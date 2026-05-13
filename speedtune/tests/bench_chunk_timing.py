"""排查 take_chunk_action 各段耗时."""
import os, sys, time
sys.path.append(os.path.abspath("."))

import h5py
import numpy as np

from envs.utils.chunk_accel import reconstruct_chunk
from envs.robot.toppra_chunk_executor import retime_chunk
from speedtune.tests.smoke_replay_expert import (
    _build_env_args, _class_decorator, setup_env, close_env,
    TASK_NAME, TASK_CONFIG, DATA_ROOT, SEED_FILE,
)

with open(SEED_FILE) as f:
    seeds = [int(s) for s in f.read().split()]
hdf5 = os.path.join(DATA_ROOT, "episode0.hdf5")
with h5py.File(hdf5, "r") as f:
    actions = f["joint_action"]["vector"][...]

env_args = _build_env_args(TASK_NAME, TASK_CONFIG)
CHUNK_N = 50


def bench_one(vel_scale: float, acc_scale: float, v_r: float, seed: int):
    env = _class_decorator(TASK_NAME)
    setup_env(env, seed=seed, env_args=env_args)

    # 先取第一个 chunk
    chunk = actions[:CHUNK_N]
    if v_r != 1.0:
        chunk = reconstruct_chunk(chunk, v_r)
    M = chunk.shape[0]

    # 打印当前状态
    current_jointstate = env.robot.get_left_arm_jointState() + env.robot.get_right_arm_jointState()
    left_dim = len(env.robot.get_left_arm_jointState()) - 1
    right_dim = len(env.robot.get_right_arm_jointState()) - 1

    # 构造 arm / gripper
    left_arm = chunk[:, :left_dim]
    left_g = chunk[:, left_dim]
    right_arm = chunk[:, left_dim+1:left_dim+1+right_dim]
    right_g = chunk[:, left_dim+1+right_dim]
    cur_state_arm = np.concatenate([
        np.array(current_jointstate[:left_dim]),
        np.array(current_jointstate[left_dim+1:left_dim+1+right_dim]),
    ])
    cur_g = np.array([env.robot.get_left_gripper_val(), env.robot.get_right_gripper_val()]).reshape(-1)
    chunk_arm = np.concatenate([left_arm, right_arm], axis=1)
    chunk_g = np.stack([left_g, right_g], axis=1)

    # limits (取 planner 内部值)
    lp = env.robot.left_mplib_planner.planner
    rp = env.robot.right_mplib_planner.planner
    vel_lim = np.concatenate([lp.joint_vel_limits, rp.joint_vel_limits])
    acc_lim = np.concatenate([lp.joint_acc_limits, rp.joint_acc_limits])
    print(f"    [limits] vel_lim={vel_lim}, acc_lim={acc_lim}")

    # 计时 TOPP
    t0 = time.perf_counter()
    retimed = retime_chunk(
        cur_state_arm, chunk_arm, cur_g, chunk_g,
        vel_lim, acc_lim, vel_scale, acc_scale, exec_hz=250,
    )
    t_topp = time.perf_counter() - t0
    if retimed["status"] != "success":
        print(f"    [TOPP fallback] {retimed['fallback_reason']}")
        close_env(env)
        return

    T = retimed["dense_arm_pos"].shape[0]
    duration = retimed["duration"]
    dense_arm = retimed["dense_arm_pos"]
    dense_vel = retimed["dense_arm_vel"]
    dense_g = retimed["dense_gripper"]

    # 计时物理步循环
    t0 = time.perf_counter()
    for t in range(T):
        env.robot.set_arm_joints(dense_arm[t, :left_dim], dense_vel[t, :left_dim], "left")
        env.robot.set_gripper(float(dense_g[t, 0]), "left")
        env.robot.set_arm_joints(dense_arm[t, left_dim:left_dim+right_dim], dense_vel[t, left_dim:left_dim+right_dim], "right")
        env.robot.set_gripper(float(dense_g[t, 1]), "right")
        env.scene.step()
    t_sim = time.perf_counter() - t0

    # 计时渲染
    t0 = time.perf_counter()
    for _ in range(T):
        env._update_render()
    t_render = time.perf_counter() - t0

    # 计时 check_success
    t0 = time.perf_counter()
    for _ in range(T):
        env.check_success()
    t_check = time.perf_counter() - t0

    print(f"  vs={vel_scale:>4.1f} as={acc_scale:>4.1f} v={v_r:>4.1f}: "
          f"M={M:>3d} T={T:>4d} dur={duration:.2f}s | "
          f"TOPP={t_topp*1000:>5.0f}ms SIM={t_sim*1000:>5.0f}ms RENDER={t_render*1000:>5.0f}ms "
          f"CHECK={t_check*1000:>5.0f}ms | "
          f"per-step sim={t_sim/T*1000:.2f}ms render={t_render/T*1000:.2f}ms check={t_check/T*1000:.2f}ms")
    close_env(env)


if __name__ == "__main__":
    configs = [
        (1.0, 1.0, 1.0),
        (2.0, 1.0, 1.0),
        (1.0, 4.0, 1.0),
        (2.0, 4.0, 1.0),
        (3.0, 16.0, 1.0),
        (1.0, 1.0, 1.5),
        (1.0, 1.0, 2.0),
        (2.0, 4.0, 1.5),
    ]
    for vs, ascale, v in configs:
        print(f"\n-- Run vs={vs} as={ascale} v={v} (seed=2) --")
        bench_one(vs, ascale, v, seed=2)
