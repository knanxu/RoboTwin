"""逐段计时 take_chunk_action 内部循环, 找真实耗时大头."""
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
with h5py.File(os.path.join(DATA_ROOT, "episode0.hdf5"), "r") as f:
    actions = f["joint_action"]["vector"][...]
env_args = _build_env_args(TASK_NAME, TASK_CONFIG)


def bench_take_chunk(vel_scale, acc_scale, v_r, seed=2):
    env = _class_decorator(TASK_NAME)
    setup_env(env, seed=seed, env_args=env_args)

    chunk = actions[:50]
    if v_r != 1.0:
        chunk = reconstruct_chunk(chunk, v_r)

    # 直接模拟 take_chunk_action 内部, 逐段计时
    t_total0 = time.perf_counter()

    left_jointstate = env.robot.get_left_arm_jointState()
    right_jointstate = env.robot.get_right_arm_jointState()
    left_arm_dim = len(left_jointstate) - 1
    right_arm_dim = len(right_jointstate) - 1

    cur_state_arm = np.concatenate([
        np.array(left_jointstate[:left_arm_dim]),
        np.array(right_jointstate[:right_arm_dim]),
    ])
    cur_g = np.array([env.robot.get_left_gripper_val(), env.robot.get_right_gripper_val()]).reshape(-1)

    left_arm = chunk[:, :left_arm_dim]
    left_g  = chunk[:, left_arm_dim]
    right_arm = chunk[:, left_arm_dim+1:left_arm_dim+1+right_arm_dim]
    right_g = chunk[:, left_arm_dim+1+right_arm_dim]
    chunk_arm = np.concatenate([left_arm, right_arm], axis=1)
    chunk_g = np.stack([left_g, right_g], axis=1)

    lp = env.robot.left_mplib_planner.planner
    rp = env.robot.right_mplib_planner.planner
    vel_lim = np.concatenate([lp.joint_vel_limits, rp.joint_vel_limits])
    acc_lim = np.concatenate([lp.joint_acc_limits, rp.joint_acc_limits])

    t0 = time.perf_counter()
    retimed = retime_chunk(
        cur_state_arm, chunk_arm, cur_g, chunk_g,
        vel_lim, acc_lim, vel_scale, acc_scale, exec_hz=250,
    )
    t_topp = time.perf_counter() - t0

    if retimed["status"] != "success":
        print(f"TOPP fallback: {retimed['fallback_reason']}")
        close_env(env)
        return

    dense_arm = retimed["dense_arm_pos"]
    dense_vel = retimed["dense_arm_vel"]
    dense_g = retimed["dense_gripper"]
    T = dense_arm.shape[0]

    # 完全复刻 take_chunk_action 内部 for 循环
    t_set = 0.0
    t_sim = 0.0
    t_render1 = 0.0
    t_render2 = 0.0
    t_check = 0.0
    t_viewer = 0.0

    for t in range(T):
        t0 = time.perf_counter()
        env._update_render()
        t_render1 += time.perf_counter() - t0

        if env.render_freq:
            t0 = time.perf_counter()
            env.viewer.render()
            t_viewer += time.perf_counter() - t0

        t0 = time.perf_counter()
        env.robot.set_arm_joints(dense_arm[t, :left_arm_dim], dense_vel[t, :left_arm_dim], "left")
        env.robot.set_gripper(float(dense_g[t, 0]), "left")
        env.robot.set_arm_joints(dense_arm[t, left_arm_dim:left_arm_dim+right_arm_dim],
                                 dense_vel[t, left_arm_dim:left_arm_dim+right_arm_dim], "right")
        env.robot.set_gripper(float(dense_g[t, 1]), "right")
        t_set += time.perf_counter() - t0

        t0 = time.perf_counter()
        env.scene.step()
        t_sim += time.perf_counter() - t0

        t0 = time.perf_counter()
        env._update_render()
        t_render2 += time.perf_counter() - t0

        t0 = time.perf_counter()
        if env.check_success():
            t_check += time.perf_counter() - t0
            break
        t_check += time.perf_counter() - t0

    t_total = time.perf_counter() - t_total0

    render_total_ms = (t_render1 + t_render2) * 1000
    print(f"vs={vel_scale:>4.1f} as={acc_scale:>4.1f} v={v_r:>4.1f} | "
          f"T={T:>4d} dur={retimed['duration']:.2f}s "
          f"TOTAL={t_total*1000:>5.0f}ms | "
          f"TOPP={t_topp*1000:>4.0f} SET={t_set*1000:>4.0f} SIM={t_sim*1000:>4.0f} "
          f"RENDER_PRE={t_render1*1000:>4.0f} RENDER_POST={t_render2*1000:>4.0f} "
          f"(render_total={render_total_ms:.0f}ms, {render_total_ms/T:.2f}ms/step) "
          f"CHECK={t_check*1000:>4.0f} VIEWER={t_viewer*1000:>4.0f}")
    close_env(env)


for vs, ac, v in [
    (1.0, 1.0, 1.0),
    (2.0, 4.0, 1.0),
    (3.0, 16.0, 1.0),
    (1.0, 1.0, 1.5),
    (2.0, 4.0, 1.5),
]:
    bench_take_chunk(vs, ac, v)
