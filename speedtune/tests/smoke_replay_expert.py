"""
Stage 0.3 专家数据回放冒烟测试.

从 hdf5 读 joint_action/vector, 切成 chunk_size 帧的块, 分三种模式回放, 对比:
  A. per_frame_take_action:  原版 take_action 逐帧 (baseline, scale=1.0)
  B. chunk_take_chunk_action: take_chunk_action 整段 TOPPRA (不重构 chunk)
  C. reconstruct_then_chunk:  reconstruct_chunk + take_chunk_action 三组 scale

每种模式跑两个 seed (2, 3) 对应已采的 episode0.hdf5 / episode1.hdf5.
打印: status, success, chunk_count, take_action_cnt, wall_time.

Run:
    cd /home/xukainan/RoboTwin
    python speedtune/tests/smoke_replay_expert.py
"""
import os
import sys
import time
import argparse
import importlib

import h5py
import numpy as np
import yaml

# ---- RoboTwin env 初始化辅助 (摘自 eval_policy.py) ----
sys.path.append(os.path.abspath("."))
from envs import CONFIGS_PATH  # noqa: E402
from envs.utils.chunk_accel import reconstruct_chunk  # noqa: E402


TASK_NAME = "shake_bottle"
TASK_CONFIG = "smoke_test"
DATA_ROOT = f"./data/{TASK_NAME}/{TASK_CONFIG}/data"
SEED_FILE = f"./data/{TASK_NAME}/{TASK_CONFIG}/seed.txt"
CHUNK_SIZE = 50


# ---------- Env setup ----------

def _class_decorator(task_name):
    mod = importlib.import_module(f"envs.{task_name}")
    return getattr(mod, task_name)()


def _build_env_args(task_name: str, task_config: str) -> dict:
    with open(f"./task_config/{task_config}.yml", "r") as f:
        args = yaml.safe_load(f)
    args["task_name"] = task_name
    args["task_config"] = task_config

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r") as f:
        emb_types = yaml.safe_load(f)

    et = args.get("embodiment")
    if len(et) == 1:
        args["left_robot_file"] = emb_types[et[0]]["file_path"]
        args["right_robot_file"] = emb_types[et[0]]["file_path"]
        args["dual_arm_embodied"] = True
    else:
        raise NotImplementedError("only single-embodiment supported in smoke test")

    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r") as f:
        cam_cfg = yaml.safe_load(f)
    head = args["camera"]["head_camera_type"]
    args["head_camera_h"] = cam_cfg[head]["h"]
    args["head_camera_w"] = cam_cfg[head]["w"]

    def _get_emb_cfg(file_path: str) -> dict:
        with open(os.path.join(file_path, "config.yml"), "r") as f:
            return yaml.safe_load(f)

    args["left_embodiment_config"] = _get_emb_cfg(args["left_robot_file"])
    args["right_embodiment_config"] = _get_emb_cfg(args["right_robot_file"])
    args["play_once_path_file_list"] = []
    # eval_mode: True 触发 step_lim 从 _eval_step_limit.yml 读
    args["eval_mode"] = True
    # 禁止采集数据 / 写 cache
    args["collect_data"] = False
    return args


def setup_env(task_env, seed: int, env_args: dict):
    """按 seed 初始化一次 episode, 返回 env (已做 setup_demo)."""
    task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **env_args)


def close_env(task_env):
    try:
        task_env.close_env()
    except Exception as e:
        print(f"[warn] close_env: {e}")


# ---------- Replay modes ----------

def replay_per_frame(task_env, actions: np.ndarray, chunk_size: int = CHUNK_SIZE):
    """Mode A: 按原版逐帧 take_action 回放整个 action 序列."""
    # 重置 per-episode success_obs 计时
    task_env._last_success_obs_time = 0.0
    t0 = time.time()
    topp_fail = 0
    n_frames = 0
    for i in range(0, len(actions), chunk_size):
        chunk = actions[i : i + chunk_size]
        for act in chunk:
            if task_env.take_action_cnt >= task_env.step_lim or task_env.eval_success:
                break
            task_env.take_action(act, action_type="qpos", vel_scale=1.0, acc_scale=1.0)
            n_frames += 1
        if task_env.take_action_cnt >= task_env.step_lim or task_env.eval_success:
            break
    wall = time.time() - t0
    obs_time = float(getattr(task_env, "_last_success_obs_time", 0.0))
    return {
        "mode": "per_frame",
        "success": bool(task_env.eval_success),
        "take_action_cnt": int(task_env.take_action_cnt),
        "n_chunk_calls": n_frames,  # 逐帧模式下 = 动作数
        "topp_fail": topp_fail,
        "wall_time": wall,
        "success_obs_time": obs_time,
        "wall_time_no_obs": max(0.0, wall - obs_time),
    }


def replay_chunk(task_env, actions: np.ndarray, vel_scale: float, acc_scale: float,
                 v_reconstruct: float = 1.0, chunk_size: int = CHUNK_SIZE):
    """Mode B/C: 切块 + (可选重构) + take_chunk_action."""
    t0 = time.time()
    topp_fail = 0
    n_chunks = 0
    obs_time_sum = 0.0
    for i in range(0, len(actions), chunk_size):
        chunk = actions[i : i + chunk_size]
        if chunk.shape[0] < 2:
            break
        if v_reconstruct != 1.0:
            chunk = reconstruct_chunk(chunk, v_reconstruct)
        info = task_env.take_chunk_action(chunk, vel_scale=vel_scale, acc_scale=acc_scale)
        n_chunks += 1
        obs_time_sum += float(info.get("success_obs_time", 0.0))
        if info["status"] == "topp_fallback":
            topp_fail += 1
        if info["status"] == "truncated":
            break
        if task_env.eval_success:
            break
    wall = time.time() - t0
    return {
        "mode": f"chunk(vs={vel_scale},as={acc_scale},v={v_reconstruct})",
        "success": bool(task_env.eval_success),
        "take_action_cnt": int(task_env.take_action_cnt),
        "n_chunk_calls": n_chunks,
        "topp_fail": topp_fail,
        "wall_time": wall,
        "success_obs_time": obs_time_sum,
        "wall_time_no_obs": max(0.0, wall - obs_time_sum),
    }


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=-1,
                        help="-1 = all available")
    args = parser.parse_args()

    # 加载 seed
    if not os.path.exists(SEED_FILE):
        raise FileNotFoundError(f"seed file missing: {SEED_FILE}")
    with open(SEED_FILE) as f:
        seeds = [int(s) for s in f.read().split()]

    # 遍历 hdf5
    hdf5_files = sorted(
        [f for f in os.listdir(DATA_ROOT) if f.endswith(".hdf5")],
        key=lambda x: int(x.replace("episode", "").replace(".hdf5", "")),
    )
    if args.episodes > 0:
        hdf5_files = hdf5_files[: args.episodes]
        seeds = seeds[: args.episodes]
    assert len(hdf5_files) == len(seeds), f"hdf5 count {len(hdf5_files)} != seed count {len(seeds)}"

    env_args = _build_env_args(TASK_NAME, TASK_CONFIG)

    # 要测的配置
    configurations = [
        ("per_frame_vs1_as1", "per_frame", {}),
        ("chunk_vs1_as1_v1",     "chunk",    {"vel_scale": 1.0, "acc_scale": 1.0, "v_reconstruct": 1.0}),
        ("chunk_vs2_as4_v1",     "chunk",    {"vel_scale": 2.0, "acc_scale": 4.0, "v_reconstruct": 1.0}),
        ("chunk_vs1_as1_v1.5",   "chunk",    {"vel_scale": 1.0, "acc_scale": 1.0, "v_reconstruct": 1.5}),
        ("chunk_vs2_as4_v1.5",   "chunk",    {"vel_scale": 2.0, "acc_scale": 4.0, "v_reconstruct": 1.5}),
    ]

    all_results = {name: [] for name, _, _ in configurations}

    for cfg_name, mode, kwargs in configurations:
        print(f"\n{'='*78}\n=== Running config: {cfg_name} ===\n{'='*78}")
        task_env = _class_decorator(TASK_NAME)
        for ep_i, (hdf5_name, seed) in enumerate(zip(hdf5_files, seeds)):
            hdf5_path = os.path.join(DATA_ROOT, hdf5_name)
            with h5py.File(hdf5_path, "r") as f:
                actions = f["joint_action"]["vector"][...]       # (T, 14)
            print(f"\n  -- episode {ep_i} (seed={seed}) actions shape {actions.shape} --")

            try:
                setup_env(task_env, seed=seed, env_args=env_args)
            except Exception as e:
                print(f"  [ERR] setup_demo: {e}")
                continue

            try:
                if mode == "per_frame":
                    result = replay_per_frame(task_env, actions)
                else:
                    result = replay_chunk(task_env, actions, **kwargs)
            except Exception as e:
                import traceback
                traceback.print_exc()
                result = {"mode": cfg_name, "success": False, "error": str(e)}
            finally:
                close_env(task_env)

            result["seed"] = seed
            result["episode"] = ep_i
            all_results[cfg_name].append(result)
            print(f"  -> {result}")

    # ---- 汇总 ----
    print(f"\n\n{'='*78}\n=== Summary (wall_time_no_obs = wall_time - success_obs_time) ===\n{'='*78}")
    header = (f"{'config':<28} {'success':<9} {'avg_cnt':<10} {'avg_chunks':<12} "
              f"{'wall(s)':<10} {'obs(s)':<10} {'wall_no_obs(s)':<14} {'topp_fail':<10}")
    print(header)
    print("-" * len(header))
    for cfg_name, _, _ in configurations:
        runs = all_results[cfg_name]
        ok = [r for r in runs if "error" not in r]
        if not ok:
            print(f"{cfg_name:<28} (all errored)")
            continue
        n = len(ok)
        succ_rate = sum(r["success"] for r in ok) / n
        avg_cnt = sum(r["take_action_cnt"] for r in ok) / n
        avg_chunks = sum(r["n_chunk_calls"] for r in ok) / n
        avg_wall = sum(r["wall_time"] for r in ok) / n
        avg_obs = sum(r.get("success_obs_time", 0.0) for r in ok) / n
        avg_wall_no_obs = sum(r.get("wall_time_no_obs", r["wall_time"]) for r in ok) / n
        avg_fail = sum(r["topp_fail"] for r in ok) / n
        print(
            f"{cfg_name:<28} {succ_rate*100:>6.1f}%  {avg_cnt:<10.1f} "
            f"{avg_chunks:<12.1f} {avg_wall:<10.2f} {avg_obs:<10.2f} "
            f"{avg_wall_no_obs:<14.2f} {avg_fail:<10.2f}"
        )


if __name__ == "__main__":
    main()
