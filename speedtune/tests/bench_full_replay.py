"""完整模拟 smoke_replay 的 chunk 回放流程, 逐 chunk 计时."""
import os, sys, time
sys.path.append(os.path.abspath("."))
import h5py
import numpy as np

from envs.utils.chunk_accel import reconstruct_chunk
from speedtune.tests.smoke_replay_expert import (
    _build_env_args, _class_decorator, setup_env, close_env,
    TASK_NAME, TASK_CONFIG, DATA_ROOT, SEED_FILE,
)

with open(SEED_FILE) as f:
    seeds = [int(s) for s in f.read().split()]
with h5py.File(os.path.join(DATA_ROOT, "episode0.hdf5"), "r") as f:
    actions = f["joint_action"]["vector"][...]
env_args = _build_env_args(TASK_NAME, TASK_CONFIG)


def replay_chunks(vs, ac, v_r, seed=2):
    env = _class_decorator(TASK_NAME)
    setup_env(env, seed=seed, env_args=env_args)
    CHUNK = 50

    t_total0 = time.perf_counter()
    chunk_times = []
    n_chunks = 0

    for i in range(0, len(actions), CHUNK):
        raw = actions[i:i+CHUNK]
        if raw.shape[0] < 2:
            break
        chunk = reconstruct_chunk(raw, v_r) if v_r != 1.0 else raw

        t0 = time.perf_counter()
        info = env.take_chunk_action(chunk, vel_scale=vs, acc_scale=ac)
        dt = time.perf_counter() - t0

        # 触发 success 后的 get_obs 也算进来; 这里再测一次让真空占用暴露
        if env.eval_success:
            t_post = time.perf_counter()
            # 不做任何额外 get_obs, 只看 dt 是不是已经涵盖了 success 分支
            t_post_dt = time.perf_counter() - t_post
        else:
            t_post_dt = 0

        chunk_times.append((dt*1000, info.get('dense_steps',0), info['status'], env.eval_success))
        n_chunks += 1
        if info['status'] == 'truncated':
            break
        if env.eval_success:
            break

    t_total = time.perf_counter() - t_total0
    print(f"vs={vs:>4.1f} as={ac:>4.1f} v={v_r:>4.1f} | "
          f"n_chunks={n_chunks} TOTAL_wall={t_total*1000:>5.0f}ms success={env.eval_success}")
    for i, (dt, steps, st, succ) in enumerate(chunk_times):
        print(f"    chunk {i}: {dt:>5.0f}ms, steps={steps:>4d}, status={st}, success={succ}")
    close_env(env)


for vs, ac, v in [
    (1.0, 1.0, 1.0),
    (2.0, 4.0, 1.0),
    (3.0, 16.0, 1.0),
    (1.0, 1.0, 1.5),
    (2.0, 4.0, 1.5),
]:
    replay_chunks(vs, ac, v)
    print()
