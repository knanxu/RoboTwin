"""
验证 expert_speed_factor (慢速专家数据生成) 是否生效.

同一 task、同一 seed -> 同一布局、同一规划目标、同一关节路径, 只是 curobo
time_dilation_factor 不同. 若透传链路正确, 慢速版规划轨迹的总步数 N 应 ≈
(1/expert_speed_factor) 倍于原速版 (例 factor=0.2 -> ≈5x).

用法:
    CUDA_VISIBLE_DEVICES=0 python speedtune/tests/verify_slow_expert.py [task_name] [seed]
默认 task=shake_bottle, seed=0.
"""
import sys
import os

sys.path.append("./")

import yaml
import importlib
from envs import *  # noqa: F401,F403  -> 暴露 CONFIGS_PATH 等


def class_decorator(task_name):
    m = importlib.import_module(f"envs.{task_name}")
    return getattr(m, task_name)()


def get_emb_cfg(robot_file):
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def build_args(task_name, task_config):
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["task_name"] = task_name
    et = args.get("embodiment")
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:  # noqa: F405
        emb = yaml.load(f.read(), Loader=yaml.FullLoader)
    gef = lambda t: emb[t]["file_path"]  # noqa: E731
    args["left_robot_file"] = gef(et[0])
    args["right_robot_file"] = gef(et[0] if len(et) == 1 else et[1])
    args["dual_arm_embodied"] = len(et) == 1
    if len(et) == 3:
        args["embodiment_dis"] = et[2]
        args["dual_arm_embodied"] = False
    args["left_embodiment_config"] = get_emb_cfg(args["left_robot_file"])
    args["right_embodiment_config"] = get_emb_cfg(args["right_robot_file"])
    args["embodiment_name"] = str(et[0])
    args["task_config"] = task_config
    args["save_path"] = os.path.join("./data", task_name, task_config)
    return args


def total_steps(env):
    def s(paths):
        return sum(p["position"].shape[0]
                   for p in paths if isinstance(p, dict) and p.get("status") == "Success")
    return s(env.left_joint_path), s(env.right_joint_path)


def run_once(task, base_args, factor, seed):
    a = dict(base_args)
    a["expert_speed_factor"] = factor
    a["need_plan"] = True
    a["render_freq"] = 0
    a["save_freq"] = None          # 跳过相机渲染, 只做规划+执行, 加速验证
    a["left_joint_path"] = []
    a["right_joint_path"] = []
    task.setup_demo(now_ep_num=0, seed=seed, **a)
    rb = task.robot
    lp = getattr(rb, "left_planner", None)
    print(f"    [debug factor={factor}] expert_time_dilation={getattr(rb, 'expert_time_dilation', 'NA')} "
          f"comm_flag={getattr(rb, 'communication_flag', 'NA')} "
          f"lp.tdf={getattr(lp, 'time_dilation_factor', 'NO_LP') if lp is not None else 'None_obj'}", flush=True)
    task.play_once()
    ok = task.plan_success
    nl, nr = total_steps(task)
    task.close_env()
    return ok, nl, nr


def safe_run(task, base_args, factor, seed):
    try:
        return run_once(task, base_args, factor, seed)
    except Exception as e:
        print(f"  [skip] factor={factor} seed={seed}: {type(e).__name__}", flush=True)
        try:
            task.close_env()
        except Exception:
            pass
        return None


if __name__ == "__main__":
    task_name = sys.argv[1] if len(sys.argv) > 1 else "shake_bottle"
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    base_args = build_args(task_name, "demo_slow")
    task = class_decorator(task_name)

    print(f"\n=== verify slow expert: task={task_name} ===", flush=True)
    # 1) 找一个原速能规划成功的稳定 seed (物理不稳定的 seed 会被跳过, 同 collect_data 行为)
    seed, rf = None, None
    for s in range(start, start + 15):
        r = safe_run(task, base_args, 1.0, s)
        if r and r[0] and (r[1] + r[2]) > 0:
            seed, rf = s, r
            break
        print(f"  seed={s} 不可用, 换下一个", flush=True)
    if rf is None:
        print(">>> 15 个 seed 内未找到可用样本", flush=True)
        sys.exit(1)
    okf, nlf, nrf = rf
    print(f"[FAST factor=1.0] seed={seed} success={okf} left={nlf} right={nrf} total={nlf + nrf}", flush=True)

    # 2) 同一 seed 跑慢速 (同布局/同规划目标, 只是时间参数化不同)
    rs = safe_run(task, base_args, 0.2, seed)
    if rs is None:
        print(">>> SLOW 运行失败", flush=True)
        sys.exit(1)
    oks, nls, nrs = rs
    print(f"[SLOW factor=0.2] seed={seed} success={oks} left={nls} right={nrs} total={nls + nrs}", flush=True)

    tf, ts = nlf + nrf, nls + nrs
    print(f"\n>>> total-step ratio = {ts / tf:.2f}x   (期望 ≈ 5.0x = 1/0.2)", flush=True)
