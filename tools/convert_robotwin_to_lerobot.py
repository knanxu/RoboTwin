#!/usr/bin/env python3
"""Convert RoboTwin hdf5 expert demos to LeRobot v2.0 format for openpi training.

RoboTwin's `script/collect_data.py` produces:
  data/<task>/<task_config>/data/episode<i>.hdf5
with this layout:
  /joint_action/{left_arm,left_gripper,right_arm,right_gripper,vector}
  /observation/{head_camera,left_camera,right_camera}/rgb   (jpeg-encoded)

openpi's `pi05_aloha_robotwin_drifting_*` configs (see
`src/openpi/training/config.py`) declare a LeRobotAlohaDataConfig with
RepackTransform that expects:
  observation.images.cam_high
  observation.images.cam_left_wrist
  observation.images.cam_right_wrist
  observation.state
  action

So we:
  1. Decode jpeg images per frame
  2. Build action vector = qpos[t+1] (next-state-as-action, matching how
     RoboTwin replays trajectories)
  3. Pack into LeRobotDataset.create + add_frame + save_episode
  4. dataset.consolidate() to write meta/

Run inside the openpi venv (Python 3.11, lerobot pinned by openpi/uv.lock).

Usage:
  python tools/convert_robotwin_to_lerobot.py \
      --raw-root /path/to/RoboTwin/data \
      --task-name shake_bottle \
      --task-config demo_clean \
      --episode-num 100 \
      --repo-id shake_bottle_drifting_repo \
      --task-prompt "shake the bottle"
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from pathlib import Path
from typing import List, Optional

import cv2
import h5py
import numpy as np
import torch
import tqdm

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset


# RoboTwin aloha-agilex single-embodiment: 6 + 1 + 6 + 1 = 14 joints,
# joint_action/vector layout = [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
ALOHA_MOTOR_NAMES = [
    "left_waist", "left_shoulder", "left_elbow",
    "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate",
    "left_gripper",
    "right_waist", "right_shoulder", "right_elbow",
    "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate",
    "right_gripper",
]
# 注意: 这里命名顺序和 RoboTwin 的 vector 完全对齐 -- 左7右7. openpi 的
# pi05_base trossen norm stats 也是这个顺序的 14 dim.

# 对齐 pi05_aloha_robotwin_drifting_shake_bottle.RepackTransform 的 image keys
ROBOTWIN_TO_PI05_CAM = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 1.0
    image_writer_processes: int = 5
    image_writer_threads: int = 5
    video_backend: Optional[str] = None
    fps: int = 25  # RoboTwin 默认 save_freq=15, 但 fps 这里只影响 LeRobot meta, 不影响训练


def _decode_jpeg(buf: bytes) -> np.ndarray:
    """RoboTwin 的 RGB 是 jpeg-encoded bytes; 解码成 (H, W, 3) RGB uint8."""
    arr = np.frombuffer(buf, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2.imdecode failed; corrupt jpeg?")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
        rgb = cv2.resize(rgb, (IMAGE_WIDTH, IMAGE_HEIGHT))
    return rgb


def _create_dataset(repo_id: str, cfg: DatasetConfig) -> LeRobotDataset:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(ALOHA_MOTOR_NAMES),),
            "names": [ALOHA_MOTOR_NAMES],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ALOHA_MOTOR_NAMES),),
            "names": [ALOHA_MOTOR_NAMES],
        },
    }
    for cam in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
        features[f"observation.images.{cam}"] = {
            "dtype": "video" if cfg.use_videos else "image",
            "shape": (3, IMAGE_HEIGHT, IMAGE_WIDTH),
            "names": ["channels", "height", "width"],
        }

    target = LEROBOT_HOME / repo_id
    if target.exists():
        print(f"[convert] removing existing dataset at {target}")
        shutil.rmtree(target)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=cfg.fps,
        robot_type="aloha",
        features=features,
        use_videos=cfg.use_videos,
        tolerance_s=cfg.tolerance_s,
        image_writer_processes=cfg.image_writer_processes,
        image_writer_threads=cfg.image_writer_threads,
        video_backend=cfg.video_backend,
    )


def _load_episode(ep_path: Path):
    with h5py.File(ep_path, "r") as f:
        # joint_action.vector: (T, 14) float
        vector = np.asarray(f["/joint_action/vector"][()], dtype=np.float32)
        if vector.ndim != 2 or vector.shape[1] != len(ALOHA_MOTOR_NAMES):
            raise RuntimeError(
                f"unexpected joint_action/vector shape {vector.shape} in {ep_path}; "
                f"expected (T, {len(ALOHA_MOTOR_NAMES)})"
            )

        cam_arrays = {}
        for src_key, dst_key in ROBOTWIN_TO_PI05_CAM.items():
            jpeg_seq = f[f"/observation/{src_key}/rgb"][()]
            cam_arrays[dst_key] = jpeg_seq  # 解码留到 frame 级, 省内存

    return vector, cam_arrays


def _maybe_load_instructions(ep_path: Path, ep_idx: int, default_prompt: str) -> str:
    """RoboTwin 的 instructions/episode<i>.json 里有 seen / unseen 多条指令, 抽一条."""
    instr_dir = ep_path.parent.parent / "instructions"
    instr_path = instr_dir / f"episode{ep_idx}.json"
    if not instr_path.exists():
        return default_prompt
    try:
        data = json.loads(instr_path.read_text())
    except Exception:
        return default_prompt
    seen = data.get("seen") or []
    if not seen:
        return default_prompt
    return str(seen[0])


def populate(
    dataset: LeRobotDataset,
    hdf5_files: List[Path],
    *,
    default_prompt: str,
):
    for ep_idx, ep_path in enumerate(tqdm.tqdm(hdf5_files, desc="episodes")):
        vector, cam_arrays = _load_episode(ep_path)
        T = vector.shape[0]
        if T < 2:
            print(f"[warn] episode {ep_idx} has only {T} frames, skipping")
            continue

        # action[t] := state[t+1]; 最后一帧不做 frame
        for t in range(T - 1):
            state_t = vector[t]
            action_t = vector[t + 1]
            frame = {
                "observation.state": torch.from_numpy(state_t),
                "action": torch.from_numpy(action_t),
            }
            for dst_key, jpeg_seq in cam_arrays.items():
                rgb = _decode_jpeg(bytes(jpeg_seq[t]))
                frame[f"observation.images.{dst_key}"] = rgb
            dataset.add_frame(frame)

        prompt = _maybe_load_instructions(ep_path, ep_idx, default_prompt)
        dataset.save_episode(task=prompt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="RoboTwin 数据根目录, 通常是 /app/RoboTwin/data")
    parser.add_argument("--task-name", type=str, required=True)
    parser.add_argument("--task-config", type=str, required=True)
    parser.add_argument("--episode-num", type=int, default=-1,
                        help="-1 = 全部")
    parser.add_argument("--repo-id", type=str, required=True,
                        help="LeRobotDataset repo_id, 必须和 openpi config 里的 repo_id 一致")
    parser.add_argument("--task-prompt", type=str, default="",
                        help="若 instructions/*.json 没有, 用作 fallback 的语言指令")
    parser.add_argument("--use-videos", action="store_true", default=True)
    parser.add_argument("--no-use-videos", dest="use_videos", action="store_false")
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    raw_dir = args.raw_root / args.task_name / args.task_config / "data"
    if not raw_dir.is_dir():
        raise SystemExit(f"raw dir not found: {raw_dir}")

    hdf5_files = sorted(raw_dir.glob("episode*.hdf5"),
                        key=lambda p: int(p.stem.replace("episode", "")))
    if args.episode_num > 0:
        hdf5_files = hdf5_files[: args.episode_num]
    if not hdf5_files:
        raise SystemExit(f"no episode*.hdf5 under {raw_dir}")
    print(f"[convert] {len(hdf5_files)} episodes from {raw_dir}")

    cfg = DatasetConfig(use_videos=args.use_videos, fps=args.fps)
    dataset = _create_dataset(args.repo_id, cfg)
    default_prompt = args.task_prompt or args.task_name.replace("_", " ")
    populate(dataset, hdf5_files, default_prompt=default_prompt)
    dataset.consolidate(run_compute_stats=True)

    out_path = LEROBOT_HOME / args.repo_id
    print(f"[convert] dataset written to {out_path}")
    print(f"[convert]   episodes: {dataset.num_episodes}")
    print(f"[convert]   frames:   {dataset.num_frames}")
    print(f"[convert] make sure openpi config repo_id == \"{args.repo_id}\"")


if __name__ == "__main__":
    main()
