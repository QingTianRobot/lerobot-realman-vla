#!/usr/bin/env python3
"""
HDF5 → LeRobot v3 数据集转换

将采集脚本生成的 HDF5 文件转换为 LeRobot 训练所需的标准格式。
转换过程包括：
  1. 读取 HDF5 中的关节角、夹爪状态、双相机图像
  2. 创建 LeRobot Dataset（含视频编码、归一化统计）
  3. 逐帧写入并保存

⚠️ 重要：--fps 必须与采集时的帧率一致！

用法:
    python scripts/convert_to_lerobot.py \\
        --input-dir data/raw_hdf5/pick_cube \\
        --output-dir data/pick_cube_30fps \\
        --repo-id lerobot/pick_cube_30fps \\
        --fps 30 \\
        --task "pick up the cube and place it in the basket"
"""

import h5py
import numpy as np
import cv2
from pathlib import Path
import argparse
import sys
import shutil
import torch
from PIL import Image

# 确保 LeRobot 可以被导入
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def convert_hdf5_episode(hdf5_path: Path):
    """读取单个 HDF5 文件，返回字典"""
    with h5py.File(hdf5_path, 'r') as f:
        data = {
            'qpos': np.array(f['observations/qpos']),        # (N, 7)
            'action': np.array(f['action']),                  # (N, 7)
            'camera_global': np.array(f['observations/images/camera_global']),    # (N, H, W, 3)
            'camera_left': np.array(f['observations/images/camera_left']),  # (N, H, W, 3)
        }
    return data


def main():
    parser = argparse.ArgumentParser(description='HDF5 → LeRobot 数据集转换')
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="HDF5 文件所在目录")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="LeRobot 数据集输出目录")
    parser.add_argument("--repo-id", type=str, required=True,
                        help="数据集标识符 (如 lerobot/pick_cube_30fps)")
    parser.add_argument("--fps", type=int, default=30,
                        help="帧率，应与采集时一致（默认30）")
    parser.add_argument("--task", type=str, default="pick up the cube and place it in the basket",
                        help="任务描述（VLA策略需要，必须与推理时完全一致）")
    args = parser.parse_args()

    # 检查输出目录
    if args.output_dir.exists():
        print(f"输出目录已存在: {args.output_dir}")
        resp = input("是否删除并重新创建？(y/n): ")
        if resp.lower() == 'y':
            shutil.rmtree(args.output_dir)
        else:
            print("取消转换")
            return

    # 获取所有 HDF5 文件
    hdf5_files = sorted(args.input_dir.glob("*.hdf5"))
    print(f"找到 {len(hdf5_files)} 个 episodes")

    if len(hdf5_files) == 0:
        print("错误：未找到 HDF5 文件")
        return

    # 检查第一个文件获取维度信息
    sample_data = convert_hdf5_episode(hdf5_files[0])
    state_dim = sample_data['qpos'].shape[1]    # 7
    action_dim = sample_data['action'].shape[1]  # 7
    img_h, img_w = sample_data['camera_global'].shape[1:3]

    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    print(f"图像尺寸: {img_h}x{img_w}")
    print(f"目标帧率: {args.fps} Hz")
    print(f"任务描述: \"{args.task}\"")

    # 定义数据集特征
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["joint_1", "joint_2", "joint_3", "joint_4",
                       "joint_5", "joint_6", "gripper"],
        },
        "observation.images.camera_global": {
            "dtype": "video",
            "shape": (img_h, img_w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.camera_left": {
            "dtype": "video",
            "shape": (img_h, img_w, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["joint_1", "joint_2", "joint_3", "joint_4",
                       "joint_5", "joint_6", "gripper"],
        },
    }

    # 创建 LeRobot 数据集
    print(f"\n创建数据集: {args.output_dir}")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.output_dir,
        robot_type="realman",
        use_videos=True,
    )

    total_frames = 0

    # 逐 episode 转换
    for ep_idx, hdf5_path in enumerate(hdf5_files):
        data = convert_hdf5_episode(hdf5_path)
        num_frames = len(data['qpos'])
        total_frames += num_frames

        for frame_idx in range(num_frames):
            # HDF5 里存的是相机 get_frame() 的 BGR (与推理脚本同源)，
            # 而 PIL.Image.fromarray 默认按 RGB 解释，直接传入会红蓝互换
            # (肤色手变蓝、蓝色筐变橙)。这里显式转成 RGB，与 inference.py 的
            # cv2.COLOR_BGR2RGB 保持一致，保证训练/推理颜色分布相同。
            frame_data = {
                "observation.state": torch.from_numpy(
                    data['qpos'][frame_idx].astype(np.float32)),
                "observation.images.camera_global": Image.fromarray(
                    cv2.cvtColor(data['camera_global'][frame_idx], cv2.COLOR_BGR2RGB)),
                "observation.images.camera_left": Image.fromarray(
                    cv2.cvtColor(data['camera_left'][frame_idx], cv2.COLOR_BGR2RGB)),
                "action": torch.from_numpy(
                    data['action'][frame_idx].astype(np.float32)),
                "task": args.task,
            }
            dataset.add_frame(frame_data)

        dataset.save_episode()

        if (ep_idx + 1) % 10 == 0 or ep_idx == len(hdf5_files) - 1:
            print(f"  已完成 {ep_idx + 1}/{len(hdf5_files)} episodes ({total_frames} frames)")

    # 收尾：LeRobot 0.4.x 中视频在 save_episode() 时已即时编码，
    # 这里 finalize() 负责关闭 parquet writer 并写 footer 元数据；
    # 不调用则数据集无效、无法被 LeRobotDataset 加载。
    # (0.3.x 的 consolidate() 在 0.4.x 已移除)
    print("\n正在收尾数据集（写入 parquet 元数据）...")
    dataset.finalize()

    print(f"\n{'=' * 50}")
    print(f"✓ 数据集转换完成!")
    print(f"{'=' * 50}")
    print(f"  路径: {args.output_dir}")
    print(f"  FPS: {args.fps}")
    print(f"  Episodes: {len(hdf5_files)}")
    print(f"  总帧数: {total_frames}")
    print(f"  Task: \"{args.task}\"")
    print(f"\n下一步: 训练与推理时都使用 --freq {args.fps}")


if __name__ == "__main__":
    main()
