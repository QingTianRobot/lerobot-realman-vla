#!/usr/bin/env python3
"""开环回放 —— 固定观测, 输出完整 chunk 的动作曲线 (无需机械臂/相机)。

用途: 排查“走6步退3步”。给定一帧固定观测, 直接调用 policy.predict_action_chunk
拿到整段 chunk (chunk_size 步) 的绝对关节角, 观察模型“本身”预测的轨迹是否
在 chunk 内就出现回退, 以及多次重规划 (fixed obs) 之间是否漂移。

这不经过动作队列/EMA/死区, 是纯策略开环输出, 用于把“模型问题”与
“推理主循环后处理问题”解耦。

用法:
  # 用数据集某帧作为固定观测, dump 整 chunk + 画图
  python scripts/open_loop_replay.py \
      --model outputs/smolvla_realman_sztu/checkpoints/last/pretrained_model \
      --dataset-root data/datasets/pick_plug_lerobot \
      --episode 0 --frame 0 \
      --task "pick up the cube and place it in the basket" \
      --replans 8 \
      --out outputs/replay/chunk.csv --plot outputs/replay/chunk.png

  # 用自采的 .npz 固定观测 (键: observation.state / *.camera_global / *.camera_left / task)
  python scripts/open_loop_replay.py --model <ckpt> --obs-npz my_obs.npz --out chunk.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_policy_and_processors(model_path: str, device):
    model_path = Path(model_path)
    with open(model_path / "config.json") as f:
        cfg = json.load(f)
    ptype = cfg.get("type", "unknown")

    if ptype == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as P
    elif ptype == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy as P
    elif ptype == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy as P
    elif ptype == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy as P
    elif ptype == "vqbet":
        from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy as P
    else:
        raise ValueError(f"不支持的策略类型: {ptype}")

    policy = P.from_pretrained(str(model_path))
    policy.to(device)
    policy.eval()

    from lerobot.processor.pipeline import DataProcessorPipeline
    pre = DataProcessorPipeline.from_pretrained(str(model_path), config_filename="policy_preprocessor.json")
    post = DataProcessorPipeline.from_pretrained(str(model_path), config_filename="policy_postprocessor.json")
    return policy, pre, post, ptype, cfg


def build_obs_from_dataset(dataset_root, episode, frame, task, include_ee_pose=True):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id="local", root=str(dataset_root))

    # (episode, frame) → 全局 index (只读小整型列, 快)
    ep = np.asarray(ds.hf_dataset["episode_index"])
    fr = np.asarray(ds.hf_dataset["frame_index"])
    mask = (ep == episode) & (fr == frame)
    hits = np.where(mask)[0]
    if len(hits) == 0:
        raise ValueError(f"episode={episode} frame={frame} 不存在 (episodes 0..{ep.max()})")
    gidx = int(hits[0])
    item = ds[gidx]

    if task is None:
        task = item.get("task", "pick up the cube")

    obs = {
        "observation.state": item["observation.state"].clone().float(),
        "observation.images.camera_global": item["observation.images.camera_global"].clone().float(),
        "observation.images.camera_left": item["observation.images.camera_left"].clone().float(),
        "task": task,
    }
    if include_ee_pose and "observation.ee_pose" in item:
        # 注: SmolVLA/ACT 的 modeling 并不消费 ee_pose(仅 observation.state),
        # 这里带上只为忠实还原训练输入; 对预测无影响。
        obs["observation.ee_pose"] = item["observation.ee_pose"].clone().float()
    return obs, task, item


def build_obs_from_npz(npz_path, task):
    d = np.load(npz_path, allow_pickle=True)
    obs = {}
    for k in d.files:
        v = d[k]
        if k == "task":
            obs["task"] = str(v)
        else:
            obs[k] = torch.from_numpy(np.asarray(v)).float()
    if task is not None:
        obs["task"] = task
    obs.setdefault("task", "pick up the cube")
    return obs, obs["task"]


def predict_chunk_abs(policy, pre, post, obs, device):
    """固定观测 → 整段 chunk 的绝对关节角 (chunk_size, action_dim) + 归一化原始输出"""
    batch = pre(dict(obs))
    with torch.no_grad():
        chunk = policy.predict_action_chunk(batch)          # (1, C, A) 归一化空间
    chunk_norm = chunk[0].detach().cpu().numpy().copy()      # (C, A)
    chunk_abs = post({"action": chunk})["action"][0].cpu().numpy()  # (C, A) 反归一化
    return chunk_norm, chunk_abs


def analyze_retreat(chunk_abs):
    """在整段 chunk 内, 逐关节找回退: 某关节先增后减(或先减后增)的拐点。"""
    C, A = chunk_abs.shape
    d = np.diff(chunk_abs, axis=0)               # (C-1, A) 每步增量
    report = []
    for j in range(A):
        dj = d[:, j]
        # 符号翻转点 = 方向反转(潜在“退步”)
        sign = np.sign(dj)
        flips = np.where(np.diff(sign) != 0)[0] + 1   # 在 chunk 中的步号
        report.append((j, flips.tolist(), float(np.min(dj)), float(np.max(dj))))
    return report


def main():
    ap = argparse.ArgumentParser(description="开环回放: 固定观测输出完整 chunk 动作曲线")
    ap.add_argument("--model", required=True, help="checkpoint 的 pretrained_model 目录")
    ap.add_argument("--dataset-root", default=None, help="LeRobot 数据集根目录 (提供固定观测)")
    ap.add_argument("--obs-npz", default=None, help="替代数据集: 从 .npz 读取固定观测")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--task", default=None, help="语言指令 (默认取数据集该帧的 task)")
    ap.add_argument("--replans", type=int, default=1,
                    help="对同一固定观测重复前向 N 次, 观察 chunk 间漂移/方差 (flow-matching 含随机性)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="outputs/replay/chunk.csv")
    ap.add_argument("--plot", default=None, help="可选: 保存关节曲线 PNG")
    ap.add_argument("--no-ee-pose", action="store_true", help="不喂 ee_pose (与 inference.py 完全一致)")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[replay] device={device}")

    policy, pre, post, ptype, cfg = load_policy_and_processors(args.model, device)
    chunk_size = cfg.get("chunk_size")
    n_action_steps = cfg.get("n_action_steps")
    print(f"[replay] policy={ptype} chunk_size={chunk_size} n_action_steps={n_action_steps}")

    if args.obs_npz:
        obs, task = build_obs_from_npz(args.obs_npz, args.task)
    elif args.dataset_root:
        obs, task, _ = build_obs_from_dataset(
            args.dataset_root, args.episode, args.frame, args.task,
            include_ee_pose=not args.no_ee_pose)
    else:
        raise SystemExit("必须提供 --dataset-root 或 --obs-npz")
    print(f"[replay] task='{task}'  obs_keys={list(obs.keys())}")
    print(f"[replay] state(deg)={np.round(obs['observation.state'].numpy(), 2).tolist()}")

    # 主回放: 第一次前向的完整 chunk
    chunk_norm, chunk_abs = predict_chunk_abs(policy, pre, post, obs, device)
    C, A = chunk_abs.shape
    print(f"[replay] 输出 chunk 形状={chunk_abs.shape} (前 {n_action_steps} 步会被执行, 其余 {C - (n_action_steps or C)} 步丢弃)")

    # 多次重规划: 记录每次 chunk 的首动作 + 末动作, 看固定观测下是否漂移/回退
    replan_first = []
    replan_last = []
    for r in range(max(0, args.replans - 1)):
        cn, ca = predict_chunk_abs(policy, pre, post, obs, device)
        replan_first.append(ca[0])
        replan_last.append(ca[-1])
    replan_first.insert(0, chunk_abs[0])
    replan_last.insert(0, chunk_abs[-1])

    # 保存 CSV
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        head = ["step_in_chunk", "executed"] + [f"j{j+1}_abs" for j in range(A - 1)] + ["gripper_abs"] \
            + [f"j{j+1}_norm" for j in range(A)]
        w.writerow(head)
        for i in range(C):
            executed = int(n_action_steps is not None and i < n_action_steps)
            w.writerow([i, executed]
                       + [f"{chunk_abs[i, j]:.5f}" for j in range(A)]
                       + [f"{chunk_norm[i, j]:.6f}" for j in range(A)])
    print(f"[replay] 已写 CSV: {out}  ({C} 行)")

    # 回退分析
    print("[replay] 整段 chunk 内各关节方向反转(潜在退步)步号:")
    for j, flips, dmin, dmax in analyze_retreat(chunk_abs):
        tag = "夹爪" if j == A - 1 else f"J{j+1}"
        print(f"    {tag}: 反转步={flips}  单步增量范围=[{dmin:.3f}, {dmax:.3f}] deg")

    if args.replans > 1:
        rf = np.array(replan_first)   # (R, A)
        print(f"[replay] {args.replans} 次固定观测重规划, 首动作标准差(各关节)= "
              f"{np.round(rf.std(axis=0), 3).tolist()}")
        print(f"[replay]   首动作均值={np.round(rf.mean(axis=0), 2).tolist()}")

    # 画图 (matplotlib 缺失时优雅降级, CSV 已包含全部曲线数据)
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[replay] 未安装 matplotlib, 跳过画图 (CSV 已含完整曲线; "
                  "可 `uv pip install matplotlib` 后重跑)")
            return
        fig, axes = plt.subplots(A, 1, figsize=(11, 2.2 * A), sharex=True)
        if A == 1:
            axes = [axes]
        steps = np.arange(C)
        for j in range(A):
            ax = axes[j]
            ax.plot(steps, chunk_abs[:, j], "-o", ms=3, label="chunk abs")
            if n_action_steps:
                ax.axvline(n_action_steps - 0.5, color="r", ls="--", lw=1,
                           label=f"n_action_steps={n_action_steps} (执行/丢弃边界)")
            ax.set_ylabel(f"J{j+1}" if j < A - 1 else "grip")
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("step_in_chunk")
        fig.suptitle(f"Open-loop chunk | {ptype} | ep{args.episode} frame{args.frame}")
        fig.tight_layout()
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=110)
        print(f"[replay] 已写曲线图: {args.plot}")


if __name__ == "__main__":
    main()
