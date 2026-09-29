#!/usr/bin/env python3
"""导出策略推理关键配置 —— 用于排查“走6步退3步”类动作异常。

读取 checkpoint 的 config.json / train_config.json / pre-post-processor，
汇总打印 (并可选 --json 导出) 以下与动作时序强相关的字段:
  chunk_size / n_action_steps / n_obs_steps / temporal_ensemble /
  action 归一化方式(决定 delta 还是 abs) / 输入输出维度 /
  控制频率(--freq 默认) 与 推理频率(=每 n_action_steps 触发一次前向)。

用法:
  python scripts/dump_policy_config.py \
      --model outputs/smolvla_realman_sztu/checkpoints/last/pretrained_model
  # 导出到文件:
  python scripts/dump_policy_config.py --model <ckpt> --json outputs/policy_config.json
"""
import argparse
import json
from pathlib import Path


def _load(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def collect(model_path: Path, control_freq: float):
    cfg = _load(model_path / "config.json")
    if cfg is None:
        raise FileNotFoundError(f"找不到 config.json: {model_path/'config.json'}")
    train_cfg = _load(model_path / "train_config.json") or {}
    post = _load(model_path / "policy_postprocessor.json") or {}

    ptype = cfg.get("type", "unknown")
    chunk_size = cfg.get("chunk_size")
    n_action_steps = cfg.get("n_action_steps")

    # 归一化方式决定 action 是 delta 还是 abs
    norm_map = cfg.get("normalization_mapping", {})
    action_norm = norm_map.get("ACTION")
    # SmolVLA/多数策略: 输出经 unnormalize 得到绝对关节目标(abs);
    # 若训练数据本身存的是 delta, 则 abs/delta 取决于数据集 action 定义。
    aloha_delta = cfg.get("use_delta_joint_actions_aloha", False)

    # temporal ensemble: ACT 才有该字段; inference.py 会强制 coeff=0.01
    temporal_ensemble_coeff = cfg.get("temporal_ensemble_coeff", None)

    # postprocessor 里的 unnormalizer 决定最终 action 语义
    post_steps = [s.get("registry_name") for s in post.get("steps", [])]

    in_feats = cfg.get("input_features", {})
    out_feats = cfg.get("output_features", {})

    # 推理频率: 每消费完 n_action_steps 个动作触发一次前向
    infer_every_n = n_action_steps if (ptype != "act" or temporal_ensemble_coeff is None) else 1
    infer_freq = (control_freq / infer_every_n) if infer_every_n else None

    rows = [
        ("policy_type", ptype, "策略类型"),
        ("chunk_size", chunk_size, "单次前向预测的动作步数"),
        ("n_action_steps", n_action_steps, "每个 chunk 实际执行/入队的步数(队列 maxlen)"),
        ("n_obs_steps", cfg.get("n_obs_steps"), "观测堆叠帧数"),
        ("temporal_ensemble_coeff", temporal_ensemble_coeff,
         "ACT 时序集成系数; 非空=每步全量前向+集成, 队列被绕过"),
        ("action_normalization", action_norm, "ACTION 归一化方式(MEAN_STD→abs 关节角)"),
        ("use_delta_joint_actions_aloha", aloha_delta, "是否使用 delta 关节动作(ALOHA 专用)"),
        ("action_dim", (out_feats.get("action") or {}).get("shape"), "输出动作维度"),
        ("state_dim", (in_feats.get("observation.state") or {}).get("shape"), "状态维度"),
        ("input_features", list(in_feats.keys()), "模型输入特征(注意 inference 是否全部喂入)"),
        ("postprocessor_steps", post_steps, "后处理管线(unnormalize→device)"),
        ("num_steps(flow)", cfg.get("num_steps"), "Flow Matching 采样步数(仅 VLA)"),
        ("control_freq_hz", control_freq, "控制频率(--freq, 应与数据 fps 一致)"),
        ("replan_every_n_steps", infer_every_n, "每 N 个控制步触发一次前向重规划"),
        ("inference_freq_hz", round(infer_freq, 3) if infer_freq else None, "推理(前向)频率"),
        ("dataset_fps(train)", (train_cfg.get("dataset") or {}).get("fps")
         if isinstance(train_cfg.get("dataset"), dict) else None, "训练数据 fps(参考)"),
    ]
    return rows


def main():
    ap = argparse.ArgumentParser(description="导出策略推理关键配置")
    ap.add_argument("--model", required=True, help="checkpoint 的 pretrained_model 目录")
    ap.add_argument("--freq", type=float, default=15.0, help="控制频率(Hz), 与推理 --freq 一致")
    ap.add_argument("--json", type=str, default=None, help="可选: 导出 JSON 到该路径")
    args = ap.parse_args()

    model_path = Path(args.model)
    rows = collect(model_path, args.freq)

    print("=" * 78)
    print(f"  策略配置导出  —  {model_path}")
    print("=" * 78)
    w = max(len(r[0]) for r in rows)
    for key, val, desc in rows:
        print(f"  {key:<{w}} : {val}")
        print(f"  {'':<{w}}   └─ {desc}")
    print("=" * 78)

    # 时序风险速判
    ptype = dict((r[0], r[1]) for r in rows)["policy_type"]
    chunk = dict((r[0], r[1]) for r in rows)["chunk_size"]
    nas = dict((r[0], r[1]) for r in rows)["n_action_steps"]
    print("  时序风险提示:")
    if ptype != "act" and chunk and nas and nas < chunk:
        print(f"    • n_action_steps({nas}) < chunk_size({chunk}): 每 {nas} 步丢弃 chunk 尾部 "
              f"{chunk-nas} 步并从当前观测重规划 → chunk 边界处易出现轨迹跳变/回退。")
    if ptype == "act":
        print("    • ACT: 若 temporal_ensemble_coeff 非空, 每步全量前向且集成偏重历史动作 → 指令滞后。")

    if args.json:
        out = {k: v for k, v, _ in rows}
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n  已导出 JSON: {args.json}")


if __name__ == "__main__":
    main()
