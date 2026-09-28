#!/usr/bin/env python3
"""分析故障 episode CSV —— 回答三个现象问题:
  1) 退步是否总在 chunk 边界?   2) 哪些关节?   3) 周期多长?

输入: inference.py --log-csv 产出的 CSV (字段见 EpisodeLogger)。
分析对象默认用 cmd_joint (下发指令) 与 actual_joint (实测) 双通道对比。

核心方法:
  • 退步事件 = 轨迹转折点(局部极值)中“朝回走”的那一段起点; 用滑动平均去抖后找拐点。
  • 边界相关性 = 统计退步起点的 step_in_chunk 分布 (集中在 0 → 边界触发; 均匀 → 策略内生抖动)。
  • 周期 = 对去趋势后的关节信号做自相关, 取首个正峰对应的 lag(步) 与秒。

用法:
  python scripts/analyze_episode.py outputs/logs/fault_episode.csv
  python scripts/analyze_episode.py fault.csv --signal actual_joint --thresh 0.8 --plot outputs/logs/fault.png
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def parse_vec(s):
    if s is None or s == "":
        return None
    return np.array([float(x) for x in s.split("|")], dtype=np.float64)


def load_csv(path):
    rows = list(csv.DictReader(open(path)))
    data = {
        "step": np.array([int(float(r["step"])) for r in rows]),
        "timestamp": np.array([float(r["timestamp"]) for r in rows]),
        "obs_timestamp": np.array([float(r["obs_timestamp"]) for r in rows]),
        "chunk_id": np.array([int(float(r["chunk_id"])) for r in rows]),
        "step_in_chunk": np.array([int(float(r["step_in_chunk"])) for r in rows]),
        "queue_len": np.array([int(float(r["queue_len"])) for r in rows]),
        "latency": np.array([float(r["latency"]) for r in rows]),
        "sent": np.array([int(float(r["sent"])) if r.get("sent", "") != "" else 1 for r in rows]),
    }
    for key in ("action_raw", "action_processed", "cmd_joint", "actual_joint"):
        vs = [parse_vec(r.get(key)) for r in rows]
        vs = [v for v in vs if v is not None]
        data[key] = np.vstack(vs) if vs else None
    data["n"] = len(rows)
    return data


def moving_average(x, w):
    if w <= 1 or x.shape[0] < w:
        return x.copy()
    pad = w // 2
    xp = np.vstack([np.tile(x[:1], (pad, 1)), x, np.tile(x[-1:], (w - 1 - pad, 1))])
    kernel = np.ones(w) / w
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 0, xp)[: x.shape[0]]


def find_retreats(sig, thresh):
    """在(已平滑)信号中找退步段: 返回每个关节的退步起点步号列表 + 退步幅度。
    退步段 = 一阶差分符号相对前一段翻转, 且该段累计位移幅度 > thresh。"""
    N, A = sig.shape
    d = np.diff(sig, axis=0)                      # (N-1, A)
    per_joint = []
    for j in range(A):
        dj = d[:, j]
        sgn = np.sign(dj)
        sgn[sgn == 0] = 1
        # 分段: 连续同号为一段
        bounds = [0] + (np.where(np.diff(sgn) != 0)[0] + 1).tolist() + [len(dj)]
        segs = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b <= a:
                continue
            disp = sig[b, j] - sig[a, j]          # 该段净位移
            segs.append((a, b, disp))
        # 退步段 = 与“主方向”相反且幅度超阈值的段; 主方向取全程净位移符号
        main_dir = np.sign(sig[-1, j] - sig[0, j]) or 1
        retreats = []
        for a, b, disp in segs:
            if np.sign(disp) == -main_dir and abs(disp) > thresh:
                retreats.append((a, disp, b - a))
        per_joint.append(retreats)
    return per_joint


def autocorr_period(x, dt, max_lag=None):
    """去趋势后自相关, 返回首个正峰 lag(步) 与周期(秒)。"""
    x = x - np.linspace(x[0], x[-1], len(x))      # 去线性趋势
    x = x - x.mean()
    n = len(x)
    if n < 8:
        return None, None
    max_lag = max_lag or n // 2
    ac = np.correlate(x, x, mode="full")[n - 1:]
    ac = ac / (ac[0] + 1e-9)
    ac = ac[:max_lag]
    # 找 lag>=2 的首个局部极大
    for i in range(2, len(ac) - 1):
        if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.15:
            return i, i * dt
    return None, None


def main():
    ap = argparse.ArgumentParser(description="分析故障 episode CSV")
    ap.add_argument("csv_path")
    ap.add_argument("--signal", default="cmd_joint",
                    choices=["cmd_joint", "actual_joint", "action_processed"],
                    help="分析哪一路信号 (默认 cmd_joint=下发指令)")
    ap.add_argument("--smooth", type=int, default=3, help="滑动平均窗口(步), 去高频抖动, 默认3")
    ap.add_argument("--thresh", type=float, default=0.5,
                    help="判定退步段的最小净位移(度), 建议≥死区, 默认0.5")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    d = load_csv(args.csv_path)
    sig_key = args.signal
    if d.get(sig_key) is None:
        raise SystemExit(f"CSV 缺少信号列 {sig_key}")
    sig = d[sig_key][:, :6]                       # 只取 6 关节(末列若是夹爪单独处理)
    A = sig.shape[0] and sig.shape[1]
    N = d["n"]
    dt = float(np.median(np.diff(d["timestamp"]))) if N > 1 else 0.0
    freq = 1.0 / dt if dt > 0 else float("nan")

    print("=" * 72)
    print(f"  故障 episode 分析: {args.csv_path}")
    print("=" * 72)
    print(f"  行数={N}  采样dt≈{dt*1000:.1f}ms (≈{freq:.1f}Hz)  信号={sig_key}  平滑窗={args.smooth}")
    n_chunks = int(d['chunk_id'].max()) if N else 0
    print(f"  chunk 数={n_chunks}  每 chunk 步数(中位)="
          f"{int(np.median(np.bincount(d['chunk_id'])[1:])) if n_chunks>1 else 'NA'}")

    sig_s = moving_average(sig, args.smooth)

    # ---- Q2: 哪些关节退步最多 ----
    retreats = find_retreats(sig_s, args.thresh)
    print("\n  [Q2] 各关节退步统计 (信号去抖后):")
    print(f"      {'关节':<6}{'退步次数':>8}{'累计回退(度)':>14}{'平均回退段长(步)':>18}")
    for j in range(6):
        r = retreats[j]
        cnt = len(r)
        tot = sum(abs(x[1]) for x in r)
        avglen = np.mean([x[2] for x in r]) if r else 0
        print(f"      J{j+1:<5}{cnt:>8}{tot:>14.2f}{avglen:>18.1f}")

    # ---- Q1: 退步是否总在 chunk 边界 ----
    all_starts = []
    for j in range(6):
        for a, disp, ln in retreats[j]:
            all_starts.append(a)                  # a 是 diff 索引, 对应步 a→a+1
    # 估计 n_action_steps(每 chunk 步数)
    nas = int(d["step_in_chunk"].max()) + 1 if N else 20
    print("\n  [Q1] 退步起点 vs chunk 边界:")
    if all_starts:
        starts = np.clip(np.array(all_starts), 0, N - 1)
        sic = d["step_in_chunk"][starts]
        total = len(sic)
        near_start = int((sic <= 1).sum())                 # 新 chunk 开头(重规划跳变)
        near_end = int((sic >= nas - 2).sum())             # chunk 末尾(旧动作耗尽前)
        print(f"      退步事件总数={total}  n_action_steps≈{nas}")
        print(f"      落在 chunk 开头(step_in_chunk≤1)={near_start} ({100.0*near_start/total:.1f}%)")
        print(f"      落在 chunk 末尾(step_in_chunk≥{nas-2})={near_end} ({100.0*near_end/total:.1f}%)")
        print(f"      合计边界相关={near_start+near_end} ({100.0*(near_start+near_end)/total:.1f}%)  "
              f"退步起点 step_in_chunk 中位数={int(np.median(sic))}")
        hist = np.bincount(sic, minlength=nas)[:nas]
        print(f"      step_in_chunk 全分布: {hist.tolist()}")
        bnd_ratio = 4.0 / nas                              # 开头2步+末尾2步占全 chunk 比例(均匀基线)
        frac = (near_start + near_end) / total
        print(f"      均匀基线(边界±2步应占)={100.0*bnd_ratio:.1f}%  实测={100.0*frac:.1f}%")
        if frac > bnd_ratio * 1.8:
            verdict = "边界强相关 → 重规划/队列边界跳变主导"
        elif frac < bnd_ratio * 0.6:
            verdict = "避离边界 → chunk 内部策略内生抖动主导"
        else:
            verdict = "与边界无明显相关 → 策略内生抖动/跟踪问题"
        print(f"      判定: {verdict}")
    else:
        print("      未检测到超阈值退步事件 (调低 --thresh 或换 --signal actual_joint)")

    # ---- Q3: 周期 ----
    print("\n  [Q3] 振荡周期 (去趋势自相关首峰):")
    for j in range(6):
        lag, per_s = autocorr_period(sig_s[:, j], dt)
        if lag:
            print(f"      J{j+1}: 周期≈{lag} 步 ≈{per_s:.2f}s  ({freq/lag:.2f} Hz)")
        else:
            print(f"      J{j+1}: 未找到显著周期峰 (可能是宽谱抖动而非固定周期)")
    bnd_steps = np.where(d["step_in_chunk"] == 0)[0]
    if len(bnd_steps) > 1:
        bper = float(np.median(np.diff(bnd_steps)))
        print(f"      chunk 边界周期={bper:.1f} 步 ≈{bper*dt:.2f}s (应≈n_action_steps)")

    # ---- 时延 & 跟踪误差 ----
    print("\n  [附] 时延与跟踪:")
    print(f"      latency  mean={d['latency'].mean()*1000:.1f}ms  "
          f"p95={np.percentile(d['latency'],95)*1000:.1f}ms  max={d['latency'].max()*1000:.1f}ms")
    over = int((d['latency'] > dt).sum()) if dt > 0 else 0
    print(f"      latency 超过控制周期({dt*1000:.0f}ms)的步数={over}/{N} "
          f"{'⚠ 主循环被推理拖慢→掉帧' if over > N*0.1 else ''}")
    if d.get("cmd_joint") is not None and d.get("actual_joint") is not None:
        cmd = d["cmd_joint"][:, :6]
        act = d["actual_joint"][:, :6]
        # 第 N 行 actual 对应第 N-1 行 cmd 的执行结果
        err = np.abs(act[1:] - cmd[:-1])
        print(f"      跟踪误差 |actual(t)-cmd(t-1)| mean={err.mean():.2f}°  "
              f"p95={np.percentile(err,95):.2f}°  max={err.max():.2f}°")
        print(f"      死区跳过步数(sent=0)={int((d['sent']==0).sum())}/{N}")

    # ---- 画图 ----
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n  未安装 matplotlib, 跳过画图")
            return
        fig, axes = plt.subplots(6, 1, figsize=(12, 13), sharex=True)
        bnds = np.where(d["step_in_chunk"] == 0)[0]
        for j in range(6):
            ax = axes[j]
            ax.plot(sig[:, j], lw=0.8, alpha=0.6, label="raw")
            ax.plot(sig_s[:, j], lw=1.4, label=f"smooth{args.smooth}")
            for b in bnds:
                ax.axvline(b, color="r", ls="--", lw=0.5, alpha=0.5)
            for a, disp, ln in retreats[j]:
                ax.axvspan(a, a + ln, color="orange", alpha=0.25)
            ax.set_ylabel(f"J{j+1}(°)")
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.legend(fontsize=8, loc="upper right")
        axes[-1].set_xlabel("step")
        fig.suptitle(f"Fault episode | {sig_key} | red=chunk boundary, orange=retreat")
        fig.tight_layout()
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=110)
        print(f"\n  已写图: {args.plot}")


if __name__ == "__main__":
    main()
