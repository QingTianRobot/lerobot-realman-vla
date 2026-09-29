#!/usr/bin/env python3
"""
Pika Sense 手持夹爪(主手)封装 — 只读开合行程, 用于遥操作机械臂夹爪(从手)

Pika Sense 是带编码器的可动主手设备(还带相机/IMU/Vive); 本封装只取夹爪开合行程
get_gripper_distance()(mm), 归一化到 0~1 (0=闭合, 1=张开), 供采集脚本把它映射到
知行从手夹爪 (ChangingtekGripper.move_normalized)。相机/IMU/Vive 一律不启用。

串口: 默认 /dev/tty_pika_left (见 /etc/udev/rules.d/98-usb-serial.rules 的 PIKA_LEFT)。
依赖: vendor/pika_sdk (submodule), 运行期只需 pyserial (Sense 模块无重依赖)。

归一化约定 (对外语义, 与从手夹爪一致):
  v = 1.0 -> 张开
  v = 0.0 -> 闭合
主手(Sense)与从手(知行)方向一致, 故默认 invert=False; 若某台 Sense 读数方向相反, 置 invert=True。

行程标定 (Sense 编码器零点/行程因个体而异, 强烈建议标定一次):
  python hardware/pika_gripper.py --calibrate
  按提示先【完全闭合】再【完全张开】各采一点, 写入 hardware/pika_calibration.json。
  未标定时用理论行程 0~109mm 兜底 (由 pika.sense.get_distance 推得), 精度较差。

自检 (实时打印行程与归一化值):
  python hardware/pika_gripper.py [--port /dev/tty_pika_left] [--hz 20]
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

# 把 vendor/pika_sdk 根目录加入模块搜索路径 (submodule, 未 pip 安装)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PIKA_SDK = os.path.abspath(os.path.join(_THIS_DIR, "..", "vendor", "pika_sdk"))
if _PIKA_SDK not in sys.path:
    sys.path.insert(0, _PIKA_SDK)

from pika.sense import Sense  # noqa: E402

# 理论行程兜底 (由 pika.sense.get_distance 推得): 闭合≈0mm, 全开≈109mm。未标定时使用。
DEFAULT_MIN_MM = 0.0
DEFAULT_MAX_MM = 109.0


class PikaGripperMaster:
    """Pika Sense 主手夹爪: 读取开合行程并归一化 (0=闭合, 1=张开)。只读, 不驱动电机。

    Sense 的串口读数由 pika 内部后台线程缓存, get_distance_mm()/get_normalized() 取缓存,
    不阻塞; 可在采集主循环/遥操线程里高频调用。
    """

    def __init__(self, port="/dev/tty_pika_left", min_mm=DEFAULT_MIN_MM,
                 max_mm=DEFAULT_MAX_MM, name="pika_left",
                 calibration_file=None, use_calibration=True, invert=False):
        self.port = port
        self.name = name
        self.min_mm = float(min_mm)
        self.max_mm = float(max_mm)
        # 主手读数方向与归一化约定相反时置 True (默认一致, 无需反转)
        self.invert = bool(invert)

        # 持久化标定文件: 由 --calibrate 写入, 封装层自动加载, 保证自检与采集同一行程基准。
        self.calibration_file = calibration_file or os.path.join(
            _THIS_DIR, "pika_calibration.json")
        self.use_calibration = use_calibration
        if self.use_calibration:
            cal = self._load_calibration()
            if cal:
                self.min_mm = float(cal.get("min_mm", self.min_mm))
                self.max_mm = float(cal.get("max_mm", self.max_mm))
                print(f"[i] 已加载 Pika 标定({self.name}): "
                      f"min={self.min_mm:.1f}mm max={self.max_mm:.1f}mm "
                      f"(来自 {os.path.basename(self.calibration_file)})")

        self.connected = False
        self.sense = Sense(port)

    # ---------- 生命周期 ----------
    def connect(self):
        """连接 Sense 串口 (内部会启动后台读数线程)。返回是否连接成功。"""
        self.connected = bool(self.sense.connect())
        if not self.connected:
            print(f"[!] Pika Sense 连接失败: {self.port}")
        return self.connected

    def disconnect(self):
        try:
            self.sense.disconnect()
        except Exception:
            pass
        self.connected = False

    # ---------- 反馈 ----------
    def get_distance_mm(self):
        """读取当前开合行程 (mm); 未连接/读失败返回 None。"""
        if not self.connected:
            return None
        try:
            return float(self.sense.get_gripper_distance())
        except Exception:
            return None

    def get_normalized(self):
        """行程(mm) → 归一化 0~1 (0=闭合, 1=张开); 未连接/读失败返回 None。"""
        d = self.get_distance_mm()
        if d is None:
            return None
        span = self.max_mm - self.min_mm
        if span <= 0:
            return 0.0
        v = (d - self.min_mm) / span
        if self.invert:
            v = 1.0 - v
        return max(0.0, min(1.0, float(v)))

    # ---------- 行程标定 (两点: 全闭 / 全开) ----------
    def _sample_distance(self, hint, samples=8):
        input(hint + " —— 摆到位后按回车采样...")
        time.sleep(0.3)   # 等编码器读数稳定
        vals = []
        for _ in range(samples):
            d = self.get_distance_mm()
            if d is not None:
                vals.append(d)
            time.sleep(0.03)
        if not vals:
            raise RuntimeError("读不到 Pika 行程, 请检查串口连接")
        return sum(vals) / len(vals)

    def calibrate(self, save=True):
        """交互式两点标定: 提示全闭/全开各采一点, 取小者为 min、大者为 max 并持久化。

        返回 dict(min_mm/max_mm/closed_raw/open_raw)。标定后立即生效。
        """
        if not self.connected:
            raise RuntimeError("calibrate 需在 connect() 成功后调用")
        closed = self._sample_distance("请把 Pika 夹爪【完全闭合】")
        opened = self._sample_distance("请把 Pika 夹爪【完全张开】")
        lo, hi = min(closed, opened), max(closed, opened)
        if hi - lo < 5.0:
            raise RuntimeError(f"标定行程异常(过小): closed={closed:.1f} open={opened:.1f}")
        self.min_mm, self.max_mm = lo, hi
        result = {"min_mm": round(lo, 2), "max_mm": round(hi, 2),
                  "closed_raw": round(closed, 2), "open_raw": round(opened, 2)}
        if save:
            self._save_calibration(result)
        return result

    # ---------- 标定持久化 ----------
    def _load_calibration(self):
        """读取本主手 (按 name 索引) 的标定条目; 无文件/无条目返回 None。"""
        try:
            with open(self.calibration_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            entry = data.get(self.name)
            return entry if isinstance(entry, dict) else None
        except (OSError, ValueError):
            return None

    def _save_calibration(self, result):
        """把标定结果按 name 合并写入 JSON (保留其他主手条目)。"""
        data = {}
        try:
            with open(self.calibration_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}
        entry = dict(result)
        entry.update({
            "port": self.port,
            "calibrated_at": datetime.now().isoformat(timespec="seconds"),
        })
        data[self.name] = entry
        with open(self.calibration_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pika Sense 主手夹爪自检 (读取开合行程)")
    parser.add_argument("--port", type=str, default="/dev/tty_pika_left",
                        help="Pika Sense 串口 (默认 /dev/tty_pika_left)")
    parser.add_argument("--name", type=str, default="pika_left", help="标定条目名 (多主手区分)")
    parser.add_argument("--hz", type=float, default=20.0, help="自检打印频率")
    parser.add_argument("--calibrate", action="store_true",
                        help="两点标定行程 (全闭/全开), 持久化到 pika_calibration.json")
    parser.add_argument("--invert", action="store_true",
                        help="反转归一化方向 (Sense 读数与 0=闭合/1=张开 约定相反时用)")
    parser.add_argument("--no-calibration-file", action="store_true",
                        help="忽略已持久化标定, 仅用理论行程 0~109mm")
    args = parser.parse_args()

    m = PikaGripperMaster(port=args.port, name=args.name, invert=args.invert,
                          use_calibration=not args.no_calibration_file)
    print("连接:", m.connect())
    if not m.connected:
        print("无法连接 Pika Sense, 请检查串口/接线。")
        raise SystemExit(1)

    try:
        time.sleep(0.3)
        if args.calibrate:
            print("\n== 标定行程 (按提示全闭/全开各采一点) ==")
            cal = m.calibrate()
            print(f"  闭合={cal['closed_raw']}mm 张开={cal['open_raw']}mm")
            print(f"  归一化范围: min={cal['min_mm']}mm max={cal['max_mm']}mm")
            print(f"  已写入: {m.calibration_file}")
        else:
            print(f"\n== 实时读数 (Ctrl+C 退出) 行程 min={m.min_mm:.1f} max={m.max_mm:.1f}mm ==")
            period = 1.0 / max(1.0, args.hz)
            while True:
                d = m.get_distance_mm()
                v = m.get_normalized()
                if d is not None:
                    print(f"\r  行程 {d:7.2f} mm  归一化 {v:.3f} "
                          f"({'张开' if v > 0.5 else '闭合'})   ", end="", flush=True)
                time.sleep(period)
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        m.disconnect()
        print("已断开")
