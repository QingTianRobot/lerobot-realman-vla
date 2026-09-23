#!/usr/bin/env python3
"""
知行 (changingtek) RTU 平动手夹爪封装

在 vendor/changingtek_rtu_sdk 的 GripperManager 之上做薄封装，暴露归一化接口，
供采集/推理脚本直接调用。夹爪走独立 RS-485 串口 (默认 /dev/realman/gripper_left)，
与机械臂完全解耦 —— 不再经机械臂 Modbus 透传，也无需工具端供电设置。

归一化约定 (对外语义, 与数据第 7 维保持一致):
  v = 1.0 -> 张开
  v = 0.0 -> 闭合
  推理端沿用 `>0.5 判为开` 的阈值。
  ⚠ 本机知行平动手物理行程为「小端(min_position, mm≈0)=张开、大端(max_position, mm≈86)=闭合」,
  与归一化约定相反, 故默认 invert=True 在 归一化<->物理位置 间反转映射, 使
  v=0 真正驱动到物理闭合(大端)、v=1 到物理张开(小端)。换用方向相反的夹爪时置 invert=False。
  min/max_position 是采集与推理必须一致的归一化基准; 构造时会自动加载
  hardware/gripper_calibration.json (若存在), 保证自检/采集/推理三端同一行程。

行程标定 (实测机械限位, 避免命令顶到硬限位长期堵转发热):
  python hardware/changingtek_gripper.py --slave-id 2 --calibrate [--margin 0.02]
  会驱动到开/合限位实测真实行程, 上限回缩 margin(默认2%) 后写入标定文件。
  标定只需在换夹爪/机械限位变化时执行一次, 平时自检与采集会自动加载该文件。

依赖:
  pip install minimalmodbus pyserial

自检:
  python hardware/changingtek_gripper.py [--port /dev/realman/gripper_left] [--slave-id 2]
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

# 把 vendor/changingtek_rtu_sdk 根目录加入模块搜索路径
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VENDOR_SDK = os.path.join(_THIS_DIR, "..", "vendor", "changingtek_rtu_sdk")
_VENDOR_SDK = os.path.abspath(_VENDOR_SDK)
if _VENDOR_SDK not in sys.path:
    sys.path.insert(0, _VENDOR_SDK)

from src.gripper_driver import GripperManager  # noqa: E402


class ChangingtekGripper:
    """知行 RTU 夹爪归一化控制器 (异步下发，读取缓存反馈)。"""

    def __init__(self, port="/dev/realman/gripper_left", slave_id=2,
                 baudrate=115200, min_position=0, max_position=9000,
                 name="left", poll_hz=25, speed_pct=50, force_pct=60,
                 calibration_file=None, use_calibration=True, invert=True):
        self.port = port
        self.slave_id = int(slave_id)
        self.baudrate = baudrate
        self.min_position = int(min_position)
        self.max_position = int(max_position)
        self.name = name
        self.poll_hz = poll_hz
        self.speed_pct = speed_pct
        self.force_pct = force_pct
        # 物理行程方向与归一化约定相反时置 True (本机知行平动手: 小端=张开、大端=闭合)
        self.invert = bool(invert)

        # 持久化标定文件: 由自检 --calibrate 写入, 封装层自动加载,
        # 保证自检 / collect_data / inference 三者归一化行程完全一致。
        self.calibration_file = calibration_file or os.path.join(
            _THIS_DIR, "gripper_calibration.json")
        self.use_calibration = use_calibration
        if self.use_calibration:
            cal = self._load_calibration()
            if cal:
                self.min_position = int(cal.get("min_position", self.min_position))
                self.max_position = int(cal.get("max_position", self.max_position))
                print(f"[i] 已加载夹爪标定({self.name}): "
                      f"min={self.min_position} max={self.max_position} "
                      f"(来自 {os.path.basename(self.calibration_file)})")

        self.connected = False
        self._last_cmd = None  # 最近一次归一化命令 (0~1)

        self.mgr = GripperManager()
        self.mgr.add_bus(port, baudrate=baudrate, poll_hz=poll_hz)
        self.mgr.add_gripper(
            port, slave_id=self.slave_id, name=name,
            speed_pct=speed_pct, force_pct=force_pct,
            min_position=self.min_position, max_position=self.max_position,
        )

    # ---------- 生命周期 ----------
    def connect(self, enable_retries=5, enable_delay=0.15):
        """连接总线、使能夹爪(带重试), 再启动后台轮询线程。返回是否连接成功。

        先使能再开轮询: 避免使能与后台读反馈争用半双工总线导致 CRC 错位;
        使能失败时清缓冲并重试, 吸收偶发的 "Checksum error in rtu mode"。
        """
        result = self.mgr.connect_all()
        self.connected = bool(result.get(self.port, False))
        if not self.connected:
            print(f"[!] 夹爪串口未连接: {self.port} ({result})")
            return False

        self.mgr.set_active(self.name)
        self._enable_with_retry(retries=enable_retries, delay=enable_delay)
        self.mgr.start_all()
        return self.connected

    def _enable_with_retry(self, retries=5, delay=0.15):
        """使能夹爪, 失败时清空串口缓冲后重试。返回是否使能成功。"""
        dev = self.mgr.get(self.name)
        bus = self.mgr.get_bus(self.port)
        for attempt in range(1, retries + 1):
            try:
                # 清掉可能残留的字节, 防止上一帧尾巴污染本次事务
                try:
                    bus._sdk.instrument.serial.reset_input_buffer()
                except Exception:
                    pass
                dev.enable(True)
                return True
            except Exception as e:
                print(f"[!] 夹爪使能失败(第 {attempt}/{retries} 次): {e}")
                time.sleep(delay)
        print("[!] 夹爪使能重试仍失败, 请检查接线/从站地址/波特率。")
        return False

    def disable(self):
        try:
            self.mgr.get(self.name).enable(False)
        except Exception:
            pass

    def disconnect(self):
        try:
            self.mgr.stop_all()
        except Exception:
            pass
        try:
            self.mgr.disconnect_all()
        except Exception:
            pass
        self.connected = False

    # ---------- 运动 (归一化 0~1) ----------
    def move_normalized(self, v):
        """下发归一化目标位置 (0=闭合, 1=张开)。异步，不阻塞主循环。

        invert=True 时反转映射: v=0 -> max_position(物理闭合), v=1 -> min_position(物理张开)。
        """
        v = max(0.0, min(1.0, float(v)))
        self._last_cmd = v  # 缓存归一化语义值 (0=闭合,1=张开), 供无有效反馈时回退
        span = self.max_position - self.min_position
        phys = (1.0 - v) if self.invert else v
        position = int(round(self.min_position + phys * span))
        self.mgr.request_move(self.name, position)
        return position

    def move_pct(self, pct):
        """按百分比下发 (0~100)。"""
        return self.move_normalized(float(pct) / 100.0)

    def open(self):
        return self.move_normalized(1.0)

    def close(self):
        return self.move_normalized(0.0)

    # ---------- 反馈 ----------
    def get_position_normalized(self, refresh=False):
        """读取当前归一化位置 (0=闭合, 1=张开)。

        后台轮询线程已按 poll_hz 缓存反馈；refresh=True 时主动读一次。
        尚未拿到有效反馈时，回退到最近一次命令值 (无命令则 0.0)。
        """
        dev = self.mgr.get(self.name)
        try:
            if refresh:
                dev.read_feedback()
            pos = dev.feedback.get("position", None)
        except Exception:
            pos = None

        if pos is None:
            return 0.5 if self._last_cmd is None else float(self._last_cmd)

        span = self.max_position - self.min_position
        if span <= 0:
            return 0.0
        v = (pos - self.min_position) / span
        if self.invert:
            v = 1.0 - v
        return max(0.0, min(1.0, float(v)))

    def get_feedback(self, refresh=True):
        """返回原始反馈 dict (position/position_mm/speed/current/ready/alarm...)。"""
        dev = self.mgr.get(self.name)
        if refresh:
            try:
                dev.read_feedback()
            except Exception:
                pass
        return dict(dev.feedback)

    # ---------- 行程标定 (实测机械限位) ----------
    def _drive_to_settle(self, target_raw, settle_s=0.4, timeout_s=4.0,
                         sample_hz=20.0, tol=20):
        """同步驱动到 target_raw (绕过归一化 clamp), 等位置稳定后返回实测原始位置。

        用于标定: 命令值超过物理行程时夹爪会顶到限位并堵转 (torque_reached),
        位置不再变化即视为到达限位。稳定 settle_s 秒即返回, 避免长时间堵转发热。
        """
        dev = self.mgr.get(self.name)
        bus = self.mgr.get_bus(self.port)
        # 绕过 dev.move_to 的 min/max clamp, 直接用官方 SDK 下发原始目标
        bus.transaction(
            dev.slave_id,
            lambda sdk: sdk.temp_move(
                position_mm=int(target_raw), speed_pct=dev.speed_pct,
                force_pct=dev.force_pct, accel=dev.accel, decel=dev.decel),
        )
        period = 1.0 / max(1.0, sample_hz)
        need_stable = max(1, int(settle_s * sample_hz))
        deadline = time.time() + timeout_s
        last_pos, stable = None, 0
        while time.time() < deadline:
            time.sleep(period)
            try:
                fb = dev.read_feedback()
            except Exception:
                continue
            pos = fb.get("position")
            if pos is None:
                continue
            if last_pos is not None and abs(pos - last_pos) <= tol and fb.get("speed", 0) == 0:
                stable += 1
                if stable >= need_stable:
                    return int(pos)
            else:
                stable = 0
            last_pos = pos
        return int(last_pos) if last_pos is not None else int(target_raw)

    def calibrate(self, margin=0.02, open_target=None, close_target=0, save=True):
        """驱动到开/合机械限位实测真实行程, 减去安全余量后作为归一化范围并持久化。

        margin: 上限相对实测行程回缩的比例 (默认 2%), 避免命令顶到硬限位长期堵转。
        返回 dict(raw_min/raw_max/min_position/max_position)。标定后 min/max 立即生效。
        注: min/max_position 只记录机械行程两端(小端/大端), 不编码开合方向;
        开合方向由 invert 在归一化层处理, 故标定结果与方向无关, 反转 invert 无需重标。
        """
        if not self.connected:
            raise RuntimeError("calibrate 需在 connect() 成功后调用")
        dev = self.mgr.get(self.name)
        if open_target is None:
            open_target = self.max_position + 3000  # 命令超过物理上限以确保顶到限位

        raw_max = self._drive_to_settle(open_target)
        raw_min = self._drive_to_settle(close_target)
        if raw_max <= raw_min:
            raise RuntimeError(f"标定失败: 实测行程异常 raw_min={raw_min} raw_max={raw_max}")

        span = raw_max - raw_min
        cal_min = int(round(raw_min))
        cal_max = int(round(raw_max - margin * span))  # 上限留余量, 下限贴合实测闭合位

        self.min_position, self.max_position = cal_min, cal_max
        dev.min_position, dev.max_position = cal_min, cal_max
        dev._last_params = None  # 行程变了, 强制下次全量下发

        result = {"raw_min": int(raw_min), "raw_max": int(raw_max),
                  "min_position": cal_min, "max_position": cal_max}
        if save:
            self._save_calibration(result, margin=margin)
        return result

    # ---------- 标定持久化 ----------
    def _load_calibration(self):
        """读取本夹爪 (按 name 索引) 的标定条目; 无文件/无条目返回 None。"""
        try:
            with open(self.calibration_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            entry = data.get(self.name)
            return entry if isinstance(entry, dict) else None
        except (OSError, ValueError):
            return None

    def _save_calibration(self, result, margin=0.02):
        """把标定结果按 name 合并写入 JSON (保留其他夹爪条目)。"""
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
            "port": self.port, "slave_id": self.slave_id,
            "margin": margin, "calibrated_at": datetime.now().isoformat(timespec="seconds"),
        })
        data[self.name] = entry
        with open(self.calibration_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="知行 RTU 夹爪自检 (开合循环)")
    parser.add_argument("--port", type=str, default="/dev/realman/gripper_left")
    parser.add_argument("--slave-id", type=int, default=2)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--max-position", type=int, default=9000)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--calibrate", action="store_true",
                        help="驱动到开/合机械限位实测行程, 持久化标定到 gripper_calibration.json")
    parser.add_argument("--margin", type=float, default=0.02,
                        help="标定时上限相对实测行程的安全余量比例 (默认 0.02=2%%)")
    parser.add_argument("--no-calibration-file", action="store_true",
                        help="忽略已持久化的标定文件, 仅用命令行 max-position")
    args = parser.parse_args()

    g = ChangingtekGripper(port=args.port, slave_id=args.slave_id,
                           baudrate=args.baudrate, max_position=args.max_position,
                           use_calibration=not args.no_calibration_file)
    print("连接:", g.connect())
    if not g.connected:
        print("无法连接夹爪，请检查端口/从站地址/接线。")
        raise SystemExit(1)

    try:
        time.sleep(0.3)
        if args.calibrate:
            print("\n== 标定行程 (驱动到机械限位, 请确保夹爪周围无障碍) ==")
            cal = g.calibrate(margin=args.margin)
            print(f"  实测行程: raw_min={cal['raw_min']} raw_max={cal['raw_max']}")
            print(f"  标定后归一化范围: min={cal['min_position']} "
                  f"max={cal['max_position']} (余量 {args.margin:.0%})")
            print(f"  已写入: {g.calibration_file}")
            time.sleep(0.3)
        print("\n初始反馈:", g.get_feedback())
        for i in range(args.cycles):
            print(f"\n== 循环 {i + 1}: 张开 ==")
            g.open()
            time.sleep(1.5)
            fb = g.get_feedback()
            print(f"  实际位置: {fb.get('position')} (mm={fb.get('position_mm')}) "
                  f"归一化: {round(g.get_position_normalized(), 3)} ready={fb.get('ready')}")
            print(f"== 循环 {i + 1}: 闭合 ==")
            g.close()
            time.sleep(1.5)
            fb = g.get_feedback()
            print(f"  实际位置: {fb.get('position')} (mm={fb.get('position_mm')}) "
                  f"归一化: {round(g.get_position_normalized(), 3)} ready={fb.get('ready')}")
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        g.disable()
        g.disconnect()
        print("已失能并断开")
