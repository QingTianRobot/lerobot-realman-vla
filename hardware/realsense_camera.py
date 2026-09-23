#!/usr/bin/env python3
"""
Intel RealSense D435 彩色相机接口 (pyrealsense2)

接口对齐 OrbbecCamera，便于采集/推理脚本无缝替换：
  - 独立后台线程持续读取最新帧
  - get_frame() 返回 BGR (H, W, 3) uint8
  - is_active / close()

依赖:
  pip install pyrealsense2>=2.55.1

⚠️ 设备占用坑：上次脚本非正常退出可能导致 "Device is already in use"，
   解决方式：pkill -f realsense 或物理拔插 USB。
   本模块启动时会尝试 hardware_reset() 来恢复设备。

自检:
  python hardware/realsense_camera.py [--serial 262322074840]
"""

import time
import threading
import argparse

import numpy as np
import pyrealsense2 as rs


class RealSenseCamera:
    """Intel RealSense 相机异步采集

    使用独立线程持续读取彩色帧，get_frame() 返回最新帧 (BGR)。
    关闭自动曝光以确保采集一致性。
    """

    def __init__(self, serial_number=None, width=640, height=480, fps=30):
        self.serial_number = str(serial_number) if serial_number else ""
        self.width, self.height = width, height
        self.fps = fps
        self.target_shape = (height, width, 3)
        self.latest_color = np.zeros(self.target_shape, dtype=np.uint8)
        self.lock = threading.Lock()
        self.stopped = False
        self.is_active = False

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # 尝试硬件复位（解决相机被占用问题）
        self._try_hardware_reset()

        if self.serial_number:
            self.config.enable_device(self.serial_number)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        try:
            profile = self.pipeline.start(self.config)
            self.is_active = True

            # 固定曝光（避免亮度波动影响训练）
            color_sensor = profile.get_device().first_color_sensor()
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            if color_sensor.supports(rs.option.exposure):
                color_sensor.set_option(rs.option.exposure, 150)

            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
        except Exception as e:
            print(f"[!] RealSense camera {self.serial_number or ''}: {e}")
            self.is_active = False

    def _try_hardware_reset(self):
        """尝试硬件复位以释放上次未正常退出的设备锁。"""
        try:
            ctx = rs.context()
            for dev in ctx.query_devices():
                sn = dev.get_info(rs.camera_info.serial_number)
                if not self.serial_number or sn == self.serial_number:
                    dev.hardware_reset()
                    time.sleep(2)
                    break
        except Exception:
            pass

    def _update_loop(self):
        while not self.stopped and self.is_active:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
                color_frame = frames.get_color_frame()
                if color_frame:
                    frame_data = np.asanyarray(color_frame.get_data())
                    if frame_data.shape == self.target_shape:
                        with self.lock:
                            self.latest_color = frame_data.copy()
            except Exception:
                pass

    def get_frame(self):
        """返回最新一帧 BGR (H, W, 3) uint8。"""
        with self.lock:
            return self.latest_color.copy()

    def close(self):
        """停止采集并释放设备。"""
        self.stopped = True
        if self.is_active and self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.is_active = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intel RealSense D435 相机自检")
    parser.add_argument("--serial", type=str, default="262322074840",
                        help="相机序列号 (留空取第一个设备)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    print(f"正在初始化 RealSense D435 (serial={args.serial or 'any'}, "
          f"{args.width}x{args.height}@{args.fps}fps)...")
    cam = RealSenseCamera(args.serial, args.width, args.height, args.fps)
    if not cam.is_active:
        print("相机初始化失败")
        print("排查建议:")
        print("  1. 检查 USB 连接 (lsusb | grep Intel)")
        print("  2. 杀掉残留进程: pkill -f realsense")
        print("  3. 物理拔插 USB 后重试")
        raise SystemExit(1)

    print("相机已启动，等待帧稳定 (1s)...")
    time.sleep(1.0)

    frame = cam.get_frame()
    nonzero = np.count_nonzero(frame) > 0
    print(f"frame shape={frame.shape} dtype={frame.dtype} nonzero={nonzero}")

    if not nonzero:
        print("[!] 警告: 帧全黑，可能曝光设置异常或镜头被遮挡")

    # 简单 FPS 测试 (采集 30 帧计算耗时)
    print("\n--- FPS 测试 (30 帧) ---")
    fps_count = 30
    t0 = time.time()
    prev = cam.get_frame()
    updated = 0
    for _ in range(fps_count):
        time.sleep(1.0 / args.fps)
        cur = cam.get_frame()
        if not np.array_equal(cur, prev):
            updated += 1
        prev = cur
    elapsed = time.time() - t0
    actual_fps = updated / elapsed if elapsed > 0 else 0
    print(f"  耗时: {elapsed:.2f}s, 更新帧数: {updated}/{fps_count}, "
          f"实际 FPS: {actual_fps:.1f}")

    # 保存测试帧到文件
    out_path = "outputs/realsense_test_frame.png"
    import cv2 as _cv2
    _cv2.imwrite(out_path, frame)
    print(f"\n测试帧已保存: {out_path}")

    cam.close()
    print("已关闭")
