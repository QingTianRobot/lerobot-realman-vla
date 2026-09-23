#!/usr/bin/env python3
"""
奥比中光 Gemini 305 彩色相机接口 (pyorbbecsdk)

接口对齐 RealSenseCamera，便于采集/推理脚本无缝替换：
  - 独立后台线程持续读取最新帧
  - get_frame() 返回 BGR (H, W, 3) uint8
  - is_active / close()

依赖:
  pip install pyorbbecsdk

⚠️ 库冲突坑：pyorbbecsdk 的 .so 可能把 libOrbbecSDK.so.2 解析到 ROS 的旧库，
   导致 `undefined symbol: ob_application_config_set_struct`。运行前需让自带库优先:
     export LD_LIBRARY_PATH=$(python -c "import pyorbbecsdk,os;print(os.path.dirname(pyorbbecsdk.__file__))"):$LD_LIBRARY_PATH
   （若已装 vendor/OrbbecSDK_v2 的 2.9.3 库，也可指向其 install/lib）

自检:
  python hardware/orbbec_camera.py [--serial CV2T66100096]
"""

import os
import time
import threading
import argparse

import numpy as np
import cv2

from pyorbbecsdk import (
    Context,
    Config,
    Pipeline,
    OBFormat,
    OBSensorType,
)


class OrbbecCamera:
    """奥比中光相机异步采集

    使用独立线程持续读取彩色帧，get_frame() 返回最新帧 (BGR)。
    优先 MJPG，回退 RGB；两者都归一化输出为 (height, width, 3) uint8 BGR。
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

        self.ctx = None
        self.pipeline = None
        self._fmt = None  # 实际协商到的彩色格式 (MJPG / RGB)
        # 诊断计数：区分"没收到帧" / "解码失败" / "正常存储"
        self._n_color_frames = 0
        self._n_decoded = 0
        self._n_stored = 0

        try:
            self.ctx = Context()
            device = self._select_device()
            if device is None:
                raise RuntimeError(
                    f"未找到 Orbbec 设备 (serial={self.serial_number or 'any'})"
                )

            self.pipeline = Pipeline(device)
            config = Config()
            profile = self._pick_color_profile()
            if profile is None:
                raise RuntimeError("未找到匹配的彩色流配置")
            self._fmt = profile.get_format()
            config.enable_stream(profile)
            self.pipeline.start(config)

            self.is_active = True
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
        except Exception as e:
            print(f"[!] Orbbec camera {self.serial_number or ''}: {e}")
            self.is_active = False

    def _select_device(self):
        """按序列号选择设备；未指定序列号则取第 0 个。"""
        device_list = self.ctx.query_devices()
        count = device_list.get_count()
        if count == 0:
            return None
        if not self.serial_number:
            return device_list.get_device_by_index(0)
        for i in range(count):
            dev = device_list.get_device_by_index(i)
            try:
                sn = dev.get_device_info().get_serial_number()
            except Exception:
                continue
            if str(sn) == self.serial_number:
                return dev
        return None

    def _pick_color_profile(self):
        """优先 MJPG@指定分辨率帧率，回退 RGB，再回退默认彩色配置。"""
        profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        for fmt in (OBFormat.MJPG, OBFormat.RGB):
            try:
                return profile_list.get_video_stream_profile(
                    self.width, self.height, fmt, self.fps
                )
            except Exception:
                continue
        # 分辨率/帧率不完全匹配时，退化到该格式的任意可用配置
        for fmt in (OBFormat.MJPG, OBFormat.RGB):
            try:
                return profile_list.get_video_stream_profile(0, 0, fmt, 0)
            except Exception:
                continue
        try:
            return profile_list.get_default_video_stream_profile()
        except Exception:
            return None

    def _to_bgr(self, color_frame):
        """把彩色帧转成 (H, W, 3) uint8 BGR。"""
        data = color_frame.get_data()
        h = color_frame.get_height()
        w = color_frame.get_width()
        fmt = color_frame.get_format()

        if fmt == OBFormat.MJPG:
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return None
        elif fmt == OBFormat.RGB:
            img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif fmt == OBFormat.BGR:
            img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
        else:
            # 兜底：按 RGB 处理
            try:
                img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except Exception:
                return None

        if img.shape[:2] != (self.height, self.width):
            img = cv2.resize(img, (self.width, self.height))
        return img

    def _update_loop(self):
        while not self.stopped and self.is_active:
            try:
                frames = self.pipeline.wait_for_frames(2000)
                if not frames:
                    continue
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                self._n_color_frames += 1
                img = self._to_bgr(color_frame)
                if img is not None:
                    self._n_decoded += 1
                if img is not None and img.shape == self.target_shape:
                    with self.lock:
                        self.latest_color = img.copy()
                    self._n_stored += 1
            except Exception:
                pass

    def get_frame(self):
        with self.lock:
            return self.latest_color.copy()

    def close(self):
        self.stopped = True
        if self.is_active and self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.is_active = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orbbec Gemini 305 相机自检")
    parser.add_argument("--serial", type=str, default="CV2T66100096",
                        help="相机序列号 (留空取第一个设备)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=float, default=10.0,
                        help="等待首个有效(非全零)帧的最长秒数")
    parser.add_argument("--save", type=str, default="",
                        help="保存帧的图片路径 (默认 orbbec_<serial>_<时间戳>.png)")
    args = parser.parse_args()

    cam = OrbbecCamera(args.serial, args.width, args.height, args.fps)
    if not cam.is_active:
        print("相机初始化失败")
        raise SystemExit(1)

    print(f"相机已启动 (format={cam._fmt})，等待首帧 (最长 {args.warmup}s)...")

    # 轮询等待第一帧非全零画面：UVC/MJPG 起流较慢，首次 wait_for_frames 可能耗掉 1~2s，
    # 固定 sleep(1.0) 会在后台存帧前就取到初始的全零 latest_color。
    deadline = time.time() + args.warmup
    frame = cam.get_frame()
    while time.time() < deadline and np.count_nonzero(frame) == 0:
        time.sleep(0.1)
        frame = cam.get_frame()

    ok = np.count_nonzero(frame) > 0
    print(f"收帧统计: 彩色帧={cam._n_color_frames} 解码成功={cam._n_decoded} 已存储={cam._n_stored}")
    if ok:
        print(f"frame shape={frame.shape} dtype={frame.dtype} "
              f"min={frame.min()} max={frame.max()} mean={frame.mean():.1f} nonzero=True")
    else:
        print(f"frame shape={frame.shape} 仍为全零 nonzero=False"
              f"（{args.warmup}s 内未取到有效画面）")

    save_path = args.save or f"orbbec_{args.serial or 'cam'}_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(save_path, frame)
    print(f"已保存: {os.path.abspath(save_path)}")

    cam.close()
    print("已关闭")
    if not ok:
        raise SystemExit(2)
