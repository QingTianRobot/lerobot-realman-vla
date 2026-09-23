#!/usr/bin/env python3
"""
数据采集脚本 — RealMan RM65 + Vive Tracker + 顶部D435/腕部Orbbec305 + 知行夹爪

支持两种模式：
  1. Vive 遥操作模式：通过 Vive Tracker 远程控制机械臂
  2. 示教模式 (--teaching)：手动拖动机械臂，无需 Vive

硬件：
  - 机械臂 RM65：TCP/IP (默认 192.168.5.123)
  - 顶部相机：Intel RealSense D435 (pyrealsense2)
  - 腕部相机：奥比中光 Gemini 305 (pyorbbecsdk)
  - 夹爪：知行 RTU 平动手，独立串口 (/dev/realman/gripper_left)，与机械臂解耦

采集数据格式：HDF5
  - observations/qpos: (N, 7)   [6关节角 + 夹爪(归一化 0~1, 1=张开)]
  - observations/images/camera_global: (N, H, W, 3)
  - observations/images/camera_left: (N, H, W, 3)
  - action: (N, 7)              [下一帧的 qpos]
  - timestamps: (N,)            [相对时间戳(秒)]

用法:
  # Vive 遥操作
  python scripts/collect_data.py --arm-ip 192.168.5.123 --save-dir data/raw_hdf5 --fps 30

  # 示教模式
  python scripts/collect_data.py --arm-ip 192.168.5.123 --save-dir data/raw_hdf5 --fps 30 --teaching
"""

import select
import tty
import termios
import time
import math
import threading
import sys
import os
import h5py
import numpy as np
import cv2
import argparse

# ============ 配置区域（根据您的硬件修改） ============
DEFAULT_ARM_IP = "192.168.5.123"
DEFAULT_ARM_PORT = 8080
DEFAULT_CAM_TOP_SERIAL = "262322074840"      # 顶部 D435 (rs-enumerate-devices | grep Serial)
DEFAULT_CAM_WRIST_SERIAL = "CV2T66100096"    # 腕部 Orbbec 305 (留空则取第一个设备)
DEFAULT_TRACKER_SERIAL = "<YOUR_TRACKER_SN>"   # SteamVR 中 Tracker 序列号

# 知行 RTU 夹爪参数（独立串口，与机械臂解耦）
GRIPPER_PORT = "/dev/realman/gripper_left"
GRIPPER_SLAVE_ID = 2          # 知行夹爪 Modbus 从站地址
GRIPPER_BAUDRATE = 115200
GRIPPER_MAX_POSITION = 9000   # 归一化行程上限兜底值 (设备单位, /100=mm); 若存在 hardware/gripper_calibration.json 则以标定值为准

# 机械臂初始位姿（Vive遥操作零点对应的机械臂笛卡尔位姿）
# 含义: 按 v 校准零点后, tracker 在零点时机械臂应处的位姿; 按 w 启用后机械臂以此为基础跟随。
# 标定: 手动把机械臂移到一个安全顺手的起始位, 读它的笛卡尔位姿填进来。
#   ⚠️ 下面两个数组是上一套实物调的值, 安装不同必改 —— 否则按 w 时机械臂会突跳到旧位姿。
#   POS 单位米 [x,y,z]; ORI 单位弧度 [rx,ry,rz]。
ROBOT_INIT_POS = np.array([-0.0847, -0.2821, 0.0872])
ROBOT_INIT_ORI = np.array([-3.102, 0.065, 1.609])

# Vive Tracker → 机械臂末端 固定外参 (现场标定): XYZRPY = {0, 0, 80mm, 90°, 0°, -90°}
#   · 平移 80mm 是 tracker 原点沿其轴到末端的固定偏置, 属常量, 已包含在 ROBOT_INIT 标定里;
#     相对运动映射不需要它, 这里只用旋转部分。
#   · 旋转 RPY(roll=90°, pitch=0°, yaw=-90°) 组成换基矩阵 M(R=Rz·Ry·Rx), 既管位置也管姿态:
#       Robot_X = -Vive_Z, Robot_Y = -Vive_X, Robot_Z = +Vive_Y。
#   · 现场若方向不对, 改这三个角度即可, 不用再逐轴翻符号/换下标。
VIVE_TO_ROBOT_RPY_DEG = np.array([90.0, 0.0, -90.0])   # [roll(X), pitch(Y), yaw(Z)] 单位度
# ============ 配置区域结束 ============


# ------ 路径设置 ------
# 如需从源码使用 RM_API2，请取消注释并填写实际路径:
# sys.path.append("/path/to/RM_API2/Python")
_HARDWARE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hardware')
if _HARDWARE_DIR not in sys.path:
    sys.path.insert(0, _HARDWARE_DIR)

from orbbec_camera import OrbbecCamera              # 腕部相机 (奥比中光)
from realsense_camera import RealSenseCamera        # 顶部相机 (Intel RealSense D435)
from changingtek_gripper import ChangingtekGripper   # 知行夹爪


# ============ 辅助函数 ============
def get_next_filename(save_dir, task_name):
    """获取下一个可用的文件名 (自增编号)"""
    os.makedirs(save_dir, exist_ok=True)
    idx = 0
    while os.path.exists(os.path.join(save_dir, f"{task_name}_{idx}.hdf5")):
        idx += 1
    return os.path.join(save_dir, f"{task_name}_{idx}.hdf5")


# ============ 姿态/坐标换算辅助 (纯 numpy, 无额外依赖) ============
def _quat_to_matrix(q):
    """四元数 [w,x,y,z] → 3×3 旋转矩阵"""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    s = 0.0 if n == 0.0 else 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1.0 - (yy + zz), xy - wz,         xz + wy        ],
        [xy + wz,         1.0 - (xx + zz), yz - wx        ],
        [xz - wy,         yz + wx,         1.0 - (xx + yy)],
    ])


def _matrix_to_quat(R):
    """3×3 旋转矩阵 → 四元数 [w,x,y,z] (标准健壮算法)"""
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return np.array([w, x, y, z])


def _euler_xyz_to_matrix(e):
    """固定轴 XYZ(RPY) 欧拉角 [rx,ry,rz](弧度) → 3×3, R = Rz·Ry·Rx (与机械臂/Vive 约定一致)"""
    rx, ry, rz = e
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy,     cy * sx,                cy * cx               ],
    ])


def _matrix_to_euler_xyz(R):
    """3×3 → 固定轴 XYZ(RPY) 欧拉角 [rx,ry,rz](弧度), 为 _euler_xyz_to_matrix 的逆"""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:  # 万向锁 (ry≈±90°)
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0.0
    return np.array([rx, ry, rz])


def _scale_rotation(R, s):
    """把旋转 R 的转角按 s 倍缩放(绕同一转轴), 用于姿态灵敏度; s=1 原样返回。"""
    if abs(s - 1.0) < 1e-9:
        return R
    q = _matrix_to_quat(R)
    if q[0] < 0:
        q = -q                                  # 保证 w≥0, 转角落在 [-π, π]
    vnorm = math.sqrt(q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    if vnorm < 1e-9:
        return np.eye(3)                        # 近乎无旋转
    half = math.atan2(vnorm, q[0]) * s          # 半角 × s
    axis = np.array([q[1], q[2], q[3]]) / vnorm
    q2 = np.concatenate(([math.cos(half)], axis * math.sin(half)))
    return _quat_to_matrix(q2)


# ============ Vive 遥控模块 ============
class ViveController:
    """Vive Tracker 遥操作控制器

    原理：读取 Tracker 的位姿增量，用旋转矩阵映射到机械臂笛卡尔空间。
    坐标系换基由固定外参 M = VIVE_TO_ROBOT_RPY_DEG 统一描述(位置/姿态共用):
      Robot_X = -Vive_Z
      Robot_Y = -Vive_X
      Robot_Z = +Vive_Y
    姿态采用相对旋转 R_delta = R_cur · R_initᵀ 换基后叠加到起始姿态,
    避免欧拉角逐轴相减在大角度/万向锁处失效。
    """

    def __init__(self, arm, arm_lock, tracker_serial=None, enable_vive=True):
        self.arm = arm
        self.arm_lock = arm_lock
        self.tracker_serial = tracker_serial or DEFAULT_TRACKER_SERIAL
        self.enable_vive = enable_vive

        self.robot_init_pos = ROBOT_INIT_POS.copy()
        self.robot_init_ori = ROBOT_INIT_ORI.copy()
        # 起始姿态的旋转矩阵形式 (固定轴 XYZ/RPY: R = Rz·Ry·Rx)
        self.robot_init_R = _euler_xyz_to_matrix(self.robot_init_ori)
        # Vive→Robot 固定外参旋转矩阵 M (坐标系换基用)
        self.vive_to_robot_rot = _euler_xyz_to_matrix(np.radians(VIVE_TO_ROBOT_RPY_DEG))

        self.vive_init_pos = None
        self.vive_init_R = None       # 校准零点时 tracker 的旋转矩阵
        self.control_enabled = False
        self.running = True
        self.vive = None
        self.tracker = None

        if self.enable_vive:
            self._init_vive()
            self.thread = threading.Thread(target=self._control_loop, daemon=True)
            self.thread.start()
        else:
            print("示教模式: 不使用 Vive 遥控，手动移动机械臂")

    def _init_vive(self):
        """阻塞式初始化 Vive：未连接时持续重试，直到发现 Tracker 才返回。

        Vive/SteamVR 未就绪时 openvr.init() 会持续抛异常, 或 SteamVR 已就绪但枚举
        不到 tracker；此处每 2s 重试一次, 直到拿到 tracker 为止。期间 Ctrl+C 可中断。
        """
        # 需要 hardware/vive_tracker.py; 依赖缺失(ImportError)属确定性错误, 直接抛出, 不重试
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hardware'))
        import vive_tracker as triad_lib

        attempt = 0
        last_msg = None
        while self.running:
            attempt += 1
            try:
                self.vive = triad_lib.triad_openvr()   # openvr.init 连接 SteamVR
                self.tracker = self._pick_tracker()
                if self.tracker is not None:
                    print(f"Vive: Tracker 已连接 (SN {self.tracker.get_serial()})")
                    return
                msg = "SteamVR 已就绪但未发现 Tracker，请检查 Tracker 电源/基站"
                # SteamVR 就绪但没枚举到 tracker: 必须先释放本次 openvr 会话,
                # 否则下一轮重试再 openvr.init() 会泄漏会话, 退出后 SteamVR 侧连接不释放(需重插拔)
                self.vive.shutdown()
                self.vive = None
            except KeyboardInterrupt:
                # 等待期间 Ctrl+C: 构造函数未完成, main 里 vive_ctrl 未赋值, 无法调 shutdown,
                # 因此必须在此释放已建立的 openvr 会话, 否则残留连接导致下次要重新插拔 Vive
                if self.vive is not None:
                    self.vive.shutdown()
                    self.vive = None
                raise
            except Exception as e:  # noqa: BLE001 - SteamVR 未就绪会持续报错, 需重试
                # 注意: 不捕获 KeyboardInterrupt, Ctrl+C 会自然传播以中断等待
                self.vive = None
                msg = f"{type(e).__name__}: {e}"

            if msg != last_msg:
                print(f"[!] Vive 未连接（{msg}）")
                print("    等待中... 启动 SteamVR 并连接 Tracker 后自动继续；Ctrl+C 退出")
                last_msg = msg
            elif attempt % 10 == 0:
                print(f"    仍在等待 Vive 连接...（{msg}）")
            time.sleep(2.0)

    def _pick_tracker(self):
        """从已枚举设备里挑目标 tracker：优先匹配序列号，否则取第一个 tracker。"""
        first = None
        for name, device in self.vive.devices.items():
            if "tracker" not in name.lower():
                continue
            try:
                if device.get_serial() == self.tracker_serial:
                    return device
            except Exception:  # noqa: BLE001 - 个别设备读序列号可能失败，跳过继续
                pass
            if first is None:
                first = device
        return first

    def calibrate(self):
        """校准：记录当前 Vive Tracker 位姿作为零点"""
        if self.tracker is None:
            print("[!] Vive 未连接")
            return False

        print("校准中... 保持 Tracker 静止")
        positions = []
        quats = []          # [w,x,y,z]
        for _ in range(30):
            pose = self.tracker.get_pose_quaternion()   # [x,y,z, w,qx,qy,qz]
            if pose:
                positions.append([pose[0], pose[1], pose[2]])
                q = np.array([pose[3], pose[4], pose[5], pose[6]])
                if quats and np.dot(q, quats[0]) < 0:
                    q = -q  # 统一到同一半球, 避免四元数符号翻转导致平均抵消
                quats.append(q)
            time.sleep(0.033)

        if not quats:
            print("[!] 校准失败")
            return False

        self.vive_init_pos = np.mean(positions, axis=0)
        q_mean = np.mean(quats, axis=0)
        self.vive_init_R = _quat_to_matrix(q_mean / np.linalg.norm(q_mean))
        print("校准完成")
        return True

    def enable(self):
        if self.vive_init_pos is None:
            print("请先校准(v)")
            return
        self.control_enabled = True
        print("遥控已启用")

    def disable(self):
        self.control_enabled = False
        print("遥控已暂停")

    def _control_loop(self):
        """20Hz 遥控循环"""
        interval = 0.05
        last_ret = 0      # 上一次 movep 返回码 (节流打印用)
        last_err = None   # 上一次异常信息 (节流打印用)

        while self.running:
            time.sleep(interval)
            if (not self.control_enabled or self.tracker is None
                    or self.vive_init_pos is None or self.vive_init_R is None):
                continue

            try:
                pose = self.tracker.get_pose_quaternion()   # [x,y,z, w,qx,qy,qz]
                if pose is None:
                    continue

                cur_pos = np.array([pose[0], pose[1], pose[2]])
                R_cur = _quat_to_matrix(np.array([pose[3], pose[4], pose[5], pose[6]]))

                scale_pos = 1   # 位置灵敏度: 调大=机械臂动得比手多, 调小=更细腻
                scale_ori = 1.4   # 姿态灵敏度: 作用于相对旋转的转角 (原 0.3 太小, 转 30° 末端只转 9°)

                # ================= 坐标映射: Vive → Robot（旋转矩阵法, 现场标定改外参）=================
                # 固定外参 VIVE_TO_ROBOT_RPY_DEG={90,0,-90} 给出换基矩阵 M, 位置/姿态共用:
                #   Robot_X=-Vive_Z, Robot_Y=-Vive_X, Robot_Z=+Vive_Y。
                # 现场方向不对时, 改外参 RPY 即可(不用再逐轴翻符号/换下标)。
                M = self.vive_to_robot_rot

                # 位置: Vive世界系位移增量 → 换基到机械臂基座系 → 叠加到起始位
                delta_pos_vive = cur_pos - self.vive_init_pos
                target_pos = self.robot_init_pos + (M @ delta_pos_vive) * scale_pos

                # 姿态: 相对旋转 R_delta=R_cur·R_initᵀ(世界系) → 换基 M·R·Mᵀ → 左乘起始姿态
                R_delta = R_cur @ self.vive_init_R.T
                R_delta = M @ R_delta @ M.T
                R_delta = _scale_rotation(R_delta, scale_ori)   # 按 scale_ori 缩放转角
                target_ori = _matrix_to_euler_xyz(R_delta @ self.robot_init_R)

                # 安全限位（笛卡尔工作空间, 单位米）: 防止手滑把机械臂推出安全区。
                # 换成你的实际可达范围; 若机械臂总在某方向到不了边界, 放宽对应上下限。
                target_pos[0] = np.clip(target_pos[0], -0.5, 0.5)
                target_pos[1] = np.clip(target_pos[1], -0.5, 0.5)
                target_pos[2] = np.clip(target_pos[2], -0.15, 0.3)

                target_6d = [
                    float(target_pos[0]), float(target_pos[1]), float(target_pos[2]),
                    float(target_ori[0]), float(target_ori[1]), float(target_ori[2])
                ]

                with self.arm_lock:
                    ret = self.arm.rm_movep_canfd(target_6d, False, 0, 60)

                # 诊断: 返回码非 0 = 逆解失败/姿态不可达, 机械臂会停在上一位姿 (节流打印)
                if ret != 0 and ret != last_ret:
                    print(f"[!] movep_canfd ret={ret} 姿态可能不可达 target_ori={target_ori}")
                last_ret = ret

            except Exception as e:  # noqa: BLE001 - 遥控循环不能因单次异常中断
                if str(e) != last_err:
                    print(f"[!] _control_loop 异常: {e}")
                    last_err = str(e)

    def shutdown(self):
        self.running = False
        # 等控制线程退出后再释放 openvr, 避免线程仍调用已 shutdown 的句柄
        if getattr(self, "thread", None) is not None:
            self.thread.join(timeout=1.0)
        if self.vive is not None:
            self.vive.shutdown()   # triad_openvr.shutdown() → openvr.shutdown()
            self.vive = None


# ============ 数据录制模块 ============
class DataRecorder:
    """以固定频率录制机械臂状态、相机图像到 HDF5

    数据格式:
      observations/qpos:             (N, 7) float32  [6关节角 + 夹爪位置]
      observations/images/camera_global:  (N, H, W, 3) uint8
      observations/images/camera_left: (N, H, W, 3) uint8
      action:                        (N, 7) float32  [下一帧的qpos，即行为克隆标签]
      timestamps:                    (N,) float64    [相对时间戳(秒)]
    """

    def __init__(self, arm, arm_lock, gripper, cam_top, cam_wrist, target_fps=30):
        self.arm = arm
        self.arm_lock = arm_lock
        self.gripper = gripper
        self.cam_top = cam_top
        self.cam_wrist = cam_wrist
        self.is_recording = False
        self.filename = None
        self.target_fps = target_fps
        self.data_buffer = {'qpos': [], 'images_top': [], 'images_wrist': [], 'timestamps': []}

    def start(self, filename):
        if self.is_recording:
            return
        self.filename = filename
        self.is_recording = True
        self.data_buffer = {'qpos': [], 'images_top': [], 'images_wrist': [], 'timestamps': []}
        self.record_start_time = time.time()
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()
        print(f"录制开始: {os.path.basename(filename)} (目标 {self.target_fps}Hz)")

    def stop(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self.thread.join()
        self._save()

    def _record_loop(self):
        interval = 1.0 / self.target_fps
        next_time = time.time()

        while self.is_recording:
            now = time.time()
            if now < next_time:
                time.sleep(0.001)
                continue
            next_time += interval

            try:
                timestamp = now - self.record_start_time

                with self.arm_lock:
                    code, state = self.arm.rm_get_current_arm_state()

                joint_angles = state['joint'] if code == 0 else [0] * 6
                # 夹爪走独立串口，后台轮询线程已缓存反馈，无需 arm_lock
                gripper_val = self.gripper.get_position_normalized()

                img_top = self.cam_top.get_frame()
                img_wrist = self.cam_wrist.get_frame()

                self.data_buffer['qpos'].append(joint_angles + [gripper_val])
                self.data_buffer['images_top'].append(img_top)
                self.data_buffer['images_wrist'].append(img_wrist)
                self.data_buffer['timestamps'].append(timestamp)
            except Exception as e:
                print(f" >> Record error: {e}")

    def _save(self):
        if not self.data_buffer['qpos']:
            print(" >> 无数据")
            return

        qpos = self.data_buffer['qpos']
        # 行为克隆标签: action[t] = qpos[t+1]
        actions = qpos[1:] + [qpos[-1]]
        timestamps = self.data_buffer['timestamps']

        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            actual_fps = 1.0 / np.mean(intervals)
            print(f"  实际帧率: {actual_fps:.1f} Hz (目标: {self.target_fps} Hz)")

        try:
            with h5py.File(self.filename, 'w') as f:
                f.attrs['sim'] = False
                f.attrs['fps'] = self.target_fps
                f.create_dataset('observations/qpos', data=np.array(qpos))
                f.create_dataset('action', data=np.array(actions))
                f.create_dataset('timestamps', data=np.array(timestamps))
                f.create_dataset('observations/images/camera_global',
                                 data=np.array(self.data_buffer['images_top']),
                                 compression="gzip")
                f.create_dataset('observations/images/camera_left',
                                 data=np.array(self.data_buffer['images_wrist']),
                                 compression="gzip")
            print(f"保存: {os.path.basename(self.filename)} ({len(qpos)} frames)")
        except Exception as e:
            print(f"[!] 保存失败: {e}")


# ============ 主程序 ============
def main():
    parser = argparse.ArgumentParser(description='RealMan RM65 数据采集')
    parser.add_argument('--arm-ip', type=str, default=DEFAULT_ARM_IP, help='机械臂IP地址')
    parser.add_argument('--arm-port', type=int, default=DEFAULT_ARM_PORT, help='机械臂端口')
    parser.add_argument('--cam-top', type=str, default=DEFAULT_CAM_TOP_SERIAL, help='顶部相机(D435)序列号')
    parser.add_argument('--cam-wrist', type=str, default=DEFAULT_CAM_WRIST_SERIAL, help='腕部相机(Orbbec 305)序列号，留空取第一个设备')
    parser.add_argument('--gripper-port', type=str, default=GRIPPER_PORT, help='知行夹爪串口')
    parser.add_argument('--gripper-slave-id', type=int, default=GRIPPER_SLAVE_ID, help='知行夹爪 Modbus 从站地址')
    parser.add_argument('--save-dir', type=str, default='data/raw_hdf5', help='数据保存目录')
    parser.add_argument('--task-name', type=str, default='task_pick_cube', help='任务名称')
    parser.add_argument('--fps', type=int, default=30, help='采集帧率')
    parser.add_argument('--tracker-serial', type=str, default=DEFAULT_TRACKER_SERIAL,
                        help='Vive Tracker 序列号（留空或保留占位符则自动选第一个 tracker）')
    parser.add_argument('--teaching', action='store_true', help='示教模式（不用Vive）')
    args = parser.parse_args()

    # 导入机械臂SDK
    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

    print("=" * 50)
    print(f"  RealMan RM65 {'示教' if args.teaching else 'Vive遥操作'} 数据采集")
    print("=" * 50)

    # 初始化双相机（顶部 D435 + 腕部 Orbbec 305）
    cam_top = RealSenseCamera(args.cam_top)
    cam_wrist = OrbbecCamera(args.cam_wrist)
    print(f"相机: top={'OK' if cam_top.is_active else 'FAIL'}, "
          f"wrist={'OK' if cam_wrist.is_active else 'FAIL'}")

    # 初始化机械臂
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(args.arm_ip, args.arm_port)
    if handle.id == -1:
        print("[!] 机械臂连接失败")
        sys.exit(1)
    print(f"机械臂: OK ({args.arm_ip}:{args.arm_port})")

    arm.rm_set_arm_run_mode(1)

    # 初始化知行夹爪（独立串口，与机械臂解耦）
    gripper = ChangingtekGripper(
        port=args.gripper_port, slave_id=args.gripper_slave_id,
        baudrate=GRIPPER_BAUDRATE, max_position=GRIPPER_MAX_POSITION,
    )
    gripper.connect()
    print(f"夹爪: {'OK' if gripper.connected else 'FAIL'} ({args.gripper_port})")
    arm_lock = threading.Lock()

    # 初始化 Vive 遥控（未连接时会阻塞等待直到 Tracker 就绪，Ctrl+C 可中断）
    try:
        vive_ctrl = ViveController(arm, arm_lock, tracker_serial=args.tracker_serial,
                                   enable_vive=not args.teaching)
    except KeyboardInterrupt:
        print("\n[!] 已取消：等待 Vive 连接被中断")
        cam_top.close()
        cam_wrist.close()
        gripper.disable()
        gripper.disconnect()
        arm.rm_delete_robot_arm()
        sys.exit(1)

    # 初始化录制器
    recorder = DataRecorder(arm, arm_lock, gripper, cam_top, cam_wrist, args.fps)

    print("\n" + "-" * 50)
    if args.teaching:
        print("命令: s=录制  d=保存  g <0-100>=夹爪  c=闭合  o=打开  q=退出")
    else:
        print("命令: v=校准  w=遥控  e=停止  s=录制  d=保存")
        print("      g <0-100>=夹爪  c=闭合  o=打开  q=退出")
    print("-" * 50)

    old_settings = termios.tcgetattr(sys.stdin)
    cmd_buffer = ""

    try:
        tty.setcbreak(sys.stdin.fileno())
        print("> ", end='', flush=True)

        while True:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.01)
            if rlist:
                char = sys.stdin.read(1)
                if char == '\n':
                    cmd = cmd_buffer.strip()
                    if cmd == 'q':
                        print()
                        break
                    elif cmd == 'v' and not args.teaching:
                        print(); vive_ctrl.calibrate()
                    elif cmd == 'w' and not args.teaching:
                        print(); vive_ctrl.enable()
                    elif cmd == 'e' and not args.teaching:
                        print(); vive_ctrl.disable()
                    elif cmd == 's':
                        print()
                        filename = get_next_filename(args.save_dir, args.task_name)
                        recorder.start(filename)
                    elif cmd == 'd':
                        print(); recorder.stop()
                    elif cmd.startswith('g '):
                        print()
                        try:
                            val = max(0, min(100, int(cmd.split()[1])))
                            gripper.move_pct(val)
                            print(f"夹爪: {val}%")
                        except Exception:
                            print("格式: g <0-100>")
                    elif cmd == 'c':
                        print()
                        gripper.close()
                        print("夹爪: 闭合")
                    elif cmd == 'o':
                        print()
                        gripper.open()
                        print("夹爪: 打开")
                    elif cmd:
                        print("\n未知命令")
                    cmd_buffer = ""
                    print("> ", end='', flush=True)
                elif char == '\x7f':
                    if cmd_buffer:
                        cmd_buffer = cmd_buffer[:-1]
                        sys.stdout.write('\b \b')
                        sys.stdout.flush()
                else:
                    cmd_buffer += char
                    sys.stdout.write(char)
                    sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        if recorder.is_recording:
            recorder.stop()
        vive_ctrl.shutdown()
        cam_top.close()
        cam_wrist.close()
        gripper.disable()
        gripper.disconnect()
        arm.rm_delete_robot_arm()
        print("已退出")


if __name__ == '__main__':
    main()
