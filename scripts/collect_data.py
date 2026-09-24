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
  - observations/ee_pose: (N, 6)  [末端笛卡尔位姿 x,y,z,rx,ry,rz (米/弧度), 机械臂直接读取非解算]
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
import json
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

# 机械臂归位 (rm_movej_p: 关节空间规划到 ROBOT_INIT 笛卡尔位姿, 大位移最稳、不撞奇异点)
ARM_HOME_SPEED_SLOW = 5      # 启动慢速归位速度百分比 v(1~100): 上电位姿未知, 求稳(用户指定 5%)
ARM_HOME_SPEED_NORMAL = 45   # 复位键常速归位速度百分比
ARM_HOME_COUNTDOWN = 3       # 启动归位前倒计时秒数, 留时间清空机械臂周围
ARM_HOME_POS_TOL = 0.02      # 慢速归位"已在起始位附近"的位置容差(米): 与 ROBOT_INIT 偏差<2cm
ARM_HOME_ORI_TOL = float(np.radians(5.0))   # 同上姿态容差(弧度): 偏差<5° 则跳过慢速归位
# ⚠️ rm_set_arm_run_mode 是"仿真(0)/真实(1)"开关, 不是运动模式! 全程保持真实(1):
#    规划运动(rm_movej_p)与 CANFD 透传(rm_movep_canfd)都在真实模式下执行, 归位无需切模式
#    (切到 0=仿真会让指令只在仿真里跑、真实机械臂不动 —— 这正是"归位没反应"的坑)。
VIVE_CALIB_FRAMES = 5        # 启用遥操时自动校准零点的平均帧数 (原 30 帧太长, 现取当前手持位)
# 遥操透传频率(Hz): rm_movep_canfd 是透传指令, 为高频稳定流设计。频率越高跟随越紧、延迟越低。
# 睿尔曼官方低跟随示例 RMDemo_MovejCANFD 用 100Hz; 原实现写死 20Hz(interval=0.05)是遥操延迟主因。
# 50~100 之间按机器负载调; 若配合录制抢锁明显, 可先取 50。
VIVE_CONTROL_HZ = 100        # 默认配合高跟随(需≥100Hz); 用 --no-high-follow 低跟随且嫌抢锁时可显式 --control-hz 50
# 高跟随模式(follow=True): 滞后最小, 但透传周期要求 ≤10ms(即 ≥100Hz)。开启后若 control_hz<100 会自动抬到 100。
#   默认开(本项目实机调定); --no-high-follow 可关(改回低跟随, 控制器自规划更软、无 ≤10ms 约束)。
VIVE_FOLLOW_HIGH = True
# SDK 控制器侧轨迹平滑(仅高跟随生效): 治理"伺服刚性嗡颤"——完全透传下伺服刚性追每个设定点激起的机械高频振,
# 发生在我们指令的下游, 软件滤波(One Euro)够不着, 只能靠控制器在伺服前重新拟合/滤波轨迹来抑制。
#   trajectory_mode: 0=完全透传(无 SDK 平滑), 1=曲线拟合, 2=滤波(默认)。
#   radio: 拟合(0~100)/滤波(0~1000)平滑系数, 越大越平滑、滞后越明显; mode=0 时无效。
# 与手抖分工: 手部生理抖/tracker 噪声(输入端)→ 下面 One Euro; 伺服嗡颤(控制器端)→ 本项。
VIVE_TRAJ_MODE = 2           # 默认滤波模式(实机调定: 高跟随下压伺服嗡颤)
VIVE_TRAJ_RADIO = 600        # 默认滤波系数(越大越平滑、滞后越大; 实机 600 手感可接受)
# One Euro 自适应滤波 (替代固定截止低通): 慢速/静止时用低截止→强平滑压手抖, 快速运动时自动升截止→几乎无滞后。
# 化解"平滑 vs 延迟"两难: 可保留高跟随@100Hz 的跟手, 同时压掉高频细碎颤 (固定低通做不到)。
#   MINCUTOFF(Hz): 零速截止频率。越低越压静止手抖(慢速运动滞后略增); 静止仍颤→降到 0.5/0.1。
#   BETA: 速度系数(截止随速度上升的斜率)。经验 10~30 越跟手; 默认取 1.0(刻意压快速段跟随、增阻尼防"太快"), 嫌拖可调大。
#   DCUTOFF(Hz): 对"速度"再做低通的截止, 一般 1.0 即可。
VIVE_OE_MINCUTOFF = 0.1      # 默认强平滑压静止手抖(实机调定)
VIVE_OE_BETA = 1.0           # 默认低速度系数: 快速段也重平滑, 抑制"跟随太快"(代价: 滞后略增)
VIVE_OE_DCUTOFF = 1.0
# ============ 配置区域结束 ============


# ------ 路径设置 ------
# 如需从源码使用 RM_API2，请取消注释并填写实际路径:
# sys.path.append("/path/to/RM_API2/Python")
_HARDWARE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hardware')
if _HARDWARE_DIR not in sys.path:
    sys.path.insert(0, _HARDWARE_DIR)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_KEYBINDINGS_PATH = os.path.join(_REPO_ROOT, "configs", "keybindings.json")

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


def _pose_deviation(cur_pose, target_pose):
    """两个 6D 位姿 [x,y,z,rx,ry,rz](米/弧度) 的偏差 → (位置偏差 米, 姿态偏差 弧度)。

    姿态用旋转矩阵测地角 arccos((tr(R_curᵀ·R_tgt)-1)/2), 避免欧拉角 wrap / ±π 歧义
    (直接比 rx/ry/rz 会把 -3.1 与 +3.1 当成差 6.2, 实则同一朝向)。
    """
    dp = float(np.linalg.norm(np.asarray(cur_pose[:3]) - np.asarray(target_pose[:3])))
    R_rel = _euler_xyz_to_matrix(np.asarray(cur_pose[3:6])).T @ _euler_xyz_to_matrix(np.asarray(target_pose[3:6]))
    cos_t = float(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
    return dp, float(np.arccos(cos_t))


class OneEuroFilter:
    """One Euro 自适应低通滤波器 (Casiez et al. 2012), 支持多维 numpy 向量。

    固定截止的低通在"平滑"与"延迟"之间此消彼长; One Euro 用"信号速度"自适应调截止:
    慢速(手抖主导)→低截止→强平滑; 快速(有意运动)→高截止→几乎无滞后。适合遥操/VR 手部去噪。

    参数:
      freq      采样频率(Hz), 取控制循环频率。
      mincutoff 零速截止频率(Hz): 越低越压静止手抖, 慢速运动滞后略增。
      beta      速度系数: 截止随速度上升的斜率。越大快速越跟手, 但快速时可能漏过抖动。
      dcutoff   对速度做低通的截止频率(Hz), 一般 1.0。
    """

    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq = max(1e-6, float(freq))
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self._x_prev = None
        self._dx_prev = None

    @property
    def prev(self):
        """上一次滤波输出 (供姿态四元数统一半球判断); 未初始化时为 None。"""
        return self._x_prev

    def _alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2.0 * math.pi * max(1e-6, cutoff))
        return 1.0 / (1.0 + tau / te)

    def reset(self):
        """清空历史; 重新 engage 时调用, 让滤波从新起点起步、不用旧值突跳。"""
        self._x_prev = None
        self._dx_prev = None

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        if self._x_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            return x.copy()
        dx = (x - self._x_prev) * self.freq            # 瞬时速度 (单位/秒)
        ad = self._alpha(self.dcutoff)
        dx_hat = ad * dx + (1.0 - ad) * self._dx_prev  # 低通后的速度估计
        speed = float(np.linalg.norm(dx_hat))          # 多维共用一个标量速度
        cutoff = self.mincutoff + self.beta * speed    # 速度越快截止越高
        a = self._alpha(cutoff)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


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

    def __init__(self, arm, arm_lock, tracker_serial=None, enable_vive=True,
                 control_hz=VIVE_CONTROL_HZ, follow_high=VIVE_FOLLOW_HIGH,
                 traj_mode=VIVE_TRAJ_MODE, traj_radio=VIVE_TRAJ_RADIO,
                 oe_mincutoff=VIVE_OE_MINCUTOFF, oe_beta=VIVE_OE_BETA,
                 oe_dcutoff=VIVE_OE_DCUTOFF):
        self.arm = arm
        self.arm_lock = arm_lock
        self.tracker_serial = tracker_serial or DEFAULT_TRACKER_SERIAL
        self.enable_vive = enable_vive
        self.follow_high = bool(follow_high)
        self.traj_mode = int(traj_mode)      # 控制器侧轨迹平滑(治伺服嗡颤), 仅高跟随生效
        self.traj_radio = int(traj_radio)
        # 高跟随要求透传周期 ≤10ms(≥100Hz); 频率不足则自动抬到 100, 否则控制器可能拒收/抖动
        self.control_hz = max(1.0, float(control_hz))
        if self.follow_high and self.control_hz < 100.0:
            print(f"[i] 高跟随模式要求 ≥100Hz, control_hz 自动 {self.control_hz:.0f}→100")
            self.control_hz = 100.0
        # One Euro 自适应滤波: 位置/姿态各一个, 采样频率=控制频率。慢速去颤、快速跟手。
        self._pos_filt = OneEuroFilter(self.control_hz, oe_mincutoff, oe_beta, oe_dcutoff)
        self._quat_filt = OneEuroFilter(self.control_hz, oe_mincutoff, oe_beta, oe_dcutoff)

        # 遥操基点(位置/姿态): 初值用全局 ROBOT_INIT 兜底, 每次 enable() 会改绑到机械臂"当前"位姿
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

    def calibrate(self, frames=None):
        """以当前 Vive Tracker 位姿为零点。frames=平均帧数(默认 VIVE_CALIB_FRAMES)。

        启用遥操(enable)时自动调用: 每次都以"当前手持位置"为零点, 无需单独手动校准、
        也不必记住某个全局固定零点。少量帧平均仅用于抑制单帧抖动/丢帧。
        """
        if self.tracker is None:
            print("[!] Vive 未连接")
            return False

        frames = VIVE_CALIB_FRAMES if frames is None else max(1, int(frames))
        positions = []
        quats = []          # [w,x,y,z]
        for _ in range(frames):
            pose = self.tracker.get_pose_quaternion()   # [x,y,z, w,qx,qy,qz]
            if pose:
                positions.append([pose[0], pose[1], pose[2]])
                q = np.array([pose[3], pose[4], pose[5], pose[6]])
                if quats and np.dot(q, quats[0]) < 0:
                    q = -q  # 统一到同一半球, 避免四元数符号翻转导致平均抵消
                quats.append(q)
            time.sleep(0.02)

        if not quats:
            print("[!] 校准失败: 读不到 Tracker 位姿")
            return False

        self.vive_init_pos = np.mean(positions, axis=0)
        q_mean = np.mean(quats, axis=0)
        self.vive_init_R = _quat_to_matrix(q_mean / np.linalg.norm(q_mean))
        return True

    def _read_arm_pose(self):
        """读机械臂当前笛卡尔位姿 [x,y,z, rx,ry,rz] (米/弧度); 失败返回 None。"""
        try:
            with self.arm_lock:
                code, state = self.arm.rm_get_current_arm_state()
            if code == 0 and state and state.get("pose"):
                return [float(x) for x in state["pose"][:6]]
        except Exception as e:  # noqa: BLE001 - 读状态失败不应中断
            print(f"[!] 读机械臂当前位姿失败: {e}")
        return None

    def enable(self):
        """启用遥操: 同时捕获「Tracker 当前位姿=零点」与「机械臂当前位姿=基点」。

        基点绑定机械臂"当前"位置(而非全局 ROBOT_INIT), 故按下 w 时机械臂原地 engage、
        不会突跳到 ROBOT_INIT; 手的相对运动从当前位置开始映射。
        """
        if self.tracker is None:
            print("[!] Vive 未连接, 无法启用遥操")
            return
        if not self.calibrate():       # Tracker 零点 = 当前手持位置
            return
        arm_pose = self._read_arm_pose()   # 机械臂基点 = 当前位姿 (不绑定 ROBOT_INIT)
        if arm_pose is None:
            print("[!] 读不到机械臂当前位姿, 无法启用遥操")
            return
        self.robot_init_pos = np.array(arm_pose[:3])
        self.robot_init_ori = np.array(arm_pose[3:6])
        self.robot_init_R = _euler_xyz_to_matrix(self.robot_init_ori)
        self._pos_filt.reset()     # 重置滤波, 从本次 engage 位姿起步, 避免用旧滤波值突跳
        self._quat_filt.reset()
        self.control_enabled = True
        print("遥控已启用 (Tracker 零点=当前手持位, 机械臂基点=当前位姿)")

    def disable(self):
        self.control_enabled = False
        print("遥控已暂停")

    def _control_loop(self):
        """高频遥控循环 (deadline 节拍, 频率 self.control_hz, 默认 VIVE_CONTROL_HZ)。

        用绝对 deadline 计时而非循环开头固定 sleep, 避免周期漂移累积成"阻塞感";
        单步超时(读位姿/透传偶发变慢)则重置 deadline 丢弃积压, 防止随后指令突发。
        """
        interval = 1.0 / self.control_hz
        last_ret = 0      # 上一次 movep 返回码 (节流打印用)
        last_err = None   # 上一次异常信息 (节流打印用)
        next_time = time.time()

        while self.running:
            now = time.time()
            if now < next_time:
                time.sleep(min(0.002, next_time - now))
                continue
            # 落后超过一个周期: 重置节拍, 不补发积压指令 (避免机械臂追旧目标点抖动)
            if now - next_time > interval:
                next_time = now
            next_time += interval

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
                R_target = R_delta @ self.robot_init_R
                target_quat = _matrix_to_quat(R_target)          # [w,x,y,z]

                # ---- One Euro 自适应滤波: 慢速去颤(强平滑)、快速跟手(几乎无滞后) ----
                # 化解"平滑 vs 延迟"两难: 静止/慢速压掉高频细碎颤, 有意快速运动不加滞后。
                # 姿态四元数滤波前先与上次输出统一半球, 避免符号翻转导致插值抵消。
                q_prev = self._quat_filt.prev
                if q_prev is not None and float(np.dot(q_prev, target_quat)) < 0:
                    target_quat = -target_quat
                filt_pos = self._pos_filt(target_pos)
                filt_quat = self._quat_filt(target_quat)
                filt_quat = filt_quat / np.linalg.norm(filt_quat)

                # 安全限位（笛卡尔工作空间, 单位米）: 防止手滑把机械臂推出安全区。
                # 换成你的实际可达范围; 若机械臂总在某方向到不了边界, 放宽对应上下限。
                filt_pos[0] = np.clip(filt_pos[0], -0.5, 0.5)
                filt_pos[1] = np.clip(filt_pos[1], -0.5, 0.5)
                filt_pos[2] = np.clip(filt_pos[2], -0.15, 0.3)

                target_ori = _matrix_to_euler_xyz(_quat_to_matrix(filt_quat))
                target_6d = [
                    float(filt_pos[0]), float(filt_pos[1]), float(filt_pos[2]),
                    float(target_ori[0]), float(target_ori[1]), float(target_ori[2])
                ]

                with self.arm_lock:
                    # trajectory_mode/radio: 控制器侧轨迹平滑(治伺服刚性嗡颤); 手抖由 One Euro 处理
                    ret = self.arm.rm_movep_canfd(target_6d, self.follow_high,
                                                  self.traj_mode, self.traj_radio)

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
      observations/ee_pose:          (N, 6) float32  [末端笛卡尔位姿 x,y,z,rx,ry,rz (米/弧度), 机械臂直接返回, 非解算]
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
        self.data_buffer = {'qpos': [], 'ee_pose': [], 'images_top': [],
                            'images_wrist': [], 'timestamps': []}

    def start(self, filename):
        if self.is_recording:
            return
        self.filename = filename
        self.is_recording = True
        self.data_buffer = {'qpos': [], 'ee_pose': [], 'images_top': [],
                            'images_wrist': [], 'timestamps': []}
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
                # 末端位姿与关节角来自同一次 rm_get_current_arm_state() 调用:
                # state['pose'] = [x,y,z,rx,ry,rz] (米/弧度), 机械臂控制器直接返回, 非正/逆解算。
                # 读失败时用 6 个 0 兜底, 与 qpos 保持帧数对齐。
                ee_pose = [float(x) for x in state['pose'][:6]] if code == 0 else [0.0] * 6
                # 夹爪走独立串口，后台轮询线程已缓存反馈，无需 arm_lock
                gripper_val = self.gripper.get_position_normalized()

                img_top = self.cam_top.get_frame()
                img_wrist = self.cam_wrist.get_frame()

                self.data_buffer['qpos'].append(joint_angles + [gripper_val])
                self.data_buffer['ee_pose'].append(ee_pose)
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
                f.create_dataset('observations/ee_pose', data=np.array(self.data_buffer['ee_pose']))
                f.create_dataset('action', data=np.array(actions))
                f.create_dataset('timestamps', data=np.array(timestamps))
                # gzip 对照片压缩率低且单线程极慢; lzf 快约 5~10x, 中间文件转完即删无需高压缩比
                # (h5py 读取时透明解压, convert_to_lerobot.py 无需改动)
                f.create_dataset('observations/images/camera_global',
                                 data=np.array(self.data_buffer['images_top']),
                                 compression="lzf", chunks=True)
                f.create_dataset('observations/images/camera_left',
                                 data=np.array(self.data_buffer['images_wrist']),
                                 compression="lzf", chunks=True)
            print(f"保存: {os.path.basename(self.filename)} ({len(qpos)} frames)")
        except Exception as e:
            print(f"[!] 保存失败: {e}")


# ============ 按键表 (terminal / web 前端共用) ============
# 内置兜底按键表: configs/keybindings.json 缺失或损坏时使用, 保证脚本仍可运行。
# 字段: key=单键; action=动作名(对应 CollectorController.action_<name>);
#       args=动作参数(可选); label=界面显示; modes=可选["vive"|"teaching"](缺省=通用)。
FALLBACK_BINDINGS = [
    {"key": "w", "action": "toggle_teleop",  "label": "遥控 开/关(自动取当前Tracker为零点)", "modes": ["vive"]},
    {"key": "s", "action": "toggle_record",  "label": "录制 开始/保存"},
    {"key": "h", "action": "reset_arm",      "label": "复位到起始位(常速)"},
    {"key": "o", "action": "gripper_open",   "label": "夹爪张开"},
    {"key": "c", "action": "gripper_close",  "label": "夹爪闭合"},
    {"key": "1", "action": "gripper_pct", "args": {"pct": 30}, "label": "夹爪 30%"},
    {"key": "2", "action": "gripper_pct", "args": {"pct": 60}, "label": "夹爪 60%"},
    {"key": "q", "action": "quit",           "label": "退出"},
]


def load_keybindings(path):
    """读取 JSON 按键表; 缺失/损坏时回退到内置默认, 保证脚本仍可运行。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        bindings = data.get("bindings") if isinstance(data, dict) else data
        if not isinstance(bindings, list) or not bindings:
            raise ValueError("bindings 为空或格式错误")
        return bindings
    except (OSError, ValueError) as e:
        print(f"[!] 按键表 {path} 加载失败({e}), 使用内置默认")
        return FALLBACK_BINDINGS


def binding_available(binding, teaching):
    """按 modes 字段判断按键在当前模式是否可用 (无 modes = 两种模式通用)。"""
    modes = binding.get("modes")
    if not modes:
        return True
    return ("teaching" if teaching else "vive") in modes


def build_keymap(bindings, teaching):
    """按键字符 -> binding 映射 (仅收录当前模式可用、且为单键的绑定)。"""
    keymap = {}
    for b in bindings:
        key = b.get("key")
        if isinstance(key, str) and len(key) == 1 and binding_available(b, teaching):
            keymap[key] = b
    return keymap


def print_command_table(bindings, teaching):
    """据按键表打印命令提示 (只列当前模式可用的键)。"""
    print("-" * 50)
    print("单键即触发 (无需回车):")
    for b in bindings:
        if binding_available(b, teaching):
            print(f"  [{b.get('key')}] {b.get('label', b.get('action'))}")
    print("  [Ctrl+C] 强制退出")
    print("-" * 50)


# ============ 动作层 (terminal / web 前端共用) ============
class CollectorController:
    """把每个操作封装成命名动作, 供 terminal 与 web 前端共用, 保证两端行为一致。

    - dispatch(name, args): 按动作名分发到 action_<name> 方法;
    - snapshot()/print_status(): 统一状态, 供 terminal 状态行与后续 web 推送复用;
    - move_to_init(): 机械臂归位统一入口 (rm_movej_p 关节空间规划到笛卡尔位姿)。
    """

    _ACTION_PREFIX = "action_"

    def __init__(self, arm, arm_lock, gripper, vive_ctrl, recorder,
                 save_dir, task_name, teaching=False):
        self.arm = arm
        self.arm_lock = arm_lock
        self.gripper = gripper
        self.vive = vive_ctrl
        self.recorder = recorder
        self.save_dir = save_dir
        self.task_name = task_name
        self.teaching = teaching
        self.quit_requested = False
        self.saved_count = 0
        # ROBOT_INIT 合成 6 维笛卡尔位姿 [x,y,z, rx,ry,rz] (米/弧度), 供 movej_p 使用
        self.robot_init_pose = [float(x) for x in
                                (list(ROBOT_INIT_POS) + list(ROBOT_INIT_ORI))]

    # ---------- 机械臂归位 ----------
    def _read_arm_pose(self):
        """读机械臂当前笛卡尔位姿 [x,y,z,rx,ry,rz] (米/弧度); 失败返回 None。"""
        try:
            with self.arm_lock:
                code, state = self.arm.rm_get_current_arm_state()
            if code == 0 and state and state.get("pose"):
                return [float(x) for x in state["pose"][:6]]
        except Exception:
            pass
        return None

    def move_to_init(self, speed, block=1, countdown=0, label="归位", skip_if_near=False):
        """rm_movej_p 关节空间规划到 ROBOT_INIT (大位移最稳、不撞奇异点)。

        机械臂全程处于真实模式(run_mode=1), 规划运动直接执行即可 —— 切勿切到
        run_mode=0(那是"仿真"模式, 指令只在仿真里跑、真实机械臂不动)。
        失败仅打印告警、不下发危险运动; ret!=0 多为位姿不可达, 见 ROBOT_INIT 标定。
        skip_if_near=True: 若已在 ROBOT_INIT 附近(位置<ARM_HOME_POS_TOL 且姿态<ARM_HOME_ORI_TOL),
        直接跳过、不下发运动 —— 供启动慢速归位用, 免得每次重启都无谓地慢速跑一遍。
        """
        if skip_if_near:
            cur = self._read_arm_pose()
            if cur is not None:
                dp, dang = _pose_deviation(cur, self.robot_init_pose)
                if dp < ARM_HOME_POS_TOL and dang < ARM_HOME_ORI_TOL:
                    print(f"[i] 已在起始位附近 (位置 {dp * 100:.1f}cm / 姿态 {np.degrees(dang):.1f}°), 跳过{label}")
                    return True
        if countdown > 0:
            print(f"[!] 机械臂即将{label}到起始位, 请清空周围! {countdown} 秒后开始...")
            for i in range(countdown, 0, -1):
                print(f"    {i}...", flush=True)
                time.sleep(1.0)
        # 非阻塞下发: 只在发送指令的瞬间持锁, 随即释放, 让录制线程能在归位过程中持续
        # 采到机械臂真实轨迹。切勿用 block=1 —— 它会全程持锁数秒, 饿死录制线程,
        # 导致这段归位运动一帧都没录进 episode (表现为"复位数据被跳过")。
        with self.arm_lock:
            ret = self.arm.rm_movej_p(self.robot_init_pose, int(speed), 0, 0, 0)
        if ret != 0:
            print(f"[!] {label}失败: rm_movej_p ret={ret} (位姿可能不可达, 检查 ROBOT_INIT 标定)")
            return False
        if block:
            self._wait_until_arrived(label)
        print(f"[✓] {label}完成 (v={speed}%)")
        return True

    def _wait_until_arrived(self, label, timeout=60.0, poll=0.05):
        """轮询等待机械臂到达 ROBOT_INIT (配合非阻塞 movej_p 使用)。

        每次只短暂持锁读一次位姿、随即释放并 sleep, 空档让录制线程采到归位轨迹;
        到位(位置<ARM_HOME_POS_TOL 且姿态<ARM_HOME_ORI_TOL)即返回。超时仅告警不阻塞。
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            cur = self._read_arm_pose()
            if cur is not None:
                dp, dang = _pose_deviation(cur, self.robot_init_pose)
                if dp < ARM_HOME_POS_TOL and dang < ARM_HOME_ORI_TOL:
                    return True
            time.sleep(poll)
        print(f"[!] {label}等待到位超时 ({timeout:.0f}s): 机械臂可能未使能/被限位/急停, 或 ret=0 却没动")
        return False

    # ---------- 动作 (每个 action_<name> 对应按键表里的一个动作) ----------
    def action_calibrate_vive(self, **_):
        if self.teaching:
            print("[i] 示教模式无 Vive 校准")
            return
        self.vive.calibrate()

    def action_toggle_teleop(self, **_):
        """单键 toggle: 遥控 开<->关 (原 w/e 合并)。"""
        if self.teaching:
            print("[i] 示教模式无遥操")
            return
        if self.vive.control_enabled:
            self.vive.disable()
        else:
            self.vive.enable()
        self.print_status()

    def action_toggle_record(self, **_):
        """单键 toggle: 录制 开始<->停止并保存 (原 s/d 合并), 自动递增文件名。

        停止时先暂停遥操再保存: 按 s 立即让机械臂停止跟随手(不必等落盘),
        随后 _save() 阻塞期间机械臂已静止, 便于复位/摆场景、也防误动;
        下一条 episode 需重新按 w 启用(会以机械臂当前位姿重新取零点)。
        """
        if self.recorder.is_recording:
            # 先暂停遥操: 避免随后的阻塞式 _save() 期间机械臂仍跟随手(lzf 后保存变快, 这段延迟更明显)
            teleop_was_on = not self.teaching and self.vive.control_enabled
            if teleop_was_on:
                self.vive.disable()
            self.recorder.stop()
            self.saved_count += 1
            if teleop_was_on:
                print("[i] 录制结束已保存, 遥操已自动暂停 (下条按 w 重新启用)")
        else:
            filename = get_next_filename(self.save_dir, self.task_name)
            self.recorder.start(filename)
        self.print_status()

    def action_reset_arm(self, **_):
        """复位键: 常速归位到 ROBOT_INIT。复位前自动暂停遥操 (需手动按 w 重新启用)。

        录制中也允许复位: 归位运动会照常录进当前 episode (按需求不拦截)。
        """
        if not self.teaching and self.vive.control_enabled:
            self.vive.disable()
            time.sleep(0.15)   # 等一个控制周期, 让在途透传指令发完, 避免与归位争锁后补发旧位姿
            print("[i] 复位前已暂停遥操; 如需遥操请按 w 重新启用 (自动取当前位姿为零点)")
        if self.recorder.is_recording:
            print("[i] 录制中复位: 归位运动会录进当前 episode")
        self.move_to_init(ARM_HOME_SPEED_NORMAL, block=1, countdown=0, label="复位")
        self.print_status()

    def action_gripper_open(self, **_):
        self.gripper.open()
        print("夹爪: 张开")

    def action_gripper_close(self, **_):
        self.gripper.close()
        print("夹爪: 闭合")

    def action_gripper_pct(self, pct=100, **_):
        pct = max(0, min(100, int(pct)))
        self.gripper.move_pct(pct)
        print(f"夹爪: {pct}%")

    def action_quit(self, **_):
        self.quit_requested = True

    # ---------- 分发 / 状态 ----------
    def dispatch(self, name, args=None):
        """按动作名分发; 单个动作异常不中断采集循环。"""
        method = getattr(self, self._ACTION_PREFIX + str(name), None)
        if not callable(method):
            print(f"[!] 未知动作: {name}")
            return False
        try:
            method(**(args or {}))
        except Exception as e:  # noqa: BLE001 - 单个动作异常不应中断整个采集循环
            print(f"[!] 动作 {name} 异常: {e}")
        return True

    def snapshot(self):
        """当前状态快照 (terminal 状态行 / 后续 web 推送共用)。"""
        return {
            "mode": "teaching" if self.teaching else "vive",
            "teleop": bool(self.vive.control_enabled) if not self.teaching else False,
            "recording": bool(self.recorder.is_recording),
            "saved": int(self.saved_count),
            "gripper": round(float(self.gripper.get_position_normalized()), 3),
            "quit": bool(self.quit_requested),
        }

    def print_status(self):
        s = self.snapshot()
        teleop = "启用" if s["teleop"] else "暂停"
        rec = "录制中" if s["recording"] else "空闲"
        print(f"[状态] 遥控:{teleop} | {rec} | 已存 {s['saved']} 条 | "
              f"夹爪 {int(s['gripper'] * 100)}%")


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
    parser.add_argument('--keybindings', type=str, default=DEFAULT_KEYBINDINGS_PATH,
                        help='按键表 JSON 路径（terminal/web 前端共用）')
    parser.add_argument('--no-home', action='store_true',
                        help='启动时不自动慢速归位到 ROBOT_INIT')
    parser.add_argument('--home-countdown', type=int, default=ARM_HOME_COUNTDOWN,
                        help='启动归位前倒计时秒数（0=不倒计时）')
    parser.add_argument('--control-hz', type=float, default=VIVE_CONTROL_HZ,
                        help='Vive 遥操透传频率(Hz)，越高跟随越紧延迟越低，默认 %(default)s')
    parser.add_argument('--high-follow', action=argparse.BooleanOptionalAction, default=VIVE_FOLLOW_HIGH,
                        help='高跟随模式(follow=True)：滞后最小，要求透传≥100Hz(不足自动抬到100)。'
                             '默认开；用 --no-high-follow 关(改回低跟随，控制器自规划更软)')
    parser.add_argument('--traj-mode', type=int, default=VIVE_TRAJ_MODE, choices=[0, 1, 2],
                        help='高跟随控制器侧轨迹平滑(治伺服嗡颤)：0=完全透传 1=曲线拟合 2=滤波(默认)，默认 %(default)s')
    parser.add_argument('--traj-radio', type=int, default=VIVE_TRAJ_RADIO,
                        help='轨迹平滑系数：拟合0~100/滤波0~1000，越大越平滑滞后越大，默认 %(default)s')
    parser.add_argument('--oe-mincutoff', type=float, default=VIVE_OE_MINCUTOFF,
                        help='One Euro 零速截止频率(Hz)，越低越压静止手抖(慢速滞后略增)，默认 %(default)s')
    parser.add_argument('--oe-beta', type=float, default=VIVE_OE_BETA,
                        help='One Euro 速度系数，越大快速运动越跟手；默认 %(default)s(刻意低=重阻尼防跟随太快)，嫌拖调大(经验10~30)')
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
                                   enable_vive=not args.teaching,
                                   control_hz=args.control_hz,
                                   follow_high=args.high_follow,
                                   traj_mode=args.traj_mode,
                                   traj_radio=args.traj_radio,
                                   oe_mincutoff=args.oe_mincutoff,
                                   oe_beta=args.oe_beta,
                                   oe_dcutoff=VIVE_OE_DCUTOFF)
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

    # 动作层: terminal 与(后续)web 前端共用同一套命名动作
    controller = CollectorController(
        arm, arm_lock, gripper, vive_ctrl, recorder,
        save_dir=args.save_dir, task_name=args.task_name, teaching=args.teaching,
    )

    # 启动慢速归位 (按要求放在 Vive 连接之后): 上电后位姿未知, 慢速求稳;
    # skip_if_near=True: 已在 ROBOT_INIT 附近就跳过, 不做无谓的慢速运动
    if not args.no_home:
        controller.move_to_init(ARM_HOME_SPEED_SLOW, block=1,
                                countdown=max(0, args.home_countdown), label="慢速归位",
                                skip_if_near=True)

    # 加载按键表 (terminal/web 共用 JSON), 据此生成命令提示与单键映射
    bindings = load_keybindings(args.keybindings)
    keymap = build_keymap(bindings, args.teaching)
    print()
    print_command_table(bindings, args.teaching)

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        controller.print_status()
        # 单键即触发: 读到一个字符立刻查表分发, 无需回车
        while not controller.quit_requested:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not rlist:
                continue
            char = sys.stdin.read(1)
            if char == '\x03':            # Ctrl+C 兜底 (cbreak 下一般已抛 KeyboardInterrupt)
                break
            binding = keymap.get(char) or keymap.get(char.lower())
            if binding:
                controller.dispatch(binding["action"], binding.get("args"))

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
