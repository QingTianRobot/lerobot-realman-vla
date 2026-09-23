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
  - observations/images/cam_high: (N, H, W, 3)
  - observations/images/cam_wrist: (N, H, W, 3)
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
ROBOT_INIT_POS = np.array([-0.218, 0.06, 0.357])
ROBOT_INIT_ORI = np.array([-3.126, 0.001, -0.015])
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


# ============ Vive 遥控模块 ============
class ViveController:
    """Vive Tracker 遥操作控制器

    原理：读取 Tracker 的笛卡尔位姿增量，映射到机械臂的笛卡尔空间。
    坐标映射关系（Vive → Robot）：
      Robot_X = -Vive_Z
      Robot_Y = -Vive_X
      Robot_Z = +Vive_Y
    """

    def __init__(self, arm, arm_lock, tracker_serial=None, enable_vive=True):
        self.arm = arm
        self.arm_lock = arm_lock
        self.tracker_serial = tracker_serial or DEFAULT_TRACKER_SERIAL
        self.enable_vive = enable_vive

        self.robot_init_pos = ROBOT_INIT_POS.copy()
        self.robot_init_ori = ROBOT_INIT_ORI.copy()

        self.vive_init_pos = None
        self.vive_init_ori = None
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
        try:
            # 需要 hardware/vive_tracker.py
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hardware'))
            import vive_tracker as triad_lib
            self.vive = triad_lib.triad_openvr()
            for name, device in self.vive.devices.items():
                if "tracker" in name.lower():
                    serial = device.get_serial()
                    if serial == self.tracker_serial:
                        self.tracker = device
                        break
            if self.tracker is None:
                for name, device in self.vive.devices.items():
                    if "tracker" in name.lower():
                        self.tracker = device
                        break
        except Exception as e:
            print(f"[!] Vive: {e}")

    def calibrate(self):
        """校准：记录当前 Vive Tracker 位姿作为零点"""
        if self.tracker is None:
            print("[!] Vive 未连接")
            return False

        print("校准中... 保持 Tracker 静止")
        poses = []
        for _ in range(30):
            euler = self.tracker.get_pose_euler()
            if euler:
                poses.append(euler)
            time.sleep(0.033)

        if not poses:
            print("[!] 校准失败")
            return False

        avg_pose = np.mean(poses, axis=0)
        self.vive_init_pos = np.array([avg_pose[0], avg_pose[1], avg_pose[2]])
        self.vive_init_ori = np.array([avg_pose[3], avg_pose[4], avg_pose[5]])
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

        while self.running:
            time.sleep(interval)
            if not self.control_enabled or self.tracker is None or self.vive_init_pos is None:
                continue

            try:
                euler = self.tracker.get_pose_euler()
                if euler is None:
                    continue

                cur_pos = np.array([euler[0], euler[1], euler[2]])
                cur_ori = np.array([euler[3], euler[4], euler[5]])

                delta_pos = cur_pos - self.vive_init_pos
                delta_ori = cur_ori - self.vive_init_ori

                scale_pos = 0.5   # 位置灵敏度: 调大=机械臂动得比手多, 调小=更细腻
                scale_ori = 0.3   # 姿态灵敏度: 同上, 作用于旋转

                # ================= 坐标映射: Vive → Robot（现场标定就改这里）=================
                # 数组三行依次对应机械臂的 X / Y / Z 轴。
                # 每行 = ±delta_pos[vive轴下标] * scale_pos
                #   delta_pos 下标: 0=Vive_X, 1=Vive_Y, 2=Vive_Z
                # 调法（按 w 启用后，手推 tracker 观察机械臂）:
                #   · 某轴方向反了        → 把那一行的符号翻转（- 改 + 或 + 改 -）
                #   · 推 A 方向却动了 B 轴 → 把那一行的 delta_pos 下标换成正确的 vive 轴
                #   · 幅度太大/太小        → 调上面的 scale_pos
                target_pos = self.robot_init_pos + np.array([
                    -delta_pos[2] * scale_pos,   # Robot_X ← -Vive_Z（前后）
                    -delta_pos[0] * scale_pos,   # Robot_Y ← -Vive_X（左右）
                    delta_pos[1] * scale_pos     # Robot_Z ← +Vive_Y（上下）
                ])

                # 姿态映射，规则同上；三行依次对应机械臂姿态的 Rx / Ry / Rz。
                # delta_ori 下标: 0=yaw, 1=pitch, 2=roll（角度，故乘 π/180 转弧度）
                target_ori = self.robot_init_ori + np.array([
                    -delta_ori[2] * scale_ori * np.pi / 180,   # Rx ← -roll
                    -delta_ori[1] * scale_ori * np.pi / 180,   # Ry ← -pitch
                    delta_ori[0] * scale_ori * np.pi / 180     # Rz ← +yaw
                ])

                # 安全限位（笛卡尔工作空间, 单位米）: 防止手滑把机械臂推出安全区。
                # 换成你的实际可达范围; 若机械臂总在某方向到不了边界, 放宽对应上下限。
                target_pos[0] = np.clip(target_pos[0], -0.5, 0.1)
                target_pos[1] = np.clip(target_pos[1], -0.3, 0.3)
                target_pos[2] = np.clip(target_pos[2], 0.1, 0.6)

                target_6d = [
                    float(target_pos[0]), float(target_pos[1]), float(target_pos[2]),
                    float(target_ori[0]), float(target_ori[1]), float(target_ori[2])
                ]

                with self.arm_lock:
                    self.arm.rm_movep_canfd(target_6d, False, 0, 60)

            except Exception:
                pass

    def shutdown(self):
        self.running = False


# ============ 数据录制模块 ============
class DataRecorder:
    """以固定频率录制机械臂状态、相机图像到 HDF5

    数据格式:
      observations/qpos:             (N, 7) float32  [6关节角 + 夹爪位置]
      observations/images/cam_high:  (N, H, W, 3) uint8
      observations/images/cam_wrist: (N, H, W, 3) uint8
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
                f.create_dataset('observations/images/cam_high',
                                 data=np.array(self.data_buffer['images_top']),
                                 compression="gzip")
                f.create_dataset('observations/images/cam_wrist',
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

    # 初始化 Vive 遥控
    vive_ctrl = ViveController(arm, arm_lock, tracker_serial=args.tracker_serial,
                               enable_vive=not args.teaching)
    if not args.teaching:
        print(f"Vive: {'OK' if vive_ctrl.tracker else 'FAIL'}")

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
