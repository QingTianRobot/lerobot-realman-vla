#!/usr/bin/env python3
"""
SmolVLA / Pi0.5 推理脚本 — RealMan RM65

自动检测并加载 SmolVLA 或 Pi0.5 策略
加载预处理/后处理管线，实时控制机械臂。

特性:
  - 自动从 config.json 检测策略类型
  - 默认使用 LeRobot 0.4.3 原生 RTC，后台生成 chunk，与动作执行重叠
  - EMA 动作平滑 + 死区过滤（减少抖动）
  - 异步夹爪控制（Modbus 写入不阻塞主循环）
  - VLA: Language instruction 支持

用法:
  # SmolVLA (需要 task 描述)
  python scripts/inference.py \\
      --model outputs/smolvla_realman/checkpoints/50000/pretrained_model \\
      --task "pick up the cube" --freq 15

  # Pi0.5 (离线模式，不下载HuggingFace)
  python scripts/inference.py \\
      --model outputs/pi05_realman/checkpoints/30000/pretrained_model \\
      --task "pick up the cube" --offline
"""

import torch
import numpy as np
import time
import os
import sys
import threading
import cv2
import json
import argparse
import math
from importlib.metadata import version
from pathlib import Path

# ============ 默认硬件配置（根据你的硬件修改） ============
DEFAULT_ARM_IP = "192.168.5.123"
DEFAULT_ARM_PORT = 8080
DEFAULT_CAM_TOP_SERIAL = "262322074840"      # 顶部 D435 (rs-enumerate-devices | grep Serial)
DEFAULT_CAM_WRIST_SERIAL = "CV2T66100096"    # 腕部 Orbbec 305 (留空取第一个设备)

# 知行 RTU 夹爪参数（独立串口，与机械臂解耦）
GRIPPER_PORT = "/dev/realman/gripper_left"
GRIPPER_SLAVE_ID = 2          # 知行夹爪 Modbus 从站地址
GRIPPER_BAUDRATE = 115200
GRIPPER_MAX_POSITION = 9000   # 归一化行程上限兜底值 (设备单位, /100=mm); 若存在 hardware/gripper_calibration.json 则以标定值为准 (需与采集时一致)
GRIPPER_SPEED_PCT = 100       # 电机行程速度(0~100); ⚠ 必须与采集端(collect_data --gripper-speed)一致, 否则开合动态与训练数据不符
GRIPPER_POLL_HZ = 25          # RS-485 总线轮询频率(Hz): 异步命令最长排队时延=1/hz
GRIPPER_DEADBAND = 0.02       # 与 collect_data.py 的 PIKA_TELEOP_DEADBAND 一致，减少无效 RTU 写入

# 与 collect_data.py ROBOT_INIT_POS + ROBOT_INIT_ORI 保持一致 (笛卡尔位姿 [x,y,z,rx,ry,rz], 米/弧度)
INIT_POSE = np.array([-0.0847, -0.2821, 0.0872, -3.102, 0.065, 1.609], dtype=np.float32)

# ------ 硬件模块路径 ------
_HARDWARE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hardware')
if _HARDWARE_DIR not in sys.path:
    sys.path.insert(0, _HARDWARE_DIR)


# ============ 版本兼容性补丁 ============
def _patch_config_compat(config_path):
    """清理 config.json 中当前版本 LeRobot 不支持的字段"""
    import dataclasses
    with open(config_path) as f:
        cfg = json.load(f)

    # 尝试获取对应策略的有效字段
    policy_type = cfg.get("type", "")
    valid_fields = set()

    try:
        if policy_type == "smolvla":
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
            valid_fields = {f.name for f in dataclasses.fields(SmolVLAConfig)}
        from lerobot.configs.policies import PreTrainedConfig
        valid_fields |= {f.name for f in dataclasses.fields(PreTrainedConfig)}
    except ImportError:
        return

    removed = []
    for key in list(cfg.keys()):
        if key not in valid_fields and key != "type":
            removed.append(key)
            del cfg[key]

    if removed:
        print(f"  ⚠️  过滤不兼容字段: {removed}")
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=4)


# ============ 策略加载器 ============
def load_policy(model_path: str, device: torch.device):
    """检测并加载 SmolVLA 或 Pi0.5 策略，返回 (policy, policy_type)。"""
    model_path = Path(model_path)
    config_path = model_path / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    policy_type = config.get("type", "unknown")
    print(f"检测到策略类型: {policy_type}")

    if policy_type == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        _patch_config_compat(config_path)
        policy = SmolVLAPolicy.from_pretrained(str(model_path))

    elif policy_type == "pi05":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        policy = PI05Policy.from_pretrained(str(model_path))

    else:
        raise ValueError(f"不支持的策略类型: {policy_type}")

    policy.to(device)
    policy.eval()
    return policy, policy_type


# ============ 辅助函数 ============
# 夹爪已改为知行 RTU 独立串口控制，归一化转换由 ChangingtekGripper 内部处理，
# 不再需要 Modbus 寄存器编解码。


# ============ 机械臂控制器 ============
class RobotController:
    """机械臂控制（支持 EMA 平滑 + 死区过滤 + 异步夹爪）

    关键设计:
      - EMA (Exponential Moving Average): 平滑关节角指令，减少抖动
      - 死区过滤: 变化小于阈值时不发送，减少通信压力
      - 异步夹爪: Modbus 写入在独立线程，不阻塞主控制循环
    """

    def __init__(self, arm, gripper, ema_alpha=0.3, joint_deadzone=0.5):
        self.arm = arm
        self.gripper = gripper
        self.lock = threading.Lock()

        self._last_gripper_cmd = None
        self._smoothed_action = None
        self._last_joint_cmd = None
        self.ema_alpha = ema_alpha        # EMA 系数 (0.3=平滑, 0.7=响应快; 1.0=不平滑)
        self.joint_deadzone = joint_deadzone  # 死区阈值(度)

    def get_qpos(self):
        """获取当前 [6关节角 + 夹爪位置]"""
        with self.lock:
            joint_state = self.arm.rm_get_current_arm_state()
            joint_angles = joint_state[1]['joint'][:6]
            # 与采集端一致：使用夹爪后台轮询到的实时归一化反馈，而非上一次命令值。
            gripper_pos = self.gripper.get_position_normalized()

            return np.array(joint_angles + [gripper_pos], dtype=np.float32)

    def set_qpos(self, qpos):
        """设置目标关节角 + 夹爪

        关节目标经过 EMA 和死区过滤后下发，夹爪目标异步下发。
        """
        with self.lock:
            joint_target = qpos[:6].copy()
            # EMA 平滑
            if self._smoothed_action is None:
                self._smoothed_action = joint_target.copy()
            else:
                self._smoothed_action = (
                    self.ema_alpha * joint_target +
                    (1 - self.ema_alpha) * self._smoothed_action
                )
            joint_target = self._smoothed_action.copy()

            # 死区过滤
            should_send = True
            if self._last_joint_cmd is not None:
                max_delta = np.max(np.abs(joint_target - self._last_joint_cmd))
                if max_delta < self.joint_deadzone:
                    should_send = False

            if should_send:
                # CANFD 透传：控制循环按 30 Hz 下发，使用低跟随模式。
                self.arm.rm_movej_canfd(joint_target.tolist(), False, 0, 0, 0)
                self._last_joint_cmd = joint_target.copy()

            # 夹爪（知行 RTU 内部异步下发）。与 collect_data 一样直接使用连续的
            # 归一化目标：0=闭合、1=张开；死区仅用于避免重复写入 RS-485 总线。
            gripper_target = float(np.clip(qpos[6], 0.0, 1.0))
            if (self._last_gripper_cmd is None or
                    abs(gripper_target - self._last_gripper_cmd) >= GRIPPER_DEADBAND):
                self.gripper.move_normalized(gripper_target)
                self._last_gripper_cmd = gripper_target

    def move_to_init(self, init_pose):
        """移动到初始位姿（阻塞）— 笛卡尔空间 movej_p, 与 collect_data 一致"""
        print(f"移动到初始位姿...")
        self.arm.rm_movej_p(init_pose[:6].tolist(), 80, 0, 0, 1)

        self.gripper.move_normalized(1.0)
        time.sleep(1.0)

    def homing_on_exit(self, init_pose):
        """退出前安全归位: 先停掉残留运动/清错误, 再 movej_p 回初始位姿并张开夹爪"""
        try:
            self.arm.rm_set_arm_stop()   # 取消推理循环遗留的排队运动指令
            self.arm.rm_clear_system_err()
            time.sleep(0.5)
            self.move_to_init(init_pose)
            print("机械臂已归位, 夹爪已张开")
        except KeyboardInterrupt:
            # 归位过程中再次 Ctrl+C: 立即急停, 不再继续归位
            print("\n归位被中断, 急停")
            self.stop()
        except Exception as e:
            print(f"[!] 退出归位失败: {type(e).__name__}: {e} (机械臂可能停留在当前位置)")

    def stop(self):
        try:
            self.arm.rm_set_arm_stop()
        except Exception:
            pass

    def close(self):
        try:
            self.gripper.disable()
            self.gripper.disconnect()
        except Exception:
            pass
        self.arm.rm_delete_robot_arm()


def _get_action_queue(policy):
    """取策略内部动作队列；无队列时返回 None。"""
    q = getattr(policy, "_queues", None)
    if isinstance(q, dict) and "action" in q:
        return q["action"]
    return getattr(policy, "_action_queue", None)


def wait_for_next_chunk(chunk_number):
    """在生成并执行一个新动作 chunk 前等待人工确认。"""
    input(f"\n[Chunk {chunk_number}] 按 Enter 生成并执行下一个 chunk (Ctrl+C 终止)...")


def _capture_image_tensor(camera):
    """按采集/训练链路读取一帧并转换成策略输入。

    相机驱动的 ``get_frame()`` 统一返回 BGR ``uint8``；采集数据在转换到
    LeRobot 时执行同样的 BGR->RGB 转换。这里保持完全相同的颜色和数值
    语义，最后才转成策略使用的 CHW ``float32`` [0, 1] tensor。

    返回 (tensor, bgr_image)，后者仅用于 OpenCV 显示，确保显示和策略看到
    的是同一帧。
    """
    frame_bgr = camera.get_frame()
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise RuntimeError(f"相机返回无效图像: shape={getattr(frame_bgr, 'shape', None)}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous().float() / 255.0
    return image_tensor, frame_bgr


# ============ 主推理循环 ============
def main():
    parser = argparse.ArgumentParser(description='RealMan SmolVLA / Pi0.5 策略推理')
    parser.add_argument('--model', type=str, required=True,
                        help='模型路径 (仅支持 SmolVLA 或 Pi0.5)')
    parser.add_argument('--arm-ip', type=str, default=DEFAULT_ARM_IP, help='机械臂IP')
    parser.add_argument('--arm-port', type=int, default=DEFAULT_ARM_PORT, help='机械臂端口')
    parser.add_argument('--cam-top', type=str, default=DEFAULT_CAM_TOP_SERIAL, help='顶部相机(D435)序列号')
    parser.add_argument('--cam-wrist', type=str, default=DEFAULT_CAM_WRIST_SERIAL, help='腕部相机(Orbbec 305)序列号，留空取第一个设备')
    parser.add_argument('--gripper-port', type=str, default=GRIPPER_PORT, help='知行夹爪串口')
    parser.add_argument('--gripper-slave-id', type=int, default=GRIPPER_SLAVE_ID, help='知行夹爪 Modbus 从站地址')
    parser.add_argument('--gripper-speed', type=int, default=GRIPPER_SPEED_PCT,
                        help='夹爪电机行程速度(0~100, 越大开合越快), 应与采集时一致, 默认 %(default)s')
    parser.add_argument('--gripper-poll-hz', type=float, default=GRIPPER_POLL_HZ,
                        help='夹爪 RS-485 总线轮询频率(Hz), 默认 %(default)s')
    parser.add_argument('--freq', type=float, default=30.0,
                        help='控制频率(Hz)，应与训练数据fps一致')
    parser.add_argument('--task', type=str, default='pick up the cube',
                        help='VLA 任务描述 (SmolVLA/Pi0.5 需要)')
    parser.add_argument('--headless', action='store_true', help='无GUI模式')
    parser.add_argument('--offline', action='store_true', help='离线模式 (不从 HuggingFace 下载)')
    parser.add_argument('--wait-for-next-chunk', '--wait_for_next_chunk',
                        dest='wait_for_next_chunk', action='store_true',
                        help='每次生成新动作 chunk 前等待 Enter 确认')
    parser.add_argument('--rtc', action=argparse.BooleanOptionalAction, default=None,
                        help='原生 RTC 异步推理，默认开启；--no-rtc 恢复串行模式')
    parser.add_argument('--rtc-delay-ms', type=float, default=125.0,
                        help='初始推理延迟估计(ms)，运行中按实测最大延迟上调，默认 %(default)s')
    parser.add_argument('--rtc-margin-steps', type=int, default=2,
                        help='提前推理的额外控制步数，默认 %(default)s')
    parser.add_argument('--rtc-execution-horizon', type=int, default=10,
                        help='RTC 前缀引导范围，至少覆盖延迟+余量，默认 %(default)s')
    parser.add_argument('--ema-alpha', type=float, default=0.3,
                        help='EMA 系数 (默认 %(default)s; 1.0=不平滑)')
    parser.add_argument('--deadzone', type=float, default=0.5,
                        help='关节死区阈值(度), 默认 %(default)s; 0=不过滤')
    args = parser.parse_args()
    if not math.isfinite(args.freq) or args.freq <= 0:
        parser.error('--freq 必须为正的有限数')
    if args.rtc and args.wait_for_next_chunk:
        parser.error('--rtc 与 --wait-for-next-chunk 不能同时使用')
    if args.rtc is None:
        args.rtc = not args.wait_for_next_chunk
    installed_version = version('lerobot')
    print(f"LeRobot: {installed_version} | Python: {sys.executable}")
    if args.rtc:
        from rtc_inference import RTCInference, check_lerobot_version
        check_lerobot_version()

    if args.offline:
        import os
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

    # 导入机械臂SDK
    from Robotic_Arm.rm_robot_interface import (
        RoboticArm, rm_thread_mode_e
    )
    from orbbec_camera import OrbbecCamera
    from realsense_camera import RealSenseCamera
    from changingtek_gripper import ChangingtekGripper
    from lerobot.processor.pipeline import DataProcessorPipeline

    print("=" * 50)
    print("  RealMan RM65 SmolVLA / Pi0.5 策略推理")
    print("=" * 50)

    # 1. 加载模型
    print("\n[1/5] Loading policy...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, policy_type = load_policy(args.model, device)
    print(f"      Device: {device}")

    preprocessor = DataProcessorPipeline.from_pretrained(
        args.model, config_filename='policy_preprocessor.json')
    postprocessor = DataProcessorPipeline.from_pretrained(
        args.model, config_filename='policy_postprocessor.json')

    rtc = None
    if args.rtc:
        rtc = RTCInference(
            policy, preprocessor, postprocessor, freq=args.freq,
            delay_ms=args.rtc_delay_ms, margin_steps=args.rtc_margin_steps,
            execution_horizon=args.rtc_execution_horizon,
        )
        print(f"      RTC: ON | chunk={rtc.chunk_size}, replan target={rtc.replan_steps} steps, "
              f"delay={rtc.delay_steps}+{rtc.margin_steps} reserve steps")
    else:
        # Checkpoints may already contain an enabled RTC config; select_action forbids it.
        policy.config.rtc_config = None
        policy.init_rtc_processor()
        print("      RTC: OFF (串行 / 人工逐 chunk 模式)")

    # 检测输入特征
    input_features = policy.config.input_features
    use_cam_high = 'observation.images.camera_global' in input_features
    use_cam_wrist = 'observation.images.camera_left' in input_features
    print(f"      Input features: {list(input_features.keys())}")

    # 2. 初始化硬件
    print("\n[2/5] Initializing hardware...")
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(args.arm_ip, args.arm_port)
    if handle.id == -1:
        raise RuntimeError(f"机械臂连接失败: {args.arm_ip}:{args.arm_port}")

    arm.rm_stop_drag_teach()

    # 初始化知行夹爪（独立串口，与机械臂解耦）
    gripper = ChangingtekGripper(
        port=args.gripper_port, slave_id=args.gripper_slave_id,
        baudrate=GRIPPER_BAUDRATE, max_position=GRIPPER_MAX_POSITION,
        speed_pct=args.gripper_speed, poll_hz=args.gripper_poll_hz,
    )
    gripper.connect()
    print(f"      夹爪: {'OK' if gripper.connected else 'FAIL'} ({args.gripper_port})")
    robot = RobotController(
        arm, gripper,
        ema_alpha=args.ema_alpha,
        joint_deadzone=args.deadzone,
    )

    cam_top = RealSenseCamera(args.cam_top) if use_cam_high else None
    # 重构后 RealSenseCamera 失败不再 raise (改为 is_active=False), 这里补上 fail-fast,
    # 避免推理时静默使用黑帧。排查参见 hardware/README.md → RealSense USB 断联 / 设备占用排查。
    if use_cam_high and not cam_top.is_active:
        raise RuntimeError(
            f"顶部 RealSense D435 初始化失败 (serial={args.cam_top}); "
            f"先试 pkill -f realsense 或物理拔插 USB 后重试"
        )
    cam_wrist = OrbbecCamera(args.cam_wrist) if use_cam_wrist else None
    if use_cam_wrist and not cam_wrist.is_active:
        raise RuntimeError(
            f"腕部 Orbbec 初始化失败 (serial={args.cam_wrist}); "
            "请检查设备连接、序列号以及 Orbbec SDK 环境"
        )
    time.sleep(1)

    # 3. 移动到初始位姿
    print("\n[3/5] Moving to initial pose...")
    robot.move_to_init(INIT_POSE)

    # 4. 等待确认
    print("\n[4/5] Ready to execute")
    if not args.headless:
        print("      按 Enter 开始推理...")
        input()
    else:
        print("      0.5秒后开始...")
        time.sleep(0.5)

    # 5. 推理循环
    print(f"\n[5/5] Executing {policy_type} at {args.freq}Hz (Ctrl+C to stop)...")
    step_count = 0
    control_period = 1.0 / args.freq
    policy.reset()

    action_queue = _get_action_queue(policy)   # 须在 reset() 之后取, 保证与 select_action 同一 deque
    chunk_id = 0
    prev_loop_start = None

    try:
        while True:
            # 用单调时钟记录循环起点；实际频率定义为相邻两次循环起点的时间差，
            # 因而包含本轮处理耗时和上一轮 sleep/调度抖动。
            loop_start = time.perf_counter()
            actual_freq = (
                1.0 / (loop_start - prev_loop_start)
                if prev_loop_start is not None and loop_start > prev_loop_start
                else 0.0
            )
            prev_loop_start = loop_start
            timings = {}

            # 仅检查 Future；推理未完成时继续消费已有动作，不等待 GPU。
            if rtc:
                t0 = time.perf_counter()
                rtc.poll()
                timings['rtc_poll'] = time.perf_counter() - t0
                chunk_id = rtc.chunk_id
            need_observation = rtc is None or rtc.needs_observation()

            # chunk 边界检测: 队列空 → 本步将触发一次前向重规划(新 chunk)
            if rtc:
                is_new_chunk = False
            elif action_queue is not None:
                is_new_chunk = (len(action_queue) == 0)
            else:
                is_new_chunk = True

            # 在采集观测前等待，以确保确认后使用的是最新画面和机械臂状态。
            if is_new_chunk and args.wait_for_next_chunk:
                t0 = time.perf_counter()
                wait_for_next_chunk(chunk_id + 1)
                timings['wait'] = time.perf_counter() - t0

            # 获取观测
            observation_time = time.perf_counter()
            qpos = robot.get_qpos()
            observation = {
                'observation.state': torch.from_numpy(qpos).float(),
            }
            display_frames = []

            if cam_top and (need_observation or not args.headless):
                image_tensor, frame_bgr = _capture_image_tensor(cam_top)
                observation['observation.images.camera_global'] = image_tensor
                display_frames.append(frame_bgr)

            if cam_wrist and (need_observation or not args.headless):
                image_tensor, frame_bgr = _capture_image_tensor(cam_wrist)
                observation['observation.images.camera_left'] = image_tensor
                display_frames.append(frame_bgr)
            timings['observe'] = time.perf_counter() - observation_time

            # VLA 需要 language instruction
            if policy_type in ("pi05", "smolvla"):
                observation['task'] = args.task

            if rtc:
                if need_observation:
                    rtc.submit(observation, observation_time=observation_time)
                if rtc.chunk_id == 0:
                    # 仅首次启动允许等待：此时尚无动作可与推理重叠。
                    warmup_start = time.perf_counter()
                    rtc.poll(wait=True)
                    print(f"[RTC] 首段就绪: {(time.perf_counter() - warmup_start) * 1000:.1f}ms")
                    loop_start = time.perf_counter()
                    prev_loop_start = loop_start
                action = rtc.pop()
                chunk_id = rtc.chunk_id
            else:
                # 串行模式保留，用于对照及人工逐 chunk 检查。
                t0 = time.perf_counter()
                observation = preprocessor(observation)
                timings['preprocess'] = time.perf_counter() - t0

                t0 = time.perf_counter()
                with torch.no_grad():
                    action_tensor = policy.select_action(observation)
                timings['inference'] = time.perf_counter() - t0

                t0 = time.perf_counter()
                action_dict = postprocessor({'action': action_tensor})
                action = action_dict['action'][0].cpu().numpy()
                timings['postprocess'] = time.perf_counter() - t0

                if is_new_chunk:
                    chunk_id += 1

            # 执行
            t0 = time.perf_counter()
            robot.set_qpos(action)
            timings['control'] = time.perf_counter() - t0

            # 可视化
            t0 = time.perf_counter()
            if not args.headless:
                try:
                    frames = [cv2.resize(frame, (320, 240)) for frame in display_frames]
                    if frames:
                        display = np.hstack(frames) if len(frames) > 1 else frames[0]
                        cv2.putText(display, f"{policy_type} Step: {step_count}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow("Inference", display)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                except cv2.error:
                    args.headless = True
            timings['visualize'] = time.perf_counter() - t0

            # 频率控制
            remaining = control_period - (time.perf_counter() - loop_start)
            if remaining > 0:
                t0 = time.perf_counter()
                time.sleep(remaining)
                timings['sleep'] = time.perf_counter() - t0

            # 日志放在 sleep 之后，确保包含完整一轮的关键步骤耗时。
            if step_count % 1 == 0:
                joints_str = " ".join(
                    f"J{i + 1}:{qpos[i]:6.1f}→{action[i]:6.1f}" for i in range(6)
                )
                timing_str = " ".join(
                    f"{name}={duration * 1000:.1f}ms"
                    for name, duration in timings.items()
                )
                if rtc:
                    timing_str += (f" | RTC queue={rtc.queue.qsize()} "
                                   f"async={rtc.last_latency * 1000:.1f}ms "
                                   f"skip={rtc.last_delay} budget={rtc.delay_steps}")
                print(f"[{policy_type}] Step {step_count:4d} | Chunk {chunk_id:2d} | "
                    #   f"{joints_str} | "
                      f"夹爪: {qpos[6]:5.2f}→{np.clip(action[6], 0.0, 1.0):5.2f} | "
                      f"freq={actual_freq:.1f}Hz | {timing_str}")

            step_count += 1

    except KeyboardInterrupt:
        print("\n\n推理终止")
    finally:
        # Shutdown order matters: stop camera SDK worker threads before native
        # arm/gripper teardown. Otherwise Ctrl+C can make a C++ destructor join
        # its own callback thread (std::system_error: Resource deadlock avoided).
        try:
            if rtc:
                robot.stop()
                rtc.close()
        except Exception as e:
            print(f"[!] RTC 退出清理失败: {type(e).__name__}: {e}")
        for camera_name, camera in (("顶部相机", cam_top), ("腕部相机", cam_wrist)):
            if camera:
                try:
                    camera.close()
                except Exception as e:
                    print(f"[!] {camera_name} 退出清理失败: {type(e).__name__}: {e}")
        # 退出前让机械臂回到 POS_INIT 并张开夹爪 (归位内部已含急停保护)
        try:
            robot.homing_on_exit(INIT_POSE)
        except Exception as e:
            print(f"[!] 机械臂归位清理失败: {type(e).__name__}: {e}")
        try:
            if not args.headless:
                cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            robot.close()
        except Exception as e:
            print(f"[!] 硬件释放失败: {type(e).__name__}: {e}")
        print("硬件已关闭")


if __name__ == "__main__":
    main()
