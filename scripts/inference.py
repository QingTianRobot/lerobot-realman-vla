#!/usr/bin/env python3
"""
通用多策略推理脚本 — RealMan RM65

自动检测策略类型 (ACT / Diffusion / VQ-BeT / SmolVLA / Pi0 / Pi0.5)
加载预处理/后处理管线，实时控制机械臂。

特性:
  - 自动从 config.json 检测策略类型
  - EMA 动作平滑 + 死区过滤（减少抖动）
  - 异步夹爪控制（Modbus 写入不阻塞主循环）
  - ACT: Temporal Ensemble 支持
  - VLA: Language instruction 支持

用法:
  # ACT
  python scripts/inference.py \\
      --model outputs/act_realman/checkpoints/100000/pretrained_model \\
      --arm-ip <YOUR_ARM_IP> --freq 30

  # SmolVLA (需要 task 描述)
  python scripts/inference.py \\
      --model outputs/smolvla_realman/checkpoints/50000/pretrained_model \\
      --task "pick up the cube" --freq 15

  # Pi0 (离线模式，不下载HuggingFace)
  python scripts/inference.py \\
      --model outputs/pi0_realman/checkpoints/30000/pretrained_model \\
      --task "pick up the cube" --offline
"""

import torch
import numpy as np
import time
import os
import sys
import csv
import threading
import cv2
import json
import argparse
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
from orbbec_camera import OrbbecCamera              # 腕部相机 (奥比中光)
from realsense_camera import RealSenseCamera        # 顶部相机 (Intel RealSense D435)
from changingtek_gripper import ChangingtekGripper   # 知行夹爪


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
def load_policy(model_path: str, device: torch.device, enable_ensemble: bool = True):
    """自动检测策略类型并加载，返回 (policy, policy_type)

    enable_ensemble=False 时 (消融: 关闭 temporal ensemble), ACT 不启用时序集成,
    改回“动作队列”模式 (每 n_action_steps 触发一次前向)。
    """
    model_path = Path(model_path)
    config_path = model_path / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    policy_type = config.get("type", "unknown")
    print(f"检测到策略类型: {policy_type}")

    if policy_type == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy
        policy = ACTPolicy.from_pretrained(str(model_path))
        # ACT 特有: Temporal Ensemble 平滑
        if hasattr(policy.config, 'temporal_ensemble_coeff'):
            if enable_ensemble:
                policy.config.temporal_ensemble_coeff = 0.01
                from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
                policy.temporal_ensembler = ACTTemporalEnsembler(
                    temporal_ensemble_coeff=0.01,
                    chunk_size=policy.config.chunk_size
                )
                print(f"  Temporal Ensemble: Enabled (coeff=0.01)")
            else:
                # 消融: 关闭集成 → 走动作队列
                policy.config.temporal_ensemble_coeff = None
                policy.temporal_ensembler = None
                policy.reset()
                print(f"  Temporal Ensemble: DISABLED (消融 --no-ensemble, 改用动作队列)")

    elif policy_type == "vqbet":
        from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
        policy = VQBeTPolicy.from_pretrained(str(model_path))

    elif policy_type == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        policy = DiffusionPolicy.from_pretrained(str(model_path))

    elif policy_type == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        _patch_config_compat(config_path)
        policy = SmolVLAPolicy.from_pretrained(str(model_path))

    elif policy_type == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        policy = PI0Policy.from_pretrained(str(model_path))

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

    def set_qpos(self, qpos, use_smoothing=True):
        """设置目标关节角 + 夹爪

        返回诊断 dict (供 CSV 日志):
          cmd_joint : 实际下发(或死区保留)的 6 关节指令
          sent      : 本步是否真正调用了 rm_movej (False=死区过滤掉)
          target    : EMA 前的原始目标关节角
        """
        with self.lock:
            joint_target = qpos[:6].copy()
            raw_target = joint_target.copy()

            # EMA 平滑
            if use_smoothing:
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
                self.arm.rm_movej(joint_target.tolist(), 50, 0, 0, 0)
                self._last_joint_cmd = joint_target.copy()

            # 夹爪（知行 RTU 内部异步下发）。与 collect_data 一样直接使用连续的
            # 归一化目标：0=闭合、1=张开；死区仅用于避免重复写入 RS-485 总线。
            gripper_target = float(np.clip(qpos[6], 0.0, 1.0))
            if (self._last_gripper_cmd is None or
                    abs(gripper_target - self._last_gripper_cmd) >= GRIPPER_DEADBAND):
                self.gripper.move_normalized(gripper_target)
                self._last_gripper_cmd = gripper_target

            return {
                'cmd_joint': joint_target.copy(),
                'sent': bool(should_send),
                'target': raw_target,
                'gripper_cmd': gripper_target,
            }

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


# ============ Episode CSV 日志器 ============
def _fmt_vec(v):
    """向量序列化为 '|' 分隔字符串 (避开 CSV 逗号冲突), 供分析脚本反解"""
    if v is None:
        return ""
    return "|".join(f"{float(x):.5f}" for x in np.asarray(v).ravel())


class EpisodeLogger:
    """将故障 episode 逐步写入 CSV。

    字段 (与需求一致):
      timestamp        本步写入时刻 (time.time, 秒)
      chunk_id         当前 chunk 序号 (从 1 开始; 队列空时触发新 chunk)
      step_in_chunk    在本 chunk 内的步序 (0-based)
      queue_len        select_action 后动作队列剩余长度
      obs_timestamp    观测采集时刻 (time.time, 秒)
      action_raw       策略原始输出 (归一化空间, 未 unnormalize)
      action_processed 后处理后动作 (绝对关节角[度] + 夹爪0~1)
      cmd_joint        实际下发给 rm_movej 的 6 关节指令 (EMA+死区后)
      actual_joint     观测时刻机械臂实测 6 关节角 (即上一步 cmd 的结果)
      latency          obs_timestamp → cmd 下发 的时延 (秒)
    额外辅助列: step, sent, inference_ms, gripper_cmd
    注: actual_joint 在每步开头采样, 因此第 N 行的 actual_joint 对应第 N-1 行 cmd_joint 的执行结果。
    """

    FIELDS = ["step", "timestamp", "chunk_id", "step_in_chunk", "queue_len",
              "obs_timestamp", "latency", "inference_ms", "sent", "gripper_cmd",
              "action_raw", "action_processed", "cmd_joint", "actual_joint"]

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.f, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.f.flush()
        self.path = path
        self.n = 0

    def log(self, **kw):
        row = {k: kw.get(k, "") for k in self.FIELDS}
        for k in ("action_raw", "action_processed", "cmd_joint", "actual_joint"):
            row[k] = _fmt_vec(kw.get(k))
        self.writer.writerow(row)
        self.n += 1
        if self.n % 20 == 0:
            self.f.flush()

    def close(self):
        try:
            self.f.flush()
            self.f.close()
        except Exception:
            pass


def _get_action_queue(policy):
    """取策略内部动作队列 (SmolVLA: _queues['action']; ACT: _action_queue)。无队列返回 None"""
    q = getattr(policy, "_queues", None)
    if isinstance(q, dict) and "action" in q:
        return q["action"]
    return getattr(policy, "_action_queue", None)


def wait_for_next_chunk(chunk_number):
    """在生成并执行一个新动作 chunk 前等待人工确认。"""
    input(f"\n[Chunk {chunk_number}] 按 Enter 生成并执行下一个 chunk (Ctrl+C 终止)...")


def apply_ablations(policy, policy_type, args):
    """根据命令行消融开关修改策略运行时行为。"""
    notes = []
    # 消融 2: 只执行每个 chunk 的第 1 步 → n_action_steps=1, 每步重规划
    if args.first_step_only:
        try:
            policy.config.n_action_steps = 1
            policy.reset()  # 重建 deque(maxlen=1)
            notes.append("--first-step-only: n_action_steps=1 (每步重规划)")
        except Exception as e:
            notes.append(f"--first-step-only 失败: {e}")
    # 消融 1: 关闭 ensemble (已在 load_policy 处理 ACT; SmolVLA 本无 ensemble)
    if args.no_ensemble and policy_type != "act":
        notes.append(f"--no-ensemble: {policy_type} 无 temporal ensemble, 忽略")
    # 消融 3: 新 chunk 清空旧队列 (运行时在主循环拦截)
    if args.clear_queue_on_new_chunk:
        notes.append("--clear-queue-on-new-chunk: 新 chunk 生成前显式清空队列")
    for n in notes:
        print(f"  [消融] {n}")
    return notes


# ============ 主推理循环 ============
def main():
    parser = argparse.ArgumentParser(description='RealMan 通用策略推理')
    parser.add_argument('--model', type=str, required=True,
                        help='模型路径 (如 outputs/act/checkpoints/100000/pretrained_model)')
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
    parser.add_argument('--freq', type=float, default=15.0,
                        help='控制频率(Hz)，应与训练数据fps一致')
    parser.add_argument('--task', type=str, default='pick up the cube',
                        help='VLA 任务描述 (Pi0/SmolVLA 需要)')
    parser.add_argument('--headless', action='store_true', help='无GUI模式')
    parser.add_argument('--offline', action='store_true', help='离线模式 (不从 HuggingFace 下载)')
    # ------ 故障排查: CSV 日志 + 消融开关 ------
    parser.add_argument('--log-csv', type=str, default=None,
                        help='将本 episode 逐步写入该 CSV (含 chunk_id/queue_len/action_raw/cmd/actual/latency)')
    parser.add_argument('--no-ensemble', action='store_true',
                        help='消融1: 关闭 temporal ensemble (仅 ACT 有效; SmolVLA 本无)')
    parser.add_argument('--first-step-only', action='store_true',
                        help='消融2: 只执行每个 chunk 的第1步 (n_action_steps=1, 每步重规划)')
    parser.add_argument('--clear-queue-on-new-chunk', action='store_true',
                        help='消融3: 新 chunk 生成前显式清空旧队列')
    parser.add_argument('--no-ema', action='store_true',
                        help='关闭 EMA 平滑 (直接下发策略目标关节角)')
    parser.add_argument('--ema-alpha', type=float, default=0.3,
                        help='EMA 系数 (默认 %(default)s; 1.0=不平滑)')
    parser.add_argument('--deadzone', type=float, default=0.5,
                        help='关节死区阈值(度), 默认 %(default)s; 0=不过滤')
    args = parser.parse_args()

    if args.offline:
        import os
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

    # 导入机械臂SDK
    from Robotic_Arm.rm_robot_interface import (
        RoboticArm, rm_thread_mode_e
    )
    from lerobot.processor.pipeline import DataProcessorPipeline

    print("=" * 50)
    print("  RealMan RM65 通用策略推理")
    print("=" * 50)

    # 1. 加载模型
    print("\n[1/5] Loading policy...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, policy_type = load_policy(args.model, device, enable_ensemble=not args.no_ensemble)
    print(f"      Device: {device}")
    apply_ablations(policy, policy_type, args)

    preprocessor = DataProcessorPipeline.from_pretrained(
        args.model, config_filename='policy_preprocessor.json')
    postprocessor = DataProcessorPipeline.from_pretrained(
        args.model, config_filename='policy_postprocessor.json')

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
        ema_alpha=(1.0 if args.no_ema else args.ema_alpha),
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
        print("      3秒后开始...")
        time.sleep(3)

    # 5. 推理循环
    print(f"\n[5/5] Executing {policy_type} at {args.freq}Hz (Ctrl+C to stop)...")
    step_count = 0
    control_period = 1.0 / args.freq
    policy.reset()

    # 故障排查: CSV 日志 + chunk 边界追踪
    logger = EpisodeLogger(args.log_csv) if args.log_csv else None
    if logger:
        print(f"      CSV 日志: {args.log_csv}")
    use_smoothing = not args.no_ema
    action_queue = _get_action_queue(policy)   # 须在 reset() 之后取, 保证与 select_action 同一 deque
    chunk_id = 0
    step_in_chunk = -1

    try:
        while True:
            start_time = time.time()

            # chunk 边界检测: 队列空 → 本步将触发一次前向重规划(新 chunk)
            if action_queue is not None:
                is_new_chunk = (len(action_queue) == 0)
                if is_new_chunk and args.clear_queue_on_new_chunk:
                    action_queue.clear()   # 消融3: 显式清空旧队列(对 refill-on-empty 策略为幂等确认)
            else:
                is_new_chunk = True   # ACT+ensemble: 无队列, 每步都是新前向

            # 在采集观测前等待，以确保确认后使用的是最新画面和机械臂状态。
            if is_new_chunk:
                wait_for_next_chunk(chunk_id + 1)

            # 获取观测
            qpos = robot.get_qpos()
            obs_timestamp = time.time()
            actual_joint = qpos[:6].copy()   # 观测时刻实测关节(=上一步 cmd 的执行结果)
            observation = {
                'observation.state': torch.from_numpy(qpos).float(),
            }

            if cam_top:
                img = cv2.cvtColor(cam_top.get_frame(), cv2.COLOR_BGR2RGB)
                observation['observation.images.camera_global'] = \
                    torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

            if cam_wrist:
                img = cv2.cvtColor(cam_wrist.get_frame(), cv2.COLOR_BGR2RGB)
                observation['observation.images.camera_left'] = \
                    torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

            # VLA 需要 language instruction
            if policy_type in ("pi0", "pi05", "smolvla"):
                observation['task'] = args.task

            # 预处理 → 推理 → 后处理
            observation = preprocessor(observation)

            with torch.no_grad():
                action_tensor = policy.select_action(observation)
            infer_done = time.time()
            action_raw = action_tensor.reshape(-1).detach().cpu().numpy().copy()

            action_dict = postprocessor({'action': action_tensor})
            action = action_dict['action'][0].cpu().numpy()

            queue_len = len(action_queue) if action_queue is not None else 0
            if is_new_chunk:
                chunk_id += 1
                step_in_chunk = 0
            else:
                step_in_chunk += 1

            # 执行
            diag = robot.set_qpos(action, use_smoothing=use_smoothing)
            cmd_time = time.time()

            # CSV 日志
            if logger:
                logger.log(
                    step=step_count,
                    timestamp=cmd_time,
                    chunk_id=chunk_id,
                    step_in_chunk=step_in_chunk,
                    queue_len=queue_len,
                    obs_timestamp=obs_timestamp,
                    latency=cmd_time - obs_timestamp,
                    inference_ms=(infer_done - obs_timestamp) * 1000.0,
                    sent=int(diag['sent']),
                    gripper_cmd=diag['gripper_cmd'],
                    action_raw=action_raw,
                    action_processed=action,
                    cmd_joint=diag['cmd_joint'],
                    actual_joint=actual_joint,
                )

            # 日志
            elapsed = time.time() - start_time
            if step_count % 1 == 0:
                actual_freq = 1.0 / elapsed if elapsed > 0 else 0
                joints_str = " ".join(
                    f"J{i + 1}:{qpos[i]:6.1f}→{action[i]:6.1f}" for i in range(6)
                )
                print(f"[{policy_type}] Step {step_count:4d} | "
                      f"{joints_str} | "
                      f"夹爪: {qpos[6]:5.2f}→{np.clip(action[6], 0.0, 1.0):5.2f} | {actual_freq:.1f}Hz")

            # 可视化
            if not args.headless:
                try:
                    frames = []
                    if cam_top:
                        frames.append(cv2.resize(cam_top.get_frame(), (320, 240)))
                    if cam_wrist:
                        frames.append(cv2.resize(cam_wrist.get_frame(), (320, 240)))
                    if frames:
                        display = np.hstack(frames) if len(frames) > 1 else frames[0]
                        cv2.putText(display, f"{policy_type} Step: {step_count}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow("Inference", display)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                except cv2.error:
                    args.headless = True

            step_count += 1

            # 频率控制
            remaining = control_period - (time.time() - start_time)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\n\n推理终止")
    finally:
        # 先 flush CSV, 保证故障 episode 数据不因退出丢失
        try:
            if logger is not None:
                logger.close()
                print(f"CSV 已写入: {logger.path} ({logger.n} 行)")
        except Exception:
            pass
        # 退出前让机械臂回到 POS_INIT 并张开夹爪 (归位内部已含急停保护)
        robot.homing_on_exit(INIT_POSE)
        try:
            if not args.headless:
                cv2.destroyAllWindows()
        except Exception:
            pass
        robot.close()
        if cam_top:
            cam_top.close()
        if cam_wrist:
            cam_wrist.close()
        print("硬件已关闭")


if __name__ == "__main__":
    main()
