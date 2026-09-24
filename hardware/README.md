

## 机械臂 — 睿尔曼 RM65-B

### 连接配置
- **通信协议**: TCP/IP
- **默认IP**: `192.168.5.123`（本项目当前机械臂；出厂默认 `192.168.2.18`，可通过示教器修改）
- **默认端口**: `8080`
- **SDK**: [RM_API2 Python](https://www.realman-robotics.cn/)

### 初始化流程
```python
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = arm.rm_create_robot_arm("192.168.5.123", 8080)

# 基础设置
arm.rm_set_arm_run_mode(1)     # 设置运行模式
# 夹爪已改为知行 RTU 独立串口，无需工具端供电 / Modbus 透传设置
```

### 运动控制
| API | 说明 | 阻塞 |
|-----|------|------|
| `rm_movej(joints, speed, 0, 0, 1)` | 关节空间运动 | 阻塞 |
| `rm_movej(joints, speed, 0, 0, 0)` | 关节空间运动 | 非阻塞 |
| `rm_movej_p(pose, speed, 0, 0, 1)` | 关节空间规划到笛卡尔位姿（采集脚本归位/复位，大位移最稳、不撞奇异点） | 阻塞 |
| `rm_movep_canfd(pose, False, 0, 60)` | 笛卡尔空间CANFD | 非阻塞 |
| `rm_get_current_arm_state()` | 获取当前状态 | - |

### 只读自检（不下发任何运动）
```bash
python hardware/realman_arm.py                       # 分层: IP冲突 / TCP / SDK状态 / 机型固件
python hardware/realman_arm.py --scan --all          # 同网段多台 RM65 时列候选并逐台读状态
python hardware/realman_arm.py --read-init           # 读当前笛卡尔位姿→打印 ROBOT_INIT 粘贴行
python hardware/realman_arm.py --read-init --write   # 同上并直接写入 collect_data.py 的 ROBOT_INIT_POS/ORI
```
> 同网段可能有多台同型号 RM65（SN 读不出、型号固件相同），靠**关节角与物理停放姿态肉眼比对**锁定本项目这台（当前 `.123`）。
> 若报 “IP 冲突: 目标 IP 是本机网卡地址”，说明本机网卡 IP 与臂 IP 撞车（ping 通是 ping 自己），用 `--scan` 重新定位。
> `--read-init`：标定遥操起始位姿——手动把臂拖到起始位后运行读取当前笛卡尔位姿；加 `--write` 直接写入 `collect_data.py` 的 `ROBOT_INIT_POS/ORI`（只改文本、不下发运动）。完整流程见 [数据采集指南 3.2](../docs/data_collection.md)。

---

## 夹爪 — 知行 RTU 平动手 (Modbus RTU / 独立串口)

夹爪通过**独立 RS-485 串口**直接控制，与机械臂**完全解耦**——不再经机械臂 Modbus 透传，
也无需工具端供电设置。驱动见 `vendor/changingtek_rtu_sdk`，封装见 `hardware/changingtek_gripper.py`。

### 连接配置
| 参数 | 值 |
|------|---|
| 串口 | `/dev/realman/gripper_left` (udev 软链 → ttyUSB2) |
| 从站地址 slave_id | `2` |
| 波特率 | 115200 |
| 行程 | `min_position`(小端、mm≈0、**物理张开**) ~ `max_position`(大端、mm≈86、**物理闭合**)，设备单位 /100 = mm |

### 归一化约定
第 7 维（qpos/action 的最后一维）统一归一化到 `0~1`（**对外语义**）：
```
v = 1.0 -> 张开
v = 0.0 -> 闭合
推理端沿用 “>0.5 判为开” 的阈值
```
⚠ 本机知行平动手的**物理行程方向与归一化约定相反**（小端=张开、大端=闭合），
封装默认 `invert=True` 已在归一化↔物理位置间反转映射：`v=0` 驱动到大端(物理闭合)、`v=1` 到小端(物理张开)。
若实测张开/闭合到位值不同，改 `ChangingtekGripper(min_position=..., max_position=...)`；若换用方向一致的夹爪，置 `invert=False`。

### 接口
```python
from changingtek_gripper import ChangingtekGripper
g = ChangingtekGripper(port="/dev/realman/gripper_left", slave_id=2)
g.connect()                 # 连接 + 启动后台轮询线程 + 使能
g.open(); g.close()         # 张开 / 闭合
g.move_pct(60)              # 按百分比 (0~100)
print(g.get_position_normalized())  # 读归一化位置 (后台线程已缓存)
g.disable(); g.disconnect()
```

### 性能说明
夹爪走独立串口，`request_move` 由所属总线的后台线程串行异步下发，**不阻塞主控制循环**；
反馈按 `poll_hz`(默认 25Hz) 缓存，读取零阻塞。自检：`python hardware/changingtek_gripper.py`。

---

## 相机 — 顶部 D435 (RealSense) + 腕部 Gemini 305 (Orbbec)

### 双相机配置
| 位置 | 型号 | 驱动 | 序列号 | 用途 |
|------|------|------|--------|------|
| 顶部 | Intel RealSense D435 | `pyrealsense2` | `262322074840` | 全局视角 |
| 腕部 | 奥比中光 Gemini 305 | `pyorbbecsdk` | `CV2T66100096` | 精细视角 |

两个相机类接口一致（`get_frame()` 返回 BGR `(H,W,3)`、`is_active`、`close()`），采集/推理脚本可无缝替换。

### 推荐参数
| 参数 | 值 | 说明 |
|------|---|------|
| 分辨率 | 640×480 | 平衡质量与速度 |
| 帧率 | 30fps | 与采集频率对齐 |
| 自动曝光 | **关闭** (D435) | 自动曝光会导致训练/推理图像不一致 |

### ⚠️ Orbbec 库冲突（必读）
`pyorbbecsdk` 的 `.so` 可能把 `libOrbbecSDK.so.2` 解析到 ROS 的旧库
(`/opt/ros/humble/lib`)，导致 `undefined symbol: ob_application_config_set_struct`。
运行任何用到腕部相机的脚本前，让自带 2.9.3 库优先：
```bash
export LD_LIBRARY_PATH=$(python -c "import pyorbbecsdk,os;print(os.path.dirname(pyorbbecsdk.__file__))"):$LD_LIBRARY_PATH
python -c "import pyorbbecsdk; print('ok')"   # 验证
```
> 运行采集/推理的终端不要先 source ROS humble，或确保上面的路径排在 `LD_LIBRARY_PATH` 最前。
> 腕部相机自检：`python hardware/orbbec_camera.py --serial CV2T66100096`。

### RealSense 自检

顶部 D435 独立自检（与夹爪、腕部相机自检命令同风格）：
```bash
python hardware/realsense_camera.py --serial 262322074840
```

自检流程：
1. **硬件复位**：`RealSenseCamera.__init__` 会先遍历 `rs.context().query_devices()` 找到目标序列号，
   调用 `dev.hardware_reset()` 并等待 2s，释放上次未正常退出的设备锁
2. **建流**：640×480@30fps，`rs.format.bgr8`（省去 RGB→BGR 转换）
3. **固定曝光**：关闭 `enable_auto_exposure`，`exposure=150`，避免亮度波动污染训练
4. **抓帧验证**：等待 1s 后取一帧，检查 `shape=(480,640,3) dtype=uint8 nonzero=True`
5. **FPS 压测**：连续 30 帧，期望更新数 ≥29、实际 FPS ≈29.7
6. **落盘**：测试帧保存到 `outputs/realsense_test_frame.png` 供肉眼确认（不使用 Qt/GUI）

### RealSense USB 断联 / 设备占用排查

**症状 A**：`RuntimeError: Device is already in use` / `failed to set power state`
通常是上次脚本 Ctrl-C 或崩溃，RealSense 内核态句柄未释放。

```bash
# 1. 找到并杀掉占用进程
pkill -f realsense
pkill -f collect_data
pkill -f inference

# 2. 若仍失败, 触发一次 hardware_reset (自检脚本已内置, 也可手动):
python -c "import pyrealsense2 as rs, time; \
    [d.hardware_reset() for d in rs.context().query_devices() \
     if d.get_info(rs.camera_info.serial_number)=='262322074840']; time.sleep(2)"

# 3. 实在不行就物理拔插 USB
```

**症状 B**：`No device connected` / `lsusb` 看不到 `8086:0b07`
- 检查 USB 线是否是**数据线**（很多 Type-C 线只供电不通数据）
- 换到主板后置 **USB 3.0** 端口（D435 需要 USB 3 带宽；USB 2.0 下 640×480@30fps 也会掉帧）
- `dmesg -w` 观察插拔时是否有枚举日志

**症状 C**：两个相机同时启动时报带宽不足 / 帧率骤降
- 顶部 D435 和腕部 Gemini 305 **不要接同一个 USB 控制器**
- `lsusb -t` 查看 USB 树，把两台相机分到不同 root hub

**硬件复位说明**：`hardware_reset()` 会让设备重新枚举一次 USB（约 2s），
等价于软拔插，无需 physically 动手。自检脚本每次启动都会主动复位，
因此**连续两次运行自检之间要留 ≥3s 间隔**，否则第二次可能在设备尚未 ready 时失败。

---

## 遥操作 — Vive Tracker (OpenVR)

> 本项目使用 **Pika Sense viva tracker**，它与 HTC Vive Tracker 同为 SteamVR/OpenVR 的
> `GenericTracker`，`hardware/vive_tracker.py` 按设备类自动发现，**无需改代码**；
> 实物安装不同时仅需重标定零点常量 `ROBOT_INIT_POS/ORI` 与坐标映射符号。

### 前置要求
1. 安装 [SteamVR](https://store.steampowered.com/app/250820/SteamVR/)
2. 安装 OpenVR Python: `pip install openvr`
3. **基站（Lighthouse Base Station）通电**、Vive Tracker 开机并配对（Tracker 靠基站定位，无基站拿不到 6DOF 位姿）
4. **只有 Tracker、没有头显**时须先配置 null 假头显（见下方「无头显运行」），否则 `openvr.init` 报 `Init_HmdNotFound`

### SteamVR 启停（推荐用一键脚本）
> ⚠️ 旧命令 `~/.steam/debian-installation/ubuntu12_64/steam-runtime/run.sh ... SteamVR/bin/vrserver` 已失效：本机无 `debian-installation` 布局、vrserver 实际在 `bin/linux64/`；且**裸跑 vrserver 会因无 vrmonitor 在约 20 秒后自动退出**（日志 `Monitor: 0`）。必须启动完整 SteamVR 栈。

`hardware/steamvr_restart.sh` 已把「清理幽灵树 → 常驻启动完整栈 → 轮询等待就绪」固化成一键流程（需在**宿主机终端**执行，沙箱内看不到宿主机进程）：

```bash
# 清理残留 + 常驻启动完整栈 (vrserver + vrmonitor)，并等待两者同时就绪后打印确认
bash hardware/steamvr_restart.sh

# 只清理幽灵进程树，不启动 (上次退出残留导致 game already running 时用)
bash hardware/steamvr_restart.sh --kill

# 停止
pkill -f vrserver
```

脚本会杀掉整棵 `AppId=250820 / reaper / steam-launch-wrapper / vrstartup / steamvr_room_setup / vrserver / vrmonitor` 幽灵树（避免 `game already running`），再 `steam steam://rungameid/250820`，最后轮询到 `pgrep -x vrserver` 与 `pgrep -x vrmonitor` 同时就绪才返回；若检测到房间设置弹窗会提示跳过。

### 无头显运行（只有 Tracker、没有头显时必做）
没有 HMD 时 SteamVR 默认拒绝初始化。启用自带的 **null 假头显驱动**即可：编辑
`~/.local/share/Steam/steamapps/common/SteamVR/resources/settings/default.vrsettings`，
在 `"steamvr"` 段设置/新增以下三项：
```json
"requireHmd": false,
"forcedDriver": "null",
"activateMultipleDrivers": true
```
并在**文件顶层**新增（null 驱动默认禁用，不加这段 `forcedDriver` 会被日志 `Ignoring ... driver is disabled` 忽略）：
```json
"driver_null": { "enable": true }
```
改完重启 SteamVR 生效。之后 `python hardware/vive_tracker.py` 应能枚举出 `hmd: Null`、`tracker`、`tracking_reference`(基站) 并实时输出位姿。

> ⚠️ **SteamVR 更新会覆盖 `default.vrsettings`**，更新后需重配（改前先备份该文件）。

### 运行自检与正确启动顺序

⚠️ **务必先启动完整 SteamVR 栈，再跑 `vive_tracker.py`**。若直接跑脚本，`openvr.init()` 会自己临时拉起一个 vrserver，而它因无 vrmonitor 会在脚本退出后约 17 秒自杀（vrserver.txt: `Shutting down server ... Monitor: 0`）；下次再跑时 vrserver 冷启动握手需约 15 秒，基站/tracker 追踪尚未就绪，脚本便判定 `No tracker found!` 退出——形成“第一次找得到、退出后就找不到”的恶性循环。

推荐直接用一键脚本（等价于下面 1~3 步，且自带幽灵树清理与就绪轮询）：

```bash
bash hardware/steamvr_restart.sh   # 清理 + 常驻启动 + 等到 vrserver/vrmonitor 都就绪
python hardware/vive_tracker.py    # 再跑自检 / collect_data.py
```

手动流程（脚本不可用时）：

```bash
# 1. 常驻启动完整栈（vrmonitor 撑着 vrserver，不会自动退出）
steam steam://rungameid/250820

# 2. 等 10~20 秒，确认两个进程都在
pgrep -x vrserver && pgrep -x vrmonitor

# 3. 再跑自检；脚本退出（已内置 openvr.shutdown）不影响 SteamVR，可反复运行
python hardware/vive_tracker.py
```

> `vive_tracker.py` 已内置：未发现 tracker 时自动等待重试（默认 8 次 × 2s）、退出时 `openvr.shutdown()` 干净释放、位姿刷新 `flush=True`。但这些只在 **vrserver 存活**时有效，替代不了上面的常驻启动。
> 若 `steam steam://rungameid/250820` 报 `game already running`：上次退出残留了 reaper 幽灵进程，直接 `bash hardware/steamvr_restart.sh`（会先清树再启动）；或手动 `pgrep -af 'AppId=250820|vrstartup|steamvr_room_setup'` 找出后 `kill -9` 再启动。

### 坐标映射
Vive Tracker 坐标系与机械臂坐标系不一致，需要映射：
```
Vive → Robot:
  Robot_X = -Vive_Z × scale
  Robot_Y = -Vive_X × scale
  Robot_Z = +Vive_Y × scale
```

### 校准流程
0. **首次/换硬件**：手动把臂拖到遥操起始位 → `python hardware/realman_arm.py --read-init --write` 写入 `ROBOT_INIT_POS/ORI`（tracker 零点对应的机械臂位姿）
1. 手持 Tracker 到顺手的起始姿态（机械臂停在你希望开始的位置即可）
2. 按 `w` 启用遥控 —— **以当前 Tracker 位为零点、机械臂当前位姿为基点**，原地 engage 不突跳（无需单独校准键）
3. 移动 Tracker 控制机械臂；再按 `w` 暂停
