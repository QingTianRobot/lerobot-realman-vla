# 数据采集指南

本文档讲**如何用 Vive 遥操作（或示教模式）采集模仿学习训练数据**，覆盖：环境准备 → 硬件自检 → 首次标定 → 采集操作 → 数据格式与检查 → 排错。

> 采集只是全流程第一步，转换/训练/推理见 [完整流程指南](pipeline_guide.md)。

相关代码：[`scripts/collect_data.py`](../scripts/collect_data.py)、硬件封装 [`hardware/`](../hardware/)。

---

## 0. 硬件与默认参数

| 组件 | 型号 | 默认参数（已写进脚本，可命令行覆盖） |
|------|------|------|
| 机械臂 | 睿尔曼 RM65 | IP `192.168.5.123`，端口 `8080` |
| 夹爪 | 知行 RTU 平动手 | 串口 `/dev/realman/gripper_left`，slave_id `2`，波特率 `115200` |
| 顶部相机 | Intel RealSense D435 | 序列号 `262322074840` |
| 腕部相机 | 奥比中光 Gemini 305 | 序列号 `CV2T66100096` |
| 遥操作 | Vive Tracker (Pika Sense viva) | 序列号可选，单 tracker 自动选第一个 |

**夹爪归一化约定**（采集/推理必须一致）：第 7 维 `0~1`，**`1=张开`、`0=闭合`**；推理端 `>0.5 判为开`。

---

## 1. 环境准备

每次开新终端，先激活环境（会自动修好腕部 Orbbec 相机的库路径）：

```bash
cd /home/robot/repo/lerobot-realman-vla
source env.sh
```

> 没装环境先跑 `bash setup.sh`（uv 建 `.venv` + 装依赖）。详见 [README 快速开始](../README.md#-快速开始)。

确认机械臂网络可达：

```bash
ping 192.168.5.123
```

> 同网段另有 `.124` / `.125` 两台同型号 RM65，别连错台架那台。确认方法：用 3.2 的片段读当前关节角，与实物姿态对比。

---

## 2. 硬件自检（强烈建议采集前逐个验证）

### 2.1 夹爪

```bash
# 开合循环自检，打印实际位置(mm)与归一化值
python hardware/changingtek_gripper.py --slave-id 2
```
观察：**能连上、`open()` 是张开、`close()` 是闭合、归一化 0↔1 对应合↔开**。

### 2.2 腕部相机（Orbbec 305）

```bash
python hardware/orbbec_camera.py --serial CV2T66100096
```
会等待首个有效帧（默认最长 10s），打印收帧统计并保存一张 `orbbec_<serial>_<时间戳>.png`。
- 可选：`--save my.png` 指定保存路径，`--warmup 15` 延长等待。
- 若报 `undefined symbol: ob_application_config_set_struct` → 没 `source env.sh`（库路径没修）。

### 2.3 顶部相机（D435）

```bash
python hardware/realsense_camera.py --serial 262322074840
```
做 30 帧 FPS 测试并保存 `outputs/realsense_test_frame.png`。
- 若初始化失败：`lsusb | grep Intel` 确认枚举；`pkill -f realsense` 杀残留进程；物理拔插 USB。

### 2.4 Vive Tracker

启动 **SteamVR**（`steam steam://rungameid/250820`），确认基站通电、tracker 在线。可单独自检：

```bash
python hardware/vive_tracker.py   # 打印发现的设备(hmd/tracker/基站)与实时位姿
```

采集脚本启动时会打印 `Vive: OK/FAIL`，`FAIL` 说明 SteamVR 没开或 tracker 没被追踪到。

> **无头显**（只有 Tracker）时，SteamVR 默认要求 HMD，`vive_tracker.py` 会报 `Init_HmdNotFound`；需配置 null 假头显驱动后才能无头显追踪，详见 [硬件配置 README 的「无头显运行」](../hardware/README.md)。
>
> **先启动 SteamVR 再跑脚本**：若直接跑 `vive_tracker.py`，`openvr.init` 会自己临时拉起 vrserver，脚本一退它就因 `Monitor: 0` 自杀，下次再跑冷启动来不及就绪而报 `No tracker found`（“第一次找得到、退出后找不到”）。正确顺序见 README「运行自检与正确启动顺序」。

---

## 3. 首次标定（换硬件/安装变动时才需重做）

### 3.1 夹爪行程标定

实测机械开/合限位，避免命令顶到硬限位长期堵转发热；结果写入 `hardware/gripper_calibration.json`，**采集与推理会自动加载**，保证三端行程一致。

```bash
python hardware/changingtek_gripper.py --slave-id 2 --calibrate --margin 0.02
```
- 会驱动到开/合限位实测真实行程，上限回缩 `margin`（默认 2%）后持久化。
- 标定前**确保夹爪周围无障碍**。
- 只需标定一次；之后自检/采集自动加载（启动会打印 `[i] 已加载夹爪标定...`）。
- 想临时忽略标定文件：加 `--no-calibration-file`。

> ⚠️ **标定纪律**：一旦开始采数据，**采集→训练→推理期间不要再 `--calibrate`**。行程基准变了会与已采数据的归一化不一致，导致模型学到矛盾映射。只有换夹爪/机械限位变化才重标，且**重标后旧数据作废、需重采**。

### 3.2 Vive 零点与坐标映射

遥操把 tracker 的位姿增量映射到机械臂笛卡尔空间，需标定两处（**都在 [`collect_data.py`](../scripts/collect_data.py) 配置区/`_control_loop`，已加详细注释**）：

**(a) `ROBOT_INIT_POS / ORI`** — tracker 在零点时机械臂应处的笛卡尔位姿（位置:米，姿态:弧度）。手动把机械臂拖到安全顺手的起始位，用只读工具读当前位姿：

```bash
# 只打印可直接粘贴的两行（默认，不改文件）
python hardware/realman_arm.py --read-init
# 读取后直接写入 collect_data.py 的 ROBOT_INIT_POS/ORI（一步到位）
python hardware/realman_arm.py --read-init --write
```
`pose` 前 3 个=位置(米)、后 3 个=姿态(弧度)；`--write` 只替换 [`collect_data.py`](../scripts/collect_data.py) 那两行的 `np.array([...])`，定位不到就放弃、不破坏文件。写入结果形如：

```python
ROBOT_INIT_POS = np.array([-0.4024, -0.0273, 0.1878])
ROBOT_INIT_ORI = np.array([3.152, 0.149, -0.137])
```
> ⚠️ 不改这里，按 `w` 时机械臂会突跳到上一套实物的旧位姿。

**(b) 坐标映射符号** — 在 `_control_loop` 里（`Robot_X←-Vive_Z`、`Robot_Y←-Vive_X`、`Robot_Z←+Vive_Y`）。按 `w` 后手推 tracker 观察：
- 某轴方向反了 → 翻转那一行符号；
- 推 A 方向却动了 B 轴 → 换那一行的 `delta_pos` 下标；
- 幅度太大/太小 → 调 `scale_pos` / `scale_ori`。

---

## 4. 启动采集

> ⚠️ **采集前务必先常驻启动 SteamVR**：`steam steam://rungameid/250820`，再用 `pgrep -x vrserver && pgrep -x vrmonitor` 确认两个进程都在。采集是长任务，若靠脚本自己临时拉起 vrserver，它无 vrmonitor 撑着会在异常/退出时 `Monitor: 0` 自杀，`_control_loop` 随即读不到位姿（`pose is None` 静默 `continue`）——**机械臂中途不跟随了却仍在录制，采到废数据**。详见 [硬件配置 README「运行自检与正确启动顺序」](../hardware/README.md)。

默认参数已填好你的硬件，最简形式：

```bash
python scripts/collect_data.py --task-name pick_cube --fps 30
```

需要时显式覆盖：

```bash
python scripts/collect_data.py \
    --arm-ip 192.168.5.123 \
    --gripper-port /dev/realman/gripper_left --gripper-slave-id 2 \
    --cam-top 262322074840 --cam-wrist CV2T66100096 \
    --tracker-serial LHR-B909D55F \
    --save-dir data/raw_hdf5 --task-name pick_cube --fps 30
```

启动后逐项确认状态行：

```
相机: top=OK, wrist=OK
机械臂: OK (192.168.5.123:8080)
夹爪: OK (/dev/realman/gripper_left)
Vive: OK
>            # 交互提示符（输入命令后按回车）
```

---

## 5. 交互命令

> 输入字母/命令后**按回车**执行。

| 命令 | 功能 | 备注 |
|------|------|------|
| `v` | 校准 Vive 零点 | 保持 tracker 静止，采 30 帧平均 |
| `w` | 启用遥控 | 机械臂开始跟随 tracker（**会动，注意安全**） |
| `e` | 暂停遥控 | 机械臂停在原地 |
| `s` | 开始录制一条 | 自动命名 `pick_cube_0.hdf5`、`_1`… |
| `d` | 停止并保存 | 打印实际帧率与帧数 |
| `g <0-100>` | 夹爪开度百分比 | **`g 0`=闭合，`g 100`=张开** |
| `c` | 夹爪闭合 | |
| `o` | 夹爪张开 | |
| `q` | 退出 | 自动松夹爪、断连、关相机 |

> 机械臂位姿由 **Vive tracker** 控制，夹爪由 **键盘 `g/c/o`** 控制（tracker 不管夹爪）。

---

## 6. 单条 episode 标准流程

1. 摆好场景（物体、容器位置；每条略变化以覆盖更多状态）
2. 手把 tracker 放到**遥操起始姿态**（对应 `ROBOT_INIT_POS`）→ 按 `v` 校准零点
3. 按 `w` 启用 → 移动 tracker，机械臂跟随；**先空移确认方向不反**
4. 把机械臂移到任务起点 → 按 `s` 开始录制
5. 用 tracker 操控机械臂完成任务，在合适时机按 `c`（抓取）/ `o`（释放）
6. 完成 → 按 `d` 保存
7. 按 `e` 暂停遥控 → 复位场景 → 回到第 2 步采下一条
8. 采够条数 → 按 `q` 退出

> 录制期间必须保持 `w` 启用，否则机械臂不动、录到的是静止轨迹。

### 示教模式（不用 Vive）

手动拖动机械臂采集（夹爪仍用键盘）：

```bash
python scripts/collect_data.py --task-name pick_cube --fps 30 --teaching
```
命令精简为：`s`=录制、`d`=保存、`g/c/o`=夹爪、`q`=退出。

---

## 7. 数据格式与检查

每条 episode 存为一个 HDF5（`--save-dir/--task-name_N.hdf5`），30Hz 逐帧记录：

| 数据集 | 形状 | 含义 |
|--------|------|------|
| `observations/qpos` | (N, 7) | 6 关节角 + 夹爪归一化(0~1, 1=张开) |
| `observations/images/camera_global` | (N, H, W, 3) | 顶部 D435 图像 |
| `observations/images/camera_left` | (N, H, W, 3) | 腕部 Orbbec 图像 |
| `action` | (N, 7) | 行为克隆标签 = 下一帧 qpos |
| `timestamps` | (N,) | 相对时间戳(秒) |

检查一条数据：

```bash
python - <<'PY'
import h5py
with h5py.File('data/raw_hdf5/pick_cube_0.hdf5', 'r') as f:
    n = f['observations/qpos'].shape[0]
    print('keys:', list(f.keys()))
    print('qpos:', f['observations/qpos'].shape, '| fps:', f.attrs.get('fps'))
    print(f'帧数: {n} → {n/30:.1f} 秒 @30Hz')
PY
```
一条 pick-and-place 通常 5~8 秒（150~240 帧）；**帧数 <100 说明录制可能中断**。

---

## 8. 数据量与质量

| 策略 | 最少 | 推荐 |
|------|------|------|
| ACT | ~50 | 100+ |
| Diffusion / VQ-BeT | ~80 | 100+ |
| VLA (SmolVLA/Pi0) | ~50 | 100+ |

**质量 > 数量**：
- 物体位置/姿态要有变化，覆盖更多状态空间；
- 相机位置、光照保持一致；
- **删掉失败的 episode**（混入失败轨迹会拉低策略表现）。

> ⚠️ **频率对齐铁律**：采集 `--fps` = 转换 `--fps` = 推理 `--freq`，三者必须一致（本项目用 30）。

---

## 9. 采集之后

转成 LeRobot 格式（注意 `--fps` 与采集一致）：

```bash
python scripts/convert_to_lerobot.py \
    --input-dir data/raw_hdf5 \
    --output-dir data/pick_cube_30fps \
    --repo-id lerobot/pick_cube_30fps \
    --fps 30 \
    --task "pick up the cube and place it in the basket"
```
后续训练/推理见 [完整流程指南](pipeline_guide.md)。

---

## 10. 采集阶段排错

| 现象 | 排查 |
|------|------|
| `Vive: FAIL` | SteamVR 没开 / tracker 未追踪 / 基站没通电；无头显报 `Init_HmdNotFound` → 需配 null 假头显（见 [硬件配置 README](../hardware/README.md)）；多 tracker 用 `--tracker-serial` 指定 |
| 按 `w` 机械臂乱跳 | `ROBOT_INIT_POS/ORI` 没标定（见 3.2） |
| 机械臂方向反了 | 坐标映射符号需翻转（见 3.2b） |
| `import pyorbbecsdk` 报 undefined symbol | 没 `source env.sh`（库路径未修） |
| 相机 "Device is already in use" | `pkill -f realsense`；确认两相机分属不同 USB 控制器（`lsusb -t`） |
| 夹爪使能失败 | 检查接线/从站地址(`--gripper-slave-id`)/波特率；脚本已带重试 |
| 夹爪开合方向反 | 确认标定文件；`g 0`=闭合、`g 100`=张开 |
| 腕部相机全黑 | 初始化顺序须 RealSense 先于 Orbbec（RealSense 的 `hardware_reset` 会打断 Orbbec UVC 流）；`collect_data.py` 已按此顺序，单独自检 Orbbec 时确保无 RealSense 进程在 reset |
| 机械臂 `socket connect err` / 连接被拒 | `--arm-ip` 要是臂的 IP，别用成本机网卡 IP（本机 enp5s0 曾配成 `192.168.5.80`）；确认臂上电、TCP 8080 可达 |
| 帧率明显低于 30 | 相机取流慢/USB 带宽不足；见 [问题排查](troubleshooting.md) |

更多见 [问题排查](troubleshooting.md)。
