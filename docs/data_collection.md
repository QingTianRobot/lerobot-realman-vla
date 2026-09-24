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
| 主手夹爪（默认启用） | Pika Sense | 串口 `/dev/tty_pika_left`；默认开且为夹爪首选控制源（`--no-pika-gripper` 关），与键盘互斥 |

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

**推荐用一键脚本启动 SteamVR**（清理幽灵进程树 → 常驻启动完整栈 → 轮询等待 `vrserver`+`vrmonitor` 就绪），需在**宿主机终端**执行（沙箱内看不到宿主机进程）：

```bash
# 清理残留 + 常驻启动完整栈 (vrserver + vrmonitor)，就绪后打印确认
bash hardware/steamvr_restart.sh

# 只清理幽灵进程树，不启动（上次退出残留导致 `game already running` 时用）
bash hardware/steamvr_restart.sh --kill
```

栈就绪后确认基站通电、tracker 在线，可单独自检：

```bash
python hardware/vive_tracker.py   # 打印发现的设备(hmd/tracker/基站)与实时位姿
```

采集脚本启动时会打印 `Vive: OK/FAIL`，`FAIL` 说明 SteamVR 没开或 tracker 没被追踪到。

> **无头显**（只有 Tracker）时，SteamVR 默认要求 HMD，`vive_tracker.py` 会报 `Init_HmdNotFound`；需配置 null 假头显驱动后才能无头显追踪，详见 [硬件配置 README 的「无头显运行」](../hardware/README.md)。
>
> **先启动 SteamVR 再跑脚本**：若直接跑 `vive_tracker.py`，`openvr.init` 会自己临时拉起 vrserver，脚本一退它就因 `Monitor: 0` 自杀，下次再跑冷启动来不及就绪而报 `No tracker found`（“第一次找得到、退出后找不到”）。一键脚本已固化正确启动顺序；手动 fallback（`steam steam://rungameid/250820`）见 README「运行自检与正确启动顺序」。

---

## 3. 首次标定（换硬件/安装变动时才需重做）

### 3.1 夹爪行程标定

实测机械开/合限位，避免命令顶到硬限位长期堵转发热；结果写入 `hardware/gripper_calibration.json`，**采集与推理会自动加载**，保证三端行程一致。

```bash
python hardware/changingtek_gripper.py --slave-id 2 --calibrate --margin 0.002
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

> ⚠️ **采集前务必先常驻启动 SteamVR**：推荐一键脚本 `bash hardware/steamvr_restart.sh`（自动清理幽灵树 + 常驻启动 + 轮询到 `vrserver` 与 `vrmonitor` 都就绪）；或手动 `steam steam://rungameid/250820`，再用 `pgrep -x vrserver && pgrep -x vrmonitor` 确认两个进程都在。采集是长任务，若靠脚本自己临时拉起 vrserver，它无 vrmonitor 撑着会在异常/退出时 `Monitor: 0` 自杀，`_control_loop` 随即读不到位姿（`pose is None` 静默 `continue`）——**机械臂中途不跟随了却仍在录制，采到废数据**。详见 [硬件配置 README「运行自检与正确启动顺序」](../hardware/README.md)。


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

> **Pika 主手夹爪默认启用且为夹爪首选控制源**（连不上自动回退键盘）；不用加 `--no-pika-gripper`，可选 `--pika-port` / `--pika-hz`。需按 `w` 开启遥操后夹爪才跟随主手，按 `p` 可与键盘互斥切换，详见 [5.1](#51-pika-主手夹爪默认启用的首选夹爪控制源)。

启动后逐项确认状态行；Vive 就绪后会**自动慢速归位到 `ROBOT_INIT`**（3 秒倒计时，请清空机械臂周围；**若机械臂本就在 `ROBOT_INIT` 附近——位置<2cm 且姿态<5°——则自动跳过、不做无谓的慢速运动**），归位到位后**夹爪也会自动复位到张开（`GRIPPER_RESET_VALUE`）**，随后打印单键命令表（无需回车）：

```
相机: top=OK, wrist=OK
机械臂: OK (192.168.5.123:8080)
夹爪: OK (/dev/realman/gripper_left)
Vive: OK
[!] 机械臂即将慢速归位到起始位, 请清空周围! 3 秒后开始...
    3...
    2...
    1...
[✓] 慢速归位完成 (v=5%)
[✓] 夹爪复位到 100%
--------------------------------------------------
单键即触发 (无需回车):
  [w] 遥控 开/关(自动取当前Tracker为零点)
  [s] 录制 开始/保存
  [h] 复位到起始位(常速)
  [o] 夹爪张开
  [c] 夹爪闭合
  [1] 夹爪 30%
  [2] 夹爪 60%
  [3] 夹爪 100%
  [p] 夹爪控制源 Pika主手/键盘 切换(互斥)
  [q] 退出
  [Ctrl+C] 强制退出
--------------------------------------------------
[状态] 遥控:暂停 | 空闲 | 已存 0 条 | 夹爪 0%
```

> 命令表由 [`configs/keybindings.json`](../configs/keybindings.json) 自动生成，改该文件即可自定义按键（terminal 与后续 web 前端共用）。不想启动即归位加 `--no-home`；启动慢速归位在机械臂已处于 `ROBOT_INIT` 附近（位置<`ARM_HOME_POS_TOL`=2cm 且姿态<`ARM_HOME_ORI_TOL`=5°）时自动跳过，打印 `[i] 已在起始位附近 … 跳过慢速归位`（此时**夹爪仍会复位**，保证起始状态一致）；归位速度/倒计时/容差见 `collect_data.py` 顶部 `ARM_HOME_*` 常量。归位配套的夹爪复位开度/等待容差见 `GRIPPER_RESET_*` 常量。

---

## 5. 交互命令

> **单键即触发，无需回车**；按键与动作由 [`configs/keybindings.json`](../configs/keybindings.json) 定义，改该文件即可自定义（terminal 与后续 web 前端共用）。

| 按键 | 动作 | 备注 |
|------|------|------|
| `w` | 遥控 **开/关**（toggle） | 单键切换：**启用时以当前 tracker 位置为零点、机械臂当前位姿为基点**（原地 engage、不突跳到 `ROBOT_INIT`），随后跟随 tracker（**会动，注意安全**）；再按暂停、停在原地（仅 Vive 模式） |
| `s` | 录制 **开始/保存**（toggle） | 单键切换：开始一条（自动命名 `pick_cube_0.hdf5`、`_1`…）↔ 停止并保存（打印实际帧率与帧数，并**立即自动暂停遥操**） |
| `h` | **复位** | 常速（`v=45%`）归位到 `ROBOT_INIT`，**到位后夹爪也复位到张开**；执行前自动暂停遥操；**录制中也可复位**（归位运动会录进当前 episode） |
| `1`/`2`/`3` | 夹爪预设开度 | 分别 30%/60%/100%；百分比在 `keybindings.json` 的 `args.pct` 自定义 |
| `c` | 夹爪闭合 | 等价 0% |
| `o` | 夹爪张开 | 等价 100% |
| `p` | 夹爪控制源 **Pika主手/键盘** 切换 | 默认 Pika 主手；互斥切换（`--no-pika-gripper` 时仅键盘），见 [5.1](#51-pika-主手夹爪默认启用的首选夹爪控制源) |
| `q` | 退出 | 自动松夹爪、断连、关相机 |

> 机械臂位姿由 **Vive tracker** 控制，夹爪默认由 **Pika 主手**控制（需按 `w` 开启遥操后跟随；按 `p` 可切到键盘 `1`/`2`/`3`/`c`/`o`，见 5.1）（tracker 不管夹爪）。

### 5.1 Pika 主手夹爪（默认启用的首选夹爪控制源）

除键盘外，可用 **Pika Sense 手持主手夹爪**的开合来遥操作机械臂从手夹爪，与键盘控制**互斥**。封装见 [`hardware/pika_gripper.py`](../hardware/pika_gripper.py)，驱动来自 submodule `vendor/pika_sdk`。

**主从映射**：主手 `get_gripper_distance()`（mm，0=闭合→~109=全开）→ 归一化 `v = clamp((d-min_mm)/(max_mm-min_mm),0,1)` → 从手 `move_normalized(v)`。两者方向一致，无需反转。

**① 标定行程**（强烈建议一次；未标定用理论 `0~109mm` 兜底、精度差）：
```bash
python hardware/pika_gripper.py --calibrate   # 按提示先【完全闭合】再【完全张开】各采一点
python hardware/pika_gripper.py               # 自检：实时打印 行程(mm) 与归一化值
```
结果写入 `hardware/pika_calibration.json`（机器相关，已 `.gitignore`），自检与采集自动加载。

**② 采集时启用**（默认已开，无需额外参数）：
```bash
python scripts/collect_data.py [--pika-port /dev/tty_pika_left] [--pika-hz 30] ...   # 默认已启用 Pika
python scripts/collect_data.py --no-pika-gripper ...                                # 不用 Pika，仅键盘
```
- **默认启用且为夹爪首选控制源**（`gripper_source=pika`）；连不上时自动回退键盘。按 `p` 在 `Pika主手 ↔ 键盘` 间互斥切换（状态行显示当前控制源）。
- **需开启遥操才控夹爪**：仅当机械臂遥操已启用（按 `w`）时，后台线程才按 `--pika-hz`（默认 30Hz）把主手行程写到从手；暂停遥操（再按 `w`/`s`/`h`）夹爪同步停写。变化 < 死区（默认 0.02）不重复下发。示教模式无遥操概念，不门控。
- Pika 为控制源时键盘 `o/c/1/2/3` 被拦截；切回键盘则 Pika 线程停写、键盘恢复。
- ⚠️ 遥操开启的**瞬间从手不再突跳到主手开度**：后台线程以从手**当前实际位置**为起点、按 `--pika-ramp-rate`（默认 `2.5`/秒，全程 0→1 约 0.4s）**平滑斜坡**逼近主手当前开合，避免夹手/撞物与录制里的快变；设 `--pika-ramp-rate 0` 可关闭限速（退回瞬间对齐）。仍建议按 `w` 前先把主手摆到期望开度附近。
- 录制记录的是从手**实际反馈**位置，与控制源无关，键盘/Pika 采出的数据第 7 维语义一致，转换/训练流程无需改动。

---

## 6. 单条 episode 标准流程

1. 摆好场景（物体、容器位置；每条略变化以覆盖更多状态）
2. 手把 tracker 拿到顺手的**遥操起始姿态**（机械臂停在你想开始的位置即可，无需正好在 `ROBOT_INIT`）
3. 按 `w` 启用遥控 —— **以当前 tracker 位为零点、机械臂当前位姿为基点**，原地 engage 不突跳（再按一次暂停）→ 移动 tracker，机械臂跟随；**先空移确认方向不反**
4. 把机械臂移到任务起点 → 按 `s` 开始录制
5. 用 tracker 操控机械臂完成任务，在合适时机按 `c`（抓取）/ `o`（释放）/ `1`/`2`/`3`（预设开度）
6. 完成 → 再按 `s` 停止并保存（**遥操会自动暂停**，机械臂不再跟随手）
7. 按 `h` 常速复位到 `ROBOT_INIT`（**到位后夹爪自动复位到张开**）→ 复位场景 → 回到第 2 步（按 `w` 重新启用遥操）采下一条
8. 采够条数 → 按 `q` 退出

> 录制期间必须保持遥控启用（`w`），否则机械臂不动、录到的是静止轨迹。**按 `s` 停止录制会立即自动暂停遥操**；`h` 复位（录制中也可按）会先自动暂停遥操再归位，**归位运动会录进当前 episode**，复位后按 `w` 即以机械臂当前位姿为新零点重新启用。

### 示教模式（不用 Vive）

手动拖动机械臂采集（夹爪仍用键盘）：

```bash
python scripts/collect_data.py --task-name pick_cube --fps 30 --teaching
```
命令精简为（单键即触发）：`s`=录制开始/保存、`1`/`2`/`3`/`c`/`o`=夹爪、`h`=复位、`q`=退出。

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
| `game already running` | 上次退出残留了 reaper 幽灵进程树；`bash hardware/steamvr_restart.sh`（先清树再启动），或 `bash hardware/steamvr_restart.sh --kill` 只清理不启动 |
| 按 `w` 机械臂乱跳 | `ROBOT_INIT_POS/ORI` 没标定（见 3.2） |
| 启动/复位打印 `[!] 归位失败: rm_movej_p ret=X` | 该次归位未执行（不会乱动）；ret≠0 多为 `ROBOT_INIT` 位姿不可达/姿态不合理，用 `--read-init --write` 重标定后再试；只想跳过启动归位加 `--no-home` |
| 机械臂方向反了 | 坐标映射符号需翻转（见 3.2b） |
| `import pyorbbecsdk` 报 undefined symbol | 没 `source env.sh`（库路径未修） |
| 相机 "Device is already in use" | `pkill -f realsense`；确认两相机分属不同 USB 控制器（`lsusb -t`） |
| 夹爪使能失败 | 检查接线/从站地址(`--gripper-slave-id`)/波特率；脚本已带重试 |
| 夹爪开合方向反 | 确认标定文件；`c`=闭合(0%)、`o`=张开(100%) |
| 腕部相机全黑 | 初始化顺序须 RealSense 先于 Orbbec（RealSense 的 `hardware_reset` 会打断 Orbbec UVC 流）；`collect_data.py` 已按此顺序，单独自检 Orbbec 时确保无 RealSense 进程在 reset |
| 机械臂 `socket connect err` / 连接被拒 | `--arm-ip` 要是臂的 IP，别用成本机网卡 IP（本机 enp5s0 曾配成 `192.168.5.80`）；确认臂上电、TCP 8080 可达 |
| 帧率明显低于 30 | 相机取流慢/USB 带宽不足；见 [问题排查](troubleshooting.md) |

更多见 [问题排查](troubleshooting.md)。
