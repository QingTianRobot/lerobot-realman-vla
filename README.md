# 🤖 LeRobot-RealMan-VLA

<div align="center">

**基于 [LeRobot](https://github.com/huggingface/lerobot) 框架的睿尔曼机械臂 VLA 全流程方案**

*数据采集 → 格式转换 → 多策略训练 → 实机推理*


[![LeRobot](https://img.shields.io/badge/Framework-LeRobot-blue)](https://github.com/huggingface/lerobot)
[![Python](https://img.shields.io/badge/Python-3.10-green)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange)](LICENSE)

[English](README_EN.md) | 中文

</div>

---

## 🎯 项目定位

为拥有**非 ALOHA 硬件**的开发者提供端到端的模仿学习参考方案：

- **非标硬件适配 LeRobot**：LeRobot 原生支持 WidowX / ViperX / ALOHA，其他机械臂需要自己写采集和推理脚本。本项目提供了 RM65 的完整适配
- **全流程实机验证**：每一步（采集→转换→训练→推理）都经过真机验证，不是纯理论
- **多策略对比**：同一数据集、同一硬件上 5 种策略的训练与推理对比，带实测数据
- **工程细节记录**：频率对齐铁律、conda/ROS2 进程隔离、推理后处理（EMA+死区）等实际部署中的坑和解决方案

> 底层硬件平台见 [realman-ros2-platform](https://github.com/Humble2Full/realman-ros2-platform)

---

## 📌 项目简介

本项目在 HuggingFace LeRobot 框架上，实现了**睿尔曼 RM65 六轴机械臂 + 自主夹爪**的完整模仿学习流程。通过 Vive Tracker 遥操作采集演示数据，支持 **ACT / Diffusion Policy / VQ-BeT / SmolVLA / Pi0** 等多种策略的训练与实机部署。

> 全流程实机验证通过。ACT 策略在 164 条演示数据、31K 训练步后达到 90%+ 的 pick-and-place 成功率。文档包含实际部署中的问题排查与解决方案。

### ✨ 核心特性

- 🎮 **Vive 遥操作采集**：基于 OpenVR 的低延迟遥操作，支持示教模式（手动拖动）
- 📷 **双相机系统**：顶部 RealSense D435 + 腕部奥比中光 Gemini 305，覆盖全局与精细观测
- 🔧 **知行 RTU 夹爪**：独立串口异步控制，与机械臂解耦，不阻塞主循环
- ✊ **Pika 主手夹爪**（默认启用）：手持主手开合遥操从手夹爪，为夹爪首选控制源（与键盘互斥，开启遥操后生效）
- 🧠 **多策略支持**：一套数据，多种策略对比训练
- ⚡ **优化推理**：EMA 平滑 + 死区过滤 + Temporal Ensemble

### 🔧 实机demo
<img src="docs/assets/4倍速实机夹取示例.gif" width="600" alt="ACT pick-and-place demo (4x speed)">


### 📊 策略对比一览

| 策略 | 类型 | 显存需求 | 训练速度 | 推理延迟 | 适合场景 |
|------|------|---------|---------|---------|---------|
| **ACT** | Transformer | ~4GB | ⚡ 快 | ~5ms | 精细操作，首选基线 |
| **Diffusion** | 扩散模型 | ~6GB | 🔄 中 | ~50ms | 多模态分布 |
| **VQ-BeT** | VQ-VAE+GPT | ~5GB | 🐢 慢 | ~10ms | 离散动作空间 |
| **SmolVLA** | VLM (500M) | ~12GB | 🐢 慢 | ~100ms | 语言条件任务 |
| **Pi0** | VLM (2B) | ~24GB | 🐌 很慢 | ~200ms | 泛化能力强 |

---

## 🔧 硬件配置

| 组件 | 型号 | 说明 |
|------|------|------|
| 机械臂 | 睿尔曼 RM65-B | 6自由度，TCP/IP通信（IP 192.168.5.123） |
| 夹爪 | 知行 RTU 平动手 | Modbus RTU，独立串口 `/dev/realman/gripper_left` |
| 主手夹爪（默认启用） | Pika Sense | 串口 `/dev/tty_pika_left`；默认开且为夹爪首选控制源（`--no-pika-gripper` 关），与键盘互斥 |
| 遥操作 | Vive Tracker (Pika Sense viva) | OpenVR/SteamVR |
| 顶部相机 | Intel RealSense D435 | 640×480@30fps |
| 腕部相机 | 奥比中光 Gemini 305 | 640×480@30fps (pyorbbecsdk) |
| 训练 GPU | NVIDIA RTX 4080 Laptop | 12GB VRAM |

<details>
<summary>🖼️ 硬件布局示意</summary>

```
        [顶部相机 D435i]
              |
              ▼
    ┌─────────────────┐
    │    工作台面       │
    │   ┌─────┐       │
    │   │ 物体 │       │
    │   └─────┘       │
    │         [RM65]──[腕部相机]
    │           │      
    │         [夹爪]   
    └─────────────────┘
```

</details>

---

## 📁 项目结构

```
lerobot-realman-vla/
├── README.md                     # 本文档
├── README_EN.md                  # English README
├── LICENSE                       # Apache 2.0
├── requirements.txt              # Python依赖 (pin lerobot==0.4.3)
├── uv.toml                       # uv 镜像配置 (阿里云)
├── .gitignore                    # Git忽略规则
├── setup.sh                      # 一键环境搭建 (uv)
├── env.sh                        # 环境激活: source env.sh
│
├── scripts/                      # 🔧 核心脚本
│   ├── collect_data.py           # Step 1: 数据采集
│   ├── convert_to_lerobot.py     # Step 2: HDF5→LeRobot格式转换
│   ├── train.sh                  # Step 3: 训练启动（本地）
│   ├── train_hpc.slurm           # Step 3: 训练启动（Slurm集群）
│   └── inference.py              # Step 4: 通用多策略推理
│
├── configs/                      # ⚙️ 训练配置
│   ├── act_realman.yaml          # ACT策略配置
│   ├── diffusion_realman.yaml    # Diffusion Policy配置
│   ├── vqbet_realman.yaml        # VQ-BeT配置
│   ├── smolvla_realman.yaml      # SmolVLA配置
│   └── pi0_realman.yaml          # Pi0配置
│
├── hardware/                     # 🔌 硬件驱动
│   ├── vive_tracker.py           # Vive Tracker OpenVR接口
│   ├── orbbec_camera.py          # 腕部相机 奥比中光 Gemini 305 (pyorbbecsdk)
│   ├── changingtek_gripper.py    # 知行 RTU 夹爪归一化封装
│   ├── pika_gripper.py           # Pika Sense 主手夹爪封装（默认首选控制源）
│   └── README.md                 # 硬件配置指南
│
├── docs/                         # 📚 技术文档
│   ├── pipeline_guide.md         # 完整流程指南（含踩坑记录）
│   ├── data_collection.md        # 数据采集指南（自检/标定/逐条流程）
│   └── troubleshooting.md        # 常见问题排查
│
├── examples/                     # 📝 示例
│   └── README.md                 # 示例数据说明
│
├── data/                         # 📦 数据集 (raw_hdf5 + lerobot 格式; 内容 git 忽略)
└── outputs/                      # 📦 训练产物 (checkpoints; 内容 git 忽略)
```

---

## 🚀 快速开始

### 1. 环境安装

本项目用 [uv](https://docs.astral.sh/uv/) 管理环境（Python 3.10，仓库内 `.venv`），镜像走阿里云（见 `uv.toml`）。

```bash
# 克隆本项目（--recursive 拉取子模块：知行夹爪 SDK vendor/changingtek_rtu_sdk、vendor/pika_sdk）
git clone --recursive https://github.com/Humble2Full/lerobot-realman-vla.git
cd lerobot-realman-vla
# 若已普通 clone（子模块目录为空），补拉：git submodule update --init --recursive

# 一键安装（uv 建 .venv + 装依赖 + 装 pyorbbecsdk2 + 验证）
bash setup.sh
```

> 未装 uv 时：`curl -LsSf https://astral.sh/uv/install.sh | sh`
>
> 手动安装等价于：
> ```bash
> uv venv --python 3.10 .venv
> uv pip install -r requirements.txt           # lerobot==0.4.3 + torch + 驱动 (勿升级 lerobot)
> uv pip install --no-deps pyorbbecsdk2==2.1.2  # 腕部 Orbbec 相机 (依赖冲突, 需 --no-deps)
> ```

**每次使用前先激活环境**（会自动修好 Orbbec 库路径）：

```bash
source env.sh
```

> ⚠️ **腕部 Orbbec 相机库路径**：`pyorbbecsdk` 的 `libOrbbecSDK.so.2` 会被 `/opt/ros/humble` 旧库覆盖，
> 直接 import 报 `undefined symbol: ob_application_config_set_struct`。`source env.sh` 已自动把 `.venv`
> 自带库目录前置到 `LD_LIBRARY_PATH` 解决；若手动激活 `.venv`，需自行 export：
> ```bash
> export LD_LIBRARY_PATH=$(python -c "import pyorbbecsdk,os;print(os.path.dirname(pyorbbecsdk.__file__))"):$LD_LIBRARY_PATH
> ```

### 2. 数据采集

> 📖 详细采集步骤（硬件自检、夹爪/Vive 标定、逐条流程、数据检查）见 [数据采集指南](docs/data_collection.md)。

```bash
# Vive 遥操作模式
python scripts/collect_data.py \
    --arm-ip 192.168.5.123 \
    --gripper-port /dev/realman/gripper_left \
    --cam-top 262322074840 \
    --cam-wrist CV2T66100096 \
    --save-dir data/raw_hdf5 \
    --task-name pick_cube \
    --fps 30

# 示教模式（手动拖动，无需Vive）
python scripts/collect_data.py \
    --arm-ip 192.168.5.123 \
    --save-dir data/raw_hdf5 \
    --task-name pick_cube \
    --fps 30 \
    --teaching
```

### 3. 数据转换

```bash
python scripts/convert_to_lerobot.py \
    --input-dir data/raw_hdf5/pick_cube \
    --output-dir data/pick_cube_30fps \
    --repo-id lerobot/pick_cube_30fps \
    --fps 30 \
    --task "pick up the cube and place it in the basket"
```

### 4. 训练

```bash
# 本地训练（ACT 推荐入门）
bash scripts/train.sh act

# 集群训练（Slurm）
sbatch scripts/train_hpc.slurm                              # 默认 ACT
sbatch --export=ALL,POLICY=smolvla scripts/train_hpc.slurm  # SmolVLA

# 断点续训
bash scripts/train.sh act resume
```

### 5. 推理

```bash
python scripts/inference.py \
    --model outputs/act_realman/checkpoints/100000/pretrained_model \
    --arm-ip 192.168.5.123 \
    --gripper-port /dev/realman/gripper_left \
    --cam-top 262322074840 \
    --cam-wrist CV2T66100096 \
    --freq 30
```

---

## 📖 完整流程图

```
┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐
│  Step 1     │     │  Step 2     │     │  Step 3     │     │  Step 4     │
│  数据采集    │────▶│  格式转换    │────▶│  策略训练    │────▶│  实机推理    │
│             │     │             │     │             │     │             │
│ Vive遥操作   │     │ HDF5→LeRobot│     │ ACT/Diff/..│     │ 实时控制     │
│ 30Hz双相机   │     │ 视频编码     │     │ GPU训练     │     │ EMA平滑     │
│ 关节+夹爪    │     │ 归一化统计   │     │ WandB日志   │     │ 异步夹爪    │
└────────────┘     └────────────┘     └────────────┘     └────────────┘
      │                   │                  │                   │
      ▼                   ▼                  ▼                   ▼
   raw_hdf5/          lerobot_v3/       checkpoints/      机械臂执行动作
   *.hdf5              videos+parquet    pretrained_model
```

---

## ⚠️ 重要注意事项

### 频率对齐铁律

**采集频率 = 转换FPS = 推理频率**，三者必须一致！

```
采集 --fps 30  →  转换 --fps 30  →  推理 --freq 30  ✅
采集 --fps 30  →  转换 --fps 15  →  推理 --freq 30  ❌ 动作会被拉伸！
```

### 任务描述一致性

VLA策略（SmolVLA/Pi0）要求训练和推理时的 `task` 描述**完全一致**：

```bash
# 转换时
python scripts/convert_to_lerobot.py --task "pick up the cube"
# 推理时
python scripts/inference.py --task "pick up the cube"
```

---

## 📚 文档导航

| 文档 | 说明 |
|------|------|
| [完整流程指南](docs/pipeline_guide.md) | 采集→转换→训练→推理的操作手册（含踩坑记录） |
| [数据采集指南](docs/data_collection.md) | 遥操/示教采集详细步骤（自检、标定、逐条流程） |
| [常见问题](docs/troubleshooting.md) | 实际遇到的问题和排查过程 |
| [硬件配置](hardware/README.md) | 机械臂、夹爪、相机、Vive 配置 |

---

## 🙏 致谢

- [**HuggingFace LeRobot**](https://github.com/huggingface/lerobot) (Apache-2.0) — 训练框架和数据格式转换流程基于 LeRobot 官方源码
- [**睿尔曼智能 RM_API2**](https://github.com/RealManRobot/RM_API2) (Apache-2.0) — 机械臂通信 SDK
- [**OpenVR / triad_openvr**](https://github.com/TriadSemi/triad_openvr) — `hardware/vive_tracker.py` 参考了其 Tracker 位姿读取方案并重写
- [ACT (Zhao et al., 2023)](https://arxiv.org/abs/2304.13705)、[Diffusion Policy (Chi et al., 2023)](https://arxiv.org/abs/2303.04137)、[VQ-BeT (Lee et al., 2024)](https://arxiv.org/abs/2403.03181)、[Pi0 (Black et al., 2024)](https://www.physicalintelligence.company/blog/pi0)、[SmolVLA (HuggingFace, 2024)](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) — 算法实现均来自 LeRobot 框架内置模块

### 合作贡献

`scripts/collect_data.py` 中 Vive Tracker 遥操作采集模式的初始代码由实验室同门合作提供，后续由本项目作者进行重构，增加了示教模式、双相机支持、数据格式化等功能。

> 详细的第三方依赖及许可证信息见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

---

## 📄 License

本项目采用 [Apache License 2.0](LICENSE)。

---

<div align="center">

**如果这个项目对你有帮助，请点个 ⭐ Star！**

</div>
