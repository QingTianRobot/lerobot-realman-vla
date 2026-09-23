# 🤖 LeRobot-RealMan-VLA

<div align="center">

**End-to-End VLA Pipeline for RealMan RM65 Robot Arm based on [LeRobot](https://github.com/huggingface/lerobot)**

*Data Collection → Format Conversion → Multi-Policy Training → Real-Robot Inference*

[![LeRobot](https://img.shields.io/badge/Framework-LeRobot-blue)](https://github.com/huggingface/lerobot)
[![Python](https://img.shields.io/badge/Python-3.10-green)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange)](LICENSE)

[中文](README.md) | English

</div>

---

## Overview

This project implements a complete imitation learning pipeline for the **RealMan RM65 6-DOF robot arm with a custom Modbus gripper**, built on the HuggingFace LeRobot framework. It supports teleoperation data collection via HTC Vive Tracker and training/deployment of multiple state-of-the-art policies including **ACT, Diffusion Policy, VQ-BeT, SmolVLA, and Pi0**.

### Key Features

- 🎮 **Vive Teleoperation**: Low-latency teleoperation via OpenVR, with teaching mode fallback
- 📷 **Dual Camera System**: RealSense top + wrist cameras for global and fine-grained observation
- 🔧 **Async Modbus Gripper**: Non-blocking gripper control that doesn't affect control frequency
- 🧠 **Multi-Policy Support**: One dataset, multiple policy comparisons
- ⚡ **Optimized Inference**: EMA smoothing + deadzone filtering + temporal ensemble

---

## Hardware Requirements

| Component | Model | Description |
|-----------|-------|-------------|
| Robot Arm | RealMan RM65-B | 6-DOF, TCP/IP communication |
| Gripper | FAE2M86C | Modbus RTU protocol |
| Teleoperation | HTC Vive Tracker 3.0 | OpenVR/SteamVR |
| Top Camera | Intel RealSense D435i | 640×480@30fps |
| Wrist Camera | Intel RealSense D435i | 640×480@30fps |
| Training GPU | NVIDIA RTX 4080 Laptop | 12GB VRAM |

---

## Quick Start

```bash
# Clone and setup (--recursive fetches the gripper SDK submodule vendor/changingtek_rtu_sdk)
git clone --recursive https://github.com/Humble2Full/lerobot-realman-vla.git
cd lerobot-realman-vla
# If already cloned without it: git submodule update --init --recursive
bash setup.sh

# Collect data
python scripts/collect_data.py --arm-ip <YOUR_ARM_IP> --save-dir data/raw --fps 30

# Convert to LeRobot format
python scripts/convert_to_lerobot.py \
    --input-dir data/raw \
    --output-dir data/lerobot \
    --repo-id lerobot/pick_cube_30fps \
    --fps 30 \
    --task "pick up the cube and place it in the basket"

# Train (ACT recommended for beginners)
bash scripts/train.sh act

# Inference
python scripts/inference.py \
    --model outputs/act_realman/checkpoints/100000/pretrained_model \
    --arm-ip <YOUR_ARM_IP> \
    --freq 30
```

---

## Documentation

- [Pipeline Guide](docs/pipeline_guide.md) — Step-by-step instructions
- [Data Collection Guide](docs/data_collection.md) — Teleoperation/teaching collection (self-checks, calibration, per-episode workflow)
- [Troubleshooting](docs/troubleshooting.md) — Common issues and solutions
- [Hardware Setup](hardware/README.md) — Hardware configuration guide

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [**HuggingFace LeRobot**](https://github.com/huggingface/lerobot) (Apache-2.0) — Training framework and data format conversion based on LeRobot official source code
- [**RealMan RM_API2**](https://github.com/RealManRobot/RM_API2) (Apache-2.0) — Robot arm communication SDK
- [**OpenVR / triad_openvr**](https://github.com/TriadSemi/triad_openvr) — `hardware/vive_tracker.py` references its Tracker pose reading approach and was rewritten for RM65
- All algorithm implementations (ACT, Diffusion Policy, VQ-BeT, SmolVLA, Pi0) come from LeRobot's built-in modules

### Collaboration

The initial Vive Tracker teleoperation code in `scripts/collect_data.py` was co-developed with a lab colleague. Subsequent refactoring (teaching mode, dual-camera support, data formatting) was done by the project author.

> See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for full third-party attribution.
