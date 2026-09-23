#!/bin/bash
# ============================================================
# LeRobot-RealMan-VLA 一键环境搭建脚本 (uv 版)
#
# 用 uv 在仓库内创建 .venv (Python 3.10)，安装:
#   - lerobot==0.4.3 (代码强绑定 0.4.x 新 API，勿升级)
#   - torch/torchvision (由 lerobot pin 到 2.7.1，自带 CUDA 12.x 运行库)
#   - 硬件驱动: robotic-arm(RM_API2) / pyrealsense2(D435) / openvr(Vive)
#                minimalmodbus+pyserial(知行夹爪) / transformations
#   - pyorbbecsdk2 (腕部 Orbbec 305，因依赖冲突需 --no-deps 单独装)
#
# 镜像: 默认走阿里云 (见仓库根 uv.toml)
# ============================================================
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
cd "$(dirname "$0")"

echo "============================================"
echo "  LeRobot-RealMan-VLA Environment Setup (uv)"
echo "============================================"

# 0. 检查 uv
if ! command -v uv >/dev/null 2>&1; then
    echo -e "${RED}未找到 uv。请先安装:${NC}"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo -e "${GREEN}uv 版本: $(uv --version)${NC}"

# 1. 创建虚拟环境
echo -e "\n${GREEN}[1/4] 创建 .venv (Python 3.10)...${NC}"
if [ -d ".venv" ]; then
    echo -e "${YELLOW}.venv 已存在，跳过创建。${NC}"
else
    uv venv --python 3.10 .venv
fi

# 2. 安装主依赖 (lerobot==0.4.3 + torch + 硬件驱动，镜像见 uv.toml)
echo -e "\n${GREEN}[2/4] 安装主依赖 (lerobot==0.4.3 + torch + 驱动)...${NC}"
VIRTUAL_ENV=.venv uv pip install -r requirements.txt

# 3. 腕部 Orbbec 相机 SDK (依赖声明与 lerobot 冲突，核心取流不需要那些依赖，故 --no-deps)
echo -e "\n${GREEN}[3/4] 安装 pyorbbecsdk2 (--no-deps)...${NC}"
VIRTUAL_ENV=.venv uv pip install --no-deps pyorbbecsdk2==2.1.2

# 4. 验证
echo -e "\n${GREEN}[4/4] 验证导入...${NC}"
# shellcheck disable=SC1091
source env.sh
python - <<'PY'
import importlib
ok = True
for m in ["lerobot","torch","Robotic_Arm","pyorbbecsdk","pyrealsense2",
          "openvr","minimalmodbus","serial","transformations","h5py"]:
    try:
        importlib.import_module(m); print(f"  OK  {m}")
    except Exception as e:
        ok = False; print(f"  FAIL {m}: {e}")
import torch; print(f"  torch.cuda.is_available = {torch.cuda.is_available()}")
import lerobot; print(f"  lerobot = {lerobot.__version__}")
PY

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "每次使用前激活环境 (含 Orbbec 库路径修正):"
echo "  source env.sh"
echo ""
echo "Next steps:"
echo "  1. 连接硬件 (RM65 @192.168.5.123, 顶部 D435, 腕部 Orbbec 305, 知行夹爪, Vive)"
echo "  2. 若用 Vive 遥操，先启动 SteamVR"
echo "  3. 硬件自检: python hardware/realman_arm.py"
echo "              python hardware/changingtek_gripper.py --slave-id 2"
echo "              python hardware/orbbec_camera.py --serial CV2T66100096"
echo "              python hardware/realsense_camera.py --serial 262322074840"
echo "  4. 采集:     python scripts/collect_data.py --help"
