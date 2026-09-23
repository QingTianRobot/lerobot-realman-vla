#!/usr/bin/env bash
# ============================================================
# 环境激活脚本  ——  用法:  source env.sh
#   1) 激活本仓库的 uv 虚拟环境 .venv
#   2) 修正腕部 Orbbec 相机的库搜索路径
#      (让 .venv 自带的 libOrbbecSDK.so.2 优先于 /opt/ros/humble 的旧库,
#       否则 import pyorbbecsdk 报 undefined symbol: ob_application_config_set_struct)
# 兼容 bash / zsh。每次开新终端跑采集/推理前先 source 本文件。
# ============================================================

# 本仓库根目录 (兼容 bash 的 BASH_SOURCE 与 zsh 的 $0)
_LRMVLA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

if [ ! -f "$_LRMVLA_DIR/.venv/bin/activate" ]; then
  echo "[env.sh] 未找到 .venv，请先执行:"
  echo "    uv venv --python 3.10 .venv"
  echo "    uv pip install -r requirements.txt"
  echo "    uv pip install --no-deps pyorbbecsdk2==2.1.2"
  return 1 2>/dev/null || exit 1
fi

# 激活虚拟环境
source "$_LRMVLA_DIR/.venv/bin/activate"

# 用 sysconfig 定位 site-packages (不 import pyorbbecsdk, 避免此时就触发符号错误)
_LRMVLA_SP="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null)"
_LRMVLA_ORBBEC="$_LRMVLA_SP/pyorbbecsdk"
if [ -d "$_LRMVLA_ORBBEC" ]; then
  export LD_LIBRARY_PATH="$_LRMVLA_ORBBEC:$LD_LIBRARY_PATH"
else
  echo "[env.sh] 警告: 未找到 $_LRMVLA_ORBBEC (pyorbbecsdk2 未安装?)"
fi

echo "[env.sh] ✓ 已激活 .venv  (Python $(python -c 'import sys;print(sys.version.split()[0])' 2>/dev/null))"
echo "[env.sh] ✓ lerobot $(python -c 'import lerobot;print(lerobot.__version__)' 2>/dev/null) | LD_LIBRARY_PATH 已前置 pyorbbecsdk 库目录"

unset _LRMVLA_DIR _LRMVLA_SP _LRMVLA_ORBBEC
