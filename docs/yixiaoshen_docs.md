
# 数据采集脚本
python scripts/collect_data.py \
    --save-dir data/raw_hdf5/test --task-name test --fps 30 \
	--task "test"



# smolVLA模型推理脚本

cd /home/robot/repo/lerobot-realman-vla
export DISPLAY=:1 ; export XAUTHORITY=/run/user/1000/gdm/Xauthority
source ./env.sh
python scripts/inference.py \
    --model /home/robot/repo/lerobot-realman-vla/outputs/smolvla_realman_sztu/checkpoints/040000/pretrained_model \
    --task "pick up the plug and place it in the box" \
    --freq 30 --ema-alpha 0.7 --deadzone 0 --offline --headless --reset-and-run --print-step


# pi05模型推理脚本
cd /home/robot/repo/lerobot-realman-vla
source ./env.sh
python scripts/inference.py \
  --model /home/robot/repo/lerobot-realman-vla/outputs/pi05_realman_sztu/checkpoints/015000/pretrained_model \
  --task "pick up the plug and place it in the box" \
  --freq 30 \
  --ema-alpha 0.7 \
  --deadzone 0 \
  --offline \
  --reset-and-run \
  --print-step \
  --headless
