# 完整流程指南

---

## Step 1: 数据采集

> 📖 详细采集指南（硬件自检、夹爪/Vive 标定、逐条流程、数据检查）见 [数据采集指南](data_collection.md)。本节为速查。

### 采集命令

> 采集前确认：机械臂可达（`ping <YOUR_ARM_IP>`）、顶部 D435（`rs-enumerate-devices`）与腕部 Orbbec 305 均识别、SteamVR 中 Tracker 在线。

```bash
# Vive 遥操作
python scripts/collect_data.py \
    --arm-ip <YOUR_ARM_IP> \
    --save-dir data/raw_hdf5 \
    --task-name pick_cube \
    --fps 30

# 示教模式（手动拖动，不需要 Vive）
python scripts/collect_data.py \
    --arm-ip <YOUR_ARM_IP> \
    --save-dir data/raw_hdf5 \
    --task-name pick_cube \
    --fps 30 \
    --teaching
```

### 操作按键

| 按键 | 功能 | 备注 |
|------|------|------|
| `w` | 遥控 开/关（toggle） | 启用时**自动以当前 Tracker 位置为零点**，无需单独校准（仅 Vive 模式） |
| `s` | 录制 开始/保存（toggle） | 单键切换开始 / 停止并保存 |
| `h` | 复位 | 常速归位到 `ROBOT_INIT`；执行前自动暂停遥操（仅 Vive 模式） |
| `1`/`2`/`3` | 夹爪预设开度 | 30%/60%/100%，可在 [`configs/keybindings.json`](../configs/keybindings.json) 自定义 |
| `c` / `o` | 夹爪闭合 / 张开 | 0% / 100% |
| `q` | 退出 | |

> 单键即触发（无需回车）；按键表见 [`configs/keybindings.json`](../configs/keybindings.json)，terminal 与后续 web 前端共用。

### 数据量参考

| 策略 | 最少 | 推荐 | 说明 |
|------|------|------|------|
| ACT | ~50 | 100+ | 164 条 + 31K 步可达 90%+ 成功率 |
| Diffusion / VQ-BeT | ~80 | 100+ | 对多样性要求更高 |
| VLA (SmolVLA/Pi0) | ~50 | 100+ | 预训练迁移能力强，理论上需要更少 |

**采集质量 > 数据数量**。注意：
- 物体位置、姿态要有变化（覆盖更多状态空间）
- 相机位置和光照保持一致
- 删掉失败的 episode（混入失败轨迹会拉低策略表现）

### 数据检查

```bash
python -c "
import h5py
with h5py.File('data/raw_hdf5/pick_cube_0.hdf5', 'r') as f:
    print('Keys:', list(f.keys()))
    print('qpos shape:', f['observations/qpos'].shape)
    n = f['observations/qpos'].shape[0]
    print(f'帧数: {n}, 30Hz → {n/30:.1f}秒')
"
```

一般一条 pick-and-place 轨迹 5-8 秒，对应 150-240 帧。帧数 <100 说明录制可能中断了。

---

## Step 2: 数据转换

### 转换命令

```bash
python scripts/convert_to_lerobot.py \
    --input-dir data/raw_hdf5/pick_cube \
    --output-dir data/pick_cube_30fps \
    --repo-id lerobot/pick_cube_30fps \
    --fps 30 \
    --task "pick up the cube and place it in the basket"
```

### 参数说明

| 参数 | 说明 | 注意 |
|------|------|------|
| `--fps` | 标注帧率 | **必须与采集一致**，见下方说明 |
| `--task` | 任务描述 | VLA 策略推理时必须完全一致 |
| `--repo-id` | 数据集 ID | 训练配置中引用 |

### ⚠️ `--fps` 不是降采样

**问题**：采集 30Hz，实际帧率有波动，转换时用 `--fps 15` 想"对齐"。

**原因**：`--fps` 只修改 `info.json` 里的标注，不做降采样。标 15 等于告诉模型"帧间隔 66ms"，但实际数据是 33ms 间距——时间尺度被拉伸 2 倍。

**方案**：采集代码设多少 Hz 就标多少。消融实验表明 15Hz 和 30Hz 标注下训练 loss 曲线完全一致，但推理频率不匹配时动作会异常。

### ⚠️ task 描述必须一致

对 ACT/Diffusion 无影响，但 VLA 策略（SmolVLA/Pi0）依赖 task 文本做条件推理。训练用 `"put the cube in the basket"`，推理用 `"pick cube"` → 动作看似合理但抓不到。改回一致后正常。

### 转换产物

```
data/pick_cube_30fps/
├── meta/
│   ├── info.json          # 数据集元信息
│   ├── episodes.jsonl     # episode 索引
│   └── stats.json         # 归一化统计量（μ, σ）
├── data/
│   └── train-*.parquet    # 状态/动作表格数据
└── videos/
    └── observation.images.camera_global/
        └── episode_000000.mp4
```

`stats.json` 用于训练归一化和推理反归一化，由 preprocessor/postprocessor 自动处理。

---

## Step 3: 训练

### 策略选择

| 场景 | 推荐 | 理由 |
|------|------|------|
| 首次跑通 | ACT | 训练快、显存小、效果稳 |
| 多模态动作 | Diffusion | 天然支持多模态分布 |
| 语言条件 + 有限显存 | SmolVLA | 500M 参数，8 GB 显存能跑 |
| 最强泛化 | Pi0 | 2B 参数，需 24GB+ |

### 训练命令

```bash
# 本地
bash scripts/train.sh act

# Slurm 集群（见 scripts/train_hpc.slurm）
sbatch scripts/train_hpc.slurm
sbatch --export=ALL,POLICY=smolvla scripts/train_hpc.slurm

# 断点续训
bash scripts/train.sh act resume
```

### 集群训练注意事项

| 问题 | 原因 | 解决 |
|------|------|------|
| HuggingFace 下载卡住 | 计算节点无外网 | `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`，预下载到 `HF_HOME` |
| Bus error | `batch × workers` 超出 `/dev/shm` | 减 batch 或 workers，Docker 检查 `--shm-size` |
| FileExistsError | 上次中断残留输出目录 | 脚本判断：有 checkpoint → resume，无 → 清理重建 |

### Loss 收敛参考

| 策略 | 关键 Loss | 经验收敛值 |
|------|----------|-----------|
| ACT | L1 + KL | L1 < 0.05（实测 31K 步 L1=0.042 → 成功率 >90%） |
| Diffusion | MSE | < 0.01 |
| VQ-BeT | 阶段1 VQ → 阶段2 GPT | GPT < 1.0（阶段切换时 loss 会跳高） |
| SmolVLA | Flow matching | < 0.1（收敛慢，50K+ 步） |
| Pi0 | Flow matching | < 0.01 |

> **注意**：loss 数值跨策略不可比。Pi0 loss=0.005 的推理效果可能不如 ACT loss=0.042。只在同策略内纵向看趋势。

### 训练量

`训练量 = batch_size × steps`。`batch=8, steps=100K` ≈ `batch=16, steps=50K`。显存不够时用梯度累积：`grad_accum=4, batch=4` ≈ `batch=16`。

### 断点续训

```bash
# 注意用等号（空格会导致参数解析错误）
python -m lerobot.scripts.lerobot_train \
    --config_path=outputs/act_realman/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
```

`--resume` 是 bool，配置恢复靠 `--config_path` 指向 checkpoint 里的 `train_config.json`。

### 显存优化（按优先级）

1. 减 `num_workers`（释放共享内存）
2. 减 `batch_size`（直接有效，同时增加 steps 补偿）
3. `gradient_checkpointing: true`（VLA 必备）
4. `freeze_vision_encoder: true` + `train_expert_only: true`
5. 换小模型（`gemma_300m` 替代 `gemma_2b`）

---

## Step 4: 推理

### 推理命令

```bash
# ACT
python scripts/inference.py \
    --model outputs/act_realman/checkpoints/100000/pretrained_model \
    --arm-ip <YOUR_ARM_IP> \
    --freq 30

# VLA（必须带 task）
python scripts/inference.py \
    --model outputs/smolvla_realman/checkpoints/50000/pretrained_model \
    --task "pick up the cube and place it in the basket" \
    --freq 15

# 无显示器
python scripts/inference.py --model ... --headless --freq 30
```

### 各环节耗时（实测）

| 环节 | 耗时 |
|------|------|
| 关节角读取 | ~1ms |
| 双相机采集 | ~1ms（异步取最新帧） |
| 预处理/后处理 | ~1ms |
| 模型推理 | 5ms (ACT) ~ 200ms (Pi0) |
| 发送指令 | ~1ms |
| 夹爪 Modbus | 0ms（异步）/ **500ms（同步）** |

> 夹爪 Modbus 同步阻塞是最大的延迟陷阱，见 [troubleshooting](troubleshooting.md#推理频率骤降至-1-2hzmodbus-延迟)。

### 推理频率

`--freq` 必须和训练数据 FPS 对齐。超过推理延迟允许的上限时脚本会跳帧（不报错但动作不连贯）。

| 策略 | 推理延迟 | 推荐 freq |
|------|---------|----------|
| ACT | ~5ms | 30 |
| Diffusion | ~50ms | 15 |
| VQ-BeT | ~10ms | 30 |
| SmolVLA | ~100ms | 10~15 |
| Pi0 | ~200ms | 5~10 |

### 快速排错

| 现象 | 可能原因 |
|------|---------|
| 机械臂不动 | preprocessor 加载失败（输出归一化值≈0） |
| 动作幅度异常 | freq 和训练 fps 不一致 |
| 夹爪不响应 | Modbus 未初始化 / 夹爪电源问题 |
| VLA 抓不到 | task 描述不匹配 |
| 频率很低 | Modbus 同步阻塞 |
