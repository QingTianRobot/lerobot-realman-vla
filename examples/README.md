# 📝 示例数据说明

## 示例数据

由于数据文件较大（单个 episode ~50MB），不包含在仓库中。

### 获取示例数据

```bash
# 采集几个测试 episode
python scripts/collect_data.py \
    --save-dir examples/sample_data \
    --task-name demo \
    --fps 30 \
    --teaching
```

### 数据格式检查

```python
import h5py
import numpy as np

with h5py.File('examples/sample_data/demo_0.hdf5', 'r') as f:
    print("=== HDF5 Structure ===")
    print(f"qpos:      {f['observations/qpos'].shape}")       # (N, 7)
    print(f"action:    {f['action'].shape}")                   # (N, 7)
    print(f"cam_global: {f['observations/images/camera_global'].shape}")  # (N, 480, 640, 3)
    print(f"cam_left:   {f['observations/images/camera_left'].shape}")   # (N, 480, 640, 3)
    print(f"FPS:       {f.attrs.get('fps', 'N/A')}")
    print(f"Frames:    {f['observations/qpos'].shape[0]}")
```

### 训练输出示例

训练完成后，输出目录结构:
```
outputs/act_realman_pick_cube/
├── checkpoints/
│   ├── 002500/
│   │   └── pretrained_model/
│   │       ├── config.json
│   │       ├── model.safetensors
│   │       ├── policy_preprocessor.json
│   │       ├── policy_postprocessor.json
│   │       └── train_config.json
│   ├── 005000/
│   ├── ...
│   └── last/ → 最新checkpoint的软链接
└── logs/
    └── training.log
```

### Loss 收敛参考

| 策略 | 10k步 | 50k步 | 100k步 | 判断标准 |
|------|-------|-------|--------|---------|
| ACT | L1≈0.08 | L1≈0.04 | L1≈0.02 | L1 < 0.02 且 KL < 0.1 |
| Diffusion | ≈0.05 | ≈0.02 | ≈0.01 | MSE < 0.01 |
| VQ-BeT | VQ阶段 | GPT≈2.0 | GPT≈0.8 | GPT loss < 1.0 |
