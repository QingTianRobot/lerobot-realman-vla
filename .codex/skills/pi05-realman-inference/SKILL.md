---
name: pi05-realman-inference
description: Run and troubleshoot local LeRobot 0.4.3 Pi0.5 RealMan checkpoints, especially offline inference, cross-machine tokenizer paths, and the required Transformers compatibility branch.
---

# Pi0.5 RealMan inference

Use this skill when a user asks how to run, validate, or troubleshoot a locally trained Pi0.5 checkpoint for this RealMan repository. The normal target is `scripts/inference.py`; do not edit that script without discussing the proposed change with the user first.

## Invariants

- Keep the repository virtual environment on `lerobot==0.4.3`.
- Pi0.5 in this LeRobot release needs the Transformers OpenPI compatibility branch, not an arbitrary PyPI Transformers build:

  ```bash
  uv pip install --python .venv/bin/python --reinstall \
    'transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi'
  ```

- Prefer local assets and offline mode. Expected local assets are:
  - `models/pi05_base/` (training/base-model asset; not downloaded again for a complete checkpoint)
  - `models/paligemma-3b-pt-224/` (tokenizer)
- A checkpoint's `model.safetensors` is the inference weight file. Its `config.json` may retain `pretrained_path: models/pi05_base` as training metadata; do not assume that field means inference must download the base model.

## Workflow

1. Inspect the checkpoint. Confirm `pretrained_model/config.json` says `"type": "pi05"`, `model.safetensors` exists, and the policy pre/postprocessor JSON plus their safetensors files exist.
2. Activate the repository environment with `source ./env.sh`. This also puts the Orbbec SDK library directory first in `LD_LIBRARY_PATH`.
3. Check versions: LeRobot must be `0.4.3`, CUDA should be available, and `from transformers.models.siglip import check` must work. If `transformers.models.siglip.check` is missing, install the compatibility branch above before testing the model.
4. Check each checkpoint's `policy_preprocessor.json` for `tokenizer_name`. Checkpoint files copied from another machine may contain an absolute path such as `/home/xiaozhang/.../models/paligemma-3b-pt-224`. If it does not exist, update only `tokenizer_name` to the current repository's `models/paligemma-3b-pt-224`; do this for every checkpoint being tested because each checkpoint stores its own copy. Do not change `scripts/inference.py` for this problem.
5. Validate offline before touching hardware:

   ```bash
   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python - <<'PY'
   from lerobot.policies.pi05.modeling_pi05 import PI05Policy
   from lerobot.processor.pipeline import DataProcessorPipeline
   p = "outputs/pi05_realman_sztu/checkpoints/005000/pretrained_model"
   policy = PI05Policy.from_pretrained(p, local_files_only=True)
   pre = DataProcessorPipeline.from_pretrained(p, config_filename="policy_preprocessor.json")
   post = DataProcessorPipeline.from_pretrained(p, config_filename="policy_postprocessor.json")
   print("OK", type(policy).__name__, len(pre.steps), len(post.steps))
   PY
   ```

   For another checkpoint, change only `p`. A successful weight load followed by a tokenizer `HFValidationError` means the checkpoint's preprocessor path is stale, not that the model weights are broken.
6. Run the real robot only after the offline check succeeds:

   ```bash
   cd /home/robot/repo/lerobot-realman-vla
   source ./env.sh
   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
   .venv/bin/python scripts/inference.py \
     --model /home/robot/repo/lerobot-realman-vla/outputs/pi05_realman_sztu/checkpoints/005000/pretrained_model \
     --task "pick up the plug and place it in the box" \
     --freq 30 --ema-alpha 0.7 --deadzone 0 --offline --headless
   ```

   Replace `005000` with the requested checkpoint. The current hardware defaults are the RM65 at `192.168.5.123:8080`, RealSense serial `262322074840`, Orbbec serial `CV2T66100096`, and gripper port `/dev/realman/gripper_left`; verify the setup before execution because `--headless` still moves the robot.

## Interpreting common output

- `✓ Loaded state dict from model.safetensors` means the checkpoint weights loaded.
- The Pi0.5 loader may warn about vision embedding keys and a missing language embedding key while still loading successfully; verify with the offline pipeline/forward smoke test rather than treating that warning alone as fatal.
- `Repo id must be in the form ... /home/.../paligemma-3b-pt-224` means `tokenizer_name` points to a nonexistent absolute path. Repair the checkpoint's `policy_preprocessor.json`.
- `An incorrect transformer version is used` or missing `transformers.models.siglip.check` means the OpenPI compatibility branch is not installed in the active venv.
- If one checkpoint works and another fails at tokenizer initialization, compare their `policy_preprocessor.json`; each checkpoint stores its own copy.
