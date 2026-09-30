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

## Concurrency and reset audit

The current inference process has four relevant execution contexts:

1. The main control loop: reads observations, polls RTC, pops an action, sends CANFD commands, and handles the reset event.
2. The reset listener thread: in headless mode, puts stdin in cbreak mode and sets `reset_requested` when it reads `r`; it never moves the arm or clears queues itself.
3. The RTC worker thread: owns preprocessing, Pi0.5 inference, and postprocessing through a single `ThreadPoolExecutor` worker. It does not call the robot SDK.
4. The RealMan state-cache thread: performs the only `rm_get_current_arm_state()` calls and publishes the newest six-joint snapshot under `_state_lock`.

### What reset currently does

The main loop checks `reset_requested` only at the beginning of an iteration. It then:

```text
robot.stop()
robot.move_to_init(INIT_POSE)
if RTC:
    rtc.reset()
    # rtc.reset waits for the old Future, then sets ActionQueue.queue/original_queue=None
else:
    policy.reset()
    action_queue = _get_action_queue(policy)
reset counters
```

For serial mode, `policy.reset()` creates new deques and the local `action_queue` is reacquired, so the old policy queue is discarded.

For RTC mode, `rtc.reset()` does clear both RTC queues (`queue` and `original_queue`) and resets `last_index`, `chunk_id`, delay bookkeeping, and policy state. However, it first calls `_future.result()`. Therefore queue clearing is delayed until the previous Pi0.5 inference finishes; it is not cancellation. The queue is not merged by another thread because `poll()`/`pop()`/`reset()` are called by the main thread only.

### Fixed reset and safety behavior

The current project implementation now applies these protections:

- `rtc.reset()` clears `ActionQueue.queue`, `original_queue`, and `last_index` under the queue lock after the old Future has completed.
- The main loop checks `reset_requested` after RTC polling and again after preprocessing/inference, before `rtc.pop()`/`robot.set_qpos()`. An action computed across a reset boundary is discarded.
- Reset stops and joins the RealMan state-cache thread before `rm_set_arm_stop()` and `rm_movej_p()`, then starts a fresh state-cache thread after homing.
- State-cache shutdown waits for the SDK call to return before the arm handle can be deleted. This is intentionally a blocking safety barrier during reset/shutdown only; the normal control loop never waits for the state RPC.
- A reset generation counter is printed in the `[RESET N]` messages so logs distinguish old and new episodes.

### Remaining timing semantics and safety implications

- If `r` arrives after the final pre-send check but during the native `robot.set_qpos()` call, that one command cannot be preempted by Python; the next loop handles reset. This is the unavoidable non-preemptible SDK call boundary.
- If `r` arrives while the RTC worker is computing, the main thread stops/homes the arm and then waits in `rtc.reset()` for the stale Future. No stale action should be merged after `reset()` because the main thread owns `poll()` and `merge()`, but reset latency can be as long as the Pi0.5 inference latency.
- Repeated `r` presses are coalesced by `threading.Event`; they do not queue multiple resets. This is desirable, but a reset-generation counter is needed if future code must distinguish events that arrive during a reset.
- During normal operation, the state-cache RPC and motion calls remain separate SDK calls from different threads. The reset/shutdown barrier prevents the especially dangerous stop/homing/delete races; if the vendor SDK requires all calls to be serialized, add a vendor-safe SDK I/O lock as a separate change and measure its effect on control latency.

### Required design for a safe future reset fix

Do not fix reset by merely assigning `action_queue = None`. Preserve these invariants:

- A reset must invalidate the current RTC generation before any next action can be sent.
- The control loop must check the reset generation immediately before `rtc.pop()`/`robot.set_qpos()` and discard an action if reset was requested during polling.
- RTC reset must stop accepting/merging the old Future result, then clear both processed and original queues atomically from the control thread.
- The state-cache thread and every arm SDK command need one shared arm-I/O synchronization policy. If SDK calls can block, shutdown must not destroy the SDK handle until the state call has exited; a timed daemon join is insufficient.
- During reset, state refresh should be invalidated or marked stale, the arm should be stopped/homed, then a fresh state sample should be obtained before submitting the next observation.
- Add a reset generation/id to logs so stale actions and post-reset actions can be distinguished.

When modifying reset or state-cache code, test at least these cases with fake SDK/policy objects before connecting hardware: reset while no RTC Future exists; reset while the RTC Future is running; repeated `r`; state RPC failure with a valid cache; state RPC blocked during reset; and shutdown while a state RPC is blocked.
