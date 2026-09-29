"""LeRobot 0.4.3 RTC adapter; only the control thread consumes/merges the queue.

The worker owns preprocessing, policy calls and postprocessing. Queue tensors
stay on CPU so consuming an action never synchronizes with GPU inference.
"""

import math
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version

import torch

from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker


def check_lerobot_version():
    installed = version("lerobot")
    if installed != "0.4.3":
        raise RuntimeError(f"RTC requires lerobot==0.4.3; installed: {installed}")
    return installed


class RTCInference:
    """Overlap full-chunk prediction with execution using native RTC guidance.

    n_action_steps sets the target interval between replans; the entire
    chunk_size prediction is retained as a reserve during inference. Replan
    earlier when the reserve reaches the measured delay plus a safety margin.
    Call submit/poll/pop/close from one control thread only.
    """

    def __init__(self, policy, preprocessor, postprocessor, *, freq,
                 delay_ms=125.0, margin_steps=2, execution_horizon=10):
        check_lerobot_version()
        if not math.isfinite(freq) or freq <= 0:
            raise ValueError("freq must be positive and finite")
        if not math.isfinite(delay_ms) or delay_ms < 0 or margin_steps < 1:
            raise ValueError("delay_ms must be nonnegative; margin_steps must be >= 1")
        self.chunk_size = policy.config.chunk_size
        if not 1 <= execution_horizon <= self.chunk_size:
            raise ValueError("RTC execution_horizon must be in [1, chunk_size]")
        self.freq = freq
        self.initial_delay = delay_ms / 1000
        self.margin_steps = margin_steps
        self.replan_steps = policy.config.n_action_steps
        self.latency = LatencyTracker(maxlen=100)
        self.cfg = RTCConfig(enabled=True, execution_horizon=execution_horizon)
        policy.config.rtc_config = self.cfg
        policy.init_rtc_processor()
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.queue = ActionQueue(self.cfg)
        self._future = None
        self._closed = False
        self.chunk_id = 0
        self.last_latency = 0.0
        self.last_delay = 0
        self._validate_reserve()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lerobot-rtc")

    @property
    def delay_steps(self):
        # Include preprocessing, transfers and postprocessing, not just CUDA launch time.
        seconds = max(self.initial_delay, self.latency.max() or 0.0)
        return math.ceil(seconds * self.freq)

    def _validate_reserve(self):
        if self.delay_steps + self.margin_steps >= self.chunk_size:
            raise RuntimeError(
                f"RTC latency budget ({self.delay_steps} steps + "
                f"{self.margin_steps} margin) does not fit chunk_size={self.chunk_size}. "
                "Reduce control frequency or inference latency."
            )

    def needs_observation(self):
        if self._closed or self._future is not None:
            return False
        return (self.chunk_id == 0
                or self.queue.get_action_index() + self.last_delay >= self.replan_steps
                or self.queue.qsize() <= self.delay_steps + self.margin_steps)

    def submit(self, observation, *, observation_time=None):
        if self._closed or self._future is not None:
            raise RuntimeError("RTC worker is closed or already predicting")
        self._index_before = self.queue.get_action_index()
        self._submitted_at = observation_time if observation_time is not None else time.perf_counter()
        leftover = self.queue.get_left_over()
        # All queue access is on this thread. The worker gets a stable CPU snapshot.
        if leftover is not None:
            leftover = leftover.clone()
        self._future = self._executor.submit(self._predict, observation, leftover, self.delay_steps)

    def _predict(self, observation, leftover, delay):
        with torch.no_grad():  # RTC internally enables gradients; do not use inference_mode.
            batch = self.preprocessor(observation)
            if leftover is not None:
                leftover = leftover.to(next(self.policy.parameters()).device)
            chunk = self.policy.predict_action_chunk(
                batch, prev_chunk_left_over=leftover, inference_delay=delay,
                execution_horizon=min(self.chunk_size, max(self.cfg.execution_horizon,
                                                          delay + self.margin_steps)),
            )
            if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[1] != self.chunk_size:
                raise RuntimeError(f"Unexpected RTC action shape: {tuple(chunk.shape)}")
            # Clone before postprocessing in case a processor modifies its input in place.
            original = chunk[0].detach().to("cpu").clone()
            processed = self.postprocessor({'action': chunk})['action'][0].detach().to("cpu")
            if processed.shape != original.shape or not torch.isfinite(processed).all():
                raise RuntimeError("Invalid RTC postprocessed actions")
            return original, processed

    def poll(self, *, wait=False):
        """Merge a completed chunk without waiting (except the first warm-up chunk)."""
        if self._future is None:
            return False
        if wait and self.chunk_id:
            raise RuntimeError("Only RTC startup may wait for inference")
        if not wait and not self._future.done():
            return False
        original, processed = self._future.result()  # Propagate worker failures to robot cleanup.
        self._future = None
        elapsed = time.perf_counter() - self._submitted_at
        # Exact executed steps, not rounded wall time. Poll and merge cannot race with pop.
        real_delay = self.queue.get_action_index() - self._index_before
        if self.chunk_id:
            self.latency.add(elapsed)
            self._validate_reserve()
        self.queue.merge(original, processed, real_delay, self._index_before)
        self.chunk_id += 1
        self.last_latency = elapsed
        self.last_delay = real_delay
        return True

    def pop(self):
        action = self.queue.get()
        if action is None:
            # Never replay stale targets or silently block the control loop on inference.
            raise RuntimeError("RTC action queue underrun: inference exceeded the action reserve")
        return action.numpy()

    def close(self):
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
