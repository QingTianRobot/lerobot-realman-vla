"""Hardware-free checks using LeRobot's installed RTC queue and guidance."""

import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rtc_inference import RTCInference, check_lerobot_version
from lerobot.policies.rtc.modeling_rtc import RTCProcessor


class FakePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(chunk_size=50, n_action_steps=20)
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.calls = []
        self.failure = None

    def init_rtc_processor(self):
        self.rtc_processor = RTCProcessor(self.config.rtc_config)

    def predict_action_chunk(self, batch, **kwargs):
        self.calls.append(kwargs)
        self.started.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError('test worker timed out')
        if self.failure:
            raise self.failure
        chunk = torch.arange(50, dtype=torch.float32).reshape(1, 50, 1)
        return chunk + (len(self.calls) - 1) * 100


class RTCInferenceTests(unittest.TestCase):
    def setUp(self):
        self.policy = FakePolicy()
        self.worker_threads = []

        def pre(obs):
            self.worker_threads.append(threading.get_ident())
            return obs

        def post(data):
            # Deliberately mutate: RTC must retain the normalized original separately.
            data['action'].mul_(10)
            return data

        self.rtc = RTCInference(self.policy, pre, post, freq=30)
        self.addCleanup(self.rtc.close)
        self.addCleanup(self.policy.release.set)

    def prime(self):
        self.rtc.submit({})
        self.assertTrue(self.rtc.poll(wait=True))

    def finish(self):
        deadline = time.monotonic() + 3
        while not self.rtc.poll():
            if time.monotonic() > deadline:
                self.fail('worker failed to complete')
            time.sleep(0.001)

    def test_startup_full_chunk_and_cpu_queue(self):
        self.assertTrue(self.rtc.needs_observation())
        self.prime()
        self.assertEqual(self.rtc.queue.qsize(), 50)  # Keep tail beyond n_action_steps.
        self.assertEqual(self.rtc.queue.queue.device.type, 'cpu')
        self.assertEqual(self.rtc.last_delay, 0)
        self.assertEqual(len(self.rtc.latency), 0)  # Cold startup is not steady latency.
        self.assertNotEqual(self.worker_threads[0], threading.get_ident())
        self.assertTrue(self.policy.config.rtc_config.enabled)

    def test_execution_continues_and_merge_skips_exact_consumption(self):
        self.prime()
        self.rtc.pop()
        self.rtc.pop()
        self.policy.started.clear()
        self.policy.release.clear()
        self.rtc.submit({})
        self.assertTrue(self.policy.started.wait(timeout=2))
        self.assertFalse(self.rtc.poll())  # Must return while prediction is blocked.
        for i in range(2, 8):
            self.assertEqual(self.rtc.pop().item(), i * 10)
        self.assertFalse(self.rtc.needs_observation())  # Only one request in flight.
        self.policy.release.set()
        self.finish()
        self.assertEqual(self.rtc.last_delay, 6)
        self.assertEqual(self.rtc.pop().item(), 1060)  # No duplicate stale prefix.
        torch.testing.assert_close(self.policy.calls[1]['prev_chunk_left_over'][:, 0],
                                   torch.arange(2, 50, dtype=torch.float32))
        self.assertEqual(self.policy.calls[1]['inference_delay'], 4)

    def test_replan_interval_and_early_reserve(self):
        self.prime()
        for _ in range(19):
            self.rtc.pop()
        self.assertFalse(self.rtc.needs_observation())
        self.rtc.pop()
        self.assertTrue(self.rtc.needs_observation())
        self.rtc.replan_steps = 50
        for _ in range(24):
            self.rtc.pop()
        self.assertEqual(self.rtc.queue.qsize(), 6)
        self.assertTrue(self.rtc.needs_observation())

    def test_measured_latency_increases_delay_budget(self):
        self.prime()
        self.rtc.submit({}, observation_time=time.perf_counter() - 0.25)
        self.finish()
        self.assertGreaterEqual(self.rtc.delay_steps, 8)

    def test_worker_error_is_propagated(self):
        self.prime()
        self.policy.failure = ValueError('inference failed')
        self.rtc.submit({})
        with self.assertRaisesRegex(ValueError, 'inference failed'):
            self.finish()

    def test_underrun_fails_instead_of_waiting_or_replaying(self):
        self.prime()
        for _ in range(50):
            self.rtc.pop()
        with self.assertRaisesRegex(RuntimeError, 'underrun'):
            self.rtc.pop()

    def test_impossible_budget_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'does not fit'):
            RTCInference(self.policy, lambda x: x, lambda x: x, freq=1000)

    def test_version_mismatch_is_rejected(self):
        with patch('rtc_inference.version', return_value='0.4.2'):
            with self.assertRaisesRegex(RuntimeError, 'installed: 0.4.2'):
                check_lerobot_version()

    def test_wait_is_forbidden_after_startup(self):
        self.prime()
        self.rtc.submit({})
        with self.assertRaisesRegex(RuntimeError, 'Only RTC startup'):
            self.rtc.poll(wait=True)

    def test_native_guidance_works_under_no_grad(self):
        # inference_mode would break the internal autograd used by native RTC.
        with torch.no_grad():
            result = self.policy.rtc_processor.denoise_step(
                x_t=torch.ones(1, 50, 1), prev_chunk_left_over=torch.zeros(10, 1),
                inference_delay=4, time=0.5,
                original_denoise_step_partial=lambda x: torch.zeros_like(x),
            )
        self.assertTrue(torch.isfinite(result).all())
        self.assertNotEqual(result[0, 0, 0].item(), 0)


if __name__ == '__main__':
    unittest.main()
