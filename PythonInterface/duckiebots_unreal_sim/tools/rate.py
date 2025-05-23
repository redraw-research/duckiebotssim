import time
from collections import deque

import numpy as np


class Rate:

    def __init__(self, target_hz: float, metrics_averaging_window: int = 100):
        self._target_seconds_per_frame = 1.0 / target_hz
        self._previous_frame_end = time.time()

        self._recent_frame_times = deque(maxlen=metrics_averaging_window)

    def sleep(self):
        delta = time.time() - self._previous_frame_end
        if delta < self._target_seconds_per_frame:
            time.sleep(self._target_seconds_per_frame - delta)
        frame_end = time.time()
        self._recent_frame_times.appendleft(frame_end - self._previous_frame_end)
        self._previous_frame_end = frame_end

    def report_avg_hz(self) -> float:
        return float(1.0 / np.mean(self._recent_frame_times))

    def clear_metrics(self):
        self._recent_frame_times.clear()
        self._previous_frame_end = time.time()

