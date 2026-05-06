import time


class FpsTracker:
    """Tracks instantaneous FPS for a repeating step.

    Call tick() on each execution. Pass a timestamp to use a custom time
    domain (e.g. elapsed_sim_time for sim-rate FPS); omit it for wall-clock FPS.
    Read .last to get the timestamp of the most recent tick (useful as a
    rate-limiting gate without a separate variable).
    """
    __slots__ = ('last',)

    def __init__(self):
        self.last: float = 0.0

    def tick(self, t: float = None) -> float:
        now = time.time() if t is None else t
        fps = 1.0 / max(now - self.last, 1e-9) if self.last > 0.0 else 0.0
        self.last = now
        return fps

    def reset(self):
        self.last = 0.0
