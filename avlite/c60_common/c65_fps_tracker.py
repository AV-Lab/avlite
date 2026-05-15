import time


class FpsTracker:
    """Tracks instantaneous FPS for a repeating step.

    Call tick() on each execution.  Always measures wall-clock elapsed time.

    Pass ``floor_dt`` to enforce a minimum inter-frame interval (e.g. pass
    ``sim_dt`` so that FPS is capped at ``1/sim_dt`` when the system runs
    faster than real-time, but falls to the true measured rate when the system
    is too slow to keep up).

    Effective FPS = 1 / max(wall_clock_dt, floor_dt)

    Read .last to get the wall-clock timestamp of the most recent tick (useful
    as a rate-limiting gate without a separate variable).
    """
    __slots__ = ('last', 'smoothed_dt', 'alpha')

    def __init__(self, alpha: float = 0.1):
        self.last: float = 0.0
        self.smoothed_dt: float = 0.0
        self.alpha: float = alpha

    def tick(self, floor_dt: float = 0.0) -> float:
        now = time.time()
        if self.last <= 0.0:
            self.last = now
            return 0.0
        raw_dt = now - self.last
        self.last = now
        effective_dt = max(raw_dt, floor_dt) if floor_dt > 0.0 else raw_dt
        if self.smoothed_dt == 0.0:
            self.smoothed_dt = effective_dt  # seed on first real measurement
        else:
            self.smoothed_dt = self.alpha * effective_dt + (1.0 - self.alpha) * self.smoothed_dt
        return 1.0 / self.smoothed_dt

    def reset(self):
        self.last = 0.0
        self.smoothed_dt = 0.0
