from c30_control.c32_base_controller import BaseController


class StanleyController(BaseController):
    def __init__(self, tj: Trajectory = None, k: float = 0.5, kv: float = 0.5):
        super().__init__(tj)
