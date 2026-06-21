from dataclasses import dataclass


@dataclass
class ControlCommand:
    steer: float = 0
    acceleration: float = 0


# Backward-compatible alias (typo in original name).
ControlComand = ControlCommand
