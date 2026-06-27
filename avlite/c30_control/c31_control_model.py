from dataclasses import dataclass


@dataclass
class ControlCommand:
    steer: float = 0
    acceleration: float = 0


# An alias for ControlCommand to maintain backward compatibility with older code that may use the previous name.
ControlComand = ControlCommand 
