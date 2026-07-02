from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ControlCommandBase:
    """Base for all control commands. Subclasses add actuation fields."""
    timestamp: float = field(default_factory=time.time)


@dataclass
class AckermannControlCommand(ControlCommandBase):
    steer: float = 0
    acceleration: float = 0


@dataclass
class DiffDriveControlCommand(ControlCommandBase):
    linear: float = 0
    angular: float = 0


@dataclass
class BodyVelocityControlCommand(ControlCommandBase):
    vx: float = 0
    vy: float = 0
    vz: float = 0
    yaw_rate: float = 0


# Backward compatibility — existing code keeps working unchanged
ControlCommand = AckermannControlCommand
ControlComand = AckermannControlCommand
