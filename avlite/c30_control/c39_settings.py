from typing import ClassVar

import numpy as np
from pydantic import Field

from avlite.c60_common.c68_settings_schema import SettingsSchema


class ControlSettingsSchema(SettingsSchema):
    filepath: ClassVar[str] = "configs/c30_control.yaml"

    c33_pid_alpha: float = Field(default=0.01, description="PID cross-track error gain.")
    c33_pid_beta: float = Field(default=0.001, description="PID cross-track integral gain.")
    c33_pid_gamma: float = Field(default=0.6, description="PID heading error gain.")
    c33_pid_valpha: float = Field(default=0.8, description="PID velocity proportional gain.")
    c33_pid_vbeta: float = Field(default=0.01, description="PID velocity integral gain.")
    c33_pid_vgamma: float = Field(default=0.3, description="PID velocity derivative gain.")
    c33_pid_lookahead: int = Field(default=2, description="PID lookahead waypoint index.")

    c34_stanley_k: float = Field(default=5, description="Stanley controller cross-track gain.")
    c34_stanley_k_soft: float = Field(default=0.01, description="Stanley softening factor at low speed.")
    c34_stanley_lookahead: int = Field(default=5, description="Stanley lookahead waypoint index.")
    c34_stanley_valpha: float = Field(default=0.8, description="Stanley velocity proportional gain.")
    c34_stanley_vbeta: float = Field(default=0.01, description="Stanley velocity integral gain.")
    c34_stanley_vgamma: float = Field(default=0.3, description="Stanley velocity derivative gain.")
    c34_stanley_slow_down_cte: float = Field(default=0.5, description="Cross-track error threshold to slow down (m).")
    c34_stanley_slow_down_heading_cte: float = Field(
        default=float(np.pi / 6), description="Heading error threshold to slow down (rad)."
    )
    c34_stanley_slow_down_vel_threshold: float = Field(default=3, description="Speed threshold for slow-down logic (m/s).")

    c30_emergency_velocity_threshold: float = Field(default=0.5, description="Speed threshold for emergency braking (m/s).")
    c30_emergency_min_moving_velocity: float = Field(default=1.0, description="Min speed treated as moving for emergency logic (m/s).")
    c30_emergency_braking_factor: float = Field(default=0.9, description="Emergency braking deceleration factor.")

    c32_ego_distance_front_axle: float = Field(default=2.5, description="Distance from rear axle to front axle (m).")
    c32_ego_max_velocity: float = Field(default=99.0, description="Maximum ego velocity (m/s).")
    c32_ego_max_acceleration: float = Field(default=10.0, description="Maximum ego acceleration (m/s²).")
    c32_ego_min_acceleration: float = Field(default=-20.0, description="Minimum ego acceleration / max decel (m/s²).")
    c32_ego_max_steering: float = Field(default=0.7, description="Maximum steering angle (rad).")
    c32_ego_min_steering: float = Field(default=-0.7, description="Minimum steering angle (rad).")


# Singleton instance: mutated in place by the loader/reset helpers — never rebind.
ControlSettings = ControlSettingsSchema()
