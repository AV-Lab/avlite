from typing import ClassVar

import numpy as np
from pydantic import Field

from avlite.c60_apps.c64_settings_schema import SettingsSchema


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

    c35_lookahead_distance: float = Field(
        default=8.0,
        description=(
            "Nominal Pure Pursuit lookahead distance in metres. Used as the fixed Ld when "
            "c35_lookahead_speed_gain is 0. Larger values smooth steering but cut corners; "
            "smaller values track tighter but can oscillate."
        ),
    )
    c35_min_lookahead: float = Field(
        default=3.0,
        description=(
            "Lower clamp (m) for speed-adaptive lookahead when c35_lookahead_speed_gain > 0. "
            "Prevents Ld from collapsing at low speed."
        ),
    )
    c35_max_lookahead: float = Field(
        default=20.0,
        description=(
            "Upper clamp (m) for speed-adaptive lookahead when c35_lookahead_speed_gain > 0. "
            "Caps how far ahead the controller aims at high speed."
        ),
    )
    c35_lookahead_speed_gain: float = Field(
        default=0.0,
        description=(
            "If > 0, sets Ld = clip(gain * ego_speed, c35_min_lookahead, c35_max_lookahead). "
            "If 0, uses the fixed c35_lookahead_distance instead."
        ),
    )
    c35_valpha: float = Field(
        default=0.8,
        description="Pure Pursuit velocity PID proportional gain (same role as Stanley/PID valpha).",
    )
    c35_vbeta: float = Field(
        default=0.01,
        description="Pure Pursuit velocity PID integral gain (same role as Stanley/PID vbeta).",
    )
    c35_vgamma: float = Field(
        default=0.3,
        description="Pure Pursuit velocity PID derivative gain (same role as Stanley/PID vgamma).",
    )
    c35_cruise_velocity: float = Field(
        default=5.0,
        description=(
            "Target speed (m/s) for FollowTheGapController when no plan/trajectory is "
            "available. Ignored when a trajectory supplies a waypoint velocity."
        ),
    )
    c35_lidar_z_min: float = Field(
        default=-1.5,
        description=(
            "Minimum height (m) kept when squashing a 3D LiDAR cloud to 2D for "
            "FollowTheGapController."
        ),
    )
    c35_lidar_z_max: float = Field(
        default=2.0,
        description=(
            "Maximum height (m) kept when squashing a 3D LiDAR cloud to 2D for "
            "FollowTheGapController."
        ),
    )
    c35_bubble_radius: float = Field(
        default=1.0,
        description=(
            "Safety bubble (m) for FollowTheGapController: ego-frame LiDAR hits "
            "closer than this are dropped before gap finding."
        ),
    )
    c35_min_gap_width: float = Field(
        default=0.2,
        description=(
            "Minimum angular gap width (rad) considered when path-biased gap "
            "selection is active in FollowTheGapController."
        ),
    )

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
