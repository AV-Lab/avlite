from typing import ClassVar

from pydantic import Field

from avlite.c60_apps.c64_settings_schema import SettingsSchema


class PlanningSettingsSchema(SettingsSchema):
    filepath: ClassVar[str] = "configs/c20_planning.yaml"

    c26_stopping_decel_factor: float = Field(default=0.8, description="Deceleration factor when stopping (velocity planner).")
    c26_fallback_deceleration: float = Field(default=3.0, description="Fallback deceleration (m/s²) for velocity planner.")
    c26_stopping_safety_buffer: float = Field(default=2.0, description="Safety buffer distance when stopping (m) for velocity planner.")
    c26_follow_gap_buffer: float = Field(default=0.5, description="Extra standoff beyond bumper lengths when following (m).")
    c26_follow_cruise_min_gap: float = Field(default=15.0, description="Extra gap beyond stopping distance before deferring to global plan speed (m).")
    c26_planning_horizon_points: int = Field(default=50, description="Max waypoints in velocity local plan window from current_wp.")

    c27_num_of_edge_points: int = Field(default=10, description="Number of points sampled along each lattice edge.")
    c27_planning_horizon: int = Field(default=3, description="Local planning horizon in seconds.")
    c27_maneuver_distance: int = Field(default=30, description="Longitudinal distance (m) for maneuver sampling.")
    c27_boundary_clearance: float = Field(default=0.5, description="Minimum clearance from race boundary (m).")
    c27_sample_size: int = Field(default=3, description="Number of lateral samples per maneuver.")
    c27_match_speed_wp_buffer: int = Field(default=4, description="Waypoint buffer for speed matching.")
    c27_replan_wait_time: float = Field(default=2.5, description="Minimum time (s) between replans.")
    c27_safety_margin_weight: float = Field(default=0.3, description="Weight for collision safety margin in cost.")
    c27_min_edge_progress_to_block: float = Field(default=0.2, description="Min edge progress before blocking replan.")
    c27_urgent_collision_threshold: int = Field(default=3, description="Frames until urgent collision response.")
    c27_disconnect_distance_threshold: float = Field(default=5.0, description="Max distance (m) before path disconnect.")
    c27_max_lateral_accel: float = Field(default=4.0, description="Max lateral acceleration for velocity profiling (m/s²).")
    c27_min_curvature_velocity: float = Field(default=3.0, description="Minimum velocity on high-curvature segments (m/s).")
    c27_d0_reference_threshold: float = Field(default=0.2, description="Frenet d₀ reference threshold (m).")
    c27_min_ramp_start_velocity: float = Field(default=3.0, description="Minimum ramp start velocity (m/s).")
    c27_allow_curvature_fallback: bool = Field(default=False, description="Allow fallback when curvature limits block plan.")
    c27_allow_boundary_violation_fallback: bool = Field(default=False, description="Allow fallback on boundary violation.")

    c20_boundary_margin: float = Field(
        default=0.0,
        description="Inset applied to global plan boundaries from race boundary or HD map lane borders (m).",
    )
    c20_collision_safety_margin: float = Field(default=0.3, description="Inflation margin for collision checks (m).")
    c20_obstacle_inflation_margin: float = Field(default=0.5, description="Obstacle inflation for lattice planning (m).")
    c20_min_velocity_threshold: float = Field(default=0.5, description="Speed below which ego is treated as stopped (m/s).")
    c20_default_ego_velocity: float = Field(default=5.0, description="Default ego velocity when unknown (m/s).")


# Singleton instance: mutated in place by the loader/reset helpers — never rebind.
PlanningSettings = PlanningSettingsSchema()
