from typing import ClassVar

from pydantic import Field

from avlite.c60_apps.c64_settings_schema import SettingsSchema


class PlanningSettingsSchema(SettingsSchema):
    filepath: ClassVar[str] = "configs/c20_planning.yaml"

    c23_behavioral_strategy: str = Field(default="", description="LocalPlanningPipeline behavioral stage class name (empty = skip).")
    c23_path_strategy: str = Field(default="GreedyLatticePlanner", description="LocalPlanningPipeline path stage class name (empty = skip).")
    c23_velocity_strategy: str = Field(default="", description="LocalPlanningPipeline velocity stage class name (empty = skip).")

    c27_max_deceleration: float = Field(default=3.0, description="Max deceleration magnitude (m/s²) used by the velocity planner for speed profiling.")
    c27_stopping_safety_buffer: float = Field(default=2.0, description="Safety buffer distance when stopping (m) for velocity planner.")
    c27_follow_gap_buffer: float = Field(default=0.5, description="Extra standoff beyond bumper lengths when following (m).")
    c27_follow_cruise_min_gap: float = Field(default=15.0, description="Extra gap beyond stopping distance before deferring to global plan speed (m).")
    c27_follow_gap_gain: float = Field(default=1.0, description="Proportional gain (1/s) mapping follow-gap deficit to speed reduction below the lead when inside the safe gap.")
    c27_planning_horizon_points: int = Field(default=50, description="Max waypoints in velocity local plan window from current_wp.")

    c28_num_of_edge_points: int = Field(default=10, description="Number of points sampled along each lattice edge.")
    c28_planning_horizon: int = Field(default=3, description="Local planning horizon in seconds.")
    c28_maneuver_distance: int = Field(default=30, description="Longitudinal distance (m) for maneuver sampling.")
    c28_boundary_clearance: float = Field(default=0.5, description="Minimum clearance from race boundary (m).")
    c28_sample_size: int = Field(default=3, description="Number of lateral samples per maneuver.")
    c28_match_speed_wp_buffer: int = Field(default=4, description="Waypoint buffer for speed matching.")
    c28_replan_wait_time: float = Field(default=2.5, description="Minimum time (s) between replans.")
    c28_safety_margin_weight: float = Field(default=0.3, description="Weight for collision safety margin in cost.")
    c28_min_edge_progress_to_block: float = Field(default=0.2, description="Min edge progress before blocking replan.")
    c28_urgent_collision_threshold: float = Field(default=10.0, description="Distance to collision (m) below which the planner switches plans immediately, bypassing the replan wait time.")
    c28_disconnect_distance_threshold: float = Field(default=5.0, description="Max distance (m) before path disconnect.")
    c28_kinematic_sampling: bool = Field(default=True, description="Bound lattice lateral sampling to the speed-dependent kinematic reach (a_lat, curvature) instead of sampling the full road width.")
    c28_sample_reach_factor: float = Field(default=1.0, description="Multiplier on the kinematic lateral reach used for sampling; <1 adds curvature-feasibility headroom, >1 widens exploration.")
    c28_sample_distribution: int = Field(default=1, description="Lateral sample placement within the reach band: 0=even spread (interior bin-centers), 1=random uniform, 2=stratified (even bins + jitter).")
    c28_max_lateral_accel: float = Field(default=4.0, description="Max lateral acceleration for velocity profiling (m/s²).")
    c28_min_curvature_velocity: float = Field(default=3.0, description="Minimum velocity on high-curvature segments (m/s).")
    c28_d0_reference_threshold: float = Field(default=0.2, description="Frenet d₀ reference threshold (m).")
    c28_allow_curvature_fallback: bool = Field(default=False, description="Allow fallback when curvature limits block plan.")
    c28_allow_boundary_violation_fallback: bool = Field(default=False, description="Allow fallback on boundary violation.")

    c20_boundary_margin: float = Field(default=0.25, description="Inset applied to global plan boundaries from race boundary or HD map lane borders (m).",)
    c20_collision_safety_margin: float = Field(
        default=0.5,
        description="Extra clearance added to the ego side of collision checks (m). Expands the buffered trajectory corridor beyond half the ego width before intersecting obstacles.",
    )
    c20_obstacle_inflation_margin: float = Field(
        default=0.5,
        description="Extra clearance added around agent obstacle polygons in collision checks (m). Inflates each agent bbox (or prediction sweep) before intersecting the ego corridor; used by lattice and velocity planners.",
    )
    c20_min_velocity_threshold: float = Field(
        default=0.5,
        description="Speed gate for agent motion in collision checks (m/s). Agents at or below this |v| are treated as static obstacles; pm.prediction trajectories are not used when building swept obstacle polygons (ahead agents only).",
    )
    c20_beside_agent_sweep_time: float = Field(default=2.0, description="Forward sweep time (s) applied to moving agents beside/just-behind the ego so the lattice stays clear of a just-passed agent before returning to the reference line (0 disables).")
    c20_beside_agent_rear_window: float = Field(default=6.0, description="Max distance behind the ego (m) for which a beside/just-behind moving agent is still swept forward (overtake cut-back protection). Agents further back are treated as static; 0 disables the beside-sweep for behind agents.")
    c20_default_ego_velocity: float = Field(default=5.0, description="Default ego velocity when unknown (m/s).")
    c20_min_ramp_start_velocity: float = Field(default=3.0, description="Minimum ramp start velocity (m/s); shared by global plan loading and the lattice planner.")


# Singleton instance: mutated in place by the loader/reset helpers — never rebind.
PlanningSettings = PlanningSettingsSchema()
