

class PlanningSettings:
    exclude = ["exclude"]
    filepath = "configs/c20_planning.yaml"

    # c26 lattice planner
    c26_num_of_edge_points = 10
    c26_planning_horizon = 3
    c26_maneuver_distance = 30
    c26_boundary_clearance = 0.5
    c26_sample_size = 3
    c26_match_speed_wp_buffer = 4
    c26_replan_wait_time = 2.5
    c26_safety_margin_weight = 0.3
    c26_min_edge_progress_to_block = 0.2
    c26_urgent_collision_threshold = 3
    c26_disconnect_distance_threshold = 5.0
    c26_max_lateral_accel = 4.0
    c26_min_curvature_velocity = 3.0
    c26_d0_reference_threshold = 0.2
    c26_min_ramp_start_velocity = 3.0
    c26_allow_curvature_fallback = False
    c26_allow_boundary_violation_fallback = False
    c26_stopping_decel_factor = 0.8
    c26_fallback_deceleration = 3.0
    c26_stopping_safety_buffer = 2.0

    # c20 shared (c26 + c27)
    c20_collision_safety_margin = 0.3
    c20_obstacle_inflation_margin = 0.5
    c20_min_velocity_threshold = 0.5
    c20_default_ego_velocity = 5.0
