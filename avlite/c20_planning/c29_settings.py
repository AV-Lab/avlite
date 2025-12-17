

class PlanningSettings:
    exclude = ["exclude"]
    filepath = "configs/c20_planning.yaml"
    
    # Race Planner Setting
    num_of_edge_points = 10
    planning_horizon = 3
    maneuver_distance = 30
    boundary_clearance = 0.5
    sample_size = 3  # number of nodes to sample in each level
    match_speed_wp_buffer = 4  # num of waypoints apart from a blocking agent
    
    # Replan stability: wait time (seconds) before switching to a better trajectory
    # Only switches immediately if current trajectory has collision risk
    replan_wait_time = 2.5
    
    # Safety: weight for preferring edges with more clearance from obstacles
    # 0.0 = only prefer closest to reference, 1.0 = strongly prefer safer edges
    safety_margin_weight = 0.3
    
    # Edge progress threshold: block switching if more than this fraction through current edge
    # 0.2 = block if >20% through edge, 0.5 = block if >50% through
    min_edge_progress_to_block = 0.2
    
    # Urgent collision threshold: switch immediately if collision within this many waypoints
    urgent_collision_threshold = 3
    
    # Collision detection settings
    collision_safety_margin = 0.3  # meters added to vehicle width for collision corridor
    min_velocity_threshold = 0.5  # m/s - agents slower than this treated as static
    default_ego_velocity = 5.0  # m/s - default velocity when ego velocity is 0 or unknown
    
    # Emergency braking settings
    stopping_decel_factor = 0.8  # fraction of max decel to use for stopping calculation
    fallback_deceleration = 3.0  # m/s^2 - fallback decel if vehicle max is too low
    stopping_safety_buffer = 2.0  # meters - safety buffer before collision point
    
    # Curvature settings (velocity-dependent)
    # max_curvature = max_lateral_accel / velocity^2
    max_lateral_accel = 4.0  # m/s^2 - maximum comfortable lateral acceleration
    min_curvature_velocity = 3.0  # m/s - minimum velocity for curvature calculation (avoids division issues)
