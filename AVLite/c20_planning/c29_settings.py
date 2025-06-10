

class PlanningSettings:
    exclude = []
    filepath: str="configs/c20_planning.yaml"
    
    # Race Planner Setting
    num_of_edge_points=10
    planning_horizon=3
    maneuver_distance=30
    boundary_clearance0.5
    sample_size=3 # number of nodes to sample in each level
    match_speed_wp_buffer=2 # num of waypoints apart from a blocking agent
