

class PlanningSettings:
    exclude = []
    filepath: str="configs/c20_planning.yaml"
    
    # Race Planner Setting
    num_of_edge_points=10
    planning_horizon=3
    maneuver_distance=35
    boundary_clearance=1
    sample_size=3 # number of nodes to sample in each level
