"""VelocityLocalPlanner must survive a replan at the final global waypoint."""

from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c20_planning.c27_local_behavioral_and_velocity_planners import VelocityLocalPlanner
from avlite.c50_common.c54_trajectory_tracker import TrajectoryTracker


def test_velocity_local_planner_replan_at_final_waypoint():
    path = [(float(i), 0.0) for i in range(20)]
    velocity = [8.0] * 20
    tj = TrajectoryTracker(path=path, velocity=velocity)
    ego = EgoState(x=19.0, y=0.0, theta=0.0, velocity=5.0)
    gp = GlobalPlan(
        trajectory=tj,
        path=path,
        velocity=velocity,
        start_point=(0.0, 0.0),
        goal_point=(19.0, 0.0),
    )
    pm = PerceptionModel(ego_vehicle=ego)
    planner = VelocityLocalPlanner(global_plan=gp, env=pm)
    planner.global_trajectory.update_waypoint_by_xy(19.0, 0.0)
    assert planner.global_trajectory.current_wp == len(path) - 1

    planner.replan(perception_model=pm)
    local = planner.get_local_plan()
    assert local is not None
    assert local.trajectory is not None
    assert len(local.trajectory.path) == 1
