import logging

import numpy as np
from shapely.geometry import LineString, Polygon

from avlite.c10_perception.c11_perception_model import PerceptionModel, AgentState
from avlite.c60_common.c63_trajectory_tracker import TrajectoryTracker
from avlite.c20_planning.c29_settings import PlanningSettings

log = logging.getLogger(__name__)


def check_collision(pm: PerceptionModel, trajectory: TrajectoryTracker, sample_size=5) -> tuple[bool, int, float]:
    """
    Check for collision along a trajectory using Shapely's buffered LineString.
    
    Instead of sampling discrete points, this creates a "corridor" around the trajectory
    (LineString buffered by half the vehicle width) and checks intersection with agent polygons.
    
    For moving agents, we predict their positions and create swept polygons.
    
    Returns: (collision_detected, collision_index, agent_velocity)
    """
    ego = pm.ego_vehicle
    min_velocity_threshold = PlanningSettings.min_velocity_threshold
    
    if trajectory is None or len(trajectory.path_x) < 2:
        # Check current position collision
        for agent in pm.agent_vehicles:
            if ego.get_bb_polygon().intersects(agent.get_bb_polygon()):
                log.info(f"Collision at current position {ego.x}, {ego.y}")
                return True, 0, agent.velocity
        return False, -1, -1
    
    path_x = trajectory.path_x
    path_y = trajectory.path_y
    
    # Create trajectory LineString and buffer it by half vehicle width (+ safety margin)
    trajectory_line = LineString(list(zip(path_x, path_y)))
    ego_half_width = ego.width / 2 + PlanningSettings.collision_safety_margin
    trajectory_corridor = trajectory_line.buffer(ego_half_width, cap_style='flat')
    
    # Get ego velocity profile for time estimation
    ego_velocities = getattr(trajectory, 'velocity', None)
    if ego_velocities is None or len(ego_velocities) == 0:
        default_vel = ego.velocity if ego.velocity > 0 else PlanningSettings.default_ego_velocity
        ego_velocities = np.ones(len(path_x)) * default_vel
    
    # Precompute cumulative distances along trajectory for time estimation
    cumulative_dist = [0.0]
    for i in range(1, len(path_x)):
        dist = np.sqrt((path_x[i] - path_x[i-1])**2 + (path_y[i] - path_y[i-1])**2)
        cumulative_dist.append(cumulative_dist[-1] + dist)
    total_length = cumulative_dist[-1]
    
    # Estimate total traversal time
    avg_velocity = np.mean(ego_velocities)
    total_time = total_length / max(avg_velocity, 1.0)
    
    for agent in pm.agent_vehicles:
        agent_polygon = agent.get_bb_polygon()
        
        if abs(agent.velocity) > min_velocity_threshold:
            # Moving agent: create swept polygon from current to predicted position
            # Predict where agent will be at end of trajectory traversal
            predicted_x = agent.x + agent.velocity * np.cos(agent.theta) * total_time
            predicted_y = agent.y + agent.velocity * np.sin(agent.theta) * total_time
            
            # Create a swept area: union of current position, predicted position, and path between
            predicted_agent = AgentState(
                x=predicted_x, y=predicted_y, theta=agent.theta,
                velocity=agent.velocity, agent_id=agent.agent_id,
                length=agent.length, width=agent.width
            )
            predicted_polygon = predicted_agent.get_bb_polygon()
            
            # Create swept polygon: convex hull of current and predicted bounding boxes
            # This is conservative but efficient
            try:
                current_corners = list(agent_polygon.exterior.coords)
                predicted_corners = list(predicted_polygon.exterior.coords)
                all_corners = current_corners + predicted_corners
                swept_polygon = Polygon(all_corners).convex_hull
            except (AttributeError, ValueError, TypeError) as e:
                # Fallback: just use union if polygon construction fails
                log.debug(f"Failed to create swept polygon: {e}, using union fallback")
                swept_polygon = agent_polygon.union(predicted_polygon).convex_hull
            
            if trajectory_corridor.intersects(swept_polygon):
                # Find approximate collision index by checking where intersection occurs
                collision_idx = _find_collision_index(trajectory_line, swept_polygon, path_x, path_y)
                log.debug(f" └─ Collision (moving agent) at idx {collision_idx}, "
                         f"agent vel: {agent.velocity:.1f}m/s, traversal time: {total_time:.2f}s")
                return True, collision_idx, agent.velocity
        else:
            # Static agent: simple intersection check
            if trajectory_corridor.intersects(agent_polygon):
                collision_idx = _find_collision_index(trajectory_line, agent_polygon, path_x, path_y)
                log.debug(f" └─ Collision (static agent) at idx {collision_idx}")
                return True, collision_idx, agent.velocity
    
    log.debug(f" └─ ✅ No Collision (corridor check)")
    return False, -1, -1


def _find_collision_index(trajectory_line: LineString, obstacle_polygon: Polygon, 
                          path_x: np.ndarray, path_y: np.ndarray) -> int:
    """
    Find the approximate trajectory index where collision with obstacle occurs.
    Uses binary search for efficiency.
    """
    n = len(path_x)
    if n < 2:
        return 0
    
    # Binary search to find first collision point
    left, right = 1, n - 1  # Start from 1 to ensure at least 2 points
    collision_idx = n - 1  # default to end if can't find
    
    while left <= right:
        mid = (left + right) // 2
        # Check if segment from start to mid intersects obstacle
        # Ensure we have at least 2 points for a valid LineString
        end_idx = max(2, mid + 1)
        partial_line = LineString(list(zip(path_x[:end_idx], path_y[:end_idx])))
        
        if partial_line.intersects(obstacle_polygon):
            collision_idx = mid
            right = mid - 1  # Search earlier
        else:
            left = mid + 1  # Search later
    
    return max(1, collision_idx)  # At least index 1
