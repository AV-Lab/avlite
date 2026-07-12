import logging
from typing import Optional

import numpy as np
from shapely.geometry import LineString, Polygon

from avlite.c10_perception.c11_perception_model import PerceptionModel, AgentState, SingleTrajectory
from avlite.c50_common.c54_trajectory_tracker import TrajectoryTracker

log = logging.getLogger(__name__)

_LARGE_CLEARANCE = 1e6


def precompute_obstacle_polygons(
    pm: PerceptionModel,
    total_time: float,
    min_velocity_threshold: float = 0.5,
    obstacle_inflation_margin: float = 0.0,
    beside_sweep_time: float = 0.0,
    beside_rear_window: float = 0.0,
) -> list:
    """Build obstacle polygons (swept for movers, plain for statics) once per replan.

    Pass the returned list to check_collision via ``obstacle_polygons`` to avoid
    rebuilding N_agents polygons for every lattice edge.

    Forward sweeping requires an active predictor: agents are only projected forward
    when ``pm.prediction`` supplies a trajectory for them. With prediction disabled every
    agent stays a static box (no constant-velocity fabrication). Agents ahead of the ego
    are swept over ``total_time``; agents abreast or just-behind the ego (within
    ``beside_rear_window`` metres) are swept over the shorter ``beside_sweep_time``
    (0 disables), keeping the lattice clear of a just-passed agent before cutting back to
    the reference line. Agents further behind than ``beside_rear_window`` are never swept.

    Returns a list of (polygon, agent_velocity) tuples.
    """
    pred = pm.prediction if isinstance(pm.prediction, SingleTrajectory) else None

    ego = pm.ego_vehicle
    ego_heading = np.array([np.cos(ego.theta), np.sin(ego.theta)])

    result = []
    for agent in pm.agent_vehicles:
        agent_polygon = agent.get_bb_polygon()
        to_agent = np.array([agent.x - ego.x, agent.y - ego.y])
        longitudinal = float(np.dot(ego_heading, to_agent))  # >=0 ahead, <0 behind
        moving = abs(agent.velocity) > min_velocity_threshold
        if longitudinal >= 0.0:
            sweep_time = total_time
        elif longitudinal >= -beside_rear_window:  # abreast / just-behind only
            sweep_time = beside_sweep_time
        else:
            sweep_time = 0.0
        agent_path = pred.trajectories.get(agent.agent_id) if pred is not None else None
        if moving and sweep_time > 0 and agent_path is not None:
            n_steps = agent_path.shape[0]
            step = min(int(sweep_time / pred.predict_delta_t), n_steps - 1)
            predicted_x, predicted_y = agent_path[step, 0], agent_path[step, 1]
            predicted_agent = AgentState(
                x=predicted_x, y=predicted_y, theta=agent.theta,
                velocity=agent.velocity, agent_id=agent.agent_id,
                length=agent.length, width=agent.width,
            )
            predicted_polygon = predicted_agent.get_bb_polygon()
            try:
                all_corners = list(agent_polygon.exterior.coords) + list(predicted_polygon.exterior.coords)
                obstacle = Polygon(all_corners).convex_hull
            except (AttributeError, ValueError, TypeError) as e:
                log.debug(f"Failed to create swept polygon: {e}, using union fallback")
                obstacle = agent_polygon.union(predicted_polygon).convex_hull
        else:
            obstacle = agent_polygon
        if obstacle_inflation_margin > 0:
            obstacle = obstacle.buffer(obstacle_inflation_margin)
        result.append((obstacle, agent.velocity))
    return result


def check_collision(
    pm: PerceptionModel,
    trajectory: TrajectoryTracker,
    obstacle_polygons: Optional[list] = None,
    min_velocity_threshold: float = 0.5,
    collision_safety_margin: float = 0.3,
    default_ego_velocity: float = 5.0,
) -> tuple[bool, int, float, float]:
    """
    Check for collision along a trajectory using Shapely's buffered LineString.

    ``obstacle_polygons``: pre-built list from :func:`precompute_obstacle_polygons`.
    When supplied the per-agent polygon construction is skipped, which is the main
    performance win when checking many lattice edges against the same agent set.

    Returns: (collision_detected, collision_index, agent_velocity, min_clearance)
    ``min_clearance`` is approx. ``line.distance(obstacle) - (width/2 + margin)``
    (0 on intersection; a large sentinel when there are no agents).
    """
    ego = pm.ego_vehicle

    if trajectory is None or len(trajectory.path_x) < 2:
        for agent in pm.agent_vehicles:
            if ego.get_bb_polygon().intersects(agent.get_bb_polygon()):
                log.info(f"Collision at current position {ego.x}, {ego.y}")
                return True, 0, agent.velocity, 0.0
        return False, -1, -1, _LARGE_CLEARANCE

    path_x = trajectory.path_x
    path_y = trajectory.path_y
    coords = [tuple(map(float, p)) for p in zip(path_x, path_y)]

    # Extend ends by half-length so the flat-cap corridor covers the ego body, not just the centerline.
    half_len = float(ego.length) / 2.0
    if half_len > 0 and len(coords) >= 2:
        x0, y0 = coords[0]
        x1, y1 = coords[1]
        n = float(np.hypot(x0 - x1, y0 - y1))
        if n > 1e-9:
            coords.insert(0, (x0 + half_len * (x0 - x1) / n, y0 + half_len * (y0 - y1) / n))
        xn1, yn1 = coords[-2]
        xn, yn = coords[-1]
        n = float(np.hypot(xn - xn1, yn - yn1))
        if n > 1e-9:
            coords.append((xn + half_len * (xn - xn1) / n, yn + half_len * (yn - yn1) / n))

    radius = ego.width / 2.0 + collision_safety_margin
    trajectory_line = LineString(coords)
    trajectory_corridor = trajectory_line.buffer(radius, cap_style='flat')

    if obstacle_polygons is not None:
        obstacles = obstacle_polygons
    else:
        # Slow path: build polygons here (used when called without pre-computed polygons)
        ego_velocities = getattr(trajectory, 'velocity', None)
        if ego_velocities is None or len(ego_velocities) == 0:
            default_vel = ego.velocity if ego.velocity > 0 else default_ego_velocity
            ego_velocities = np.ones(len(path_x)) * default_vel

        cumulative_dist = [0.0]
        for i in range(1, len(path_x)):
            dist = np.sqrt((path_x[i] - path_x[i - 1]) ** 2 + (path_y[i] - path_y[i - 1]) ** 2)
            cumulative_dist.append(cumulative_dist[-1] + dist)
        total_length = cumulative_dist[-1]
        avg_velocity = np.mean(ego_velocities)
        total_time = total_length / max(avg_velocity, 1.0)

        obstacles = []
        for agent in pm.agent_vehicles:
            agent_polygon = agent.get_bb_polygon()
            if abs(agent.velocity) > min_velocity_threshold:
                predicted_x = agent.x + agent.velocity * np.cos(agent.theta) * total_time
                predicted_y = agent.y + agent.velocity * np.sin(agent.theta) * total_time
                predicted_agent = AgentState(
                    x=predicted_x, y=predicted_y, theta=agent.theta,
                    velocity=agent.velocity, agent_id=agent.agent_id,
                    length=agent.length, width=agent.width,
                )
                predicted_polygon = predicted_agent.get_bb_polygon()
                try:
                    all_corners = list(agent_polygon.exterior.coords) + list(predicted_polygon.exterior.coords)
                    swept_polygon = Polygon(all_corners).convex_hull
                except (AttributeError, ValueError, TypeError) as e:
                    log.debug(f"Failed to create swept polygon: {e}, using union fallback")
                    swept_polygon = agent_polygon.union(predicted_polygon).convex_hull
                obstacles.append((swept_polygon, agent.velocity))
            else:
                obstacles.append((agent_polygon, agent.velocity))

    if not obstacles:
        log.debug(" └─ ✅ No Collision (no obstacles)")
        return False, -1, -1, _LARGE_CLEARANCE

    best_idx, best_vel = None, None
    min_clearance = _LARGE_CLEARANCE
    for obstacle, agent_velocity in obstacles:
        if trajectory_corridor.intersects(obstacle):
            idx = _find_collision_index(trajectory_line, obstacle, path_x, path_y)
            if best_idx is None or idx < best_idx:
                best_idx, best_vel = idx, agent_velocity
            min_clearance = 0.0
        else:
            # Line–polygon distance is much cheaper than corridor.distance (polygon–polygon).
            min_clearance = min(min_clearance, float(trajectory_line.distance(obstacle)) - radius)

    if best_idx is not None:
        log.debug(f" └─ Nearest collision at idx {best_idx}, agent vel: {best_vel:.1f}m/s")
        return True, best_idx, best_vel, 0.0

    log.debug(" └─ ✅ No Collision (corridor check)")
    return False, -1, -1, min_clearance


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
