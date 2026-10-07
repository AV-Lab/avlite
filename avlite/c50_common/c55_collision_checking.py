"""Planar bird's-eye collision in x, y. z is ignored. Not a 3D volume check."""

import logging
import math
from typing import Optional

import numpy as np
from shapely.geometry import LineString, Polygon

from avlite.c10_perception.c11_perception_model import (
    AgentState,
    AggregatedOccupancyFlow,
    GMM,
    GP,
    MultiTrajectory,
    OccupancyFlow,
    PerceptionModel,
    SingleTrajectory,
)
from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c50_common.c54_trajectory_tracker import TrajectoryTracker

log = logging.getLogger(__name__)

_LARGE_CLEARANCE = 1e6
_PROBABILISTIC_PREDICTIONS = (MultiTrajectory, GMM, GP, OccupancyFlow, AggregatedOccupancyFlow)


def precompute_obstacle_polygons_2d(
    pm: PerceptionModel,
    total_time: float,
    min_velocity_threshold: float = 0.5,
    obstacle_inflation_margin: float = 0.0,
    beside_sweep_time: float = 0.0,
    beside_rear_window: float = 0.0,
) -> list:
    """Build obstacle polygons (swept for movers, plain for statics) once per replan.

    Bird's-eye polygons in x, y. z is ignored.

    Pass the returned list to check_collision_2d via ``obstacle_polygons`` to avoid
    rebuilding N_agents polygons for every lattice edge.

    Forward sweeping uses one polyline per agent: ``SingleTrajectory.trajectories``,
    ``GP.means``, or the highest-weight mode of ``GMM.trajectories`` (first mode on a
    tie; covariances are unused). With prediction disabled, or when an agent has no
    polyline, the agent stays a static box (no constant-velocity fabrication).
    ``OccupancyFlow``, ``AggregatedOccupancyFlow``, ``MultiTrajectory``, and any other
    forecast type are ignored — a warning on each call — and agents stay current
    boxes. ``collision_probability_2d`` still reads ``MultiTrajectory``
    and every ``GMM`` mode.

    Agents ahead of the ego are swept over ``total_time``; agents abreast or just-behind the ego (within
    ``beside_rear_window`` metres) are swept over the shorter ``beside_sweep_time``
    (0 disables), keeping the lattice clear of a just-passed agent before cutting back to
    the reference line. Agents further behind than ``beside_rear_window`` are never swept.

    Returns a list of (polygon, agent_velocity) tuples.
    """
    def sweep_polyline(pred, agent_id: int) -> Optional[np.ndarray]:
        """``[n_steps, 2]`` centerline to sweep, or None to keep the current box."""
        if isinstance(pred, SingleTrajectory):
            return pred.trajectories.get(agent_id)
        if isinstance(pred, GP):
            return pred.means.get(agent_id)
        if isinstance(pred, GMM):
            modes = pred.trajectories.get(agent_id)
            weights = pred.weights.get(agent_id)
            if modes is None or weights is None or len(modes) == 0 or len(weights) == 0:
                return None
            n = min(len(modes), len(weights))
            mode = int(np.argmax(np.asarray(weights[:n], dtype=float)))
            return np.asarray(modes[mode])
        return None

    pred = pm.prediction
    if pred is not None and not isinstance(pred, (SingleTrajectory, GP, GMM)):
        log.warning(
            "Swept polygons ignore %s and keep current agent boxes. "
            "SingleTrajectory paths, GP means, and the highest-weight GMM mean are swept.",
            type(pred).__name__,
        )
        pred = None

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
        agent_path = sweep_polyline(pred, agent.agent_id) if pred is not None else None
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


def check_collision_2d(
    pm: PerceptionModel,
    trajectory: TrajectoryTracker,
    obstacle_polygons: Optional[list] = None,
    min_velocity_threshold: float = 0.5,
    ego_inflation_margin: float = 0.3,
    default_ego_velocity: float = 5.0,
) -> tuple[bool, int, float, float]:
    """
    Bird's-eye check in x, y (z is ignored). Not a 3D volume test.

    Check for collision along a trajectory using Shapely's buffered LineString.

    ``obstacle_polygons``: pre-built list from :func:`precompute_obstacle_polygons_2d`.
    When supplied the per-agent polygon construction is skipped, which is the main
    performance win when checking many lattice edges against the same agent set.
    When omitted, polygons are built with that function over this trajectory's travel
    time, so the same forecast rule applies (``SingleTrajectory`` path, ``GP`` mean,
    or highest-weight ``GMM`` mean; other forecasts warn and stay current boxes).

    When ``c15_probabilistic_collision_checking`` is on and the forecast is
    ``MultiTrajectory``, ``GMM``, ``GP``, ``OccupancyFlow``, or
    ``AggregatedOccupancyFlow``, this uses :func:`collision_probability_2d` instead
    and ignores ``obstacle_polygons``. An edge collides when that probability is
    greater than ``c15_max_local_collision_probability``.

    Returns: (collision_detected, collision_index, agent_velocity, min_clearance)
    ``min_clearance`` is approx. ``line.distance(obstacle) - (width/2 + margin)``
    (0 on intersection; a large sentinel when there are no agents).
    """
    if (
        PerceptionSettings.c15_probabilistic_collision_checking
        and isinstance(pm.prediction, _PROBABILISTIC_PREDICTIONS)
    ):
        found: list[tuple[int, float]] = []
        probability = collision_probability_2d(
            pm,
            trajectory,
            ego_inflation_margin=ego_inflation_margin,
            default_ego_velocity=default_ego_velocity,
            index_out=found,
        )
        idx, velocity = found[0]
        if probability > PerceptionSettings.c15_max_local_collision_probability:
            return True, idx, velocity, 0.0
        return False, -1, -1.0, _LARGE_CLEARANCE

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

    radius = ego.width / 2.0 + ego_inflation_margin
    trajectory_line = LineString(coords)
    trajectory_corridor = trajectory_line.buffer(radius, cap_style='flat')

    if obstacle_polygons is not None:
        obstacles = obstacle_polygons
    else:
        # Same forecast rule as a caller-supplied polygon list, over this path's travel time.
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
        total_time = total_length / max(float(avg_velocity), 1.0)
        obstacles = precompute_obstacle_polygons_2d(
            pm,
            total_time=total_time,
            min_velocity_threshold=min_velocity_threshold,
        )

    if not obstacles:
        log.debug(" └─ ✅ No Collision (no obstacles)")
        return False, -1, -1, _LARGE_CLEARANCE

    best_idx, best_vel = None, None
    min_clearance = _LARGE_CLEARANCE
    n_path = len(path_x)
    for obstacle, agent_velocity in obstacles:
        if trajectory_corridor.intersects(obstacle):
            # Earliest centerline index whose prefix meets the obstacle.
            # Search path_x/path_y, not the half-length-extended corridor.
            # left starts at 1 so the probe LineString always has two points.
            left, right = 1, n_path - 1
            idx = n_path - 1  # fall back to the last point if the search misses
            while left <= right:
                mid = (left + right) // 2
                end_idx = max(2, mid + 1)
                partial = LineString(list(zip(path_x[:end_idx], path_y[:end_idx])))
                if partial.intersects(obstacle):
                    idx = mid
                    right = mid - 1  # hit is at or before mid
                else:
                    left = mid + 1  # hit is after mid
            idx = max(1, idx)
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


def collision_probability_2d(
    pm: PerceptionModel,
    trajectory: TrajectoryTracker,
    *,
    ego_inflation_margin: float = 0.3,
    default_ego_velocity: float = 5.0,
    index_out: list[tuple[int, float]] | None = None,
) -> float:
    """Probability in ``[0, 1]`` that *trajectory* meets an agent at the same time.

    Bird's-eye check in x, y (z is ignored). Not a 3D volume test.

    ``MultiTrajectory`` and ``GMM`` add a mode's weight when that polyline overlaps
    the ego box at the matching forecast time (step ``k`` at ``(k + 1) * predict_delta_t``).
    ``SingleTrajectory`` is one mode of weight 1. ``OccupancyFlow`` and
    ``AggregatedOccupancyFlow`` use the maximum occupancy among grid cells that box
    intersects, then the maximum across forecast steps. ``AggregatedOccupancyFlow``
    is one scene value. Any other forecast, or a missing agent id, uses the agent's
    current box. Weights are not renormalized. Agents are independent:
    ``1 - prod(1 - p)``. ``ego_inflation_margin`` is buffered onto each ego box.
    No agents returns ``0``.
    """
    agents = pm.agent_vehicles
    if not agents:
        if index_out is not None:
            index_out.append((-1, -1.0))
        return 0.0

    ego = pm.ego_vehicle
    best_idx: int | None = None
    best_vel = -1.0

    def note(idx: int, velocity: float) -> None:
        nonlocal best_idx, best_vel
        if best_idx is None or idx < best_idx:
            best_idx, best_vel = idx, float(velocity)

    def finish(survive: float) -> float:
        if index_out is not None:
            if best_idx is None:
                index_out.append((-1, -1.0))
            else:
                index_out.append((best_idx, best_vel))
        return 1.0 - survive

    def boxes_meet(
        ax: float, ay: float, atheta: float, alen: float, awid: float,
        bx: float, by: float, btheta: float, blen: float, bwid: float,
    ) -> bool:
        """True when the first rectangle, buffered by a round ``ego_inflation_margin``, meets the second.

        The round buffer meets the other box when the distance between the two
        rectangles is at most that margin, the same test as ``Polygon.buffer(margin).intersects``.
        """
        ahx = float(alen) * 0.5
        ahy = float(awid) * 0.5
        bhx = float(blen) * 0.5
        bhy = float(bwid) * 0.5
        ac = math.cos(float(atheta))
        asn = math.sin(float(atheta))
        bc = math.cos(float(btheta))
        bsn = math.sin(float(btheta))
        dx = float(bx) - float(ax)
        dy = float(by) - float(ay)
        margin = float(ego_inflation_margin)
        separated = False
        for nx, ny in ((ac, asn), (-asn, ac), (bc, bsn), (-bsn, bc)):
            ra = ahx * abs(nx * ac + ny * asn) + ahy * abs(-nx * asn + ny * ac)
            rb = bhx * abs(nx * bc + ny * bsn) + bhy * abs(-nx * bsn + ny * bc)
            gap = abs(dx * nx + dy * ny) - ra - rb
            if gap > margin + 1e-8:
                return False
            if gap > 1e-8:
                separated = True
        if not separated:
            return True

        def corners(cx: float, cy: float, c: float, s: float, hx: float, hy: float):
            return (
                (cx + hx * c - hy * s, cy + hx * s + hy * c),
                (cx - hx * c - hy * s, cy - hx * s + hy * c),
                (cx - hx * c + hy * s, cy - hx * s - hy * c),
                (cx + hx * c + hy * s, cy + hx * s - hy * c),
            )

        def nearest(verts, edges) -> float:
            best = float("inf")
            for px, py in verts:
                for i in range(4):
                    x0, y0 = edges[i]
                    x1, y1 = edges[(i + 1) % 4]
                    abx = x1 - x0
                    aby = y1 - y0
                    den = abx * abx + aby * aby
                    if den <= 1e-24:
                        ddx = px - x0
                        ddy = py - y0
                        d2 = ddx * ddx + ddy * ddy
                    else:
                        t = ((px - x0) * abx + (py - y0) * aby) / den
                        if t < 0.0:
                            t = 0.0
                        elif t > 1.0:
                            t = 1.0
                        ddx = px - (x0 + t * abx)
                        ddy = py - (y0 + t * aby)
                        d2 = ddx * ddx + ddy * ddy
                    if d2 < best:
                        best = d2
            return best

        a_pts = corners(float(ax), float(ay), ac, asn, ahx, ahy)
        b_pts = corners(float(bx), float(by), bc, bsn, bhx, bhy)
        dist2 = min(nearest(a_pts, b_pts), nearest(b_pts, a_pts))
        limit = margin + 1e-8
        return dist2 <= limit * limit

    def meet_agent(x: float, y: float, theta: float, agent) -> bool:
        return boxes_meet(
            x, y, theta, ego.length, ego.width,
            agent.x, agent.y, agent.theta, agent.length, agent.width,
        )

    short = trajectory is None or len(trajectory.path_x) < 2
    if short:
        survive = 1.0
        for agent in agents:
            if meet_agent(ego.x, ego.y, ego.theta, agent):
                survive = 0.0
                note(0, agent.velocity)
                break
        return finish(survive)

    path_x = np.asarray(trajectory.path_x, dtype=float)
    path_y = np.asarray(trajectory.path_y, dtype=float)
    heading = np.asarray(trajectory.path_heading, dtype=float)
    arc = np.asarray(trajectory.path_s, dtype=float)
    speed = np.asarray(trajectory.velocity, dtype=float)
    n = len(path_x)
    if len(speed) < n:
        speed = np.pad(speed, (0, n - len(speed)), constant_values=default_ego_velocity)
    speed = np.where(speed[:n] > 1e-6, speed[:n], default_ego_velocity)
    ego_t = np.concatenate([[0.0], np.cumsum(np.diff(arc) / speed[:-1])])

    pred = pm.prediction
    half_length = float(ego.length) / 2.0 + ego_inflation_margin
    half_width = float(ego.width) / 2.0 + ego_inflation_margin

    def ego_pose(t_k: float):
        if t_k > float(ego_t[-1]):
            return None
        hi = int(np.searchsorted(ego_t, t_k, side="left"))
        hi = min(max(hi, 1), len(ego_t) - 1)
        lo = hi - 1
        span = float(ego_t[hi] - ego_t[lo])
        frac = 0.0 if span < 1e-9 else (t_k - float(ego_t[lo])) / span
        ex = float(path_x[lo] + frac * (path_x[hi] - path_x[lo]))
        ey = float(path_y[lo] + frac * (path_y[hi] - path_y[lo]))
        return ex, ey, float(heading[lo]), hi

    def max_cell(grid, ex: float, ey: float, theta: float, origin_x: float, origin_y: float, resolution: float) -> float:
        grid = np.asarray(grid, dtype=float)
        if grid.size == 0 or resolution <= 0.0:
            return 0.0
        rows, cols = grid.shape
        c, s = float(np.cos(theta)), float(np.sin(theta))
        ext_x = half_length * abs(c) + half_width * abs(s)
        ext_y = half_length * abs(s) + half_width * abs(c)
        col0 = max(0, int(np.floor((ex - ext_x - origin_x) / resolution)))
        col1 = min(cols - 1, int(np.floor((ex + ext_x - origin_x) / resolution)))
        row0 = max(0, int(np.floor((ey - ext_y - origin_y) / resolution)))
        row1 = min(rows - 1, int(np.floor((ey + ext_y - origin_y) / resolution)))
        if col0 > col1 or row0 > row1:
            return 0.0
        cc, rr = np.meshgrid(np.arange(col0, col1 + 1), np.arange(row0, row1 + 1))
        half = resolution / 2.0
        dx = origin_x + (cc + 0.5) * resolution - ex
        dy = origin_y + (rr + 0.5) * resolution - ey
        cell_on_axis = half * (abs(c) + abs(s))
        separate = (
            (np.abs(dx) > ext_x + half)
            | (np.abs(dy) > ext_y + half)
            | (np.abs(dx * c + dy * s) > half_length + cell_on_axis)
            | (np.abs(-dx * s + dy * c) > half_width + cell_on_axis)
        )
        hit = ~separate
        if not np.any(hit):
            return 0.0
        return float(grid[rr[hit], cc[hit]].max())

    def occupancy_along(grids, origin_x: float, origin_y: float, resolution: float) -> tuple[float, int | None]:
        p_best = 0.0
        hit_idx: int | None = None
        dt = float(pred.predict_delta_t)
        for k, grid in enumerate(grids):
            pose = ego_pose((k + 1) * dt)
            if pose is None:
                break
            ex, ey, theta, hi = pose
            step_p = max_cell(grid, ex, ey, theta, origin_x, origin_y, resolution)
            if step_p > p_best:
                p_best = step_p
            if hit_idx is None and step_p > 0.0:
                hit_idx = hi
        return p_best, hit_idx

    if isinstance(pred, AggregatedOccupancyFlow) and pred.occupancy_flow:
        for agent in agents:
            if meet_agent(ego.x, ego.y, ego.theta, agent):
                note(0, agent.velocity)
                return finish(0.0)
        p_scene, scene_idx = occupancy_along(
            pred.occupancy_flow, pred.origin_x, pred.origin_y, pred.resolution,
        )
        if scene_idx is not None:
            note(scene_idx, 0.0)
        return finish(1.0 - min(1.0, max(0.0, p_scene)))

    survive = 1.0
    for agent in agents:
        modes = weights = None
        if isinstance(pred, (MultiTrajectory, GMM)):
            modes = pred.trajectories.get(agent.agent_id)
            weights = pred.weights.get(agent.agent_id)
        elif isinstance(pred, SingleTrajectory):
            path = pred.trajectories.get(agent.agent_id)
            if path is not None:
                modes = np.asarray(path, dtype=float)[None, ...]
                weights = np.ones(1)
        if meet_agent(ego.x, ego.y, ego.theta, agent):
            survive *= 0.0
            note(0, agent.velocity)
            continue
        if isinstance(pred, OccupancyFlow):
            grids = pred.occupancy_flow.get(agent.agent_id)
            if grids:
                p_agent, agent_idx = occupancy_along(
                    grids, pred.origin_x, pred.origin_y, pred.resolution,
                )
                if agent_idx is not None:
                    note(agent_idx, agent.velocity)
                survive *= 1.0 - min(1.0, max(0.0, p_agent))
                continue
        if modes is None or weights is None or len(modes) == 0:
            for i in range(n):
                if meet_agent(float(path_x[i]), float(path_y[i]), float(heading[i]), agent):
                    survive *= 0.0
                    note(i, agent.velocity)
                    break
            continue

        dt = pred.predict_delta_t
        p_agent = 0.0
        agent_idx: int | None = None
        for m in range(min(len(modes), len(weights))):
            mode = np.asarray(modes[m], dtype=float)
            hit = False
            hit_idx = 0
            for k in range(len(mode)):
                pose = ego_pose((k + 1) * dt)
                if pose is None:
                    break
                ex, ey, ego_theta, hi = pose
                prev_x, prev_y = (agent.x, agent.y) if k == 0 else (float(mode[k - 1, 0]), float(mode[k - 1, 1]))
                dx, dy = float(mode[k, 0]) - prev_x, float(mode[k, 1]) - prev_y
                theta = agent.theta if dx * dx + dy * dy < 1e-12 else float(np.arctan2(dy, dx))
                if boxes_meet(
                    ex, ey, ego_theta, ego.length, ego.width,
                    float(mode[k, 0]), float(mode[k, 1]), theta, agent.length, agent.width,
                ):
                    hit = True
                    hit_idx = hi
                    break
            if hit:
                weight = float(weights[m])
                p_agent += weight
                if weight > 0 and (agent_idx is None or hit_idx < agent_idx):
                    agent_idx = hit_idx
        if agent_idx is not None:
            note(agent_idx, agent.velocity)
        survive *= 1.0 - min(1.0, max(0.0, p_agent))

    return finish(survive)
