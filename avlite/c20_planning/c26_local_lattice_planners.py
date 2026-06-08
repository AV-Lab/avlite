from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Type
import math
import time

import numpy as np

from avlite.c10_perception.c12_perception_strategy import PerceptionModel
from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c20_planning.c21_planning_model import GlobalPlan, LocalPlan
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c20_planning.c27_lattice import Lattice, Node, Edge
from avlite.c20_planning.c29_settings import PlanningSettings
from avlite.c60_common.c64_collision_checking import check_collision, precompute_obstacle_polygons

if TYPE_CHECKING:
    from avlite.c30_control.c32_control_strategy import ControlStrategy

import logging
log = logging.getLogger(__name__)


class LatticePlanningStrategy(LocalPlanningStrategy, abstract=True):
    """Local planning over a sampled frenet lattice.

    Builds a :class:`Lattice` of candidate edges along the global trajectory,
    commits to a chain of :class:`Edge` objects (``selected_local_plan``), and
    advances along that chain as the ego moves. Concrete planners implement
    :meth:`replan` to populate ``selected_local_plan`` from the lattice.
    """

    def __init__(self, global_plan: GlobalPlan, pm: PerceptionModel,
                 planning_horizon: int = 3, num_of_edge_points: int = 10,
                 controller: Optional['ControlStrategy'] = None,
                 setting: Type[PlanningSettings] = PlanningSettings):
        super().__init__(global_plan=global_plan, pm=pm, controller=controller, setting=setting)

        self.planning_horizon: int = planning_horizon
        self.num_of_edge_points: int = num_of_edge_points
        self.selected_local_plan: Optional[Edge] = None
        self.lattice: Lattice = Lattice(
            self.global_trajectory, global_plan.left_boundary_d, global_plan.right_boundary_d,
            planning_horizon=self.planning_horizon, num_of_points=self.num_of_edge_points)

        # Replan stability: track when plan was last changed
        self._last_plan_change_time: float = 0.0
        self._replan_wait_time: float = setting.c26_replan_wait_time
        self._min_edge_progress_to_block: float = setting.c26_min_edge_progress_to_block
        self._urgent_collision_threshold: int = setting.c26_urgent_collision_threshold
        self._disconnect_distance_threshold: float = setting.c26_disconnect_distance_threshold

    def set_global_plan(self, global_plan: GlobalPlan, ego_xy=None) -> None:
        super().set_global_plan(global_plan, ego_xy=ego_xy)
        self.lattice = Lattice(
            self.global_trajectory, global_plan.left_boundary_d, global_plan.right_boundary_d,
            planning_horizon=self.planning_horizon, num_of_points=self.num_of_edge_points)

    def reset(self, wp: int = 0):
        super().reset(wp)
        self.selected_local_plan = None
        self.lattice.reset()
        self._last_plan_change_time = 0.0

    def get_local_plan(self) -> LocalPlan:
        if self.selected_local_plan is not None:
            return LocalPlan.from_trajectory(self.selected_local_plan.local_trajectory)
        return LocalPlan.from_trajectory(self.global_trajectory)

    def should_switch_plan(self, new_plan: Edge, force_if_collision: bool = True) -> bool:
        """
        Determine if we should switch to a new plan.

        Switches only if:
        - No current plan exists
        - Current edge is fully traversed with no successor
        - Current plan has an urgent collision (within threshold waypoints)

        Otherwise the planner commits to the current edge and follows it to
        completion, preventing jitter from replanning on every cycle.
        """
        # No plan yet — take the first one available
        if self.selected_local_plan is None:
            return True

        # Zero-velocity recovery: escape an all-zero emergency-stop plan to any clean alternative.
        # This covers both (a) collision-blocked plans and (b) boundary-violation-blocked plans
        # where the emergency plan has collision=False and the normal collision guard never fires.
        cur_vel = getattr(self.selected_local_plan.local_trajectory, 'velocity', None)
        if cur_vel is not None and len(cur_vel) > 0:
            _cva = np.asarray(cur_vel)
            _is_emergency_stop = (float(_cva[-1]) < 0.5 and float(np.mean(_cva)) < 3.0)
            if (_is_emergency_stop
                    and not new_plan.collision
                    and not new_plan.boundary_violation):
                log.info("Emergency-stop plan detected — recovering to clean plan")
                return True

        # Current edge is done and has no queued successor — must switch
        if (self.selected_local_plan.local_trajectory.is_traversed()
                and self.selected_local_plan.selected_next_local_plan is None):
            return True

        # Current plan is colliding — attempt to escape to a collision-free plan
        if force_if_collision and self.selected_local_plan.collision:
            collision_idx = getattr(self.selected_local_plan, 'collision_idx', -1)
            current_wp = self.selected_local_plan.local_trajectory.current_wp
            waypoints_to_collision = collision_idx - current_wp if collision_idx >= 0 else float('inf')
            if waypoints_to_collision <= self._urgent_collision_threshold:
                # Imminent — switch immediately regardless of wait time
                log.debug(f"Switching plan: urgent collision in {waypoints_to_collision} waypoints")
                return True
            # Agent cleared: new plan is collision-free with a materially better speed profile.
            # Allows immediate recovery when an obstacle leaves the path, without waiting
            # for the full replan_wait_time.
            if not new_plan.collision and not new_plan.boundary_violation:
                cur_v = float(np.mean(np.asarray(self.selected_local_plan.local_trajectory.velocity)))
                new_v = float(np.mean(np.asarray(new_plan.local_trajectory.velocity)))
                if new_v > cur_v + 0.5:
                    log.info(f"Switching plan: agent cleared ({new_v:.1f} > {cur_v:.1f} m/s)")
                    return True
            # Non-urgent but colliding — switch only after wait time to avoid oscillation
            if not new_plan.collision and (time.time() - self._last_plan_change_time >= self._replan_wait_time):
                log.debug("Switching plan: escaping blocked plan to collision-free alternative")
                return True

        # Geometric disconnect: car has fallen behind the plan start
        local_tj = self.selected_local_plan.local_trajectory
        if local_tj is not None:
            cwp = local_tj.current_wp
            dist = math.hypot(
                local_tj.path_x[cwp] - self.location_xy[0],
                local_tj.path_y[cwp] - self.location_xy[1],
            )
            if dist > self._disconnect_distance_threshold:
                log.debug(f"Switching plan: geometric disconnect — {dist:.1f}m from plan")
                return True

        # Commit to current plan — do not switch
        return False

    def set_selected_plan(self, new_plan: Edge) -> None:
        """Set the selected plan and update the change timestamp."""
        self.selected_local_plan = new_plan
        self._last_plan_change_time = time.time()

    def _on_edge_traversed(self) -> None:
        """Called once when step() advances to the next edge in the committed chain.

        Subclasses override this to extend the planning horizon incrementally
        (sliding-window replan). The base implementation is a no-op.
        """

    def _advance_local_plan(self, state: EgoState) -> None:
        """Advance the committed edge chain based on the current ego state."""
        if self.selected_local_plan is None:
            return

        self.selected_local_plan.local_trajectory.update_waypoint_by_xy(state.x, state.y)

        if self.selected_local_plan.local_trajectory.is_traversed() and self.selected_local_plan.selected_next_local_plan is not None:
            log.info("Local Plan Traversed, choosing next selected Local Plan")
            self.selected_local_plan = self.selected_local_plan.selected_next_local_plan
            self.selected_local_plan.local_trajectory.update_to_next_waypoint()
            self._on_edge_traversed()
        elif self.selected_local_plan.local_trajectory.is_traversed() and self.selected_local_plan.selected_next_local_plan is None:
            log.info("Local plan traversed, no next local plan — holding last trajectory until replan")

    def step_wp(self):
        """
        Advances the planner to the next waypoint and updates the traversed path.
        """
        log.info(f"Step: {self.global_trajectory.current_wp}")
        # next edge selected, but not finished
        if self.selected_local_plan is not None and not self.selected_local_plan.local_trajectory.is_traversed():
            self.selected_local_plan.local_trajectory.update_to_next_waypoint()
            x_new, y_new = self.selected_local_plan.local_trajectory.get_current_xy()

        # next edge selected, but finished
        elif (
            self.selected_local_plan is not None
            and self.selected_local_plan.local_trajectory.is_traversed()
            and self.selected_local_plan.selected_next_local_plan is not None
        ):
            log.info("Local Plan Completed, choosing next selected Local Plan")
            self.selected_local_plan = self.selected_local_plan.selected_next_local_plan
            self.selected_local_plan.local_trajectory.update_to_next_waypoint()
            x_new, y_new = self.selected_local_plan.local_trajectory.get_current_xy()
        # no edge selected — hold last trajectory until replan provides a new one
        elif (
            self.selected_local_plan is not None
            and self.selected_local_plan.local_trajectory.is_traversed()
            and self.selected_local_plan.selected_next_local_plan is None
        ):
            log.info("Local Plan Traversed. No next Local Plan selected — holding last trajectory until replan")
            x_new = self.selected_local_plan.local_trajectory.path_x[-1]
            y_new = self.selected_local_plan.local_trajectory.path_y[-1]
        else:
            log.warning("No Local Plan, back to closest next reference point")
            x_new = self.global_trajectory.path_x[self.global_trajectory.next_wp]
            y_new = self.global_trajectory.path_y[self.global_trajectory.next_wp]

        self.traversed_x.append(x_new)
        self.traversed_y.append(y_new)
        current_orientation = self.global_trajectory.get_current_heading()
        log.debug(f"global tj current orientation: {current_orientation}")

        # TODO some error check might be needed
        self.global_trajectory.update_waypoint_by_xy(x_new, y_new)
        if self.selected_local_plan is not None:
            self.selected_local_plan.local_trajectory.update_waypoint_by_xy(x_new, y_new)

        #### Frenet Coordinates
        s_, d_ = self.global_trajectory.convert_xy_to_sd(x_new, y_new)
        self.traversed_d.append(d_)
        self.traversed_s.append(s_)

        if self.global_trajectory.is_traversed() and self.global_plan.race_mode:
            self.lap += 1
            log.info(f"Lap {self.lap} Done")

        self.location_xy = (self.traversed_x[-1], self.traversed_y[-1])
        self.location_sd = (self.traversed_s[-1], self.traversed_d[-1])

    def local_plan_len(self, tmp_plan=None):
        edge = self.selected_local_plan if tmp_plan is None else tmp_plan
        return 1 + self.__plan_len(edge=edge.selected_next_local_plan)

    def __plan_len(self, edge):
        if edge is None:
            return 0
        return 1 + self.__plan_len(edge=edge.selected_next_local_plan)


class GreedyLatticePlanner(LatticePlanningStrategy):
    def __init__(self, global_plan: GlobalPlan, env: PerceptionModel, setting: Type[PlanningSettings] = PlanningSettings, controller=None):

        super().__init__(global_plan=global_plan, pm=env, num_of_edge_points=setting.c26_num_of_edge_points, planning_horizon=setting.c26_planning_horizon, controller=controller, setting=setting)
        self.maneuver_distance: float = setting.c26_maneuver_distance
        self.boundary_clearance: float = setting.c26_boundary_clearance
        self.sample_size: int = setting.c26_sample_size
        self.match_speed_wp_buffer: int = setting.c26_match_speed_wp_buffer
        self.safety_margin_weight: float = setting.c26_safety_margin_weight
        self.max_lateral_accel: float = setting.c26_max_lateral_accel
        self.min_curvature_velocity: float = setting.c26_min_curvature_velocity
        self._min_ramp_start_velocity: float = setting.c26_min_ramp_start_velocity
        self._allow_curvature_fallback: bool = setting.c26_allow_curvature_fallback
        self._allow_boundary_violation_fallback: bool = setting.c26_allow_boundary_violation_fallback
        self._stopping_decel_factor: float = setting.c26_stopping_decel_factor
        self._fallback_deceleration: float = setting.c26_fallback_deceleration
        self._stopping_safety_buffer: float = setting.c26_stopping_safety_buffer

    def _get_max_curvature_for_velocity(self, velocity: float) -> float:
        """
        Compute velocity-dependent max curvature.
        Based on: a_lateral = v^2 * curvature, so curvature_max = a_lat_max / v^2
        """
        v = max(velocity, self.min_curvature_velocity)
        return self.max_lateral_accel / (v * v)

    def _is_curvature_feasible(self, edge) -> bool:
        """Check if edge trajectory curvature is within velocity-dependent limits."""
        if edge.local_trajectory is None:
            return True

        max_curv = edge.local_trajectory.max_curvature()
        velocity = self.pm.ego_vehicle.velocity if self.pm.ego_vehicle.velocity > 0 else self.min_curvature_velocity
        max_allowed = self._get_max_curvature_for_velocity(velocity)

        feasible = max_curv <= max_allowed
        if not feasible:
            log.debug(f"Edge curvature {max_curv:.4f} exceeds limit {max_allowed:.4f} at v={velocity:.1f} m/s")

        return feasible

    def _edge_cost(self, edge) -> float:
        """
        Compute cost for edge selection balancing reference tracking and safety.
        Lower cost = better edge.
        """
        # Cost for deviation from reference (d=0)
        ref_cost = abs(edge.end.d)

        # Safety cost: prefer edges with more clearance (higher min_clearance = lower cost)
        clearance = getattr(edge, 'min_clearance', 10.0)
        safety_cost = 1.0 / (clearance + 0.1)  # inverse: more clearance = lower cost

        return ref_cost + self.safety_margin_weight * safety_cost * 10.0

    def _select_best_edge(self, edges: list):
        """Select best edge from candidates considering both reference and safety.
        Hard-prefers edges ending at d=0 (within tolerance) when any exist."""
        if not edges:
            return None
        d0_edges = [e for e in edges if abs(e.end.d) < PlanningSettings.c26_d0_reference_threshold]
        if d0_edges:
            return min(d0_edges, key=self._edge_cost)
        return min(edges, key=self._edge_cost)

    def replan(self, back_to_ref_horizon=10):
        if len(self.traversed_s) == 0:
            log.debug("Location unkown. Cannot replan")
            return

        # self.selected_local_plan = None
        # delete previous plans
        self.lattice.reset()
        self.lattice.sample_nodes(
            s=self.location_sd[0],
            d=self.location_sd[1],
            maneuver_distance=self.maneuver_distance,
            boundary_clearance=self.boundary_clearance,
            sample_size=self.sample_size,
            # orientation = np.tan(self.pm.ego_vehicle.theta)/2 -  0.1* self.location_sd[1],
        )

        self.lattice.generate_lattice_from_nodes(pm=self.pm)

        # Filter edges: no collision and curvature within limits
        feasible_edges = [edge for edge in self.lattice.level0_edges
                         if not edge.collision and not edge.boundary_violation and self._is_curvature_feasible(edge)]

        # Fallback 1: drop curvature requirement (gated by settings)
        if not feasible_edges and self._allow_curvature_fallback:
            feasible_edges = [edge for edge in self.lattice.level0_edges if not edge.collision and not edge.boundary_violation]
            if feasible_edges:
                log.debug("No curvature-feasible edges, using collision-free edges")

        # Fallback 2: accept boundary-violation edges when NO real collision exists.
        # Boundary violations are soft constraints — a trajectory slightly outside the
        # clearance margin should not trigger an emergency stop. (Gated by settings)
        if not feasible_edges and self._allow_boundary_violation_fallback:
            feasible_edges = [edge for edge in self.lattice.level0_edges if not edge.collision]
            if feasible_edges:
                log.warning("All edges have boundary violations but no collision — "
                            "proceeding with best collision-free edge despite boundary violations")

        if feasible_edges:
            # Select best edge considering both reference tracking and safety
            edge = self._select_best_edge(feasible_edges)

            current_plan = edge
            while edge is not None and len(edge.next_edges) > 0:
                # Filter next edges by collision and curvature
                next_feasible = [e for e in edge.next_edges
                                 if not e.collision and not e.boundary_violation and self._is_curvature_feasible(e)]
                if not next_feasible and self._allow_curvature_fallback:
                    next_feasible = [e for e in edge.next_edges if not e.collision and not e.boundary_violation]
                if not next_feasible and self._allow_boundary_violation_fallback:
                    next_feasible = [e for e in edge.next_edges if not e.collision]
                if not next_feasible:
                    edge.selected_next_local_plan = None
                    break
                edge.selected_next_local_plan = self._select_best_edge(next_feasible)
                edge = edge.selected_next_local_plan

            log.debug(f"current plan len {self.local_plan_len(current_plan)}")
            if self.local_plan_len(current_plan) == self.planning_horizon:
                # Only switch if allowed (no recent change or current plan has collision)
                if self.should_switch_plan(current_plan):
                    log.debug("Switching to new plan")
                    self.set_selected_plan(current_plan)
                    # Velocity continuity: ramp from current ego speed up to reference
                    # to prevent a sudden speed jump when recovering from a stop/obstacle.
                    ego_v = max(self._min_ramp_start_velocity, self.pm.ego_vehicle.velocity)  # ensure positive creep speed to avoid current_wp=0 deadlock
                    tj = current_plan.local_trajectory
                    # Only ramp when ego is slower than the plan's opening speed (recovering from
                    # a stop or emergency brake). Skip when already at or above plan speed so
                    # that a distant/passing obstacle does not suppress normal acceleration.
                    if ego_v < tj.velocity[0]:
                        n = min(self.match_speed_wp_buffer, len(tj.velocity))
                        ramp = np.linspace(ego_v, tj.velocity[n - 1], n)
                        tj.velocity[:n] = np.maximum(0.0, np.minimum(tj.velocity[:n], ramp))
                        log.debug(f"Velocity ramp applied: {ego_v:.1f} -> {tj.velocity[n-1]:.1f} m/s over {n} waypoints")
                    _g_start = self.global_trajectory.get_closest_waypoint_frm_sd(current_plan.start.s, 0)
                    _g_end = self.global_trajectory.get_closest_waypoint_frm_sd(current_plan.end.s, 0)
                    _gv = np.asarray(self.global_trajectory.velocity)[_g_start:_g_end + 1]
                    _lv = np.asarray(tj.velocity)
                    if len(_gv) > 0:
                        log.info(
                            f"Plan velocity — local: start={float(_lv[0]):.1f} mean={float(np.mean(_lv)):.1f} m/s | "
                            f"global_ref: start={float(_gv[0]):.1f} mean={float(np.mean(_gv)):.1f} m/s | "
                            f"discrepancy(mean)={float(np.mean(_gv)) - float(np.mean(_lv)):+.1f} m/s"
                        )
                else:
                    log.debug("Keeping current plan (wait time not elapsed)")

            log.debug(
                f"Sampled Lattice has {len(self.lattice.edges)} edges and {len(self.lattice.nodes)} nodes"
            )
        elif len(self.lattice.level0_edges) != 0:
            # No collision-free edges — pick edge with latest collision (more reaction time)
            edges_sorted = sorted(
                self.lattice.level0_edges,
                key=lambda e: getattr(e, 'collision_idx', 0),
                reverse=True
            )
            self.set_selected_plan(edges_sorted[0])
            vel = getattr(self.selected_local_plan, 'collision_agent_velocity', 0)
            idx = getattr(self.selected_local_plan, 'collision_idx', 0)

            if not self.selected_local_plan.collision:
                # All edges have boundary violations only — no real collision.
                # The boundary-violation fallback above should have handled this;
                # if we still end up here, keep the existing velocity profile unchanged.
                log.warning("Emergency branch: all edges have boundary violations but no collision "
                            "— keeping current velocity profile")
                return

            log.warning(f"No feasible edges. Collision at idx {idx}, initiating speed-match/emergency stop.")

            tj = self.selected_local_plan.local_trajectory
            current_vel = self.pm.ego_vehicle.velocity if self.pm.ego_vehicle.velocity > 0 else (tj.velocity[0] if len(tj.velocity) > 0 else 0)
            target_vel = max(0.0, vel)  # match obstacle speed, not necessarily zero

            if self.controller is not None:
                max_decel = abs(self.controller.ego_min_acceleration) * self._stopping_decel_factor
            else:
                max_decel = self._fallback_deceleration
            if max_decel < 0.1:
                max_decel = self._fallback_deceleration

            # Distance needed to decelerate from current_vel to target_vel
            stopping_distance = max(0.0, current_vel**2 - target_vel**2) / (2 * max_decel)

            # Calculate distance to collision point
            collision_distance = 0.0
            for i in range(1, min(idx + 1, len(tj.path_x))):
                collision_distance += np.sqrt(
                    (tj.path_x[i] - tj.path_x[i-1])**2 +
                    (tj.path_y[i] - tj.path_y[i-1])**2
                )

            log.warning(f"Collision distance: {collision_distance:.1f}m, Speed-match distance: {stopping_distance:.1f}m, "
                        f"Current vel: {current_vel:.1f}m/s, Target vel: {target_vel:.1f}m/s")

            if stopping_distance >= collision_distance - self._stopping_safety_buffer:
                # Not enough room — ramp from current speed down to obstacle speed over the trajectory
                log.warning(f"Cannot match speed in time — ramping to obstacle speed {target_vel:.1f} m/s")
                tj.velocity = np.maximum(0.0, np.linspace(current_vel, target_vel, len(tj.path)))
            else:
                # Enough room — hold current speed then decelerate smoothly to target_vel
                cumulative_dist = 0.0
                brake_start_idx = 0
                target_brake_dist = collision_distance - stopping_distance - self._stopping_safety_buffer

                for i in range(1, len(tj.path_x)):
                    cumulative_dist += np.sqrt(
                        (tj.path_x[i] - tj.path_x[i-1])**2 +
                        (tj.path_y[i] - tj.path_y[i-1])**2
                    )
                    if cumulative_dist >= target_brake_dist:
                        brake_start_idx = i
                        break

                new_velocity = np.empty(len(tj.path))
                for i in range(len(tj.path)):
                    if i <= brake_start_idx:
                        new_velocity[i] = current_vel
                    else:
                        progress = (i - brake_start_idx) / max(1, len(tj.path) - brake_start_idx - 1)
                        new_velocity[i] = max(target_vel, current_vel - progress * (current_vel - target_vel))

                tj.velocity = new_velocity
                log.info(f"Speed-match profile: hold {current_vel:.1f} m/s until idx {brake_start_idx}, "
                         f"then ramp to {target_vel:.1f} m/s")
        else:
            # No edges at all - emergency stop
            log.error("No lattice edges generated - emergency stop")
            if self.selected_local_plan is not None:
                tj = self.selected_local_plan.local_trajectory
                stop_vel = self.pm.ego_vehicle.velocity if self.pm.ego_vehicle.velocity > 0 else (float(tj.velocity[0]) if len(tj.velocity) > 0 else 0.0)
                tj.velocity = np.maximum(0.0, np.linspace(stop_vel, 0.0, len(tj.path)))

    def _on_edge_traversed(self) -> None:
        self._partial_replan()

    def _partial_replan(self) -> None:
        """Slide the planning horizon forward by appending one new edge at the tail.

        Finds the last committed edge in the chain (where selected_next_local_plan
        is None), samples candidate nodes one maneuver_distance further along the
        reference, generates Edges from the tail node to each candidate, runs
        collision checking, and assigns the best feasible edge as the new tail.

        Falls back to a full replan when no committed chain exists yet.
        """
        if self.selected_local_plan is None:
            log.debug("_partial_replan: no current plan, falling back to full replan")
            self.replan()
            return

        # Walk to the tail of the current committed chain.
        tail_plan = self.selected_local_plan
        while tail_plan.selected_next_local_plan is not None:
            tail_plan = tail_plan.selected_next_local_plan

        tail_node: Node = tail_plan.end
        s_new = tail_node.s + self.maneuver_distance

        if s_new > self.global_trajectory.path_s[-2]:
            log.debug("partial_replan: approaching track end, skipping extension")
            return

        # Sample candidate nodes at s_new (one on reference line, rest random).
        candidate_nodes: list[Node] = []

        wp_ref = self.global_trajectory.get_closest_waypoint_frm_sd(s_new, 0)
        _, d_ref = self.global_trajectory.get_sd_by_waypoint(wp_ref)
        x_ref, y_ref = self.global_trajectory.convert_sd_to_xy(s_new, d_ref)
        candidate_nodes.append(Node(s_new, d_ref, x_ref, y_ref))

        d_left = self.lattice.ref_left_boundary_d[wp_ref] - self.boundary_clearance
        d_right = self.lattice.ref_right_boundary_d[wp_ref] + self.boundary_clearance
        for _ in range(self.sample_size - 1):
            d_rnd = np.random.uniform(d_left, d_right)
            x_rnd, y_rnd = self.global_trajectory.convert_sd_to_xy(s_new, d_rnd)
            candidate_nodes.append(Node(s_new, d_rnd, x_rnd, y_rnd))

        # Build obstacle polygons once for all candidate edges.
        obstacle_polygons = None
        if len(self.pm.agent_vehicles) > 0:
            ego_vel = max(self.pm.ego_vehicle.velocity, PlanningSettings.c20_default_ego_velocity)
            obstacle_polygons = precompute_obstacle_polygons(
                self.pm,
                total_time=self.maneuver_distance / ego_vel,
                min_velocity_threshold=PlanningSettings.c20_min_velocity_threshold,
                obstacle_inflation_margin=PlanningSettings.c20_obstacle_inflation_margin,
            )

        # Create and evaluate edges from tail_node to each candidate.
        new_edges: list[Edge] = []
        for node in candidate_nodes:
            edge = Edge(
                start=tail_node,
                end=node,
                global_tj=self.global_trajectory,
                num_of_points=self.num_of_edge_points,
            )
            edge.collision, edge.collision_idx, edge.collision_agent_velocity = check_collision(
                self.pm, edge.local_trajectory,
                obstacle_polygons=obstacle_polygons,
                min_velocity_threshold=PlanningSettings.c20_min_velocity_threshold,
                collision_safety_margin=PlanningSettings.c20_collision_safety_margin,
                default_ego_velocity=PlanningSettings.c20_default_ego_velocity,
            )
            edge.boundary_violation = self.lattice._check_boundary_violation(edge)
            new_edges.append(edge)

        feasible = [e for e in new_edges if not e.collision and not e.boundary_violation and self._is_curvature_feasible(e)]
        if not feasible and self._allow_curvature_fallback:
            feasible = [e for e in new_edges if not e.collision and not e.boundary_violation]
            if feasible:
                log.debug("partial_replan: no curvature-feasible edges, using collision-free only")
        if not feasible and self._allow_boundary_violation_fallback:
            feasible = [e for e in new_edges if not e.collision]
            if feasible:
                log.warning("partial_replan: all edges have boundary violations — "
                            "using collision-free edge despite boundary violations")

        if feasible:
            best = self._select_best_edge(feasible)
            tail_plan.selected_next_local_plan = best
            log.debug(
                "_partial_replan: extended chain by 1 edge (tail s=%.1f -> s=%.1f, d=%.2f)",
                tail_node.s, s_new, best.end.d,
            )
        else:
            log.warning("_partial_replan: no feasible extension edges at s=%.1f", s_new)
