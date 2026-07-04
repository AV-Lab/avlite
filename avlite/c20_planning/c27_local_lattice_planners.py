from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import math
import time

import numpy as np

from avlite.c10_perception.c12_perception_strategy import PerceptionModel
from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c20_planning.c21_planning_model import GlobalPlan, LocalPlan
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c20_planning.c26_local_planners import VelocityLocalPlanner
from avlite.c20_planning.c28_lattice import Lattice, Node, Edge
from avlite.c20_planning.c29_settings import PlanningSettings, PlanningSettingsSchema
from avlite.c50_common.c54_collision_checking import check_collision, precompute_obstacle_polygons

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
                 setting: PlanningSettingsSchema = PlanningSettings):
        super().__init__(global_plan=global_plan, pm=pm, controller=controller, setting=setting)

        self.planning_horizon: int = planning_horizon
        self.num_of_edge_points: int = num_of_edge_points
        self.selected_local_plan: Optional[Edge] = None
        self._committed_trajectory = None
        self.lattice: Lattice = Lattice(
            self.global_trajectory, global_plan.left_boundary_d, global_plan.right_boundary_d,
            planning_horizon=self.planning_horizon, num_of_points=self.num_of_edge_points)

        # Replan stability: track when plan was last changed
        self._last_plan_change_time: float = 0.0
        self._replan_wait_time: float = setting.c27_replan_wait_time
        self._min_edge_progress_to_block: float = setting.c27_min_edge_progress_to_block
        self._urgent_collision_threshold: int = setting.c27_urgent_collision_threshold
        self._disconnect_distance_threshold: float = setting.c27_disconnect_distance_threshold

    def set_global_plan(self, global_plan: GlobalPlan, ego_xy=None) -> None:
        super().set_global_plan(global_plan, ego_xy=ego_xy)
        self.lattice = Lattice(
            self.global_trajectory, global_plan.left_boundary_d, global_plan.right_boundary_d,
            planning_horizon=self.planning_horizon, num_of_points=self.num_of_edge_points)

    def reset(self, wp: int = 0):
        super().reset(wp)
        self.selected_local_plan = None
        self._committed_trajectory = None
        self.lattice.reset()
        self._last_plan_change_time = 0.0

    def get_local_plan(self) -> LocalPlan:
        if self._committed_trajectory is not None:
            return LocalPlan.from_trajectory(self._committed_trajectory)
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

        new_clean = not new_plan.collision and not new_plan.boundary_violation
        if new_clean:
            prev_len = self.local_plan_len()
            new_len = self.local_plan_len(new_plan)
            waited = time.time() - self._last_plan_change_time >= self._replan_wait_time
            cur_v = float(np.mean(np.asarray(cur_vel))) if cur_vel is not None and len(cur_vel) > 0 else 0.0
            new_v = float(np.mean(np.asarray(new_plan.local_trajectory.velocity)))
            material_gain = new_len >= prev_len + 2
            speed_gain = new_v > cur_v + 0.5

            old_colliding = False
            edge = self.selected_local_plan
            while edge is not None:
                if edge.collision:
                    old_colliding = True
                    break
                edge = edge.selected_next_local_plan

            wants_switch = (new_len > prev_len) or old_colliding
            if wants_switch and (waited or material_gain):
                return True
            if speed_gain and waited:
                return True

        # Commit to current plan — do not switch
        return False

    def set_selected_plan(self, new_plan: Edge) -> None:
        """Set the selected plan and update the change timestamp."""
        self.selected_local_plan = new_plan
        self._last_plan_change_time = time.time()
        traj = self.selected_local_plan.local_trajectory
        edge = self.selected_local_plan.selected_next_local_plan
        while edge is not None:
            traj = traj.concatenate(edge.local_trajectory)
            edge = edge.selected_next_local_plan
        traj.update_waypoint_by_xy(self.location_xy[0], self.location_xy[1])
        self._committed_trajectory = traj

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

        if self._committed_trajectory is not None:
            self._committed_trajectory.update_waypoint_by_xy(state.x, state.y)

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
        if self._committed_trajectory is not None:
            self._committed_trajectory.update_waypoint_by_xy(x_new, y_new)

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
    def __init__(self, global_plan: GlobalPlan, env: PerceptionModel, setting: PlanningSettingsSchema = PlanningSettings, controller=None):

        super().__init__(global_plan=global_plan, pm=env, num_of_edge_points=setting.c27_num_of_edge_points, planning_horizon=setting.c27_planning_horizon, controller=controller, setting=setting)
        self.maneuver_distance: float = setting.c27_maneuver_distance
        self.boundary_clearance: float = setting.c27_boundary_clearance
        self.sample_size: int = setting.c27_sample_size
        self.match_speed_wp_buffer: int = setting.c27_match_speed_wp_buffer
        self.safety_margin_weight: float = setting.c27_safety_margin_weight
        self.max_lateral_accel: float = setting.c27_max_lateral_accel
        self.min_curvature_velocity: float = setting.c27_min_curvature_velocity
        self._min_ramp_start_velocity: float = setting.c27_min_ramp_start_velocity
        self._allow_curvature_fallback: bool = setting.c27_allow_curvature_fallback
        self._allow_boundary_violation_fallback: bool = setting.c27_allow_boundary_violation_fallback
        self._velocity_planner = VelocityLocalPlanner(global_plan, env, controller, setting)

    def set_global_plan(self, global_plan: GlobalPlan, ego_xy=None) -> None:
        super().set_global_plan(global_plan, ego_xy=ego_xy)
        self._velocity_planner.set_global_plan(global_plan, ego_xy=ego_xy)

    def reset(self, wp: int = 0):
        super().reset(wp)
        self._velocity_planner.reset(wp)

    def _profile_edge_velocity(self, edge: Edge) -> None:
        if not edge.collision:
            return
        ref_vel = np.asarray(edge.local_trajectory.velocity, dtype=float)
        self._velocity_planner.apply_speed_match(
            edge.local_trajectory,
            edge.collision_idx,
            max(0.0, edge.collision_agent_velocity),
            ref_velocity=ref_vel,
        )

    def _profile_lattice_edges(self, edges: list[Edge] | None = None) -> None:
        for edge in edges if edges is not None else self.lattice.edges:
            self._profile_edge_velocity(edge)

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
        d0_edges = [e for e in edges if abs(e.end.d) < PlanningSettings.c27_d0_reference_threshold]
        if d0_edges:
            return min(d0_edges, key=self._edge_cost)
        return min(edges, key=self._edge_cost)

    def _agent_blocks_ahead(self) -> bool:
        if len(self.pm.agent_vehicles) == 0:
            return False
        ego = self.pm.ego_vehicle
        ego_heading = np.array([math.cos(ego.theta), math.sin(ego.theta)])
        s_horizon = self.location_sd[0] + self.planning_horizon * self.maneuver_distance
        for agent in self.pm.agent_vehicles:
            to_agent = np.array([agent.x - ego.x, agent.y - ego.y])
            if float(np.dot(ego_heading, to_agent)) < 0:
                continue
            agent_s, _ = self.global_trajectory.convert_xy_to_sd(agent.x, agent.y)
            if agent_s <= s_horizon:
                return True
        return False

    def _feasible_candidates(self, edges: list[Edge], agent_blocks_ahead: bool) -> list[Edge]:
        relax_boundary = agent_blocks_ahead or self._allow_boundary_violation_fallback
        relax_curvature = self._allow_curvature_fallback
        candidates = [
            e for e in edges
            if not e.collision and not e.boundary_violation and self._is_curvature_feasible(e)
        ]
        if not candidates and relax_curvature:
            candidates = [e for e in edges if not e.collision and not e.boundary_violation]
        if not candidates and relax_boundary:
            candidates = [e for e in edges if not e.collision]
        return candidates

    def _build_selected_chain(self, feasible_level0: list[Edge], agent_blocks_ahead: bool) -> Edge:
        edge = self._select_best_edge(feasible_level0)
        current_plan = edge
        d0_threshold = PlanningSettings.c27_d0_reference_threshold
        while edge is not None and len(edge.next_edges) > 0:
            next_feasible = self._feasible_candidates(edge.next_edges, agent_blocks_ahead)
            if not next_feasible:
                edge.selected_next_local_plan = None
                break
            candidates = next_feasible
            if agent_blocks_ahead:
                lateral = [e for e in candidates if abs(e.end.d) >= d0_threshold]
                if lateral:
                    candidates = lateral
            edge.selected_next_local_plan = self._select_best_edge(candidates)
            edge = edge.selected_next_local_plan
        return current_plan

    def _fill_planning_horizon(self) -> None:
        while self.local_plan_len() < self.planning_horizon:
            prev_len = self.local_plan_len()
            self._partial_replan()
            if self.local_plan_len() <= prev_len:
                break

    def replan(self, back_to_ref_horizon=10):
        if len(self.traversed_s) == 0:
            log.debug("Location unkown. Cannot replan")
            return

        track_end_s = self.global_trajectory.path_s[-2]
        if self.location_sd[0] + self.maneuver_distance > track_end_s:
            log.debug("replan: approaching track end, hand off to global decel profile")
            self.selected_local_plan = None
            self._committed_trajectory = None
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
        self._profile_lattice_edges()

        agent_blocks_ahead = self._agent_blocks_ahead()
        feasible_edges = self._feasible_candidates(self.lattice.level0_edges, agent_blocks_ahead)

        if feasible_edges:
            current_plan = self._build_selected_chain(feasible_edges, agent_blocks_ahead)

            new_len = self.local_plan_len(current_plan)
            prev_len = self.local_plan_len() if self.selected_local_plan else None
            log.debug(f"current plan len {new_len}")
            new_clean = not current_plan.collision and not current_plan.boundary_violation
            old_colliding = False
            if self.selected_local_plan is not None:
                edge = self.selected_local_plan
                while edge is not None:
                    if edge.collision:
                        old_colliding = True
                        break
                    edge = edge.selected_next_local_plan
            acceptable = (prev_len is None and new_len >= 1) or (
                prev_len is not None and (
                    new_len >= prev_len
                    or abs(new_len - prev_len) <= 1
                    or (old_colliding and new_clean)
                )
            )
            if acceptable:
                # Only switch if allowed (no recent change or current plan has collision)
                if self.should_switch_plan(current_plan):
                    log.debug("Switching to new plan")
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
                    self.set_selected_plan(current_plan)
                    self._fill_planning_horizon()
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
            passing_feasible = self._feasible_candidates(self.lattice.level0_edges, agent_blocks_ahead=True)
            if passing_feasible:
                passing_plan = self._build_selected_chain(passing_feasible, agent_blocks_ahead=True)
                if not passing_plan.collision and self.should_switch_plan(passing_plan):
                    log.debug("Emergency avoided: committing collision-free passing chain")
                    self.set_selected_plan(passing_plan)
                    self._fill_planning_horizon()
                    return

            # No collision-free edges — pick edge with latest collision (more reaction time)
            edges_sorted = sorted(
                self.lattice.level0_edges,
                key=lambda e: getattr(e, 'collision_idx', 0),
                reverse=True
            )
            if self.should_switch_plan(edges_sorted[0]):
                self.set_selected_plan(edges_sorted[0])
            else:
                return
            idx = getattr(self.selected_local_plan, 'collision_idx', 0)

            if not self.selected_local_plan.collision:
                # All edges have boundary violations only — no real collision.
                # The boundary-violation fallback above should have handled this;
                # if we still end up here, keep the existing velocity profile unchanged.
                log.debug("Emergency branch: all edges have boundary violations but no collision "
                            "— keeping current velocity profile")
                return

            log.warning(f"No feasible edges. Collision at idx {idx}, initiating speed-match/emergency stop.")
        else:
            # No edges at all - emergency stop
            log.error("No lattice edges generated - emergency stop")
            if self.selected_local_plan is not None:
                tj = self.selected_local_plan.local_trajectory
                self._velocity_planner.apply_speed_match(tj, len(tj.path) - 1, 0.0)
                traj = self.selected_local_plan.local_trajectory
                edge = self.selected_local_plan.selected_next_local_plan
                while edge is not None:
                    traj = traj.concatenate(edge.local_trajectory)
                    edge = edge.selected_next_local_plan
                traj.update_waypoint_by_xy(self.location_xy[0], self.location_xy[1])
                self._committed_trajectory = traj

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

        self._profile_lattice_edges(new_edges)

        agent_blocks_ahead = self._agent_blocks_ahead()
        feasible = self._feasible_candidates(new_edges, agent_blocks_ahead)
        if agent_blocks_ahead:
            d0_threshold = PlanningSettings.c27_d0_reference_threshold
            lateral = [e for e in feasible if abs(e.end.d) >= d0_threshold]
            if lateral:
                feasible = lateral

        if feasible:
            best = self._select_best_edge(feasible)
            tail_plan.selected_next_local_plan = best
            traj = self.selected_local_plan.local_trajectory
            edge = self.selected_local_plan.selected_next_local_plan
            while edge is not None:
                traj = traj.concatenate(edge.local_trajectory)
                edge = edge.selected_next_local_plan
            traj.update_waypoint_by_xy(self.location_xy[0], self.location_xy[1])
            self._committed_trajectory = traj
            log.debug(
                "_partial_replan: extended chain by 1 edge (tail s=%.1f -> s=%.1f, d=%.2f)",
                tail_node.s, s_new, best.end.d,
            )
        else:
            log.warning("_partial_replan: no feasible extension edges at s=%.1f", s_new)
