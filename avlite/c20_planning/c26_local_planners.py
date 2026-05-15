from __future__ import annotations
from typing import TYPE_CHECKING, Type
from avlite.c10_perception.c12_perception_strategy import PerceptionModel
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c20_planning.c23_local_planning_strategy import LocalPlannerStrategy
from avlite.c20_planning.c27_lattice import Lattice
from avlite.c20_planning.c29_settings import PlanningSettings
import numpy as np
import logging

if TYPE_CHECKING:
    from avlite.c60_common.c63_trajectory_tracker import TrajectoryTracker

log = logging.getLogger(__name__)


class GreedyLatticePlanner(LocalPlannerStrategy):
    def __init__( self, global_plan: GlobalPlan, env: PerceptionModel, setting: Type[PlanningSettings] = PlanningSettings):

        super().__init__(global_plan=global_plan, pm=env, num_of_edge_points=setting.num_of_edge_points, planning_horizon=setting.planning_horizon,)
        self.maneuver_distance: float = setting.maneuver_distance
        self.boundary_clearance: float = setting.boundary_clearance
        self.sample_size: int = setting.sample_size
        self.match_speed_wp_buffer: int = setting.match_speed_wp_buffer
        self._replan_wait_time: float = setting.replan_wait_time
        self.safety_margin_weight: float = setting.safety_margin_weight
        self._min_edge_progress_to_block: float = setting.min_edge_progress_to_block
        self._urgent_collision_threshold: int = setting.urgent_collision_threshold
        self.max_lateral_accel: float = setting.max_lateral_accel
        self.min_curvature_velocity: float = setting.min_curvature_velocity
        # TODO: 
        self.lattice.targetted_num_edges = setting.sample_size * setting.sample_size**(self.planning_horizon - 1)

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
        d0_edges = [e for e in edges if abs(e.end.d) < PlanningSettings.d0_reference_threshold]
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
                         if not edge.collision and self._is_curvature_feasible(edge)]
        
        # Fallback to collision-free only if no curvature-feasible edges
        if not feasible_edges:
            feasible_edges = [edge for edge in self.lattice.level0_edges if not edge.collision]
            if feasible_edges:
                log.debug("No curvature-feasible edges, using collision-free edges")
        
        if feasible_edges:
            # Select best edge considering both reference tracking and safety
            edge = self._select_best_edge(feasible_edges)

            current_plan = edge
            while edge is not None and len(edge.next_edges) > 0:
                # Filter next edges by collision and curvature
                next_feasible = [e for e in edge.next_edges 
                                if not e.collision and self._is_curvature_feasible(e)]
                if not next_feasible:
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
                else:
                    log.debug("Keeping current plan (wait time not elapsed)")
            
            log.debug(
                f"Sampled Lattice has {len(self.lattice.edges)} edges and {len(self.lattice.nodes)} nodes"
            )
        elif len(self.lattice.level0_edges) != 0:
            # No collision-free edges - pick edge with latest collision (more reaction time)
            edges_sorted = sorted(
                self.lattice.level0_edges,
                key=lambda e: getattr(e, 'collision_idx', 0),
                reverse=True
            )
            self.selected_local_plan = edges_sorted[0]
            vel = getattr(self.selected_local_plan, 'collision_agent_velocity', 0)
            idx = getattr(self.selected_local_plan, 'collision_idx', 0)
            log.warning(f"No feasible edges. Collision at idx {idx}, initiating emergency stop.")
            
            tj = self.selected_local_plan.local_trajectory
            current_vel = self.pm.ego_vehicle.velocity if self.pm.ego_vehicle.velocity > 0 else (tj.velocity[0] if len(tj.velocity) > 0 else 0)
            
            # Calculate required stopping distance: d = v^2 / (2*a_max)
            # Use conservative deceleration (not max to be safe)
            max_decel = abs(self.pm.ego_vehicle.min_acceleration) * 0.8  # 80% of max decel for safety margin
            if max_decel < 0.1:
                max_decel = 3.0  # fallback deceleration m/s^2
            
            # Calculate distance to collision point
            collision_distance = 0.0
            for i in range(1, min(idx + 1, len(tj.path_x))):
                collision_distance += np.sqrt(
                    (tj.path_x[i] - tj.path_x[i-1])**2 + 
                    (tj.path_y[i] - tj.path_y[i-1])**2
                )
            
            # Required stopping distance
            stopping_distance = (current_vel ** 2) / (2 * max_decel)
            
            log.warning(f"Collision distance: {collision_distance:.1f}m, Required stopping distance: {stopping_distance:.1f}m, Current vel: {current_vel:.1f}m/s")
            
            # If we can't stop in time, apply maximum braking immediately (set velocity to 0 everywhere)
            if stopping_distance >= collision_distance - 2.0:  # 2m safety buffer
                log.error(f"Cannot stop in time! Setting emergency brake (all velocities to 0)")
                tj.velocity = np.zeros(len(tj.path))
            else:
                # We can stop - create smooth deceleration profile
                # Find the waypoint where we need to start braking
                cumulative_dist = 0.0
                brake_start_idx = 0
                target_brake_dist = collision_distance - stopping_distance - 2.0  # Start braking with 2m margin
                
                for i in range(1, len(tj.path_x)):
                    cumulative_dist += np.sqrt(
                        (tj.path_x[i] - tj.path_x[i-1])**2 + 
                        (tj.path_y[i] - tj.path_y[i-1])**2
                    )
                    if cumulative_dist >= target_brake_dist:
                        brake_start_idx = i
                        break
                
                # Create velocity profile: maintain speed until brake_start_idx, then decelerate to 0
                new_velocity = np.zeros(len(tj.path))
                for i in range(len(tj.path)):
                    if i <= brake_start_idx:
                        new_velocity[i] = current_vel
                    else:
                        # Linear deceleration (simplified - actual would be sqrt profile)
                        progress = (i - brake_start_idx) / max(1, len(tj.path) - brake_start_idx - 1)
                        new_velocity[i] = max(0, current_vel * (1 - progress))
                
                tj.velocity = new_velocity
                log.info(f"Created braking profile: maintain {current_vel:.1f}m/s until idx {brake_start_idx}, then brake to stop")
        else:
            # No edges at all - emergency stop
            log.error("No lattice edges generated - emergency stop")
            if self.selected_local_plan is not None:
                tj = self.selected_local_plan.local_trajectory
                tj.velocity = np.zeros(len(tj.path))


