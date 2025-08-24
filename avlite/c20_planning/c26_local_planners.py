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
    from c20_planning.c28_trajectory import Trajectory

log = logging.getLogger(__name__)


class GreedyLatticePlanner(LocalPlannerStrategy):
    def __init__( self, global_plan: GlobalPlan, env: PerceptionModel, setting: Type[PlanningSettings] = PlanningSettings):

        super().__init__(global_plan=global_plan, pm=env, num_of_edge_points=setting.num_of_edge_points, planning_horizon=setting.planning_horizon,)
        self.maneuver_distance: float = setting.maneuver_distance
        self.boundary_clearance: float = setting.boundary_clearance
        self.sample_size: int = setting.sample_size
        self.match_speed_wp_buffer: int = setting.match_speed_wp_buffer
        # TODO: 
        self.lattice.targetted_num_edges = setting.sample_size * setting.sample_size**(self.planning_horizon - 1)

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

        no_collision_edges = [edge for edge in self.lattice.level0_edges if not edge.collision]
        if no_collision_edges:
            sorted_edges = sorted(no_collision_edges, key=lambda edge: abs(edge.end.d))
            edge = sorted_edges[0]

            current_plan = edge
            while edge is not None and len(edge.next_edges) > 0:
                no_collision_edges = [edge for edge in edge.next_edges if not edge.collision]
                if not no_collision_edges:
                    edge.selected_next_local_plan = None
                    break
                sorted_edges = sorted(no_collision_edges, key=lambda edge: abs(edge.end.d))
                edge.selected_next_local_plan = sorted_edges[0]
                edge = edge.selected_next_local_plan

            
            # self.selected_local_plan = current_plan if self.selected_local_plan is None else self.selected_local_plan
            log.debug(f"current plan len {self.local_plan_len(current_plan)}")
            if self.local_plan_len(current_plan) == self.planning_horizon: # If this plan is infeasible, revert to previous plan
                log.debug("Reverting to previous plan")
                self.selected_local_plan = current_plan
                # log.info(f"local plan len {self.local_plan_len()}")
            
            log.debug(
                f"Sampled Lattice has {len(self.lattice.edges)} edges and {len(self.lattice.nodes)} nodes"
            )
        elif len(self.lattice.level0_edges) != 0:
            self.selected_local_plan = self.lattice.level0_edges[0]
            vel = self.selected_local_plan.collision_agent_velocity
            idx = self.selected_local_plan.collision_idx
            log.warning(f"No feasible edges found in the lattice. Matching agent speed at idx {idx} with velocity {vel:.2f} m/s. Reverting to previous plan.")
            tj = self.selected_local_plan.local_trajectory
            current_vel = tj.velocity[0]
            tj.velocity = np.linspace(current_vel, 0, max(0,idx - self.match_speed_wp_buffer))
            tj.velocity = np.concatenate((tj.velocity, np.zeros(len(tj.path) - idx + self.match_speed_wp_buffer)))  # pad with zeros

            # log.warning(f"local velocity profile {self.selected_local_plan.local_trajectory.velocity} of length {len(self.selected_local_plan.local_trajectory.velocity)} vs {len(tj.path)}.")



