from __future__ import annotations
from typing import TYPE_CHECKING
from c10_perceive.c11_base_perception import PerceptionModel
from c20_plan.c24_base_local_planner import BaseLocalPlanner
from c20_plan.c26_lattice import Lattice
import numpy as np
import logging

if TYPE_CHECKING:
    from c20_plan.c27_trajectory import Trajectory

log = logging.getLogger(__name__)


class RNDPlanner(BaseLocalPlanner):
    def __init__(
        self,
        global_path: list[tuple[float, float]],
        global_velocity: list[float],
        ref_left_boundary_d: list[float],
        ref_right_boundary_d: list[float],
        env: PerceptionModel,
        num_of_edge_points=10,
        planning_horizon=4,
        maneuver_distance=35,
        boundary_clearance=1,
        sample_size=3, # number of nodes to sample in each level
    ):
        super().__init__(
            global_path,
            global_velocity, 
            ref_left_boundary_d,
            ref_right_boundary_d,
            pm=env,
            num_of_edge_points=num_of_edge_points,
            planning_horizon=planning_horizon,
        )
        self.maneuver_distance: float = maneuver_distance
        self.boundary_clearance: int = boundary_clearance
        self.sample_size: int = sample_size
        self.lattice.targetted_num_edges = 3 * 3**(self.planning_horizon - 1)

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

        self.lattice.generate_lattice_from_nodes(env=self.pm)

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


