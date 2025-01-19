from c10_perceive.c11_environment import Environment
from c20_plan.c21_planner import Planner
from c20_plan.c23_lattice import Lattice
import numpy as np
import logging


log = logging.getLogger(__name__)


class RNDPlanner(Planner):
    def __init__(
        self,
        global_path: list[tuple[float, float]],
        ref_left_boundary_d: list[float],
        ref_right_boundary_d: list[float],
        env: Environment,
        num_of_edge_points=10,
        planning_horizon=3,
        maneuver_distance=20,
        boundary_clearance=1,
        sample_size=3,
    ):
        super().__init__(global_path, ref_left_boundary_d, ref_right_boundary_d, env=env)
        self.planning_horizon: int = planning_horizon
        self.maneuver_distance: float = maneuver_distance
        self.boundary_clearance: int = boundary_clearance
        self.sample_size: int = sample_size
        self.num_of_edge_points: int = num_of_edge_points

        self.lattice = Lattice(
            self.global_trajectory,
            self.ref_left_boundary_d,
            self.ref_right_boundary_d,
            planning_horizon=self.planning_horizon,
            num_of_points=self.num_of_edge_points,
        )

    def replan(self, back_to_ref_horizon=10):
        if len(self.traversed_s) == 0:
            log.debug("Location unkown. Cannot replan")
            return

        # delete previous plans
        self.lattice.reset()
        self.lattice.sample_nodes(
            s=self.location_sd[0],
            d=self.location_sd[1],
            maneuver_distance=self.maneuver_distance,
            boundary_clearance=self.boundary_clearance,
            sample_size=self.sample_size,
        )
        self.lattice.generate_lattice_from_nodes()
        self.selected_local_plan = np.random.choice(self.lattice.level0_edges) if len(self.lattice.level0_edges) > 0 else None
        edge = self.selected_local_plan
        while edge is not None and len(edge.next_edges) > 0:
            edge.selected_next_local_plan = np.random.choice(edge.next_edges)
            edge = edge.selected_next_local_plan

        log.debug(f"Sampled Lattice has {len(self.lattice.edges)} edges and {len(self.lattice.nodes)} nodes") 

        return


