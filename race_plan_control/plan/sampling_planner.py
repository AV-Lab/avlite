from race_plan_control.plan.planner import Planner
from race_plan_control.plan.lattice import LatticeGraph, EdgeTmp
import numpy as np
import logging


log = logging.getLogger(__name__)


class RNDPlanner(Planner):
    def __init__(
        self,
        reference_path,
        ref_left_boundary_d,
        ref_right_boundary_d,
        planning_horizon=10,
        minimum_s_distance=50,
        minimum_boundary_distance=4,
    ):

        super().__init__(reference_path, ref_left_boundary_d, ref_right_boundary_d)

        self.planning_horizon = planning_horizon
        self.minimum_planning_distance = minimum_s_distance
        self.minimum_boundary_distance = minimum_boundary_distance

    def replan(self, sample_size=1, back_to_ref_horizon=10, sample=True):
        if len(self.traversed_s) == 0:
            log.debug("Location unkown. Cannot replan")
            return

        s = self.traversed_s[-1]
        d = self.traversed_d[-1]

        # delete old edges that already passed its starting point
        self.lattice_graph = {}

        # TODO
        # 1. Sample points from Frenet space
        # 2. Generate edges from the sampled points in the Eucledian space

        # ---------------------------------------------------------------------
        ### Group 1
        # ---------------------------------------------------------------------
        # add edge on the reference trajectory
        target_wp = (self.global_trajectory.current_wp + back_to_ref_horizon) % len(
            self.global_trajectory.path_s
        )
        s1_ = self.global_trajectory.path_s[target_wp]
        d1_ = 0
        ep1 = EdgeTmp(
            s,
            d,
            s1_,
            d1_,
            self.global_trajectory,
            num_of_points=back_to_ref_horizon + 2,
        )
        self.lattice_graph[(s1_, d1_)] = ep1

        s2_ = s1_ + 10
        d2_ = 5
        ep2 = EdgeTmp(
            s1_,
            d1_,
            s2_,
            d2_,
            self.global_trajectory,
            num_of_points=back_to_ref_horizon + 2,
        )
        ep1.append_next_edges(ep2)

        s3_ = s2_ + 10
        d3_ = 0
        ep3 = EdgeTmp(
            s2_,
            d2_,
            s3_,
            d3_,
            self.global_trajectory,
            num_of_points=back_to_ref_horizon + 2,
        )
        ep2.append_next_edges(ep3)

        # current_wp = self.global_trajectory.current_wp
        # if sample:
        # for _ in range(sample_size):
        #     s_e = np.random.uniform(self.minimum_planning_distance,self.planning_horizon)
        #     s_ = self.traversed_s[-1] + s_e
        #     d_ = np.random.uniform(self.ref_left_boundary_d[current_wp]-self.minimum_boundary_distance,
        #                            self.ref_right_boundary_d[current_wp]+self.minimum_boundary_distance)
        #     log.info(f"Sampling: ({s_:.2f},{d_:.2f})")
        #
        #     ep = Edge(s,d, s_, d_, self.global_trajectory)
        #     self.lattice_graph[(s_,d_)] = ep

        # ---------------------------------------------------------------------
        ### Group 2
        # ---------------------------------------------------------------------
        # group2_edges = []
        # if sample:
        #     for _ in range(sample_size):
        #         current_edges = list(self.lattice_graph.values())
        #         for e in current_edges:
        #             s = e.end_s
        #             d = e.end_d
        #             s_e = np.random.uniform(self.minimum_planning_distance,self.planning_horizon)
        #             # Also making sure that the sampled d is within the boundary
        #             d_ = np.random.uniform(self.ref_left_boundary_d[current_wp]-self.minimum_boundary_distance,
        #                                    self.ref_right_boundary_d[current_wp]+self.minimum_boundary_distance)
        #
        #             s_ = e.end_s + s_e
        #
        #             log.info(f"Group 2 Sampling: ({s_:.2f},{d_:.2f})")
        #
        #             # d_1st_derv = e.local_trajectory.poly.coef[-2]
        #             # d_2nd_derv = e.local_trajectory.poly.coef[-3]
        #             # ep = Edge(s,d, s_, d_, self.global_trajectory, d_1st_derv=d_1st_derv, d_2nd_derv=d_2nd_derv)
        #             x_1st_derv = e.local_trajectory.poly_x.coef[-2]
        #             x_2nd_derv = e.local_trajectory.poly_x.coef[-3]
        #
        #             y_1st_derv = e.local_trajectory.poly_y.coef[-2]
        #             y_2nd_derv = e.local_trajectory.poly_y.coef[-3]
        #
        #             ep = Edge(s,d, s_, d_, self.global_trajectory, x_1st_derv=x_1st_derv, x_2nd_derv=x_2nd_derv, y_1st_derv=y_1st_derv, y_2nd_derv=y_2nd_derv)
        #
        #
        #             e.append_next_edges(ep)
        #             e.selected_next_edge =  np.random.choice(e.next_edges) # if len(e.next_edges) > 0 else None
        #             group2_edges.append(ep)

        # ---------------------------------------------------------------------
        ### Group 3
        # ---------------------------------------------------------------------
        # if sample:
        #     for _ in range(sample_size):
        #         for e in group2_edges:
        #             s = e.end_s
        #             d = e.end_d
        #             # s_e = np.random.uniform(self.minimum_planning_distance,self.planning_horizon)
        #
        #             # d_ = np.random.uniform(self.ref_left_boundary_d[current_wp]-self.minimum_boundary_distance,
        #                                    # self.ref_right_boundary_d[current_wp]+self.minimum_boundary_distance)
        #
        #             # d_1st_derv = e.local_trajectory.poly.coef[-2]
        #             # d_2nd_derv = e.local_trajectory.poly.coef[-3]
        #             # ep = Edge(s,d, s_, d_, self.global_trajectory, d_1st_derv=d_1st_derv, d_2nd_derv=d_2nd_derv)
        #             x_1st_derv = e.local_trajectory.poly_x.coef[-2]
        #             x_2nd_derv = e.local_trajectory.poly_x.coef[-3]
        #
        #             y_1st_derv = e.local_trajectory.poly_y.coef[-2]
        #             y_2nd_derv = e.local_trajectory.poly_y.coef[-3]
        #             ep = Edge(s,d, s_, d_, self.global_trajectory, x_1st_derv=x_1st_derv, x_2nd_derv=x_2nd_derv, y_1st_derv=y_1st_derv, y_2nd_derv=y_2nd_derv)
        #
        #             e.append_next_edges(ep)
        #             e.selected_next_edge =  np.random.choice(e.next_edges) # if len(e.next_edges) > 0 else None
        #
        # ---------------------------------------------------------------------
        ## Plan --------------------------------------------------------------
        # ---------------------------------------------------------------------

        # Select a random edge from the lattice graph
        self.selected_edge = np.random.choice(list(self.lattice_graph.values()))
        self.selected_edge.is_selected = True
        self.selected_edge.selected_next_edge = (
            np.random.choice(self.selected_edge.next_edges)
            if len(self.selected_edge.next_edges) > 0
            else None
        )


if __name__ == "__main__":
    import race_plan_control.main as main

    main.run()
