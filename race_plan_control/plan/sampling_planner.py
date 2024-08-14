from race_plan_control.plan.planner import Planner
from race_plan_control.plan.lattice import Edge, Node, create_edge
import numpy as np
import logging
from icecream import ic


log = logging.getLogger(__name__)


class RNDPlanner(Planner):
    def __init__(
        self,
        reference_path,
        ref_left_boundary_d,
        ref_right_boundary_d,
        num_of_edge_points=10,
        planning_horizon=10,
        minimum_s_distance=50,
        minimum_boundary_distance=4,
        sample_size=1,
    ):
        self.planning_horizon: int = planning_horizon
        self.minimum_planning_distance: int = minimum_s_distance
        self.minimum_boundary_distance: int = minimum_boundary_distance
        self.sample_size: int = sample_size

        super().__init__(
            reference_path, ref_left_boundary_d, ref_right_boundary_d, num_of_edge_points=num_of_edge_points
        )

    def replan(self, back_to_ref_horizon=10):
        if len(self.traversed_s) == 0:
            log.debug("Location unkown. Cannot replan")
            return

        s = self.traversed_s[-1]
        d = self.traversed_d[-1]

        # delete old edges that already passed its starting point
        self.edges = []
        self.selected_edge = None

        # ---------------------------------------------------------------------
        ### Group 1
        # ---------------------------------------------------------------------
        # add edge on the reference trajectory
        target_wp = (self.global_trajectory.current_wp + back_to_ref_horizon) % len(self.global_trajectory.path_s)
        s1_ = self.global_trajectory.path_s[target_wp]
        d1_ = 1

        s2_ = s1_ + 20
        d2_ = 2

        s3_ = s2_ + 20
        d3_ = -2

        s4_ = s3_ + 20
        d4_ = 2

        s5_ = s4_ + 20
        d5_ = 0

        ep1 = create_edge(
            Node(s, d, self.global_trajectory),
            Node(s1_, d1_, self.global_trajectory),
            self.global_trajectory,
            num_of_points=back_to_ref_horizon + 2,
        )
        self.selected_edge = ep1
        self.edges.append(ep1)

        coef = ep1.local_trajectory.poly_d.coef
        ic(coef)
        ic(ep1.local_trajectory.path_s)

        d_1st_derv = 0  
        d_2nd_derv = 0


        ep2 = create_edge(
            Node(s1_, d1_, self.global_trajectory,d_1st_derv=d_1st_derv, d_2nd_derv=d_2nd_derv),
            Node(s2_, d2_, self.global_trajectory),
            self.global_trajectory,
            num_of_points=back_to_ref_horizon + 2,
        )
        ep1.next_edges.append(ep2)
        ep1.selected_next_edge = ep2
        coef = ep2.local_trajectory.poly_d.coef
        ic(coef)
        ic(ep2.local_trajectory.path_s)

        return 

        d_1st_derv = ep2.local_trajectory.poly_d.coef[-2]
        d_2nd_derv = ep2.local_trajectory.poly_d.coef[-3]
        coef = ep2.local_trajectory.poly_d.coef
        ic(coef)
        ep3 = Edge(
            Node(s2_, d2_, self.global_trajectory,d_1st_derv=d_1st_derv, d_2nd_derv=d_2nd_derv),
            Node(s3_, d3_, self.global_trajectory),
            self.global_trajectory,
            num_of_points=back_to_ref_horizon + 2,
        )
        ep2.next_edges.append(ep3)
        ep2.selected_next_edge = ep3

        d_1st_derv = ep3.local_trajectory.poly_d.coef[-2]
        d_2nd_derv = ep3.local_trajectory.poly_d.coef[-3]
        coef = ep3.local_trajectory.poly_d.coef
        ic(coef)
        ep4 = Edge(
            Node(s3_, d3_, self.global_trajectory,d_1st_derv=d_1st_derv, d_2nd_derv=d_2nd_derv),
            Node(s4_, d4_, self.global_trajectory),
            self.global_trajectory,
            num_of_points=back_to_ref_horizon + 2,
        )
        ep3.next_edges.append(ep4)
        ep3.selected_next_edge = ep4

        d_1st_derv = ep4.local_trajectory.poly_d.coef[-2]
        d_2nd_derv = ep4.local_trajectory.poly_d.coef[-3]
        coef = ep4.local_trajectory.poly_d.coef
        ic(coef)
        ep5 = Edge(
            Node(s4_, d4_, self.global_trajectory,d_1st_derv=d_1st_derv, d_2nd_derv=d_2nd_derv),
            Node(s5_, d5_, self.global_trajectory),
            self.global_trajectory,
            num_of_points=back_to_ref_horizon + 2,
        )
        ep4.next_edges.append(ep5)
        ep4.selected_next_edge = ep5


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


if __name__ == "__main__":
    import race_plan_control.main as main

    main.run()
