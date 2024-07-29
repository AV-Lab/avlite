from race_plan_control.plan.planner import Planner, LatticeGraph
import numpy as np
import logging


log = logging.getLogger(__name__)

class RNDPlanner(Planner):
    def __init__(self, reference_path, ref_left_boundary_d, ref_right_boundary_d, planning_horizon = 15, minimum_s_distance=5,
                  minimum_boundary_distance=1.5):
        super().__init__(reference_path, ref_left_boundary_d, ref_right_boundary_d)
        
        self.planning_horizon = planning_horizon
        self.minimum_planning_distance = minimum_s_distance
        self.minimum_boundary_distance = minimum_boundary_distance

    def replan(self, sample_size=2, back_to_ref_horizon=10, sample=True):
        if len(self.past_s) == 0:
            log.debug("No data to replan")
            return

        s = self.past_s[-1]
        d = self.past_d[-1]
        
        # delete old edges that already passed its starting point
        self.lattice_graph = {} 

        # TODO
        # 1. Sample points from Frenet space
        # 2. Generate edges from the sampled points in the Eucledian space

        ### Group 1
        # add edge on the reference trajectory
        idx = (self.global_trajectory.next_wp + back_to_ref_horizon)%len(self.global_trajectory.path_s)
        next_s = self.global_trajectory.path_s[idx]
        next_d = 0
        ep = LatticeGraph.Edge(s,d, next_s, next_d, self.global_trajectory, num_of_points= back_to_ref_horizon+2)
        self.lattice_graph[(next_s,next_d)] = ep

        # sample new edges
        if sample:
            for _ in range(sample_size):
                s_e = np.random.uniform(self.minimum_planning_distance,self.planning_horizon)
                s_ = self.past_s[-1] + s_e
                d_ = np.random.uniform(self.ref_left_boundary_d[-1]-self.minimum_boundary_distance, self.ref_right_boundary_d[-1]+self.minimum_boundary_distance)
                log.info(f"Sampling: ({s_:.2f},{d_:.2f})")
                
                ep = LatticeGraph.Edge(s,d, s_, d_, self.global_trajectory)
                self.lattice_graph[(s_,d_)] = ep
        ### Group 2
        if sample:
            for _ in range(sample_size):
                s_e = np.random.uniform(self.minimum_planning_distance,self.planning_horizon)
                d_ = np.random.uniform(self.ref_left_boundary_d[-1]-self.minimum_boundary_distance, self.ref_right_boundary_d[-1]+self.minimum_boundary_distance)
                current_edges = list(self.lattice_graph.values())
                for e in current_edges:
                    s_ = e.end_s + s_e
                    s = e.end_s
                    d = e.end_d
                    ep = LatticeGraph.Edge(s,d, s_, d_, self.global_trajectory)
                    e.append_next_edges(ep)
                    e.selected_next_edge =  np.random.choice(e.next_edges) # if len(e.next_edges) > 0 else None
        
        ### Plan
        # Select a random edge from the lattice graph
        self.selected_edge = np.random.choice(list(self.lattice_graph.values()))
        self.selected_edge.is_selected = True
        self.selected_edge.selected_next_edge =  np.random.choice(self.selected_edge.next_edges) if len(self.selected_edge.next_edges) > 0 else None

if __name__ == "__main__":
    import race_plan_control.main as main
    main.run()