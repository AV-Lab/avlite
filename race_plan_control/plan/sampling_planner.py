from race_plan_control.plan.planner import Planner
from race_plan_control.plan.lattice import Edge, Node, create_edge
from typing import Optional, Dict
import numpy as np
import logging
from icecream import ic
from collections import defaultdict


log = logging.getLogger(__name__)


class RNDPlanner(Planner):
    def __init__(
        self,
        reference_path,
        ref_left_boundary_d,
        ref_right_boundary_d,
        num_of_edge_points=10,
        planning_horizon=3,
        maneuver_distance=20,
        boundary_clearance=1,
        sample_size=3,
    ):
        self.planning_horizon: int = planning_horizon
        self.maneuver_distance: float = maneuver_distance
        self.boundary_clearance: int = boundary_clearance
        self.sample_size: int = sample_size

        super().__init__(
            reference_path, ref_left_boundary_d, ref_right_boundary_d, num_of_edge_points=num_of_edge_points
        )

    def replan(self, back_to_ref_horizon=10):
        if len(self.traversed_s) == 0:
            log.debug("Location unkown. Cannot replan")
            return

        # delete previous plan
        self.next_edges = []
        self.selected_next_edge = None
        self.lattice_nodes:Dict[int,list] = defaultdict(list) # key is level, value is list of nodes
        self.incoming_edges:Dict[Node,list[Edge]] = defaultdict(list) # key is node, value is incoming edge
        self.outgoing_edges:Dict[Node,list[Edge]] = defaultdict(list) # key is node, value is incoming edge

        self.__sample_nodes()
        self.__create_edges()

        # select a random edge
        self.selected_next_edge = np.random.choice(self.next_edges) if len(self.next_edges) > 0 else None
        edge: Optional[Edge] = self.selected_next_edge
        while edge is not None and len(edge.next_edges) > 0:
            edge.selected_next_edge = np.random.choice(edge.next_edges)
            edge = edge.selected_next_edge


    def __sample_nodes(self):
        s1_ = self.traversed_s[-1]
        d = self.traversed_d[-1]
        self.lattice_nodes[0].append(Node(s1_, d, self.global_trajectory))

        for l in range(1,self.planning_horizon+1):
            s1_ = s1_ + self.maneuver_distance
            if s1_ > self.global_trajectory.path_s[-2]:  # at -1 path_s is zero
                log.warn("No Replan, reaching the end of lap")
                return
            self.lattice_nodes[l].append(Node(s1_, 0, self.global_trajectory)) # always a node at track line
            for _ in np.arange(self.sample_size-1):
                target_wp = self.global_trajectory.get_closest_waypoint_frm_sd(s1_, 0)
                d1_ = np.random.uniform(
                    self.ref_left_boundary_d[target_wp] - self.boundary_clearance,
                    self.ref_right_boundary_d[target_wp] + self.boundary_clearance,
                )
                self.lattice_nodes[l].append(Node(s1_, d1_, self.global_trajectory))
            


    def __create_edges(self):
        for l in range(self.planning_horizon+1):
            for node in self.lattice_nodes[l]:
                for next_node in self.lattice_nodes[l+1]:
                    assert node != next_node
                    edge = create_edge(node, next_node, self.global_trajectory)
                    self.incoming_edges[next_node].append(edge)
                    self.outgoing_edges[node].append(edge)
                    if l == 0:
                        self.next_edges.append(edge)
                for e in self.incoming_edges[node]: 
                    for o in self.outgoing_edges[node]:
                        e.next_edges.append(o) 
                        
                



if __name__ == "__main__":
    import race_plan_control.main as main

    main.run()
