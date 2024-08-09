from race_plan_control.plan.trajectory import Trajectory
from typing import Dict 

import logging
log = logging.getLogger(__name__)

class Edge:
    local_trajectory:Trajectory

    def __init__(self, start_s, start_d, end_s, end_d, global_tj:Trajectory, num_of_points = 10, d_1st_derv = None, d_2nd_derv = None):
        self.start_s = start_s
        self.start_d = start_d
        self.end_s = end_s
        self.end_d = end_d
        self.num_of_points = num_of_points
        if d_1st_derv is None or d_2nd_derv is None:
            self.local_trajectory = global_tj.create_default_trajectory_sd(s_start=start_s, d_start = start_d, s_end=end_s, d_end=end_d, num_points=num_of_points)
        else:
            # self.local_trajectory = global_tj.create_cubic_trajectory_sd(s_start=start_s,
            #     d_start=start_d, s_end=end_s, d_end=end_d, d_start_1st_derv= d_1st_derv, d_start_2nd_derv =  d_2nd_derv, num_points = num_of_points)
            
            self.local_trajectory = global_tj.create_quintic_trajectory_sd(start_s,
                                start_d, end_s, end_d, d_1st_derv=d_1st_derv, d_2nd_derv=d_2nd_derv,
                                 num_points=num_of_points)

        self.selected_next_edge = None
        self.next_edges = []
        self.is_selected = False

    def is_next_edge_selected(self):
        return self.selected_next_edge is not None

    def append_next_edges(self, edge):
        self.next_edges.append(edge)



    def __hash__(self):
        return hash((self.start_s, self.start_d, self.end_s, self.end_d, self.num_of_points))

    def __eq__(self, other):
        if isinstance(other, Edge):
            return (self.start_s == other.start_s and self.start_d == other.start_d and
                    self.end_s == other.end_s and self.end_d == other.end_d and
                    self.num_of_points == other.num_of_points)
        return False

class Node:
    pass

class LatticeGraph:
    nodes:Dict[Node, list[Node]] # adjancency list of nodes
    edges:Dict[Edge, list[Edge]] # adjancency list of edges

    def __init__(self, global_tj:Trajectory, num_of_edge_points = 10, max_curvature=3, start_vel = None, end_vel = None):
        self.__global_tj = global_tj
        self.__num_of_points = num_of_edge_points
        
        self.selected_edge:Edge = None
        self.lattice_graph = {} 

        self.__iter_idx = 0

    def add_edge(self, start_s, start_d, end_s, end_d):
        edge = Edge(start_s, start_d, end_s, end_d,global_tj=self.__global_tj,
                    num_of_points=self.__num_of_points)
        self.lattice_graph[edge] = self.__global_tj.create_quintic_trajectory_sd(start_s, start_d, end_s, end_d, num_points=self.__num_of_points)

    def clear(self):
        self.lattice_graph = {}
        self.selected_edge = None
        self.__iter_idx = 0

    def __iter__(self):
        self.__iter_idx = 0
        return self

    def __next__(self):
        if self.__iter_idx < len(self.lattice_graph):
            self.__iter_idx += 1
            return self.lattice_graph[self.__iter_idx-1]
        else:
            raise StopIteration

