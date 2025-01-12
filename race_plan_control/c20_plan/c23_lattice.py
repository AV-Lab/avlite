from os import wait
from c20_plan.c24_trajectory import Trajectory
from typing import Dict
from dataclasses import dataclass, field
from typing import Iterator, Optional
import logging
from icecream import ic
import math
import numpy as np
from collections import defaultdict

log = logging.getLogger(__name__)


@dataclass
class Node:
    def __init__(self, s, d, global_tj: Optional[Trajectory] = None, d_1st_derv=0, d_2nd_derv=0):
        self.s: float = s
        self.d: float = d
        self.x: float = 0
        self.y: float = 0
        self.x_1st_derv: float = 0
        self.y_1st_derv: float = 0
        self.x_2nd_derv: float = 0
        self.y_2nd_derv: float = 0
        self.d_1st_derv: float = d_1st_derv
        self.d_2nd_derv: float = d_2nd_derv
        if global_tj is not None:
            self.x, self.y = global_tj.convert_sd_to_xy(s, d)

    def __eq__(self, other):
        tol = 1e-9

        return (
            math.isclose(self.s, other.s, abs_tol=tol)
            and math.isclose(self.d, other.d, abs_tol=tol)
            and math.isclose(self.x, other.x, abs_tol=tol)
            and math.isclose(self.y, other.y, abs_tol=tol)
            and math.isclose(self.x_1st_derv, other.x_1st_derv, abs_tol=tol)
            and math.isclose(self.y_1st_derv, other.y_1st_derv, abs_tol=tol)
            and math.isclose(self.x_2nd_derv, other.x_2nd_derv, abs_tol=tol)
            and math.isclose(self.y_2nd_derv, other.y_2nd_derv, abs_tol=tol)
            and math.isclose(self.d_1st_derv, other.d_1st_derv, abs_tol=tol)
            and math.isclose(self.d_2nd_derv, other.d_2nd_derv, abs_tol=tol)
        )

    def __hash__(self):
        return hash(
            (
                self.s,
                self.d,
                self.x,
                self.y,
                self.x_1st_derv,
                self.y_1st_derv,
                self.x_2nd_derv,
                self.y_2nd_derv,
                self.d_1st_derv,
                self.d_2nd_derv,
            )
        )

    def __repr__(self):
        return (
            f"Node(s={self.s}, d={self.d}, x={self.x}, y={self.y}, "
            f"x_1st_derv={self.x_1st_derv}, y_1st_derv={self.y_1st_derv}, "
            f"x_2nd_derv={self.x_2nd_derv}, y_2nd_derv={self.y_2nd_derv}, "
            f"d_1st_derv={self.d_1st_derv}, d_2nd_derv={self.d_2nd_derv})"
        )


@dataclass
class Edge:
    start: Node
    end: Node
    local_trajectory: Trajectory
    selected_next_local_plan: Optional["Edge"]
    next_edges: list["Edge"] = field(default_factory=list["Edge"])
    num_of_points: int = 30
    cost: float = 0
    risk: float = 0


    @staticmethod
    def edge_factory(start: Node, end: Node, global_tj: Trajectory, num_of_points=30) -> "Edge":
        local_trajectory = global_tj.create_quintic_trajectory_sd(
            s_start=start.s,
            d_start=start.d,
            s_end=end.s,
            d_end=end.d,
            num_points=num_of_points,
        )
        edge = Edge(start, end, local_trajectory, None, [], num_of_points)

        return edge


class Lattice:
    """
    Lattice class to generate lattice from sample_nodes
    """
    def __init__(
        self,
        global_tj: Trajectory,
        ref_left_boundary_d: list,
        ref_right_boundary_d: list,
        planning_horizon=5,
        num_of_points=30,
    ):
        self.global_trajectory = global_tj
        self.num_of_points = num_of_points
        self.planning_horizon = planning_horizon # number of levels in the lattice
        self.__ref_left_boundary_d = ref_left_boundary_d
        self.__ref_right_boundary_d = ref_right_boundary_d
        
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self.level0_edges: list[Edge] = []

        self.lattice_nodes_by_level: Dict[int, list] = defaultdict(list)  # key is level, value is list of nodes
        self.incoming_edges: Dict[Node, list[Edge]] = defaultdict(list)  # key is node, value is incoming edge
        self.outgoing_edges: Dict[Node, list[Edge]] = defaultdict(list)  # key is node, value is incoming edge


    def sample_nodes(self, s, d, sample_size, maneuver_distance, boundary_clearance):
        s1_ = s
        self.lattice_nodes_by_level[0].append(Node(s1_, d, self.global_trajectory))

        for l in range(1, self.planning_horizon + 1):
            s1_ = s1_ + maneuver_distance
            if s1_ > self.global_trajectory.path_s[-2]:  # at -1 path_s is zero
                log.warning("No Replan, reaching the end of lap")
                return
            node = Node(s1_, d, self.global_trajectory)
            self.lattice_nodes_by_level[l].append(node)  # always a node at track line
            self.nodes.append(node)
            for _ in np.arange(sample_size - 1):
                target_wp = self.global_trajectory.get_closest_waypoint_frm_sd(s1_, 0)
                d1_ = np.random.uniform(
                    self.__ref_left_boundary_d[target_wp] - boundary_clearance,
                    self.__ref_right_boundary_d[target_wp] + boundary_clearance,
                )
                n_ = Node(s1_, d1_, self.global_trajectory)
                self.nodes.append(n_)
                self.lattice_nodes_by_level[l].append(n_)



    def generate_lattice_from_nodes(self):
        for l in range(self.planning_horizon + 1):
            for node in self.lattice_nodes_by_level[l]:
                for next_node in self.lattice_nodes_by_level[l + 1]:
                    assert node != next_node
                    edge = Edge.edge_factory(node, next_node, self.global_trajectory)
                    self.edges.append(edge)
                    self.incoming_edges[next_node].append(edge)
                    self.outgoing_edges[node].append(edge)
                    if l == 0:
                        self.level0_edges.append(edge)
                for e in self.incoming_edges[node]:
                    for o in self.outgoing_edges[node]:
                        e.next_edges.append(o)
    def reset(self):
        self.lattice_nodes_by_level.clear()
        self.incoming_edges.clear()
        self.outgoing_edges.clear()
        self.level0_edges.clear()
        self.nodes.clear()
        self.edges.clear()


