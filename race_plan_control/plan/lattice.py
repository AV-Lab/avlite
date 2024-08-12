from numpy import who
from race_plan_control.plan.trajectory import Trajectory
from typing import Dict
from dataclasses import dataclass
import networkx as nx

import logging

log = logging.getLogger(__name__)


class EdgeTmp:
    local_trajectory: Trajectory

    def __init__(
        self,
        start_s,
        start_d,
        end_s,
        end_d,
        global_tj: Trajectory,
        num_of_points=30,
        d_1st_derv=0,
        d_2nd_derv=0,
        x_1st_derv=0,
        x_2nd_derv=0,
        y_1st_derv=0,
        y_2nd_derv=0,
    ):
        self.start_s = start_s
        self.start_d = start_d
        self.end_s = end_s
        self.end_d = end_d
        self.num_of_points = num_of_points

        if d_1st_derv is None or d_2nd_derv is None:
            self.local_trajectory = global_tj.create_default_trajectory_sd(
                s_start=start_s,
                d_start=start_d,
                s_end=end_s,
                d_end=end_d,
                num_points=num_of_points,
            )
        else:
            # self.local_trajectory = global_tj.create_cubic_trajectory_sd(s_start=start_s,
            #     d_start=start_d, s_end=end_s, d_end=end_d, d_start_1st_derv= d_1st_derv, d_start_2nd_derv =  d_2nd_derv, num_points = num_of_points)

            # self.local_trajectory = global_tj.create_quintic_trajectory_sd(start_s,
            #                     start_d, end_s, end_d, d_1st_derv=d_1st_derv, d_2nd_derv=d_2nd_derv,
            #                      num_points=num_of_points)

            log.debug(f"Creating xy trajectory..")
            start_x, start_y = global_tj.convert_sd_to_xy(start_s, start_d)
            end_x, end_y = global_tj.convert_sd_to_xy(end_s, end_d)

            self.local_trajectory = global_tj.create_cubic_trajectory_xy(
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y,
                start_x_1st_derv=x_1st_derv,
                start_y_1st_derv=y_1st_derv,
                start_x_2nd_derv=x_2nd_derv,
                start_y_2nd_derv=y_2nd_derv,
                num_points=num_of_points,
            )

            # self.local_trajectory = global_tj.create_quintic_trajectory_xy(
            #         start_x = start_x, start_y = start_y, end_x = end_x,
            #         end_y = end_y,
            #         start_x_1st_derv=x_1st_derv, start_y_1st_derv=y_1st_derv,
            #         start_x_2nd_derv=x_2nd_derv, start_y_2nd_derv=y_2nd_derv,
            #         end_x_1st_derv=0, end_y_1st_derv=0,
            #         end_x_2nd_derv=0, end_y_2nd_derv=0,
            #         num_points=num_of_points)

        self.selected_next_edge = None
        self.next_edges = []
        self.is_selected = False

    def is_next_edge_selected(self):
        return self.selected_next_edge is not None

    def append_next_edges(self, edge):
        self.next_edges.append(edge)


@dataclass
class Node:
    s: float = 0
    d: float = 0
    x: float = 0
    y: float = 0
    x_1st_derv: float = 0
    y_1st_derv: float = 0
    x_2nd_derv: float = 0
    y_2nd_derv: float = 0

    def __init__(self, s, d, global_tj: Trajectory = None):
        self.s = s
        self.d = d
        if global_tj is not None:
            self.x, self.y = global_tj.convert_sd_to_xy(s, d)


@dataclass
class Edge:
    start: Node
    end: Node
    local_trajectory: Trajectory
    cost: float = 0


class LatticeGraph:
    nodes: list[Node]
    edges: list[Edge]
    solution: list[Edge]
    G: nx.DiGraph  # networkx graph

    def __init__(
        self,
        global_tj: Trajectory,
        num_of_edge_points=10,
        max_curvature=3,
        start_vel=None,
        end_vel=None,
    ):
        self.__global_tj = global_tj
        self.__num_of_points = num_of_edge_points
        G = nx.DiGraph()

    def create_path(
        self, start_node: Node, s_values: list[float], d_values: list[float]
    ) -> list[Edge]:
        assert len(s_values) == len(d_values)
        for i in range(len(s_values) - 1):
            pass
            # start_node = nodes[i]
            # end_node = nodes[i + 1]
            # create continious trajectories
            # trajectories = self.__global_tj.create_multiple_cubic_trajectories_xy(
            # x_values

    def add_node(self, node: Node):
        self.nodes.append(node)
        self.G.add_node(node)

    def add_edge(self, start_node: Node, end_node: Node):
        local_trajectory = self.__global_tj.create_cubic_trajectory_xy(
            start_x=start_node.x,
            start_y=start_node.y,
            end_x=end_node.x,
            end_y=end_node.y,
            start_x_1st_derv=start_node.x_1st_derv,
            start_y_1st_derv=start_node.y_1st_derv,
            start_x_2nd_derv=start_node.x_2nd_derv,
            start_y_2nd_derv=start_node.y_2nd_derv,
            num_points=self.__num_of_points,
        )
        edge = Edge(start=start_node, end=end_node, local_trajectory=local_trajectory)
        self.edges.append(edge)
        self.G.add_edge(start_node, end_node, data=edge)

    def clear(self):
        self.nodes.clear()
        self.edges.clear()
        self.solution.clear()
        self.G.clear()
