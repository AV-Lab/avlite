from race_plan_control.plan.trajectory import Trajectory
from typing import Dict
from dataclasses import dataclass, field
from typing import Iterator, Optional
import logging
from icecream import ic
import math

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
        return (math.isclose(self.s, other.s, abs_tol=tol) and
                    math.isclose(self.d, other.d, abs_tol=tol) and
                    math.isclose(self.x, other.x, abs_tol=tol) and
                    math.isclose(self.y, other.y, abs_tol=tol) and
                    math.isclose(self.x_1st_derv, other.x_1st_derv, abs_tol=tol) and
                    math.isclose(self.y_1st_derv, other.y_1st_derv, abs_tol=tol) and
                    math.isclose(self.x_2nd_derv, other.x_2nd_derv, abs_tol=tol) and
                    math.isclose(self.y_2nd_derv, other.y_2nd_derv, abs_tol=tol) and
                    math.isclose(self.d_1st_derv, other.d_1st_derv, abs_tol=tol) and
                    math.isclose(self.d_2nd_derv, other.d_2nd_derv, abs_tol=tol))
        return False



    def __hash__(self):
        return hash((self.s, self.d, self.x, self.y, self.x_1st_derv, self.y_1st_derv,
                     self.x_2nd_derv, self.y_2nd_derv, self.d_1st_derv, self.d_2nd_derv))

    def __repr__(self):
        return (f"Node(s={self.s}, d={self.d}, x={self.x}, y={self.y}, "
                f"x_1st_derv={self.x_1st_derv}, y_1st_derv={self.y_1st_derv}, "
                f"x_2nd_derv={self.x_2nd_derv}, y_2nd_derv={self.y_2nd_derv}, "
                f"d_1st_derv={self.d_1st_derv}, d_2nd_derv={self.d_2nd_derv})")

@dataclass
class Edge:
    start: Node
    end: Node
    local_trajectory: Trajectory
    selected_next_edge: Optional["Edge"]
    next_edges: list["Edge"] = field(default_factory=list["Edge"])
    num_of_points:int =30
    cost: float = 0
    risk: float = 0


def create_edge(start: Node, end: Node, global_tj: Trajectory, num_of_points=30) -> Edge:
    local_trajectory = global_tj.create_quintic_trajectory_sd(
        s_start=start.s,
        d_start=start.d,
        s_end=end.s,
        d_end=end.d,
        num_points=num_of_points,
    )
    edge = Edge(start, end, local_trajectory, None, [], num_of_points)
    
    return edge

def create_path(start_node: Node, s_values: list[float], d_values: list[float]) -> list[Edge]:
    assert len(s_values) == len(d_values)
    for s, d in zip(s_values, d_values):
        pass
        # start_node = nodes[i]
        # end_node = nodes[i + 1]
        # create continious trajectories
        # trajectories = self.__global_tj.create_multiple_cubic_trajectories_xy(
        # x_values
