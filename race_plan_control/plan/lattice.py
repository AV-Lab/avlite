from race_plan_control.plan.trajectory import Trajectory
from typing import Dict
from dataclasses import dataclass, field
from typing import Iterator, Optional
import logging
from icecream import ic

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
        self.d_1st_derv: Optional[float] = d_1st_derv
        self.d_2nd_derv: Optional[float] = d_2nd_derv
        if global_tj is not None:
            self.x, self.y = global_tj.convert_sd_to_xy(s, d)

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
    local_trajectory = global_tj.create_cubic_trajectory_sd(
        s_start=start.s,
        d_start=start.d,
        s_end=end.s,
        d_end=end.d,
        d_start_1st_derv=start.d_1st_derv,
        d_start_2nd_derv=start.d_2nd_derv,
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
