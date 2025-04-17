from c10_perception.c11_perception_model import PerceptionModel, EgoState
from c20_planning.c28_trajectory import Trajectory
from c10_perception.c11_perception_model import State
from typing import Dict
from dataclasses import dataclass, field
from typing import Iterator, Optional
import logging

import math
import numpy as np
from collections import defaultdict

log = logging.getLogger(__name__)


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
    d_1st_derv: float = 0
    d_2nd_derv: float = 0

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

@dataclass
class Edge:
    start: Node
    end: Node
    global_tj: Trajectory
    num_of_points: int = 30
    local_trajectory: Optional[Trajectory] = None
    selected_next_local_plan: Optional[Trajectory] = None
    next_edges: list["Edge"] = field(default_factory=list)
    collision: bool = False
    cost: float = 0
    risk: float = 0
    
    def __post_init__(self):
        # Create the local trajectory during initialization
        self.local_trajectory = self.global_tj.create_quintic_trajectory_sd(
            s_start=self.start.s,
            d_start=self.start.d,
            s_end=self.end.s,
            d_end=self.end.d,
            num_points=self.num_of_points,
            start_d_1st_derv=self.start.d_1st_derv,
            start_d_2nd_derv=self.start.d_2nd_derv,
        )

    def __str__(self):
        return f"Edge: {self.start} -> {self.end}"


@dataclass
class Lattice:
    """
    Lattice class to generate lattice from sample_nodes
    """
    global_trajectory: Trajectory
    ref_left_boundary_d: list
    ref_right_boundary_d: list
    planning_horizon: int = 5
    num_of_points: int = 30
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    level0_edges: list[Edge] = field(default_factory=list)
    lattice_nodes_by_level: Dict[int, list] = field(default_factory=lambda: defaultdict(list))
    incoming_edges: Dict[Node, list] = field(default_factory=lambda: defaultdict(list))
    outgoing_edges: Dict[Node, list] = field(default_factory=lambda: defaultdict(list))
    targetted_num_edges: int = 0

    def sample_nodes(self, s, d, sample_size, maneuver_distance, boundary_clearance, orientation=0):
        s1_ = s
        x, y = self.global_trajectory.convert_sd_to_xy(s1_, d)
        self.lattice_nodes_by_level[0].append(Node(s1_, d, x, y, d_1st_derv=orientation))

        for l in range(1, self.planning_horizon + 1):
            s1_ = s1_ + maneuver_distance
            if s1_ > self.global_trajectory.path_s[-2]:  # at -1 path_s is zero
                log.warning("No Replan, reaching the end of lap")
                return

            # One line always at track line
            wp = self.global_trajectory.get_closest_waypoint_frm_sd(s1_, 0)
            _, dg = self.global_trajectory.get_sd_by_waypoint(wp)
            x, y = self.global_trajectory.convert_sd_to_xy(s1_, dg)
            node = Node(s1_, dg, x, y)
            self.lattice_nodes_by_level[l].append(node)  # always a node at track line
            self.nodes.append(node)

            for _ in np.arange(sample_size - 1):
                target_wp = self.global_trajectory.get_closest_waypoint_frm_sd(s1_, 0)
                d1_ = np.random.uniform(
                    self.ref_left_boundary_d[target_wp] - boundary_clearance,
                    self.ref_right_boundary_d[target_wp] + boundary_clearance,
                )
                x, y = self.global_trajectory.convert_sd_to_xy(s1_, d1_)
                n_ = Node(s1_, d1_, x, y)
                self.nodes.append(n_)
                self.lattice_nodes_by_level[l].append(n_)

    def generate_lattice_from_nodes(self, pm: Optional[PerceptionModel] = None):
        for l in range(self.planning_horizon + 1):
            for node in self.lattice_nodes_by_level[l]:
                for next_node in self.lattice_nodes_by_level[l + 1]:
                    assert node != next_node
                    edge = Edge(start=node, end=next_node, global_tj = self.global_trajectory)
                    if pm is not None:
                        edge.collision = not check_collision(pm, edge.local_trajectory)
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



def check_collision(pm:PerceptionModel, trajectory: Trajectory = None, sample_size=5)-> bool:
    ego = pm.ego_vehicle
    if trajectory is not None:
        path_x = trajectory.path_x
        path_y = trajectory.path_y
        indices = np.linspace(1, len(trajectory.path_x) - 1, sample_size, dtype=int)
        for i in indices:
            dx = path_x[i] - path_x[i - 1]
            dy = path_y[i] - path_y[i - 1]
            theta = np.arctan2(dy, dx)
            ego = EgoState(
                x = float(path_x[i]),
                y = float(path_y[i]),
                theta = float(theta)
            )

            for agent in pm.agent_vehicles:
                if ego.get_bb_polygon().intersects(agent.get_bb_polygon()):
                    log.debug(f"Collision at {ego.x}, {ego.y}")
                    return False
    else:
        for agent in pm.agent_vehicles:
            if ego.get_bb_polygon().intersects(agent.get_bb_polygon()):
                log.info(f"Collision at {ego.x}, {ego.y}")
                return False
    return True
