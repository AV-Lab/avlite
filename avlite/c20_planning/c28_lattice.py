from typing import Dict
from dataclasses import dataclass, field
from typing import Optional
import logging

import numpy as np
from collections import defaultdict

from avlite.c10_perception.c11_perception_model import PerceptionModel, EgoState
from avlite.c60_common.c63_trajectory_tracker import TrajectoryTracker
from avlite.c20_planning.c29_settings import PlanningSettings
from avlite.c60_common.c64_collision_checking import check_collision, precompute_obstacle_polygons


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

    def __hash__(self):
        return hash((self.s, self.d, self.x, self.y, self.x_1st_derv, self.y_1st_derv, self.x_2nd_derv, self.y_2nd_derv, self.d_1st_derv, self.d_2nd_derv,))
@dataclass
class Edge:
    start: Node
    end: Node
    global_tj: TrajectoryTracker
    num_of_points: int = 30
    local_trajectory: Optional[TrajectoryTracker] = None
    selected_next_local_plan: Optional["Edge"] = None
    next_edges: list["Edge"] = field(default_factory=list)
    collision: bool = False
    collision_agent_velocity: float = 0.0
    collision_idx: int = -1 # Index of the collision point in the local trajectory
    cost: float = 0
    risk: float = 0
    boundary_violation: bool = False  # True if the path exits road boundaries (with clearance)

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
    global_trajectory: TrajectoryTracker
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

    def sample_nodes(self, s, d, sample_size, maneuver_distance, boundary_clearance, orientation=0):
        self.boundary_clearance = boundary_clearance  # stored for full-path boundary checks
        self.maneuver_distance = maneuver_distance   # stored for collision-prediction horizon
        s1_ = s
        x, y = self.global_trajectory.convert_sd_to_xy(s1_, d)
        self.lattice_nodes_by_level[0].append(Node(s1_, d, x, y, d_1st_derv=orientation))

        for l in range(1, self.planning_horizon + 1):
            s1_ = s1_ + maneuver_distance
            if s1_ > self.global_trajectory.path_s[-2]:  # at -1 path_s is zero
                log.debug("sample_nodes: approaching track end, truncating lattice horizon")
                break

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
        # Pre-build all obstacle polygons once (swept for movers, plain for statics).
        # This avoids re-constructing N_agents polygons inside every edge's check_collision call.
        obstacle_polygons = None
        if pm is not None and len(pm.agent_vehicles) > 0:
            # Predict obstacles over the full planning horizon: horizon_dist / ego_vel
            ego_vel = max(pm.ego_vehicle.velocity, PlanningSettings.c20_default_ego_velocity)
            maneuver_dist = getattr(self, 'maneuver_distance', 30.0)
            obstacle_polygons = precompute_obstacle_polygons(
                pm,
                total_time=self.planning_horizon * maneuver_dist / ego_vel,
                min_velocity_threshold=PlanningSettings.c20_min_velocity_threshold,
                obstacle_inflation_margin=PlanningSettings.c20_obstacle_inflation_margin,
            )
        for l in range(self.planning_horizon + 1):
            for node in self.lattice_nodes_by_level[l]:
                for next_node in self.lattice_nodes_by_level[l + 1]:
                    assert node != next_node
                    edge = Edge(start=node, end=next_node, global_tj = self.global_trajectory, num_of_points=self.num_of_points)
                    if pm is not None:
                        edge.collision, edge.collision_idx, edge.collision_agent_velocity = check_collision(
                            pm, edge.local_trajectory,
                            obstacle_polygons=obstacle_polygons,
                            min_velocity_threshold=PlanningSettings.c20_min_velocity_threshold,
                            collision_safety_margin=PlanningSettings.c20_collision_safety_margin,
                            default_ego_velocity=PlanningSettings.c20_default_ego_velocity,
                        )
                    edge.boundary_violation = self._check_boundary_violation(edge)
                    self.edges.append(edge)
                    self.incoming_edges[next_node].append(edge)
                    self.outgoing_edges[node].append(edge)
                    if l == 0:
                        self.level0_edges.append(edge)
                for e in self.incoming_edges[node]:
                    for o in self.outgoing_edges[node]:
                        e.next_edges.append(o)

    def _check_boundary_violation(self, edge: "Edge") -> bool:
        """Return True if any point on the edge path exits the road boundaries (with clearance)."""
        tj = edge.local_trajectory
        if (tj is None
                or not hasattr(tj, 'path_s_from_parent') or tj.path_s_from_parent is None
                or not hasattr(tj, 'path_d_from_parent') or tj.path_d_from_parent is None):
            return False
        path_s_arr = np.asarray(self.global_trajectory.path_s)
        max_wp = len(self.ref_left_boundary_d) - 1
        clearance = getattr(self, 'boundary_clearance', 0.0)
        for s, d in zip(tj.path_s_from_parent, tj.path_d_from_parent):
            wp = int(np.clip(np.searchsorted(path_s_arr, s, side='left'), 0, max_wp))
            if d > self.ref_left_boundary_d[wp] - clearance:
                return True
            if d < self.ref_right_boundary_d[wp] + clearance:
                return True
        return False

    def reset(self):
        self.lattice_nodes_by_level.clear()
        self.incoming_edges.clear()
        self.outgoing_edges.clear()
        self.level0_edges.clear()
        self.nodes.clear()
        self.edges.clear()


