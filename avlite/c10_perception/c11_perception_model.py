from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import copy
import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import Polygon

from avlite.c10_perception.c19_settings import PerceptionSettings

if TYPE_CHECKING:
    from avlite.c40_execution.c43_task_strategy import StackEvent

log = logging.getLogger(__name__)

EGO_AGENT_ID: int = 0




@dataclass
class PerceptionModel:
    static_obstacles: list[State] = field(default_factory=list)
    agent_vehicles: list[AgentState] = field(default_factory=list)
    ego_vehicle: EgoState= field(default_factory=lambda: EgoState())
    max_agent_vehicles: int = field(default_factory=lambda: PerceptionSettings.c11_max_agents)
    
    prediction: Optional[PredictionModelBase] = None

    # Optional map (HDMap or RaceMap)
    map: Optional[Map] = None


    # Raw LiDAR points that passed segmentation + range gating (diagnostic overlay)
    detection_clusters: Optional[np.ndarray] = None

    # Optional outcome signal for TaskRunner harvest (see StackEvent); default None.
    stack_event: Optional[StackEvent] = None

    def add_agent_vehicle(self, agent: AgentState) -> int: # return agent_id
        """ Add an agent vehicle to the perception model and assign a unique agent_id."""
        if self.max_agent_vehicles <= 0:
            log.info("Max num of agents is %s; not adding", self.max_agent_vehicles)
            return -1
        while len(self.agent_vehicles) >= self.max_agent_vehicles:
            evicted = self.agent_vehicles.pop(0)
            log.info(
                "Max num of agents reached. Deleting oldest agent %s",
                evicted.agent_id,
            )
        ids = {a.agent_id for a in self.agent_vehicles}
        agent.agent_id = next(i for i in range(1, len(ids) + 2) if i not in ids)
        self.agent_vehicles.append(agent)
        log.info(f"Total agent vehicles {len(self.agent_vehicles)}")

        return agent.agent_id

    def reset(self):
        self.static_obstacles = []
        self.agent_vehicles = []
        self.prediction = None
        self.stack_event = None


@dataclass
class PredictionModelBase:
    """Shared metadata for prediction outputs on ``PerceptionModel.prediction``."""

    # Seconds between consecutive forecast samples (t, t+dt, …).
    predict_delta_t: float = field(
        default_factory=lambda: PerceptionSettings.c11_predict_delta_t
    )


@dataclass
class SingleTrajectory(PredictionModelBase):
    """Deterministic (x, y) polyline per agent."""

    # agent_id -> [n_steps, 2] world x,y [m]; step k at (k+1) * predict_delta_t.
    trajectories: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class GP(PredictionModelBase):
    """Gaussian-process forecast per agent (mean + joint covariance)."""

    # agent_id -> [n_steps, 2] predictive mean trajectory.
    means: dict[int, np.ndarray] = field(default_factory=dict)
    # agent_id -> [2*n_steps, 2*n_steps] joint covariance; state order [x0,y0,x1,y1,...].
    covariance: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class GMM(PredictionModelBase):
    """Gaussian-mixture multi-modal forecast per agent."""

    # agent_id -> [n_modes, n_steps, 2] mode means.
    trajectories: dict[int, np.ndarray] = field(default_factory=dict)
    # agent_id -> [n_modes] mode weights (sum ≈ 1).
    weights: dict[int, np.ndarray] = field(default_factory=dict)
    # agent_id -> [n_modes, n_steps, 2, 2] position covariance per mode/step.
    covariances: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class OccupancyFlow(PredictionModelBase):
    """Per-agent occupancy grid sequences."""

    # agent_id -> n_steps grids, each [grid_size, grid_size].
    occupancy_flow: dict[int, list[np.ndarray]] = field(default_factory=dict)
    grid_bounds: dict[str, float] = field(default_factory=dict)
    grid_size: int = field(default_factory=lambda: PerceptionSettings.c11_prediction_grid_size)


@dataclass
class AggregatedOccupancyFlow(PredictionModelBase):
    """Lump-sum occupancy grids for all agents combined."""

    occupancy_flow: list[np.ndarray] = field(default_factory=list)
    grid_bounds: dict[str, float] = field(default_factory=dict)
    grid_size: int = field(default_factory=lambda: PerceptionSettings.c11_prediction_grid_size)


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    theta: float = PerceptionSettings.c11_state_default_heading
    width: float = 2.0
    length: float = 4.5

    def __post_init__(self):
        self.__start = self.get_copy()

    def set_start(self):
        """Capture the current state as the snapshot restored by :meth:`reset`."""
        self.__start.copy_from(self)

    def get_bb_corners(self) -> np.ndarray:
        """Get the bounding box corners of the vehicle in world coordinates."""
        corners_x = np.array(
            [
                -self.length / 2,
                +self.length / 2,
                +self.length / 2,
                -self.length / 2,
            ]
        )
        corners_y = np.array(
            [
                -self.width / 2,
                -self.width / 2,
                +self.width / 2,
                +self.width / 2,
            ]
        )

        rotation_matrix = np.array(
            [
                [np.cos(self.theta), -np.sin(self.theta)],
                [np.sin(self.theta), np.cos(self.theta)],
            ]
        )
        rotated_corners = np.dot(rotation_matrix, np.array([corners_x, corners_y]))

        rotated_corners_x = rotated_corners[0, :] + self.x
        rotated_corners_y = rotated_corners[1, :] + self.y

        return np.c_[rotated_corners_x, rotated_corners_y]
    
    def reset(self):
        self.copy_from(self.__start)

    def copy_from(self, other: State) -> None:
        """Copy dataclass fields from *other* in place (preserves object identity)."""
        for f in fields(self):
            setattr(self, f.name, getattr(other, f.name))

    def get_copy(self):
        return copy.deepcopy(self)

    def get_bb_polygon(self):
        return Polygon(self.get_bb_corners())


class AgentType(Enum):
    ACKERMANN = auto()
    DIFF_DRIVE = auto()
    AERIAL = auto()
    SURFACE_VESSEL = auto()  # boats, USVs — water surface
    UNDERWATER = auto()      # AUVs — subsurface
    CYCLIST = auto()
    PEDESTRIAN = auto()
    DYNAMIC_OBJECT = auto()

@dataclass
class AgentState(State):
    velocity: float = 0.0
    agent_id: int = -1
    agent_type: AgentType = AgentType.ACKERMANN


@dataclass
class EgoState(AgentState):
    """Ego vehicle state with additional properties in future."""
    agent_id: int = EGO_AGENT_ID
    agent_type: AgentType = AgentType.ACKERMANN


class Map(ABC):
    """Static world geometry with an optional WGS84 reference point."""

    source_path: str

    @property
    @abstractmethod
    def reference_point(self) -> tuple[float, float] | None:
        """WGS84 (lat_deg, lon_deg) when available."""

    @staticmethod
    @abstractmethod
    def is_loadable(path: Path | str) -> bool:
        """Return True when *path* is a supported map file."""

    @classmethod
    @abstractmethod
    def from_path(cls, path: Path | str) -> Map:
        """Load a map instance from *path*."""

    @staticmethod
    def open(path: Path | str) -> Map | None:
        """Dispatch to ``HDMap`` or ``RaceMap`` based on file format."""
        path = Path(path)
        if HDMap.is_loadable(path):
            return HDMap.from_path(path)
        if RaceMap.is_loadable(path):
            return RaceMap.from_path(path)
        return None


@dataclass
class RaceMap(Map):
    """Race corridor map from left/right boundary JSON."""

    source_path: str
    left_bound: np.ndarray = field(default_factory=lambda: np.array([]))
    right_bound: np.ndarray = field(default_factory=lambda: np.array([]))
    _reference_point: tuple[float, float] | None = None

    @property
    def reference_point(self) -> tuple[float, float] | None:
        return self._reference_point

    @staticmethod
    def is_loadable(path: Path | str) -> bool:
        """True when *path* is a race-boundary JSON with bounds and ReferencePoint."""
        path = Path(path)
        if path.suffix.lower() != ".json" or not path.is_file():
            return False
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        if not all(k in data for k in ("LeftBound", "RightBound", "ReferencePoint")):
            return False
        left = data["LeftBound"]
        right = data["RightBound"]
        if not left or not right:
            return False
        if not isinstance(left[0], list) or not isinstance(right[0], list):
            return False
        ref = data["ReferencePoint"]
        if not isinstance(ref, list) or len(ref) < 2:
            return False
        try:
            float(ref[0])
            float(ref[1])
        except (TypeError, ValueError):
            return False
        return True

    @classmethod
    def from_path(cls, path: Path | str) -> RaceMap:
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        left = np.array(data["LeftBound"])[:, :2]
        right = np.array(data["RightBound"])[:, :2]
        ref = data["ReferencePoint"]
        ref_pt = (float(ref[0]), float(ref[1]))
        return cls(source_path=str(path), left_bound=left, right_bound=right, _reference_point=ref_pt,)


@dataclass
class HDMap(Map):
    """Compact HD map representation for global planning."""

    @dataclass
    class Lane:
        id: int
        uid: str
        lane_element: ET.Element
        center_line: np.ndarray = field(default_factory=lambda: np.array([]))
        left_d: list[float] = field(default_factory=list)
        right_d: list[float] = field(default_factory=list)
        road: Optional["HDMap.Road"] = None
        pred_id: str = ""
        succ_id: str = ""
        pred_type: str = "lane"
        succ_type: str = "lane"
        side: str = "left"
        type: str = "driving"
        road_id: str = ""
        width: float = 0.0
        lane_section_idx: int = 0
        predecessors: list["HDMap.Lane"] = field(default_factory=list)
        successors: list["HDMap.Lane"] = field(default_factory=list)
        neighbors: set["HDMap.Lane"] = field(default_factory=set)
        drivable_neighbors: set["HDMap.Lane"] = field(default_factory=set)

        def __hash__(self):
            return hash(self.uid)

    @dataclass
    class Road:
        """Compact road representation for global planning."""

        id: str
        road_element: ET.Element
        pred_id: str = ""
        succ_id: str = ""
        pred_type: str = "road"
        succ_type: str = "road"
        length: float = 0.0
        junction_id: str = ""
        center_line: np.ndarray = field(default_factory=lambda: np.array([]))
        predecessors: list["HDMap.Road"] = field(default_factory=list)
        successors: list["HDMap.Road"] = field(default_factory=list)
        lane_sections: list[list["HDMap.Lane"]] = field(default_factory=list)
        lane_section_s_vals: list[float] = field(default_factory=list)
        reversed: bool = False

    xodr_file_name: str = ""
    sampling_resolution: float = 0.1
    roads: list[Road] = field(default_factory=list)
    lanes: list[Lane] = field(default_factory=list)
    road_by_id: dict[str, Road] = field(default_factory=dict)
    lane_by_uid: dict[str, Lane] = field(default_factory=dict)
    junction_by_id: dict[str, list[Road]] = field(default_factory=dict)
    road_network: nx.DiGraph = field(default_factory=nx.DiGraph)
    lane_network: nx.DiGraph = field(default_factory=nx.DiGraph)
    root: ET.Element | None = field(default=None, repr=False)

    _point_to_road: dict[tuple[float, float], Road] = field(default_factory=dict, init=False, repr=False)
    _point_to_drivable_lane: dict[tuple[float, float], Lane] = field(default_factory=dict, init=False, repr=False)
    _road_kdtree: Optional[KDTree] = field(default=None, init=False, repr=False)
    _lane_kdtree_drivable: Optional[KDTree] = field(default=None, init=False, repr=False)
    _all_road_points: list[tuple[float, float]] = field(default_factory=list, init=False, repr=False)
    _all_drivable_lane_points: list[tuple[float, float]] = field(default_factory=list, init=False, repr=False)
    _reference_point: tuple[float, float] | None = field(default=None, init=False, repr=False)

    @property
    def source_path(self) -> str:
        return self.xodr_file_name

    @property
    def reference_point(self) -> tuple[float, float] | None:
        return self._reference_point

    @staticmethod
    def is_loadable(path: Path | str) -> bool:
        path = Path(path)
        return path.suffix.lower() == ".xodr" and path.is_file()

    @classmethod
    def from_path(cls, path: Path | str) -> HDMap:
        return cls(xodr_file_name=str(Path(path).resolve()))

    def __post_init__(self) -> None:
        if not self.xodr_file_name:
            log.error("No OpenDRIVE file specified.")
            return
        from avlite.c10_perception.c18_hdmap_parser import parse_opendrive

        parse_opendrive(self)

    def find_nearest_road(self, x: float, y: float) -> Road | None:
        if self._road_kdtree is not None:
            _, index = self._road_kdtree.query((x, y))
            if index >= 0 and index < len(self._all_road_points):
                px, py = self._all_road_points[index]
                if (px, py) not in self._point_to_road:
                    log.error("Point not found in point_to_road mapping: (%s, %s)", px, py)
                return self._point_to_road.get((px, py), None)

    def find_nearest_lane(self, x: float, y: float) -> Lane | None:
        if self._lane_kdtree_drivable is not None:
            _, index = self._lane_kdtree_drivable.query((x, y))
            if 0 <= index < len(self._all_drivable_lane_points):
                lx, ly = self._all_drivable_lane_points[index]
                if (lx, ly) not in self._point_to_drivable_lane:
                    log.error("Point not found in point_to_lane mapping: (%s, %s)", x, y)
                return self._point_to_drivable_lane.get((lx, ly), None)

    def find_nearest_lane_and_idx(self, x: float, y: float) -> tuple[Lane | None, int]:
        lane = self.find_nearest_lane(x, y)
        if lane is None or lane.center_line.size == 0:
            return None, -1
        dists = np.linalg.norm(lane.center_line - np.array([[x], [y]]), axis=0)
        idx = int(np.argmin(dists))
        return lane, idx

    def can_laneA_access_laneB(self, lane_a: Lane, lane_b: Lane) -> bool:
        check1 = lane_b in lane_a.neighbors
        b_start_end = [lane_b.center_line[:, 0], lane_b.center_line[:, -1]]
        a = lane_a.center_line[:, -1] if int(lane_a.id) < 0 else lane_a.center_line[:, 0]
        dists = [(np.linalg.norm(a - b), j) for j, b in enumerate(b_start_end)]
        min_dist, b_idx = min(dists, key=lambda item: item[0])
        check2 = min_dist < 0.5
        check3 = False
        if check2:
            if int(lane_a.id) < 0:
                vec_a = lane_a.center_line[:, -3] - lane_a.center_line[:, -1]
            else:
                vec_a = lane_a.center_line[:, 2] - lane_a.center_line[:, 0]
            if int(lane_b.id) < 0:
                vec_b = (
                    lane_b.center_line[:, 0] - lane_b.center_line[:, 2]
                    if b_idx == 0
                    else lane_b.center_line[:, -1] - lane_b.center_line[:, -3]
                )
            else:
                vec_b = (
                    lane_b.center_line[:, 1] - lane_b.center_line[:, 0]
                    if b_idx == 0
                    else lane_b.center_line[:, -1] - lane_b.center_line[:, -3]
                )
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            if norm_a > 0 and norm_b > 0:
                check3 = np.dot(vec_a / norm_a, vec_b / norm_b) > 0.9
        return check1 and check2 and check3

    def road_has_driving_lanes(self, road: Road) -> bool:
        road_element = road.road_element
        for section in road_element.findall(".//laneSection"):
            for lane in section.findall(".//lane"):
                if lane.get("type") == "driving":
                    return True
        return False

    def road_is_bidirectional(self, road: Road) -> bool:
        road_element = road.road_element
        right = False
        left = False
        for section in road_element.findall(".//laneSection"):
            for lane in section.findall(".//lane"):
                lane_type = lane.get("type")
                lane_id = int(lane.get("id", "0"))
                if lane_type == "driving" and lane_id < 0:
                    right = True
                if lane_type == "driving" and lane_id > 0:
                    left = True
                if right and left:
                    return True
        return False

