from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import copy
import numpy as np
from shapely.geometry import Polygon

from avlite.c10_perception.c19_settings import PerceptionSettings

log = logging.getLogger(__name__)

EGO_AGENT_ID: int = 0


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

    def add_agent_vehicle(self, agent: AgentState) -> int: # return agent_id
        if len(self.agent_vehicles) == self.max_agent_vehicles:
            log.info("Max num of agent reached. Deleteing Old agents")
            self.agent_vehicles = []
        ids = {a.agent_id for a in self.agent_vehicles}
        agent.agent_id = next(i for i in range(1, len(ids) + 2) if i not in ids)
        self.agent_vehicles.append(agent)
        log.info(f"Total agent vehicles {len(self.agent_vehicles)}")

        return agent.agent_id

    def reset(self):
        self.static_obstacles = []
        self.agent_vehicles = []
        self.prediction = None


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
        # initial x,y position, useful for reset
        self.__init_x = self.x
        self.__init_y = self.y
        self.__init_theta = self.theta


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
        self.x = self.__init_x
        self.y = self.__init_y
        self.theta = self.__init_theta

    def get_copy(self):
        return copy.deepcopy(self)

    def get_bb_polygon(self):
        return Polygon(self.get_bb_corners())


@dataclass
class AgentState(State):
    velocity: float = 0.0
    agent_id: int = -1
    agent_type: AgentType = AgentType.ACKERMANN

    def __post_init__(self):
        super().__post_init__()
        self.__init_speed = self.velocity

    def reset(self):
        super().reset()
        self.velocity = self.__init_speed


    #TODO 
    def predict(self, dt):
        pass
        # self.x += self.velocity * np.cos(self.theta) * dt
        # self.y += self.velocity * np.sin(self.theta) * dt


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

    @classmethod
    def open(cls, path: Path | str) -> Map | None:
        """Dispatch to ``HDMap`` or ``RaceMap`` based on file format."""
        path = Path(path)
        from avlite.c10_perception.c18_hdmap_parser import HDMap

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
        return cls(
            source_path=str(path),
            left_bound=left,
            right_bound=right,
            _reference_point=ref_pt,
        )

