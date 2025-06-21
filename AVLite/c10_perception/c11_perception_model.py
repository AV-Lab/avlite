import numpy as np
from numpy.matlib import ndarray
from shapely.geometry import Polygon
from typing import Optional, Dict, Any
import copy
from dataclasses import dataclass, field
import logging

from c10_perception.c19_settings import PerceptionSettings


log = logging.getLogger(__name__)


@dataclass
class State:
    id:str = "-1"
    x: float = 0.0
    y: float = 0.0
    theta: float = -np.pi / 4
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

    def get_transformed_bb_corners(self, func):
        corners = self.get_bb_corners()
        return np.apply_along_axis(func, 1, corners)

@dataclass
class AgentState(State):
    velocity: float = 0.0
    agent_id: int = -1

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
    max_valocity: float = PerceptionSettings.max_valocity
    max_acceleration: float = PerceptionSettings.max_acceleration
    min_acceleration: float = PerceptionSettings.min_acceleration
    max_steering: float = PerceptionSettings.max_steering  # in radians
    min_steering: float = PerceptionSettings.min_steering
    L_f: float = PerceptionSettings.L_f  # Distance from center of mass to front axle


@dataclass
class PerceptionModel:
    static_obstacles: list[State] = field(default_factory=list)
    agent_vehicles: list[AgentState] = field(default_factory=list)
    ego_vehicle: EgoState = field(default_factory=EgoState)
    max_agent_vehicles:int = PerceptionSettings.max_agent_vehicles
    # agent_history: dict[str, list[AgentState]] = field(default_factory=dict) # agent_id -> list of states

    # Occupancy grid fields (NumPy arrays)
    occupancy_grid: Optional[np.ndarray] = field(default=None)
    grid_bounds: Optional[Dict[str, float]] = field(default=None)

    trajectories : Optional[np.ndarray] = None # For single, multi,GMM results of predictor


    def add_agent_vehicle(self, agent: AgentState):
        if len(self.agent_vehicles) == self.max_agent_vehicles:
            log.info("Max num of agent reached. Deleteing Old agents")
            self.agent_vehicles = []
        self.agent_vehicles.append(agent)
        log.info(f"Total agent vehicles {len(self.agent_vehicles)}")


    def reset(self):
        self.static_obstacles = []
        self.agent_vehicles = []


