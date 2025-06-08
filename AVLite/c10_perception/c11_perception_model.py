import numpy as np
from shapely.geometry import Polygon
import copy
from dataclasses import dataclass, field

import logging


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
    max_valocity: float = 30
    max_acceleration: float = 10    
    min_acceleration: float = -20
    max_steering: float = 0.7  # in radians
    min_steering: float = -0.7
    L_f: float = 2.5  # Distance from center of mass to front axle


@dataclass
class PerceptionModel:
    static_obstacles: list[State] = field(default_factory=list)
    agent_vehicles: list[AgentState] = field(default_factory=list)
    ego_vehicle: EgoState = field(default_factory=EgoState)
    max_agent_vehicles:int = 12
    agent_history: dict[str, list[AgentState]] = field(default_factory=dict) # agent_id -> list of states
    agent_occupancy_flow: dict[str, list[np.ndarray]] = field(default_factory=dict) # agent_id -> list of occupancy flow polygons

    def add_agent_vehicle(self, agent: AgentState):
        if len(self.agent_vehicles) == self.max_agent_vehicles:
            log.info("Max num of agent reached. Deleteing Old agents")
            self.agent_vehicles = []
        self.agent_vehicles.append(agent)
        log.info(f"Total agent vehicles {len(self.agent_vehicles)}")


    def reset(self):
        self.static_obstacles = []
        self.agent_vehicles = []


