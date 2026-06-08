from __future__ import annotations

from enum import Enum
import numpy as np
from numpy.matlib import ndarray
from shapely.geometry import Polygon
from typing import Optional, Dict, Any
import copy
from dataclasses import dataclass, field
import logging

from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c60_common.c68_hdmap import HDMap


log = logging.getLogger(__name__)

class PredictionMode(Enum):
    TRAJECTORY = 1 # Outputs a single predicted trajectory for each agent, represented as a sequence of future positions and velocities over a specified prediction horizon.
    GMM = 2 # Gaussian Mixture Model - outputs a set of weighted trajectories
    OCCUPANCY_FLOW = 2 # Outputs a time sequence of occupancy grids representing the predicted positions of the agent over time. Each grid cell contains a probability of occupancy, and the flow aspect captures how these probabilities evolve across the prediction horizon.
    OCCUPANCY_FLOW_PER_AGENT = 3 # Similar to OCCUPANCY_FLOW but provides separate occupancy flow predictions for each individual agent, allowing for more granular and agent-specific future state estimations. Each entry in the list corresponds to a specific agent and contains its own sequence of occupancy grids, enabling the model to capture distinct movement patterns and interactions between agents in the environment.
    NONE = 4

@dataclass
class PerceptionModel:
    static_obstacles: list[State] = field(default_factory=list)
    agent_vehicles: list[AgentState] = field(default_factory=list)
    ego_vehicle: EgoState= field(default_factory=lambda: EgoState())
    max_agent_vehicles: int = PerceptionSettings.c11_max_agents
   
    # Optional HDMap
    hd_map: Optional[HDMap] = None
   
    # Optional Agent Prediction 
    prediction_mode: PredictionMode = PredictionMode.NONE
    trajectories : Optional[np.ndarray] = None # For single, multi,GMM results of predictor
    occupancy_flow: Optional[list[np.ndarray]] = None # list of 2D grids. Each list corresponds to a timestep in the prediction
    grid_bounds: Optional[Dict[str, float]] = None # Dictionary with bounds of the grid (min_x, max_x, min_y, max_y, resolution)
    predict_delta_t: float = 0.1
    grid_size: int = PerceptionSettings.c11_prediction_grid_size  # Size of the occupancy grid -> 100x100

    occupancy_flow_per_object:  Optional[list[tuple[int,list[np.ndarray]]]] = None # list(agent_id, list(2D grid))

    # Raw LiDAR points that passed segmentation + range gating (diagnostic overlay)
    detection_clusters: Optional[np.ndarray] = None



    def add_agent_vehicle(self, agent: AgentState) -> int: # return agent_id
        if len(self.agent_vehicles) == self.max_agent_vehicles:
            log.info("Max num of agent reached. Deleteing Old agents")
            self.agent_vehicles = []
        ids = {a.agent_id for a in self.agent_vehicles} # update agent id
        agent.agent_id = next(i for i in range(len(ids)+1) if i not in ids)
        self.agent_vehicles.append(agent)
        log.info(f"Total agent vehicles {len(self.agent_vehicles)}")

        return agent.agent_id

    
    def filter_agent_vehicles(self, distance_threshold: float = 100.0):
        """ Filter agent vehicles based on distance from the ego vehicle. -vectorized"""

        if not self.ego_vehicle:
            log.warning("Ego vehicle is not set. Cannot filter agent vehicles.")
            return
        
        if len(self.agent_vehicles) == 0:
            log.info("No agent vehicles to filter.")
            return
        
        ego_position = np.array([self.ego_vehicle.x, self.ego_vehicle.y])
        
        # Get agent positions as a 2D array
        agent_positions = np.array([[agent.x, agent.y] for agent in self.agent_vehicles])
       
        # Vectorized distance calculation
        distances = np.linalg.norm(agent_positions - ego_position, axis=1)

        # Boolean mask
        mask = distances < distance_threshold

        filtered_count = np.sum(mask)
        
        if filtered_count == 0:
            # log.warning(f"[{method_name}] No agents within {distance_threshold}m threshold. All agents filtered out.")
            self.agent_vehicles = []
        else:
            # Filter using boolean indexing
            agents_array = np.array(self.agent_vehicles, dtype=object)  
            self.agent_vehicles = list(agents_array[mask])
            
            # log.info(f"[{method_name}] Filtered agents: {filtered_count} kept")
        
    def agents_sizes_as_np(self) -> np.ndarray:
        """Get the sizes of all agent vehicles."""
        return np.array([[agent.length,agent.width,agent.theta] for agent in self.agent_vehicles])

    def reset(self):
        self.static_obstacles = []
        self.agent_vehicles = []



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
    """Ego vehicle state with additional properties in future."""
    pass


