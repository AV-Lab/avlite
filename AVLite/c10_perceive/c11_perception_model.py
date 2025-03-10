
from shapely.geometry import Polygon
from c10_perceive.c12_state import State, AgentState, EgoState
from c20_plan.c24_trajectory import Trajectory
import numpy as np


import logging

log = logging.getLogger(__name__)


class PerceptionModel:
    # static_obstacles: list[State]
    agent_vehicles: list[AgentState]
    ego_vehicle: EgoState
    max_agent_vehicles:int = 12

    def __init__(self, ego_state: EgoState = None, satatic_obstacles: list[State] = [], agent_vehicles: list[AgentState] = []):
        self.static_obstacles: list[State] = satatic_obstacles
        self.agent_vehicles: list[AgentState] = agent_vehicles
        self.ego_vehicle: EgoState = ego_state if ego_state is not None else EgoState()

    def add_agent_vehicle(self, agent: AgentState):
        if len(self.agent_vehicles) == self.max_agent_vehicles:
            log.info("Max num of agent reached. Deleteing Old agents")
            self.agent_vehicles = []
        self.agent_vehicles.append(agent)
        log.info(f"Total agent vehicles {len(self.agent_vehicles)}")

    def is_tj_collision_free(self, trajectory: Trajectory = None, sample_size=5):
        ego = self.ego_vehicle
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

                for agent in self.agent_vehicles:
                    if ego.get_bb_polygon().intersects(agent.get_bb_polygon()):
                        log.debug(f"Collision at {ego.x}, {ego.y}")
                        return False
        else:
            for agent in self.agent_vehicles:
                if ego.get_bb_polygon().intersects(agent.get_bb_polygon()):
                    log.info(f"Collision at {ego.x}, {ego.y}")
                    return False
        return True

    def reset(self):
        self.static_obstacles = []
        self.agent_vehicles = []
