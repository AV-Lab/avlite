from os import wait
from shapely.geometry import Polygon

from c10_perceive.c12_state import State, AgentState, EgoState
import logging
log = logging.getLogger(__name__)

class Environment:
    def __init__(self,  satatic_obstacles:list[State] = [], agent_vehicles:list[AgentState] = []):
        self.static_obstacles:list[State] = satatic_obstacles
        self.agent_vehicles:list[AgentState] = agent_vehicles

    def add_agent_vehicle(self, agent:AgentState):
        self.agent_vehicles.append(agent)
        log.info(f"Added agent vehicle {agent}")
        log.info(f"Total agent vehicles {len(self.agent_vehicles)}")

    # used for steping dynamic objects in the environment such as agent vehicles
    def step(self):
        pass

    def reset(self):
        for o in self.static_obstacles:
            o.reset()
        for o in self.agent_vehicles:
            o.reset()

    def is_collision_free(self):
        pass
        # v = self.ego_vehicle
        # for o in self.static_obstacles:
        #     if v.polygon.intersects(o.polygon):
        #         return False
        # for o in self.agent_vehicles:
        #     if v.polygon.intersects(o.polygon):
        #         return False




