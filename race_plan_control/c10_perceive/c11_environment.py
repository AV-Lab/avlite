from shapely.geometry import Polygon
from c10_perceive.c12_state import State, AgentState, EgoState
from c20_plan.c24_trajectory import Trajectory


import logging
log = logging.getLogger(__name__)

class Environment:
    def __init__(self, ego_state : EgoState,  satatic_obstacles:list[State] = [], agent_vehicles:list[AgentState] = []):
        self.static_obstacles:list[State] = satatic_obstacles
        self.agent_vehicles:list[AgentState] = agent_vehicles
        self.ego_vehicle:EgoState = ego_state

    def add_agent_vehicle(self, agent:AgentState):
        self.agent_vehicles.append(agent)
        log.info(f"Total agent vehicles {len(self.agent_vehicles)}")



    def is_collision_free(self, ego: EgoState, trajectory: Trajectory):
        pass
        # v = self.ego_vehicle
        # for o in self.static_obstacles:
        #     if v.polygon.intersects(o.polygon):
        #         return False
        # for o in self.agent_vehicles:
        #     if v.polygon.intersects(o.polygon):
        #         return False




    def reset(self):
        self.static_obstacles = []
        self.agent_vehicles = []
