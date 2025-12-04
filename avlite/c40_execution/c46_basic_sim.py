import math
from typing import Optional

from avlite.c10_perception.c11_perception_model import AgentState, PerceptionModel
from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c30_control.c32_control_strategy import ControlComand
from avlite.c40_execution.c41_execution_model import WorldBridge, WorldCapability, ExecutionSettings
from avlite.c30_control.c34_stanley import StanleyController


import logging

log = logging.getLogger(__name__)

class BasicSim(WorldBridge):
    @property
    def capabilities(self) -> frozenset[WorldCapability]:
        return frozenset({
            WorldCapability.GT_DETECTION,
            WorldCapability.GT_TRACKING,
            WorldCapability.GT_LOCALIZATION,
        })

    def __init__(self,ego_state:EgoState, pm:Optional[PerceptionModel] = None,
                 npc_control=ExecutionSettings.basic_sim_npc_control,
                 speed_factor=ExecutionSettings.basic_sim_npc_speed_factor,
                 default_trajectory=ExecutionSettings.basic_sim_default_trajectory):
        self.ego_state = ego_state
        self.pm = pm
        self.supports_ground_truth_detection = True
        self.supports_ground_truth_localization = True
        self.npc_control = npc_control
        self.speed_factor = speed_factor
        self.npc_controllers = {}

        log.info(f"Loading default trajectory from {default_trajectory}")
        if pm is not None and default_trajectory:
            try:  
                from avlite.c20_planning.c21_planning_model import GlobalPlan
                self.default_global_plan = GlobalPlan.from_file(default_trajectory)
                self.npc_control = True
            except Exception as e:
                log.error(f"Failed to load default trajectory {default_trajectory}: {e}")
    

    def control_ego_state(self, cmd:ControlComand, dt=0.01):
        acceleration = cmd.acceleration
        steering_angle = cmd.steer

        self.ego_state.x += self.ego_state.velocity * math.cos(self.ego_state.theta) * dt
        self.ego_state.y += self.ego_state.velocity * math.sin(self.ego_state.theta) * dt
        self.ego_state.velocity += acceleration * dt
        self.ego_state.theta += self.ego_state.velocity / self.ego_state.L_f * steering_angle * dt

        if self.npc_control:
            self.__control_npc_agents(dt)

    def __control_npc_agents(self, dt: float):
        """ Control NPC agents in the simulation. """
        for agent in self.pm.agent_vehicles:

            cmd = self.npc_controllers[agent.agent_id].control(agent,control_dt=dt) 

            agent_acceleration = cmd.acceleration
            agent_steering_angle = cmd.steer
            agent.x += agent.velocity * math.cos(agent.theta) * dt
            agent.y += agent.velocity * math.sin(agent.theta) * dt
            agent.velocity += agent_acceleration * dt
            agent.theta += agent.velocity / agent.L_f * agent_steering_angle * dt


        
    def spawn_agent(self, agent_state:AgentState):
        id = self.pm.add_agent_vehicle(agent_state)

        if self.npc_control:
            controllable_agent = EgoState(agent_state.x, agent_state.y, agent_state.theta, agent_state.velocity)

            controller = StanleyController(tj=self.default_global_plan.trajectory)
            controller.tj.velocity = [v* self.speed_factor for v in controller.tj.velocity] 
            controller.set_trajectory(self.default_global_plan.trajectory)

            self.npc_controllers[id] = controller

        

    def get_ego_state(self):

        return self.ego_state

    def teleport_ego(self, x: float, y: float, theta: Optional[float] = None):
        self.ego_state.x = x
        self.ego_state.y = y
        if theta is not None:
            self.ego_state.theta = theta


    def get_ground_truth_perception_model(self) -> PerceptionModel:
        return self.pm
    
