from typing import Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from c30_control.c31_control_model import  ControlComand
from c10_perception.c11_perception_model import PerceptionModel, EgoState, AgentState

@dataclass
class WorldInterface(ABC):
    """
    Abstract class for the world interface. This class is used to control the ego vehicle and spawn agents in the world.
    It provides an interface for the simulator or ROS bridge to implement its own world logic.
    """
    
    ego_state: EgoState
    perception_model: Optional[PerceptionModel] = None # Simulators can provide ground truth perception model

    @abstractmethod
    def control_ego_state(self, cmd: ControlComand, dt:Optional[float]=0.01):
        """
        Update the ego state.

        Parameters
        cmd (ControlCommand): The control command containing acceleration and steering angle.
        dt (float): Time delta for the update if supported. Default is 0.01.
        """
        pass

    def get_ego_state(self) -> EgoState:
        return self.ego_state

    def get_perception_model(self) -> PerceptionModel:
        raise NotImplementedError("This method should be implemented by the simulator or ROS bridge.")

    @abstractmethod
    def teleport_ego(self, x: float, y: float, theta: Optional[float] = None):
        """
        Teleport the ego vehicle to a new position and orientation.

        Parameters
        x (float): The new x-coordinate.
        y (float): The new y-coordinate.
        theta (float): The new orientation in radians.
        """
        pass

    @abstractmethod
    def spawn_agent(self, agent_state: AgentState):
        """ Spawn an agent vehicled in a (simulated) world. Its optional if the world allows that. """
        pass

    def safety_stop(self):
        """ Stop the ego vehicle safely. This method should be implemented by the simulator or ROS bridge. """
        raise NotImplementedError("This method should be implemented by the simulator or ROS bridge.")

    def reset(self):
        pass
