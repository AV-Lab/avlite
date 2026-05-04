from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import threading
import time
import numpy as np
from enum import Enum, auto


from avlite.c10_perception.c11_perception_model import PerceptionModel, EgoState, AgentState
from avlite.c30_control.c31_control_model import  ControlComand
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c10_perception.c12_perception_strategy import PerceptionStrategy
from avlite.c10_perception.c13_localization_strategy import LocalizationStrategy
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c23_local_planning_strategy import LocalPlannerStrategy
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c60_common.c61_setting_utils import reload_lib, get_absolute_path, import_all_modules
from avlite.c60_common.c62_capabilities import WorldCapability


log = logging.getLogger(__name__)


@dataclass
class WorldBridge(ABC):
    """
    Abstract class for the world interface. This class is used to control the ego vehicle and spawn agents in the world.
    It provides an interface for the simulator or ROS bridge to implement its own world logic.
    """
    
    ego_state: EgoState
    perception_model: Optional[PerceptionModel] = None # Simulators can provide ground truth perception model

    registry = {}

    
    @property
    @abstractmethod
    def capabilities(self) -> set[WorldCapability]:
        """Set of supported capabilities (must be implemented by subclass)."""
        pass

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

    def teleport_ego(self, x: float, y: float, theta: Optional[float] = None):
        """
        Teleport the ego vehicle to a new position and orientation.

        Parameters
        x (float): The new x-coordinate.
        y (float): The new y-coordinate.
        theta (float): The new orientation in radians.
        """
        raise NotImplementedError("This method should be implemented by the simulator or ROS bridge.")

    def spawn_agent(self, agent_state: AgentState):
        """ Spawn an agent vehicled in a (simulated) world. Its optional if the world allows that. """
        raise NotImplementedError("This method should be implemented by the simulator or ROS bridge.")

    def get_ground_truth_perception_model(self) -> PerceptionModel:
        """ Returns the perception model of the world. This method should be implemented by simulators  """
        raise NotImplementedError("This method should be implemented by the simulator or ROS bridge.")

    def get_rgb_image(self) -> np.ndarray | None:
        """ Returns the RGB image of the world. This method should be implemented by simulators """
        return None

    def get_depth_image(self) -> np.ndarray | None:
        """ Returns the depth image of the world. This method should be implemented by simulators """
        return None
    
    def get_lidar_data(self) -> np.ndarray | None:
        """ Returns the lidar data of the world. This method should be implemented by simulators """
        return None

    def reset(self):
        pass
    
    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:  
            WorldBridge.registry[cls.__name__] = cls


class Executer(ABC):
    registry = {}
    
    def __init__(
        self,
        perception_model: PerceptionModel,
        perception: Optional[PerceptionStrategy],
        global_planner: GlobalPlannerStrategy,
        local_planner: LocalPlannerStrategy,
        controller: ControlStrategy,
        world: WorldBridge,
        localization: Optional[LocalizationStrategy] = None,
        perception_dt=0.5,
        replan_dt=0.5,
        control_dt=0.01,
        localization_dt=0.1,
    ):
        """
        Initializes the SyncExecuter with the given perception model, global planner, local planner, control strategy, and world interface.
        """
        self.pm: PerceptionModel = perception_model
        self.perception: Optional[PerceptionStrategy] = perception  
        self.localization: Optional[LocalizationStrategy] = localization
        self.ego_state: EgoState = perception_model.ego_vehicle
        self.global_planner: GlobalPlannerStrategy = global_planner
        self.local_planner: LocalPlannerStrategy = local_planner
        self.controller: ControlStrategy = controller
        self.world: WorldBridge = world

        self.perception_fps: float = 0.0
        self.planner_fps: float = 0.0
        self.control_fps: float = 0.0
        self.localization_fps: float = 0.0

        self.perception_dt: float = perception_dt
        self.replan_dt: float = replan_dt
        self.control_dt: float = control_dt
        self.localization_dt: float = localization_dt

        self.elapsed_real_time = 0
        self.elapsed_sim_time = 0

        self._stop_event = threading.Event()

    @abstractmethod
    def step(self, perception_dt=0.01, control_dt=0.01, replan_dt=0.01, localization_dt=0.01, sim_dt=0.01, call_replan=True, call_control=True, call_perceive=True, call_localize=True,) -> None:
        """ Steps the executer for one time step. This method should be implemented by the specific executer class. """
        pass

    def run(self, replan_dt=None, control_dt=None, call_replan=True, call_control=True, call_perceive=False, call_localize=True):
        """Generic run loop that repeatedly calls step() until stop() is called.

        Subclasses may override for specialized scheduling.
        """
        self.reset()
        self._stop_event.clear()
        rdt = replan_dt if replan_dt is not None else self.replan_dt
        cdt = control_dt if control_dt is not None else self.control_dt
        pdt = self.perception_dt
        ldt = self.localization_dt
        sdt = ExecutionSettings.sim_dt
        while not self._stop_event.is_set():
            try:
                self.step(
                    perception_dt=pdt,
                    control_dt=cdt,
                    replan_dt=rdt,
                    localization_dt=ldt,
                    sim_dt=sdt,
                    call_replan=call_replan,
                    call_control=call_control,
                    call_perceive=call_perceive,
                    call_localize=call_localize,
                )
            except Exception as e:
                log.error(f"Executer step error: {e}", exc_info=True)
            if self._stop_event.wait(timeout=cdt):
                break

    def stop(self):
        """Signal run() to exit. Subclasses may override to also tear down threads/resources."""
        self._stop_event.set()

    def reset(self):
        self.pm.reset()
        self.ego_state.reset()
        if self.perception:
            self.perception.reset()
        if self.localization:
            self.localization.reset()
        self.local_planner.reset()
        self.controller.reset()
        self.world.reset()
        self.elapsed_real_time = 0
        self.elapsed_sim_time = 0


    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:  
            Executer.registry[cls.__name__] = cls


    
