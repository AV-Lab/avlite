from typing import Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging 
import numpy as np


from c10_perception.c11_perception_model import PerceptionModel, EgoState, AgentState
from c30_control.c31_control_model import  ControlComand
from c40_execution.c49_settings import ExecutionSettings
from c10_perception.c12_perception_strategy import PerceptionStrategy
from c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from c20_planning.c23_local_planning_strategy import LocalPlannerStrategy
from c30_control.c32_control_strategy import ControlStrategy
# from c20_planning.c21_planning_model import GlobalPlan
# from c20_planning.c24_global_planners import RaceGlobalPlanner
# from c40_execution.c44_basic_sim import BasicSim
from c60_tools.c61_utils import reload_lib, get_absolute_path

log = logging.getLogger(__name__)

@dataclass
class WorldInterface(ABC):
    """
    Abstract class for the world interface. This class is used to control the ego vehicle and spawn agents in the world.
    It provides an interface for the simulator or ROS bridge to implement its own world logic.
    """
    
    ego_state: EgoState
    perception_model: Optional[PerceptionModel] = None # Simulators can provide ground truth perception model
    supports_ground_truth_perception: bool = False  # Whether the world supports ground truth perception model
    supports_rgb_image: bool = False  # Whether the world supports RGB support_rgb_image
    supports_depth_image: bool = False  # Whether the world supports depth image 
    supports_lidar_data: bool = False  # Whether the world supports lidar data  

    registry = {}

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

    def get_rgb_image(self) -> np.ndarray:
        """ Returns the RGB image of the world. This method should be implemented by simulators """
        raise NotImplementedError("This method should be implemented by the simulator or ROS bridge.")

    def get_depth_image(self) -> np.ndarray:
        """ Returns the depth image of the world. This method should be implemented by simulators """
        raise NotImplementedError("This method should be implemented by the simulator or ROS bridge.")
    
    def get_lidar_data(self) -> np.ndarray:
        """ Returns the lidar data of the world. This method should be implemented by simulators """
        raise NotImplementedError("This method should be implemented by the simulator or ROS bridge.")

    def reset(self):
        pass
    
    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:  
            Executer.registry[cls.__name__] = cls


class Executer(ABC):
    registry = {}
    
    def __init__(
        self,
        perception_model: PerceptionModel,
        perception: PerceptionStrategy,
        global_planner: GlobalPlannerStrategy,
        local_planner: LocalPlannerStrategy,
        controller: ControlStrategy,
        world: WorldInterface,
        replan_dt=0.5,
        control_dt=0.01,
    ):
        """
        Initializes the SyncExecuter with the given perception model, global planner, local planner, control strategy, and world interface.
        """
        self.pm: PerceptionModel = perception_model
        self.perception: PerceptionStrategy = perception  
        self.ego_state: EgoState = perception_model.ego_vehicle
        self.global_planner: GlobalPlannerStrategy = global_planner
        self.local_planner: LocalPlannerStrategy = local_planner
        self.controller: ControlStrategy = controller
        self.world: WorldInterface = world
        self.planner_fps: float = 0.0
        self.control_fps: float = 0.0

        self.replan_dt: float = replan_dt
        self.control_dt: float = control_dt

        self.elapsed_real_time = 0
        self.elapsed_sim_time = 0

    @abstractmethod
    def step(self, control_dt=0.01, replan_dt=0.01, sim_dt=0.01, call_replan=True, call_control=True, call_perceive=True,) -> None:
        """ Steps the executer for one time step. This method should be implemented by the specific executer class. """
        pass

    def run(self, replan_dt=0.5, control_dt=0.01, call_replan=True, call_control=True, call_perceive=False):
        raise NotImplementedError("This method should be implemented by the specific executer class.")

    def stop(self):
        raise NotImplementedError("This method should be implemented by the specific executer class.")

    def reset(self):
        self.pm.reset()
        self.ego_state.reset()
        self.perception.reset()
        self.local_planner.reset()
        self.controller.reset()
        self.world.reset()
        self.elapsed_real_time = 0
        self.elapsed_sim_time = 0


    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:  
            Executer.registry[cls.__name__] = cls


    @classmethod
    def executor_factory(cls,
        async_mode = ExecutionSettings.async_mode,
        bridge = ExecutionSettings.bridge,
        perception = ExecutionSettings.perception,
        global_planner = ExecutionSettings.global_planner,
        local_planner = ExecutionSettings.local_planner,
        controller = ExecutionSettings.controller,
        replan_dt = ExecutionSettings.replan_dt,
        control_dt = ExecutionSettings.control_dt,
        global_trajectory = ExecutionSettings.global_trajectory,
        hd_map = ExecutionSettings.hd_map,
        reload_code = True,
        exclude_reload_settings = False,
        profile=None
    ) -> "Executer":
        """
        Factory method to create an instance of the Executer class based on the provided configuration.
        """

        if reload_code:
            reload_lib(exclude_settings=exclude_reload_settings,)

        from c10_perception.c11_perception_model import PerceptionModel, EgoState
        from c10_perception.c12_perception_strategy import PerceptionStrategy
        from c20_planning.c21_planning_model import GlobalPlan
        from c20_planning.c24_global_planners import HDMapGlobalPlanner
        from c20_planning.c24_global_planners import RaceGlobalPlanner
        from c20_planning.c26_local_planners import RNDPlanner
        from c30_control.c33_pid import PIDController
        from c30_control.c34_stanley import StanleyController
        from c40_execution.c42_sync_executer import SyncExecuter
        from c40_execution.c44_async_mproc_executer import AsyncExecuter
        from c40_execution.c43_async_threaded_executer import AsyncThreadedExecuter
        from c40_execution.c44_basic_sim import BasicSim


        #TODO: this should be dynamic loading (fix later with config and profiles, to load only selected extensions
        from extensions.multi_object_prediction.e10_perception.perception import MultiObjectPredictor

        # if reload_code and profile is not None:
            # load_all_settings(profile)


        #################
        global_plan_path =  get_absolute_path(global_trajectory)

        # Loading default
        # global planner
        #################
        default_global_plan = GlobalPlan.from_file(global_plan_path)
        
        if global_planner is None:
            gp = RaceGlobalPlanner()
            gp.global_plan = default_global_plan
            log.debug("RaceGlobalPlanner loaded by default. If you want to use a different global planner, please specify it in the config file or as an argument.")

        elif global_planner == RaceGlobalPlanner.__name__:
            gp = RaceGlobalPlanner()
            gp.global_plan = default_global_plan
            log.debug("RaceGlobalPlanner loaded")
        elif global_planner == HDMapGlobalPlanner.__name__:
            gp = HDMapGlobalPlanner(xodr_file=hd_map)
            log.debug("GlobalHDMapPlanner loaded")

        ego_state = EgoState(x=default_global_plan.start_point[0], y=default_global_plan.start_point[1], velocity=20, theta=-np.pi / 4)
        pm = PerceptionModel(ego_vehicle=ego_state)

        ############################
        # Loading perception strategy
        ##############################
        pr = None
        if perception is not None and perception != "" and perception in  PerceptionStrategy.registry:
            # load the class
            cls = PerceptionStrategy.registry[perception]
            pr = cls(perception_model=pm)
            log.info("Perception Module Loaded!")
        # perception = MultiObjectPredictor(perception_model=pm)

        #################
        # Loading world
        #################
        if bridge == "CarlaBridge":
            print("Loading Carla bridge...")
            from c40_execution.c45_carla_bridge import CarlaBridge
            world = CarlaBridge(ego_state=ego_state)
        elif bridge == "GazeboIgnitionBridge":
            print("Loading Gazebo bridge...")
            from c40_execution.c47_gazebo_bridge import GazeboIgnitionBridge
            world = GazeboIgnitionBridge(ego_state=ego_state)
        else:
            world = BasicSim(ego_state=ego_state, pm = pm)


        #################
        # Loading planner
        #################
        if local_planner is None or local_planner == RNDPlanner.__name__:
            local_planner = RNDPlanner(global_plan=default_global_plan, env=pm)
        else:
            log.error(f"Local planner {local_planner} not recognized. Using RNDPlanner as default.")
            local_planner = RNDPlanner(global_plan=default_global_plan, env=pm)

        #################
        # Loading planner
        #################
        if controller is None or controller == PIDController.__name__:
            controller = PIDController()
        elif controller == StanleyController.__name__:
            controller = StanleyController()
        else:
            log.error(f"Controller {controller} not recognized. Using PIDController as default.")
            controller = PIDController()


        #################
        # Creating Executer
        #################
        executer = (
            SyncExecuter(perception_model=pm,perception=pr, global_planner=gp, local_planner=local_planner, controller=controller, world=world, replan_dt=replan_dt, control_dt=control_dt)
            if not async_mode
            else AsyncThreadedExecuter(perception_model=pm,perception=pr, global_planner=gp, local_planner=local_planner, controller=controller, world=world, replan_dt=replan_dt, control_dt=control_dt)
        )

        return executer
