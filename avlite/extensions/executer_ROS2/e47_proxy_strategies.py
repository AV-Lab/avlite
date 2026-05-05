"""
Proxy strategies for ROS executor.

These proxy classes don't perform actual computation - they store and return
data received from external ROS nodes, making it visible to AVLite's visualizer.
"""
import logging
from typing import Optional

from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c10_perception.c12_perception_strategy import PerceptionModel
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c20_planning.c23_local_planning_strategy import LocalPlannerStrategy
from avlite.c20_planning.c28_trajectory import Trajectory
from avlite.c30_control.c31_control_model import ControlComand
from avlite.c30_control.c32_control_strategy import ControlStrategy

log = logging.getLogger(__name__)


class ProxyLocalPlanner(LocalPlannerStrategy):
    """
    Proxy local planner that stores trajectory received from external ROS planner.
    
    This doesn't compute anything - it just holds the latest trajectory received
    via ROS topics, allowing the visualizer to display it.
    """
    
    def __init__(self, global_plan: GlobalPlan, pm: PerceptionModel, **kwargs):
        super().__init__(global_plan, pm, **kwargs)
        self.last_plan: Optional[Trajectory] = None
        log.info("ProxyLocalPlanner initialized - will receive plans from ROS")
    
    def replan(self) -> Trajectory:
        """
        Return the last received trajectory (no actual planning).
        
        The trajectory is updated externally by ROSExecuter._sync_ros_to_avlite().
        """
        # Return the externally-set plan, or fall back to global trajectory
        if self.last_plan is not None:
            return self.last_plan
        return self.global_trajectory
    
    def get_local_plan(self) -> Trajectory:
        """Return the last received trajectory."""
        if self.last_plan is not None:
            return self.last_plan
        return self.global_trajectory
    
    def reset(self, wp: int = 0):
        """Reset the proxy planner."""
        super().reset(wp)
        self.last_plan = None


class ProxyController(ControlStrategy):
    """
    Proxy controller that stores control commands received from external ROS controller.
    
    This doesn't compute anything - it just holds the latest control command
    received via ROS topics, allowing the visualizer to display it.
    """
    
    def __init__(self, tj: Optional[Trajectory] = None):
        super().__init__(tj)
        self.last_command: Optional[ControlComand] = None
        log.info("ProxyController initialized - will receive commands from ROS")
    
    def control(
        self,
        ego: EgoState,
        tj: Optional[Trajectory] = None,
        control_dt: float = None
    ) -> ControlComand:
        """
        Return the last received control command (no actual control computation).
        
        The command is updated externally by ROSExecuter._sync_ros_to_avlite().
        """
        if self.last_command is not None:
            self.cmd = self.last_command
            return self.last_command
        return self.cmd  # Return default (zero) command
    
    def reset(self):
        """Reset the proxy controller."""
        self.cmd = ControlComand()
        self.last_command = None
        self.cte_steer = 0
        self.cte_velocity = 0
