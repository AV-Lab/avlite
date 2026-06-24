"""
Proxy strategies for ROS executor.

These proxy classes don't perform actual computation - they store and return
data received from external ROS nodes, making it visible to AVLite's visualizer.
"""
import logging
from typing import Optional

from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c10_perception.c12_perception_strategy import PerceptionModel
from avlite.c20_planning.c21_planning_model import GlobalPlan, LocalPlan
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c60_common.c63_trajectory_tracker import TrajectoryTracker
from avlite.c30_control.c31_control_model import ControlCommand
from avlite.c30_control.c32_control_strategy import ControlStrategy

log = logging.getLogger(__name__)


class ProxyLocalPlanner(LocalPlanningStrategy):
    """
    Proxy local planner that stores trajectory received from external ROS planner.
    
    This doesn't compute anything - it just holds the latest trajectory received
    via ROS topics, allowing the visualizer to display it.
    """
    
    def __init__(self, global_plan: GlobalPlan, pm: PerceptionModel, **kwargs):
        super().__init__(global_plan, pm, **kwargs)
        self.last_plan: Optional[TrajectoryTracker] = None
        log.info("ProxyLocalPlanner initialized - will receive plans from ROS")
    
    def replan(self) -> TrajectoryTracker:
        """
        Return the last received trajectory (no actual planning).
        
        The trajectory is updated externally by ROSExecuter._sync_ros_to_avlite().
        """
        # Return the externally-set plan, or fall back to global trajectory
        if self.last_plan is not None:
            return self.last_plan
        return self.global_trajectory
    
    def get_local_plan(self) -> LocalPlan:
        """Return the last received trajectory wrapped as a LocalPlan."""
        if self.last_plan is not None:
            return LocalPlan.from_trajectory(self.last_plan)
        return LocalPlan.from_trajectory(self.global_trajectory)
    
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
    
    def __init__(self, tj: Optional[TrajectoryTracker] = None):
        super().__init__(tj)
        self.last_command: Optional[ControlCommand] = None
        log.info("ProxyController initialized - will receive commands from ROS")
    
    def control(
        self,
        ego: EgoState,
        plan: Optional[LocalPlan] = None,
        control_dt: float = None
    ) -> ControlCommand:
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
        self.cmd = ControlCommand()
        self.last_command = None
        self.cte_steer = 0
        self.cte_velocity = 0
