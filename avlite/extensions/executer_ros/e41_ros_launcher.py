"""
ROS Executor for AVLite with Autoware message support.

This executor subscribes to external ROS planner/controller outputs using Autoware
message types, syncs them into AVLite's internal state, and exposes the data to
the visualizer through proxy strategies.

Architecture:
- External ROS nodes handle planning and control (e.g., Autoware stack)
- ROSExecuter subscribes to their outputs via Autoware message topics
- Received data is converted and stored in AVLite data structures
- Visualizer sees data through standard AVLite interfaces
"""

import threading
import time
import logging
from typing import Optional
from dataclasses import dataclass, field

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rcl_interfaces.msg import Log

from avlite.c40_execution.c41_execution_model import Executer
from avlite.c10_perception.c11_perception_model import EgoState, AgentState
from avlite.c20_planning.c28_trajectory import Trajectory
from avlite.c30_control.c31_control_model import ControlComand

from .settings import ExtensionSettings
from .e46_autoware_converters import (
    AUTOWARE_AVAILABLE,
    ego_state_from_kinematic_state,
    trajectory_from_autoware,
    control_from_ackermann,
    agents_from_tracked_objects,
)

log = logging.getLogger(__name__)

# Import Autoware messages if available
if AUTOWARE_AVAILABLE:
    from autoware_auto_planning_msgs.msg import Trajectory as AutowareTrajectory
    from autoware_auto_control_msgs.msg import AckermannControlCommand
    from autoware_auto_vehicle_msgs.msg import VehicleKinematicState
    from autoware_auto_perception_msgs.msg import TrackedObjects

@dataclass
class ROSData:
    """Thread-safe container for data received from ROS topics."""
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    # Latest received data
    ego_state: Optional[EgoState] = None
    local_plan: Optional[Trajectory] = None
    control_cmd: Optional[ControlComand] = None
    agents: list[AgentState] = field(default_factory=list)
    
    # Timestamps for staleness checking
    ego_stamp: float = 0.0
    plan_stamp: float = 0.0
    control_stamp: float = 0.0
    perception_stamp: float = 0.0


class CollectorNode(Node):
    """
    ROS2 node that subscribes to Autoware topics and collects data.
    
    Subscribes to:
    - Localization (VehicleKinematicState) - ego pose and velocity
    - Planning (Trajectory) - trajectory from external planner
    - Control (AckermannControlCommand) - control from external controller
    - Perception (TrackedObjects) - detected/tracked objects
    """
    
    def __init__(self, ros_data: ROSData, settings: ExtensionSettings):
        super().__init__('avlite_collector')
        self.ros_data = ros_data
        self.settings = settings
        
        if not AUTOWARE_AVAILABLE:
            self.get_logger().error("Autoware messages not available!")
            return
        
        # Subscribe to localization
        self.create_subscription(
            VehicleKinematicState,
            settings.localization_topic,
            self._localization_callback,
            10
        )
        
        # Subscribe to trajectory from external planner
        self.create_subscription(
            AutowareTrajectory,
            settings.trajectory_topic,
            self._trajectory_callback,
            10
        )
        
        # Subscribe to control command from external controller
        self.create_subscription(
            AckermannControlCommand,
            settings.control_cmd_topic,
            self._control_callback,
            10
        )
        
        # Subscribe to tracked objects
        self.create_subscription(
            TrackedObjects,
            settings.perception_topic,
            self._perception_callback,
            10
        )
        
        # Subscribe to ROS log for debugging
        self.create_subscription(Log, '/rosout', self._rosout_callback, 10)
        
        self.get_logger().info(f"CollectorNode initialized, subscribing to:")
        self.get_logger().info(f"  Localization: {settings.localization_topic}")
        self.get_logger().info(f"  Trajectory: {settings.trajectory_topic}")
        self.get_logger().info(f"  Control: {settings.control_cmd_topic}")
        self.get_logger().info(f"  Perception: {settings.perception_topic}")
    
    def _localization_callback(self, msg: 'VehicleKinematicState'):
        """Handle localization message."""
        with self.ros_data.lock:
            if self.ros_data.ego_state is None:
                self.ros_data.ego_state = EgoState(x=0, y=0, theta=0)
            ego_state_from_kinematic_state(msg, self.ros_data.ego_state)
            self.ros_data.ego_stamp = time.time()
    
    def _trajectory_callback(self, msg: 'AutowareTrajectory'):
        """Handle trajectory message from external planner."""
        with self.ros_data.lock:
            self.ros_data.local_plan = trajectory_from_autoware(msg)
            self.ros_data.local_plan.name = "ROS Trajectory"
            self.ros_data.plan_stamp = time.time()
    
    def _control_callback(self, msg: 'AckermannControlCommand'):
        """Handle control command from external controller."""
        with self.ros_data.lock:
            self.ros_data.control_cmd = control_from_ackermann(msg)
            self.ros_data.control_stamp = time.time()
    
    def _perception_callback(self, msg: 'TrackedObjects'):
        """Handle tracked objects message."""
        with self.ros_data.lock:
            self.ros_data.agents = agents_from_tracked_objects(msg)
            self.ros_data.perception_stamp = time.time()
    
    def _rosout_callback(self, msg: Log):
        """Forward ROS log to Python logging."""
        level = msg.level
        text = msg.msg
        if level >= 40:  # ERROR
            log.error(f"[ROS] {text}")
        elif level >= 30:  # WARN
            log.warning(f"[ROS] {text}")
        elif level >= 20:  # INFO
            log.debug(f"[ROS] {text}")  # Demote to debug to reduce noise


class ROSExecuter(Executer):
    """
    Executer that interfaces with ROS/Autoware system.
    
    Runs AVLite planner/controller as ROS nodes that publish Autoware messages.
    Also subscribes to external topics for localization and perception.
    
    Topics Published:
    - trajectory_out_topic: Trajectory from local planner
    - control_out_topic: Control command from controller
    
    Topics Subscribed:
    - localization_topic: Ego state (from external localization)
    - perception_topic: Tracked objects (from external perception)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.settings = ExtensionSettings()
        
        # Shared data container for ROS callbacks
        self.ros_data = ROSData()

        # ROS components
        self.collector_node: Optional[CollectorNode] = None
        self.planner_node = None
        self.controller_node = None
        self.ros_exec: Optional[SingleThreadedExecutor] = None
        self.spin_thread: Optional[threading.Thread] = None
        self.ros_started = False
        
        log.info("ROSExecuter initialized")

    def step(
        self,
        perception_dt=0.01,
        control_dt=0.01,
        replan_dt=0.01,
        sim_dt=0.01,
        call_replan=True,
        call_control=True,
        call_perceive=True
    ) -> None:
        """
        Steps the executer for one time step.
        
        Runs AVLite planner/controller and publishes to ROS topics.
        Also syncs any received ROS data (localization, perception) to AVLite.
        """
        # Start ROS infrastructure if not already started
        if not self.ros_started:
            self._start_ros()
        
        # Get ego state from world
        self.ego_state = self.world.get_ego_state()
        
        # Sync external ROS data (localization, perception) to AVLite
        self._sync_ros_to_avlite()
        
        # Update local planner step (updates waypoint tracking)
        if self.local_planner:
            self.local_planner.step(self.ego_state)
        
        # Run planner (replan if needed)
        if call_replan and self.local_planner:
            self.local_planner.replan()
        
        # Run controller and apply to world
        if call_control and self.controller:
            local_tj = self.local_planner.get_local_plan()
            cmd = self.controller.control(self.ego_state, local_tj, control_dt=sim_dt)
            
            # Apply control command to world/simulator
            if cmd and self.world:
                self.world.control_ego_state(cmd, dt=sim_dt)

    def _start_ros(self):
        """Initialize and start ROS components including planner/controller nodes."""
        if self.ros_started:
            return
            
        # Initialize ROS
        if not rclpy.ok():
            rclpy.init()
        
        # Import node classes
        from .e43_planner_node import PlannerNode
        from .e44_controller_node import ControllerNode
        
        # Create collector node (subscribes to external topics)
        self.collector_node = CollectorNode(self.ros_data, self.settings)
        
        # Create planner node (publishes trajectory)
        self.planner_node = PlannerNode(
            planner=self.local_planner,
            ego_state=self.ego_state
        )
        
        # Create controller node (publishes control commands)
        self.controller_node = ControllerNode(
            controller=self.controller,
            ego_state=self.ego_state
        )
        
        # Create executor and add all nodes
        self.ros_exec = SingleThreadedExecutor()
        self.ros_exec.add_node(self.collector_node)
        self.ros_exec.add_node(self.planner_node)
        self.ros_exec.add_node(self.controller_node)
        
        # Start spinning in separate thread
        self.spin_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self.spin_thread.start()
        
        self.ros_started = True
        log.info("ROS infrastructure started with planner and controller nodes")

    def _spin_ros(self):
        """Spin ROS executor in background thread."""
        try:
            self.ros_exec.spin()
        except Exception as e:
            log.error(f"ROS spin error: {e}")

    def _sync_ros_to_avlite(self):
        """
        Sync data received from ROS to AVLite's internal state.
        
        This makes data visible to the visualizer through standard interfaces.
        """
        with self.ros_data.lock:
            # Sync ego state
            if self.ros_data.ego_state is not None:
                self.ego_state.x = self.ros_data.ego_state.x
                self.ego_state.y = self.ros_data.ego_state.y
                self.ego_state.theta = self.ros_data.ego_state.theta
                self.ego_state.velocity = self.ros_data.ego_state.velocity
            
            # Sync local plan to local_planner's last_plan
            if self.ros_data.local_plan is not None:
                self.local_planner.last_plan = self.ros_data.local_plan
            
            # Sync control command to controller's last_command
            if self.ros_data.control_cmd is not None:
                self.controller.last_command = self.ros_data.control_cmd
            
            # Sync perceived agents
            if self.ros_data.agents:
                self.pm.agent_vehicles = self.ros_data.agents

    def stop(self):
        """Clean shutdown of ROS components."""
        if not self.ros_started:
            return
            
        log.info("Stopping ROS infrastructure...")
        
        # Stop executor
        if self.ros_exec:
            self.ros_exec.shutdown()
        
        # Destroy nodes
        if self.collector_node:
            self.collector_node.destroy_node()
        if self.planner_node:
            self.planner_node.destroy_node()
        if self.controller_node:
            self.controller_node.destroy_node()
        
        # Shutdown ROS
        if rclpy.ok():
            rclpy.shutdown()
        
        # Wait for thread to finish
        if self.spin_thread and self.spin_thread.is_alive():
            self.spin_thread.join(timeout=2.0)
        
        self.ros_started = False
        log.info("ROS infrastructure stopped")
