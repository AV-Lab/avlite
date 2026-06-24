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
import json
from typing import Optional
from dataclasses import dataclass, field

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rcl_interfaces.msg import Log
from std_msgs.msg import String

from avlite.c40_execution.c42_executer import Executer
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c10_perception.c11_perception_model import EgoState, AgentState
from avlite.c60_common.c63_trajectory_tracker import TrajectoryTracker
from avlite.c60_common.c69_setting_utils import load_setting
from avlite.c30_control.c31_control_model import ControlComand

from .settings import PluginSettings
from .p46_autoware_converters import (
    AUTOWARE_AVAILABLE,
    ego_state_from_kinematic_state,
    trajectory_from_autoware,
    control_from_vehicle_cmd,
    agents_from_bounding_boxes,
)

log = logging.getLogger(__name__)

# Import Autoware messages if available
if AUTOWARE_AVAILABLE:
    from autoware_auto_msgs.msg import Trajectory as AutowareTrajectory
    from autoware_auto_msgs.msg import VehicleControlCommand
    from autoware_auto_msgs.msg import VehicleKinematicState
    from autoware_auto_msgs.msg import BoundingBoxArray

@dataclass
class ROSData:
    """Thread-safe container for data received from ROS topics."""
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    # Latest received data
    ego_state: Optional[EgoState] = None
    local_plan: Optional[TrajectoryTracker] = None
    control_cmd: Optional[ControlComand] = None
    agents: list[AgentState] = field(default_factory=list)
    
    # Prediction/heatmap data from perception
    prediction_mode: Optional[int] = None  # PredictionMode enum value
    occupancy_flow: Optional[list] = None  # list of 2D numpy arrays
    grid_bounds: Optional[dict] = None  # min_x, max_x, min_y, max_y, resolution
    predict_delta_t: float = 0.1
    trajectories: Optional[any] = None  # For trajectory predictions
    occupancy_flow_per_object: Optional[list] = None  # per-agent occupancy
    
    # Timestamps for staleness checking
    ego_stamp: float = 0.0
    plan_stamp: float = 0.0
    control_stamp: float = 0.0
    perception_stamp: float = 0.0
    
    # FPS counters (updated by nodes)
    perception_fps: float = 0.0
    planner_fps: float = 0.0
    control_fps: float = 0.0
    sim_fps: float = 0.0
    
    # Elapsed simulation time (from WorldNode)
    elapsed_sim_time: float = 0.0


class CollectorNode(Node):
    """
    ROS2 node that subscribes to Autoware topics and collects data.
    
    Subscribes to:
    - Localization (VehicleKinematicState or String) - ego pose and velocity
    - Planning (Trajectory or String) - trajectory from external planner
    - Control (VehicleControlCommand or String) - control from external controller
    - Perception (BoundingBoxArray or String) - detected/tracked objects
    """
    
    def __init__(self, ros_data: ROSData, settings: PluginSettings):
        super().__init__('avlite_collector')
        self.ros_data = ros_data
        self.settings = settings

        # Validate topic names early so misconfigured profiles surface a clear
        # error instead of a deep rclpy traceback.
        for attr in ("localization_topic", "trajectory_topic",
                     "control_cmd_topic", "perception_topic"):
            value = getattr(settings, attr, None)
            if not isinstance(value, str) or not value or " " in value or not value.startswith("/"):
                raise ValueError(
                    f"Invalid ROS topic for setting '{attr}': {value!r}. "
                    f"Topic names must be non-empty strings starting with '/' and contain no spaces. "
                    f"Check your plugin_ros_executer.yaml profile."
                )

        # Use Autoware messages only if available AND enabled in settings
        self.use_autoware = AUTOWARE_AVAILABLE and settings.use_autoware_msgs
        
        if self.use_autoware:
            # Subscribe to Autoware messages
            self.create_subscription(
                VehicleKinematicState,
                settings.localization_topic,
                self._localization_callback,
                10
            )
            self.create_subscription(
                AutowareTrajectory,
                settings.trajectory_topic,
                self._trajectory_callback,
                10
            )
            self.create_subscription(
                VehicleControlCommand,
                settings.control_cmd_topic,
                self._control_callback,
                10
            )
            self.create_subscription(
                BoundingBoxArray,
                settings.perception_topic,
                self._perception_callback,
                10
            )
        else:
            # Subscribe to JSON string messages
            self.create_subscription(
                String,
                settings.localization_topic,
                self._localization_json_callback,
                10
            )
            self.create_subscription(
                String,
                settings.trajectory_topic,
                self._trajectory_json_callback,
                10
            )
            self.create_subscription(
                String,
                settings.control_cmd_topic,
                self._control_json_callback,
                10
            )
            self.create_subscription(
                String,
                settings.perception_topic,
                self._perception_json_callback,
                10
            )
        
        # Subscribe to ROS log for debugging
        self.create_subscription(Log, '/rosout', self._rosout_callback, 10)
        
        self.get_logger().info(f"CollectorNode initialized (autoware={self.use_autoware})")
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
    
    def _control_callback(self, msg: 'VehicleControlCommand'):
        """Handle control command from external controller."""
        with self.ros_data.lock:
            self.ros_data.control_cmd = control_from_vehicle_cmd(msg)
            self.ros_data.control_stamp = time.time()
    
    def _perception_callback(self, msg: 'BoundingBoxArray'):
        """Handle bounding box array message."""
        with self.ros_data.lock:
            self.ros_data.agents = agents_from_bounding_boxes(msg)
            self.ros_data.perception_stamp = time.time()

    # JSON fallback callbacks
    def _localization_json_callback(self, msg: String):
        """Handle localization as JSON string."""
        try:
            data = json.loads(msg.data)
            with self.ros_data.lock:
                if self.ros_data.ego_state is None:
                    self.ros_data.ego_state = EgoState(x=0, y=0, theta=0)
                self.ros_data.ego_state.x = data.get('x', 0)
                self.ros_data.ego_state.y = data.get('y', 0)
                self.ros_data.ego_state.theta = data.get('theta', 0)
                self.ros_data.ego_state.velocity = data.get('velocity', 0)
                self.ros_data.ego_stamp = time.time()
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Invalid JSON in localization: {e}")

    def _trajectory_json_callback(self, msg: String):
        """Handle trajectory as JSON string."""
        try:
            data = json.loads(msg.data)
            with self.ros_data.lock:
                path = [(p['x'], p['y']) for p in data.get('points', [])]
                velocity = [p.get('velocity', 0) for p in data.get('points', [])]
                self.ros_data.local_plan = TrajectoryTracker(path=path, velocity=velocity)
                self.ros_data.local_plan.name = "ROS Trajectory"
                self.ros_data.plan_stamp = time.time()
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Invalid JSON in trajectory: {e}")

    def _control_json_callback(self, msg: String):
        """Handle control command as JSON string."""
        try:
            data = json.loads(msg.data)
            with self.ros_data.lock:
                self.ros_data.control_cmd = ControlComand(
                    steer=data.get('steer', 0),
                    acceleration=data.get('acceleration', 0)
                )
                self.ros_data.control_stamp = time.time()
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Invalid JSON in control: {e}")

    def _perception_json_callback(self, msg: String):
        """Handle perception as JSON string."""
        try:
            data = json.loads(msg.data)
            with self.ros_data.lock:
                agents = []
                for obj in data.get('objects', []):
                    agent = AgentState(
                        x=obj.get('x', 0),
                        y=obj.get('y', 0),
                        theta=obj.get('theta', 0),
                        velocity=obj.get('velocity', 0),
                        agent_id=obj.get('id', 0)
                    )
                    agents.append(agent)
                self.ros_data.agents = agents
                self.ros_data.perception_stamp = time.time()
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Invalid JSON in perception: {e}")
    
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
        
        self.settings = PluginSettings()
        load_setting(PluginSettings)

        # Override timing from settings
        self.perception_dt = self.settings.perception_dt
        self.replan_dt = self.settings.replan_dt
        self.control_dt = self.settings.control_dt
        
        # Shared data container for ROS callbacks
        self.ros_data = ROSData()

        # ROS components
        self.collector_node: Optional[CollectorNode] = None
        self.planner_node = None
        self.controller_node = None
        self.perception_node = None
        self.world_node = None
        self.ros_exec: Optional[SingleThreadedExecutor] = None
        self.spin_thread: Optional[threading.Thread] = None
        self.ros_started = False
        
        # Timing tracking
        self._start_real_time: float = 0.0
        self._planner_last_real_time: float = 0.0
        
        log.info("ROSExecuter initialized")

    def teleport_ego(self, x: float, y: float, theta: Optional[float] = None) -> None:
        """Teleport the ego vehicle and immediately flush ros_data to prevent revert.

        Without this override, _sync_ros_to_avlite() would overwrite the teleported
        position with the stale pre-teleport value still sitting in ros_data (which
        lags one ROS round-trip behind the world state).
        """
        self.world.teleport_ego(x, y, theta)
        with self.ros_data.lock:
            if self.ros_data.ego_state is None:
                self.ros_data.ego_state = EgoState(x=x, y=y, theta=theta or 0.0)
            else:
                self.ros_data.ego_state.x = x
                self.ros_data.ego_state.y = y
                if theta is not None:
                    self.ros_data.ego_state.theta = theta

    def step(
        self,
        perception_dt=0.01,
        control_dt=0.01,
        replan_dt=0.01,
        localization_dt=0.01,
        sim_dt=0.01,
        call_replan=True,
        call_control=True,
        call_perceive=True,
        call_localize=True,
    ) -> None:
        """
        Steps the executer for one time step.
        
        Runs AVLite planner/controller and publishes to ROS topics.
        Also syncs any received ROS data (localization, perception) to AVLite.
        """
        # Start ROS infrastructure if not already started
        if not self.ros_started:
            self._start_ros()
            self._start_real_time = time.time()
        
        # Get ego state from world
        self.ego_state = self.world.get_ego_state()
        
        # Run localization strategy (before perception)
        if call_localize and self.localization:
            try:
                self._localization_step()
            except Exception as e:
                log.debug(f"Localization step error: {e}")

        # Run perception strategy to populate occupancy_flow, etc.
        if call_perceive and self.perception:
            try:
                self._perception_step()
                # Update perception node's reference to pm
                if self.perception_node:
                    self.perception_node.pm = self.pm
            except Exception as e:
                log.debug(f"Perception step error: {e}")
        
        # Sync external ROS data (localization, perception) to AVLite
        self._sync_ros_to_avlite()
        
        # Sync FPS values from ROS nodes
        self._sync_fps()
        
        # Update elapsed real time
        self.elapsed_real_time = time.time() - self._start_real_time
        
        # Update local planner step (updates waypoint tracking)
        if self.local_planner:
            self.local_planner.step(self.ego_state)

        # Replan in main thread (not ROS spin thread — avoids blocking callbacks)
        now = time.time()
        if call_replan and self.local_planner and (now - self._planner_last_real_time) >= replan_dt:
            self._planner_last_real_time = now
            self.local_planner.replan()
            self.planner_fps = self._planner_fps_tracker.tick()
        
        # Run controller and apply to world
        if call_control and self.controller:
            local_tj = self.local_planner.get_local_plan()
            cmd = self.controller.control(self.ego_state, local_tj, control_dt=sim_dt)
            
            # Apply control command to world/simulator
            if cmd and self.world:
                self.world.control_ego_state(cmd, dt=sim_dt)
    
    def _sync_fps(self):
        """Sync FPS values from ROS nodes to executer."""
        with self.ros_data.lock:
            self.perception_fps = self.ros_data.perception_fps
            # planner_fps set at replan() call site in step()
            self.control_fps = self.ros_data.control_fps
            self.elapsed_sim_time = self.ros_data.elapsed_sim_time

    def _spin_ros(self):
        """Spin ROS executor in background thread."""
        try:
            self.ros_exec.spin()
        except Exception as e:
            log.error(f"ROS spin error: {e}")

    def _start_ros(self):
        """Initialize and start ROS components including world/perception/planner/controller nodes.

        When the world bridge owns its own ROS topics (e.g. ROS2WorldBridge), we skip the
        internal WorldNode, PerceptionNode, and ControllerNode to avoid duplicate publishers
        and feedback loops on the shared localization / perception / control topics.
        """
        if self.ros_started:
            return

        # Detect whether the world bridge manages its own ROS topics.
        # ROS2WorldBridge (and any future external bridge) sets owns_ros_topics = True.
        bridge_owns = getattr(self.world, "owns_ros_topics", False)

        # Initialize ROS
        if not rclpy.ok():
            rclpy.init()

        # Import node classes
        from .p42_perception_node import PerceptionNode
        from .p43_planner_node import PlannerNode
        from .p44_controller_node import ControllerNode
        from .p45_world_node import WorldNode

        # Create collector node (subscribes to external topics for visualizer sync)
        self.collector_node = CollectorNode(self.ros_data, self.settings)

        self.ros_exec = SingleThreadedExecutor()
        self.ros_exec.add_node(self.collector_node)

        if not bridge_owns:
            # Create world node (runs simulation step asynchronously)
            self.world_node = WorldNode(
                world=self.world,
                ros_data=self.ros_data,
                sim_dt=self.settings.sim_dt,
            )
            # Create perception node (publishes ego state and tracked objects)
            self.perception_node = PerceptionNode(
                ego_state=self.ego_state,
                perception_model=self.pm,
                world=self.world,
                ros_data=self.ros_data,
                perception_dt=self.settings.perception_dt,
            )
            # Create controller node (publishes control commands)
            self.controller_node = ControllerNode(
                controller=self.controller,
                ego_state=self.ego_state,
                ros_data=self.ros_data,
                control_dt=self.settings.control_dt,
            )
            self.ros_exec.add_node(self.world_node)
            self.ros_exec.add_node(self.perception_node)
            self.ros_exec.add_node(self.controller_node)
        else:
            log.info(
                "World bridge owns ROS topics – skipping WorldNode, PerceptionNode, ControllerNode"
            )

        # Create planner node (publishes trajectory; always active)
        self.planner_node = PlannerNode(
            planner=self.local_planner,
            ego_state=self.ego_state,
            ros_data=self.ros_data,
            replan_dt=self.settings.replan_dt,
        )
        self.ros_exec.add_node(self.planner_node)

        # Start spinning in separate thread
        self.spin_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self.spin_thread.start()

        self.ros_started = True
        log.info(
            "ROS infrastructure started (bridge_owns=%s) – "
            "sim_dt=%s  perception_dt=%s  replan_dt=%s  control_dt=%s",
            bridge_owns,
            self.settings.sim_dt,
            self.settings.perception_dt,
            self.settings.replan_dt,
            self.settings.control_dt,
        )

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
            
            # Sync prediction/heatmap data
            if self.ros_data.occupancy_flow is not None:
                self.pm.occupancy_flow = self.ros_data.occupancy_flow
            if self.ros_data.grid_bounds is not None:
                self.pm.grid_bounds = self.ros_data.grid_bounds
            if self.ros_data.prediction_mode is not None:
                from avlite.c10_perception.c11_perception_model import PredictionMode
                self.pm.prediction_mode = PredictionMode(self.ros_data.prediction_mode)
            if self.ros_data.predict_delta_t:
                self.pm.predict_delta_t = self.ros_data.predict_delta_t
            if self.ros_data.trajectories is not None:
                self.pm.trajectories = self.ros_data.trajectories
            if self.ros_data.occupancy_flow_per_object is not None:
                self.pm.occupancy_flow_per_object = self.ros_data.occupancy_flow_per_object

    @property
    def ui_poll_delay(self):
        # step() is lightweight — replanning runs in _replan_worker background thread.
        # Tell the UI to poll at 20 Hz rather than burning the main thread.
        return 0.05

    def stop(self):
        """Clean shutdown of ROS components."""
        self._stop_event.set()
        if not self.ros_started:
            return
            
        log.info("Stopping ROS infrastructure...")
        
        # Stop executor
        if self.ros_exec:
            self.ros_exec.shutdown()
        
        # Destroy nodes
        if self.collector_node:
            self.collector_node.destroy_node()
        if self.world_node:
            self.world_node.destroy_node()
        if self.perception_node:
            self.perception_node.destroy_node()
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
