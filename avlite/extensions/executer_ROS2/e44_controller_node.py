#!/usr/bin/env python3
"""
ROS2 Controller Node - Runs AVLite controller and publishes control commands.

This node:
1. Subscribes to localization (ego state)
2. Subscribes to trajectory from planner
3. Runs AVLite controller
4. Publishes control command as Autoware message (or std_msgs as fallback)

Launch standalone:
    ros2 run avlite controller_node --ros-args -p controller_name:=StanleyController -p control_dt:=0.02
"""
import logging
import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c20_planning.c28_trajectory import Trajectory
from avlite.c30_control.c32_control_strategy import ControlStrategy

from .e46_autoware_converters import (
    AUTOWARE_AVAILABLE,
    ego_state_from_kinematic_state,
    trajectory_from_autoware,
    control_to_vehicle_cmd,
)
from .settings import ExtensionSettings

log = logging.getLogger(__name__)

if AUTOWARE_AVAILABLE:
    from autoware_auto_msgs.msg import VehicleKinematicState
    from autoware_auto_msgs.msg import Trajectory as AutowareTrajectory
    from autoware_auto_msgs.msg import VehicleControlCommand


class ControllerNode(Node):
    """
    ROS2 node that runs AVLite controller and publishes control commands.
    
    Can be used in two ways:
    1. Standalone: Pass controller via ROS params (loads from registry)
    2. Embedded: Pass controller instance directly to constructor
    """

    def __init__(self, controller: ControlStrategy = None, ego_state: EgoState = None, ros_data=None, control_dt: float = 0.02):
        super().__init__('avlite_controller')
        
        self.settings = ExtensionSettings()
        self.controller = controller
        self.ego_state = ego_state if ego_state else EgoState(x=0, y=0, theta=0)
        self.current_trajectory: Trajectory = None
        self.ros_data = ros_data
        # Use Autoware messages only if available AND enabled in settings
        self.use_autoware = AUTOWARE_AVAILABLE and self.settings.use_autoware_msgs
        
        # FPS tracking
        self._tick_count: int = 0
        self._fps_update_time: float = time.time()
        self._shutdown: bool = False
        
        # Declare parameters
        self.declare_parameter('controller_name', '')
        self.declare_parameter('control_dt', control_dt)
        
        controller_name = self.get_parameter('controller_name').get_parameter_value().string_value
        control_dt = self.get_parameter('control_dt').get_parameter_value().double_value
        
        # If no controller passed, try to load from registry
        if self.controller is None and controller_name:
            if controller_name in ControlStrategy.registry:
                self.get_logger().info(f"Loading controller from registry: {controller_name}")
                ControllerClass = ControlStrategy.registry[controller_name]
                self.controller = ControllerClass()
            else:
                self.get_logger().warn(f"Controller '{controller_name}' not found in registry")
        
        if self.use_autoware:
            # Subscribe to localization
            self.loc_sub = self.create_subscription(
                VehicleKinematicState,
                self.settings.localization_topic,
                self._on_localization,
                10
            )
            
            # Subscribe to trajectory from planner
            self.traj_sub = self.create_subscription(
                AutowareTrajectory,
                self.settings.trajectory_out_topic,
                self._on_trajectory,
                10
            )
            
            # Publisher for Autoware control command
            self.ctrl_pub = self.create_publisher(
                VehicleControlCommand,
                self.settings.control_out_topic,
                10
            )
        else:
            # Fallback: publish control as JSON string
            self.ctrl_pub = self.create_publisher(
                String,
                self.settings.control_out_topic,
                10
            )
        
        # Timer for control loop
        self.timer = self.create_timer(control_dt, self._control_tick)
        
        self.get_logger().info(f"ControllerNode initialized (autoware={self.use_autoware})")
        self.get_logger().info(f"  Rate: {1.0/control_dt:.1f} Hz")

    def _on_localization(self, msg: 'VehicleKinematicState'):
        """Update ego state from localization."""
        ego_state_from_kinematic_state(msg, self.ego_state)

    def _on_trajectory(self, msg: 'AutowareTrajectory'):
        """Receive trajectory from planner."""
        self.current_trajectory = trajectory_from_autoware(msg)
        if self.controller:
            self.controller.set_trajectory(self.current_trajectory)

    def _control_tick(self):
        """Run controller and publish command."""
        # Skip if shutting down or ROS context invalid
        if self._shutdown or not rclpy.ok():
            return
            
        if self.controller is None:
            return
        
        # Get trajectory from controller if available
        traj = self.controller.tj if hasattr(self.controller, 'tj') else self.current_trajectory
        if traj is None:
            return
        
        try:
            # Run the controller
            cmd = self.controller.control(self.ego_state, traj)
            
            if cmd:
                if self.use_autoware:
                    # Convert to Autoware message
                    msg = control_to_vehicle_cmd(cmd)
                    msg.stamp = self.get_clock().now().to_msg()
                else:
                    # Fallback: publish as JSON
                    msg = String()
                    msg.data = json.dumps({
                        'steer': cmd.steer,
                        'acceleration': cmd.acceleration,
                    })
                
                self.ctrl_pub.publish(msg)
                self.get_logger().debug(f"Published control: steer={cmd.steer:.3f}, accel={cmd.acceleration:.3f}")
                
                # Update FPS tracking
                self._tick_count += 1
                now = time.time()
                elapsed = now - self._fps_update_time
                if elapsed >= 1.0:  # Update FPS every second
                    fps = self._tick_count / elapsed
                    self._tick_count = 0
                    self._fps_update_time = now
                    
                    # Update ros_data with FPS
                    if self.ros_data:
                        with self.ros_data.lock:
                            self.ros_data.control_fps = fps
        except (rclpy.exceptions.InvalidHandle, RuntimeError):
            # Suppress errors during shutdown (invalid handle or runtime errors)
            pass
        except Exception as e:
            if not self._shutdown:
                self.get_logger().error(f"Control failed: {e}")

    def set_controller(self, controller: ControlStrategy):
        """Set or update the controller instance."""
        self.controller = controller
        self.get_logger().info(f"Controller set: {controller.__class__.__name__}")

    def destroy_node(self):
        """Clean shutdown."""
        self._shutdown = True
        if self.timer:
            self.timer.cancel()
        super().destroy_node()


def main(args=None):
    """Standalone entry point."""
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
