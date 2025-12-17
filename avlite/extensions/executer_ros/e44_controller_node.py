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
    control_to_ackermann,
)
from .settings import ExtensionSettings

log = logging.getLogger(__name__)

if AUTOWARE_AVAILABLE:
    from autoware_auto_vehicle_msgs.msg import VehicleKinematicState
    from autoware_auto_planning_msgs.msg import Trajectory as AutowareTrajectory
    from autoware_auto_control_msgs.msg import AckermannControlCommand


class ControllerNode(Node):
    """
    ROS2 node that runs AVLite controller and publishes control commands.
    
    Can be used in two ways:
    1. Standalone: Pass controller via ROS params (loads from registry)
    2. Embedded: Pass controller instance directly to constructor
    """

    def __init__(self, controller: ControlStrategy = None, ego_state: EgoState = None):
        super().__init__('avlite_controller')
        
        self.settings = ExtensionSettings()
        self.controller = controller
        self.ego_state = ego_state if ego_state else EgoState(x=0, y=0, theta=0)
        self.current_trajectory: Trajectory = None
        self.use_autoware = AUTOWARE_AVAILABLE
        
        # Declare parameters
        self.declare_parameter('controller_name', '')
        self.declare_parameter('control_dt', 0.02)
        
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
                AckermannControlCommand,
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
        self.get_logger().info(f"  Publish: {self.settings.control_out_topic}")
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
                    msg = control_to_ackermann(cmd)
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
        except Exception as e:
            self.get_logger().error(f"Control failed: {e}")

    def set_controller(self, controller: ControlStrategy):
        """Set or update the controller instance."""
        self.controller = controller
        self.get_logger().info(f"Controller set: {controller.__class__.__name__}")


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
