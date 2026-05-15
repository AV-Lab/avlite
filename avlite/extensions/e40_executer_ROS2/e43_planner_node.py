#!/usr/bin/env python3
"""
ROS2 Planner Node - Runs AVLite local planner and publishes trajectory.

This node:
1. Subscribes to localization (ego state)
2. Runs AVLite local planner
3. Publishes trajectory as Autoware message (or std_msgs as fallback)

Launch standalone:
    ros2 run avlite planner_node --ros-args -p planner_name:=LatticePlanner -p replan_dt:=0.1
"""
import logging
import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header

from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c20_planning.c23_local_planning_strategy import LocalPlannerStrategy
from avlite.c60_common.c63_trajectory_tracker import TrajectoryTracker

from .e46_autoware_converters import (
    AUTOWARE_AVAILABLE,
    ego_state_from_kinematic_state,
    trajectory_to_autoware,
)
from avlite.c60_common.c65_fps_tracker import FpsTracker
from .settings import ExtensionSettings

log = logging.getLogger(__name__)

if AUTOWARE_AVAILABLE:
    from autoware_auto_msgs.msg import VehicleKinematicState
    from autoware_auto_msgs.msg import Trajectory as AutowareTrajectory


class PlannerNode(Node):
    """
    ROS2 node that runs AVLite local planner and publishes trajectory.
    
    Can be used in two ways:
    1. Standalone: Pass planner via ROS params (loads from registry)
    2. Embedded: Pass planner instance directly to constructor
    """

    def __init__(self, planner: LocalPlannerStrategy = None, ego_state: EgoState = None, ros_data=None, replan_dt: float = 0.1):
        super().__init__('avlite_planner')
        
        self.settings = ExtensionSettings()
        self.planner = planner
        self.ego_state = ego_state if ego_state else EgoState(x=0, y=0, theta=0)
        self.ros_data = ros_data
        # Use Autoware messages only if available AND enabled in settings
        self.use_autoware = AUTOWARE_AVAILABLE and self.settings.use_autoware_msgs
        
        # FPS tracking
        self._fps_tracker = FpsTracker()
        self._shutdown: bool = False
        
        # Declare parameters
        self.declare_parameter('planner_name', '')
        self.declare_parameter('replan_dt', replan_dt)
        
        planner_name = self.get_parameter('planner_name').get_parameter_value().string_value
        replan_dt = self.get_parameter('replan_dt').get_parameter_value().double_value
        self._node_period: float = replan_dt
        
        # If no planner passed, try to load from registry
        if self.planner is None and planner_name:
            if planner_name in LocalPlannerStrategy.registry:
                self.get_logger().info(f"Loading planner from registry: {planner_name}")
                self.get_logger().warn(f"Standalone mode requires global_plan - planner not loaded")
            else:
                self.get_logger().warn(f"Planner '{planner_name}' not found in registry")
        
        # Subscribe to localization (if Autoware available)
        if self.use_autoware:
            self.loc_sub = self.create_subscription(
                VehicleKinematicState,
                self.settings.localization_topic,
                self._on_localization,
                10
            )
            # Publisher for Autoware trajectory
            self.traj_pub = self.create_publisher(
                AutowareTrajectory,
                self.settings.trajectory_out_topic,
                10
            )
        else:
            # Fallback: publish trajectory as JSON string
            self.traj_pub = self.create_publisher(
                String,
                self.settings.trajectory_out_topic,
                10
            )
        
        # Timer for planning loop
        self.timer = self.create_timer(replan_dt, self._plan_tick)
        
        self.get_logger().info(f"PlannerNode initialized (autoware={self.use_autoware})")
        self.get_logger().info(f"  Publish: {self.settings.trajectory_out_topic}")
        self.get_logger().info(f"  Rate: {1.0/replan_dt:.1f} Hz")

    def _on_localization(self, msg: 'VehicleKinematicState'):
        """Update ego state from localization."""
        ego_state_from_kinematic_state(msg, self.ego_state)

    def _plan_tick(self):
        """Run planner and publish trajectory."""
        # Skip if shutting down or ROS context invalid
        if self._shutdown or not rclpy.ok():
            return
            
        if self.planner is None:
            self.get_logger().debug("No planner set")
            return
        
        try:
            # Only publish — replan() runs in step() to avoid blocking the ROS spin thread
            trajectory = self.planner.get_local_plan()
            
            if trajectory is None:
                self.get_logger().debug("Trajectory is None")
                return
                
            # Check if path exists and has data
            has_path = (hasattr(trajectory, 'path') and trajectory.path and len(trajectory.path) > 0)
            
            if not has_path:
                self.get_logger().debug(f"No valid path in trajectory")
                return
            
            if self.use_autoware:
                # Convert to Autoware message
                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = self.settings.map_frame
                msg = trajectory_to_autoware(trajectory, header)
            else:
                # Fallback: publish as JSON
                msg = String()
                path_list = list(trajectory.path)[:50] if hasattr(trajectory.path, '__iter__') else []
                vel_list = list(trajectory.velocity)[:50] if hasattr(trajectory.velocity, '__len__') and len(trajectory.velocity) > 0 else []
                msg.data = json.dumps({
                    'path': [(float(p[0]), float(p[1])) for p in path_list],
                    'velocity': [float(v) for v in vel_list],
                })
            
            self.traj_pub.publish(msg)
            self.get_logger().debug(f"Published trajectory with {len(trajectory.path)} points")
            
            # Update FPS tracking
            fps = self._fps_tracker.tick()
            if self.ros_data:
                with self.ros_data.lock:
                    self.ros_data.planner_fps = fps
        except (rclpy.exceptions.InvalidHandle, RuntimeError):
            # Suppress errors during shutdown (invalid handle or runtime errors)
            pass
        except Exception as e:
            if not self._shutdown:
                self.get_logger().error(f"Planning failed: {e}")

    def set_planner(self, planner: LocalPlannerStrategy):
        """Set or update the planner instance."""
        self.planner = planner
        self.get_logger().info(f"Planner set: {planner.__class__.__name__}")

    def destroy_node(self):
        """Clean shutdown."""
        self._shutdown = True
        if self.timer:
            self.timer.cancel()
        super().destroy_node()


def main(args=None):
    """Standalone entry point."""
    rclpy.init(args=args)
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
