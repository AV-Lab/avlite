#!/usr/bin/env python3
"""
ROS2 World Node - Runs world simulation step asynchronously.

This node:
1. Subscribes to control commands
2. Steps the world/simulator with the control input
3. Publishes updated ego state and perception data

This allows the world simulation to run independently as a ROS node,
with the visualizer syncing through the collector like other components.
"""
import logging
import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header

from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c40_execution.c41_world_bridge import WorldBridge
from avlite.c30_control.c31_control_model import ControlComand

from .p46_autoware_converters import (
    AUTOWARE_AVAILABLE,
    ego_state_to_kinematic_state,
    control_from_vehicle_cmd,
)
from .settings import PluginSettings

log = logging.getLogger(__name__)

if AUTOWARE_AVAILABLE:
    from autoware_auto_msgs.msg import VehicleKinematicState
    from autoware_auto_msgs.msg import VehicleControlCommand
    from autoware_auto_msgs.msg import BoundingBoxArray, BoundingBox


class WorldNode(Node):
    """
    ROS2 node that runs world simulation asynchronously.
    
    Subscribes to:
    - Control commands (to apply to world)
    
    Publishes:
    - Ego state (localization)
    - Tracked objects (perception from world)
    
    This allows the simulation to run as a standalone ROS node.
    """

    def __init__(self, world: WorldBridge = None, ros_data=None, sim_dt: float = 0.02):
        super().__init__('avlite_world')
        
        self.settings = PluginSettings()
        self.world = world
        self.ros_data = ros_data
        self.last_cmd: ControlComand = None
        # Use Autoware messages only if available AND enabled in settings
        self.use_autoware = AUTOWARE_AVAILABLE and self.settings.use_autoware_msgs
        
        # Use provided sim_dt or parameter
        self.declare_parameter('sim_dt', sim_dt)
        self.sim_dt = self.get_parameter('sim_dt').get_parameter_value().double_value
        
        # FPS tracking
        self._last_tick_time: float = 0.0
        self._tick_count: int = 0
        self._fps_update_time: float = time.time()
        self._elapsed_sim_time: float = 0.0
        self._shutdown: bool = False
        
        self._setup_subscribers()
        self._setup_publishers()
        
        # Timer for simulation loop
        self.timer = self.create_timer(self.sim_dt, self._sim_tick)
        
        self.get_logger().info(f"WorldNode initialized (autoware={self.use_autoware})")
        self.get_logger().info(f"  Sim rate: {1.0/self.sim_dt:.1f} Hz")
        self.get_logger().info(f"  Subscribe control: {self.settings.control_out_topic}")
        self.get_logger().info(f"  Publish ego: {self.settings.localization_topic}")
        self.get_logger().info(f"  Publish perception: {self.settings.perception_topic}")

    def _setup_subscribers(self):
        """Setup control command subscriber."""
        if self.use_autoware:
            self.ctrl_sub = self.create_subscription(
                VehicleControlCommand,
                self.settings.control_out_topic,  # Subscribe to AVLite controller output
                self._on_control_autoware,
                10
            )
        else:
            self.ctrl_sub = self.create_subscription(
                String,
                self.settings.control_out_topic,
                self._on_control_json,
                10
            )

    def _setup_publishers(self):
        """Setup ego state and perception publishers."""
        if self.use_autoware:
            self.ego_pub = self.create_publisher(
                VehicleKinematicState,
                self.settings.localization_topic,
                10
            )
            self.perception_pub = self.create_publisher(
                BoundingBoxArray,
                self.settings.perception_topic,
                10
            )
        else:
            self.ego_pub = self.create_publisher(
                String,
                self.settings.localization_topic,
                10
            )
            self.perception_pub = self.create_publisher(
                String,
                self.settings.perception_topic,
                10
            )

    def _on_control_autoware(self, msg: 'VehicleControlCommand'):
        """Handle Autoware control command."""
        self.last_cmd = control_from_vehicle_cmd(msg)

    def _on_control_json(self, msg: String):
        """Handle JSON control command."""
        try:
            data = json.loads(msg.data)
            self.last_cmd = ControlComand(
                steer=data.get('steer', 0),
                acceleration=data.get('acceleration', 0)
            )
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Invalid JSON control: {e}")

    def _sim_tick(self):
        """Run one simulation step and publish results."""
        # Skip if shutting down or ROS context invalid
        if self._shutdown or not rclpy.ok():
            return
            
        if self.world is None:
            return
        
        # Apply control and step the world
        if self.last_cmd:
            self.world.control_ego_state(self.last_cmd, dt=self.sim_dt)
        
        # Track elapsed simulation time
        self._elapsed_sim_time += self.sim_dt
        
        # Update FPS tracking
        self._tick_count += 1
        now = time.time()
        elapsed = now - self._fps_update_time
        if elapsed >= 1.0:  # Update FPS every second
            fps = self._tick_count / elapsed
            self._tick_count = 0
            self._fps_update_time = now
            
            # Update ros_data with FPS and sim time
            if self.ros_data:
                with self.ros_data.lock:
                    self.ros_data.sim_fps = fps
                    self.ros_data.elapsed_sim_time = self._elapsed_sim_time
        
        # Get updated state
        ego_state = self.world.get_ego_state()
        
        # Publish ego state
        self._publish_ego_state(ego_state)
        
        # Publish perception (other agents in world)
        self._publish_perception()

    def _publish_ego_state(self, ego_state: EgoState):
        """Publish ego vehicle state."""
        if ego_state is None:
            return
        
        try:
            if self.use_autoware:
                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = self.settings.map_frame
                msg = ego_state_to_kinematic_state(ego_state, header)
            else:
                msg = String()
                msg.data = json.dumps({
                    'x': float(ego_state.x),
                    'y': float(ego_state.y),
                    'theta': float(ego_state.theta),
                    'velocity': float(ego_state.velocity),
                })
            
            self.ego_pub.publish(msg)
        except (rclpy.exceptions.InvalidHandle, RuntimeError):
            # Suppress errors during shutdown (invalid handle or runtime errors)
            pass
        except Exception as e:
            if not self._shutdown:
                self.get_logger().error(f"Failed to publish ego state: {e}")

    def _publish_perception(self):
        """Publish perception data from world."""
        try:
            # Get ground truth perception from world
            pm = self.world.get_ground_truth_perception_model()
            if pm is None:
                return
            
            agents = getattr(pm, 'agent_vehicles', [])
            if not agents:
                return
            
            if self.use_autoware:
                msg = BoundingBoxArray()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.settings.map_frame
                
                for agent in agents:
                    box = BoundingBox()
                    box.centroid.x = float(agent.x)
                    box.centroid.y = float(agent.y)
                    box.centroid.z = 0.0
                    box.size.x = 4.5  # length
                    box.size.y = 2.0  # width
                    box.size.z = 1.5  # height
                    box.heading = float(agent.theta)
                    box.velocity = float(agent.velocity)
                    box.vehicle_label = 1  # car
                    msg.boxes.append(box)
            else:
                msg = String()
                objects_list = []
                for agent in agents:
                    objects_list.append({
                        'id': getattr(agent, 'agent_id', 0),
                        'x': float(agent.x),
                        'y': float(agent.y),
                        'theta': float(agent.theta),
                        'velocity': float(agent.velocity),
                    })
                msg.data = json.dumps({'objects': objects_list})
            
            self.perception_pub.publish(msg)
        except (rclpy.exceptions.InvalidHandle, RuntimeError):
            # Suppress errors during shutdown (invalid handle or runtime errors)
            pass
        except Exception as e:
            if not self._shutdown:
                self.get_logger().error(f"Failed to publish perception: {e}")

    def set_world(self, world: WorldBridge):
        """Set or update the world bridge."""
        self.world = world
        self.get_logger().info(f"World set: {world.__class__.__name__}")

    def destroy_node(self):
        """Clean shutdown."""
        self._shutdown = True
        if self.timer:
            self.timer.cancel()
        super().destroy_node()


def main(args=None):
    """Standalone entry point."""
    rclpy.init(args=args)
    node = WorldNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
