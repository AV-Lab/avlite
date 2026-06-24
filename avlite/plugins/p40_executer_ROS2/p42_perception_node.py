#!/usr/bin/env python3
"""
ROS2 Perception Node - Publishes ego state and tracked objects from AVLite perception.

This node:
1. Gets ego state from world bridge
2. Gets tracked agents from perception model
3. Publishes as Autoware messages (or std_msgs as fallback)

Topics Published:
- /localization/kinematic_state (VehicleKinematicState or String)
- /perception/object_recognition/tracking/objects (TrackedObjects or String)
"""
import logging
import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header

from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c40_execution.c41_world_bridge import WorldBridge

from .p46_autoware_converters import (
    AUTOWARE_AVAILABLE,
    ego_state_to_kinematic_state,
    euler_to_quaternion,
)
from avlite.c60_common.c65_fps_tracker import FpsTracker
from .settings import PluginSettings

log = logging.getLogger(__name__)

if AUTOWARE_AVAILABLE:
    from autoware_auto_msgs.msg import VehicleKinematicState
    from autoware_auto_msgs.msg import BoundingBoxArray, BoundingBox


class PerceptionNode(Node):
    """
    ROS2 node that publishes AVLite perception data.
    
    Publishes:
    - Ego vehicle state (localization)
    - Tracked objects (perception)
    """

    def __init__(
        self,
        ego_state: EgoState = None,
        perception_model: PerceptionModel = None,
        world: WorldBridge = None,
        ros_data=None,
        perception_dt: float = 0.1
    ):
        super().__init__('avlite_perception')
        
        self.settings = PluginSettings()
        self.ego_state = ego_state
        self.pm = perception_model
        self.world = world
        self.ros_data = ros_data
        # Use Autoware messages only if available AND enabled in settings
        self.use_autoware = AUTOWARE_AVAILABLE and self.settings.use_autoware_msgs
        
        # Use provided perception_dt or parameter
        self.declare_parameter('perception_dt', perception_dt)
        perception_dt = self.get_parameter('perception_dt').get_parameter_value().double_value
        
        # FPS tracking
        self._fps_tracker = FpsTracker()
        self._node_period: float = perception_dt
        self._shutdown: bool = False
        
        if self.use_autoware:
            # Publisher for ego state (localization)
            self.ego_pub = self.create_publisher(
                VehicleKinematicState,
                self.settings.localization_topic,
                10
            )
            
            # Publisher for bounding boxes (perception)
            self.objects_pub = self.create_publisher(
                BoundingBoxArray,
                self.settings.perception_topic,
                10
            )
        else:
            # Fallback: publish as JSON strings
            self.ego_pub = self.create_publisher(
                String,
                self.settings.localization_topic,
                10
            )
            self.objects_pub = self.create_publisher(
                String,
                self.settings.perception_topic,
                10
            )
        
        # Timer for perception publishing loop
        self.timer = self.create_timer(perception_dt, self._perception_tick)
        
        self.get_logger().info(f"PerceptionNode initialized (autoware={self.use_autoware})")
        self.get_logger().info(f"  Publish ego: {self.settings.localization_topic}")
        self.get_logger().info(f"  Publish objects: {self.settings.perception_topic}")
        self.get_logger().info(f"  Rate: {1.0/perception_dt:.1f} Hz")

    def _perception_tick(self):
        """Publish perception data."""
        # Skip if shutting down or ROS context invalid
        if self._shutdown or not rclpy.ok():
            return
            
        self._publish_ego_state()
        self._publish_tracked_objects()
        self._sync_prediction_data()
        
        # Update FPS tracking
        fps = self._fps_tracker.tick()
        if self.ros_data:
            with self.ros_data.lock:
                self.ros_data.perception_fps = fps
    
    def _sync_prediction_data(self):
        """Sync prediction/heatmap data from perception model to ros_data."""
        if self.pm is None or self.ros_data is None:
            return
        
        try:
            with self.ros_data.lock:
                # Sync occupancy flow (heatmap)
                if hasattr(self.pm, 'occupancy_flow') and self.pm.occupancy_flow is not None:
                    self.ros_data.occupancy_flow = self.pm.occupancy_flow
                
                # Sync grid bounds
                if hasattr(self.pm, 'grid_bounds') and self.pm.grid_bounds is not None:
                    self.ros_data.grid_bounds = self.pm.grid_bounds
                
                # Sync prediction mode
                if hasattr(self.pm, 'prediction_mode') and self.pm.prediction_mode is not None:
                    self.ros_data.prediction_mode = self.pm.prediction_mode.value
                
                # Sync predict delta t
                if hasattr(self.pm, 'predict_delta_t'):
                    self.ros_data.predict_delta_t = self.pm.predict_delta_t
                
                # Sync trajectory predictions
                if hasattr(self.pm, 'trajectories') and self.pm.trajectories is not None:
                    self.ros_data.trajectories = self.pm.trajectories
                
                # Sync per-object occupancy flow
                if hasattr(self.pm, 'occupancy_flow_per_object') and self.pm.occupancy_flow_per_object is not None:
                    self.ros_data.occupancy_flow_per_object = self.pm.occupancy_flow_per_object
        except Exception as e:
            if not self._shutdown:
                self.get_logger().debug(f"Failed to sync prediction data: {e}")

    def _publish_ego_state(self):
        """Publish ego vehicle state."""
        # Get ego state from world if available
        if self.world:
            self.ego_state = self.world.get_ego_state()
        
        if self.ego_state is None:
            return
        
        try:
            if self.use_autoware:
                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = self.settings.map_frame
                msg = ego_state_to_kinematic_state(self.ego_state, header)
            else:
                msg = String()
                msg.data = json.dumps({
                    'x': float(self.ego_state.x),
                    'y': float(self.ego_state.y),
                    'theta': float(self.ego_state.theta),
                    'velocity': float(self.ego_state.velocity),
                })
            
            self.ego_pub.publish(msg)
        except (rclpy.exceptions.InvalidHandle, RuntimeError):
            # Suppress errors during shutdown (invalid handle or runtime errors)
            pass
        except Exception as e:
            if not self._shutdown:
                self.get_logger().error(f"Failed to publish ego state: {e}")

    def _publish_tracked_objects(self):
        """Publish tracked objects from perception model as bounding boxes."""
        if self.pm is None:
            return
        
        agents = getattr(self.pm, 'agent_vehicles', [])
        if not agents:
            return
        
        try:
            if self.use_autoware:
                msg = BoundingBoxArray()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.settings.map_frame
                
                for agent in agents:
                    box = BoundingBox()
                    
                    # Set centroid position
                    box.centroid.x = float(agent.x)
                    box.centroid.y = float(agent.y)
                    box.centroid.z = 0.0
                    
                    # Set size (default vehicle size)
                    box.size.x = 4.5  # length
                    box.size.y = 2.0  # width
                    box.size.z = 1.5  # height
                    
                    # Set heading and velocity
                    box.heading = float(agent.theta)
                    box.velocity = float(agent.velocity)
                    
                    # Set vehicle label (1 = car)
                    box.vehicle_label = 1
                    
                    msg.boxes.append(box)
            else:
                # Fallback: publish as JSON
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
            
            self.objects_pub.publish(msg)
        except (rclpy.exceptions.InvalidHandle, RuntimeError):
            # Suppress errors during shutdown (invalid handle or runtime errors)
            pass
        except Exception as e:
            if not self._shutdown:
                self.get_logger().error(f"Failed to publish tracked objects: {e}")

    def set_perception_model(self, pm: PerceptionModel):
        """Set or update the perception model."""
        self.pm = pm

    def set_world(self, world: WorldBridge):
        """Set or update the world bridge."""
        self.world = world

    def destroy_node(self):
        """Clean shutdown."""
        self._shutdown = True
        if self.timer:
            self.timer.cancel()
        super().destroy_node()


def main(args=None):
    """Standalone entry point."""
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
