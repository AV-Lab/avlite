"""
ROS Executor extension for AVLite with Autoware message support.

This extension provides:
- ROSExecuter: Executor that syncs with external ROS planner/controller
- PerceptionNode: ROS node that publishes AVLite ego state and tracked objects
- PlannerNode: ROS node that runs AVLite planner, publishes Autoware Trajectory
- ControllerNode: ROS node that runs AVLite controller, publishes Autoware ControlCommand
- WorldNode: ROS node that runs world simulation asynchronously
- ProxyLocalPlanner: Placeholder planner that stores ROS-received trajectories
- ProxyController: Placeholder controller that stores ROS-received commands
- Autoware message converters for EgoState, Trajectory, ControlCommand

Usage:
    1. Configure topics in ExtensionSettings (or ext_ros_executer.yaml)
    2. Launch PlannerNode, ControllerNode, WorldNode (or external Autoware nodes)
    3. Use ROSExecuter with ProxyLocalPlanner and ProxyController
    4. Collector subscribes to topics, syncs to AVLite, visualizer displays
"""

from .e41_ros_launcher import ROSExecuter
from .e42_perception_node import PerceptionNode
from .e43_planner_node import PlannerNode
from .e44_controller_node import ControllerNode
from .e45_world_node import WorldNode
from .e47_proxy_strategies import ProxyLocalPlanner, ProxyController
from .settings import ExtensionSettings

__all__ = [
    "ROSExecuter",
    "PerceptionNode",
    "PlannerNode",
    "ControllerNode",
    "WorldNode",
    "ProxyLocalPlanner", 
    "ProxyController",
    "ExtensionSettings",
]