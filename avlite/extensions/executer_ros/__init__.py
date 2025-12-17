"""
ROS Executor extension for AVLite with Autoware message support.

This extension provides:
- ROSExecuter: Executor that syncs with external ROS planner/controller
- PlannerNode: ROS node that runs AVLite planner, publishes Autoware Trajectory
- ControllerNode: ROS node that runs AVLite controller, publishes Autoware ControlCommand
- ProxyLocalPlanner: Placeholder planner that stores ROS-received trajectories
- ProxyController: Placeholder controller that stores ROS-received commands
- Autoware message converters for EgoState, Trajectory, ControlCommand

Usage:
    1. Configure topics in ExtensionSettings (or ext_ros_executer.yaml)
    2. Launch PlannerNode and ControllerNode (or external Autoware nodes)
    3. Use ROSExecuter with ProxyLocalPlanner and ProxyController
    4. Collector subscribes to topics, syncs to AVLite, visualizer displays
"""

from .e41_ros_launcher import ROSExecuter
from .e43_planner_node import PlannerNode
from .e44_controller_node import ControllerNode
from .e47_proxy_strategies import ProxyLocalPlanner, ProxyController
from .settings import ExtensionSettings

__all__ = [
    "ROSExecuter",
    "PlannerNode",
    "ControllerNode",
    "ProxyLocalPlanner", 
    "ProxyController",
    "ExtensionSettings",
]