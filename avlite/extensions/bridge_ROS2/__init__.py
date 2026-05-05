"""
ROS2 World Bridge plugin for AVLite.

Provides ROS2WorldBridge: a WorldBridge implementation that connects AVLite
to an externally-running ROS stack (e.g. Autoware). It:

- Subscribes to /localization/kinematic_state  → updates ego state
- Subscribes to perception tracking topic      → updates perception model
- Publishes to /control/command/control_cmd    → forwards AVLite control output

Use in execution settings::

    bridge = "ROS2WorldBridge"
"""

from .ros2_world_bridge import ROS2WorldBridge
from .settings import ExtensionSettings

__all__ = ["ROS2WorldBridge", "ExtensionSettings"]
