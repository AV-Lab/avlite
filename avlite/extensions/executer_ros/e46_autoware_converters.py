"""
Autoware message converters for AVLite.

Converts between AVLite types and Autoware message types.

Required Autoware packages:
- ros-humble-autoware-auto-msgs (apt install ros-humble-autoware-auto-msgs)
"""
import logging
import math
from typing import Optional

import numpy as np

from avlite.c10_perception.c11_perception_model import EgoState, AgentState
from avlite.c20_planning.c28_trajectory import Trajectory
from avlite.c30_control.c31_control_model import ControlComand

log = logging.getLogger(__name__)

# Try importing Autoware messages - these may not be available
try:
    from autoware_auto_msgs.msg import Trajectory as AutowareTrajectory
    from autoware_auto_msgs.msg import TrajectoryPoint
    from autoware_auto_msgs.msg import VehicleControlCommand
    from autoware_auto_msgs.msg import VehicleKinematicState
    from geometry_msgs.msg import Pose, Point, Quaternion
    from std_msgs.msg import Header
    from builtin_interfaces.msg import Time
    AUTOWARE_AVAILABLE = True
except ImportError:
    log.warning("Autoware messages not found. Install: sudo apt install ros-humble-autoware-auto-msgs")
    AUTOWARE_AVAILABLE = False


def euler_to_quaternion(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> tuple[float, float, float, float]:
    """Convert Euler angles to quaternion (x, y, z, w)."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return (x, y, z, w)


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Extract yaw angle from quaternion."""
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


# -----------------------------------------------------------------------------
# EgoState <-> VehicleKinematicState
# -----------------------------------------------------------------------------

def ego_state_from_kinematic_state(msg, ego: EgoState) -> EgoState:
    """
    Update EgoState from Autoware VehicleKinematicState message.
    
    Args:
        msg: VehicleKinematicState message
        ego: EgoState to update (modified in place)
    
    Returns:
        Updated EgoState
    """
    if not AUTOWARE_AVAILABLE:
        return ego
    
    # VehicleKinematicState.state is a TrajectoryPoint with x, y, heading (Complex32)
    state = msg.state
    ego.x = state.x
    ego.y = state.y
    # heading is Complex32 with real and imag (cos and sin of heading)
    ego.theta = math.atan2(state.heading.imag, state.heading.real)
    ego.velocity = state.longitudinal_velocity_mps
    return ego


def ego_state_to_kinematic_state(ego: EgoState, header: Optional['Header'] = None) -> 'VehicleKinematicState':
    """
    Convert EgoState to Autoware VehicleKinematicState message.
    
    Args:
        ego: EgoState to convert
        header: Optional ROS header with timestamp and frame_id
    
    Returns:
        VehicleKinematicState message
    """
    if not AUTOWARE_AVAILABLE:
        raise RuntimeError("Autoware messages not available")
    
    from autoware_auto_msgs.msg import Complex32
    
    msg = VehicleKinematicState()
    if header:
        msg.header = header
    
    msg.state.x = float(ego.x)
    msg.state.y = float(ego.y)
    # heading as Complex32 (cos + i*sin)
    msg.state.heading.real = float(math.cos(ego.theta))
    msg.state.heading.imag = float(math.sin(ego.theta))
    msg.state.longitudinal_velocity_mps = float(ego.velocity)
    
    return msg


# -----------------------------------------------------------------------------
# Trajectory <-> Autoware Trajectory
# -----------------------------------------------------------------------------

def trajectory_from_autoware(msg) -> Trajectory:
    """
    Convert Autoware Trajectory message to AVLite Trajectory.
    
    Args:
        msg: Autoware Trajectory message
    
    Returns:
        AVLite Trajectory
    """
    if not AUTOWARE_AVAILABLE:
        return Trajectory()
    
    path = []
    velocity = []
    
    for point in msg.points:
        # TrajectoryPoint has x, y directly (not pose.position)
        path.append((point.x, point.y))
        velocity.append(point.longitudinal_velocity_mps)
    
    return Trajectory(path=path, velocity=velocity)


def trajectory_to_autoware(traj: Trajectory, header: Optional['Header'] = None) -> 'AutowareTrajectory':
    """
    Convert AVLite Trajectory to Autoware Trajectory message.
    
    Args:
        traj: AVLite Trajectory
        header: Optional ROS header
    
    Returns:
        Autoware Trajectory message
    """
    if not AUTOWARE_AVAILABLE:
        raise RuntimeError("Autoware messages not available")
    
    msg = AutowareTrajectory()
    if header:
        msg.header = header
    
    for i, (x, y) in enumerate(traj.path):
        point = TrajectoryPoint()
        point.x = float(x)
        point.y = float(y)
        
        # Set heading from path_heading if available (as Complex32)
        if i < len(traj.path_heading):
            heading = traj.path_heading[i]
            point.heading.real = float(math.cos(heading))
            point.heading.imag = float(math.sin(heading))
        
        # Set velocity
        if hasattr(traj, 'velocity') and hasattr(traj.velocity, '__len__') and i < len(traj.velocity):
            point.longitudinal_velocity_mps = float(traj.velocity[i])
        
        msg.points.append(point)
    
    return msg
    
    return msg


# -----------------------------------------------------------------------------
# ControlCommand <-> VehicleControlCommand
# -----------------------------------------------------------------------------

def control_from_vehicle_cmd(msg) -> ControlComand:
    """
    Convert Autoware VehicleControlCommand to AVLite ControlCommand.
    
    Args:
        msg: VehicleControlCommand message
    
    Returns:
        AVLite ControlCommand
    """
    if not AUTOWARE_AVAILABLE:
        return ControlComand()
    
    return ControlComand(
        steer=msg.front_wheel_angle_rad,
        acceleration=msg.long_accel_mps2
    )


def control_to_vehicle_cmd(cmd: ControlComand, header: Optional['Header'] = None) -> 'VehicleControlCommand':
    """
    Convert AVLite ControlCommand to Autoware VehicleControlCommand.
    
    Args:
        cmd: AVLite ControlCommand
        header: Optional ROS header
    
    Returns:
        VehicleControlCommand message
    """
    if not AUTOWARE_AVAILABLE:
        raise RuntimeError("Autoware messages not available")
    
    msg = VehicleControlCommand()
    if header:
        msg.stamp = header.stamp
    
    msg.front_wheel_angle_rad = float(cmd.steer)
    msg.long_accel_mps2 = float(cmd.acceleration)
    
    return msg


# Aliases for backward compatibility
control_from_ackermann = control_from_vehicle_cmd
control_to_ackermann = control_to_vehicle_cmd


# -----------------------------------------------------------------------------
# AgentState <-> BoundingBoxArray (perception)
# Note: autoware_auto_msgs uses BoundingBoxArray, not TrackedObjects
# -----------------------------------------------------------------------------

def agents_from_bounding_boxes(msg) -> list[AgentState]:
    """
    Convert Autoware BoundingBoxArray to list of AVLite AgentState.
    
    Args:
        msg: BoundingBoxArray message
    
    Returns:
        List of AgentState
    """
    if not AUTOWARE_AVAILABLE:
        return []
    
    agents = []
    for i, box in enumerate(msg.boxes):
        agent = AgentState(
            x=box.centroid.x,
            y=box.centroid.y,
            theta=box.heading,
            velocity=box.velocity,
            agent_id=i
        )
        agents.append(agent)
    
    return agents


# Alias for backward compatibility (in case code uses old function name)
agents_from_tracked_objects = agents_from_bounding_boxes
