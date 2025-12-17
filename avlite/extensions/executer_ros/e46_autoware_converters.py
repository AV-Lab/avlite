"""
Autoware message converters for AVLite.

Converts between AVLite types and Autoware message types.

Required Autoware packages:
- autoware_auto_vehicle_msgs
- autoware_auto_planning_msgs  
- autoware_auto_control_msgs
- autoware_auto_perception_msgs
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
    from autoware_auto_planning_msgs.msg import Trajectory as AutowareTrajectory
    from autoware_auto_planning_msgs.msg import TrajectoryPoint
    from autoware_auto_control_msgs.msg import AckermannControlCommand
    from autoware_auto_vehicle_msgs.msg import VehicleKinematicState
    from autoware_auto_perception_msgs.msg import TrackedObjects, TrackedObject
    from geometry_msgs.msg import Pose, Point, Quaternion
    from std_msgs.msg import Header
    from builtin_interfaces.msg import Time
    AUTOWARE_AVAILABLE = True
except ImportError:
    log.warning("Autoware messages not found. Install autoware_auto_msgs package.")
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
    
    pose = msg.state.pose
    ego.x = pose.position.x
    ego.y = pose.position.y
    ego.theta = quaternion_to_yaw(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
    )
    ego.velocity = msg.state.longitudinal_velocity_mps
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
    
    msg = VehicleKinematicState()
    if header:
        msg.header = header
    
    qx, qy, qz, qw = euler_to_quaternion(ego.theta)
    msg.state.pose.position.x = ego.x
    msg.state.pose.position.y = ego.y
    msg.state.pose.position.z = 0.0
    msg.state.pose.orientation.x = qx
    msg.state.pose.orientation.y = qy
    msg.state.pose.orientation.z = qz
    msg.state.pose.orientation.w = qw
    msg.state.longitudinal_velocity_mps = ego.velocity
    
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
        path.append((point.pose.position.x, point.pose.position.y))
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
        point.pose.position.x = float(x)
        point.pose.position.y = float(y)
        point.pose.position.z = 0.0
        
        # Set orientation from heading if available
        if i < len(traj.path_heading):
            qx, qy, qz, qw = euler_to_quaternion(traj.path_heading[i])
            point.pose.orientation.x = qx
            point.pose.orientation.y = qy
            point.pose.orientation.z = qz
            point.pose.orientation.w = qw
        
        # Set velocity
        if i < len(traj.velocity):
            point.longitudinal_velocity_mps = float(traj.velocity[i])
        
        msg.points.append(point)
    
    return msg


# -----------------------------------------------------------------------------
# ControlCommand <-> AckermannControlCommand
# -----------------------------------------------------------------------------

def control_from_ackermann(msg) -> ControlComand:
    """
    Convert Autoware AckermannControlCommand to AVLite ControlCommand.
    
    Args:
        msg: AckermannControlCommand message
    
    Returns:
        AVLite ControlCommand
    """
    if not AUTOWARE_AVAILABLE:
        return ControlComand()
    
    return ControlComand(
        steer=msg.lateral.steering_tire_angle,
        acceleration=msg.longitudinal.acceleration
    )


def control_to_ackermann(cmd: ControlComand, header: Optional['Header'] = None) -> 'AckermannControlCommand':
    """
    Convert AVLite ControlCommand to Autoware AckermannControlCommand.
    
    Args:
        cmd: AVLite ControlCommand
        header: Optional ROS header
    
    Returns:
        AckermannControlCommand message
    """
    if not AUTOWARE_AVAILABLE:
        raise RuntimeError("Autoware messages not available")
    
    msg = AckermannControlCommand()
    if header:
        msg.stamp = header.stamp
    
    msg.lateral.steering_tire_angle = float(cmd.steer)
    msg.longitudinal.acceleration = float(cmd.acceleration)
    
    return msg


# -----------------------------------------------------------------------------
# AgentState <-> TrackedObjects
# -----------------------------------------------------------------------------

def agents_from_tracked_objects(msg) -> list[AgentState]:
    """
    Convert Autoware TrackedObjects to list of AVLite AgentState.
    
    Args:
        msg: TrackedObjects message
    
    Returns:
        List of AgentState
    """
    if not AUTOWARE_AVAILABLE:
        return []
    
    agents = []
    for obj in msg.objects:
        pose = obj.kinematics.pose_with_covariance.pose
        twist = obj.kinematics.twist_with_covariance.twist
        
        yaw = quaternion_to_yaw(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        )
        
        # Calculate velocity magnitude
        vx = twist.linear.x
        vy = twist.linear.y
        velocity = math.sqrt(vx * vx + vy * vy)
        
        agent = AgentState(
            x=pose.position.x,
            y=pose.position.y,
            theta=yaw,
            velocity=velocity,
            agent_id=hash(obj.object_id.uuid.tobytes()) % 10000  # Simple ID conversion
        )
        agents.append(agent)
    
    return agents
