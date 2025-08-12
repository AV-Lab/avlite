import subprocess
import json
import logging
import math
from typing import Optional
from c40_execution.c42_sync_executer import WorldBridge
from c10_perception.c11_perception_model import EgoState, AgentState
from c30_control.c32_control_strategy import ControlComand
import numpy as np


log = logging.getLogger(__name__)

try: 
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
except ImportError:
    log.warning("ROS2 Python packages not found. Make sure ROS2 is installed and sourced.")


class GazeboIgnitionBridge(WorldBridge, Node):
    def __init__(self, ego_state: Optional[EgoState], model_name: str = "gen0_model", world_name: str = "default"):
        """
        Initialize Gazebo Ignition Bridge
        
        Args:
            ego_state: The ego vehicle state
            model_name: Name of the model in Gazebo (default: "gen0_model")
            world_name: Name of the world in Gazebo (default: "default")
        """
        ## ROS2 stuff
        if not rclpy.ok():
            rclpy.init()  # Only initialize if not already done
        Node.__init__(self, 'gazebo_ignition_bridge')
        self.cmd_vel_publisher = self.create_publisher(Twist, '/control/cmd_vel', 10)

         # Vehicle parameters based on Gen0
        self.wheel_base = 2.8
        self.wheel_track = 1.385
        self.wheel_radius = 0.33
        self.max_velocity_forward = 5.6
        self.min_velocity_forward = 0.3
        self.max_steering = 0.31  # rad
        self.min_steering = -0.31
        self.min_turning_radius = self.wheel_base / (2 * math.tan(self.max_steering))
        self.max_linear_velocity = self.max_velocity_forward  # 5.6 m/s
        self.max_angular_velocity = self.max_velocity_forward / self.min_turning_radius  # Calculate max angular velocity

        self.ego_state = ego_state
        self.model_name = model_name
        self.world_name = world_name
        
        # Test connection to Gazebo
        if not self._test_gazebo_connection():
            raise ConnectionError("Cannot connect to Gazebo Ignition. Make sure it's running.")
        print(f"Connected to Gazebo Ignition with model: {model_name}")

    def _test_gazebo_connection(self) -> bool:
        """Test if Gazebo Ignition is running and accessible"""
        try:
            result = subprocess.run(
                ["ign", "service", "-l"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _run_ign_command(self, command: list, timeout: int = 5) -> tuple[bool, str]:
        """
        Run an ignition command and return success status and output
        
        Args:
            command: List of command arguments
            timeout: Command timeout in seconds
            
        Returns:
            Tuple of (success, output_message)
        """
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                log.debug(f"Command succeeded: {' '.join(command)}")
                return True, result.stdout
            else:
                log.error(f"Command failed: {' '.join(command)}")
                log.error(f"Error output: {result.stderr}")
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            log.error(f"Command timed out: {' '.join(command)}")
            return False, "Command timed out"
        except FileNotFoundError:
            log.error("Ignition command not found. Make sure Gazebo Ignition is installed.")
            return False, "Ignition command not found"
        except Exception as e:
            log.error(f"Unexpected error running command: {e}")
            return False, str(e)

    def _euler_to_quaternion(self, yaw: float, pitch: float = 0.0, roll: float = 0.0) -> dict:
        """
        Convert Euler angles to quaternion
        
        Args:
            yaw: Yaw angle in radians
            pitch: Pitch angle in radians (default: 0.0)
            roll: Roll angle in radians (default: 0.0)
            
        Returns:
            Dictionary with quaternion components {x, y, z, w}
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        return {
            "x": sr * cp * cy - cr * sp * sy,
            "y": cr * sp * cy + sr * cp * sy,
            "z": cr * cp * sy - sr * sp * cy,
            "w": cr * cp * cy + sr * sp * sy
        }

    def teleport_ego(self, x: float, y: float, theta: Optional[float] = None):
        """
        Teleport the ego vehicle to a new position
        
        Args:
            x: X coordinate in meters
            y: Y coordinate in meters
            theta: Orientation in radians (optional)
            z: Z coordinate in meters (default: 0.0)
        """
        # Update ego state
        if self.ego_state:
            self.ego_state.x = x
            self.ego_state.y = y
            if theta is not None:
                self.ego_state.theta = theta

        # Convert theta to quaternion if provided
        if theta is not None:
            quat = self._euler_to_quaternion(theta)
        else:
            quat = {"x": 0.0, "y": 0.0, "z": 5.0, "w": 1.0}  # No rotation

        # Construct the pose request message
        pose_msg = (
            f'name: "{self.model_name}", '
            f'position: {{x: {x}, y: {y}, z: {5.0}}}, '
            f'orientation: {{x: {quat["x"]}, y: {quat["y"]}, z: {quat["z"]}, w: {quat["w"]}}}'
        )

        # Build the ignition service command
        command = [
            "ign", "service",
            "-s", f"/world/{self.world_name}/set_pose",
            "--reqtype", "ignition.msgs.Pose",
            "--reptype", "ignition.msgs.Boolean",
            "--timeout", "1000",
            "--req", pose_msg
        ]

        # Execute the command
        success, output = self._run_ign_command(command)
        
        if success:
            log.info(f"Successfully teleported {self.model_name} to ({x}, {y}, {5.0}) with rotation {theta}")
        else:
            log.error(f"Failed to teleport {self.model_name}: {output}")
            
        return success

    def get_model_pose(self) -> Optional[dict]:
        """
        Get the current pose of the model (main model only, not sub-components)
        
        Returns:
            Dictionary with pose information or None if failed
        """
        # Subscribe to pose topic and get current pose
        command = [
            "ign", "topic",
            "-t", f"/model/{self.model_name}/pose",
            "-e", "-n", "1"  # Echo once
        ]
        
        success, output = self._run_ign_command(command, timeout=10)
        
        if success and output.strip():
            try:
                return self._parse_main_model_pose(output)
            except Exception as e:
                log.error(f"Failed to parse pose data: {e}")
                return None
        else:
            log.error(f"Failed to get model pose: {output}")
            return None

    def _parse_main_model_pose(self, output: str) -> Optional[dict]:
        """
        Parse the pose output and extract only the main model pose
        
        Args:
            output: Raw output from ign topic command
            
        Returns:
            Dictionary with position and orientation, or None if not found
        """
        lines = output.strip().split('\n')
        
        # Look for the main model pose block (name: "gen0_model" without :: prefix)
        in_main_model_block = False
        in_position_block = False
        in_orientation_block = False
        
        pose_data = {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
        }
        
        for line in lines:
            line = line.strip()
            
            # Check if we're in the main model block (exact match for model name)
            if f'name: "{self.model_name}"' in line and '::' not in line:
                in_main_model_block = True
                continue
            
            # If we hit another pose block, we're done with the main model
            if line.startswith('pose {') and in_main_model_block:
                break
                
            if in_main_model_block:
                # Parse position block
                if line == 'position {':
                    in_position_block = True
                    in_orientation_block = False
                elif line == 'orientation {':
                    in_orientation_block = True
                    in_position_block = False
                elif line == '}':
                    in_position_block = False
                    in_orientation_block = False
                elif in_position_block:
                    if line.startswith('x: '):
                        pose_data['position']['x'] = float(line.split(': ')[1])
                    elif line.startswith('y: '):
                        pose_data['position']['y'] = float(line.split(': ')[1])
                    elif line.startswith('z: '):
                        pose_data['position']['z'] = float(line.split(': ')[1])
                elif in_orientation_block:
                    if line.startswith('x: '):
                        pose_data['orientation']['x'] = float(line.split(': ')[1])
                    elif line.startswith('y: '):
                        pose_data['orientation']['y'] = float(line.split(': ')[1])
                    elif line.startswith('z: '):
                        pose_data['orientation']['z'] = float(line.split(': ')[1])
                    elif line.startswith('w: '):
                        pose_data['orientation']['w'] = float(line.split(': ')[1])
        
        if in_main_model_block:
            return pose_data
        else:
            log.warning(f"Could not find main model pose for {self.model_name}")
            return None

    def teleport_ego_relative(self, dx: float, dy: float, dtheta: float = 0.0):
        """
        Teleport the ego vehicle relative to its current position
        
        Args:
            dx: Change in X coordinate
            dy: Change in Y coordinate
            dtheta: Change in orientation (radians)
        """
        if not self.ego_state:
            log.error("Cannot perform relative teleport: ego_state is None")
            return False
            
        new_x = self.ego_state.x + dx
        new_y = self.ego_state.y + dy
        new_theta = self.ego_state.theta + dtheta
        
        return self.teleport_ego(new_x, new_y, new_theta)

    def get_ego_state(self) -> Optional[EgoState]:
        """
        Get the current state of the ego vehicle from Gazebo
        
        Returns:
            Updated EgoState or None if failed
        """
        pose_data = self.get_model_pose()
        
        if pose_data and self.ego_state:
            # Update ego state with current pose
            self.ego_state.x = pose_data['position']['x']
            self.ego_state.y = pose_data['position']['y']
            
            # Convert quaternion to yaw angle (theta)
            quat = pose_data['orientation']
            # Simple yaw extraction from quaternion (assuming vehicle moves in XY plane)
            self.ego_state.theta = math.atan2(
                2.0 * (quat['w'] * quat['z'] + quat['x'] * quat['y']),
                1.0 - 2.0 * (quat['y'] * quat['y'] + quat['z'] * quat['z'])
            )
            
            log.debug(f"Updated Ego State: x={self.ego_state.x}, y={self.ego_state.y}, theta={self.ego_state.theta}")
            return self.ego_state
        else:
            log.error("Failed to get ego state from Gazebo")
            return None
        
    def spawn_agent(self, agent_state: AgentState):
        pass
    
    def control_ego_state(self, cmd: ControlComand, dt=0.01):
        """Update the ego state with the given command.
        This method applies control commands to the vehicle and updates the state.
        If the vehicle doesn't exist yet, it will be spawned.
        """

        log.debug(f"Applying control: {cmd}")
        assert self.ego_state is not None, "Ego state is None. Cannot update state without a reference."

        current_velocity = self.ego_state.velocity

        # Calculate throttle and brake values
        throttle = np.abs(cmd.acceleration) / self.ego_state.max_acceleration if cmd.acceleration > 0 else 0.0
        brake = np.abs(cmd.acceleration) / self.ego_state.min_acceleration if cmd.acceleration < 0 else 0.0

        # Convert to float to ensure correct type
        throttle = float(throttle)
        brake = float(brake)
        steer = float(cmd.steer)

        # Determine reverse state
        is_nearly_stopped = current_velocity < 0.1  # threshold for "stopped"
        wants_reverse = cmd.acceleration < 0
        is_reverse = wants_reverse and is_nearly_stopped

        # In reverse mode, use throttle instead of brake for backward movement
        if is_reverse and wants_reverse:
            throttle = float(np.abs(cmd.acceleration) / self.ego_state.max_acceleration)
            target_linear_velocity = -self.max_velocity_forward * throttle  # Reverse velocity
            brake = 0.0
        else:
            throttle = float(np.abs(cmd.acceleration) / self.ego_state.max_acceleration)
            target_linear_velocity = self.max_velocity_forward * throttle

        # When steering with zero throttle, maintain a small throttle to prevent stopping
        if throttle == 0.0 and brake == 0.0 and abs(cmd.steer) > 0.01:
            throttle = 0.05  # Small throttle value to maintain momentum during steering

        log.debug(f"Velocity: {current_velocity}, Throttle: {throttle}, Brake: {brake}, Reverse: {is_reverse}")
        
        # Convert steering to angular velocity
        # Using bicycle model: angular_velocity = (velocity * tan(steering_angle)) / wheelbase
        # For small angles: tan(angle) ≈ angle, and we assume steer is normalized [-1, 1]
        if abs(target_linear_velocity) > 0.01:
            # Scale steering command to actual steering angle (assuming max steer = 30 degrees)
            max_steer_angle = np.pi / 6  # 30 degrees in radians
            steering_angle = steer * max_steer_angle
            target_angular_velocity = (target_linear_velocity * np.tan(steering_angle)) / 2.8  # Assuming a wheelbase of 2.8 meters
        else:
            # If not moving, allow in-place rotation
            target_angular_velocity = steer * self.max_angular_velocity
        
        # Clamp velocities to safe limits
        target_linear_velocity = np.clip(target_linear_velocity, 
                                       -self.max_linear_velocity, 
                                       self.max_linear_velocity)
        target_angular_velocity = np.clip(target_angular_velocity, 
                                        -self.max_angular_velocity, 
                                        self.max_angular_velocity)
        
        log.debug(f"Current velocity: {current_velocity}, Target linear: {target_linear_velocity}, "
                 f"Target angular: {target_angular_velocity}, Reverse: {is_reverse}")
        
        # Create and publish cmd_vel message
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = float(target_linear_velocity)
        cmd_vel_msg.linear.y = 0.0
        cmd_vel_msg.linear.z = 0.0
        cmd_vel_msg.angular.x = 0.0
        cmd_vel_msg.angular.y = 0.0
        cmd_vel_msg.angular.z = float(target_angular_velocity)
        
        self.cmd_vel_publisher.publish(cmd_vel_msg)
        log.debug(f"Published cmd_vel: linear.x={cmd_vel_msg.linear.x}, angular.z={cmd_vel_msg.angular.z}")
        
        # Update self.ego_state (you'll need to implement this based on your state tracking)
        self.get_ego_state()

    def _quaternion_to_yaw(self, x: float, y: float, z: float, w: float) -> float:
        """
        Convert quaternion to yaw angle (rotation around Z-axis)
        
        Args:
            x, y, z, w: Quaternion components
            
        Returns:
            Yaw angle in radians
        """
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def reset_model_to_spawn_point(self):
        """
        Reset the model to its original spawn point
        """
        # This would teleport back to the original spawn location
        # You'd need to store the original spawn point during initialization
        if hasattr(self, 'original_spawn_point'):
            return self.teleport_ego(
                self.original_spawn_point['x'],
                self.original_spawn_point['y'],
                self.original_spawn_point['theta']
            )
        else:
            log.warning("Original spawn point not stored, cannot reset")
            return False
