from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from avlite.c10_perception.c11_perception_model import AgentState, EgoState, EGO_AGENT_ID, PerceptionModel
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c30_control.c31_control_model import ControlCommandBase
from avlite.c30_control.c38_control_mapping import control_type_for_agent
from avlite.c60_common.c61_capabilities import WorldCapability
from avlite.c60_common.c62_sensor_data import (
    GnssReading,
    ImuReading,
    SensorFrame,
    WheelOdometry,
    DepthImage,
    LidarCloud,
    RgbImage,
)


@dataclass
class WorldBridge(ABC):
    """
    Abstract class for the world interface. This class is used to control the ego vehicle and spawn agents in the world.
    It provides an interface for the simulator or ROS bridge to implement its own world logic.
    """

    ego_state: EgoState
    perception_model: Optional[PerceptionModel] = None  # Simulators can provide ground truth perception model
    reference_point: tuple[float, float] | None = None  # WGS84 (lat_deg, lon_deg) map origin

    registry = {}

    @property
    @abstractmethod
    def capabilities(self) -> set[WorldCapability]:
        """Set of supported capabilities (must be implemented by subclass)."""
        pass

    @abstractmethod
    def control_ego_state(self, cmd: ControlCommandBase, dt: Optional[float] = 0.01):
        """
        Update the ego state.

        Parameters
        cmd (ControlCommandBase): The control command (typically AckermannControlCommand).
        dt (float): Time delta for the update if supported. Default is 0.01.
        """
        pass

    def control_type(self, agent: AgentState) -> type[ControlCommandBase]:
        """Command class this bridge expects for the given agent."""
        return control_type_for_agent(agent)

    def control_agent(
        self,
        agent_id: int,
        cmd: ControlCommandBase,
        dt: Optional[float] = 0.01,
    ) -> None:
        """Apply control to any agent. Default: delegate ego to control_ego_state."""
        if agent_id == EGO_AGENT_ID:
            self.control_ego_state(cmd, dt=dt)
            return
        raise NotImplementedError(
            f"{type(self).__name__} does not support control of agent {agent_id}"
        )

    def step(self, dt: Optional[float] = 0.01) -> None:
        """Advance the world by dt without a new command from the control stack."""
        pass

    def get_ego_state(self) -> EgoState:
        return self.ego_state

    def teleport_ego(self, x: float, y: float, theta: Optional[float] = None):
        """
        Teleport the ego vehicle to a new position and orientation.

        Parameters
        x (float): The new x-coordinate.
        y (float): The new y-coordinate.
        theta (float): The new orientation in radians.
        """
        raise NotImplementedError("This method should be implemented by the simulator or ROS bridge.")

    def teleport_agent(
        self,
        agent_id: int,
        x: float,
        y: float,
        theta: Optional[float] = None,
    ) -> None:
        """Teleport any agent. Default: ego delegates to teleport_ego; NPC raises."""
        if agent_id == EGO_AGENT_ID:
            self.teleport_ego(x, y, theta)
            return
        raise NotImplementedError(
            f"{type(self).__name__} does not support teleport of agent {agent_id}"
        )

    def spawn_agent(self, agent_state: AgentState, global_plan: Optional[GlobalPlan] = None):
        """Spawn an agent. ``global_plan`` is optional ego route context for route-following NPCs."""
        raise NotImplementedError("This method should be implemented by the simulator or ROS bridge.")

    def get_ground_truth_perception_model(self) -> PerceptionModel:
        """ Returns the perception model of the world. This method should be implemented by simulators  """
        raise NotImplementedError("This method should be implemented by the simulator or ROS bridge.")

    def get_rgb_image(self, agent_id: int = EGO_AGENT_ID) -> RgbImage | None:
        """Returns the RGB image. Layout: ``RgbImage`` in c62_sensor_data."""
        self._require_ego_agent(agent_id, "rgb")
        return None

    def get_depth_image(self, agent_id: int = EGO_AGENT_ID) -> DepthImage | None:
        """Returns the depth image. Layout: ``DepthImage`` in c62_sensor_data."""
        self._require_ego_agent(agent_id, "depth")
        return None

    def get_lidar_data(self, agent_id: int = EGO_AGENT_ID) -> LidarCloud | None:
        """Returns the lidar point cloud. Layout: ``LidarCloud`` in c62_sensor_data."""
        self._require_ego_agent(agent_id, "lidar")
        return None

    def get_imu(self, agent_id: int = EGO_AGENT_ID) -> ImuReading | None:
        self._require_ego_agent(agent_id, "imu")
        return None

    def get_gnss(self, agent_id: int = EGO_AGENT_ID) -> GnssReading | None:
        self._require_ego_agent(agent_id, "gnss")
        return None

    def get_wheel_odometry(self, agent_id: int = EGO_AGENT_ID) -> WheelOdometry | None:
        self._require_ego_agent(agent_id, "wheel odometry")
        return None

    def get_sensor_frame(self, agent_id: int = EGO_AGENT_ID) -> SensorFrame:
        """Compose a sensor snapshot from individual getters. Override for atomic reads."""
        if agent_id == EGO_AGENT_ID:
            return SensorFrame(
                rgb=self.get_rgb_image(),
                depth=self.get_depth_image(),
                lidar=self.get_lidar_data(),
                imu=self.get_imu(),
                gnss=self.get_gnss(),
                wheel_odometry=self.get_wheel_odometry(),
            )
        return SensorFrame(
            rgb=self.get_rgb_image(agent_id=agent_id),
            depth=self.get_depth_image(agent_id=agent_id),
            lidar=self.get_lidar_data(agent_id=agent_id),
            imu=self.get_imu(agent_id=agent_id),
            gnss=self.get_gnss(agent_id=agent_id),
            wheel_odometry=self.get_wheel_odometry(agent_id=agent_id),
        )

    def _require_ego_agent(self, agent_id: int, method: str) -> None:
        if agent_id != EGO_AGENT_ID:
            raise NotImplementedError(
                f"{type(self).__name__} does not support {method} for agent {agent_id}"
            )

    def reset(self):
        pass

    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:
            WorldBridge.registry[cls.__name__] = cls
