from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import ClassVar, Optional

from avlite.c10_perception.c11_perception_model import AgentState, EgoState, EGO_AGENT_ID, Map, PerceptionModel
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c30_control.c31_control_model import ControlCommandBase
from avlite.c50_common.c51_capabilities import StackCapability, StackRequirement, WorldCapability
from avlite.c50_common.c53_stack_datatypes import control_type_for_agent
from avlite.c50_common.c52_world_sensor_datatypes import (
    WORLD_CAPABILITY_SENSOR_FIELDS,
    Camera,
    Gnss,
    Imu,
    Lidar,
    SensorFrame,
    WheelOdometry,
    DepthImage,
    LidarCloud,
    RgbImage,
)

import logging

log = logging.getLogger(__name__)


@dataclass
class WorldBridge(ABC):
    """
    Abstract class for the world interface. This class is used to control the ego vehicle and spawn agents in the world.
    It provides an interface for the simulator or ROS bridge to implement its own world logic.
    """

    ego_state: EgoState
    perception_model: Optional[PerceptionModel] = None  # Simulators can provide ground truth perception model
    reference_point: tuple[float, float] | None = None  # WGS84 (lat_deg, lon_deg) map origin
    map: Map | None = None  # Static map for simulation (LiDAR geometry, GT MAP); None for real-world bridges

    registry = {}

    # Soft default: bridges may declare stack deps (e.g. CONTROL) without subclass boilerplate.
    stack_requirements: ClassVar[frozenset[StackRequirement]] = frozenset()

    @property
    @abstractmethod
    def world_capabilities(self) -> frozenset[WorldCapability]:
        """Sensors / actuation this bridge exposes to the stack."""

    @property
    @abstractmethod
    def stack_capabilities(self) -> frozenset[StackCapability]:
        """Ground-truth stack capabilities this bridge provides (may be empty)."""

    @abstractmethod
    def control_ego_state(self, cmd: ControlCommandBase, dt: Optional[float] = 0.01):
        """
        Update the ego state.

        Parameters
        cmd (ControlCommandBase): The control command (typically AckermannControlCommand).
        dt (float): Time delta for the update if supported. Default is 0.01.
        """
        pass

    def control_agent(self, agent_id: int, cmd: ControlCommandBase, dt: Optional[float] = 0.01,) -> None:
        """Apply control to any agent. Default: delegate ego to control_ego_state."""
        if agent_id == EGO_AGENT_ID:
            self.control_ego_state(cmd, dt=dt)
            return
        raise NotImplementedError( f"{type(self).__name__} does not support control of agent {agent_id}")

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

    def teleport_agent(self, agent_state: AgentState) -> None:
        """Teleport any agent to the pose in ``agent_state``.

        Identity comes from ``agent_state.agent_id`` (same pattern as
        :meth:`spawn_agent`). Only pose is applied (``x``, ``y``, ``theta``;
        later ``z`` for aerial/etc.); velocity, size, and type are ignored.

        Default: ego delegates to :meth:`teleport_ego`; NPC raises.
        """
        if agent_state.agent_id == EGO_AGENT_ID:
            self.teleport_ego(agent_state.x, agent_state.y, agent_state.theta)
            return
        raise NotImplementedError(
            f"{type(self).__name__} does not support teleport of agent {agent_state.agent_id}"
        )

    def spawn_agent(self, agent_state: AgentState, global_plan: Optional[GlobalPlan] = None):
        """Spawn an agent. ``global_plan`` is optional ego route context for route-following NPCs."""
        raise NotImplementedError("This method should be implemented by the simulator or ROS bridge.")

    def get_ground_truth_perception_model(self) -> PerceptionModel:
        """ Returns the perception model of the world. This method should be implemented by simulators  """
        raise NotImplementedError("This method should be implemented by the simulator or ROS bridge.")

    def get_rgb_image(self, agent_id: int = EGO_AGENT_ID) -> RgbImage | None:
        """Returns the RGB image. Layout: ``RgbImage`` in c52_world_sensor_datatypes."""
        return None

    def get_depth_image(self, agent_id: int = EGO_AGENT_ID) -> DepthImage | None:
        """Returns the depth image. Layout: ``DepthImage`` in c52_world_sensor_datatypes."""
        return None

    def get_camera_sensor(self, agent_id: int = EGO_AGENT_ID) -> Camera | None:
        """Static description of the camera that produced rgb/depth. Layout: ``Camera``.

        Bridges exposing CAMERA_RGB or CAMERA_DEPTH must override this; without
        it, LiDAR cannot be projected into the image. ``base_to_sensor`` is the
        static mount of the optical frame in the ego body frame.
        """
        return None

    def get_lidar_data(self, agent_id: int = EGO_AGENT_ID) -> LidarCloud | None:
        """Returns the lidar point cloud in the lidar's own coordinate frame.

        Layout: ``LidarCloud`` in c52_world_sensor_datatypes. The stack places
        it in the map frame from its own pose estimate.
        """
        return None

    def get_lidar_sensor(self, agent_id: int = EGO_AGENT_ID) -> Lidar:
        """Static description of the lidar (mount in the ego body frame). Default: identity."""
        return Lidar()

    def get_imu(self, agent_id: int = EGO_AGENT_ID) -> Imu | None:
        """Return the IMU snapshot, including its mount, identity, and timestamp."""
        return None

    def get_gnss(self, agent_id: int = EGO_AGENT_ID) -> Gnss | None:
        """Return the GNSS snapshot, including antenna mount and acquisition time."""
        return None

    def get_wheel_odometry(self, agent_id: int = EGO_AGENT_ID) -> WheelOdometry | None:
        """Return source metadata and body-relative wheel-odometry readings."""
        return None

    def get_sensor_frame(self, agent_id: int = EGO_AGENT_ID) -> SensorFrame:
        """Compose primary camera/lidar snapshots from the single-device getters.

        Existing getter implementations can keep returning calibration and
        readings separately. This adapter creates fresh sensor objects and
        selects each as primary, using its name or "camera"/"lidar" as fallback.
        Unknown timestamps remain None; assembly time is not acquisition time.
        IMU, GNSS, and wheel getters return complete single-source sensors;
        their metadata is copied together with their reading fields.

        Override for atomic reads or multiple devices, populating ``cameras``
        and ``lidars`` directly with explicit primary names, then call
        :meth:`_apply_world_capability_filter`. Reading buffers must remain
        stable for the lifetime of a snapshot; copy any driver-reused buffers.

        Non-ego ``agent_id`` requires ``WorldCapability.AGENT_SENSING``.
        """
        if (
            agent_id != EGO_AGENT_ID
            and WorldCapability.AGENT_SENSING not in self.world_capabilities
        ):
            raise NotImplementedError(
                f"{type(self).__name__} does not support sensors for agent {agent_id}"
            )
        # BasicSim and older ego-only bridges define getters without kwargs.
        kwargs = {} if agent_id == EGO_AGENT_ID else {"agent_id": agent_id}
        rgb = self.get_rgb_image(**kwargs)
        depth = self.get_depth_image(**kwargs)
        camera = self.get_camera_sensor(**kwargs)
        points = self.get_lidar_data(**kwargs)
        lidar = self.get_lidar_sensor(**kwargs)

        cameras: dict[str, Camera] = {}
        camera_name = None
        if camera is not None:
            camera_name = camera.sensor_name if camera.sensor_name is not None else "camera"
            cameras[camera_name] = replace(camera, sensor_name=camera_name, rgb=rgb, depth=depth)
        elif rgb is not None or depth is not None:
            raise ValueError("Camera readings require calibration from get_camera_sensor()")

        lidars: dict[str, Lidar] = {}
        lidar_name = None
        if points is not None or self.world_capabilities.intersection(
            {WorldCapability.LIDAR_2D, WorldCapability.LIDAR_3D}
        ):
            lidar_name = lidar.sensor_name if lidar.sensor_name is not None else "lidar"
            lidars[lidar_name] = replace(lidar, sensor_name=lidar_name, points=points)

        imu = self.get_imu(**kwargs)
        gnss = self.get_gnss(**kwargs)
        wheel_odometry = self.get_wheel_odometry(**kwargs)
        frame = SensorFrame(
            cameras=cameras,
            lidars=lidars,
            primary_camera_name=camera_name,
            primary_lidar_name=lidar_name,
            imu=replace(imu) if imu is not None else None,
            gnss=replace(gnss) if gnss is not None else None,
            wheel_odometry=replace(wheel_odometry) if wheel_odometry is not None else None,
        )

        log.debug("Sensor frame before world capability filter: %s", frame)
        return self._apply_world_capability_filter(frame)

    def reset(self):
        pass

    @staticmethod
    def _apply_world_capability_filter(frame: SensorFrame) -> SensorFrame:
        """Clear disabled camera/lidar payloads, preserving metadata and primaries.

        Single-source IMU/GNSS/wheel entries become None when disabled.
        Mutates this fresh frame; bridges must not share its sensor-state
        objects with previous snapshots or their static calibration objects.
        """
        cleared: set[str] = set()
        for cap, field in WORLD_CAPABILITY_SENSOR_FIELDS.items():
            if field is None or field in cleared:
                continue
            if not is_world_capability_enabled(cap):
                # LiDAR: keep the field if either 2D or 3D is still provided.
                peers = [
                    c for c, f in WORLD_CAPABILITY_SENSOR_FIELDS.items() if f == field
                ]
                if any(is_world_capability_enabled(c) for c in peers):
                    continue
                collection, separator, payload = field.partition(".")
                if separator:
                    for sensor in getattr(frame, collection).values():
                        setattr(sensor, payload, None)
                else:
                    setattr(frame, field, None)
                cleared.add(field)
        return frame

    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:
            WorldBridge.registry[cls.__name__] = cls


def is_world_capability_enabled(cap: WorldCapability) -> bool:
    """Whether *cap* is enabled in the Bridge Setting world-capability filter.

    ``None`` on ``ExecutionSettings.c41_world_capabilities`` means all enabled.
    """
    from avlite.c40_execution.c49_settings import ExecutionSettings

    val = ExecutionSettings.c41_world_capabilities
    return True if val is None else cap.name in val


def is_world_stack_capability_enabled(cap: StackCapability) -> bool:
    """Whether bridge GT *cap* is enabled in the Bridge Setting stack filter.

    ``None`` on ``ExecutionSettings.c41_world_stack_capabilities`` means all enabled.
    """
    from avlite.c40_execution.c49_settings import ExecutionSettings

    val = ExecutionSettings.c41_world_stack_capabilities
    return True if val is None else cap.name in val
