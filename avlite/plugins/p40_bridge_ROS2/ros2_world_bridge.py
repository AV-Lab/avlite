"""
ROS2 World Bridge for AVLite.

Implements WorldBridge by subscribing to localization / perception / sensor topics
from a live ROS stack (e.g. Autoware) and publishing control commands back.

Typical use
-----------
Set in execution config::

    bridge = "ROS2WorldBridge"

Then the executer calls:
  - ``world.get_ego_state()``       → returns latest ego state from localization topic
  - ``world.control_ego_state(cmd)``→ publishes cmd to control_out_topic
  - ``world.get_ground_truth_perception_model()`` → returns perception model populated
                                                    from the perception topic
  - ``world.get_rgb_image()``       → returns latest ``RgbImage`` (see c62_sensor_data)
  - ``world.get_lidar_data()``      → returns latest ``LidarCloud`` (see c62_sensor_data)

Capabilities advertised depend on which sensor topics are configured in settings.
"""
from __future__ import annotations

import json
import logging
import math
import threading
from typing import Optional

import numpy as np

from avlite.c10_perception.c11_perception_model import AgentState, EgoState, PerceptionModel
from avlite.c30_control.c31_control_model import ControlComand
from avlite.c40_execution.c41_world_bridge import WorldBridge
from avlite.c60_common.c61_capabilities import WorldCapability
from avlite.c60_common.c62_sensor_data import LidarCloud, RgbImage, SensorFrame

from .settings import PluginSettings

log = logging.getLogger(__name__)

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except ImportError:
    log.warning("rclpy not found – ROS2WorldBridge will not function.")
    ROS_AVAILABLE = False

try:
    from sensor_msgs.msg import Image, PointCloud2
    import sensor_msgs_py.point_cloud2 as pc2
    SENSOR_MSGS_AVAILABLE = True
except ImportError:
    SENSOR_MSGS_AVAILABLE = False

try:
    from autoware_auto_msgs.msg import BoundingBoxArray
    from autoware_auto_msgs.msg import VehicleControlCommand
    from autoware_auto_msgs.msg import VehicleKinematicState
    AUTOWARE_AVAILABLE = True
except ImportError:
    AUTOWARE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Inline lightweight converters (avoids circular imports with executer_ROS2)
# ---------------------------------------------------------------------------

def _ego_from_kinematic(msg, ego: EgoState) -> None:
    """Update *ego* in-place from a VehicleKinematicState message."""
    state = msg.state
    ego.x = state.x
    ego.y = state.y
    ego.theta = math.atan2(state.heading.imag, state.heading.real)
    ego.velocity = state.longitudinal_velocity_mps


def _agents_from_bounding_boxes(msg) -> list[AgentState]:
    agents: list[AgentState] = []
    for i, box in enumerate(msg.boxes):
        c = box.centroid
        agents.append(
            AgentState(
                x=c.x,
                y=c.y,
                theta=math.atan2(box.heading.imag, box.heading.real),
                velocity=math.hypot(
                    box.velocity.x if hasattr(box, "velocity") else 0.0,
                    box.velocity.y if hasattr(box, "velocity") else 0.0,
                ),
                agent_id=i,
            )
        )
    return agents


def _control_to_vehicle_cmd(cmd: ControlComand, stamp=None) -> "VehicleControlCommand":
    msg = VehicleControlCommand()
    if stamp is not None:
        msg.stamp = stamp
    msg.front_wheel_angle_rad = float(cmd.steer)
    msg.long_accel_mps2 = float(cmd.acceleration)
    return msg


def _image_msg_to_ndarray(msg: "Image") -> Optional[np.ndarray]:
    """Convert a sensor_msgs/Image to an (H, W, 3) uint8 RGB ndarray."""
    try:
        data = np.frombuffer(msg.data, dtype=np.uint8)
        img = data.reshape((msg.height, msg.width, -1))
        if img.shape[2] < 3:
            log.warning(
                "Image encoding '%s' has %d channel(s) – expected a colour image (bgr8/rgb8). Skipping.",
                msg.encoding, img.shape[2],
            )
            return None
        if msg.encoding in ("bgr8", "bgra8"):
            img = img[:, :, :3][..., ::-1].copy()  # BGR → RGB
        else:
            img = img[:, :, :3].copy()
        return img
    except Exception as exc:
        log.error("Failed to decode Image message: %s", exc)
        return None


def _pointcloud2_to_ndarray(msg: "PointCloud2") -> Optional[np.ndarray]:
    """Convert a sensor_msgs/PointCloud2 to an (N, 4) [x, y, z, intensity] float32 ndarray."""
    try:
        points = list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True))
        if not points:
            return None
        return np.array(points, dtype=np.float32)
    except Exception as exc:
        log.error("Failed to decode PointCloud2 message: %s", exc)
        return None


# ---------------------------------------------------------------------------
# ROS2WorldBridge
# ---------------------------------------------------------------------------

class ROS2WorldBridge(WorldBridge, Node if ROS_AVAILABLE else object):
    """
    WorldBridge that connects AVLite to a live ROS2 / Autoware stack.

    Subscribers
    -----------
    - localization_topic  → VehicleKinematicState (or JSON String) → ego state
    - perception_topic    → BoundingBoxArray (or JSON String)       → agent vehicles
    - lidar_topic         → PointCloud2  → lidar buffer (when non-empty and sensor_msgs_py available)
    - rgb_topic           → Image        → RGB buffer  (when non-empty and sensor_msgs available)

    Publisher
    ---------
    - control_out_topic   → VehicleControlCommand (or JSON String)

    ``capabilities`` is dynamic: CAMERA_RGB and LIDAR_3D are included only when
    the respective topic is configured and sensor_msgs is installed.

    Set ``owns_ros_topics = True`` so that ROSExecuter skips its internal
    WorldNode / PerceptionNode / ControllerNode when this bridge is active,
    preventing duplicate publishers and feedback loops.
    """

    owns_ros_topics: bool = True  # coordination flag read by ROSExecuter

    @property
    def capabilities(self) -> set[WorldCapability]:
        caps = {
            WorldCapability.GT_LOCALIZATION,
            WorldCapability.GT_DETECTION,
            WorldCapability.GT_TRACKING,
        }
        if self._rgb_enabled:
            caps.add(WorldCapability.CAMERA_RGB)
        if self._lidar_enabled:
            caps.add(WorldCapability.LIDAR_3D)
        return caps

    def __init__(self, ego_state: Optional[EgoState] = None, pm: Optional[PerceptionModel] = None):
        if not ROS_AVAILABLE:
            raise RuntimeError("rclpy is not available. Install and source ROS2 first.")

        # WorldBridge is a dataclass; initialise it explicitly before Node so
        # that self.ego_state is set before any callbacks fire.
        if ego_state is None:
            ego_state = EgoState(x=0.0, y=0.0, theta=0.0)
        self.ego_state = ego_state
        self.perception_model = pm if pm is not None else PerceptionModel(ego_vehicle=ego_state)

        self._lock = threading.Lock()
        self.settings = PluginSettings()
        self.use_autoware = AUTOWARE_AVAILABLE and self.settings.use_autoware_msgs

        # Sensor buffers
        self._rgb_buffer: Optional[np.ndarray] = None    # (H, W, 3) uint8
        self._lidar_buffer: Optional[np.ndarray] = None  # (N, 4) float32
        self._rgb_lock = threading.Lock()
        self._lidar_lock = threading.Lock()
        self._rgb_enabled = False
        self._lidar_enabled = False

        if not rclpy.ok():
            rclpy.init()

        Node.__init__(self, "avlite_ros2_world_bridge")

        self._setup_subscribers()
        self._setup_publishers()

        # Spin in a daemon thread so the bridge is self-contained
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

        log.info(
            "ROS2WorldBridge started (autoware=%s)\n"
            "  subscribe localization : %s\n"
            "  subscribe perception   : %s\n"
            "  subscribe lidar        : %s\n"
            "  subscribe rgb          : %s\n"
            "  publish  control       : %s",
            self.use_autoware,
            self.settings.localization_topic,
            self.settings.perception_topic,
            self.settings.lidar_topic or "<disabled>",
            self.settings.rgb_topic or "<disabled>",
            self.settings.control_out_topic,
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_subscribers(self) -> None:
        # --- Localization & Perception ---
        if self.use_autoware:
            self.create_subscription(
                VehicleKinematicState,
                self.settings.localization_topic,
                self._on_localization_autoware,
                10,
            )
            self.create_subscription(
                BoundingBoxArray,
                self.settings.perception_topic,
                self._on_perception_autoware,
                10,
            )
        else:
            self.create_subscription(
                String,
                self.settings.localization_topic,
                self._on_localization_json,
                10,
            )
            self.create_subscription(
                String,
                self.settings.perception_topic,
                self._on_perception_json,
                10,
            )

        # --- LiDAR (optional) ---
        if self.settings.lidar_topic and SENSOR_MSGS_AVAILABLE:
            self.create_subscription(
                PointCloud2,
                self.settings.lidar_topic,
                self._on_lidar,
                rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value,
            )
            self._lidar_enabled = True
        elif self.settings.lidar_topic and not SENSOR_MSGS_AVAILABLE:
            log.warning(
                "lidar_topic is set but sensor_msgs_py is not installed – LiDAR disabled. "
                "Install: pip install sensor-msgs-py"
            )

        # --- RGB Camera (optional) ---
        if self.settings.rgb_topic and SENSOR_MSGS_AVAILABLE:
            self.create_subscription(
                Image,
                self.settings.rgb_topic,
                self._on_rgb,
                rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value,
            )
            self._rgb_enabled = True
        elif self.settings.rgb_topic and not SENSOR_MSGS_AVAILABLE:
            log.warning(
                "rgb_topic is set but sensor_msgs is not installed – RGB camera disabled."
            )

    def _setup_publishers(self) -> None:
        if self.use_autoware:
            self._ctrl_pub = self.create_publisher(
                VehicleControlCommand,
                self.settings.control_out_topic,
                10,
            )
        else:
            self._ctrl_pub = self.create_publisher(
                String,
                self.settings.control_out_topic,
                10,
            )

    # ------------------------------------------------------------------
    # Subscription callbacks
    # ------------------------------------------------------------------

    def _on_localization_autoware(self, msg: "VehicleKinematicState") -> None:
        with self._lock:
            _ego_from_kinematic(msg, self.ego_state)
            self.perception_model.ego_vehicle = self.ego_state

    def _on_perception_autoware(self, msg: "BoundingBoxArray") -> None:
        agents = _agents_from_bounding_boxes(msg)
        with self._lock:
            self.perception_model.agent_vehicles = agents

    def _on_localization_json(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            with self._lock:
                self.ego_state.x = float(data.get("x", self.ego_state.x))
                self.ego_state.y = float(data.get("y", self.ego_state.y))
                self.ego_state.theta = float(data.get("theta", self.ego_state.theta))
                self.ego_state.velocity = float(data.get("velocity", self.ego_state.velocity))
                self.perception_model.ego_vehicle = self.ego_state
        except (json.JSONDecodeError, ValueError) as exc:
            self.get_logger().error(f"Invalid JSON localization: {exc}")

    def _on_perception_json(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            agents = []
            for obj in data.get("objects", []):
                agents.append(
                    AgentState(
                        x=float(obj.get("x", 0)),
                        y=float(obj.get("y", 0)),
                        theta=float(obj.get("theta", 0)),
                        velocity=float(obj.get("velocity", 0)),
                        agent_id=int(obj.get("id", 0)),
                    )
                )
            with self._lock:
                self.perception_model.agent_vehicles = agents
        except (json.JSONDecodeError, ValueError) as exc:
            self.get_logger().error(f"Invalid JSON perception: {exc}")

    def _on_lidar(self, msg: "PointCloud2") -> None:
        arr = _pointcloud2_to_ndarray(msg)
        if arr is not None:
            with self._lidar_lock:
                self._lidar_buffer = arr

    def _on_rgb(self, msg: "Image") -> None:
        arr = _image_msg_to_ndarray(msg)
        if arr is not None:
            with self._rgb_lock:
                self._rgb_buffer = arr

    # ------------------------------------------------------------------
    # WorldBridge interface
    # ------------------------------------------------------------------

    def control_ego_state(self, cmd: ControlComand, dt: Optional[float] = 0.01) -> None:
        """Publish the control command to the ROS control topic."""
        if self.use_autoware:
            stamp = self.get_clock().now().to_msg()
            ros_msg = _control_to_vehicle_cmd(cmd, stamp)
        else:
            ros_msg = String()
            ros_msg.data = json.dumps(
                {"steer": float(cmd.steer), "acceleration": float(cmd.acceleration)}
            )
        self._ctrl_pub.publish(ros_msg)

    def get_ego_state(self) -> EgoState:
        with self._lock:
            return self.ego_state

    def get_ground_truth_perception_model(self) -> PerceptionModel:
        with self._lock:
            return self.perception_model

    def get_rgb_image(self) -> Optional[RgbImage]:
        """Return the latest RGB frame, or None if not yet received."""
        with self._rgb_lock:
            return self._rgb_buffer

    def get_lidar_data(self) -> Optional[LidarCloud]:
        """Return the latest point cloud, or None."""
        with self._lidar_lock:
            return self._lidar_buffer

    def get_sensor_frame(self) -> SensorFrame:
        """Return an atomic snapshot of buffered sensor data."""
        with self._rgb_lock:
            rgb = self._rgb_buffer
        with self._lidar_lock:
            lidar = self._lidar_buffer
        return SensorFrame(rgb=rgb, lidar=lidar)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _spin(self) -> None:
        try:
            rclpy.spin(self)
        except Exception as exc:
            log.error("ROS2WorldBridge spin error: %s", exc)
