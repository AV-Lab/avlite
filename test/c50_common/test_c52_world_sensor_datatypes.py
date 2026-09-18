from dataclasses import replace

import numpy as np
import pytest

from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c50_common.c52_world_sensor_datatypes import (
    Camera,
    GnssDatum,
    Gnss,
    Imu,
    Lidar,
    Sensor,
    SensorFrame,
    WheelOdometry,
)


def _mount(x=0.0, y=0.0, z=0.0, yaw=0.0) -> np.ndarray:
    t = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    t[:2, :2] = [[c, -s], [s, c]]
    t[:3, 3] = [x, y, z]
    return t


@pytest.mark.parametrize("name", ["Camera", "Lidar", "Imu", "Gnss", "WheelOdometry"])
def test_sensor_types_are_exported_from_public_api(name):
    import avlite
    from avlite.c50_common import c52_world_sensor_datatypes as sensors

    assert getattr(avlite, name) is getattr(sensors, name)
    assert issubclass(getattr(avlite, name), sensors.Sensor)


def test_sensor_frame_defaults():
    frame = SensorFrame()
    assert frame.cameras == {}
    assert frame.lidars == {}
    assert frame.camera is None
    assert frame.lidar is None
    assert frame.get_camera() is None
    assert frame.get_lidar() is None
    assert frame.stamp is None
    assert frame.imu is None
    assert frame.gnss is None
    assert frame.wheel_odometry is None
    assert not hasattr(frame, "imu_sensor")
    assert not hasattr(frame, "gnss_sensor")


def test_camera_coerces_to_float64():
    params = Camera(
        intrinsic=[[400, 0, 320], [0, 400, 240], [0, 0, 1]],
        width=640,
        height=480,
        base_to_sensor=np.eye(4, dtype=np.float32),
    )
    assert params.intrinsic.shape == (3, 3)
    assert params.intrinsic.dtype == np.float64
    assert params.base_to_sensor.dtype == np.float64
    assert params.intrinsic[0, 2] == pytest.approx(320.0)

    frame = SensorFrame(cameras={"front": params}, primary_camera_name="front")
    assert frame.camera.width == 640


def test_camera_rejects_bad_intrinsic_shape():
    with pytest.raises(ValueError, match=r"\(3, 3\) intrinsic"):
        Camera(intrinsic=np.zeros((3, 4)), width=640, height=480)


def test_sensor_mount_defaults_to_identity_and_validates_shape():
    for sensor in (
        Camera(intrinsic=np.eye(3), width=4, height=4),
        Lidar(),
        Sensor(),
    ):
        assert isinstance(sensor, Sensor)
        np.testing.assert_array_equal(sensor.base_to_sensor, np.eye(4))

    with pytest.raises(ValueError, match=r"\(4, 4\) base_to_sensor"):
        Lidar(base_to_sensor=np.eye(3))


def test_single_source_readings_are_sensors():
    for measurement in (
        Imu(linear_accel=(0, 0, 9.8), angular_velocity=(0, 0, 0.0)),
        Gnss(latitude=1.0, longitude=2.0, altitude=3.0),
        WheelOdometry(linear_velocity=1.0, yaw_rate=0.0),
    ):
        assert isinstance(measurement, Sensor)
        np.testing.assert_array_equal(measurement.base_to_sensor, np.eye(4))
        assert measurement.sensor_name is None
        assert measurement.sensor_id is None
        assert measurement.stamp is None


@pytest.mark.parametrize("sensor_type, reading_fields", [
    (Imu, {"linear_accel": (0, 0, 9.8), "angular_velocity": (0, 0, 0.1)}),
    (Gnss, {"latitude": 24.0, "longitude": 54.0, "altitude": 10.0}),
    (WheelOdometry, {"linear_velocity": 5.0, "yaw_rate": 0.05}),
])
def test_single_source_sensor_metadata_and_validation(sensor_type, reading_fields):
    mount = _mount(x=2.0).astype(np.float32)
    sensor = sensor_type(
        **reading_fields, base_to_sensor=mount,
        sensor_name="body", sensor_id="device-01", stamp=1.0,
    )
    assert sensor.base_to_sensor.dtype == np.float64
    np.testing.assert_array_equal(sensor.base_to_sensor, mount)
    newer = replace(sensor, stamp=2.0)
    assert newer is not sensor
    assert newer.sensor_name == "body" and newer.sensor_id == "device-01"
    assert newer.stamp == 2.0 and sensor.stamp == 1.0
    for name, value in reading_fields.items():
        assert getattr(newer, name) == value
    with pytest.raises(ValueError, match=r"\(4, 4\) base_to_sensor"):
        sensor_type(**reading_fields, base_to_sensor=np.eye(3))


def test_to_base_applies_mount_and_keeps_intensity():
    mount = Lidar(base_to_sensor=_mount(x=1.0, z=0.5, yaw=np.pi / 2))
    cloud = np.array([[2.0, 0.0, 0.0, 7.0]], dtype=np.float32)
    out = mount.to_base(cloud)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[1.0, 2.0, 0.5, 7.0]], atol=1e-6)
    # 2D input stays 2D; None passes through.
    np.testing.assert_allclose(mount.to_base(np.array([[2.0, 0.0]])), [[1.0, 2.0]], atol=1e-9)
    assert mount.to_base(None) is None


def test_to_map_composes_mount_and_state_pose():
    mount = Lidar(base_to_sensor=_mount(x=1.0))
    cloud = np.array([[1.0, 0.0, 0.3, 9.0]], dtype=np.float32)
    # Lidar 1 m ahead of the body origin, ego at (10, 5) facing +y.
    ego = EgoState(x=10.0, y=5.0, theta=np.pi / 2)
    np.testing.assert_allclose(mount.to_map(cloud, ego), [[10.0, 7.0, 0.3, 9.0]], atol=1e-6)
    # Identity mount = already in the body frame; ego z is added to the cloud z.
    ego = EgoState(x=10.0, y=5.0, z=0.2, theta=0.0)
    np.testing.assert_allclose(Lidar().to_map(cloud, ego), [[11.0, 5.0, 0.5, 9.0]], atol=1e-6)
    assert Lidar().to_map(None, ego) is None
    assert Lidar().to_map(np.zeros((0, 4)), ego).shape == (0, 4)


def test_world_capability_sensor_fields_cover_all_caps():
    from avlite.c50_common.c51_capabilities import WorldCapability
    from avlite.c50_common.c52_world_sensor_datatypes import WORLD_CAPABILITY_SENSOR_FIELDS

    assert set(WORLD_CAPABILITY_SENSOR_FIELDS) == set(WorldCapability)
    assert WORLD_CAPABILITY_SENSOR_FIELDS[WorldCapability.CAMERA_RGB] == "cameras.rgb"
    assert WORLD_CAPABILITY_SENSOR_FIELDS[WorldCapability.CAMERA_DEPTH] == "cameras.depth"
    assert WORLD_CAPABILITY_SENSOR_FIELDS[WorldCapability.LIDAR_2D] == "lidars.points"
    assert WORLD_CAPABILITY_SENSOR_FIELDS[WorldCapability.LIDAR_3D] == "lidars.points"
    assert WORLD_CAPABILITY_SENSOR_FIELDS[WorldCapability.RADAR] is None


def test_gnss_reading_datum():
    fix = Gnss(latitude=24.0, longitude=54.0, altitude=10.0)
    assert fix.datum == GnssDatum.WGS84


def test_imu_and_wheel_odometry():
    imu = Imu(linear_accel=(0, 0, 9.8), angular_velocity=(0, 0, 0.1))
    odom = WheelOdometry(linear_velocity=5.0, yaw_rate=0.05)
    frame = SensorFrame(imu=imu, wheel_odometry=odom)
    assert frame.imu.linear_accel[2] == pytest.approx(9.8)
    assert frame.wheel_odometry.yaw_rate == pytest.approx(0.05)


@pytest.fixture(params=["camera", "lidar"])
def modality(request):
    """Exercise the same collection contract for both sensor types."""
    if request.param == "camera":

        def make_sensor(**kwargs):
            return Camera(intrinsic=np.eye(3), width=4, height=3, **kwargs)
    else:
        make_sensor = Lidar
    return request.param, make_sensor


def test_lookup_by_name_id_and_primary_returns_same_sensor(modality):
    kind, make_sensor = modality
    front = make_sensor(sensor_name="front", sensor_id="device-01")
    rear = make_sensor(sensor_name="rear", sensor_id="device-02")
    frame = SensorFrame(
        **{
            f"{kind}s": {"front": front, "rear": rear},
            f"primary_{kind}_name": "rear",
        }
    )
    get_sensor = getattr(frame, f"get_{kind}")
    assert get_sensor("front") is front
    assert get_sensor("device-01") is front
    assert get_sensor("rear") is rear
    assert get_sensor("device-02") is rear
    assert get_sensor() is rear
    assert get_sensor("primary") is rear
    assert getattr(frame, kind) is getattr(frame, f"{kind}s")["rear"]


def test_nonempty_collection_does_not_implicitly_select_primary(modality):
    kind, make_sensor = modality
    sensor = make_sensor()
    frame = SensorFrame(**{f"{kind}s": {"front": sensor}})
    assert getattr(frame, kind) is None
    assert getattr(frame, f"get_{kind}")() is None
    assert getattr(frame, f"get_{kind}")("front") is sensor


def test_missing_lookup_raises_key_error(modality):
    kind, _ = modality
    with pytest.raises(KeyError, match="Unknown sensor: missing"):
        getattr(SensorFrame(), f"get_{kind}")("missing")


@pytest.mark.parametrize("primary", ["missing", "device-01", "primary"])
def test_primary_must_be_an_existing_collection_name(modality, primary):
    kind, make_sensor = modality
    with pytest.raises(ValueError, match="Unknown primary"):
        SensorFrame(
            **{
                f"{kind}s": {"front": make_sensor(sensor_id="device-01")},
                f"primary_{kind}_name": primary,
            }
        )


def test_sensor_name_must_match_collection_key(modality):
    kind, make_sensor = modality
    with pytest.raises(ValueError, match="does not match key"):
        SensorFrame(**{f"{kind}s": {"front": make_sensor(sensor_name="rear")}})


@pytest.mark.parametrize("name", ["primary", "", 12])
def test_collection_names_are_nonempty_strings_and_primary_is_reserved(modality, name):
    kind, make_sensor = modality
    with pytest.raises(ValueError, match="Sensor names must be"):
        SensorFrame(**{f"{kind}s": {name: make_sensor()}})


@pytest.mark.parametrize("sensor_id", ["primary", "", 12])
def test_sensor_ids_are_nonempty_strings_and_primary_is_reserved(modality, sensor_id):
    kind, make_sensor = modality
    with pytest.raises(ValueError, match="Sensor IDs must be"):
        SensorFrame(**{f"{kind}s": {"front": make_sensor(sensor_id=sensor_id)}})


def test_duplicate_ids_are_rejected(modality):
    kind, make_sensor = modality
    with pytest.raises(ValueError, match="Duplicate sensor ID"):
        SensorFrame(
            **{
                f"{kind}s": {
                    "front": make_sensor(sensor_id="duplicate"),
                    "rear": make_sensor(sensor_id="duplicate"),
                }
            }
        )


def test_id_cannot_shadow_another_sensor_name(modality):
    kind, make_sensor = modality
    with pytest.raises(ValueError, match="Ambiguous sensor name/ID"):
        SensorFrame(
            **{
                f"{kind}s": {
                    "front": make_sensor(sensor_id="rear"),
                    "rear": make_sensor(),
                }
            }
        )


def test_same_sensor_can_have_matching_name_and_id(modality):
    kind, make_sensor = modality
    sensor = make_sensor(sensor_name="front", sensor_id="front")
    frame = SensorFrame(**{f"{kind}s": {"front": sensor}})
    assert getattr(frame, f"get_{kind}")("front") is sensor


def test_lookup_rejects_ambiguity_if_collection_was_modified(modality):
    kind, make_sensor = modality
    frame = SensorFrame(**{f"{kind}s": {"front": make_sensor(sensor_id="device-01")}})
    getattr(frame, f"{kind}s")["device-01"] = make_sensor()
    with pytest.raises(ValueError, match="Ambiguous sensor name/ID"):
        getattr(frame, f"get_{kind}")("device-01")


def test_collections_reject_wrong_sensor_type(modality):
    kind, _ = modality
    with pytest.raises(TypeError, match="Expected"):
        SensorFrame(**{f"{kind}s": {"front": Sensor()}})


def test_collections_are_independent_between_frames():
    first, second = SensorFrame(), SensorFrame()
    first.cameras["front"] = Camera(np.eye(3), 4, 3)
    first.lidars["top"] = Lidar()
    assert second.cameras == {}
    assert second.lidars == {}


def test_ids_are_scoped_to_modality():
    camera = Camera(np.eye(3), 4, 3, sensor_id="device-01")
    lidar = Lidar(sensor_id="device-01")
    frame = SensorFrame(cameras={"front": camera}, lidars={"front": lidar})
    assert frame.get_camera("device-01") is camera
    assert frame.get_lidar("device-01") is lidar


def test_readings_and_timestamps_belong_to_each_sensor():
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    depth = np.ones((3, 4), dtype=np.float32)
    cloud = np.array([[1, 2, 3, 4]], dtype=np.float32)
    front = Camera(np.eye(3), 4, 3, rgb=image, depth=depth, stamp=1.0)
    rear = Camera(np.eye(3), 4, 3, stamp=0.9)
    top = Lidar(points=cloud, stamp=0.95)
    frame = SensorFrame(
        cameras={"front": front, "rear": rear},
        lidars={"top": top},
        primary_camera_name="front",
        primary_lidar_name="top",
        stamp=1.1,
    )
    assert frame.camera.rgb is image
    assert frame.camera.depth is depth
    assert frame.lidar.points is cloud
    assert frame.get_camera("rear").rgb is None
    assert frame.camera.stamp == 1.0
    assert frame.get_camera("rear").stamp == 0.9
    assert frame.lidar.stamp == 0.95
    assert frame.stamp == 1.1


def test_primary_without_reading_does_not_fall_back_to_another_sensor():
    missing_camera = Camera(np.eye(3), 4, 3)
    missing_lidar = Lidar()
    frame = SensorFrame(
        cameras={
            "front": missing_camera,
            "rear": Camera(np.eye(3), 4, 3, rgb=np.zeros((3, 4, 3), dtype=np.uint8)),
        },
        lidars={"top": missing_lidar, "rear": Lidar(points=np.zeros((1, 4)))},
        primary_camera_name="front",
        primary_lidar_name="top",
    )
    assert frame.camera is missing_camera
    assert frame.camera.rgb is None
    assert frame.lidar is missing_lidar
    assert frame.lidar.points is None


def test_fresh_acquisition_with_replace_preserves_previous_snapshot():
    old_image = np.zeros((3, 4, 3), dtype=np.uint8)
    new_image = np.ones((3, 4, 3), dtype=np.uint8)
    first = Camera(np.eye(3), 4, 3, sensor_name="front", rgb=old_image, stamp=1.0)
    second = replace(first, rgb=new_image, stamp=2.0)
    first_frame = SensorFrame(cameras={"front": first}, primary_camera_name="front")
    second_frame = SensorFrame(cameras={"front": second}, primary_camera_name="front")
    assert first_frame.camera is not second_frame.camera
    assert first_frame.camera.rgb is old_image
    assert second_frame.camera.rgb is new_image
    assert first_frame.camera.stamp == 1.0
    assert second_frame.camera.stamp == 2.0
    assert second.sensor_name == first.sensor_name
    np.testing.assert_array_equal(first.intrinsic, second.intrinsic)
    np.testing.assert_array_equal(first.base_to_sensor, second.base_to_sensor)


def test_capability_paths_resolve_to_readings():
    from avlite.c50_common.c52_world_sensor_datatypes import WORLD_CAPABILITY_SENSOR_FIELDS

    frame = SensorFrame(
        cameras={"front": Camera(np.eye(3), 4, 3)},
        lidars={"top": Lidar()},
    )
    for path in WORLD_CAPABILITY_SENSOR_FIELDS.values():
        if path is None:
            continue
        collection, separator, payload = path.partition(".")
        target = getattr(frame, collection)
        if separator:
            assert target
            assert all(hasattr(sensor, payload) for sensor in target.values())


def test_sensor_frame_accepts_retained_sensors_after_module_reload():
    # Keep the reload isolated: other tests import these classes at collection.
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", """
import importlib
import numpy as np
import avlite.c50_common.c52_world_sensor_datatypes as sensors
camera = sensors.Camera(np.eye(3), 4, 3)
lidar = sensors.Lidar(points=np.zeros((1, 4)))
old_frame_type = sensors.SensorFrame
importlib.reload(sensors)
for frame_type in (old_frame_type, sensors.SensorFrame):
    frame = frame_type(cameras={'front': camera}, lidars={'top': lidar},
                       primary_camera_name='front', primary_lidar_name='top')
    assert frame.camera is camera
    assert frame.lidar is lidar
try:
    sensors.SensorFrame(lidars={'wrong': camera})
except TypeError:
    pass
else:
    raise AssertionError('Camera accepted as a lidar')
"""],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
