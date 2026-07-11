import time

from avlite.c30_control.c31_control_model import (
    AckermannControlCommand,
    BodyVelocityControlCommand,
    CONTROL_COMMAND_REGISTRY,
    ControlCommand,
    ControlCommandBase,
    ControlComand,
    DiffDriveControlCommand,
)


def test_ackermann_has_timestamp():
    before = time.time()
    cmd = AckermannControlCommand()
    after = time.time()
    assert before <= cmd.timestamp <= after


def test_control_command_alias():
    assert ControlCommand is AckermannControlCommand


def test_control_comand_alias():
    assert ControlComand is AckermannControlCommand


def test_alias_construction():
    cmd = ControlCommand(steer=0.1, acceleration=1.0)
    assert cmd.steer == 0.1
    assert cmd.acceleration == 1.0
    assert isinstance(cmd, ControlCommandBase)


def test_isinstance_base():
    assert isinstance(ControlCommand(), ControlCommandBase)


def test_diff_drive_command_fields():
    cmd = DiffDriveControlCommand(linear=1.0, angular=0.5)
    assert cmd.linear == 1.0
    assert cmd.angular == 0.5
    assert isinstance(cmd, ControlCommandBase)


def test_body_velocity_command_fields():
    cmd = BodyVelocityControlCommand(vx=1.0, vy=0.5, vz=-0.2, yaw_rate=0.1)
    assert cmd.vx == 1.0
    assert cmd.vy == 0.5
    assert cmd.vz == -0.2
    assert cmd.yaw_rate == 0.1
    assert isinstance(cmd, ControlCommandBase)


def test_control_command_registry():
    assert set(CONTROL_COMMAND_REGISTRY) == {
        "AckermannControlCommand",
        "DiffDriveControlCommand",
        "BodyVelocityControlCommand",
    }
