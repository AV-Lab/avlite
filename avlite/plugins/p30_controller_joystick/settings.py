"""Settings for the JoystickController extension (Xbox-style gamepad).

Usage: select **JoystickController** in the Control dropdown, then run
simulation or press Step. Input is read inside ``control()`` each tick.

Axis mapping (Xbox controller via pygame; triggers are analog axes, not buttons):

| Control          | Xbox input       | Default axis | Output                                      |
|------------------|------------------|--------------|---------------------------------------------|
| Steer left/right | Left stick X     | 0            | ``steer``, scaled to ``ego_max_steering``   |
| Accelerate       | Right trigger RT | 4            | ``acceleration``, normalized from rest to full press |
| Brake            | Left trigger LT  | 5            | added to acceleration, same normalization            |

Trigger rest positions are auto-calibrated at init (release triggers when
starting). Devices differ: some axes rest at 0, others at -1.

Not mapped: face buttons (A/B/X/Y), bumpers (LB/RB), D-pad, right stick.
Axis indices can differ per device; adjust ``steer_axis``, ``accel_axis``,
``brake_axis`` in the extension config if needed.

Deadzones: ``deadzone`` for steer stick; ``accel_deadzone`` and ``brake_deadzone``
for normalized trigger input (0..1 after rest calibration).

Steering gain scales down with ``ego.velocity`` when ``steer_speed_scale`` is enabled:
full stick authority at or below ``steer_full_gain_below`` m/s, reduced toward
``steer_min_gain`` at ``ego_max_velocity``.
"""

from pydantic import Field

from avlite.c60_common.c68_settings_schema import SettingsSchema


class PluginSettingsSchema(SettingsSchema):
    device_index: int = Field(
        default=0,
        description="pygame joystick index (0 = first connected gamepad).",
    )
    deadzone: float = Field(
        default=0.02,
        description="Steer-axis deadzone; stick values below this are treated as zero.",
    )
    accel_deadzone: float = Field(
        default=0.02,
        description="Throttle deadzone on normalized 0..1 input (after rest calibration).",
    )
    brake_deadzone: float = Field(
        default=0.02,
        description="Brake deadzone on normalized 0..1 input (after rest calibration).",
    )
    steer_axis: int = Field(
        default=0,
        description="Axis for steering (Xbox: left stick X).",
    )
    accel_axis: int = Field(
        default=4,
        description="Axis for throttle (Xbox: right trigger RT; pygame range -1..1).",
    )
    brake_axis: int = Field(
        default=5,
        description="Axis for brake (Xbox: left trigger LT; pygame range -1..1).",
    )
    steer_invert: bool = Field(
        default=True,
        description="If true, push stick right steers right (negates axis before scaling).",
    )
    steer_speed_scale: bool = Field(
        default=True,
        description="Reduce max steer as ego velocity increases (less sensitive at high speed).",
    )
    steer_full_gain_below: float = Field(
        default=3.0,
        description="Velocity (m/s) at or below which steering uses full stick gain (1.0).",
    )
    steer_min_gain: float = Field(
        default=0.25,
        description="Minimum steer fraction at ego_max_velocity (0..1).",
    )
    fail_if_missing: bool = Field(
        default=False,
        description="Raise at init if pygame is missing or no gamepad at device_index.",
    )


# Settings singleton; filepath is assigned by the plugin loader from the directory name.
PluginSettings = PluginSettingsSchema()
