import logging
from typing import Optional

from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c20_planning.c21_planning_model import LocalPlan
from avlite.c30_control.c32_control_strategy import ControlStrategy, ControlComand
from avlite.c60_common.c63_trajectory_tracker import TrajectoryTracker
from avlite.plugins.p30_controller_joystick.settings import PluginSettings

log = logging.getLogger(__name__)

try:
    import pygame
except ImportError:
    pygame = None


class JoystickController(ControlStrategy):
    def __init__(self, tj: Optional[TrajectoryTracker] = None):
        super().__init__(tj)
        self._joystick = None
        self._accel_rest = 0.0
        self._brake_rest = -1.0
        self._unavailable_reason: Optional[str] = None
        self._warned_on_control = False
        self._init_joystick()

    def _fail_or_log(self, reason: str, message: str) -> None:
        self._unavailable_reason = reason
        if PluginSettings.fail_if_missing:
            raise RuntimeError(message)
        log.error(message)

    def _init_joystick(self) -> None:
        if pygame is None:
            self._fail_or_log(
                "pygame",
                "JoystickController: pygame not installed; install with `pip install pygame`",
            )
            return
        try:
            pygame.init()
            pygame.joystick.init()
            device_index = PluginSettings.device_index
            if pygame.joystick.get_count() <= device_index:
                count = pygame.joystick.get_count()
                self._fail_or_log(
                    "no_device",
                    f"JoystickController: no gamepad at device_index={device_index} "
                    f"({count} device(s) connected)",
                )
                return
            self._joystick = pygame.joystick.Joystick(device_index)
            self._joystick.init()
            pygame.event.pump()
            self._accel_rest = self._joystick.get_axis(PluginSettings.accel_axis)
            self._brake_rest = self._joystick.get_axis(PluginSettings.brake_axis)
            log.info(
                "JoystickController: using '%s' (device_index=%s); "
                "trigger rest accel=%.4f brake=%.4f (release triggers on startup)",
                self._joystick.get_name(),
                device_index,
                self._accel_rest,
                self._brake_rest,
            )
        except RuntimeError:
            raise
        except Exception as e:
            self._fail_or_log(
                "init_error",
                f"JoystickController: error initializing gamepad: {e}",
            )

    def control(
        self,
        ego: EgoState,
        plan: Optional[LocalPlan] = None,
        control_dt: float = None,
    ) -> ControlComand:
        if self._joystick is None:
            if not self._warned_on_control:
                self._warned_on_control = True
                log.warning(
                    "JoystickController: no gamepad available (%s); returning zero command",
                    self._unavailable_reason or "unknown",
                )
            return ControlComand()

        pygame.event.pump()

        steer_raw = self._joystick.get_axis(PluginSettings.steer_axis)
        accel_raw = self._joystick.get_axis(PluginSettings.accel_axis)
        brake_raw = self._joystick.get_axis(PluginSettings.brake_axis)

        if abs(steer_raw) < PluginSettings.deadzone:
            steer_raw = 0

        accel = self._trigger_value(accel_raw, self._accel_rest, PluginSettings.accel_deadzone)
        brake = self._trigger_value(brake_raw, self._brake_rest, PluginSettings.brake_deadzone)

        sign = -1 if PluginSettings.steer_invert else 1
        velocity = ego.velocity if ego is not None else 0.0
        steer_gain = self._steer_speed_gain(velocity)
        steering = sign * steer_raw * self.ego_max_steering * steer_gain

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "Steer axis %s: %s (gain %.3f @ v=%.2f), Accel axis %s: %s (norm %s), "
                "Brake axis %s: %s (norm %s)",
                PluginSettings.steer_axis, steer_raw, steer_gain, velocity,
                PluginSettings.accel_axis, accel_raw, accel,
                PluginSettings.brake_axis, brake_raw, brake,
            )

        acceleration = accel * self.ego_max_acceleration
        braking = brake * self.ego_min_acceleration

        cmd = ControlComand(steer=steering, acceleration=acceleration + braking)
        self.cmd = cmd
        return cmd

    def reset(self):
        pass

    def _steer_speed_gain(self, velocity: float) -> float:
        if not PluginSettings.steer_speed_scale:
            return 1.0
        v = max(0.0, velocity)
        if v <= PluginSettings.steer_full_gain_below:
            return 1.0
        v_max = max(self.ego_max_velocity, PluginSettings.steer_full_gain_below + 1e-6)
        t = min(1.0, (v - PluginSettings.steer_full_gain_below) / (v_max - PluginSettings.steer_full_gain_below))
        return 1.0 - t * (1.0 - PluginSettings.steer_min_gain)

    @staticmethod
    def _trigger_value(raw: float, rest: float, deadzone: float) -> float:
        """Map trigger axis to 0..1 using rest position measured at init."""
        pressed = 1.0 if rest <= 0 else -1.0
        span = pressed - rest
        if abs(span) < 1e-6:
            return 0.0
        value = max(0.0, min(1.0, (raw - rest) / span))
        if value < deadzone:
            return 0.0
        return value
