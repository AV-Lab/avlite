"""Unit tests for KeyboardController (avlite.c30_control.c36_keyboard)."""

import pytest

from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c30_control.c36_keyboard import KeyboardController, keyboard_drive_owns, normalize_key
from avlite.c30_control.c39_settings import ControlSettingsSchema
from avlite.c50_common.c51_capabilities import StackCapability, satisfies_requirements


def _kb_settings(**overrides) -> ControlSettingsSchema:
    defaults = {
        "c36_key_accel": ["w", "up"],
        "c36_key_brake": ["s", "down"],
        "c36_key_steer_left": ["a", "left"],
        "c36_key_steer_right": ["d", "right"],
        "c36_keyboard_acceleration": 3.0,
        "c36_keyboard_steering": 0.7,
    }
    defaults.update(overrides)
    return ControlSettingsSchema(**defaults)


def _ego() -> EgoState:
    return EgoState(x=0.0, y=0.0, theta=0.0, velocity=0.0)


class TestNormalizeKey:
    def test_aliases_and_case(self):
        assert normalize_key("W") == "w"
        assert normalize_key("arrowup") == "up"
        assert normalize_key("Key.up") == "up"


class TestKeyboardController:
    def test_default_map_accel_and_steer(self):
        ctrl = KeyboardController(setting=_kb_settings(), listen=False)
        ego = _ego()

        ctrl.press("w")
        assert ctrl.control(ego).acceleration == pytest.approx(3.0)
        ctrl.release("w")
        ctrl.press("up")
        assert ctrl.control(ego).acceleration == pytest.approx(3.0)
        ctrl.release("up")

        ctrl.press("s")
        assert ctrl.control(ego).acceleration == pytest.approx(-3.0)
        ctrl.release("s")
        ctrl.press("down")
        assert ctrl.control(ego).acceleration == pytest.approx(-3.0)
        ctrl.release("down")

        ctrl.press("a")
        assert ctrl.control(ego).steer == pytest.approx(0.7)
        ctrl.release("a")
        ctrl.press("left")
        assert ctrl.control(ego).steer == pytest.approx(0.7)
        ctrl.release("left")

        ctrl.press("d")
        assert ctrl.control(ego).steer == pytest.approx(-0.7)
        ctrl.release("d")
        ctrl.press("right")
        assert ctrl.control(ego).steer == pytest.approx(-0.7)

    def test_remap_is_honored(self):
        setting = _kb_settings(c36_key_accel=["i"], c36_key_brake=[], c36_key_steer_left=[], c36_key_steer_right=[])
        ctrl = KeyboardController(setting=setting, listen=False)
        ego = _ego()
        ctrl.press("w")
        assert ctrl.control(ego).acceleration == pytest.approx(0.0)
        ctrl.release("w")
        ctrl.press("i")
        assert ctrl.control(ego).acceleration == pytest.approx(3.0)

    def test_release_and_reset_zero(self):
        ctrl = KeyboardController(setting=_kb_settings(), listen=False)
        ego = _ego()
        ctrl.press("w")
        ctrl.press("a")
        ctrl.release("w")
        ctrl.release("a")
        cmd = ctrl.control(ego)
        assert cmd.acceleration == pytest.approx(0.0)
        assert cmd.steer == pytest.approx(0.0)

        ctrl.press("w")
        ctrl.reset()
        cmd = ctrl.control(ego)
        assert cmd.acceleration == pytest.approx(0.0)
        assert cmd.steer == pytest.approx(0.0)

    def test_opposing_keys_cancel(self):
        ctrl = KeyboardController(setting=_kb_settings(), listen=False)
        ego = _ego()
        ctrl.press("w")
        ctrl.press("s")
        ctrl.press("a")
        ctrl.press("d")
        cmd = ctrl.control(ego)
        assert cmd.acceleration == pytest.approx(0.0)
        assert cmd.steer == pytest.approx(0.0)

    def test_stack_requirements_empty(self):
        ctrl = KeyboardController(listen=False)
        assert ctrl.stack_requirements == frozenset()
        assert StackCapability.CONTROL in ctrl.stack_capabilities
        assert satisfies_requirements(ctrl.stack_requirements, set())

    def test_keyboard_drive_owns(self):
        ctrl = KeyboardController(setting=_kb_settings(), listen=False)
        assert keyboard_drive_owns(ctrl, "W")
        assert keyboard_drive_owns(ctrl, "arrowup")
        assert not keyboard_drive_owns(ctrl, "x")
        assert not keyboard_drive_owns(None, "w")
