"""Keyboard teleop controller (c36).

Maps held keys (defaults: WASD and arrows) to Ackermann steer/accel.
Bindings and magnitudes live in ``ControlSettings`` (``c36_*``) and are read
live each :meth:`control` tick.
"""

from __future__ import annotations

import logging
import threading
import weakref
from typing import Iterable, Optional

import numpy as np

from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c20_planning.c21_planning_model import GlobalPlan, LocalPlan
from avlite.c30_control.c31_control_model import ControlCommand
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c30_control.c39_settings import ControlSettings, ControlSettingsSchema
from avlite.c50_common.c51_capabilities import StackCapability
from avlite.c50_common.c52_world_sensor_datatypes import SensorFrame
from avlite.c50_common.c54_trajectory_tracker import TrajectoryTracker

log = logging.getLogger(__name__)

_KEY_ALIASES = {
    "arrowup": "up",
    "arrowdown": "down",
    "arrowleft": "left",
    "arrowright": "right",
}


def normalize_key(key: str) -> str:
    """Lowercase a key name and fold arrow aliases (``arrowup`` → ``up``)."""
    name = str(key).strip().lower()
    if name.startswith("key."):
        name = name[4:]
    return _KEY_ALIASES.get(name, name)


def _pynput_key_name(key) -> str | None:
    """Return a normalized name for a pynput ``Key`` / ``KeyCode``, or ``None``."""
    name = getattr(key, "name", None)
    if name:
        return normalize_key(name)
    char = getattr(key, "char", None)
    if char:
        return normalize_key(char)
    return None


def keyboard_drive_owns(controller: object, key: str) -> bool:
    """True when *controller* is a live ``KeyboardController`` that maps *key*."""
    if not isinstance(controller, KeyboardController):
        return False
    return normalize_key(key) in controller.mapped_keys()


class KeyboardController(ControlStrategy):
    """Teleop: held keys → Ackermann ``steer`` / ``acceleration``."""

    stack_requirements = frozenset()
    stack_capabilities = frozenset({StackCapability.CONTROL})

    _listener = None
    _active: weakref.ReferenceType[KeyboardController] | None = None

    def __init__(
        self,
        tj: Optional[TrajectoryTracker] = None,
        setting: ControlSettingsSchema = ControlSettings,
        listen: bool = True,
    ):
        super().__init__(tj)
        self._setting = setting
        self._held: set[str] = set()
        self._lock = threading.Lock()
        if listen:
            type(self)._attach(self)

    @classmethod
    def _attach(cls, instance: KeyboardController) -> None:
        cls._active = weakref.ref(instance)
        if cls._listener is not None:
            return
        try:
            from pynput import keyboard
        except ImportError:
            log.warning(
                "pynput not installed; KeyboardController needs press()/release() "
                "or pip install pynput"
            )
            return
        try:
            listener = keyboard.Listener(on_press=cls._on_press, on_release=cls._on_release)
            listener.start()
            cls._listener = listener
        except Exception as exc:
            log.warning("Could not start keyboard listener: %s", exc)

    @classmethod
    def _on_press(cls, key) -> None:
        name = _pynput_key_name(key)
        ctrl = cls._active() if cls._active is not None else None
        if name and ctrl is not None:
            ctrl.press(name)

    @classmethod
    def _on_release(cls, key) -> None:
        name = _pynput_key_name(key)
        ctrl = cls._active() if cls._active is not None else None
        if name and ctrl is not None:
            ctrl.release(name)

    def press(self, key: str) -> None:
        name = normalize_key(key)
        if not name:
            return
        with self._lock:
            self._held.add(name)

    def release(self, key: str) -> None:
        name = normalize_key(key)
        if not name:
            return
        with self._lock:
            self._held.discard(name)

    def mapped_keys(self) -> set[str]:
        """Normalized names currently bound on any drive axis."""
        s = self._setting
        out: set[str] = set()
        for names in (
            s.c36_key_accel,
            s.c36_key_brake,
            s.c36_key_steer_left,
            s.c36_key_steer_right,
        ):
            out.update(normalize_key(n) for n in names if n)
        return out

    def _held_copy(self) -> set[str]:
        with self._lock:
            return set(self._held)

    def _mapped(self, names: Iterable[str]) -> set[str]:
        return {normalize_key(n) for n in names if n}

    def _axis(self, held: set[str], positive: Iterable[str], negative: Iterable[str]) -> int:
        plus = bool(held & self._mapped(positive))
        minus = bool(held & self._mapped(negative))
        if plus and not minus:
            return 1
        if minus and not plus:
            return -1
        return 0

    def control(
        self,
        ego: EgoState,
        plan: GlobalPlan | LocalPlan | None = None,
        control_dt: float | None = None,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
    ) -> ControlCommand:
        held = self._held_copy()
        s = self._setting
        accel_dir = self._axis(held, s.c36_key_accel, s.c36_key_brake)
        steer_dir = self._axis(held, s.c36_key_steer_left, s.c36_key_steer_right)
        acc = accel_dir * float(s.c36_keyboard_acceleration)
        steer = steer_dir * float(s.c36_keyboard_steering)
        acc = float(np.clip(acc, s.c32_ego_min_acceleration, s.c32_ego_max_acceleration))
        steer = float(np.clip(steer, s.c32_ego_min_steering, s.c32_ego_max_steering))
        cmd = ControlCommand(steer=steer, acceleration=acc)
        self.cmd = cmd
        return cmd

    def reset(self) -> None:
        with self._lock:
            self._held.clear()
        self.cmd = ControlCommand()
        self.cte_steer = 0
        self.cte_velocity = 0
