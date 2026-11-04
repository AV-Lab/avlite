"""Keyboard shortcut catalog (mirrors the Tk visualizer defaults)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Shortcut:
    id: str
    group: str
    label: str
    default: str  # Qt key sequence string, e.g. "Q", "Ctrl+S"


SHORTCUTS: tuple[Shortcut, ...] = (
    Shortcut("quit", "App", "Quit", "Q"),
    Shortcut("toggle_shortcuts", "App", "Toggle shortcut panel", "S"),
    Shortcut("switch_profile", "App", "Switch profile", "F"),
    Shortcut("reload", "App", "Reload stack", "R"),
    Shortcut("step_plan", "Plan", "Step plan", "N"),
    Shortcut("step_plan_back", "Plan", "Step plan back", "B"),
    Shortcut("replan", "Plan", "Local replan", "Shift+R"),
    Shortcut("zoom_in", "Plan", "Zoom in", "+"),
    Shortcut("zoom_out", "Plan", "Zoom out", "-"),
    Shortcut("step_control", "Control", "Control step", "H"),
    Shortcut("align_control", "Control", "Align control", "I"),
    Shortcut("accelerate", "Control", "Accelerate", "W"),
    Shortcut("steer_left", "Control", "Steer left", "A"),
    Shortcut("steer_right", "Control", "Steer right", "D"),
    Shortcut("decelerate", "Control", "Decelerate", "S"),  # conflict with toggle; drive when not shortcut mode letter alone handled in window
    Shortcut("step_exec", "Execute", "Step execution", "C"),
    Shortcut("reset_exec", "Execute", "Reset execution", "T"),
    Shortcut("toggle_exec", "Execute", "Toggle execution", "X"),
    Shortcut("clear_log", "Log", "Clear log", "L"),
)

SHORTCUT_BY_ID = {s.id: s for s in SHORTCUTS}
