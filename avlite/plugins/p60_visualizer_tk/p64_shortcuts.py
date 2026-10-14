"""Visualizer shortcut catalog. Defaults match the historical key bindings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Shortcut:
    id: str
    group: str
    label: str
    default: str
    bind_all: bool = False
    press_release: bool = False
    drive_guard: bool = False
    skip_text: bool = False
    settings_window: bool = False


SHORTCUTS: tuple[Shortcut, ...] = (
    Shortcut("quit", "App", "Quit", "Q", bind_all=True),
    Shortcut("toggle_shortcuts", "App", "Toggle shortcut panel", "S"),
    Shortcut("switch_profile", "App", "Switch profile", "F", bind_all=True),
    Shortcut("reload", "App", "Reload stack", "R", bind_all=True),
    Shortcut("open_settings", "App", "Open settings", "T"),
    Shortcut("save_settings", "App", "Save settings", "<Control-s>", settings_window=True),
    Shortcut("focus", "App", "Focus window", "<Escape>"),
    Shortcut("step_plan", "Plan", "Step plan", "n"),
    Shortcut("step_plan_back", "Plan", "Step plan back", "b"),
    Shortcut("replan", "Plan", "Replan", "r"),
    Shortcut("zoom_in", "Plan", "Zoom in", "<plus>"),
    Shortcut("zoom_out", "Plan", "Zoom out", "<minus>"),
    Shortcut("zoom_in_frenet", "Plan", "Zoom in Frenet", "<Control-plus>"),
    Shortcut("zoom_out_frenet", "Plan", "Zoom out Frenet", "<Control-minus>"),
    Shortcut("step_control", "Control", "Control step", "h"),
    Shortcut("align_control", "Control", "Align control", "i"),
    Shortcut("accelerate", "Control", "Accelerate", "w", drive_guard=True),
    Shortcut("steer_left", "Control", "Steer left", "a", press_release=True, drive_guard=True),
    Shortcut("steer_right", "Control", "Steer right", "d", press_release=True, drive_guard=True),
    Shortcut("decelerate", "Control", "Decelerate", "s", drive_guard=True),
    Shortcut("step_exec", "Execute", "Step execution", "c"),
    Shortcut("reset_exec", "Execute", "Reset execution", "t"),
    Shortcut("toggle_exec", "Execute", "Toggle execution", "x"),
    Shortcut("log_up", "Log", "Scroll log up", "k", skip_text=True, settings_window=True),
    Shortcut("log_down", "Log", "Scroll log down", "j", skip_text=True, settings_window=True),
    Shortcut("log_page_up", "Log", "Log half page up", "<Control-u>", settings_window=True),
    Shortcut("log_page_down", "Log", "Log half page down", "<Control-d>", settings_window=True),
    Shortcut("log_top", "Log", "Log top", "g", skip_text=True, settings_window=True),
    Shortcut("log_bottom", "Log", "Log bottom", "G", skip_text=True, settings_window=True),
    Shortcut("log_up_arrow", "Log", "Scroll log up (arrow)", "<Up>", drive_guard=True),
    Shortcut("log_down_arrow", "Log", "Scroll log down (arrow)", "<Down>", drive_guard=True),
    Shortcut("toggle_log_height", "Log", "Expand or collapse log", "E"),
    Shortcut("clear_log", "Log", "Clear log", "L"),
)

SHORTCUT_BY_ID: dict[str, Shortcut] = {entry.id: entry for entry in SHORTCUTS}

_MODIFIER_KEYS = frozenset({
    "Control_L", "Control_R", "Shift_L", "Shift_R",
    "Alt_L", "Alt_R", "Meta_L", "Meta_R", "Super_L", "Super_R",
})
_NAMED_KEYS = {
    "plus": "+",
    "minus": "-",
    "Escape": "Esc",
    "Up": "Up",
    "Down": "Down",
    "Left": "Left",
    "Right": "Right",
}


def sequence(entry: Shortcut, overrides: dict[str, str] | None = None) -> str:
    """Return the bound Tk sequence for *entry*, or its default."""
    custom = (overrides or {}).get(entry.id, "")
    if isinstance(custom, str) and custom:
        return custom
    return entry.default


def conflicting_id(entry_id: str, seq: str, overrides: dict[str, str] | None = None) -> str | None:
    """Return another shortcut id already bound to *seq*, if any."""
    for other in SHORTCUTS:
        if other.id != entry_id and sequence(other, overrides) == seq:
            return other.id
    return None


def event_sequence(event) -> str | None:
    """Turn a KeyPress into a Tk sequence. Modifier keys alone return None."""
    keysym = str(getattr(event, "keysym", "") or "")
    if not keysym or keysym in _MODIFIER_KEYS:
        return None
    state = int(getattr(event, "state", 0) or 0)
    control = bool(state & 0x4)
    shift = bool(state & 0x1)
    alt = bool(state & 0x8)
    if len(keysym) == 1 and keysym.isalpha():
        letter = keysym.upper() if shift else keysym.lower()
        mods = _mods(control, alt, shift=False)
        if mods:
            return "<" + "-".join([*mods, letter.lower()]) + ">"
        return letter
    mods = _mods(control, alt, shift)
    if mods:
        return "<" + "-".join([*mods, keysym]) + ">"
    if len(keysym) == 1:
        return keysym
    return f"<{keysym}>"


def display_sequence(seq: str) -> str:
    """Short label for a Tk sequence, such as ``Ctrl+S`` or ``+``."""
    if not (seq.startswith("<") and seq.endswith(">")):
        return seq
    inner = seq[1:-1]
    parts = inner.split("-")
    key = parts[-1]
    prefix = ""
    if "Control" in parts:
        prefix += "Ctrl+"
    if "Alt" in parts:
        prefix += "Alt+"
    if "Shift" in parts:
        prefix += "Shift+"
    shown = _NAMED_KEYS.get(key, key)
    if len(shown) == 1 and shown.isalpha():
        shown = shown.upper()
    return prefix + shown


def bind_sequences(entry: Shortcut, seq: str) -> tuple[str, ...]:
    """Sequences to bind. Steer uses press and release of the same key."""
    if not entry.press_release:
        return (seq,)
    key = seq[1:-1].split("-")[-1] if seq.startswith("<") and seq.endswith(">") else seq
    mods = ""
    if seq.startswith("<") and seq.endswith(">"):
        parts = seq[1:-1].split("-")[:-1]
        if parts:
            mods = "-".join(parts) + "-"
    return (f"<{mods}KeyPress-{key}>", f"<{mods}KeyRelease-{key}>")


def _mods(control: bool, alt: bool, shift: bool) -> list[str]:
    mods: list[str] = []
    if control:
        mods.append("Control")
    if alt:
        mods.append("Alt")
    if shift:
        mods.append("Shift")
    return mods
