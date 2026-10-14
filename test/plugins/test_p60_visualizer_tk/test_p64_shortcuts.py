"""Shortcut catalog resolution, without a Tk display."""

from types import SimpleNamespace

from avlite.plugins.p60_visualizer_tk.p64_shortcuts import (
    SHORTCUTS,
    conflicting_id,
    event_sequence,
    sequence,
)

# Sequences that were hardcoded before the catalog existed.
_TODAY = {
    "quit": "Q",
    "toggle_shortcuts": "S",
    "switch_profile": "F",
    "reload": "R",
    "open_settings": "T",
    "save_settings": "<Control-s>",
    "focus": "<Escape>",
    "step_plan": "n",
    "step_plan_back": "b",
    "replan": "r",
    "zoom_in": "<plus>",
    "zoom_out": "<minus>",
    "zoom_in_frenet": "<Control-plus>",
    "zoom_out_frenet": "<Control-minus>",
    "step_control": "h",
    "align_control": "i",
    "accelerate": "w",
    "steer_left": "a",
    "steer_right": "d",
    "decelerate": "s",
    "step_exec": "c",
    "reset_exec": "t",
    "toggle_exec": "x",
    "log_up": "k",
    "log_down": "j",
    "log_page_up": "<Control-u>",
    "log_page_down": "<Control-d>",
    "log_top": "g",
    "log_bottom": "G",
    "log_up_arrow": "<Up>",
    "log_down_arrow": "<Down>",
    "toggle_log_height": "E",
    "clear_log": "L",
}


def test_empty_overrides_use_todays_sequences():
    assert {entry.id: entry.default for entry in SHORTCUTS} == _TODAY
    assert {entry.id: sequence(entry, {}) for entry in SHORTCUTS} == _TODAY


def test_override_replaces_only_that_shortcut():
    overrides = {"quit": "Z"}
    by_id = {entry.id: entry for entry in SHORTCUTS}
    assert sequence(by_id["quit"], overrides) == "Z"
    assert sequence(by_id["replan"], overrides) == "r"


def test_duplicate_sequence_is_reported():
    assert conflicting_id("replan", "n", {}) == "step_plan"
    assert conflicting_id("replan", "r", {}) is None
    assert conflicting_id("replan", "n", {"step_plan": "m"}) is None
    assert conflicting_id("step_plan", "Z", {"quit": "Z"}) == "quit"


def test_event_sequence_maps_ctrl_and_letters():
    assert event_sequence(SimpleNamespace(keysym="s", state=0x4)) == "<Control-s>"
    assert event_sequence(SimpleNamespace(keysym="G", state=0x1)) == "G"
    assert event_sequence(SimpleNamespace(keysym="g", state=0)) == "g"
    assert event_sequence(SimpleNamespace(keysym="plus", state=0x4)) == "<Control-plus>"
    assert event_sequence(SimpleNamespace(keysym="Control_L", state=0x4)) is None
