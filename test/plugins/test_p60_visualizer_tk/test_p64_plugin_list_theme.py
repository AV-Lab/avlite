"""Settings plugin lists follow the light and dark theme."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from avlite.plugins.p60_visualizer_tk.p64_setting_views import SettingWindow
from avlite.plugins.p60_visualizer_tk.p65_ui_lib import apply_ttk_theme


def _field_bg(widget: tk.Misc, *, dark: bool) -> str:
    style = ttk.Style(widget)
    frame_bg = style.lookup("TFrame", "background") or ("gray14" if dark else "white")
    return style.lookup("TEntry", "fieldbackground") or frame_bg


def _same(widget: tk.Misc, left: str, right: str) -> bool:
    return widget.winfo_rgb(left) == widget.winfo_rgb(right)


def test_plugin_lists_follow_dark_and_light_theme():
    root = tk.Tk()
    root.withdraw()
    try:
        apply_ttk_theme(root, dark=True)
        dark_mode = tk.BooleanVar(master=root, value=True)
        view = SettingWindow.__new__(SettingWindow)
        view.host = type("Host", (), {})()
        view.host.setting = type("Setting", (), {"p60_dark_mode": dark_mode})()
        view.window = root
        view.profile_note = tk.Text(root)
        view.listbox_default_plugins = tk.Listbox(root, bg="white", fg="black")
        view.listbox_community_plugins = tk.Listbox(root, bg="white", fg="black")

        view.apply_profile_note_theme()
        dark_field = _field_bg(root, dark=True)
        for box in (view.listbox_default_plugins, view.listbox_community_plugins):
            assert _same(box, box.cget("bg"), dark_field)
            assert not _same(box, box.cget("bg"), "white")

        apply_ttk_theme(root, dark=False)
        dark_mode.set(False)
        view.apply_profile_note_theme()
        light_field = _field_bg(root, dark=False)
        for box in (view.listbox_default_plugins, view.listbox_community_plugins):
            assert _same(box, box.cget("bg"), light_field)
    finally:
        root.destroy()
