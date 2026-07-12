from __future__ import annotations

import ctypes
import logging
import os
import platform
import webbrowser
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import ttk

try:
    from ttkthemes import ThemedStyle
except ImportError:
    ThemedStyle = None  # type: ignore[misc, assignment]

from avlite.c10_perception.c11_perception_model import HDMap, RaceMap
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c50_common.c51_capabilities import (
    AnyOf,
    CapabilityGroup,
    MayUse,
    StackCapability,
    WorldCapability,
    combine_stack_requirements,
)
from avlite.c60_apps.c64_settings_schema import PlainBinder, field_tooltip_text
from avlite.c60_apps.c68_paths import DataPaths

log = logging.getLogger(__name__)

class ValueGauge(ttk.Frame):
    def __init__(
        self,
        parent,
        name: str = "",
        min_value: float = 0,
        max_value: float = 100,
        variable=None,
        height=12,
        dpi_scale: float | None = None,
        **kwargs,
    ):
        super().__init__(parent, **kwargs)
        if dpi_scale is None:
            dpi_scale = getattr(parent, "_dpi_scale", None)
        if dpi_scale is None:
            dpi_scale = DpiScale.for_widget(parent)

        self.min_value = min_value
        self.max_value = max_value
        self.current_value = 0
        self.marker_value = 0
        self.variable = variable
        self.pack_propagate(False)
        self.font = ("Helvetica", 7, "bold")
        self._old_value = 0  # used to not draw if value is same
        gauge_height = DpiScale.scaled(height, dpi_scale)

        if name != "":
            tk.Label(self, text=name, font=self.font).pack(side=tk.LEFT, padx=0, pady=1)

        min_label = tk.Label(self, text=f"{min_value:+.2f}", font=self.font)
        min_label.pack(side=tk.LEFT, padx=0, pady=1)

        max_label = tk.Label(self, text=f"{max_value:+.2f}", font=self.font)
        max_label.pack(side=tk.RIGHT, padx=0, pady=1)

        self.canvas = tk.Canvas(self, height=gauge_height, bg="gray", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)

        total_height = gauge_height
        self.config(height=total_height)

        self.bind("<Configure>", lambda e: self.__draw())

        if self.variable is not None:
            self.variable.trace_add("write", self.__variable_changed)

        
        # Schedule a delayed draw to ensure widget dimensions are set
        self.after(100, self.__draw)
    
    def __variable_changed(self, *args):
        self.set_value(self.variable.get())

    def set_value(self, value):
        if value == self.current_value:
            return  # Skip if no change
        self.current_value = value
        self.after_idle(self.__draw)  # Schedule redraw for next idle time

    def set_marker(self, value):
        if value == self.marker_value:
            return  # Skip if no change
        self.marker_value = value
        self.after_idle(self.__draw)  # Schedule redraw for next idle time
        # self.__draw()

    def __draw(self):
        # Avoid unnecessary redraws
        if not self.winfo_ismapped():
            return
            
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            return
            
        self.canvas.delete("all")

        # Bound values to prevent drawing errors
        bounded_marker = max(self.min_value, min(self.max_value, self.marker_value))
        bounded_current = max(self.min_value, min(self.max_value, self.current_value))
        
        # Calculate marker position
        marker_x = ((bounded_marker - self.min_value) / (self.max_value - self.min_value)) * width
        self.canvas.create_line(marker_x, 0, marker_x, height, fill="red", width=2, tags="marker")

        # Draw value text with black background highlight
        text = f"{bounded_current:+4.2f}"
        
        # Calculate text position
        x = ((bounded_current - self.min_value) / (self.max_value - self.min_value)) * width
        y = height / 2

        # Tint the value box by magnitude: calm near zero, warmer as it deviates.
        half_span = max((self.max_value - self.min_value) / 2.0, 1e-6)
        magnitude = abs(bounded_current) / half_span
        if magnitude <= 0.1:
            box_fill = "#1b5e20"  # green: within deadband
        elif magnitude >= 0.6:
            box_fill = "#9d0006"  # red: large deviation
        else:
            box_fill = "#d65d0e"  # orange: moderate deviation

        # Create rectangle first (will be behind text)
        text_id = self.canvas.create_text(x, y, text=text, fill="white", font=self.font, anchor="center", tags="value")
        bbox = self.canvas.bbox(text_id)
        # self.canvas.coords(text_id, x, y + 2)  # Move text down within the box
        self.canvas.create_rectangle(bbox, fill=box_fill, tags="bg")
        self.canvas.tag_raise("value")
        self.canvas.tag_raise(text_id)


class ThemedInputDialog:
    def __init__(self, parent, title, prompt, initial=""):
        self.result = None

        # Create the dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.transient(parent)
        s = DpiScale.for_widget(self.dialog, parent=parent)
        self.dialog.minsize(DpiScale.scaled(300, s), DpiScale.scaled(100, s))
        
        self.dialog.bind("<Escape>", lambda e: self.on_cancel())  # Bind Escape key to cancel

        
        
        frame = ttk.Frame(self.dialog)
        frame.pack(expand=True, fill=tk.BOTH)

        top_frame = ttk.Frame(frame)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Add prompt and entry field
        ttk.Label(top_frame, text=prompt).pack(side=tk.LEFT, padx=10)
        self.entry = ttk.Entry(top_frame, width=max(10, DpiScale.scaled(20, s)))
        self.entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10, pady=5)
        if initial:
            self.entry.insert(0, initial)
            self.entry.select_range(0, tk.END)
        self.entry.focus_set()
        
        # Add button frame
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Button(btn_frame, text="OK", command=self.on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.on_cancel).pack(side=tk.RIGHT , padx=5)
        
        # Make dialog modal
        self.dialog.grab_set()
        self.dialog.wait_window()
    
    def on_ok(self):
        self.result = self.entry.get()
        self.dialog.destroy()
    
    def on_cancel(self):
        self.dialog.destroy()

class ThemedTwoInputDialog:
    def __init__(self, parent, title, prompt1="", prompt2="", initial1="", initial2=""):
        self.result = None

        # Create the dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.transient(parent)
        s = DpiScale.for_widget(self.dialog, parent=parent)
        self.dialog.minsize(DpiScale.scaled(360, s), DpiScale.scaled(140, s))
        
        self.dialog.bind("<Escape>", lambda e: self.on_cancel())  # Bind Escape key to cancel
        
        frame = ttk.Frame(self.dialog)
        # frame.pack(expand=True, fill=tk.BOTH)
        frame.grid(row=0, column=0, sticky="nsew")
        self.dialog.grid_rowconfigure(0, weight=1)
        self.dialog.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=0)  # Entry row doesn't expand
        frame.grid_rowconfigure(1, weight=0)  # Entry row doesn't expand
        frame.grid_rowconfigure(2, weight=1)  # Button row expands to push buttons down
        frame.grid_columnconfigure(1, weight=1)


        # Add prompt and entry field
        ttk.Label(frame, text=prompt1).grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        self.first_entry = ttk.Entry(frame)
        self.first_entry.grid(row=0, column=1, padx=10, pady=5, sticky=tk.EW)
        
        self.first_entry.insert(0, initial1)
        
        ttk.Label(frame, text=prompt2).grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        self.second_entry = ttk.Entry(frame, width=max(20, DpiScale.scaled(30, s)))
        self.second_entry.grid(row=1, column=1, padx=10, pady=5, sticky=tk.EW)
        self.second_entry.insert(0, initial2)
        
        
        ttk.Button(frame, text="OK", command=self.on_ok).grid(row=2, column=0, padx=5, pady=5, sticky="sw")
        ttk.Button(frame, text="Cancel", command=self.on_cancel).grid(row=2, column=1, padx=5, pady=5, sticky="se")
        
        self.dialog.transient(parent)
        self.dialog.update_idletasks()
        self.dialog.deiconify()  # Ensure window is visible
        self.dialog.wait_visibility()  # Wait until window is visible
        self.dialog.grab_set()
        self.dialog.wait_window()
    
    def on_ok(self):
        self.result = (self.first_entry.get(), self.second_entry.get())
        self.dialog.destroy()
    
    def on_cancel(self):
        self.dialog.destroy()


class ThemedListPickerDialog:
    """Modal list picker; returns selected item text or None."""

    def __init__(
        self,
        parent,
        title: str,
        items: list[str],
        *,
        initial: str | None = None,
        readonly: bool = False,
    ):
        self.result: str | None = None
        self.readonly = readonly
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.transient(parent)
        s = DpiScale.for_widget(self.dialog, parent=parent)
        self.dialog.bind("<Escape>", lambda _e: self.on_cancel())

        frame = ttk.Frame(self.dialog)
        frame.pack(fill=tk.BOTH, expand=True)

        list_frame = ttk.Frame(frame)
        list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        _max_visible_rows = 12
        listbox_height = max(1, min(_max_visible_rows, len(items)))
        self.listbox = tk.Listbox(
            list_frame,
            height=listbox_height,
            selectmode=tk.SINGLE,
            exportselection=False,
            width=max(30, DpiScale.scaled(40, s)),
        )
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        if len(items) > _max_visible_rows:
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for item in items:
            self.listbox.insert(tk.END, item)

        if initial is not None and initial in items:
            idx = items.index(initial)
            self.listbox.selection_set(idx)
            self.listbox.see(idx)

        if not readonly:
            self.listbox.bind("<Double-Button-1>", lambda _e: self.on_ok())
            self.listbox.bind("<Return>", lambda _e: self.on_ok())

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=4)
        if readonly:
            ttk.Button(btn_frame, text="Close", command=self.on_cancel).pack(side=tk.RIGHT, padx=4)
        else:
            ttk.Button(btn_frame, text="OK", command=self.on_ok).pack(side=tk.LEFT, padx=4)
            ttk.Button(btn_frame, text="Cancel", command=self.on_cancel).pack(side=tk.RIGHT, padx=4)

        self.dialog.update_idletasks()
        self.dialog.minsize(self.dialog.winfo_reqwidth(), self.dialog.winfo_reqheight())
        self.dialog.resizable(False, False)
        self.dialog.deiconify()
        self.dialog.wait_visibility()
        self.dialog.grab_set()
        self.dialog.wait_window()

    def on_ok(self):
        if not self.readonly:
            sel = self.listbox.curselection()
            if sel:
                self.result = self.listbox.get(sel[0])
        self.dialog.destroy()

    def on_cancel(self):
        self.dialog.destroy()


class ThemedReadOnlyTwoFieldDialog:
    """Show two read-only label/value pairs (view-only)."""

    def __init__(
        self,
        parent,
        title: str,
        label1: str = "",
        label2: str = "",
        value1: str = "",
        value2: str = "",
        github_url: str | None = None,
    ):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.transient(parent)
        s = DpiScale.for_widget(self.dialog, parent=parent)
        self.dialog.minsize(DpiScale.scaled(360, s), DpiScale.scaled(120, s))
        self.dialog.bind("<Escape>", lambda _e: self.dialog.destroy())

        frame = ttk.Frame(self.dialog)
        frame.grid(row=0, column=0, sticky="nsew")
        self.dialog.grid_rowconfigure(0, weight=1)
        self.dialog.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        ttk.Label(frame, text=label1).grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        ttk.Label(frame, text=value1, wraplength=DpiScale.scaled(280, s)).grid(
            row=0, column=1, padx=10, pady=10, sticky=tk.W
        )
        ttk.Label(frame, text=label2).grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        ttk.Label(frame, text=value2, wraplength=DpiScale.scaled(280, s)).grid(
            row=1, column=1, padx=10, pady=10, sticky=tk.W
        )
        footer = ttk.Frame(frame)
        footer.grid(row=2, column=0, columnspan=2, padx=5, pady=10, sticky=tk.E)
        if github_url:
            btn = ttk.Button(
                footer, text="Open on GitHub", command=lambda: webbrowser.open(github_url)
            )
            btn.pack(side=tk.LEFT, padx=(0, 6))
            HoverTooltip.attach(btn, BUTTON_TOOLTIPS["cp_github"])
        ttk.Button(footer, text="OK", command=self.dialog.destroy).pack(side=tk.RIGHT)

        self.dialog.transient(parent)
        self.dialog.update_idletasks()
        self.dialog.deiconify()
        self.dialog.wait_visibility()
        self.dialog.grab_set()
        self.dialog.wait_window()


class HoverTooltip:
    """Show *text* in a small popup while the pointer is over *widget*."""

    def __init__(self, widget: tk.Widget, text: str, *, delay_ms: int = 400) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._tip: tk.Toplevel | None = None
        self._after_id: str | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    @classmethod
    def attach(cls, widget, text: str) -> HoverTooltip | None:
        if text:
            tooltip = cls(widget, text)
            widget._hover_tooltip = tooltip
            return tooltip
        return None

    @classmethod
    def attach_schema(cls, widget, settings_cls, field_name: str) -> None:
        cls.attach(widget, field_tooltip_text(settings_cls, field_name) or "")

    @classmethod
    def update_schema(cls, widget, settings_cls, field_name: str) -> None:
        text = field_tooltip_text(settings_cls, field_name) or ""
        tooltip = getattr(widget, "_hover_tooltip", None)
        if tooltip is not None:
            tooltip.text = text
        else:
            cls.attach(widget, text)

    @classmethod
    def attach_capability(cls, widget, cap) -> HoverTooltip | None:
        return cls.attach(widget, CAPABILITY_TOOLTIPS.get(cap, cap.name))

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self) -> None:
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self) -> None:
        self._after_id = None
        if self._tip is not None:
            return

        s = DpiScale.for_widget(self.widget)
        x = self.widget.winfo_rootx() + DpiScale.scaled(20, s)
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + DpiScale.scaled(4, s)
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tip,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            wraplength=DpiScale.scaled(360, s),
            padx=DpiScale.scaled(6, s),
            pady=DpiScale.scaled(4, s),
        )
        label.pack()
        self._tip = tip

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


CAPABILITY_TOOLTIPS: dict = {
    WorldCapability.CAMERA_RGB: "RGB camera image feed from the world.",
    WorldCapability.CAMERA_DEPTH: "Depth camera image feed from the world.",
    WorldCapability.LIDAR_3D: "3D LiDAR point cloud from the world.",
    WorldCapability.LIDAR_2D: "2D LiDAR scan from the world.",
    WorldCapability.RADAR: "Radar returns from the world.",
    WorldCapability.WHEEL_ENCODER: "Wheel encoder odometry from the world.",
    WorldCapability.IMU: "Inertial measurement unit data from the world.",
    WorldCapability.GNSS: "GNSS / GPS receiver data from the world.",
    StackCapability.DETECTION: "Ground-truth object detections provided by the world.",
    StackCapability.TRACKING: "Ground-truth object tracks provided by the world.",
    StackCapability.PREDICTION: "Ground-truth agent trajectory predictions provided by the world.",
    StackCapability.LOCALIZATION: "Ground-truth ego localization provided by the world.",
    StackCapability.MAP_HD: "HD / OpenDRIVE map provided by a mapping module.",
    StackCapability.MAP_RACE_TRACK: "Race-track corridor map provided by a mapping module.",
    StackCapability.SLAM: "Ground-truth simultaneous localization and mapping from the world.",
    StackCapability.LOCAL_PLAN: "Ground-truth local plan provided by the world.",
    StackCapability.GLOBAL_PLAN: "Ground-truth global plan provided by the world.",
    StackCapability.CONTROL: "Ground-truth control commands provided by the world.",
}


BUTTON_TOOLTIPS: dict[str, str] = {
    # Toolbar
    "toolbar_settings": "Open settings to edit profiles, plugins, and visualization options.",
    "toolbar_plugins": "Browse and install community plugins from GitHub.",
    "toolbar_reload_stack": "Rebuild the perception, planning, control, and execution stack with current settings.",
    "toolbar_reset_settings": "Reload configuration from disk and discard unsaved UI changes.",
    "toolbar_save_settings": "Save the current settings to the active profile YAML files.",
    # Execution
    "exec_start": "Run the simulation loop continuously at the configured rates.",
    "exec_stop": "Stop the loop and halt the world bridge.",
    "exec_step": "Advance one execution tick without continuous run.",
    "exec_reset": "Reset the executer and world to the initial state.",
    # Planning
    "plan_global_replan": "Recompute the global route from the map and planner.",
    "plan_save_global": "Save the current global plan to a JSON file.",
    "plan_set_waypoint": "Jump the local planner to the waypoint index in the field.",
    "plan_wp_back": "Move to the previous waypoint on the global plan.",
    "plan_step": "Advance the local planner to the next waypoint.",
    "plan_align": "Snap the planned path to the current ego pose.",
    "plan_local_replan": "Replan the local trajectory from the current waypoint.",
    # Control
    "control_step": "Run one control cycle and update the gauges.",
    "control_align": "Re-align the controller with the current ego state.",
    "control_steer_left": "Nudge steering left (manual override).",
    "control_steer_right": "Nudge steering right (manual override).",
    "control_accel": "Apply a short acceleration pulse (manual override).",
    "control_decel": "Apply a short brake pulse (manual override).",
    # Log
    "log_clear": "Clear the log output.",
    "log_copy": "Copy log text to the clipboard.",
    "log_toggle_height": "Expand or collapse the log panel.",
    # Settings window — profiles
    "profile_new": "Create a new execution profile folder.",
    "profile_delete": "Delete the selected profile (the default profile is protected).",
    "profile_save": "Save all settings into the selected profile.",
    "profile_export": "Export the profile as a zip archive.",
    "profile_import": "Import a profile from a zip archive.",
    "profile_rename": "Rename the selected profile.",
    "profile_reset_all": "Reset every module setting to source-code defaults.",
    "profile_reset_non_exec": "Reset all settings except execution to defaults.",
    "section_reset_stack": "Reset this layer's settings to source-code defaults.",
    "section_reset_plugin": "Reset this plugin's settings to source-code defaults.",
    # Settings window — plugins
    "plugins_reset_builtin": "Restore the built-in plugin list to defaults.",
    "plugins_remove_builtin": (
        "Remove the selected built-in plugin entry "
        "(the app hosting this settings window cannot be removed)."
    ),
    "plugins_reset_community": "Sync community plugin list with installed packages.",
    "plugins_add": "Add a community plugin entry manually.",
    "plugins_remove_community": "Remove the selected community plugin entry.",
    "plugins_browse": "Open the community plugins browser.",
    "settings_close": "Close the settings window without saving.",
    "settings_save": "Save profile and close the settings window.",
    "edit_repo_configs": (
        "Write core stack and built-in plugin YAML to repository configs/ "
        "instead of user data. Community and member plugin settings always "
        "stay in your user config directory."
    ),
    # Community plugins
    "cp_refresh": "Refresh the plugin list from the registry.",
    "cp_install": "Install the selected plugin.",
    "cp_add_profile": "Add the installed plugin to the active profile.",
    "cp_uninstall": "Uninstall the selected plugin.",
    "cp_update": "Update the selected plugin to the latest version.",
    "cp_update_all": "Update all installed plugins that have updates.",
    "cp_github": "Open the plugin repository on GitHub.",
    "cp_open_folder": "Open the plugin install folder in the file manager.",
    "cp_close": "Close this window.",
    "cp_sign_in": "Sign in with GitHub to browse member-only plugins.",
    "cp_sign_out": "Sign out of GitHub and clear saved credentials.",
    "cp_sign_in_browser": "Open GitHub in your browser to authorize AVLite.",
    "cp_copy_code": "Copy the GitHub sign-in code to the clipboard.",
    "cp_show_installed": (
        "Show only Installed plugins (downloaded locally, not in the active profile)."
    ),
    "cp_show_active": (
        "Show only Active plugins (registered in the active profile and ready to load)."
    ),
}


class TkSettingsBinder:
    """Read/write settings backed by ``tk.Variable`` attributes."""

    def get_value(self, setting, attr_name: str):
        attr_value = getattr(setting if not isinstance(setting, type) else setting, attr_name)
        if isinstance(attr_value, tk.Variable):
            return attr_value.get()
        return PlainBinder().get_value(setting, attr_name)

    def set_value(self, setting, attr_name: str, value) -> None:
        attr_value = getattr(setting if not isinstance(setting, type) else setting, attr_name)
        if isinstance(attr_value, tk.BooleanVar):
            attr_value.set(bool(value))
        elif isinstance(attr_value, tk.IntVar):
            attr_value.set(int(value))
        elif isinstance(attr_value, tk.DoubleVar):
            attr_value.set(float(value))
        elif isinstance(attr_value, tk.Variable):
            attr_value.set(value)
        else:
            PlainBinder().set_value(setting, attr_name, value)


def apply_ttk_theme(root: tk.Misc, *, dark: bool) -> None:
    """Apply equilux (dark) or default (light) ttk theme and Tk option_add overrides."""
    if dark:
        root.configure(bg="gray14")
        try:
            if ThemedStyle is None:
                raise ImportError("ttkthemes not available")
            style = ThemedStyle(root)
            style.set_theme("equilux")
            style.configure("Big.TLabel", font=("Arial", 16, "bold"))
            style.configure("TLabelframe.Label", font=("Arial", 11, "bold"))
            gruvbox_red = "#9d0006"
            gruvbox_orange = "#d65d0e"

            style.layout(
                "Start.TButton",
                [("Button.border", {"sticky": "nswe", "children": [
                    ("Button.padding", {"sticky": "nswe", "children": [
                        ("Button.label", {"sticky": "nswe"})
                    ]})
                ]})],
            )
            style.layout(
                "Stop.TButton",
                [("Button.border", {"sticky": "nswe", "children": [
                    ("Button.padding", {"sticky": "nswe", "children": [
                        ("Button.label", {"sticky": "nswe"})
                    ]})
                ]})],
            )
            style.configure("Start.TButton", background=gruvbox_orange, foreground="white")
            style.configure("Stop.TButton", background=gruvbox_red, foreground="white")
            style.map(
                "Start.TButton",
                background=[("active", "#ff8800")],
                foreground=[("active", "white")],
            )
            style.map(
                "Stop.TButton",
                background=[("active", "#ff4444")],
                foreground=[("active", "white")],
            )
        except ImportError:
            log.error("Please install ttkthemes to use dark mode.")
            return

        root.option_add("*Listbox.background", "#222222")
        root.option_add("*Listbox.foreground", "#ffffff")
        root.option_add("*Listbox.selectBackground", "#444444")
        root.option_add("*Listbox.selectForeground", "#dddddd")
        root.option_add("*Listbox.highlightBackground", "#1a1a1a")
        root.option_add("*Listbox.highlightColor", "#333333")
        root.option_add("*Listbox.borderWidth", 1)
        root.option_add("*selectBackground", "#666699")
        root.option_add("*selectForeground", "#ffffff")
        root.option_add("*Entry.selectBackground", "#444466")
        root.option_add("*Entry.selectForeground", "#cccccc")
        root.option_add("*Text.selectBackground", "#444466")
        root.option_add("*Text.selectForeground", "#cccccc")
        style = ttk.Style(root)
        style.map(
            "TEntry",
            selectbackground=[("!disabled", "#444466")],
            selectforeground=[("!disabled", "#cccccc")],
        )
    else:
        root.configure(bg="white")
        style = ttk.Style(root)
        style.theme_use("default")
        root.option_add("*Listbox.background", "white")
        root.option_add("*Listbox.foreground", "black")
        root.option_add("*Listbox.selectBackground", "#0078d7")
        root.option_add("*Listbox.selectForeground", "white")
        root.option_add("*Listbox.highlightBackground", "white")
        root.option_add("*Listbox.highlightColor", "#0078d7")
        root.option_add("*Listbox.borderWidth", 2)
        style.configure("Big.TLabel", font=("Arial", 16, "bold"))
        style.configure("TLabelframe.Label", font=("Arial", 10, "bold"))


_DPI_MIN = 1.0
_DPI_MAX = 3.0


class DpiScale:
    """DPI-aware pixel and font scaling for Tk widgets."""

    MIN = _DPI_MIN
    MAX = _DPI_MAX

    @staticmethod
    def setup() -> None:
        """Configure process DPI awareness and Tk OpenGL before any window is created."""
        if platform.system() == "Linux":
            os.environ.setdefault("TK_WINDOWS_FORCE_OPENGL", "1")
            # GDK_SCALE (when set by the desktop) is read by for_widget().
        else:
            try:  # >= win 8.
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except (AttributeError, OSError):  # win 8.0 or less
                ctypes.windll.user32.SetProcessDPIAware()
            os.environ["TK_WINDOWS_FORCE_OPENGL"] = "1"

    @staticmethod
    def for_widget(widget: tk.Misc, parent: tk.Misc | None = None) -> float:
        """Pixels-per-inch normalised to 96 dpi, with font and GDK fallbacks."""
        if parent is not None:
            inherited = getattr(parent, "_dpi_scale", None)
            if inherited is not None:
                return float(inherited)

        ppi_scale = _DPI_MIN
        try:
            widget.update_idletasks()
            ppi_scale = max(_DPI_MIN, min(_DPI_MAX, widget.winfo_fpixels("1i") / 96.0))
        except tk.TclError:
            pass

        gdk_scale = os.environ.get("GDK_SCALE")
        env_scale = _DPI_MIN
        if gdk_scale:
            try:
                env_scale = max(_DPI_MIN, min(_DPI_MAX, float(gdk_scale)))
            except ValueError:
                pass

        font_scale = DpiScale._font_scale_from_default()
        return max(ppi_scale, env_scale, font_scale)

    @staticmethod
    def scaled(n: float, scale: float) -> int:
        """Scale a pixel value and round to int."""
        return round(n * scale)

    @staticmethod
    def scaled_font(scale: float, family: str, size: int, **kwargs) -> tuple:
        """Return a font tuple scaled for geometry scale (plugin README rendering only)."""
        font_size = max(1, DpiScale.scaled(size, scale))
        parts: list = [family, font_size]
        if "weight" in kwargs:
            parts.append(kwargs["weight"])
        if "slant" in kwargs:
            parts.append(kwargs["slant"])
        return tuple(parts)

    @staticmethod
    def _font_scale_from_default() -> float:
        """Estimate scale from TkDefaultFont when PPI alone is insufficient (common on Linux)."""
        try:
            font = tkfont.nametofont("TkDefaultFont")
            size = font.cget("size")
            if isinstance(size, str):
                size = int(size)
            size = abs(int(size))
            if size <= 0:
                return _DPI_MIN
            # Negative size is pixels; positive is points. Baseline ~10 pt / 12 px.
            baseline = 12 if font.cget("size") < 0 else 10
            return max(_DPI_MIN, min(_DPI_MAX, size / baseline))
        except (tk.TclError, ValueError, TypeError):
            return _DPI_MIN


def configure_treeview_style(style: ttk.Style, name: str, scale: float = 1.0) -> None:
    """Set Treeview row height and heading font to match the default UI font."""
    font = tkfont.nametofont("TkDefaultFont")
    rowheight = font.metrics("linespace") + DpiScale.scaled(4, scale)
    style.configure(f"{name}.Treeview", rowheight=rowheight)
    style.configure(f"{name}.Treeview.Heading", font=font)


class UiAssets:
    """Resolve shipped UI image assets from the repository data tree."""

    @staticmethod
    def resolve(name: str):
        path = DataPaths._repo_data_root() / "imgs" / name
        if not path.is_file():
            raise FileNotFoundError(f"UI asset not found: {name}")
        return path


class DataPicker:
    """File-picker helpers for map, plan, and data path display."""

    @staticmethod
    def display_path(stored: str) -> str:
        """Format a stored settings path for picker display."""
        if not stored:
            return ""
        if stored.startswith("~/"):
            return stored
        abs_path = Path(DataPaths.resolve(stored)).resolve()
        user_root = DataPaths.user_dir().resolve()
        repo_root = DataPaths._repo_data_root().resolve()
        try:
            abs_path.relative_to(user_root)
            return DataPicker._format_user_path(abs_path)
        except ValueError:
            pass
        try:
            rel = abs_path.relative_to(repo_root)
            return "data/" + rel.as_posix()
        except ValueError:
            return stored

    @staticmethod
    def default_map_display_path() -> str:
        return DataPicker.display_path(ExecutionSettings.c40_map)

    @staticmethod
    def default_map_settings_field() -> str:
        return "c40_map"

    @staticmethod
    def default_global_plan_display_path() -> str:
        return DataPicker.display_path(ExecutionSettings.c40_global_trajectory)

    @staticmethod
    def list_map_candidates() -> list[str]:
        def _is_map(path: Path) -> bool:
            return HDMap.is_loadable(path) or RaceMap.is_loadable(path)

        return [""] + DataPicker._collect_candidates(_is_map)

    @staticmethod
    def list_global_plan_candidates() -> list[str]:
        return [""] + DataPicker._collect_candidates(
            lambda path: GlobalPlan.is_loadable(path)
        )

    @staticmethod
    def _format_user_path(abs_path) -> str:
        try:
            rel = abs_path.resolve().relative_to(Path.home())
            return "~/" + rel.as_posix()
        except ValueError:
            return str(abs_path.resolve())

    @staticmethod
    def _format_repo_path(abs_path) -> str:
        rel = abs_path.relative_to(DataPaths._repo_data_root())
        return "data/" + rel.as_posix()

    @staticmethod
    def _path_for_file(file_path, data_root) -> str:
        if data_root.resolve() == DataPaths.user_dir().resolve():
            return DataPicker._format_user_path(file_path)
        return DataPicker._format_repo_path(file_path)

    @staticmethod
    def _iter_data_files(*roots):
        for root in roots:
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if path.is_file():
                    yield path, root

    @staticmethod
    def _collect_candidates(predicate) -> list[str]:
        repo_data = DataPaths._repo_data_root()
        user_data = DataPaths.user_dir()
        seen: set[str] = set()
        user_candidates: list[str] = []
        repo_candidates: list[str] = []

        for path, root in DataPicker._iter_data_files(user_data, repo_data):
            if not predicate(path):
                continue
            picker_path = DataPicker._path_for_file(path, root)
            if picker_path in seen:
                continue
            seen.add(picker_path)
            if root.resolve() == user_data.resolve():
                user_candidates.append(picker_path)
            else:
                repo_candidates.append(picker_path)

        return sorted(user_candidates) + sorted(repo_candidates)


# ---------------------------------------------------------------------------
# Strategy contract popup (world / stack requirements + capabilities)
# ---------------------------------------------------------------------------

_CONTRACT_MET = "#6abf69"
_CONTRACT_UNMET = "#e57373"
_CONTRACT_CONSUMED = "#6abf69"
_CONTRACT_UNUSED = "#888888"
_CONTRACT_REDUNDANT = "#f0a040"


def _contract_sets(target):
    """Return (world_reqs, stack_reqs, provided) from a live instance or class.

    Raises TypeError when *target* is a class whose contract is still an
    instance ``@property`` (e.g. pipelines — stage-dependent).
    """
    def _get(name: str) -> set:
        val = getattr(target, name, None)
        if isinstance(val, property):
            raise TypeError(name)
        return set(val or ())

    return (
        _get("world_requirements"),
        _get("stack_requirements"),
        _get("stack_capabilities"),
    )


def _strategy_type_name(target) -> str:
    """Class name of a live instance or class (reload-safe identity)."""
    cls = target if isinstance(target, type) else type(target)
    return cls.__name__


def _live_strategy_from_exec(executer, cls):
    """Return the live module instance if *cls* is currently loaded on *executer*.

    Matches by class ``__name__`` so identity survives ``importlib.reload``.
    """
    if executer is None or cls is None:
        return None
    want = cls.__name__ if isinstance(cls, type) else type(cls).__name__
    modules = [
        getattr(executer, "perception", None),
        getattr(executer, "localization", None),
        getattr(executer, "mapping", None),
        getattr(executer, "global_planner", None),
        getattr(executer, "local_planner", None),
        getattr(executer, "controller", None),
    ]
    stage_attrs = (
        "_detector", "_tracker", "_predictor",
        "_behavioral", "_path", "_velocity",
    )
    for m in modules:
        if m is None:
            continue
        if type(m).__name__ == want:
            return m
        for attr in stage_attrs:
            stage = getattr(m, attr, None)
            if stage is not None and type(stage).__name__ == want:
                return stage
    runner = getattr(executer, "_task_runner", None)
    if runner is not None:
        for task in getattr(runner, "tasks", []) or []:
            if type(task).__name__ == want:
                return task
    return None


def _finish_contract_popup(pop: tk.Toplevel, frame: ttk.Frame, anchor) -> tk.Toplevel:
    """Legend, Close, Escape, place — shared by every popup path."""
    legend = ttk.Frame(frame)
    legend.pack(anchor="w", pady=(8, 0))
    style = ttk.Style(pop)
    style.configure("ContractLegend.Title.TLabel", foreground=_CONTRACT_UNUSED, font=("", 8, "bold"))
    ttk.Label(legend, text="Legend", style="ContractLegend.Title.TLabel").grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 2)
    )
    entries = (
        ("met / consumed", _CONTRACT_MET, "Met"),
        ("unmet", _CONTRACT_UNMET, "Unmet"),
        ("redundant", _CONTRACT_REDUNDANT, "Redundant"),
        ("unused", _CONTRACT_UNUSED, "Unused"),
    )
    for i, (text, color, key) in enumerate(entries):
        style_name = f"ContractLegend.{key}.TLabel"
        style.configure(style_name, foreground=color, font=("", 8))
        ttk.Label(legend, text=text, style=style_name).grid(
            row=1 + i // 2, column=i % 2, sticky="w", padx=(0, 12)
        )
    ttk.Button(frame, text="Close", command=pop.destroy).pack(anchor="e", pady=(6, 0))
    pop.bind("<Escape>", lambda e: pop.destroy())
    _place_popup(pop, anchor)
    pop.focus_set()
    return pop


def _pack_labeled_cap_row(parent, label: str, caps, available: set, *, soft: bool, joiner: str) -> None:
    """Pack ``label · A | B`` (or ``&``) with per-member colors."""
    row = ttk.Frame(parent)
    row.pack(anchor="w")
    ttk.Label(row, text=f"  {label} · ", foreground=_CONTRACT_UNUSED).pack(side=tk.LEFT)
    for i, cap in enumerate(sorted(caps, key=lambda c: c.name)):
        if i:
            ttk.Label(row, text=f" {joiner} ", foreground=_CONTRACT_UNUSED).pack(side=tk.LEFT)
        present = cap in available
        if soft:
            color = _CONTRACT_MET if present else _CONTRACT_UNUSED
        else:
            color = _CONTRACT_MET if present else _CONTRACT_UNMET
        ttk.Label(row, text=cap.name, foreground=color).pack(side=tk.LEFT)


def _mentioned_caps(requirements: set) -> set:
    """Plain caps + AnyOf/MayUse members — for ``cap in consumed`` coloring."""
    out: set = set()
    for req in requirements:
        if CapabilityGroup.matches(req):
            out |= set(req.capabilities)
        else:
            out.add(req)
    return out


def _pack_requirement_rows(parent, requirements: set, available: set) -> None:
    if not requirements:
        ttk.Label(parent, text="  (none)", foreground=_CONTRACT_UNUSED).pack(anchor="w")
        return
    plain: list = []
    wrappers: list = []
    for req in requirements:
        if AnyOf.matches(req) or MayUse.matches(req):
            wrappers.append(req)
        else:
            plain.append(req)
    if plain:
        _pack_labeled_cap_row(parent, "all", plain, available, soft=False, joiner="&")
    for req in sorted(wrappers, key=lambda r: " | ".join(sorted(c.name for c in r.capabilities))):
        if AnyOf.matches(req):
            _pack_labeled_cap_row(parent, "any", req.capabilities, available, soft=False, joiner="|")
        else:
            _pack_labeled_cap_row(parent, "optional", req.capabilities, available, soft=True, joiner="|")

def _is_perception_stage(pipeline, target) -> bool:
    """True when *target* is (or is the type of) a detect/track/predict stage."""
    if pipeline is None or target is None:
        return False
    stages = (
        getattr(pipeline, "_detector", None),
        getattr(pipeline, "_tracker", None),
        getattr(pipeline, "_predictor", None),
    )
    want = _strategy_type_name(target)
    return any(s is not None and type(s).__name__ == want for s in stages)


def _other_providers(executer, target) -> set:
    """Stack caps provided by world GT and other top-level modules (not *target*).

    World GT is filtered by Bridge Setting (``is_world_stack_capability_enabled``).
    The parent ``PerceptionPipeline`` is excluded when *target* is one of its stages
    so parent advertising does not mark a stage's caps as redundant.
    """
    caps: set = set()
    if executer is None:
        return caps
    target_name = _strategy_type_name(target)
    if getattr(executer, "world", None) is not None:
        from avlite.c40_execution.c41_world_bridge import is_world_stack_capability_enabled

        caps |= {
            c for c in executer.world.stack_capabilities if is_world_stack_capability_enabled(c)
        }
    perception = getattr(executer, "perception", None)
    for m in (
        perception,
        getattr(executer, "localization", None),
        getattr(executer, "mapping", None),
        getattr(executer, "global_planner", None),
        getattr(executer, "local_planner", None),
        getattr(executer, "controller", None),
    ):
        if m is None or type(m).__name__ == target_name:
            continue
        if m is perception and _is_perception_stage(perception, target):
            continue
        caps |= set(getattr(m, "stack_capabilities", set()) or set())
    return caps

def show_strategy_contract_popup(
    anchor,
    *,
    name: str,
    registry: dict,
    get_exec,
    title: str | None = None,
):
    """Show a color-coded contract popup for the selected strategy name."""
    parent = anchor.winfo_toplevel()
    pop = tk.Toplevel(parent)
    pop.title(title or (name or "(none)"))
    pop.transient(parent)
    pop.resizable(False, False)

    frame = ttk.Frame(pop, padding=8)
    frame.pack(fill=tk.BOTH, expand=True)

    if not name:
        ttk.Label(frame, text="(none) — module omitted").pack(anchor="w")
        return _finish_contract_popup(pop, frame, anchor)

    cls = registry.get(name)
    if cls is None:
        ttk.Label(frame, text=f"Unknown strategy: {name}").pack(anchor="w")
        return _finish_contract_popup(pop, frame, anchor)

    executer = get_exec() if callable(get_exec) else get_exec
    live = _live_strategy_from_exec(executer, cls)
    target = live if live is not None else cls
    try:
        world_reqs, stack_reqs, provided = _contract_sets(target)
    except TypeError:
        ttk.Label(
            frame,
            text="Contract depends on configured stages — reload to inspect",
            foreground=_CONTRACT_UNUSED,
        ).pack(anchor="w")
        return _finish_contract_popup(pop, frame, anchor)

    world_caps: set = set()
    stack_available: set = set()
    if executer is not None:
        if getattr(executer, "world", None) is not None:
            from avlite.c40_execution.c41_world_bridge import is_world_capability_enabled

            world_caps = {
                c for c in executer.world.world_capabilities if is_world_capability_enabled(c)
            }
        if hasattr(executer, "available_stack_capabilities"):
            stack_available = set(executer.available_stack_capabilities())

    target_name = _strategy_type_name(target)
    other_modules = []
    if executer is not None:
        for m in (
            executer.perception,
            executer.localization,
            getattr(executer, "mapping", None),
            executer.global_planner,
            executer.local_planner,
            executer.controller,
        ):
            if m is not None and type(m).__name__ != target_name:
                other_modules.append(m)
    combined = combine_stack_requirements(other_modules, soft=True)
    if executer is not None and getattr(executer, "world", None) is not None:
        combined |= combine_stack_requirements([executer.world], soft=True)
    consumed = _mentioned_caps(combined)
    redundant = _other_providers(executer, target)

    ttk.Label(frame, text="World requirements", font=("", 9, "bold")).pack(anchor="w")
    _pack_requirement_rows(frame, world_reqs, world_caps)

    ttk.Label(frame, text="Stack requirements", font=("", 9, "bold")).pack(
        anchor="w", pady=(6, 0)
    )
    _pack_requirement_rows(frame, stack_reqs, stack_available)

    ttk.Label(frame, text="Stack capabilities", font=("", 9, "bold")).pack(
        anchor="w", pady=(6, 0)
    )
    if not provided:
        ttk.Label(frame, text="  (none)", foreground=_CONTRACT_UNUSED).pack(anchor="w")
    else:
        for cap in sorted(provided, key=lambda c: c.name):
            if cap in redundant:
                color = _CONTRACT_REDUNDANT
            elif cap in consumed:
                color = _CONTRACT_CONSUMED
            else:
                color = _CONTRACT_UNUSED
            ttk.Label(frame, text=f"  {cap.name}", foreground=color).pack(anchor="w")

    return _finish_contract_popup(pop, frame, anchor)


def _place_popup(pop: tk.Toplevel, anchor) -> None:
    pop.update_idletasks()
    try:
        x = anchor.winfo_rootx()
        y = anchor.winfo_rooty() + anchor.winfo_height()
    except tk.TclError:
        x, y = 100, 100
    pop.geometry(f"+{x}+{y}")


def make_strategy_contract_controls(parent, combobox, registry, get_exec):
    """Bind right-click on *combobox*; return ``(show_popup, info_btn)``.

    Caller places *info_btn* next to the Combobox.
    """

    def show_popup(_event=None):
        show_strategy_contract_popup(
            combobox,
            name=combobox.get(),
            registry=registry,
            get_exec=get_exec,
        )

    combobox.bind("<Button-3>", show_popup)
    info_btn = ttk.Button(parent, text="ⓘ", width=2, command=show_popup)
    HoverTooltip.attach(info_btn, "World requirements, stack requirements & capabilities")
    return show_popup, info_btn


def show_world_bridge_contract_popup(
    anchor,
    *,
    name: str,
    registry: dict,
    get_exec,
    title: str | None = None,
):
    """Show a color-coded contract popup for the selected world bridge."""
    from avlite.c40_execution.c41_world_bridge import (
        is_world_capability_enabled,
        is_world_stack_capability_enabled,
    )

    parent = anchor.winfo_toplevel()
    pop = tk.Toplevel(parent)
    pop.title(title or (name or "(none)"))
    pop.transient(parent)
    pop.resizable(False, False)

    frame = ttk.Frame(pop, padding=8)
    frame.pack(fill=tk.BOTH, expand=True)

    if not name:
        ttk.Label(frame, text="(none) — no bridge selected").pack(anchor="w")
        return _finish_contract_popup(pop, frame, anchor)

    cls = registry.get(name)
    if cls is None:
        ttk.Label(frame, text=f"Unknown bridge: {name}").pack(anchor="w")
        return _finish_contract_popup(pop, frame, anchor)

    executer = get_exec() if callable(get_exec) else get_exec
    live = getattr(executer, "world", None) if executer is not None else None
    target = live if live is not None and type(live).__name__ == cls.__name__ else cls

    def _attr(obj, name_: str) -> set:
        val = getattr(obj, name_, None)
        if isinstance(val, property):
            return set()
        return set(val or ())

    world_caps = _attr(target, "world_capabilities")
    stack_reqs = _attr(target, "stack_requirements")
    provided = _attr(target, "stack_capabilities")

    stack_available: set = set()
    if executer is not None and hasattr(executer, "available_stack_capabilities"):
        stack_available = set(executer.available_stack_capabilities())

    other_modules = []
    if executer is not None:
        for m in (
            getattr(executer, "perception", None),
            getattr(executer, "localization", None),
            getattr(executer, "mapping", None),
            getattr(executer, "global_planner", None),
            getattr(executer, "local_planner", None),
            getattr(executer, "controller", None),
        ):
            if m is not None:
                other_modules.append(m)
    consumed = _mentioned_caps(combine_stack_requirements(other_modules, soft=True))
    redundant = set()
    for m in other_modules:
        redundant |= set(getattr(m, "stack_capabilities", set()) or set())

    ttk.Label(frame, text="World capabilities", font=("", 9, "bold")).pack(anchor="w")
    if not world_caps:
        ttk.Label(frame, text="  (none)", foreground=_CONTRACT_UNUSED).pack(anchor="w")
    else:
        for cap in sorted(world_caps, key=lambda c: c.name):
            if executer is None:
                color = _CONTRACT_MET
            else:
                color = _CONTRACT_MET if is_world_capability_enabled(cap) else _CONTRACT_UNUSED
            ttk.Label(frame, text=f"  {cap.name}", foreground=color).pack(anchor="w")

    ttk.Label(frame, text="Stack requirements", font=("", 9, "bold")).pack(
        anchor="w", pady=(6, 0)
    )
    _pack_requirement_rows(frame, stack_reqs, stack_available)

    ttk.Label(frame, text="Stack capabilities", font=("", 9, "bold")).pack(
        anchor="w", pady=(6, 0)
    )
    if not provided:
        ttk.Label(frame, text="  (none)", foreground=_CONTRACT_UNUSED).pack(anchor="w")
    else:
        for cap in sorted(provided, key=lambda c: c.name):
            enabled = executer is None or is_world_stack_capability_enabled(cap)
            if not enabled:
                color = _CONTRACT_UNUSED
            elif cap in redundant:
                color = _CONTRACT_REDUNDANT
            elif cap in consumed:
                color = _CONTRACT_CONSUMED
            else:
                color = _CONTRACT_UNUSED
            ttk.Label(frame, text=f"  {cap.name}", foreground=color).pack(anchor="w")

    return _finish_contract_popup(pop, frame, anchor)


def make_world_bridge_contract_controls(parent, combobox, get_exec):
    """Bind right-click on bridge *combobox*; return ``(show_popup, info_btn)``."""
    from avlite.c40_execution.c41_world_bridge import WorldBridge

    def show_popup(_event=None):
        show_world_bridge_contract_popup(
            combobox,
            name=combobox.get(),
            registry=WorldBridge.registry,
            get_exec=get_exec,
        )

    combobox.bind("<Button-3>", show_popup)
    info_btn = ttk.Button(parent, text="ⓘ", width=2, command=show_popup)
    HoverTooltip.attach(info_btn, "World capabilities, stack requirements & capabilities")
    return show_popup, info_btn

