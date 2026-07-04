import ctypes
import logging
import os
import platform
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
from avlite.c20_planning.c24_global_hdmap_planners import HDMapGlobalPlanner
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c50_apps.c54_settings_schema import PlainBinder, field_tooltip_text
from avlite.c50_apps.c58_paths import DataPaths

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
            dpi_scale = get_dpi_scale(parent)

        self.min_value = min_value
        self.max_value = max_value
        self.current_value = 0
        self.marker_value = 0
        self.variable = variable
        self.pack_propagate(False)
        self.font = ("Helvetica", 7, "bold")
        self._old_value = 0  # used to not draw if value is same
        gauge_height = scaled(height, dpi_scale)

        if name != "":
            tk.Label(self, text=name, font=self.font).pack(side=tk.LEFT, padx=0)

        min_label = tk.Label(self, text=f"{min_value:+.2f}", font=self.font)
        min_label.pack(side=tk.LEFT, padx=0)

        max_label = tk.Label(self, text=f"{max_value:+.2f}", font=self.font)
        max_label.pack(side=tk.RIGHT, padx=0)

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

        # Create rectangle first (will be behind text)
        text_id = self.canvas.create_text(x, y, text=text, fill="white", font=self.font, anchor="center", tags="value")
        bbox = self.canvas.bbox(text_id)
        # self.canvas.coords(text_id, x, y + 2)  # Move text down within the box
        self.canvas.create_rectangle(bbox, fill="black", tags="bg")
        self.canvas.tag_raise("value")
        self.canvas.tag_raise(text_id)


class ThemedInputDialog:
    def __init__(self, parent, title, prompt, initial=""):
        self.result = None

        # Create the dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.transient(parent)
        s = get_dpi_scale(self.dialog, parent=parent)
        self.dialog.minsize(scaled(300, s), scaled(100, s))
        
        self.dialog.bind("<Escape>", lambda e: self.on_cancel())  # Bind Escape key to cancel

        
        
        frame = ttk.Frame(self.dialog)
        frame.pack(expand=True, fill=tk.BOTH)

        top_frame = ttk.Frame(frame)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Add prompt and entry field
        ttk.Label(top_frame, text=prompt).pack(side=tk.LEFT, padx=10)
        self.entry = ttk.Entry(top_frame, width=max(10, scaled(20, s)))
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
        s = get_dpi_scale(self.dialog, parent=parent)
        self.dialog.minsize(scaled(360, s), scaled(140, s))
        
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
        self.second_entry = ttk.Entry(frame, width=max(20, scaled(30, s)))
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
        s = get_dpi_scale(self.dialog, parent=parent)
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
            width=max(30, scaled(40, s)),
        )
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        if len(items) > _max_visible_rows:
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for item in items:
            self.listbox.insert(tk.END, item)

        if initial and initial in items:
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
    ):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.transient(parent)
        s = get_dpi_scale(self.dialog, parent=parent)
        self.dialog.minsize(scaled(360, s), scaled(120, s))
        self.dialog.bind("<Escape>", lambda _e: self.dialog.destroy())

        frame = ttk.Frame(self.dialog)
        frame.grid(row=0, column=0, sticky="nsew")
        self.dialog.grid_rowconfigure(0, weight=1)
        self.dialog.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        ttk.Label(frame, text=label1).grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        ttk.Label(frame, text=value1, wraplength=scaled(280, s)).grid(
            row=0, column=1, padx=10, pady=10, sticky=tk.W
        )
        ttk.Label(frame, text=label2).grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        ttk.Label(frame, text=value2, wraplength=scaled(280, s)).grid(
            row=1, column=1, padx=10, pady=10, sticky=tk.W
        )
        ttk.Button(frame, text="OK", command=self.dialog.destroy).grid(
            row=2, column=0, columnspan=2, padx=5, pady=10, sticky=tk.E
        )

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

        s = get_dpi_scale(self.widget)
        x = self.widget.winfo_rootx() + scaled(20, s)
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + scaled(4, s)
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
            wraplength=scaled(360, s),
            padx=scaled(6, s),
            pady=scaled(4, s),
        )
        label.pack()
        self._tip = tip

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


def attach_tooltip(widget, text: str) -> HoverTooltip | None:
    if text:
        tooltip = HoverTooltip(widget, text)
        widget._hover_tooltip = tooltip
        return tooltip
    return None


def attach_schema_tooltip(widget, settings_cls, field_name: str) -> None:
    attach_tooltip(widget, field_tooltip_text(settings_cls, field_name) or "")


def update_schema_tooltip(widget, settings_cls, field_name: str) -> None:
    text = field_tooltip_text(settings_cls, field_name) or ""
    tooltip = getattr(widget, "_hover_tooltip", None)
    if tooltip is not None:
        tooltip.text = text
    else:
        attach_tooltip(widget, text)


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
    "profile_delete": "Delete the selected profile.",
    "profile_save": "Save all settings into the selected profile.",
    "profile_export": "Export the profile as a zip archive.",
    "profile_import": "Import a profile from a zip archive.",
    "profile_rename": "Rename the selected profile.",
    "profile_reset_all": "Reset every module setting to source-code defaults.",
    "profile_reset_non_exec": "Reset all settings except execution to defaults.",
    # Settings window — plugins
    "plugins_reset_builtin": "Restore the built-in plugin list to defaults.",
    "plugins_remove_builtin": "Remove the selected built-in plugin entry.",
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
        setup_dpi_impl()

    @staticmethod
    def for_widget(widget: tk.Misc, parent: tk.Misc | None = None) -> float:
        return get_dpi_scale_impl(widget, parent=parent)

    @staticmethod
    def scaled(n: float, scale: float) -> int:
        return scaled_impl(n, scale)

    @staticmethod
    def scaled_font(scale: float, family: str, size: int, **kwargs) -> tuple:
        return scaled_font_impl(scale, family, size, **kwargs)


def scaled(n: float, scale: float) -> int:
    """Scale a pixel value and round to int."""
    return scaled_impl(n, scale)


def scaled_font(scale: float, family: str, size: int, **kwargs) -> tuple:
    """Return a font tuple scaled for geometry scale (plugin README rendering only)."""
    return scaled_font_impl(scale, family, size, **kwargs)


def setup_dpi() -> None:
    """Configure process DPI awareness and Tk OpenGL before any window is created."""
    setup_dpi_impl()


def get_dpi_scale(widget: tk.Misc, parent: tk.Misc | None = None) -> float:
    """Pixels-per-inch normalised to 96 dpi, with font and GDK fallbacks."""
    return get_dpi_scale_impl(widget, parent=parent)


def scaled_impl(n: float, scale: float) -> int:
    """Scale a pixel value and round to int."""
    return round(n * scale)


def scaled_font_impl(scale: float, family: str, size: int, **kwargs) -> tuple:
    """Return a font tuple scaled for geometry scale (plugin README rendering only)."""
    font_size = max(1, scaled_impl(size, scale))
    parts: list = [family, font_size]
    if "weight" in kwargs:
        parts.append(kwargs["weight"])
    if "slant" in kwargs:
        parts.append(kwargs["slant"])
    return tuple(parts)


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


def setup_dpi_impl() -> None:
    """Configure process DPI awareness and Tk OpenGL before any window is created."""
    if platform.system() == "Linux":
        os.environ.setdefault("TK_WINDOWS_FORCE_OPENGL", "1")
        # GDK_SCALE (when set by the desktop) is read by get_dpi_scale().
    else:
        try:  # >= win 8.
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):  # win 8.0 or less
            ctypes.windll.user32.SetProcessDPIAware()
        os.environ["TK_WINDOWS_FORCE_OPENGL"] = "1"


def get_dpi_scale_impl(widget: tk.Misc, parent: tk.Misc | None = None) -> float:
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

    font_scale = _font_scale_from_default()
    return max(ppi_scale, env_scale, font_scale)


def configure_treeview_style(style: ttk.Style, name: str, scale: float = 1.0) -> None:
    """Set Treeview row height and heading font to match the default UI font."""
    font = tkfont.nametofont("TkDefaultFont")
    rowheight = font.metrics("linespace") + scaled(4, scale)
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
        if ExecutionSettings.c40_global_planner == HDMapGlobalPlanner.__name__:
            return DataPicker.display_path(ExecutionSettings.c40_hd_map)
        return DataPicker.display_path(ExecutionSettings.c43_race_boundary_map)

    @staticmethod
    def default_map_settings_field() -> str:
        if ExecutionSettings.c40_global_planner == HDMapGlobalPlanner.__name__:
            return "c40_hd_map"
        return "c43_race_boundary_map"

    @staticmethod
    def default_global_plan_display_path() -> str:
        return DataPicker.display_path(ExecutionSettings.c40_global_trajectory)

    @staticmethod
    def list_map_candidates() -> list[str]:
        def _is_map(path: Path) -> bool:
            return HDMap.is_loadable(path) or RaceMap.is_loadable(path)

        return DataPicker._collect_candidates(_is_map)

    @staticmethod
    def list_global_plan_candidates() -> list[str]:
        return DataPicker._collect_candidates(
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

