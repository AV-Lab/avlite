"""Visualizer toolbar: profile dropdown, shortcuts, settings/plugins launchers."""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c50_apps.c59_settings import AppSettings
from avlite.plugins.p50_visualizer_tk.p52_setting_views import SettingWindow
from avlite.plugins.p50_visualizer_tk.p53_ui_lib import (
    BUTTON_TOOLTIPS,
    TkSettingsBinder,
    attach_schema_tooltip,
    attach_tooltip,
    scaled,
)
from avlite.plugins.p50_visualizer_tk.settings import VisualizationSettings
from avlite.c50_apps.c58_paths import PluginPaths
from avlite.c50_apps.c55_setting_utils import save_setting
from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.plugins.p50_visualizer_tk.p51_plugins_app import CommunityPluginsApp

if TYPE_CHECKING:
    from avlite.plugins.p50_visualizer_tk.p51_visualizer_app import VisualizerApp

log = logging.getLogger(__name__)


class ConfigShortcutView(ttk.LabelFrame):
    def __init__(self, root: VisualizerApp):
        super().__init__(root, text="Config")

        self.root: VisualizerApp = root
        # ----------------------------------------------------------------------
        # Key Bindings --------------------------------------------------------
        # ----------------------------------------------------------------------
        ## App shortcuts
        self.root.bind("T", lambda e: self.open_settings_window())
        self.root.bind_all("Q", lambda e: self.root.quit())
        self.root.bind_all("R", lambda e: self.root.reload_stack())
        self.root.bind_all("F", lambda e: self.root.switch_profile() )
        self.root.bind("S", lambda e: self.root.update_shortcut_mode(reverse=True))
        self.root.bind("<Control-s>", lambda e: self.save_config())
        
        self.root.bind("<Control-plus>", lambda e: self.root.local_plan_plot_view.zoom_in_frenet())
        self.root.bind("<Control-minus>", lambda e: self.root.local_plan_plot_view.zoom_out_frenet())
        self.root.bind("<plus>", lambda e: self.root.local_plan_plot_view.zoom_in())
        self.root.bind("<minus>", lambda e: self.root.local_plan_plot_view.zoom_out())

        ## Exec shortcuts
        self.root.bind("x", lambda e: self.root.exec_visualize_view.toggle_exec())
        self.root.bind("c", lambda e: self.root.exec_visualize_view.step_exec())
        self.root.bind("t", lambda e: self.root.exec_visualize_view.reset_exec())

        ## Perception, Planning and Control shortcuts
        self.root.bind("n", lambda e: self.root.perceive_plan_control_view.plan_frame.step_plan())
        self.root.bind("b", lambda e: self.root.perceive_plan_control_view.plan_frame.step_waypoint_back())
        self.root.bind("r", lambda e: self.root.perceive_plan_control_view.plan_frame.replan())

        self.root.bind("h", lambda e: self.root.perceive_plan_control_view.control_frame.step_control())
        self.root.bind("i", lambda e: self.root.perceive_plan_control_view.control_frame.align_control())

        self.root.bind("<KeyPress-a>", lambda e: self.root.perceive_plan_control_view.control_frame.step_steer_left())
        self.root.bind("<KeyPress-d>", lambda e: self.root.perceive_plan_control_view.control_frame.step_steer_right())
        self.root.bind("<KeyRelease-a>", lambda e: self.root.perceive_plan_control_view.control_frame.reset_steer())
        self.root.bind("<KeyRelease-d>", lambda e: self.root.perceive_plan_control_view.control_frame.reset_steer())
        self.root.bind("w", lambda e: self.root.perceive_plan_control_view.control_frame.step_acc())
        self.root.bind("s", lambda e: self.root.perceive_plan_control_view.control_frame.step_dec())


        ## Log shortcuts 
        # scroll log text using vim motion
        self.root.bind("k", lambda e: self.root.log_view.log_area.yview_scroll(-1, "units")) 
        self.root.bind("j", lambda e: self.root.log_view.log_area.yview_scroll(1, "units"))
        self.root.bind("<Control-u>",  lambda e: self.root.log_view.log_area.yview_scroll(-5, "units"))
        self.root.bind("<Control-d>",  lambda e: self.root.log_view.log_area.yview_scroll(int(0.5*self.root.setting.p57_log_view_default_height.get()), "units"))
        self.root.bind("G", lambda e: self.root.log_view.log_area.yview_moveto(1.0))
        self.root.bind("g", lambda e: self.root.log_view.log_area.yview_moveto(0.0))
        
        self.root.bind("<Up>", lambda e: self.root.log_view.log_area.yview_scroll(-1, "units")) 
        self.root.bind("<Down>", lambda e: self.root.log_view.log_area.yview_scroll(1, "units"))

        self.root.bind("E", lambda e: self.root.log_view.update_log_view_height(reverse=True))  # Toggle log view height
        self.root.bind("L", lambda e: self.root.log_view.clear_log())  # Toggle log view height
    

        ## Additional
        self.root.bind("<Escape>", lambda e: self.root.focus_set()) # Unfocus any entry fields including widgets.
        # ----------------------------------------------------------------------

        btn_settings = ttk.Button(self, text="⚙", command=self.open_settings_window, width=2)
        btn_settings.pack(side=tk.RIGHT)
        attach_tooltip(btn_settings, BUTTON_TOOLTIPS["toolbar_settings"])
        btn_plugins = ttk.Button(self, text="Plugins", command=self.open_plugins_window)
        btn_plugins.pack(side=tk.RIGHT)
        attach_tooltip(btn_plugins, BUTTON_TOOLTIPS["toolbar_plugins"])
        btn_reload = ttk.Button(self, text="Reload Stack", command=self.root.reload_stack)
        btn_reload.pack(side=tk.RIGHT)
        attach_tooltip(btn_reload, BUTTON_TOOLTIPS["toolbar_reload_stack"])
        btn_reset = ttk.Button(self, text="Reset Config", command=self.root.load_configs)
        btn_reset.pack(side=tk.RIGHT)
        attach_tooltip(btn_reset, BUTTON_TOOLTIPS["toolbar_reset_config"])
        btn_save = ttk.Button(self, text="Save Config", command=self.save_config)
        btn_save.pack(side=tk.RIGHT)
        attach_tooltip(btn_save, BUTTON_TOOLTIPS["toolbar_save_config"])


        self.profile_dropdown_menu = ttk.Combobox(self, width=10, textvariable=self.root.app.c50_selected_profile, state="readonly",
            justify=tk.CENTER, font=("Arial", 10, "bold"))
        self.profile_dropdown_menu["values"] = self.root.setting.profile_list
        self.profile_dropdown_menu.state(["readonly"])
        self.profile_dropdown_menu.bind("<<ComboboxSelected>>", self.__on_profile_dropdown_change)
        self.profile_dropdown_menu.pack(side=tk.RIGHT)
        attach_schema_tooltip(self.profile_dropdown_menu, AppSettings, "c50_selected_profile")

        shortcut_cb = ttk.Checkbutton(self, text="Shortcut Mode", variable=self.root.setting.p50_shortcut_mode,
            command=self.root.update_shortcut_mode,)
        shortcut_cb.pack(anchor=tk.W, side=tk.LEFT)
        attach_schema_tooltip(shortcut_cb, VisualizationSettings, "p50_shortcut_mode")

        dark_cb = ttk.Checkbutton(self, text="Dark Mode", variable=self.root.setting.p50_dark_mode, command=self.toggle_dark_mode)
        dark_cb.pack(anchor=tk.W, side=tk.LEFT)
        attach_schema_tooltip(dark_cb, VisualizationSettings, "p50_dark_mode")
        
        ttk.Label(self, textvariable=self.root.setting.perception_status_text, width=30).pack(side=tk.LEFT, padx=(25,5), pady=5)

        # ----------------------------------------------------------------------
        # Shortcut frame
        # ------------------------------------------------------
        # TODO: this is a dirty trick because its parent is root
        _s = getattr(root, "_dpi_scale", 1.0)
        self.shortcut_frame = ttk.LabelFrame(root, text="Shortcuts")
        self.help_text = tk.Text(
            self.shortcut_frame,
            wrap=tk.WORD,
            width=max(30, scaled(50, _s)),
            height=max(5, scaled(7, _s)),
        )
        key_binding_info = """
App:      Q - Quit             S - Toggle shortcut          F - Switch to next Profile  R - Reload imports     
          T - Open Settings    E - Expand/collapse log      ↑/↓- use Up/Down or vim motion to scroll log   
Plan:     n - Step plan        b - Step Back                r - Replan            
          + - Zoom In          - - Zoom Out           <Ctrl+> - Zoom In F         <Ctrl-> - Zoom Out F
Control:  h - Control Step     i - Re-align control         w - Accelerate 
          a - Steer left       d - Steer right              s - Deccelerate
Execute:  c - Step Execution   t - Reset execution          x - Toggle execution
         """.strip()
        self.help_text.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.help_text.insert(tk.END, key_binding_info)
        self.help_text.config(state=tk.DISABLED)  # Make the text area read-only



    def __on_profile_dropdown_change(self, event):
        log.info(f"Selected profile: {event.widget.get()}")
        self.root.load_configs()
        self.root.reload_stack(reload_code=False)


    def toggle_dark_mode(self):
        self.root.set_dark_mode_themed() if self.root.setting.p50_dark_mode.get() else self.root.set_light_mode()

    def save_config(self):
        profile = self.root.app.c50_selected_profile.get()
        binder = TkSettingsBinder()
        save_setting(self.root.setting, profile=profile, binder=binder)
        save_setting(PerceptionSettings, profile=profile, binder=binder)
        AppSettings.c50_community_plugins = PluginPaths.normalize_map(
            AppSettings.c50_community_plugins
        )
        save_setting(ExecutionSettings, profile=profile, binder=binder)


    def open_settings_window(self):
        if hasattr(self, "setting_view") and hasattr(self.setting_view, "window") and self.setting_view.window.winfo_exists():
            # Show existing window
            self.root.load_configs(only_stack=True)
            self.setting_view.show()
            log.info("Showing existing settings window")
        else:
            self.root.load_configs(only_stack=True)
            self.setting_view = SettingWindow(self.root)
            log.info("Creating new settings window")

    def update_setting_window(self):
        """Update the settings window with the latest settings."""
        if hasattr(self, "setting_view") and hasattr(self.setting_view, "window") and self.setting_view.window.winfo_exists():
            self.setting_view.update_core_widgets()
            self.setting_view.update_plugins_widgets()
            self.setting_view.update_community_plugin_list()
            log.info("Updated existing settings window")

    def open_plugins_window(self):
        """Open the community plugins manager window."""
        CommunityPluginsApp.open(parent=self.root)


