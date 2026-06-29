from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
import importlib
import tkinter as tk
from tkinter import filedialog, ttk, messagebox


from avlite.c60_common.c67_paths import (
    bundled_config_dir,
    can_edit_repo_configs,
    community_plugin_settings_display_path,
    effective_config_path,
    format_user_path,
    get_config_dir,
    installed_community_plugins_map,
    is_repo_config_target,
    normalize_community_plugin_stored,
    normalize_community_plugins_map,
    set_repo_config_target,
    set_startup_profile,
)
from avlite.c60_common.c60_plugins import (
    import_plugin_modules,
    list_plugins,
    load_all_stack_settings,
    load_builtin_plugin_settings,
    load_community_plugin_setting,
    plugin_module_prefix,
    reload_lib,
)
from avlite.c60_common.c69_setting_utils import (
    delete_setting_profile,
    export_profile,
    import_profile,
    list_profiles,
    load_setting,
    rename_setting_profile,
    save_setting,
)
from avlite.c50_visualization.c58_ui_lib import (
    BUTTON_TOOLTIPS,
    HoverTooltip,
    ThemedInputDialog,
    ThemedReadOnlyTwoFieldDialog,
    ThemedTwoInputDialog,
    TkSettingsBinder,
    attach_schema_tooltip,
    attach_tooltip,
    get_dpi_scale,
    scaled,
)
from avlite.c50_visualization.c59_settings import VisualizationSettings
from avlite.c60_common.c68_settings_schema import field_tooltip_text, setting_key

from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c20_planning.c29_settings import PlanningSettings
from avlite.c30_control.c39_settings import ControlSettings
from avlite.c40_execution.c49_settings import ExecutionSettings

import logging

log = logging.getLogger(__name__)


def _widget_key(setting, plugin_name: str = "") -> str:
    return setting_key(setting) + plugin_name


if TYPE_CHECKING:
    from avlite.c50_visualization.c51_visualizer_app import VisualizerApp


def _refresh_profile_dropdowns(win: "SettingWindow", *, select: str | None = None) -> str:
    """Re-read profile names from the active config target; update all comboboxes."""
    profiles = list_profiles(win.root.setting)
    win.root.setting.profile_list = profiles
    for combo in (
        win.profile_dropdown_menu,
        win.next_profile_dropdown_menu,
        win.root.config_shortcut_view.profile_dropdown_menu,
    ):
        combo["values"] = profiles
    if select and select in profiles:
        current = select
        win.root.setting.selected_profile.set(current)
    else:
        current = win.root.setting.selected_profile.get()
        if current not in profiles:
            current = "default" if "default" in profiles else (profiles[0] if profiles else "default")
            win.root.setting.selected_profile.set(current)
    if win.root.setting.next_profile.get() not in profiles:
        win.root.setting.next_profile.set(current)
    return current


def _setting_window_edit_repo_configs_toggle(win: "SettingWindow") -> None:
    enabled = win._edit_repo_configs_var.get()
    if enabled and not messagebox.askyesno(
        "Edit repository configs",
        f"Save and load will use files under\n{bundled_config_dir()}\n"
        f"instead of your user config dir ({get_config_dir()}).\n\nContinue?",
        parent=win.window,
    ):
        win._edit_repo_configs_var.set(False)
        return
    set_repo_config_target(enabled)
    profile = _refresh_profile_dropdowns(win)
    win.root.load_configs(profile=profile)
    win.load_profile(profile)


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
        self.root.bind("<Control-d>",  lambda e: self.root.log_view.log_area.yview_scroll(int(0.5*self.root.setting.log_view_default_height.get()), "units"))
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


        self.profile_dropdown_menu = ttk.Combobox(self, width=10, textvariable=self.root.setting.selected_profile, state="readonly",
            justify=tk.CENTER, font=("Arial", 10, "bold"))
        self.profile_dropdown_menu["values"] = self.root.setting.profile_list
        self.profile_dropdown_menu.state(["readonly"])
        self.profile_dropdown_menu.bind("<<ComboboxSelected>>", self.__on_profile_dropdown_change)
        self.profile_dropdown_menu.pack(side=tk.RIGHT)
        attach_schema_tooltip(self.profile_dropdown_menu, VisualizationSettings, "selected_profile")

        shortcut_cb = ttk.Checkbutton(self, text="Shortcut Mode", variable=self.root.setting.shortcut_mode,
            command=self.root.update_shortcut_mode,)
        shortcut_cb.pack(anchor=tk.W, side=tk.LEFT)
        attach_schema_tooltip(shortcut_cb, VisualizationSettings, "shortcut_mode")

        dark_cb = ttk.Checkbutton(self, text="Dark Mode", variable=self.root.setting.dark_mode, command=self.toggle_dark_mode)
        dark_cb.pack(anchor=tk.W, side=tk.LEFT)
        attach_schema_tooltip(dark_cb, VisualizationSettings, "dark_mode")
        
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
        self.root.set_dark_mode_themed() if self.root.setting.dark_mode.get() else self.root.set_light_mode()

    def save_config(self):
        profile = self.root.setting.selected_profile.get()
        binder = TkSettingsBinder()
        save_setting(self.root.setting, profile=profile, binder=binder)
        ExecutionSettings.c40_community_plugins = normalize_community_plugins_map(
            ExecutionSettings.c40_community_plugins
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
        from avlite.c50_visualization.c54_plugins import CommunityPluginsApp
        CommunityPluginsApp.open(parent=self.root)


class SettingWindow:
    """
    A view to display and edit settings.
    """
    def __init__(self, root: VisualizerApp):
        self.root = root
        self.setting = root.setting
        self.window = tk.Toplevel(root)
        # settings_window.title("Settings")
        _s = get_dpi_scale(self.window, parent=root)
        self.window.geometry(f"{scaled(550, _s)}x{scaled(450, _s)}")
        # self.root.bind("Q", lambda e: settings_window.destroy())

        self.frame = ttk.Frame(self.window)
        self.frame.pack(fill=tk.BOTH, expand=True)

        
        self.window.bind("<Control-s>", lambda e: self.save_profile())
        self.window.bind("k", lambda e: self.canvas.yview_scroll(-1, "units")) 
        self.window.bind("j", lambda e: self.canvas.yview_scroll(1, "units"))
        self.window.bind("<Control-u>",  lambda e: self.canvas.yview_scroll(-5, "units"))
        self.window.bind("<Control-d>",  lambda e: self.canvas.yview_scroll(int(0.5*self.root.setting.log_view_default_height.get()), "units"))
        self.window.bind("G", lambda e: self.canvas.yview_moveto(1.0))
        self.window.bind("g", lambda e: self.canvas.yview_moveto(0.0))
        
        #########
        # Main Layout
        #########
        self.frame.rowconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=1)  # Settings frame
        
        profile_ext_frame = ttk.Frame(self.frame)
        profile_ext_frame.grid(row=0, column=0, sticky="nswe", padx=10, pady=10)

        settings_frame = ttk.Frame(self.frame)
        settings_frame.grid(row=0, column=1, rowspan=1, sticky="nsew", padx=10, pady=10)

        additional_setting_frame = ttk.LabelFrame(self.frame, text="Additional Settings")
        additional_setting_frame.grid(row=1, column=0, columnspan=2, sticky="sew", padx=5, pady=5)

        ##########
        # Profiles & Plugins
        ##########
        profile_ext_frame.rowconfigure(8, weight=1)


        ttk.Label(profile_ext_frame, text="Execution Profiles",style="Big.TLabel").grid(row=0, column=0, sticky="w", columnspan=3, padx=10, pady=5)
        ttk.Label(profile_ext_frame, text="Load Profile").grid(row=1, column=0, padx=5, pady=5)
        self.profile_dropdown_menu = ttk.Combobox(profile_ext_frame, textvariable=self.root.setting.selected_profile, state="readonly",)
        self.profile_dropdown_menu["values"] = self.root.setting.profile_list
        # self.global_planner_dropdown_menu.current(0)  
        self.profile_dropdown_menu.state(["readonly"])
        self.profile_dropdown_menu.bind("<<ComboboxSelected>>", self.__on_profile_dropdown_change)
        self.profile_dropdown_menu.grid(row=1, column=1, columnspan=2, padx=5, pady=5)

        btn_profile_new = ttk.Button(profile_ext_frame, text="New", command=self.create_profile)
        btn_profile_new.grid(row=2, column=0, padx=5, pady=5, sticky="we")
        attach_tooltip(btn_profile_new, BUTTON_TOOLTIPS["profile_new"])
        btn_profile_delete = ttk.Button(profile_ext_frame, text="Delete", command=self.delete_profile)
        btn_profile_delete.grid(row=2, column=1, padx=5, pady=5, sticky="we")
        attach_tooltip(btn_profile_delete, BUTTON_TOOLTIPS["profile_delete"])
        btn_profile_save = ttk.Button(profile_ext_frame, text="Save", underline=0, command=self.save_profile)
        btn_profile_save.grid(row=2, column=2, padx=5, pady=5, sticky="we")
        attach_tooltip(btn_profile_save, BUTTON_TOOLTIPS["profile_save"])
        btn_profile_export = ttk.Button(profile_ext_frame, text="Export", command=self.export_profile_zip)
        btn_profile_export.grid(row=3, column=0, padx=5, pady=5, sticky="we")
        attach_tooltip(btn_profile_export, BUTTON_TOOLTIPS["profile_export"])
        btn_profile_import = ttk.Button(profile_ext_frame, text="Import", command=self.import_profile_zip)
        btn_profile_import.grid(row=3, column=1, padx=5, pady=5, sticky="we")
        attach_tooltip(btn_profile_import, BUTTON_TOOLTIPS["profile_import"])
        btn_profile_rename = ttk.Button(profile_ext_frame, text="Rename", command=self.rename_profile)
        btn_profile_rename.grid(row=3, column=2, padx=5, pady=5, sticky="we")
        attach_tooltip(btn_profile_rename, BUTTON_TOOLTIPS["profile_rename"])

        ttk.Label(profile_ext_frame, text="Cycle Next (Shortcut F)").grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        self.next_profile_dropdown_menu = ttk.Combobox(profile_ext_frame, width=10, textvariable=self.root.setting.next_profile, state="readonly",)
        self.next_profile_dropdown_menu["values"] = self.root.setting.profile_list
        self.next_profile_dropdown_menu.state(["readonly"])
        # next_profile_dropdown_menu.bind("<<ComboboxSelected>>", self.__on_dropdown_change)
        self.next_profile_dropdown_menu.grid(row=4, column=2, padx=5, pady=5, sticky="we")
        
        btn_reset_all = ttk.Button(
            profile_ext_frame, text="Reset all to source code defaults", command=self.reset_to_to_source_stack_values
        )
        btn_reset_all.grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky="we")
        attach_tooltip(btn_reset_all, BUTTON_TOOLTIPS["profile_reset_all"])
        btn_reset_non_exec = ttk.Button(
            profile_ext_frame, text="Reset all except Exectution",
            command=lambda: self.reset_to_to_source_stack_values(exclude_execution=True),
        )
        btn_reset_non_exec.grid(row=6, column=0, columnspan=3, padx=5, pady=5, sticky="we")
        attach_tooltip(btn_reset_non_exec, BUTTON_TOOLTIPS["profile_reset_non_exec"])
        if can_edit_repo_configs():
            self._edit_repo_configs_var = tk.BooleanVar(value=is_repo_config_target())
            cb_edit_repo = ttk.Checkbutton(
                profile_ext_frame, text="Edit repository configs", variable=self._edit_repo_configs_var,
                command=lambda: _setting_window_edit_repo_configs_toggle(self),
            )
            cb_edit_repo.grid(row=7, column=0, columnspan=3, padx=5, pady=5, sticky="w")
            attach_tooltip(cb_edit_repo, BUTTON_TOOLTIPS["edit_repo_configs"])
        ## Plugins
        ##############################################
        plugin_frame = ttk.LabelFrame(profile_ext_frame, text="Plugins")
        plugin_frame.grid(row=8, column=0, columnspan=3, sticky="sew", padx=5, pady=5)
        plugin_frame.rowconfigure(2, weight=1)
        plugin_frame.rowconfigure(5, weight=1)
        plugin_frame.columnconfigure(0, weight=1)
        plugin_frame.columnconfigure(1, weight=1)

        listbox_height = max(6, scaled(10, _s))

        cb_load_plugins = ttk.Checkbutton(
            plugin_frame, text="Load Plugins", variable=self.root.setting.load_plugins,
            command=self._on_load_plugins_toggle,
        )
        cb_load_plugins.grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        attach_schema_tooltip(cb_load_plugins, VisualizationSettings, "load_plugins")

        # built-in plugins
        ttk.Label(plugin_frame, text="Plugins").grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        self.listbox_default_plugins = tk.Listbox(
            plugin_frame, height=listbox_height, selectmode=tk.SINGLE, exportselection=False, width=30,
        )
        self.listbox_default_plugins.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        # Convert comma-separated string to list items

        for plugin in ExecutionSettings.c40_default_plugins:
            self.listbox_default_plugins.insert(tk.END, plugin)
        
        btn_reset_plugins = ttk.Button(plugin_frame, text="Reset Plugins", command=self.reset_default_plugins)
        btn_reset_plugins.grid(row=3, column=0, sticky="we", padx=5, pady=5)
        attach_tooltip(btn_reset_plugins, BUTTON_TOOLTIPS["plugins_reset_builtin"])
        btn_remove_builtin = ttk.Button(plugin_frame, text="Remove Plugin", command=self.remove_default_plugin)
        btn_remove_builtin.grid(row=3, column=1, sticky="we", padx=5, pady=5)
        attach_tooltip(btn_remove_builtin, BUTTON_TOOLTIPS["plugins_remove_builtin"])


        # community plugins
        ttk.Label(plugin_frame, text="Community Plugins").grid(row=4, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        self.listbox_community_plugins = tk.Listbox(
            plugin_frame, height=listbox_height, selectmode=tk.SINGLE, exportselection=False, width=30,
        )
        self.listbox_community_plugins.grid(row=5, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        # Convert comma-separated string to list items

        for plugin in ExecutionSettings.c40_community_plugins.keys() if self.root.setting.load_plugins.get() else []:
            self.listbox_community_plugins.insert(tk.END, plugin)

        self.listbox_community_plugins.bind("<Double-Button-1>", lambda e: self.edit_community_plugin())
        self.listbox_community_plugins.bind("<<ListboxSelect>>", lambda e: self._scroll_to_selected_plugin())
        self.listbox_default_plugins.bind("<Double-Button-1>", lambda e: self.edit_default_plugin())
        self.listbox_default_plugins.bind("<<ListboxSelect>>", lambda e: self._scroll_to_selected_builtin_plugin())



        btn_reset_community = ttk.Button(plugin_frame, text="Reset to Installed", command=self.reset_community_plugins)
        btn_reset_community.grid(row=6, column=0, columnspan=2, sticky="we", padx=5, pady=5)
        attach_tooltip(btn_reset_community, BUTTON_TOOLTIPS["plugins_reset_community"])
        btn_add_plugin = ttk.Button(plugin_frame, text="Add Plugin", command=self.add_community_plugin)
        btn_add_plugin.grid(row=7, column=0, sticky="we", padx=5, pady=5)
        attach_tooltip(btn_add_plugin, BUTTON_TOOLTIPS["plugins_add"])
        btn_remove_community = ttk.Button(plugin_frame, text="Remove Plugin", command=self.delete_community_plugin)
        btn_remove_community.grid(row=7, column=1, sticky="we", padx=5, pady=5)
        attach_tooltip(btn_remove_community, BUTTON_TOOLTIPS["plugins_remove_community"])
        btn_browse_plugins = ttk.Button(plugin_frame, text="Browse Community Plugins…", command=self.open_plugins_window)
        btn_browse_plugins.grid(row=8, column=0, columnspan=2, sticky="we", padx=5, pady=5)
        attach_tooltip(btn_browse_plugins, BUTTON_TOOLTIPS["plugins_browse"])


        #############################################
        # settings
        #############################################
        settings_frame.columnconfigure(0, weight=1)
        settings_frame.rowconfigure(0, weight=1)
        
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        settings_frame.bind_all("<MouseWheel>", _on_mousewheel)  # Windows/macOS
        settings_frame.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))  # Linux
        settings_frame.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))   # Linux

        def unbind_mousewheel_events(event=None):
            settings_frame.unbind_all("<MouseWheel>")
            settings_frame.unbind_all("<Button-4>")
            settings_frame.unbind_all("<Button-5>")
        settings_frame.bind("<Destroy>", unbind_mousewheel_events)

        style = ttk.Style()
        bg_color = style.lookup("TFrame", "background")
        self.canvas = tk.Canvas(settings_frame, highlightthickness=0, bd=0, background=bg_color)
        self.scrollbar = ttk.Scrollbar(settings_frame, orient="vertical", command=self.canvas.yview)
        self.settings_frame = ttk.Frame(self.canvas)

        # Configure scrolling
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.settings_frame.bind( "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        # Create window with fixed width that matches self.canvas
        self.canvas.create_window((0, 0), window=self.settings_frame, anchor="nw")

        # Grid layout with proper weights
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        
        ########
        ########
        # Set minimum width/height for the container
        self.window.update_idletasks()  # Force layout update
        self.window.minsize(scaled(500, _s), scaled(400, _s))
        # to prevent killing the window afte close
        self.window.protocol("WM_DELETE_WINDOW", self.hide) 

        ######################
        ## Stack widgetes
        ######################
        ttk.Label(self.settings_frame, text="Core Stack Settings",style="Big.TLabel").pack(anchor=tk.W, padx=5, pady=5)
        # to keep track of all widgets
        self.widget_entries = {}
        self.settings_section_frames = {}
        self.create_widgets(PerceptionSettings, "Perception Settings")
        self.create_widgets(PlanningSettings, "Planning Settings")
        self.create_widgets(ControlSettings, "Control Settings")
        self.create_widgets(ExecutionSettings, "Execution Settings")
        #######
        #######
        if self.root.setting.load_plugins.get():
            ttk.Separator(self.settings_frame, orient='horizontal').pack(fill='x', pady=10)
            ttk.Label(self.settings_frame, text="Plugin Settings",style="Big.TLabel").pack(anchor=tk.W, padx=5, pady=5)
            self.create_plugin_widgets()

        # Pre-create separator and label for community plugin settings section;
        # they are pack()ed lazily inside create_community_plugin_widgets().
        self._cp_sep = ttk.Separator(self.settings_frame, orient='horizontal')
        self._cp_label = ttk.Label(self.settings_frame, text="Community Plugin Settings", style="Big.TLabel")
        self.create_community_plugin_widgets()

        ################
        # Additional settings
        ################

        ## UI Elements for Visualize - Checkboxes
        additional_setting_row_1 = ttk.Frame(additional_setting_frame)
        ttk.Label(additional_setting_row_1, text="Local Plan Plot View:").pack(anchor=tk.W, side=tk.LEFT, padx=5)
        additional_setting_row_1.pack(fill=tk.X)
        for text, field in (
            ("Legend", "show_legend"),
            ("Locations", "show_past_locations"),
            ("Global Plan", "show_global_plan"),
            ("Local Plan", "show_local_plan"),
            ("Local Lattice", "show_local_lattice"),
            ("State", "show_state"),
        ):
            cb = ttk.Checkbutton(
                additional_setting_row_1, text=text,
                variable=getattr(self.root.setting, field), command=self.root.update_ui,
            )
            cb.pack(anchor=tk.W, side=tk.LEFT)
            attach_schema_tooltip(cb, VisualizationSettings, field)

        for text, field in (
            ("Follow Planner in Global", "global_view_follow_planner"),
            ("Follow Planner in Frenet", "frenet_view_follow_planner"),
        ):
            cb = ttk.Checkbutton(additional_setting_row_1, text=text, variable=getattr(self.root.setting, field))
            cb.pack(side=tk.LEFT)
            attach_schema_tooltip(cb, VisualizationSettings, field)

        additional_setting_row_1b = ttk.Frame(additional_setting_frame)
        additional_setting_row_1b.pack(fill=tk.X)
        ttk.Label(additional_setting_row_1b, text="Local Plan Plot View:").pack(anchor=tk.W, side=tk.LEFT, padx=5)
        for text, field in (
            ("Local Global View", "show_local_global_view"),
            ("Local Frenet View", "show_local_frenet_view"),
            ("LiDAR in Global", "show_lidar_global"),
            ("LiDAR in Frenet", "show_lidar_frenet"),
            ("Clustered Pts", "show_lidar_clusters"),
            ("Race Boundary", "show_race_boundary"),
        ):
            cb = ttk.Checkbutton(
                additional_setting_row_1b, text=text,
                variable=getattr(self.root.setting, field), command=self.root.update_ui,
            )
            cb.pack(side=tk.LEFT)
            attach_schema_tooltip(cb, VisualizationSettings, field)

        additional_setting_row_2 = ttk.Frame(additional_setting_frame)
        additional_setting_row_2.pack(fill=tk.X, padx=5)

        ttk.Label(additional_setting_row_2, text="Log View:").pack(anchor=tk.W, side=tk.LEFT, padx=0)
        cb_expand_log = ttk.Checkbutton(additional_setting_row_2, text="Expand Log View", variable=self.root.setting.log_view_expanded)
        cb_expand_log.pack(side=tk.LEFT)
        attach_schema_tooltip(cb_expand_log, VisualizationSettings, "log_view_expanded")
        
        ttk.Label(additional_setting_row_2, text="Default Log Height:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(additional_setting_row_2, textvariable=self.root.setting.log_view_default_height, width=5,
                  validatecommand=self.root.validate_cmd).pack(side=tk.LEFT, padx=5)

        ttk.Label(additional_setting_row_2, text="Expanded Log Height").pack(side=tk.LEFT, padx=5)
        ttk.Entry(additional_setting_row_2, textvariable=self.root.setting.log_view_expended_height, width=5,
                  validatecommand=self.root.validate_cmd).pack(side=tk.LEFT, padx=5)
        
        additional_setting_row_3 = ttk.Frame(additional_setting_frame)
        additional_setting_row_3.pack(fill=tk.X, padx=0)
        ttk.Label(additional_setting_row_3, text="Menu bar:").pack(anchor=tk.W, side=tk.LEFT, padx=5)
        cb_hide_menubar = ttk.Checkbutton(additional_setting_row_3, text="Hide", variable=self.root.setting.hide_menubar)
        cb_hide_menubar.pack(anchor=tk.W, side=tk.LEFT)
        attach_schema_tooltip(cb_hide_menubar, VisualizationSettings, "hide_menubar")
        btn_settings_close = ttk.Button(additional_setting_row_2, text="Close", width=5, underline=0, command=self.hide)
        btn_settings_close.pack(side=tk.RIGHT, padx=5)
        attach_tooltip(btn_settings_close, BUTTON_TOOLTIPS["settings_close"])
        btn_settings_save = ttk.Button(additional_setting_row_2, text="Save", width=5, underline=0, command=self.save_profile)
        btn_settings_save.pack(side=tk.RIGHT, padx=5)
        attach_tooltip(btn_settings_save, BUTTON_TOOLTIPS["settings_save"])

    def reset_community_plugins(self):
        """Reset community plugins to all plugins installed under the plugins directory."""
        log.info("Resetting community plugins to installed set.")
        ExecutionSettings.c40_community_plugins = installed_community_plugins_map()
        self.update_community_plugin_list()
        if self.root.setting.load_plugins.get():
            self.update_community_plugin_widgets()

    def reset_default_plugins(self):
        """ Reset the default plugins to the source code defaults. """
        log.info("Resetting default plugins to source code defaults.")

        self.listbox_default_plugins.delete(0, tk.END)
        ExecutionSettings.c40_default_plugins = list_plugins()

        for plugin in ExecutionSettings.c40_default_plugins:
            self.listbox_default_plugins.insert(tk.END, plugin)

        import_plugin_modules(plugins_filter=ExecutionSettings.c40_default_plugins)
        self.root.perceive_plan_control_view.reset()
        self.root.exec_visualize_view.update_data()

    def remove_default_plugin(self):
        """ Remove the selected default plugin from the list. """
        selected = self.listbox_default_plugins.curselection()
        if selected:
            plugin_name = self.listbox_default_plugins.get(selected)
            if plugin_name in ExecutionSettings.c40_default_plugins:
                ExecutionSettings.c40_default_plugins.remove(plugin_name)
                self.listbox_default_plugins.delete(selected)
                self._unregister_plugin(plugin_name)
                self.root.perceive_plan_control_view.reset()
                self.root.exec_visualize_view.update_data()
                log.info(f"Removed and unloaded plugin: {plugin_name}")
            else:
                log.warning(f"Plugin {plugin_name} not found in default plugins.")
        else:
            log.warning("No plugin selected to remove.")

    def _unregister_plugin(self, plugin_name: str):
        """Remove all strategy classes registered by a plugin from all registries and sys.modules."""
        import sys
        from avlite.c10_perception.c12_perception_strategy import PerceptionStrategy
        from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
        from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
        from avlite.c30_control.c32_control_strategy import ControlStrategy
        from avlite.c40_execution.c41_world_bridge import WorldBridge
        from avlite.c40_execution.c42_executer import Executer

        plugin_module_prefix_str = plugin_module_prefix(plugin_name)
        for registry in [PerceptionStrategy.registry, GlobalPlannerStrategy.registry,
                         LocalPlanningStrategy.registry, ControlStrategy.registry,
                         Executer.registry, WorldBridge.registry]:
            to_remove = [name for name, cls in registry.items()
                         if cls.__module__.startswith(plugin_module_prefix_str)]
            for name in to_remove:
                del registry[name]
                log.info(f"Unregistered {name} from {plugin_module_prefix_str}")

        # Purge plugin modules from sys.modules so reload_lib won't re-register them
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith(plugin_module_prefix_str):
                del sys.modules[mod_name]
                log.debug(f"Removed {mod_name} from sys.modules")


    def add_community_plugin(self):
        dialog = ThemedTwoInputDialog(self.root, "Community Plugins", "Package Name", "Package Directory")
        name, dir =  dialog.result if dialog.result else (None, None)
        if not name:
            return

        log.info(f"Adding plugin: {name}")
        ExecutionSettings.c40_community_plugins[name] = normalize_community_plugin_stored(name, dir)
        self.listbox_community_plugins.insert(tk.END, name)
        
    def delete_community_plugin(self):
        selected = self.listbox_community_plugins.curselection()
        if selected:
            plugin_name = self.listbox_community_plugins.get(selected)
            ExecutionSettings.c40_community_plugins.pop(plugin_name, None)
            self.listbox_community_plugins.delete(selected)
            log.info(f"Deleted community plugin: {plugin_name}")
        else:
            log.warning("No community plugin selected to delete.")

    def edit_default_plugin(self):
        selected = self.listbox_default_plugins.curselection()
        if not selected:
            log.warning("No plugin selected to edit.")
            return

        plugin_name = self.listbox_default_plugins.get(selected[0])
        cls = load_builtin_plugin_settings(plugin_name)
        if cls is not None and getattr(cls, "filepath", None):
            settings_path = format_user_path(
                effective_config_path(cls.filepath, for_write=False)
            )
        else:
            settings_path = "\u2014"
        ThemedReadOnlyTwoFieldDialog(
            self.root,
            "Plugin",
            "Package Name",
            "Settings file",
            plugin_name,
            settings_path,
        )

    def edit_community_plugin(self):
        selected = self.listbox_community_plugins.curselection()
        if not selected:
            log.warning("No community plugin selected to edit.")
            return

        plugin_name = self.listbox_community_plugins.get(selected)
        ThemedReadOnlyTwoFieldDialog(
            self.root,
            "Community Plugin",
            "Package Name",
            "Settings file",
            plugin_name,
            community_plugin_settings_display_path(plugin_name),
        )

    def update_community_plugin_list(self):
        """ Load the community plugins from the settings. """

        self.listbox_community_plugins.delete(0, tk.END)
        for name, dir in ExecutionSettings.c40_community_plugins.items():
            self.listbox_community_plugins.insert(tk.END, name)

    def open_plugins_window(self):
        """Open the community plugins manager and refresh the list on close."""
        from avlite.c50_visualization.c54_plugins import CommunityPluginsApp
        app = CommunityPluginsApp.open(parent=self.root)
        app.window.bind("<Destroy>", lambda _e: self.update_community_plugin_list(), add="+")


    def create_profile(self):
        """ Load a profile from the settings. """
        
        # text = simpledialog.askstring("Profile", "Enter profile Name")
        dialog = ThemedInputDialog(self.root, "Profile", "Name")
        text =  dialog.result.strip() if dialog.result else None
        if not text:
            return
        log.info(f"Creating profile: {text}")
        self.root.setting.selected_profile.set(text)
        self.root.setting.profile_list.append(text)
        self.profile_dropdown_menu["values"] = self.root.setting.profile_list
        self.root.config_shortcut_view.profile_dropdown_menu["values"] = self.root.setting.profile_list
        self.next_profile_dropdown_menu["values"] = self.root.setting.profile_list

        self.save_profile()


    def export_profile_zip(self):
        """Export the selected profile to a zip file."""
        profile = self.root.setting.selected_profile.get()
        if not messagebox.askyesno(
            "Export profile",
            f"Export reads from saved files on disk.\nSave profile '{profile}' first if you have unsaved changes.\n\nContinue?",
            parent=self.window,
        ):
            return
        zip_path = filedialog.asksaveasfilename(
            parent=self.window,
            title="Export profile",
            defaultextension=".zip",
            initialfile=f"{profile}.zip",
            filetypes=[("AVLite profile", "*.zip"), ("All files", "*.*")],
        )
        if not zip_path:
            return
        try:
            load_setting(ExecutionSettings, profile=profile)
            count = export_profile(
                profile,
                zip_path,
                community_plugins=ExecutionSettings.c40_community_plugins,
            )
        except ValueError as e:
            messagebox.showerror("Export profile", str(e), parent=self.window)
            return
        except OSError as e:
            messagebox.showerror("Export profile", f"Failed to write zip: {e}", parent=self.window)
            return
        messagebox.showinfo(
            "Export profile",
            f"Exported profile '{profile}' ({count} file(s)) to\n{zip_path}",
            parent=self.window,
        )

    def import_profile_zip(self):
        """Import a profile from a zip file."""
        zip_path = filedialog.askopenfilename(
            parent=self.window,
            title="Import profile",
            filetypes=[("AVLite profile", "*.zip"), ("All files", "*.*")],
        )
        if not zip_path:
            return
        overwrite = False
        try:
            import zipfile
            import yaml as _yaml

            with zipfile.ZipFile(zip_path, "r") as zf:
                for name in zf.namelist():
                    if not name.endswith(".yaml"):
                        continue
                    snippet = _yaml.safe_load(zf.read(name)) or {}
                    if isinstance(snippet, dict) and len(snippet) == 1:
                        imported_name = next(iter(snippet.keys()))
                        break
                else:
                    messagebox.showerror("Import profile", "No profile found in zip.", parent=self.window)
                    return
        except (OSError, _yaml.YAMLError) as e:
            messagebox.showerror("Import profile", str(e), parent=self.window)
            return

        if imported_name in self.root.setting.profile_list:
            if not messagebox.askyesno(
                "Import profile",
                f"Profile '{imported_name}' already exists.\nOverwrite it?",
                parent=self.window,
            ):
                return
            overwrite = True

        try:
            profile_name = import_profile(zip_path, overwrite=overwrite)
        except ValueError as e:
            messagebox.showerror("Import profile", str(e), parent=self.window)
            return
        except OSError as e:
            messagebox.showerror("Import profile", f"Failed to import: {e}", parent=self.window)
            return

        profile_name = _refresh_profile_dropdowns(self, select=profile_name)
        self.load_profile(profile_name)
        messagebox.showinfo(
            "Import profile",
            f"Imported profile '{profile_name}'.",
            parent=self.window,
        )


    def delete_profile(self):
        """ Delete a profile from the settings. """

        from tkinter import messagebox
        result = messagebox.askyesno("Confirmation", f"Are you sure you want to delete {self.root.setting.selected_profile.get()}?")
        if result:
            log.info(f"Deleting profile: {self.root.setting.selected_profile.get()}")
            delete_setting_profile(PerceptionSettings, profile=self.root.setting.selected_profile.get())
            delete_setting_profile(PlanningSettings, profile=self.root.setting.selected_profile.get())
            delete_setting_profile(ControlSettings, profile=self.root.setting.selected_profile.get())
            delete_setting_profile(ExecutionSettings, profile=self.root.setting.selected_profile.get())
            delete_setting_profile(self.root.setting, profile=self.root.setting.selected_profile.get())
            if self.root.setting.load_plugins.get():
                for plugin in ExecutionSettings.c40_default_plugins:
                    try:
                        module = importlib.import_module(f"{plugin_module_prefix(plugin)}.settings")
                        PluginSettings = getattr(module, "PluginSettings")
                        delete_setting_profile(PluginSettings, profile=self.root.setting.selected_profile.get())
                    except Exception as e:
                        log.error(f"Failed to delete plugin settings for {plugin}: {e}")

            self.root.setting.profile_list.remove(self.root.setting.selected_profile.get())
            self.profile_dropdown_menu["values"] = self.root.setting.profile_list
            self.root.config_shortcut_view.profile_dropdown_menu["values"] = self.root.setting.profile_list
            self.root.setting.selected_profile.set("default")  
            self.load_profile("default")



    def rename_profile(self):
        """ Rename the selected profile across all settings files. """

        from tkinter import messagebox
        old_name = self.root.setting.selected_profile.get()
        if old_name == "default":
            messagebox.showwarning("Rename", "Cannot rename the 'default' profile.")
            return

        dialog = ThemedInputDialog(self.root, "Rename Profile", "New name", initial=old_name)
        new_name = dialog.result.strip() if dialog.result else None
        if not new_name or new_name == old_name:
            return
        if new_name in self.root.setting.profile_list:
            messagebox.showwarning("Rename", f"Profile '{new_name}' already exists.")
            return

        log.info(f"Renaming profile '{old_name}' to '{new_name}'")
        rename_setting_profile(PerceptionSettings, old_name, new_name)
        rename_setting_profile(PlanningSettings, old_name, new_name)
        rename_setting_profile(ControlSettings, old_name, new_name)
        rename_setting_profile(ExecutionSettings, old_name, new_name)
        rename_setting_profile(self.root.setting, old_name, new_name)
        if self.root.setting.load_plugins.get():
            for plugin in ExecutionSettings.c40_default_plugins:
                try:
                    module = importlib.import_module(f"{plugin_module_prefix(plugin)}.settings")
                    PluginSettings = getattr(module, "PluginSettings")
                    rename_setting_profile(PluginSettings, old_name, new_name)
                except Exception as e:
                    log.error(f"Failed to rename plugin settings for {plugin}: {e}")

        idx = self.root.setting.profile_list.index(old_name)
        self.root.setting.profile_list[idx] = new_name
        self.root.setting.selected_profile.set(new_name)
        self.profile_dropdown_menu["values"] = self.root.setting.profile_list
        self.next_profile_dropdown_menu["values"] = self.root.setting.profile_list
        self.root.config_shortcut_view.profile_dropdown_menu["values"] = self.root.setting.profile_list



    def save_profile(self):
        """ Save the current settings to the selected profile. """

        log.info(f"Saving profile: {self.root.setting.selected_profile.get()}")
        profile = self.root.setting.selected_profile.get()
        binder = TkSettingsBinder()
        self.save_from_widgets(PerceptionSettings)
        save_setting(PerceptionSettings, profile=profile, binder=binder)
        self.save_from_widgets(PlanningSettings) 
        save_setting(PlanningSettings, profile=profile, binder=binder)
        self.save_from_widgets(ControlSettings)
        save_setting(ControlSettings, profile=profile, binder=binder)
        self.save_from_widgets(ExecutionSettings)
        ExecutionSettings.c40_community_plugins = normalize_community_plugins_map(
            ExecutionSettings.c40_community_plugins
        )
        save_setting(ExecutionSettings, profile=profile, binder=binder)

        save_setting(self.root.setting, profile=profile, binder=binder)

        if self.root.setting.load_plugins.get():
            for plugin in ExecutionSettings.c40_default_plugins:
                try:
                    module = importlib.import_module(f"{plugin_module_prefix(plugin)}.settings")
                    PluginSettings = getattr(module, "PluginSettings")
                    self.save_from_widgets(PluginSettings, plugin_name=plugin)
                    save_setting(PluginSettings, profile=profile, binder=binder)
                except Exception as e:
                    log.error(f"Failed to save plugin settings for {plugin}: {e}")

        # Save community plugin settings
        if self.root.setting.load_plugins.get():
            profile = self.root.setting.selected_profile.get()
            for name, stored in ExecutionSettings.c40_community_plugins.items():
                try:
                    cls = load_community_plugin_setting(
                        name, stored, profile=profile, binder=TkSettingsBinder()
                    )
                    if cls is None:
                        continue
                    self.save_from_widgets(cls, plugin_name=f"community_{name}")
                    save_setting(cls, profile=profile, binder=binder)
                except Exception as e:
                    log.error("Failed to save plugin settings for '%s': %s", name, e)

        # just to save the profile 
        save_setting(self.root.setting, profile=profile, binder=binder)
    

    def load_profile(self, profile="default"):
        """ Load a profile from the settings. """

        log.info(f"loading profile: {profile}")
        binder = TkSettingsBinder()
        load_all_stack_settings(profile=profile, load_plugins=self.root.setting.load_plugins.get())
        load_setting(self.root.setting, profile=profile, binder=binder)
        self.root.setting.selected_profile.set(profile)
        set_startup_profile(profile)

        self.update_core_widgets()
        self.update_plugins_widgets()
        self.update_community_plugin_list()

        self.listbox_default_plugins.delete(0, tk.END)
        for plugin in ExecutionSettings.c40_default_plugins:
            self.listbox_default_plugins.insert(tk.END, plugin)

    def update_core_widgets(self):
        self.update_widgets(PerceptionSettings)
        self.update_widgets(PlanningSettings)
        self.update_widgets(ControlSettings)
        self.update_widgets(ExecutionSettings)
        
        if self.root.setting.load_plugins.get():
            for plugin in ExecutionSettings.c40_default_plugins:
                try:
                    module = importlib.import_module(f"{plugin_module_prefix(plugin)}.settings")
                    PluginSettings = getattr(module, "PluginSettings")
                    load_setting(PluginSettings, profile=self.root.setting.selected_profile.get(), binder=TkSettingsBinder())
                    self.update_widgets(PluginSettings, plugin_name=plugin)
                    log.debug(f"loaded plugin settings for {plugin}")
                except Exception as e:
                    log.error(f"Failed to load plugin settings for {plugin}: {e}")

        self.update_community_plugin_widgets()

    def update_community_plugin_widgets(self):
        """Reload and refresh widgets for community plugins that have ``PluginSettings``."""
        if not self.root.setting.load_plugins.get():
            return
        profile = self.root.setting.selected_profile.get()
        for name, stored in ExecutionSettings.c40_community_plugins.items():
            plugin_name = f"community_{name}"
            cls = load_community_plugin_setting(
                name, stored, profile=profile, binder=TkSettingsBinder()
            )
            if cls is None:
                continue
            try:
                self.update_widgets(cls, plugin_name=plugin_name)
            except Exception as e:
                log.warning("Could not reload plugin settings for '%s': %s", name, e)

    def update_plugins_widgets(self):
        """ Update the plugin widgets with the current settings. """ 

        if self.root.setting.load_plugins.get():
            for plugin in ExecutionSettings.c40_default_plugins:
                try:
                    module = importlib.import_module(f"{plugin_module_prefix(plugin)}.settings")
                    PluginSettings = getattr(module, "PluginSettings")
                    self.update_widgets(PluginSettings, plugin_name=plugin)
                except Exception as e:
                    log.error(f"Failed to update plugin settings for {plugin}: {e}")

    def __on_profile_dropdown_change(self, event):
        log.info(f"Selected profile: {event.widget.get()}")
        self.load_profile(event.widget.get())

    def reset_to_to_source_stack_values(self, exclude_execution=False):
        """ Reset the stack settings to the source code defaults, except for the UI as it is using some 
            some instant variables for tkinter.
        """
        reload_lib(exclude_stack=True, reload_plugins=True)
        from avlite.c10_perception.c19_settings import PerceptionSettings
        from avlite.c20_planning.c29_settings import PlanningSettings
        from avlite.c30_control.c39_settings import ControlSettings
        self.update_widgets(PerceptionSettings)
        self.update_widgets(PlanningSettings)
        self.update_widgets(ControlSettings)

        if not exclude_execution:
            from avlite.c40_execution.c49_settings import ExecutionSettings
            self.update_widgets(ExecutionSettings)


        self.update_plugins_widgets()
        

    def _on_load_plugins_toggle(self):
        self.recreate_plugin_widgets()
        self.root.reload_stack(reload_code=False)

    def recreate_plugin_widgets(self):
        """ Recreate the plugin widgets based on the current setting."""

        if self.root.setting.load_plugins.get():
            self.create_plugin_widgets()
            self.create_community_plugin_widgets()
        else:
            # Clear existing plugin widgets
            for plugin in ExecutionSettings.c40_default_plugins:
                plugin_key = f"PluginSettings{plugin}"
                if plugin_key in self.widget_entries:
                    entry_dict = self.widget_entries[plugin_key]
                    if entry_dict:
                        first_widget = next(iter(entry_dict.values()))
                        parent_frame = first_widget.master
                        parent_frame.pack_forget()  # or grid_forget() if using grid
                    del self.widget_entries[plugin_key]
                    log.debug(f"Removed widgets for {plugin_key}")

            self.plugin_widget_created = False

            # Clear community plugin widgets
            for name in list(ExecutionSettings.c40_community_plugins.keys()):
                cp_key = f"PluginSettingscommunity_{name}"
                if cp_key in self.widget_entries:
                    entry_dict = self.widget_entries[cp_key]
                    if entry_dict:
                        first_widget = next(iter(entry_dict.values()))
                        parent_frame = first_widget.master
                        parent_frame.pack_forget()
                    del self.widget_entries[cp_key]
                    log.debug("Removed community plugin widgets for %s", name)

            # Hide the community plugin section header
            self._cp_sep.pack_forget()
            self._cp_label.pack_forget()
            self._cp_widget_created = False


    def create_plugin_widgets(self):
        if hasattr(self, "plugin_widget_created") and self.plugin_widget_created:
            log.warning("Plugin widgets already created, skipping.")
            return

        for plugin in ExecutionSettings.c40_default_plugins:
            try:
                module = importlib.import_module(f"{plugin_module_prefix(plugin)}.settings")
                PluginSettings = getattr(module, "PluginSettings")
                self.create_widgets(PluginSettings, f"{plugin}", plugin_name=plugin)
                self.plugin_widget_created = True
            except Exception as e:
                log.error(f"Failed to load plugin settings for {plugin}: {e}")

    def create_community_plugin_widgets(self):
        """Create settings widgets for community plugins that expose a ``PluginSettings`` class."""
        if getattr(self, "_cp_widget_created", False):
            log.warning("Community plugin widgets already created, skipping.")
            return

        found = []
        if self.root.setting.load_plugins.get():
            profile = self.root.setting.selected_profile.get()
            for name, stored in ExecutionSettings.c40_community_plugins.items():
                cls = load_community_plugin_setting(
                    name, stored, profile=profile, binder=TkSettingsBinder()
                )
                if cls is not None:
                    found.append((name, cls))

        if not found:
            return

        self._cp_sep.pack(fill='x', pady=10)
        self._cp_label.pack(anchor=tk.W, padx=5, pady=5)
        for name, cls in found:
            self.create_widgets(cls, f"Plugin: {name}", plugin_name=f"community_{name}")

        self._cp_widget_created = True


    def create_widgets(self, setting, setting_name="Settings", plugin_name=""):
        """ Create widgets for the settings view. """

        frame = ttk.Labelframe(self.settings_frame, text=setting_name)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        key = _widget_key(setting, plugin_name)
        self.widget_entries[key] = {}
        self.settings_section_frames[key] = frame
        row = 0
        skip = {"filepath", "schema", "exclude"}
        for field in dir(setting):
            if field.startswith("__") or field in skip or callable(getattr(setting, field)):
                continue
            value = getattr(setting, field)
            if isinstance(value, (str, int, float, bool)):
                label = ttk.Label(frame, text=field)
                label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
                entry = ttk.Entry(frame)
                entry.insert(0, str(value))
                entry.grid(row=row, column=1, padx=5, pady=2, sticky="ew")
                self.widget_entries[key][field] = entry
                tip = field_tooltip_text(setting, field)
                if tip:
                    HoverTooltip(label, tip)
                    HoverTooltip(entry, tip)
                row += 1



    def save_from_widgets(self, setting, plugin_name=""):
        """ Save the settings from the widgets to the setting class. """

        if _widget_key(setting, plugin_name) not in self.widget_entries:
            log.warning(f"No widgets found for setting: {setting_key(setting)}+{plugin_name}")
            return

        # log.warning(f"keys in widget_entries: {self.widget_entries.keys()}")
        for field, entry in self.widget_entries[_widget_key(setting, plugin_name)].items():
            if field.startswith("__") or callable(getattr(setting, field)) or field == "filepath":
                continue
            
            if not hasattr(setting, field):  # Changed from app.data to data
                log.warning(f"Skipping unknown attribute: {field}")
                continue

            val = entry.get()
            orig = getattr(setting, field)
            log.debug(f"Saving {field} with value {val} of type {type(val)} to setting {setting_key(setting)}")

            if isinstance(orig, bool):
                if val.lower() in ["true", "1", "yes"]:
                    setattr(setting, field, True)
                elif val.lower() in ["false", "0", "no"]:
                    setattr(setting, field, False)
                else:
                    log.warning(f"Invalid boolean value for {field}: {val}. Keeping original value: {orig}")
            elif isinstance(orig, int):
                setattr(setting, field, int(val))
            elif isinstance(orig, float):
                setattr(setting, field, float(val))
            else:
                setattr(setting, field, val)

    def update_widgets(self, setting, plugin_name=""):
        """ Update the widgets with the current settings. """
        if _widget_key(setting, plugin_name) not in self.widget_entries:
            log.warning(f"No widgets found for setting: {plugin_name} {setting_key(setting)}")
            return

        for field, entry in self.widget_entries[_widget_key(setting, plugin_name)].items():
            if field.startswith("__") or callable(getattr(setting, field)) or field == "filepath":
                continue
            
            if not hasattr(setting, field):
                log.error(f"Skipping unknown attribute: {field}")
                continue
            value = getattr(setting, field)
            log.debug(f"Updating {field} with value {value} of type {type(value)} in setting {setting_key(setting)}")
            if isinstance(value, bool):
                entry.delete(0, tk.END)
                entry.insert(0, "True" if value else "False")
            elif isinstance(value, (int, float)):
                entry.delete(0, tk.END)
                entry.insert(0, str(value))
            else:
                entry.delete(0, tk.END)
                entry.insert(0, value)

    def _scroll_to_settings(self, key: str) -> None:
        """Scroll the settings canvas so the section for *key* is visible."""
        frame = self.settings_section_frames.get(key)
        if not frame:
            return
        self.window.update_idletasks()
        total = self.settings_frame.winfo_height()
        if total > 0:
            self.canvas.yview_moveto(frame.winfo_y() / total)

    def _scroll_to_selected_builtin_plugin(self) -> None:
        sel = self.listbox_default_plugins.curselection()
        if sel:
            plugin = self.listbox_default_plugins.get(sel[0])
            self._scroll_to_settings(f"PluginSettings{plugin}")

    def _scroll_to_selected_plugin(self) -> None:
        sel = self.listbox_community_plugins.curselection()
        if sel:
            name = self.listbox_community_plugins.get(sel[0])
            self._scroll_to_settings(f"PluginSettingscommunity_{name}")

    def hide(self):
        """Hide the window instead of destroying it"""
        self.window.withdraw()
        
    def show(self):
        """Show the hidden window"""
        self.window.deiconify()
        self.window.lift()
        self.window.focus_set()
        # Update widgets with latest settings
        self.update_widgets(PerceptionSettings)
        self.update_widgets(PlanningSettings)
        self.update_widgets(ControlSettings)
        self.update_widgets(ExecutionSettings)
