"""Shared settings editor UI for AVLite Tk apps."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from avlite.plugins.p60_visualizer_tk.p61_visualizer_app import VisualizerApp
import importlib
import logging
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

from avlite.c10_perception import c19_settings
from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c20_planning import c29_settings
from avlite.c20_planning.c29_settings import PlanningSettings
from avlite.c30_control import c39_settings
from avlite.c30_control.c39_settings import ControlSettings
from avlite.c40_execution import c49_settings
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c60_apps.c69_settings import AppSettings
from avlite.c60_apps.c68_paths import ConfigPaths, PluginPaths
from avlite.c60_apps.c63_plugins import (
    import_plugin_modules,
    list_plugins,
    load_builtin_plugin_settings,
    load_community_plugin_setting,
    plugin_module_prefix,
    reload_lib,
    unregister_plugin_package,
)
from avlite.c60_apps.c62_factory import load_stack_settings
from avlite.plugins.p60_visualizer_tk.settings import (
    PluginSettings,
    VisualizationSettings,
    sync_stack_settings_to_ui,
)
from avlite.c60_apps.c65_setting_utils import (
    can_remove_builtin_plugin,
    delete_profile as delete_profile_file,
    dev_mode_export_warning,
    export_profile,
    import_profile,
    list_profiles,
    load_setting,
    order_profiles_for_dropdown,
    profile_file_path,
    rename_profile as rename_profile_file,
    save_setting,
)
from avlite.plugins.p60_visualizer_tk.p65_ui_lib import (
    BUTTON_TOOLTIPS,
    HoverTooltip,
    ThemedInputDialog,
    ThemedReadOnlyTwoFieldDialog,
    ThemedTwoInputDialog,
    TkSettingsBinder,
    apply_ttk_theme,
    attach_schema_tooltip,
    attach_tooltip,
    get_dpi_scale,
    scaled,
    setup_dpi,
)
from avlite.c60_apps.c64_settings_schema import (
    apply_validated_to_setting,
    field_tooltip_text,
    schema_of,
    setting_key,
)
from avlite.plugins.p60_visualizer_tk.p63_plugins_app import CommunityPluginsApp

log = logging.getLogger(__name__)

_STACK_SETTINGS_MODULES = {
    type(PerceptionSettings): c19_settings,
    type(PlanningSettings): c29_settings,
    type(ControlSettings): c39_settings,
    type(ExecutionSettings): c49_settings,
}


class SettingsHost(Protocol):
    setting: VisualizationSettings
    validate_cmd: tuple

    def load_settings(self, only_stack: bool = False, profile: str | None = None) -> None: ...

    def on_stack_settings_changed(self) -> None: ...


class SettingWindow:
    """
    A view to display and edit settings.
    """
    def __init__(self, host: SettingsHost, show_visualization_settings: bool = True):
        self.host = host
        self.show_visualization_settings = show_visualization_settings
        self.setting = host.setting
        self.window = tk.Toplevel(host)
        self._dpi_scale = get_dpi_scale(self.window, parent=host)
        self.window.geometry(f"{scaled(550, self._dpi_scale)}x{scaled(450, self._dpi_scale)}")

        self.frame = ttk.Frame(self.window)
        self.frame.pack(fill=tk.BOTH, expand=True)
        self.frame.rowconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=1)

        profile_ext_frame = ttk.Frame(self.frame)
        profile_ext_frame.grid(row=0, column=0, sticky="nswe", padx=10, pady=10)
        settings_frame = ttk.Frame(self.frame)
        settings_frame.grid(row=0, column=1, rowspan=1, sticky="nsew", padx=10, pady=10)
        additional_setting_frame = ttk.LabelFrame(self.frame, text="Additional Settings")
        if show_visualization_settings:
            additional_setting_frame.grid(row=1, column=0, columnspan=2, sticky="sew", padx=5, pady=5)

        self._bind_window_keys()
        self._build_profile_panel(profile_ext_frame)
        self._build_settings_canvas(settings_frame)
        if show_visualization_settings:
            self._build_viz_settings_panel(additional_setting_frame)

        footer_row = 2 if show_visualization_settings else 1
        footer_frame = ttk.Frame(self.frame)
        footer_frame.grid(row=footer_row, column=0, columnspan=2, sticky="se", padx=5, pady=5)
        self._build_action_buttons(footer_frame)

        if show_visualization_settings:
            self.window.protocol("WM_DELETE_WINDOW", self.hide)
        else:
            self.window.protocol("WM_DELETE_WINDOW", self._close_standalone)

    def _build_action_buttons(self, parent: ttk.Frame) -> None:
        close_cmd = self.hide if self.show_visualization_settings else self._close_standalone
        btn_settings_close = ttk.Button(parent, text="Close", width=5, underline=0, command=close_cmd)
        btn_settings_close.pack(side=tk.RIGHT, padx=5)
        attach_tooltip(btn_settings_close, BUTTON_TOOLTIPS["settings_close"])
        btn_settings_save = ttk.Button(parent, text="Save", width=5, underline=0, command=self.save_profile)
        btn_settings_save.pack(side=tk.RIGHT, padx=5)
        attach_tooltip(btn_settings_save, BUTTON_TOOLTIPS["settings_save"])

    def _widget_key(self, setting, plugin_name: str = "") -> str:
        return setting_key(setting) + plugin_name

    def _refresh_profile_dropdowns(self, *, select: str | None = None) -> str:
        """Re-read profile names from the active config target; update all comboboxes."""
        profiles = list_profiles(self.host.setting)
        self.host.setting.profile_list = profiles
        combos = [
            self.profile_dropdown_menu,
            self.next_profile_dropdown_menu,
        ]
        shortcut = getattr(self.host, "setting_shortcut_view", None)
        if shortcut is not None:
            combos.append(shortcut.profile_dropdown_menu)
        for combo in combos:
            combo["values"] = profiles
        if select and select in profiles:
            current = select
            self.host.setting.c60_selected_profile.set(current)
        else:
            current = self.host.setting.c60_selected_profile.get()
            if current not in profiles:
                current = profiles[0] if profiles else "default"
                self.host.setting.c60_selected_profile.set(current)
        if self.host.setting.p60_next_profile.get() not in profiles:
            self.host.setting.p60_next_profile.set(current)
        self._update_profile_action_states(current)
        return current

    def _hosting_plugin_name(self) -> str:
        return getattr(self.host, "hosting_plugin_name", "p60_visualizer_tk")

    def _update_profile_action_states(self, profile: str | None = None) -> None:
        profile = profile or self.host.setting.c60_selected_profile.get()
        if profile == "default":
            self._btn_profile_delete.state(["disabled"])
        else:
            self._btn_profile_delete.state(["!disabled"])

    def _update_builtin_plugin_action_states(self) -> None:
        selected = self.listbox_default_plugins.curselection()
        if not selected:
            self._btn_remove_builtin.state(["disabled"])
            return
        plugin_name = self.listbox_default_plugins.get(selected)
        if can_remove_builtin_plugin(plugin_name, self._hosting_plugin_name()) is not None:
            self._btn_remove_builtin.state(["disabled"])
        else:
            self._btn_remove_builtin.state(["!disabled"])

    def _edit_repo_configs_toggle(self) -> None:
        enabled = self._edit_repo_configs_var.get()
        if enabled and not messagebox.askyesno(
            "Edit repository configs",
            f"Core stack and built-in plugin settings will use files under\n"
            f"{ConfigPaths.bundled_dir()}\n"
            f"instead of your user config dir ({ConfigPaths.user_dir()}).\n\n"
            f"Community and member plugin settings always stay in your user "
            f"config directory.\n\nContinue?",
            parent=self.window,
        ):
            self._edit_repo_configs_var.set(False)
            return
        ConfigPaths.set_repo_target(enabled)
        profile = self._refresh_profile_dropdowns()
        self.host.load_settings(profile=profile)
        self.load_profile(profile)

    def _bind_window_keys(self) -> None:
        self.window.bind("<Control-s>", lambda e: self.save_profile())
        self.window.bind("k", lambda e: self.canvas.yview_scroll(-1, "units"))
        self.window.bind("j", lambda e: self.canvas.yview_scroll(1, "units"))
        self.window.bind("<Control-u>", lambda e: self.canvas.yview_scroll(-5, "units"))
        self.window.bind(
            "<Control-d>",
            lambda e: self.canvas.yview_scroll(int(0.5 * self.host.setting.p68_log_view_default_height.get()), "units"),
        )
        self.window.bind("G", lambda e: self.canvas.yview_moveto(1.0))
        self.window.bind("g", lambda e: self.canvas.yview_moveto(0.0))

    def _build_profile_panel(self, profile_ext_frame: ttk.Frame) -> None:
        _s = self._dpi_scale
        profile_ext_frame.rowconfigure(8, weight=1)

        ttk.Label(profile_ext_frame, text="Execution Profiles",style="Big.TLabel").grid(row=0, column=0, sticky="w", columnspan=3, padx=10, pady=5)
        ttk.Label(profile_ext_frame, text="Load Profile").grid(row=1, column=0, padx=5, pady=5)
        self.profile_dropdown_menu = ttk.Combobox(profile_ext_frame, textvariable=self.host.setting.c60_selected_profile, state="readonly",)
        self.profile_dropdown_menu["values"] = self.host.setting.profile_list
        # self.global_planner_dropdown_menu.current(0)  
        self.profile_dropdown_menu.state(["readonly"])
        self.profile_dropdown_menu.bind("<<ComboboxSelected>>", self.__on_profile_dropdown_change)
        self.profile_dropdown_menu.grid(row=1, column=1, columnspan=2, padx=5, pady=5)

        btn_profile_new = ttk.Button(profile_ext_frame, text="New", command=self.create_profile)
        btn_profile_new.grid(row=2, column=0, padx=5, pady=5, sticky="we")
        attach_tooltip(btn_profile_new, BUTTON_TOOLTIPS["profile_new"])
        self._btn_profile_delete = ttk.Button(profile_ext_frame, text="Delete", command=self.delete_profile)
        self._btn_profile_delete.grid(row=2, column=1, padx=5, pady=5, sticky="we")
        attach_tooltip(self._btn_profile_delete, BUTTON_TOOLTIPS["profile_delete"])
        btn_profile_save = ttk.Button(profile_ext_frame, text="Save", underline=0, command=self.save_profile)
        btn_profile_save.grid(row=2, column=2, padx=5, pady=5, sticky="we")
        attach_tooltip(btn_profile_save, BUTTON_TOOLTIPS["profile_save"])
        btn_profile_export = ttk.Button(profile_ext_frame, text="Export", command=self.export_profile_file)
        btn_profile_export.grid(row=3, column=0, padx=5, pady=5, sticky="we")
        attach_tooltip(btn_profile_export, BUTTON_TOOLTIPS["profile_export"])
        btn_profile_import = ttk.Button(profile_ext_frame, text="Import", command=self.import_profile_file)
        btn_profile_import.grid(row=3, column=1, padx=5, pady=5, sticky="we")
        attach_tooltip(btn_profile_import, BUTTON_TOOLTIPS["profile_import"])
        btn_profile_rename = ttk.Button(profile_ext_frame, text="Rename", command=self.rename_profile)
        btn_profile_rename.grid(row=3, column=2, padx=5, pady=5, sticky="we")
        attach_tooltip(btn_profile_rename, BUTTON_TOOLTIPS["profile_rename"])

        ttk.Label(profile_ext_frame, text="Cycle Next (Shortcut F)").grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        self.next_profile_dropdown_menu = ttk.Combobox(profile_ext_frame, width=10, textvariable=self.host.setting.p60_next_profile, state="readonly",)
        self.next_profile_dropdown_menu["values"] = self.host.setting.profile_list
        self.next_profile_dropdown_menu.state(["readonly"])
        attach_schema_tooltip(self.next_profile_dropdown_menu, PluginSettings, "p60_next_profile")
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
        if ConfigPaths.can_edit_bundled():
            self._edit_repo_configs_var = tk.BooleanVar(value=ConfigPaths.is_repo_target())
            cb_edit_repo = ttk.Checkbutton(
                profile_ext_frame, text="Edit repository configs", variable=self._edit_repo_configs_var,
                command=self._edit_repo_configs_toggle,
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
            plugin_frame, text="Load Plugins", variable=self.host.setting.c62_load_plugins,
            command=self._on_load_plugins_toggle,
        )
        cb_load_plugins.grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        attach_schema_tooltip(cb_load_plugins, AppSettings, "c62_load_plugins")

        # built-in plugins
        ttk.Label(plugin_frame, text="Plugins").grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        self.listbox_default_plugins = tk.Listbox(
            plugin_frame, height=listbox_height, selectmode=tk.SINGLE, exportselection=False, width=30,
        )
        self.listbox_default_plugins.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        # Convert comma-separated string to list items

        for plugin in AppSettings.c62_default_plugins:
            self.listbox_default_plugins.insert(tk.END, plugin)
        
        btn_reset_plugins = ttk.Button(plugin_frame, text="Reset Plugins", command=self.reset_default_plugins)
        btn_reset_plugins.grid(row=3, column=0, sticky="we", padx=5, pady=5)
        attach_tooltip(btn_reset_plugins, BUTTON_TOOLTIPS["plugins_reset_builtin"])
        self._btn_remove_builtin = ttk.Button(plugin_frame, text="Remove Plugin", command=self.remove_default_plugin)
        self._btn_remove_builtin.grid(row=3, column=1, sticky="we", padx=5, pady=5)
        attach_tooltip(self._btn_remove_builtin, BUTTON_TOOLTIPS["plugins_remove_builtin"])


        # community plugins
        ttk.Label(plugin_frame, text="Community Plugins").grid(row=4, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        self.listbox_community_plugins = tk.Listbox(
            plugin_frame, height=listbox_height, selectmode=tk.SINGLE, exportselection=False, width=30,
        )
        self.listbox_community_plugins.grid(row=5, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        # Convert comma-separated string to list items

        for plugin in AppSettings.c62_community_plugins.keys() if self.host.setting.c62_load_plugins.get() else []:
            self.listbox_community_plugins.insert(tk.END, plugin)

        self.listbox_community_plugins.bind("<Double-Button-1>", lambda e: self.edit_community_plugin())
        self.listbox_community_plugins.bind("<<ListboxSelect>>", lambda e: self._scroll_to_selected_plugin())
        self.listbox_default_plugins.bind("<Double-Button-1>", lambda e: self.edit_default_plugin())
        self.listbox_default_plugins.bind("<<ListboxSelect>>", lambda e: self._on_builtin_plugin_select())
        self._update_profile_action_states()
        self._update_builtin_plugin_action_states()



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

    def _build_settings_canvas(self, settings_frame: ttk.Frame) -> None:
        _s = self._dpi_scale
        settings_frame.columnconfigure(0, weight=1)
        settings_frame.rowconfigure(0, weight=1)

        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        settings_frame.bind_all("<MouseWheel>", _on_mousewheel)
        settings_frame.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
        settings_frame.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))

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

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.settings_frame.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.settings_frame, anchor="nw")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.window.update_idletasks()
        self.window.minsize(scaled(500, _s), scaled(400, _s))

        ttk.Label(self.settings_frame, text="Core Stack Settings", style="Big.TLabel").pack(
            anchor=tk.W, padx=5, pady=5
        )
        self.widget_entries = {}
        self.settings_section_frames = {}
        self.create_widgets(PerceptionSettings, "Perception Settings")
        self.create_widgets(PlanningSettings, "Planning Settings")
        self.create_widgets(ControlSettings, "Control Settings")
        self.create_widgets(ExecutionSettings, "Execution Settings")
        if self.host.setting.c62_load_plugins.get():
            ttk.Separator(self.settings_frame, orient="horizontal").pack(fill="x", pady=10)
            ttk.Label(self.settings_frame, text="Plugin Settings", style="Big.TLabel").pack(
                anchor=tk.W, padx=5, pady=5
            )
            self.create_plugin_widgets()

        self._cp_sep = ttk.Separator(self.settings_frame, orient="horizontal")
        self._cp_label = ttk.Label(
            self.settings_frame, text="Community Plugin Settings", style="Big.TLabel"
        )
        self.create_community_plugin_widgets()

    def _build_viz_settings_panel(self, additional_setting_frame: ttk.LabelFrame) -> None:
        ## UI Elements for Visualize - Checkboxes
        additional_setting_row_1 = ttk.Frame(additional_setting_frame)
        ttk.Label(additional_setting_row_1, text="Local Plan Plot View:").pack(anchor=tk.W, side=tk.LEFT, padx=5)
        additional_setting_row_1.pack(fill=tk.X)
        for text, field in (
            ("Legend", "p66_show_legend"),
            ("Locations", "p66_show_past_locations"),
            ("Global Plan", "p66_show_global_plan"),
            ("Local Plan", "p66_show_local_plan"),
            ("Local Lattice", "p66_show_local_lattice"),
            ("State", "p66_show_state"),
        ):
            cb = ttk.Checkbutton(
                additional_setting_row_1, text=text,
                variable=getattr(self.host.setting, field), command=self.host.update_ui,
            )
            cb.pack(anchor=tk.W, side=tk.LEFT)
            attach_schema_tooltip(cb, VisualizationSettings, field)

        for text, field in (
            ("Follow Planner in Global", "p66_global_view_follow_planner"),
            ("Follow Planner in Frenet", "p66_frenet_view_follow_planner"),
        ):
            cb = ttk.Checkbutton(additional_setting_row_1, text=text, variable=getattr(self.host.setting, field))
            cb.pack(side=tk.LEFT)
            attach_schema_tooltip(cb, VisualizationSettings, field)

        additional_setting_row_1b = ttk.Frame(additional_setting_frame)
        additional_setting_row_1b.pack(fill=tk.X)
        ttk.Label(additional_setting_row_1b, text="Local Plan Plot View:").pack(anchor=tk.W, side=tk.LEFT, padx=5)
        for text, field in (
            ("Local Global View", "p67_show_local_global_view"),
            ("Local Frenet View", "p67_show_local_frenet_view"),
            ("LiDAR in Global", "p66_show_lidar_global"),
            ("LiDAR in Frenet", "p66_show_lidar_frenet"),
            ("Clustered Pts", "p66_show_lidar_clusters"),
            ("Race Boundary", "p66_show_race_boundary"),
        ):
            cb = ttk.Checkbutton(
                additional_setting_row_1b, text=text,
                variable=getattr(self.host.setting, field), command=self.host.update_ui,
            )
            cb.pack(side=tk.LEFT)
            attach_schema_tooltip(cb, VisualizationSettings, field)

        additional_setting_row_1c = ttk.Frame(additional_setting_frame)
        additional_setting_row_1c.pack(fill=tk.X)
        ttk.Label(additional_setting_row_1c, text="Global Plan Plot View:").pack(anchor=tk.W, side=tk.LEFT, padx=5)
        cb_plan_boundaries = ttk.Checkbutton(
            additional_setting_row_1c,
            text="Plan boundaries",
            variable=self.host.setting.p66_show_global_plan_boundaries,
            command=self.host.update_ui,
        )
        cb_plan_boundaries.pack(side=tk.LEFT)
        attach_schema_tooltip(cb_plan_boundaries, VisualizationSettings, "p66_show_global_plan_boundaries")
        ttk.Label(additional_setting_row_1c, text="Velocity scale:").pack(side=tk.LEFT, padx=(10, 5))
        velocity_scale_cb = ttk.Combobox(
            additional_setting_row_1c,
            textvariable=self.host.setting.p66_global_plan_velocity_scale,
            values=("relative", "absolute"),
            state="readonly",
            width=10,
        )
        velocity_scale_cb.pack(side=tk.LEFT)
        velocity_scale_cb.bind("<<ComboboxSelected>>", lambda e: self.host.update_ui())
        attach_schema_tooltip(velocity_scale_cb, VisualizationSettings, "p66_global_plan_velocity_scale")

        additional_setting_row_1d = ttk.Frame(additional_setting_frame)
        additional_setting_row_1d.pack(fill=tk.X)
        ttk.Label(additional_setting_row_1d, text="Perception:").pack(anchor=tk.W, side=tk.LEFT, padx=5)
        cb_show_prediction = ttk.Checkbutton(
            additional_setting_row_1d, text="Show prediction",
            variable=self.host.setting.p67_show_prediction, command=self.host.update_ui,
        )
        cb_show_prediction.pack(side=tk.LEFT)
        attach_schema_tooltip(cb_show_prediction, VisualizationSettings, "p67_show_prediction")
        cb_occupancy_flow = ttk.Checkbutton(
            additional_setting_row_1d, text="Occupancy flow",
            variable=self.host.setting.p67_show_occupancy_flow, command=self.host.update_ui,
        )
        cb_occupancy_flow.pack(side=tk.LEFT)
        attach_schema_tooltip(cb_occupancy_flow, VisualizationSettings, "p67_show_occupancy_flow")
        ttk.Label(additional_setting_row_1d, text="Mapping:").pack(side=tk.LEFT, padx=(10, 5))
        # mapping_cb = ttk.Combobox(
        #     additional_setting_row_1d,
        #     textvariable=self.host.setting.mapping_type,
        #     values=("",) + tuple(MappingStrategy.registry.keys()),
        #     state="readonly",
        #     width=12,
        # )
        # mapping_cb.pack(side=tk.LEFT)
        # mapping_cb.bind("<<ComboboxSelected>>", lambda e: self.host.reload_stack(reload_code=False))
        # attach_schema_tooltip(mapping_cb, ExecutionSettings, "c40_mapping")

        additional_setting_row_2 = ttk.Frame(additional_setting_frame)
        additional_setting_row_2.pack(fill=tk.X, padx=5)

        ttk.Label(additional_setting_row_2, text="Log View:").pack(anchor=tk.W, side=tk.LEFT, padx=0)
        cb_expand_log = ttk.Checkbutton(additional_setting_row_2, text="Expand Log View", variable=self.host.setting.p68_log_view_expanded)
        cb_expand_log.pack(side=tk.LEFT)
        attach_schema_tooltip(cb_expand_log, VisualizationSettings, "p68_log_view_expanded")

        ttk.Label(additional_setting_row_2, text="Default Log Height:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(additional_setting_row_2, textvariable=self.host.setting.p68_log_view_default_height, width=5,
                  validatecommand=self.host.validate_cmd).pack(side=tk.LEFT, padx=5)

        ttk.Label(additional_setting_row_2, text="Expanded Log Height").pack(side=tk.LEFT, padx=5)
        ttk.Entry(additional_setting_row_2, textvariable=self.host.setting.p68_log_view_expended_height, width=5,
                  validatecommand=self.host.validate_cmd).pack(side=tk.LEFT, padx=5)

        additional_setting_row_3 = ttk.Frame(additional_setting_frame)
        additional_setting_row_3.pack(fill=tk.X, padx=0)
        ttk.Label(additional_setting_row_3, text="Menu bar:").pack(anchor=tk.W, side=tk.LEFT, padx=5)
        cb_hide_menubar = ttk.Checkbutton(additional_setting_row_3, text="Hide", variable=self.host.setting.p60_hide_menubar)
        cb_hide_menubar.pack(anchor=tk.W, side=tk.LEFT)
        attach_schema_tooltip(cb_hide_menubar, VisualizationSettings, "p60_hide_menubar")

    def refresh_widgets(self) -> None:
        """Reload stack and plugin settings from disk into all editor widgets."""
        self.update_widgets(PerceptionSettings)
        self.update_widgets(PlanningSettings)
        self.update_widgets(ControlSettings)
        self.update_widgets(ExecutionSettings)

        if self.host.setting.c62_load_plugins.get():
            for plugin in AppSettings.c62_default_plugins:
                try:
                    module = importlib.import_module(f"{plugin_module_prefix(plugin)}.settings")
                    PluginSettings = getattr(module, "PluginSettings")
                    load_setting(
                        PluginSettings,
                        profile=self.host.setting.c60_selected_profile.get(),
                        binder=TkSettingsBinder(),
                    )
                    self.update_widgets(PluginSettings, plugin_name=plugin)
                    log.debug("loaded plugin settings for %s", plugin)
                except Exception as e:
                    log.error("Failed to load plugin settings for %s: %s", plugin, e)

            profile = self.host.setting.c60_selected_profile.get()
            for name, stored in AppSettings.c62_community_plugins.items():
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

    def rebuild_plugin_sections(self) -> None:
        """Create or tear down built-in and community plugin setting sections."""
        if self.host.setting.c62_load_plugins.get():
            self.create_plugin_widgets()
            self.create_community_plugin_widgets()
            return

        for plugin in AppSettings.c62_default_plugins:
            plugin_key = f"PluginSettings{plugin}"
            if plugin_key in self.widget_entries:
                entry_dict = self.widget_entries[plugin_key]
                if entry_dict:
                    first_widget = next(iter(entry_dict.values()))
                    parent_frame = first_widget.master
                    parent_frame.pack_forget()
                del self.widget_entries[plugin_key]
                log.debug("Removed widgets for %s", plugin_key)
        self.plugin_widget_created = False

        for name in list(AppSettings.c62_community_plugins.keys()):
            cp_key = f"PluginSettingscommunity_{name}"
            if cp_key in self.widget_entries:
                entry_dict = self.widget_entries[cp_key]
                if entry_dict:
                    first_widget = next(iter(entry_dict.values()))
                    parent_frame = first_widget.master
                    parent_frame.pack_forget()
                del self.widget_entries[cp_key]
                log.debug("Removed community plugin widgets for %s", name)

        self._cp_sep.pack_forget()
        self._cp_label.pack_forget()
        self._cp_widget_created = False

    def reset_community_plugins(self):
        """Reset community plugins to all plugins installed under the plugins directory."""
        log.info("Resetting community plugins to installed set.")
        AppSettings.c62_community_plugins = PluginPaths.installed_map()
        self.update_community_plugin_list()
        if self.host.setting.c62_load_plugins.get():
            self.update_community_plugin_widgets()

    def reset_default_plugins(self):
        """ Reset the default plugins to the source code defaults. """
        log.info("Resetting default plugins to source code defaults.")

        self.listbox_default_plugins.delete(0, tk.END)
        AppSettings.c62_default_plugins = list_plugins()

        for plugin in AppSettings.c62_default_plugins:
            self.listbox_default_plugins.insert(tk.END, plugin)

        import_plugin_modules(plugins_filter=AppSettings.c62_default_plugins)
        self.host.on_stack_settings_changed()
        self._update_builtin_plugin_action_states()

    def remove_default_plugin(self):
        """ Remove the selected default plugin from the list. """
        selected = self.listbox_default_plugins.curselection()
        if not selected:
            log.warning("No plugin selected to remove.")
            return
        plugin_name = self.listbox_default_plugins.get(selected)
        blocked = can_remove_builtin_plugin(plugin_name, self._hosting_plugin_name())
        if blocked is not None:
            messagebox.showwarning("Remove Plugin", blocked, parent=self.window)
            return
        if plugin_name in AppSettings.c62_default_plugins:
            AppSettings.c62_default_plugins.remove(plugin_name)
            self.listbox_default_plugins.delete(selected)
            unregister_plugin_package(plugin_name)
            self.host.on_stack_settings_changed()
            log.info(f"Removed and unloaded plugin: {plugin_name}")
            self._update_builtin_plugin_action_states()
        else:
            log.warning(f"Plugin {plugin_name} not found in default plugins.")

    def add_community_plugin(self):
        dialog = ThemedTwoInputDialog(self.host, "Community Plugins", "Package Name", "Package Directory")
        name, dir =  dialog.result if dialog.result else (None, None)
        if not name:
            return

        log.info(f"Adding plugin: {name}")
        AppSettings.c62_community_plugins[name] = PluginPaths.normalize_stored(name, dir)
        self.listbox_community_plugins.insert(tk.END, name)
        
    def delete_community_plugin(self):
        selected = self.listbox_community_plugins.curselection()
        if selected:
            plugin_name = self.listbox_community_plugins.get(selected)
            AppSettings.c62_community_plugins.pop(plugin_name, None)
            self.listbox_community_plugins.delete(selected)
            log.info(f"Deleted community plugin: {plugin_name}")
        else:
            log.warning("No community plugin selected to delete.")

    def _plugin_settings_location(self, plugin_name: str) -> str:
        profile = self.host.setting.c60_selected_profile.get()
        path = PluginPaths.format_display(profile_file_path(profile, for_write=False))
        return f"{path} (plugins.{plugin_name})"

    def edit_default_plugin(self):
        selected = self.listbox_default_plugins.curselection()
        if not selected:
            log.warning("No plugin selected to edit.")
            return

        plugin_name = self.listbox_default_plugins.get(selected[0])
        settings_path = (
            self._plugin_settings_location(plugin_name)
            if load_builtin_plugin_settings(plugin_name) is not None
            else "\u2014"
        )
        ThemedReadOnlyTwoFieldDialog(
            self.host,
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
            self.host,
            "Community Plugin",
            "Package Name",
            "Settings file",
            plugin_name,
            self._plugin_settings_location(plugin_name),
        )

    def update_community_plugin_list(self):
        """ Load the community plugins from the settings. """

        self.listbox_community_plugins.delete(0, tk.END)
        for name, dir in AppSettings.c62_community_plugins.items():
            self.listbox_community_plugins.insert(tk.END, name)

    def open_plugins_window(self):
        """Open the community plugins manager and refresh the list on close."""
        app = CommunityPluginsApp.open(parent=self.window, host=self.host)
        app.window.bind("<Destroy>", lambda _e: self.update_community_plugin_list(), add="+")


    def create_profile(self):
        """ Load a profile from the settings. """
        
        # text = simpledialog.askstring("Profile", "Enter profile Name")
        dialog = ThemedInputDialog(self.host, "Profile", "Name")
        text =  dialog.result.strip() if dialog.result else None
        if not text:
            return
        log.info(f"Creating profile: {text}")
        self.host.setting.c60_selected_profile.set(text)
        self.host.setting.profile_list.append(text)
        self.host.setting.profile_list = order_profiles_for_dropdown(self.host.setting.profile_list)
        self.profile_dropdown_menu["values"] = self.host.setting.profile_list
        shortcut = getattr(self.host, "setting_shortcut_view", None)
        if shortcut is not None:
            shortcut.profile_dropdown_menu["values"] = self.host.setting.profile_list
        self.next_profile_dropdown_menu["values"] = self.host.setting.profile_list

        self.save_profile()


    def export_profile_file(self):
        """Export the selected profile to a single YAML file."""
        profile = self.host.setting.c60_selected_profile.get()
        if not messagebox.askyesno(
            "Export profile",
            f"Export reads from saved files on disk.\nSave profile '{profile}' first if you have unsaved changes.\n\nContinue?",
            parent=self.window,
        ):
            return
        options = self._ask_export_options()
        if options is None:
            return
        include_stack, include_app, include_plugins = options
        warning = dev_mode_export_warning(AppSettings.c62_community_plugins)
        if warning and not messagebox.askyesno(
            "Export profile",
            warning + "\n\nContinue export?",
            parent=self.window,
        ):
            return
        out_path = filedialog.asksaveasfilename(
            parent=self.window,
            title="Export profile",
            defaultextension=".yaml",
            initialfile=f"{profile}.yaml",
            filetypes=[("AVLite profile", "*.yaml"), ("All files", "*.*")],
        )
        if not out_path:
            return
        try:
            count = export_profile(
                profile,
                out_path,
                include_stack=include_stack,
                include_app=include_app,
                include_plugins=include_plugins,
            )
        except ValueError as e:
            messagebox.showerror("Export profile", str(e), parent=self.window)
            return
        except OSError as e:
            messagebox.showerror("Export profile", f"Failed to write file: {e}", parent=self.window)
            return
        messagebox.showinfo(
            "Export profile",
            f"Exported profile '{profile}' ({count} section(s)) to\n{out_path}",
            parent=self.window,
        )

    def _ask_export_options(self) -> tuple[bool, bool, bool] | None:
        """Modal dialog: returns (include_stack, include_app, include_plugins) or None if cancelled."""
        dlg = tk.Toplevel(self.window)
        dlg.title("Export options")
        dlg.transient(self.window)
        try:
            bg = ttk.Style(dlg).lookup("TFrame", "background")
            if bg:
                dlg.configure(background=bg)
        except tk.TclError:
            pass
        include_stack = tk.BooleanVar(value=True)
        include_app = tk.BooleanVar(value=True)
        include_plugins = tk.BooleanVar(value=True)
        ttk.Label(dlg, text="Choose what to include in the exported profile:").pack(
            padx=12, pady=(12, 6), anchor="w"
        )
        ttk.Checkbutton(
            dlg,
            text="Stack settings (perception, planning, control, execution)",
            variable=include_stack,
        ).pack(padx=12, anchor="w")
        ttk.Checkbutton(
            dlg, text="App settings (plugin list, active profile)", variable=include_app
        ).pack(padx=12, anchor="w")
        ttk.Checkbutton(
            dlg, text="Plugin settings (built-in and community)", variable=include_plugins
        ).pack(padx=12, anchor="w")
        state = {"ok": False}
        buttons = ttk.Frame(dlg)
        buttons.pack(padx=12, pady=12, fill="x")

        def _ok() -> None:
            state["ok"] = True
            dlg.destroy()

        ttk.Button(buttons, text="Export", command=_ok).pack(side="right")
        ttk.Button(buttons, text="Cancel", command=dlg.destroy).pack(side="right", padx=(0, 6))
        dlg.protocol("WM_DELETE_WINDOW", dlg.destroy)
        dlg.grab_set()
        self.window.wait_window(dlg)
        if not state["ok"]:
            return None
        return include_stack.get(), include_app.get(), include_plugins.get()

    def import_profile_file(self):
        """Import a profile from a single YAML file."""
        in_path = filedialog.askopenfilename(
            parent=self.window,
            title="Import profile",
            filetypes=[("AVLite profile", "*.yaml"), ("All files", "*.*")],
        )
        if not in_path:
            return
        imported_name = Path(in_path).stem
        overwrite = False
        if imported_name in self.host.setting.profile_list:
            if not messagebox.askyesno(
                "Import profile",
                f"Profile '{imported_name}' already exists.\nOverwrite it?",
                parent=self.window,
            ):
                return
            overwrite = True

        try:
            profile_name = import_profile(in_path, overwrite=overwrite)
        except ValueError as e:
            messagebox.showerror("Import profile", str(e), parent=self.window)
            return
        except OSError as e:
            messagebox.showerror("Import profile", f"Failed to import: {e}", parent=self.window)
            return

        profile_name = self._refresh_profile_dropdowns(select=profile_name)
        self.load_profile(profile_name)
        messagebox.showinfo(
            "Import profile",
            f"Imported profile '{profile_name}'.",
            parent=self.window,
        )


    def delete_profile(self):
        """ Delete a profile (single per-profile YAML file). """

        profile = self.host.setting.c60_selected_profile.get()
        if profile == "default":
            messagebox.showwarning("Delete", "Cannot delete the 'default' profile.", parent=self.window)
            return
        if not messagebox.askyesno("Confirmation", f"Are you sure you want to delete {profile}?", parent=self.window):
            return
        log.info(f"Deleting profile: {profile}")
        if not delete_profile_file(profile):
            messagebox.showerror("Delete", f"Could not delete profile '{profile}'.", parent=self.window)
            return
        profile = self._refresh_profile_dropdowns(select="default")
        self.load_profile(profile)

    def rename_profile(self):
        """ Rename the selected profile (single per-profile YAML file). """

        old_name = self.host.setting.c60_selected_profile.get()
        if old_name == "default":
            messagebox.showwarning("Rename", "Cannot rename the 'default' profile.")
            return

        dialog = ThemedInputDialog(self.host, "Rename Profile", "New name", initial=old_name)
        new_name = dialog.result.strip() if dialog.result else None
        if not new_name or new_name == old_name:
            return
        if new_name in self.host.setting.profile_list:
            messagebox.showwarning("Rename", f"Profile '{new_name}' already exists.")
            return

        log.info(f"Renaming profile '{old_name}' to '{new_name}'")
        if not rename_profile_file(old_name, new_name):
            messagebox.showerror("Rename", f"Could not rename profile '{old_name}'.")
            return

        idx = self.host.setting.profile_list.index(old_name)
        self.host.setting.profile_list[idx] = new_name
        self.host.setting.profile_list = order_profiles_for_dropdown(self.host.setting.profile_list)
        self.host.setting.c60_selected_profile.set(new_name)
        self.profile_dropdown_menu["values"] = self.host.setting.profile_list
        self.next_profile_dropdown_menu["values"] = self.host.setting.profile_list
        shortcut = getattr(self.host, "setting_shortcut_view", None)
        if shortcut is not None:
            shortcut.profile_dropdown_menu["values"] = self.host.setting.profile_list



    def save_profile(self):
        """ Save the current settings to the selected profile. """

        log.info(f"Saving profile: {self.host.setting.c60_selected_profile.get()}")
        profile = self.host.setting.c60_selected_profile.get()
        binder = TkSettingsBinder()
        self.save_from_widgets(PerceptionSettings)
        save_setting(PerceptionSettings, profile=profile, binder=binder)
        self.save_from_widgets(PlanningSettings)
        save_setting(PlanningSettings, profile=profile, binder=binder)
        self.save_from_widgets(ControlSettings)
        save_setting(ControlSettings, profile=profile, binder=binder)
        self.save_from_widgets(ExecutionSettings)
        AppSettings.c62_community_plugins = PluginPaths.normalize_map(
            AppSettings.c62_community_plugins
        )
        save_setting(ExecutionSettings, profile=profile, binder=binder)
        self.host.setting.sync_app_to_singleton()
        save_setting(AppSettings, profile=profile, binder=binder)
        save_setting(self.host.setting, profile=profile, binder=binder)

        if self.host.setting.c62_load_plugins.get():
            for plugin in AppSettings.c62_default_plugins:
                try:
                    module = importlib.import_module(f"{plugin_module_prefix(plugin)}.settings")
                    PluginSettings = getattr(module, "PluginSettings")
                    self.save_from_widgets(PluginSettings, plugin_name=plugin)
                    save_setting(PluginSettings, profile=profile, binder=binder)
                except Exception as e:
                    log.error(f"Failed to save plugin settings for {plugin}: {e}")

        # Save community plugin settings
        if self.host.setting.c62_load_plugins.get():
            profile = self.host.setting.c60_selected_profile.get()
            for name, stored in AppSettings.c62_community_plugins.items():
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
        save_setting(self.host.setting, profile=profile, binder=binder)
    

    def load_profile(self, profile="default"):
        """ Load a profile from the settings. """

        log.info(f"loading profile: {profile}")
        binder = TkSettingsBinder()
        load_setting(AppSettings, profile=profile)
        self.host.setting.sync_app_from_singleton()
        load_stack_settings(profile=profile)
        load_setting(self.host.setting, profile=profile, binder=binder)
        sync_stack_settings_to_ui(self.host.setting)
        self.host.setting.c60_selected_profile.set(profile)
        ConfigPaths.set_startup_profile(profile)

        self.refresh_widgets()
        self.update_community_plugin_list()

        self.listbox_default_plugins.delete(0, tk.END)
        for plugin in AppSettings.c62_default_plugins:
            self.listbox_default_plugins.insert(tk.END, plugin)

        self._update_profile_action_states(profile)
        self._update_builtin_plugin_action_states()
        self.host.on_stack_settings_changed()

    def update_community_plugin_widgets(self):
        """Reload and refresh widgets for community plugins that have ``PluginSettings``."""
        if not self.host.setting.c62_load_plugins.get():
            return
        profile = self.host.setting.c60_selected_profile.get()
        for name, stored in AppSettings.c62_community_plugins.items():
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

    def __on_profile_dropdown_change(self, event):
        log.info(f"Selected profile: {event.widget.get()}")
        self.load_profile(event.widget.get())

    def _settings_module(self, setting, plugin_name: str = ""):
        if plugin_name:
            name = plugin_name.removeprefix("community_")
            return importlib.import_module(f"{plugin_module_prefix(name)}.settings")
        return _STACK_SETTINGS_MODULES.get(type(setting))

    def _schema_for_setting(self, setting, plugin_name: str = "", *, reload_module: bool = True):
        mod = self._settings_module(setting, plugin_name)
        if mod is None:
            return schema_of(setting)
        if reload_module:
            importlib.reload(mod)
        fresh = getattr(mod, setting_key(setting), None)
        return schema_of(fresh) if fresh is not None else schema_of(setting)

    def reset_section_to_source_defaults(
        self, setting, plugin_name: str = "", *, reload_module: bool = True
    ) -> None:
        """Reset one settings section to source-code schema defaults."""
        schema = self._schema_for_setting(setting, plugin_name, reload_module=reload_module)
        if schema is None:
            log.warning("No schema for %s", setting_key(setting))
            return
        apply_validated_to_setting(
            setting,
            schema.model_validate({}),
            binder=TkSettingsBinder(),
        )
        self.update_widgets(setting, plugin_name=plugin_name)

    def reset_to_to_source_stack_values(self, exclude_execution=False):
        """ Reset the stack settings to the source code defaults, except for the UI as it is using some 
            some instant variables for tkinter.
        """
        reload_lib(exclude_stack=True, reload_plugins=True)
        for layer in (PerceptionSettings, PlanningSettings, ControlSettings):
            self.reset_section_to_source_defaults(layer, reload_module=False)
        if not exclude_execution:
            self.reset_section_to_source_defaults(ExecutionSettings, reload_module=False)

        self.update_plugins_widgets()
        

    def _on_load_plugins_toggle(self):
        self.rebuild_plugin_sections()
        self.host.reload_stack(reload_code=False)

    def update_core_widgets(self):
        """Backward-compatible alias for :meth:`refresh_widgets`."""
        self.refresh_widgets()

    def update_plugins_widgets(self):
        """Backward-compatible alias; plugin refresh is handled by :meth:`refresh_widgets`."""
        pass

    def create_plugin_widgets(self):
        if hasattr(self, "plugin_widget_created") and self.plugin_widget_created:
            log.warning("Plugin widgets already created, skipping.")
            return

        for plugin in AppSettings.c62_default_plugins:
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
        if self.host.setting.c62_load_plugins.get():
            profile = self.host.setting.c60_selected_profile.get()
            for name, stored in AppSettings.c62_community_plugins.items():
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

        key = self._widget_key(setting, plugin_name)
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

        frame.columnconfigure(1, weight=1)
        reset_label = "Reset to plugin defaults" if plugin_name else "Reset to stack defaults"
        reset_tooltip = (
            BUTTON_TOOLTIPS["section_reset_plugin"]
            if plugin_name
            else BUTTON_TOOLTIPS["section_reset_stack"]
        )
        reset_btn = ttk.Button(
            frame,
            text=reset_label,
            command=lambda s=setting, pn=plugin_name: self.reset_section_to_source_defaults(s, pn),
        )
        reset_btn.grid(row=row, column=0, columnspan=2, sticky="ew", padx=5, pady=(8, 2))
        attach_tooltip(reset_btn, reset_tooltip)

    def save_from_widgets(self, setting, plugin_name=""):
        """ Save the settings from the widgets to the setting class. """

        if self._widget_key(setting, plugin_name) not in self.widget_entries:
            log.warning(f"No widgets found for setting: {setting_key(setting)}+{plugin_name}")
            return

        # log.warning(f"keys in widget_entries: {self.widget_entries.keys()}")
        for field, entry in self.widget_entries[self._widget_key(setting, plugin_name)].items():
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
        if self._widget_key(setting, plugin_name) not in self.widget_entries:
            log.warning(f"No widgets found for setting: {plugin_name} {setting_key(setting)}")
            return

        for field, entry in self.widget_entries[self._widget_key(setting, plugin_name)].items():
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

    def _on_builtin_plugin_select(self) -> None:
        self._update_builtin_plugin_action_states()
        self._scroll_to_selected_builtin_plugin()

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
        """Hide the window instead of destroying it."""
        self.window.withdraw()

    def _close_standalone(self) -> None:
        self.window.destroy()
        self.host.destroy()

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


class SettingShortcutView(ttk.LabelFrame):
    """Visualizer toolbar: profile dropdown, shortcuts, settings/plugins launchers."""

    def __init__(self, root: VisualizerApp):
        super().__init__(root, text="Settings")

        self.root: VisualizerApp = root
        self.root.bind("T", lambda e: self.open_settings_window())
        self.root.bind_all("Q", lambda e: self.root.quit())
        self.root.bind_all("R", lambda e: self.root.reload_stack())
        self.root.bind_all("F", lambda e: self.root.switch_profile())
        self.root.bind("S", lambda e: self.root.update_shortcut_mode(reverse=True))
        self.root.bind("<Control-s>", lambda e: self.save_settings())

        self.root.bind("<Control-plus>", lambda e: self.root.local_plan_plot_view.zoom_in_frenet())
        self.root.bind("<Control-minus>", lambda e: self.root.local_plan_plot_view.zoom_out_frenet())
        self.root.bind("<plus>", lambda e: self.root.local_plan_plot_view.zoom_in())
        self.root.bind("<minus>", lambda e: self.root.local_plan_plot_view.zoom_out())

        self.root.bind("x", lambda e: self.root.exec_visualize_view.toggle_exec())
        self.root.bind("c", lambda e: self.root.exec_visualize_view.step_exec())
        self.root.bind("t", lambda e: self.root.exec_visualize_view.reset_exec())

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

        self.root.bind("k", lambda e: self.root.log_view.log_area.yview_scroll(-1, "units"))
        self.root.bind("j", lambda e: self.root.log_view.log_area.yview_scroll(1, "units"))
        self.root.bind("<Control-u>", lambda e: self.root.log_view.log_area.yview_scroll(-5, "units"))
        self.root.bind(
            "<Control-d>",
            lambda e: self.root.log_view.log_area.yview_scroll(
                int(0.5 * self.root.setting.p68_log_view_default_height.get()), "units"
            ),
        )
        self.root.bind("G", lambda e: self.root.log_view.log_area.yview_moveto(1.0))
        self.root.bind("g", lambda e: self.root.log_view.log_area.yview_moveto(0.0))
        self.root.bind("<Up>", lambda e: self.root.log_view.log_area.yview_scroll(-1, "units"))
        self.root.bind("<Down>", lambda e: self.root.log_view.log_area.yview_scroll(1, "units"))

        self.root.bind("E", lambda e: self.root.log_view.update_log_view_height(reverse=True))
        self.root.bind("L", lambda e: self.root.log_view.clear_log())
        self.root.bind("<Escape>", lambda e: self.root.focus_set())

        btn_settings = ttk.Button(self, text="⚙", command=self.open_settings_window, width=2)
        btn_settings.pack(side=tk.RIGHT)
        attach_tooltip(btn_settings, BUTTON_TOOLTIPS["toolbar_settings"])
        btn_plugins = ttk.Button(self, text="Plugins", command=self.open_plugins_window)
        btn_plugins.pack(side=tk.RIGHT)
        attach_tooltip(btn_plugins, BUTTON_TOOLTIPS["toolbar_plugins"])
        btn_reload = ttk.Button(self, text="Reload Stack", command=self.root.reload_stack)
        btn_reload.pack(side=tk.RIGHT)
        attach_tooltip(btn_reload, BUTTON_TOOLTIPS["toolbar_reload_stack"])
        btn_reset = ttk.Button(self, text="Reset Settings", command=self.root.load_settings)
        btn_reset.pack(side=tk.RIGHT)
        attach_tooltip(btn_reset, BUTTON_TOOLTIPS["toolbar_reset_settings"])
        btn_save = ttk.Button(self, text="Save Settings", command=self.save_settings)
        btn_save.pack(side=tk.RIGHT)
        attach_tooltip(btn_save, BUTTON_TOOLTIPS["toolbar_save_settings"])

        self.profile_dropdown_menu = ttk.Combobox(
            self,
            width=10,
            textvariable=self.root.setting.c60_selected_profile,
            state="readonly",
            justify=tk.CENTER,
            font=("Arial", 10, "bold"),
        )
        self.profile_dropdown_menu["values"] = self.root.setting.profile_list
        self.profile_dropdown_menu.state(["readonly"])
        self.profile_dropdown_menu.bind("<<ComboboxSelected>>", self.__on_profile_dropdown_change)
        self.profile_dropdown_menu.pack(side=tk.RIGHT)
        attach_schema_tooltip(self.profile_dropdown_menu, AppSettings, "c60_selected_profile")

        shortcut_cb = ttk.Checkbutton(
            self,
            text="Shortcut Mode",
            variable=self.root.setting.p60_shortcut_mode,
            command=self.root.update_shortcut_mode,
        )
        shortcut_cb.pack(anchor=tk.W, side=tk.LEFT)
        attach_schema_tooltip(shortcut_cb, VisualizationSettings, "p60_shortcut_mode")

        dark_cb = ttk.Checkbutton(
            self, text="Dark Mode", variable=self.root.setting.p60_dark_mode, command=self.toggle_dark_mode
        )
        dark_cb.pack(anchor=tk.W, side=tk.LEFT)
        attach_schema_tooltip(dark_cb, VisualizationSettings, "p60_dark_mode")

        ttk.Label(self, textvariable=self.root.setting.perception_status_text, width=30).pack(
            side=tk.LEFT, padx=(25, 5), pady=5
        )

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
        self.help_text.config(state=tk.DISABLED)

    def __on_profile_dropdown_change(self, event):
        log.info("Selected profile: %s", event.widget.get())
        self.root.load_settings()
        self.root.reload_stack(reload_code=False)

    def toggle_dark_mode(self):
        self.root.set_dark_mode_themed() if self.root.setting.p60_dark_mode.get() else self.root.set_light_mode()

    def save_settings(self):
        profile = self.root.setting.c60_selected_profile.get()
        binder = TkSettingsBinder()
        save_setting(self.root.setting, profile=profile, binder=binder)
        save_setting(PerceptionSettings, profile=profile, binder=binder)
        AppSettings.c62_community_plugins = PluginPaths.normalize_map(AppSettings.c62_community_plugins)
        save_setting(ExecutionSettings, profile=profile, binder=binder)

    def open_settings_window(self):
        if hasattr(self, "setting_view") and hasattr(self.setting_view, "window") and self.setting_view.window.winfo_exists():
            self.root.load_settings(only_stack=True)
            self.setting_view.show()
            log.info("Showing existing settings window")
        else:
            self.root.load_settings(only_stack=True)
            self.setting_view = SettingWindow(self.root)
            log.info("Creating new settings window")

    def update_setting_window(self):
        if hasattr(self, "setting_view") and hasattr(self.setting_view, "window") and self.setting_view.window.winfo_exists():
            self.setting_view.update_core_widgets()
            self.setting_view.update_plugins_widgets()
            self.setting_view.update_community_plugin_list()
            log.info("Updated existing settings window")

    def open_plugins_window(self):
        CommunityPluginsApp.open(parent=self.root)
