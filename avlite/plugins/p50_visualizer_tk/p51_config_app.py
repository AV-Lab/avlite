"""Standalone ``avlite config`` settings GUI."""

from __future__ import annotations

import logging
import tkinter as tk

from avlite.c50_apps.c51_app_strategy import AppStrategy
from avlite.c50_apps.c59_settings import AppSettings
from avlite.plugins.p50_visualizer_tk.settings import AppSettingsUI, VisualizationSettings, sync_stack_settings_to_ui
from avlite.c50_apps.c52_factory import load_stack_settings
from avlite.c50_apps.c53_plugins import reload_lib
from avlite.c50_apps.c58_paths import ConfigPaths
from avlite.c50_apps.c55_setting_utils import list_profiles, load_setting

log = logging.getLogger(__name__)


class ConfigAppHost(tk.Tk):
    """Minimal host for the standalone ``avlite config`` settings GUI."""

    def __init__(self) -> None:
        setup_dpi()
        super().__init__()
        apply_ttk_theme(self, dark=True)
        self.title("AVLite Config")
        _s = get_dpi_scale(self)
        self.geometry(f"{scaled(900, _s)}x{scaled(700, _s)}")

        self.app = AppSettingsUI()
        self.setting = VisualizationSettings()
        self.setting.profile_list = list_profiles(AppSettings)
        startup = ConfigPaths.startup_profile()
        if startup and startup in self.setting.profile_list:
            self.app.c50_selected_profile.set(startup)

        self.validate_cmd = (self.register(self._validate_float_input), "%P")

    def _validate_float_input(self, user_input: str) -> bool:
        if user_input in ("", "-"):
            return True
        try:
            float(user_input)
            return True
        except ValueError:
            return False

    def load_configs(self, only_stack: bool = False, profile: str | None = None) -> None:
        if profile:
            self.app.c50_selected_profile.set(profile)
        else:
            profile = self.app.c50_selected_profile.get()
        binder = TkSettingsBinder()
        load_setting(AppSettings, profile=profile)
        self.app.sync_from_singleton()
        if not only_stack:
            load_setting(self.setting, profile=profile, binder=binder)
        load_stack_settings(profile=profile)
        sync_stack_settings_to_ui(self.setting)
        ConfigPaths.set_startup_profile(profile)
        log.info("Loaded settings from profile: %s", profile)

    def on_stack_settings_changed(self) -> None:
        pass

    def reload_stack(self, reload_code: bool = True) -> None:
        if reload_code:
            reload_lib(exclude_settings=True, reload_plugins=self.app.c50_load_plugins.get())
        load_stack_settings(profile=self.app.c50_selected_profile.get())
        sync_stack_settings_to_ui(self.setting)
        self.on_stack_settings_changed()


class ConfigApp(AppStrategy):
    """``avlite config`` — standalone settings GUI (no visualizer panels)."""

    cli_name = "config"
    help = "Open the settings GUI"

    def run(self, args, unknown):
        host = ConfigAppHost()
        host.withdraw()
        host.load_configs()
        view = SettingWindow(host, show_visualization_settings=False)
        view.window.protocol("WM_DELETE_WINDOW", host.destroy)
        host.mainloop()
