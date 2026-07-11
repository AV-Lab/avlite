"""Standalone ``avlite setting`` settings GUI."""

from __future__ import annotations

import logging
import tkinter as tk

from avlite.c60_apps.c61_app_strategy import AppStrategy
from avlite.c60_apps.c69_settings import AppSettings
from avlite.c60_apps.c62_factory import load_stack_settings
from avlite.c60_apps.c63_plugins import reload_lib
from avlite.c60_apps.c68_paths import ConfigPaths
from avlite.c60_apps.c65_setting_utils import list_profiles, load_setting
from avlite.plugins.p60_visualizer_tk.p64_setting_views import SettingWindow
from avlite.plugins.p60_visualizer_tk.p65_ui_lib import (
    DpiScale,
    TkSettingsBinder,
    apply_ttk_theme,
)
from avlite.plugins.p60_visualizer_tk.settings import VisualizationSettings, sync_stack_settings_to_ui

log = logging.getLogger(__name__)


class SettingAppHost(tk.Tk):
    """Minimal host for the standalone ``avlite setting`` settings GUI."""

    hosting_plugin_name = "p60_visualizer_tk"

    def __init__(self) -> None:
        DpiScale.setup()
        super().__init__()
        apply_ttk_theme(self, dark=True)
        self.title("AVLite Settings")
        _s = DpiScale.for_widget(self)
        self.geometry(f"{DpiScale.scaled(900, _s)}x{DpiScale.scaled(700, _s)}")

        self.setting = VisualizationSettings()
        self.setting.profile_list = list_profiles(AppSettings)
        startup = ConfigPaths.startup_profile()
        if startup and startup in self.setting.profile_list:
            self.setting.c60_selected_profile.set(startup)

        self.validate_cmd = (self.register(self._validate_float_input), "%P")

    def _validate_float_input(self, user_input: str) -> bool:
        if user_input in ("", "-"):
            return True
        try:
            float(user_input)
            return True
        except ValueError:
            return False

    def load_settings(self, only_stack: bool = False, profile: str | None = None) -> None:
        if profile:
            self.setting.c60_selected_profile.set(profile)
        else:
            profile = self.setting.c60_selected_profile.get()
        binder = TkSettingsBinder()
        load_setting(AppSettings, profile=profile)
        self.setting.sync_app_from_singleton()
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
            reload_lib(exclude_settings=True, reload_plugins=self.setting.c62_load_plugins.get())
        load_stack_settings(profile=self.setting.c60_selected_profile.get())
        sync_stack_settings_to_ui(self.setting)
        self.on_stack_settings_changed()


class SettingApp(AppStrategy):
    """``avlite setting`` — standalone settings GUI (no visualizer panels)."""

    cli_name = "setting"
    help = "Open the settings GUI"

    def run(self, args, unknown):
        host = SettingAppHost()
        host.withdraw()
        host.load_settings()
        view = SettingWindow(host, show_visualization_settings=False)
        host.mainloop()
