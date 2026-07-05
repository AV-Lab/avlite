"""App-layer settings: plugin loading, profiles, and bootstrap (c60_apps)."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from avlite.c60_apps.c64_settings_schema import SettingsSchema

_DEFAULT_BUILTIN_PLUGINS = [
    "p60_headless_mode",
    "p60_setting_cli",
    "p60_visualizer_tk",
]


class AppSettingsSchema(SettingsSchema):
    filepath: ClassVar[str] = "configs/c69_apps.yaml"

    c62_load_plugins: bool = Field(
        default=True, description="Load built-in and community plugins on startup."
    )
    c62_default_plugins: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_BUILTIN_PLUGINS),
        description="Built-in plugin packages to load on startup.",
    )
    c62_community_plugins: dict[str, str] = Field(
        default_factory=dict,
        description="Community plugin name to install directory map.",
    )
    c60_selected_profile: str = Field(default="default", description="Active settings profile name.")


AppSettings = AppSettingsSchema()
