"""App-layer settings: plugin loading, profiles, and bootstrap (c50_apps)."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from avlite.c50_apps.c54_settings_schema import SettingsSchema

_DEFAULT_BUILTIN_PLUGINS = [
    "p50_headless_mode",
    "p50_config_cli",
    "p50_visualizer_tk",
]


class AppSettingsSchema(SettingsSchema):
    filepath: ClassVar[str] = "configs/c59_apps.yaml"

    c52_load_plugins: bool = Field(
        default=True, description="Load built-in and community plugins on startup."
    )
    c52_default_plugins: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_BUILTIN_PLUGINS),
        description="Built-in plugin packages to load on startup.",
    )
    c52_community_plugins: dict[str, str] = Field(
        default_factory=dict,
        description="Community plugin name to install directory map.",
    )
    c50_selected_profile: str = Field(default="default", description="Active settings profile name.")


AppSettings = AppSettingsSchema()
