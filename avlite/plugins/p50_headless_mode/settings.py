from pydantic import Field

from avlite.c60_common.c68_settings_schema import SettingsSchema


class PluginSettingsSchema(SettingsSchema):
    log_buffer_size: int = Field(default=500, description="Max log lines in headless dashboard buffer.")
    dashboard_refresh_hz: float = Field(default=10.0, description="Terminal dashboard refresh rate (Hz).")
    stats_panel_height: int = Field(default=18, description="Rows reserved for stats panel in dashboard.")


class PluginSettings:
    schema = PluginSettingsSchema
    exclude = ["exclude", "filepath", "schema"]
    filepath: str = "configs/plugin_headless_mode.yaml"

    log_buffer_size: int = 500
    dashboard_refresh_hz: float = 10.0
    stats_panel_height: int = 18
