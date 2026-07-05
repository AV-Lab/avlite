from pydantic import Field

from avlite.c60_apps.c64_settings_schema import SettingsSchema


class PluginSettingsSchema(SettingsSchema):
    log_buffer_size: int = Field(default=500, description="Max log lines in headless dashboard buffer.")
    dashboard_refresh_hz: float = Field(default=10.0, description="Terminal dashboard refresh rate (Hz).")
    stats_panel_height: int = Field(default=18, description="Rows reserved for stats panel in dashboard.")


# Settings singleton; filepath is assigned by the plugin loader from the directory name.
PluginSettings = PluginSettingsSchema()
