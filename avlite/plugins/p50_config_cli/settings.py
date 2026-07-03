from avlite.c50_apps.c54_settings_schema import SettingsSchema


class PluginSettingsSchema(SettingsSchema):
    """No tunable settings; plugin exists to register the config-cli app."""


PluginSettings = PluginSettingsSchema()
