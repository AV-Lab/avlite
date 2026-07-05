from avlite.c60_apps.c64_settings_schema import SettingsSchema


class PluginSettingsSchema(SettingsSchema):
    """No tunable settings; plugin exists to register the setting-cli app."""


PluginSettings = PluginSettingsSchema()
