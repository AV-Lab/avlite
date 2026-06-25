from avlite.c60_common.c68_settings_schema import SettingsSchema


class PluginSettingsSchema(SettingsSchema):
    pass


class PluginSettings:
    schema = PluginSettingsSchema
    exclude = ["exclude", "filepath", "schema"]
    filepath: str = "configs/plugin_carla.yaml"
