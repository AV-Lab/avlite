from avlite.c60_common.c68_settings_schema import SettingsSchema


class ExtensionSettingsSchema(SettingsSchema):
    pass


class ExtensionSettings:
    schema = ExtensionSettingsSchema
    exclude = ["exclude", "filepath", "schema"]
    filepath: str = "configs/ext_gazebo_worldbridge.yaml"
