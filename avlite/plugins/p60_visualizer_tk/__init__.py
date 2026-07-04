"""Default AVLite visualizer GUI plugin (visualizer, setting, plugins apps)."""

from avlite.plugins.p60_visualizer_tk.p62_setting_app import SettingApp, SettingAppHost
from avlite.plugins.p60_visualizer_tk.p63_plugins_app import CommunityPluginsApp, PluginsApp
from avlite.plugins.p60_visualizer_tk.p61_visualizer_app import VisualizerApp, VisualizationApp

__all__ = [
    "VisualizerApp",
    "VisualizationApp",
    "SettingAppHost",
    "SettingApp",
    "CommunityPluginsApp",
    "PluginsApp",
]
