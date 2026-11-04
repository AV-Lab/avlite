"""AVLite Qt5 visualizer community plugin (``avlite qt``).

Layout matches sibling plugins such as ``avlite-bridge-carla`` so this directory
can be published as its own GitHub repository under the AV-Lab org without
restructuring. A thin built-in shim under ``avlite/plugins/p60_visualizer_qt``
loads this package at CLI bootstrap so ``avlite qt`` is registered.
"""

# AVLite loads this file as ``avlite.plugins.avlite_app_visualizer_qt``.
if __package__:
    from .app_strategy import VisualizationQtApp
    from .settings import PluginSettings, PluginSettingsSchema

    __all__ = [
        "VisualizationQtApp",
        "PluginSettings",
        "PluginSettingsSchema",
    ]
else:  # pragma: no cover — pytest collecting the dashed checkout path
    __all__ = []
