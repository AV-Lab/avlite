"""Each registered plot has its own visibility flag, and sash math does not collapse the next pane."""

from __future__ import annotations

import tkinter as tk

from tkinter import ttk

from avlite.c60_apps.c61_app_strategy import PlotStrategy
from avlite.plugins.p60_visualizer_tk.p61_visualizer_app import VisualizerApp
from avlite.plugins.p60_visualizer_tk.p64_setting_views import SettingShortcutView
from avlite.plugins.p60_visualizer_tk.p65_ui_lib import attach_panel_fold
from avlite.plugins.p60_visualizer_tk.settings import VisualizationSettings


def test_update_views_restores_a_matching_sash_set_and_splits_otherwise():
    root = tk.Tk()
    root.withdraw()
    added = []
    try:
        app = VisualizerApp.__new__(VisualizerApp)
        app.plot_pane = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        app.plot_sash_token = 0
        app.setting = VisualizationSettings()
        app.after_idle = root.after_idle
        app.after = lambda _delay_ms, callback: None
        app.log_view = type("Log", (), {"update_log_view_height": lambda self: None})()
        app.update_shortcut_mode = lambda: None
        app.plots = {}
        root.setting = app.setting

        def weights():
            return [int(app.plot_pane.pane(path, "weight")) for path in app.plot_pane.panes()]

        def add(class_name):
            cls = type(class_name, (PlotStrategy,), {"label": class_name, "plot": lambda self: None})
            added.append(class_name)
            app.plots[class_name] = cls(root)

        add("PaneA")
        app.setting.p67_plot_sashes = {"PaneA": 1.0}
        app.update_views()
        assert weights() == [1]

        add("PaneB")
        app.plots["PaneA"].enabled.set(False)
        app.setting.p67_plot_sashes = {"PaneA": 0.25, "PaneB": 0.75}
        app.update_views()
        assert weights() == [1]

        app.plots["PaneA"].enabled.set(True)
        app.update_views()
        assert weights() == [250, 750]

        app.setting.p67_plot_sashes = {"PaneA": 0.0, "PaneB": 1.0}
        app.update_views()
        assert weights() == [1, 1]

        add("PaneC")
        app.plots["PaneA"].enabled.set(False)
        app.setting.p67_plot_sashes = {"PaneA": 0.3, "PaneB": 0.3, "PaneC": 0.4}
        app.update_views()
        assert weights() == [1, 1]
        for _ in range(6):
            root.update()
    finally:
        for name in added:
            PlotStrategy.registry.pop(name, None)
        root.destroy()


def test_each_plot_seeds_and_toggles_only_its_own_flag():
    root = tk.Tk()
    root.withdraw()
    try:
        setting = VisualizationSettings()
        root.setting = setting
        setting.p67_show_local_global_view.set(False)
        setting.p67_show_local_frenet_view.set(True)

        class SeededPlot(PlotStrategy):
            label = "Seeded"
            enabled_field = "p67_show_local_global_view"

            def plot(self) -> None:
                return None

        class FreshPlot(PlotStrategy):
            label = "Fresh"

            def plot(self) -> None:
                return None

        seeded = SeededPlot(root)
        fresh = FreshPlot(root)
        assert seeded.enabled.get() is False
        assert setting.p67_plot_visible["SeededPlot"] is False
        assert fresh.enabled.get() is True
        assert setting.p67_plot_visible["FreshPlot"] is True

        seeded.enabled.set(True)
        assert setting.p67_plot_visible["SeededPlot"] is True
        assert setting.p67_show_local_frenet_view.get() is True
        assert fresh.enabled.get() is True

        setting.p67_plot_visible["SeededPlot"] = False
        again = setting.ensure_plot_enabled("SeededPlot", "p67_show_local_global_view")
        assert again is seeded.enabled
        assert again.get() is False
        assert setting.p67_plot_visible["FreshPlot"] is True
    finally:
        PlotStrategy.registry.pop("SeededPlot", None)
        PlotStrategy.registry.pop("FreshPlot", None)
        root.destroy()


def test_plot_checks_show_on_an_open_settings_panel():
    root = tk.Tk()
    root.withdraw()
    try:
        settings = ttk.LabelFrame(root, text="Settings")
        settings.grid(row=1, column=0, sticky="ew")
        plot_checks = ttk.Frame(settings)
        plot_checks.pack(side=tk.TOP, fill=tk.X)
        body = ttk.Label(settings, text="Dark")
        body.pack(side=tk.LEFT)
        folded = tk.BooleanVar(master=root, value=False)
        attach_panel_fold(settings, folded)

        class VisiblePlot(PlotStrategy):
            label = "Visible"
            enabled_field = "p67_show_local_global_view"

            def plot(self) -> None:
                return None

        root.setting = VisualizationSettings()
        root.plots = {"VisiblePlot": VisiblePlot(root)}
        root.update_views = lambda: None
        root.update_ui = lambda: None
        host = type("Host", (), {"root": root, "plot_checks": plot_checks})()
        SettingShortcutView.refresh_plot_checks(host)
        root.update_idletasks()

        checks = [
            child for child in plot_checks.winfo_children() if child.winfo_class() == "TCheckbutton"
        ]
        assert len(checks) == 1
        assert checks[0].cget("text") == "Visible"
        assert plot_checks.master is settings
        assert plot_checks.winfo_manager() == "pack"
        assert checks[0].winfo_ismapped() or not root.winfo_viewable()
        assert checks[0].winfo_manager() == "pack"
    finally:
        PlotStrategy.registry.pop("VisiblePlot", None)
        root.destroy()
