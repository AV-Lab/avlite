"""Each registered plot has its own visibility flag, and sash math does not collapse the next pane."""

from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

from types import SimpleNamespace

from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c60_apps.c61_app_strategy import PlotStrategy
from avlite.plugins.p60_visualizer_tk.p61_visualizer_app import VisualizerApp
from avlite.plugins.p60_visualizer_tk.p66_plot_views import ControlPlot
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


def test_control_plot_records_acceleration_and_starts_on():
    root = tk.Tk()
    root.withdraw()
    try:
        root.setting = VisualizationSettings()
        view = ControlPlot(root)
        assert view.enabled.get()
        ego = EgoState()
        ego.velocity = 10.0
        ego.theta = 0.0
        root.exec = SimpleNamespace(ego_state=ego, elapsed_sim_time=0.0)
        root.overrideredirect(True)
        root.geometry("700x700+0+0")
        view.pack(fill=tk.BOTH, expand=True)
        root.deiconify()
        root.update()
        view.plot()
        ego.velocity = 12.0
        ego.theta = 0.1
        root.exec.elapsed_sim_time = 1.0
        view.plot()
        lateral, longitudinal = view.trail[-1]
        assert abs(longitudinal - 2.0) < 1e-9
        assert abs(lateral - 1.2) < 1e-9
        assert view.gg.itemcget("along", "text") == f"{longitudinal / 9.80665:+.1f}"
        assert view.gg.itemcget("alat", "text") == f"{lateral / 9.80665:+.1f}"
    finally:
        root.destroy()


def test_control_plot_draws_labeled_rings():
    root = tk.Tk()
    root.withdraw()
    try:
        root.setting = VisualizationSettings()
        view = ControlPlot(root)
        ego = EgoState()
        root.exec = SimpleNamespace(ego_state=ego, elapsed_sim_time=0.0)
        root.overrideredirect(True)
        root.geometry("700x700+0+0")
        view.pack(fill=tk.BOTH, expand=True)
        root.deiconify()
        root.update()
        view.plot()
        assert len(view.gg.find_withtag("ring")) == 3
        labels = [view.gg.itemcget(item, "text") for item in view.gg.find_withtag("label")]
        assert labels.count("0.0") == 1
        assert "1.0" in labels
        assert "2.0" in labels
        assert "3.0" in labels
        assert "long" not in labels
        assert "lat" not in labels
        assert "g" not in labels
        assert view.gg.itemcget("along", "text") == "0.0"
        assert view.gg.itemcget("alat", "text") == "0.0"
        plot_w, plot_h = view.winfo_width(), view.winfo_height()
        assert plot_h > 48
        widgets = (view.speed_gauge, view.steer_wheel, view.gg)
        ordered = sorted(widgets, key=lambda widget: widget.winfo_y())
        assert ordered == list(widgets)
        for above, below in zip(ordered, ordered[1:]):
            assert below.winfo_y() == above.winfo_y() + above.winfo_height()
        boxes = [
            (
                widget.winfo_x(),
                widget.winfo_y(),
                widget.winfo_x() + widget.winfo_width(),
                widget.winfo_y() + widget.winfo_height(),
            )
            for widget in widgets
        ]
        slack_w = plot_w - (max(box[2] for box in boxes) - min(box[0] for box in boxes))
        slack_h = plot_h - (max(box[3] for box in boxes) - min(box[1] for box in boxes))
        assert min(slack_w, slack_h) <= 24
        gg_mid = view.gg.winfo_x() + view.gg.winfo_width() / 2
        steer_mid = view.steer_wheel.winfo_x() + view.steer_wheel.winfo_width() / 2
        assert abs(gg_mid - steer_mid) <= 2
        scale = {}
        zero = None
        zero_item = None
        scale_item = None
        for item in view.gg.find_withtag("label"):
            text = view.gg.itemcget(item, "text")
            x0, y0, x1, y1 = view.gg.bbox(item)
            assert x0 >= 0 and y0 >= 0
            assert x1 <= view.gg.winfo_width() and y1 <= view.gg.winfo_height()
            if text == "0.0":
                zero = (x0, y0, x1, y1)
                zero_item = item
            elif text in ("1.0", "2.0", "3.0"):
                scale[text] = (x0, y0, x1, y1)
                scale_item = item
        assert zero is not None and set(scale) == {"1.0", "2.0", "3.0"}
        assert view.gg.itemcget(zero_item, "fill").lower() in {"#888", "#888888"}
        origin_font = tkfont.Font(font=view.gg.itemcget(zero_item, "font"))
        assert abs(origin_font.cget("size")) == 8
        used = tkfont.Font(font=view.gg.itemcget(scale_item, "font"))
        assert abs(used.cget("size")) == 8
        ring_boxes = [view.gg.bbox(item) for item in view.gg.find_withtag("ring")]
        ring_boxes.sort(key=lambda box: box[2] - box[0])
        assert abs((ring_boxes[-1][2] - ring_boxes[-1][0]) - view.dial) <= 12
        ring_mid = view.gg.winfo_x() + (ring_boxes[-1][0] + ring_boxes[-1][2]) / 2
        assert abs(ring_mid - steer_mid) <= 2
        centers = []
        for text, ring in zip(("1.0", "2.0", "3.0"), ring_boxes):
            x0, y0, x1, y1 = scale[text]
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            rx, ry = (ring[0] + ring[2]) / 2, (ring[1] + ring[3]) / 2
            rr = (ring[2] - ring[0]) / 2
            assert mx < rx and my < ry
            assert abs(((mx - rx) ** 2 + (my - ry) ** 2) ** 0.5 - rr) <= 3
            centers.append((mx, my))
        (ax, ay), (bx, by), (cx_, cy_) = centers
        cross = (bx - ax) * (cy_ - ay) - (by - ay) * (cx_ - ax)
        span = ((cx_ - ax) ** 2 + (cy_ - ay) ** 2) ** 0.5
        assert abs(cross) / span <= 2
        assert abs((ax - cx_) - (ay - cy_)) <= 3
        assert cx_ < bx < ax and cy_ < by < ay
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = scale[("1.0", "2.0", "3.0")[i]], scale[("1.0", "2.0", "3.0")[j]]
                assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
        for tag in ("along", "alat"):
            x0, y0, x1, y1 = view.gg.bbox(tag)
            assert x0 >= 0 and y0 >= 0
            assert x1 <= view.gg.winfo_width() and y1 <= view.gg.winfo_height()
            assert x1 <= zero[0] or x0 >= zero[2] or y1 <= zero[1] or y0 >= zero[3]
        assert view.gg.bbox("along")[3] <= ring_boxes[-1][1] + 2
        assert 6 <= view.gg.bbox("alat")[0] - ring_boxes[-1][2] <= 14
        for box in scale.values():
            x0, y0, x1, y1 = box
            assert x1 <= zero[0] or x0 >= zero[2] or y1 <= zero[1] or y0 >= zero[3]
        rings = view.gg.find_withtag("ring")
        speed, steer = view.speed_gauge, view.steer_wheel
        view.plot()
        assert view.gg.find_withtag("ring") == rings
        assert view.speed_gauge is speed
        assert view.steer_wheel is steer
        root.geometry("220x220+0+0")
        root.update()
        view.plot()
        tight = [item for item in view.gg.find_withtag("label") if view.gg.itemcget(item, "text") == "3.0"]
        assert len(tight) == 1
        for item in (*tight, *view.gg.find_withtag("along"), *view.gg.find_withtag("alat")):
            x0, y0, x1, y1 = view.gg.bbox(item)
            assert x0 >= 0 and y0 >= 0
            assert x1 <= view.gg.winfo_width() and y1 <= view.gg.winfo_height()
    finally:
        root.destroy()
