"""pyqtgraph plot widget unit tests (offscreen)."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

from avlite.c60_apps.c61_app_strategy import bootstrap_apps


@pytest.fixture(scope="module")
def qapp():
    bootstrap_apps()
    from avlite.plugins.avlite_app_visualizer_qt.qt_binding import configure_qt_env, import_qt

    configure_qt_env(offscreen=True)
    _QtCore, _QtGui, QtWidgets = import_qt()
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_closed_poly_helper():
    from avlite.plugins.avlite_app_visualizer_qt.plot_widgets import _closed_poly

    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    xs, ys = _closed_poly(corners)
    assert len(xs) == 5
    assert xs[0] == xs[-1]
    assert ys[0] == ys[-1]


def test_global_plot_redraw_without_exec(qapp):
    from avlite.plugins.avlite_app_visualizer_qt.plot_widgets import GlobalPlotWidget
    from avlite.plugins.avlite_app_visualizer_qt.setting_model import VisualizationSettings

    class _Root:
        setting = VisualizationSettings()

        def __init__(self):
            self.exec = None

        def teleport_ego(self, *a, **k):
            pass

        def update_ui(self):
            pass

        def replan_global(self):
            pass

    root = _Root()
    w = GlobalPlotWidget(root)
    assert w.redraw() == 0.0
    w.reset()
    w.update_theme("#000", "#fff")
    w.close()
