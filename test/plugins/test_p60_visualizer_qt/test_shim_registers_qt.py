"""Built-in shim registers ``avlite qt`` even when collected from core tests."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

from avlite.c60_apps.c61_app_strategy import AppStrategy, bootstrap_apps


def test_p60_visualizer_qt_shim_registers_cli():
    bootstrap_apps()
    assert "qt" in AppStrategy.registry
