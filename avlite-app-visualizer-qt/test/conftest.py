"""Ensure the community Qt plugin stays imported under ``avlite.plugins.*``."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

from _plugin_util import ensure_qt_plugin_loaded  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _bootstrap_qt_plugin():
    ensure_qt_plugin_loaded()


def pytest_runtest_setup(item):
    import sys

    if "avlite.plugins.avlite_app_visualizer_qt" not in sys.modules:
        ensure_qt_plugin_loaded()
