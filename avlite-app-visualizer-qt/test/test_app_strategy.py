"""AppStrategy registration and CLI wiring for the Qt visualizer."""

from __future__ import annotations

import os

import pytest

from avlite.c60_apps.c61_app_strategy import AppStrategy, bootstrap_apps


@pytest.fixture(autouse=True)
def _offscreen():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"


def test_bootstrap_registers_qt_cli():
    bootstrap_apps()
    assert "qt" in AppStrategy.registry
    cls = AppStrategy.registry["qt"]
    assert cls.cli_name == "qt"
    assert "Qt" in (cls.help or "") or "qt" in (cls.help or "").lower()


def test_configure_parser_adds_offscreen_flag():
    bootstrap_apps()
    import argparse

    parser = argparse.ArgumentParser()
    AppStrategy.registry["qt"]().configure_parser(parser)
    args = parser.parse_args(["--offscreen", "--profile", "default"])
    assert args.offscreen is True
    assert args.profile == "default"
