"""Qt binding helpers.

**Why PyQt5 (not PySide2):** AVLite targets Python ≥3.10 and this environment runs
3.12. PySide2's last public wheels stop at 5.13.2 and do not cover 3.12. PyQt5
5.15.x publishes ``cp38-abi3`` manylinux wheels that import cleanly on 3.12, and
pyqtgraph's Qt autodetection prefers PyQt5 when both are absent. LGPL PySide6
would work but is Qt6 — this plugin intentionally stays on the Qt5 stack
requested for AVLite.
"""

from __future__ import annotations

import os


def configure_qt_env(*, offscreen: bool | None = None) -> None:
    """Set Qt platform / pyqtgraph library before creating QApplication."""
    if offscreen or os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        # Mesa/offscreen + pyqtgraph OpenGL frequently SIGSEGVs in CI VMs.
        os.environ.setdefault("AVLITE_QT_NO_OPENGL", "1")
    # Force pyqtgraph onto PyQt5 before it probes bindings.
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")


def import_qt():
    """Import PyQt5 widgets/core/gui. Raises ImportError with install hint."""
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyQt5 is required for the AVLite Qt visualizer.\n"
            "  pip install -r avlite-app-visualizer-qt/requirements.txt"
        ) from exc
    return QtCore, QtGui, QtWidgets


def import_pyqtgraph(*, use_opengl: bool | None = None):
    """Import and configure pyqtgraph for high-FPS PlotWidget updates."""
    configure_qt_env()
    import pyqtgraph as pg

    if use_opengl is None:
        use_opengl = os.environ.get("AVLITE_QT_NO_OPENGL") != "1"
    pg.setConfigOptions(
        antialias=True,
        background="#1e1e1e",
        foreground="#dddddd",
        useOpenGL=bool(use_opengl),
        enableExperimental=True,
    )
    return pg
