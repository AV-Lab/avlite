"""Log panel that drains the root logging handlers into a QPlainTextEdit."""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

from .qt_binding import import_qt

if TYPE_CHECKING:
    from .main_window import VisualizerQtWindow

QtCore, QtGui, QtWidgets = import_qt()


class _QtLogHandler(logging.Handler):
    def __init__(self, sink: "LogPanel"):
        super().__init__()
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.sink.enqueue(self.format(record), record.name)
        except Exception:
            self.handleError(record)


class LogPanel(QtWidgets.QGroupBox):
    def __init__(self, root: "VisualizerQtWindow"):
        super().__init__("Log")
        self.root = root
        self._queue: deque[tuple[str, str]] = deque(maxlen=root.setting.p68_max_log_lines)
        layout = QtWidgets.QVBoxLayout(self)
        filters = QtWidgets.QHBoxLayout()
        self._filters = {
            "core": QtWidgets.QCheckBox("Core"),
            "perceive": QtWidgets.QCheckBox("Perception"),
            "plan": QtWidgets.QCheckBox("Planning"),
            "control": QtWidgets.QCheckBox("Control"),
            "execute": QtWidgets.QCheckBox("Execution"),
            "plugins": QtWidgets.QCheckBox("Plugins"),
        }
        mapping = {
            "core": root.setting.p68_show_core_logs,
            "perceive": root.setting.p68_show_perceive_logs,
            "plan": root.setting.p68_show_plan_logs,
            "control": root.setting.p68_show_control_logs,
            "execute": root.setting.p68_show_execute_logs,
            "plugins": root.setting.p68_show_plugins_logs,
        }
        for key, box in self._filters.items():
            box.setChecked(bool(mapping[key].get()))
            box.stateChanged.connect(lambda state, k=key, v=mapping[key]: v.set(bool(state)))
            filters.addWidget(box)
        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.clicked.connect(self.clear)
        filters.addWidget(clear_btn)
        filters.addStretch(1)
        layout.addLayout(filters)

        self.view = QtWidgets.QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(root.setting.p68_max_log_lines)
        self.view.setFont(QtGui.QFont("monospace", 10))
        layout.addWidget(self.view)

        self._handler = _QtLogHandler(self)
        self._handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", "%H:%M:%S"))
        logging.getLogger().addHandler(self._handler)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._flush)
        self._timer.start(int(root.setting.p68_log_pull_time))

    def enqueue(self, text: str, logger_name: str) -> None:
        self._queue.append((text, logger_name))

    def _allowed(self, name: str) -> bool:
        s = self.root.setting
        if s.p68_disable_log.get():
            return False
        low = name.lower()
        if "plugin" in low:
            return bool(s.p68_show_plugins_logs.get())
        if "c10" in low or "perception" in low:
            return bool(s.p68_show_perceive_logs.get())
        if "c20" in low or "plan" in low:
            return bool(s.p68_show_plan_logs.get())
        if "c30" in low or "control" in low:
            return bool(s.p68_show_control_logs.get())
        if "c40" in low or "exec" in low:
            return bool(s.p68_show_execute_logs.get())
        return bool(s.p68_show_core_logs.get())

    def _flush(self) -> None:
        if not self._queue:
            return
        lines = []
        while self._queue:
            text, name = self._queue.popleft()
            if self._allowed(name):
                lines.append(text)
        if lines:
            self.view.appendPlainText("\n".join(lines))

    def clear(self) -> None:
        self._queue.clear()
        self.view.clear()

    def shutdown(self) -> None:
        self._timer.stop()
        logging.getLogger().removeHandler(self._handler)
