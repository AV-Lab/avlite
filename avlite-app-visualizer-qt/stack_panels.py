"""Perception / Planning / Control / Execution panels for the Qt visualizer."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from avlite.c10_perception.c12_perception_strategy import PerceptionPipeline
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningPipeline
from avlite.c30_control.c31_control_model import ControlCommand
from avlite.c40_execution.c43_task_strategy import TaskStrategy

from .qt_binding import import_qt
from .setting_model import Var

if TYPE_CHECKING:
    from .main_window import VisualizerQtWindow

QtCore, QtGui, QtWidgets = import_qt()
log = logging.getLogger(__name__)


def _combo(parent, var: Var, choices: list[str], on_change: Callable[[], None] | None = None) -> QtWidgets.QComboBox:
    box = QtWidgets.QComboBox(parent)
    box.addItems([str(c) for c in choices])
    current = str(var.get() or "")
    idx = box.findText(current)
    box.setCurrentIndex(idx if idx >= 0 else 0)

    def _sync() -> None:
        text = box.currentText()
        if text != str(var.get() or ""):
            var.set(text)
            if on_change:
                on_change()

    box.currentTextChanged.connect(lambda _t: _sync())
    var.bind(lambda: _set_combo(box, var))
    return box


def _set_combo(box: QtWidgets.QComboBox, var: Var) -> None:
    text = str(var.get() or "")
    idx = box.findText(text)
    if idx >= 0 and box.currentIndex() != idx:
        box.blockSignals(True)
        box.setCurrentIndex(idx)
        box.blockSignals(False)


def _check(parent, text: str, var: Var, on_change: Callable[[], None] | None = None) -> QtWidgets.QCheckBox:
    box = QtWidgets.QCheckBox(text, parent)
    box.setChecked(bool(var.get()))

    def _sync(state: int) -> None:
        var.set(bool(state))
        if on_change:
            on_change()

    box.stateChanged.connect(_sync)
    var.bind(lambda: box.setChecked(bool(var.get())) if box.isChecked() != bool(var.get()) else None)
    return box


def _dbl_spin(parent, var: Var, lo=0.001, hi=10.0, step=0.01) -> QtWidgets.QDoubleSpinBox:
    spin = QtWidgets.QDoubleSpinBox(parent)
    spin.setRange(lo, hi)
    spin.setSingleStep(step)
    spin.setDecimals(3)
    spin.setValue(float(var.get()))
    spin.valueChanged.connect(lambda v: var.set(float(v)))
    var.bind(lambda: spin.setValue(float(var.get())) if abs(spin.value() - float(var.get())) > 1e-9 else None)
    return spin


class StackPanels(QtWidgets.QWidget):
    def __init__(self, root: "VisualizerQtWindow"):
        super().__init__()
        self.root = root
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        row = QtWidgets.QHBoxLayout()
        self.perception = PerceptionPanel(root)
        self.planning = PlanningPanel(root)
        self.control = ControlPanel(root)
        row.addWidget(self.perception, 1)
        row.addWidget(self.planning, 1)
        row.addWidget(self.control, 1)
        layout.addLayout(row)

        self.execution = ExecutionPanel(root)
        layout.addWidget(self.execution)

    def refresh(self) -> None:
        self.perception.refresh()
        self.planning.refresh()
        self.control.refresh()
        self.execution.refresh()


class PerceptionPanel(QtWidgets.QGroupBox):
    def __init__(self, root: "VisualizerQtWindow"):
        super().__init__("Perception")
        self.root = root
        grid = QtWidgets.QGridLayout(self)
        choices = root.setting.registry_choices()
        grid.addWidget(QtWidgets.QLabel("Strategy"), 0, 0)
        self.perc = _combo(
            self, root.setting.perception_type, choices["perception"],
            on_change=lambda: root.reload_stack(reload_code=False),
        )
        grid.addWidget(self.perc, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Detect"), 1, 0)
        self.det = _combo(self, root.setting.detection_strategy_type, choices["detection"], root.refresh_pipeline)
        grid.addWidget(self.det, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Track"), 2, 0)
        self.trk = _combo(self, root.setting.tracking_strategy_type, choices["tracking"], root.refresh_pipeline)
        grid.addWidget(self.trk, 2, 1)
        grid.addWidget(QtWidgets.QLabel("Predict"), 3, 0)
        self.prd = _combo(self, root.setting.prediction_strategy_type, choices["prediction"], root.refresh_pipeline)
        grid.addWidget(self.prd, 3, 1)
        grid.addWidget(QtWidgets.QLabel("Map"), 4, 0)
        self.mapping = _combo(
            self, root.setting.mapping_type, choices["mapping"],
            on_change=lambda: root.reload_stack(reload_code=False),
        )
        grid.addWidget(self.mapping, 4, 1)
        grid.addWidget(QtWidgets.QLabel("Loc"), 5, 0)
        self.loc = _combo(
            self, root.setting.localization_type, choices["localization"],
            on_change=lambda: root.reload_stack(reload_code=False),
        )
        grid.addWidget(self.loc, 5, 1)
        self.state_lbl = QtWidgets.QLabel()
        self.state_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        grid.addWidget(self.state_lbl, 6, 0, 1, 2)
        root.setting.vehicle_state.bind(lambda: self.state_lbl.setText(str(root.setting.vehicle_state.get())))
        self._update_pipeline_enabled()

    def _update_pipeline_enabled(self) -> None:
        pipe = self.root.setting.perception_type.get() == PerceptionPipeline.__name__
        for w in (self.det, self.trk, self.prd):
            w.setEnabled(pipe)

    def refresh(self) -> None:
        choices = self.root.setting.registry_choices()
        for box, key in (
            (self.perc, "perception"), (self.det, "detection"), (self.trk, "tracking"),
            (self.prd, "prediction"), (self.mapping, "mapping"), (self.loc, "localization"),
        ):
            cur = box.currentText()
            box.blockSignals(True)
            box.clear()
            box.addItems(choices[key])
            idx = box.findText(cur)
            box.setCurrentIndex(idx if idx >= 0 else 0)
            box.blockSignals(False)
        self._update_pipeline_enabled()


class PlanningPanel(QtWidgets.QGroupBox):
    def __init__(self, root: "VisualizerQtWindow"):
        super().__init__("Planning")
        self.root = root
        grid = QtWidgets.QGridLayout(self)
        choices = root.setting.registry_choices()
        grid.addWidget(QtWidgets.QLabel("Global"), 0, 0)
        self.gplan = _combo(
            self, root.setting.global_planner_type, choices["global_planner"],
            on_change=lambda: root.reload_stack(reload_code=False),
        )
        grid.addWidget(self.gplan, 0, 1)
        btn_greplan = QtWidgets.QPushButton("Global Replan")
        btn_greplan.clicked.connect(root.replan_global)
        grid.addWidget(btn_greplan, 0, 2)
        grid.addWidget(QtWidgets.QLabel("Local"), 1, 0)
        self.lplan = _combo(
            self, root.setting.local_planner_type, choices["local_planner"],
            on_change=lambda: root.reload_stack(reload_code=False),
        )
        grid.addWidget(self.lplan, 1, 1)
        btn_lreplan = QtWidgets.QPushButton("Local Replan")
        btn_lreplan.clicked.connect(self._local_replan)
        grid.addWidget(btn_lreplan, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Behavior"), 2, 0)
        self.beh = _combo(self, root.setting.behavioral_strategy_type, [""] + [], root.refresh_pipeline)
        grid.addWidget(self.beh, 2, 1)
        grid.addWidget(QtWidgets.QLabel("Path"), 3, 0)
        self.path = _combo(self, root.setting.path_strategy_type, [""], root.refresh_pipeline)
        grid.addWidget(self.path, 3, 1)
        grid.addWidget(QtWidgets.QLabel("Speed"), 4, 0)
        self.vel = _combo(self, root.setting.velocity_strategy_type, [""], root.refresh_pipeline)
        grid.addWidget(self.vel, 4, 1)
        self.prob = _check(self, "Probabilistic collision", root.setting.probabilistic_collision_checking)
        grid.addWidget(self.prob, 5, 0, 1, 2)
        self.wp_lbl = QtWidgets.QLabel("WP: 0   Lap: 0")
        grid.addWidget(self.wp_lbl, 6, 0, 1, 3)
        root.setting.current_wp.bind(self._update_wp)
        root.setting.lap.bind(self._update_wp)
        self.refresh()

    def _update_wp(self) -> None:
        self.wp_lbl.setText(f"WP: {self.root.setting.current_wp.get()}   Lap: {self.root.setting.lap.get()}")

    def _local_replan(self) -> None:
        if self.root.exec is None or self.root.exec.local_planner is None:
            return
        self.root.exec.local_planner.replan(
            perception_model=self.root.exec.pm,
            sensors=self.root.exec.world.get_sensor_frame(),
        )
        self.root.update_ui()

    def refresh(self) -> None:
        from avlite.c20_planning.c23_local_planning_strategy import (
            LocalBehavioralPlanningStrategy,
            LocalPathPlanningStrategy,
            LocalVelocityPlanningStrategy,
        )

        choices = self.root.setting.registry_choices()
        for box, items in (
            (self.gplan, choices["global_planner"]),
            (self.lplan, choices["local_planner"]),
            (self.beh, [""] + list(LocalBehavioralPlanningStrategy.registry.keys())),
            (self.path, [""] + list(LocalPathPlanningStrategy.registry.keys())),
            (self.vel, [""] + list(LocalVelocityPlanningStrategy.registry.keys())),
        ):
            cur = box.currentText()
            box.blockSignals(True)
            box.clear()
            box.addItems(items)
            idx = box.findText(cur)
            box.setCurrentIndex(idx if idx >= 0 else 0)
            box.blockSignals(False)
        pipe = self.root.setting.local_planner_type.get() == LocalPlanningPipeline.__name__
        for w in (self.beh, self.path, self.vel):
            w.setEnabled(pipe)


class ControlPanel(QtWidgets.QGroupBox):
    def __init__(self, root: "VisualizerQtWindow"):
        super().__init__("Control")
        self.root = root
        grid = QtWidgets.QGridLayout(self)
        choices = root.setting.registry_choices()
        grid.addWidget(QtWidgets.QLabel("Controller"), 0, 0)
        self.ctrl = _combo(
            self, root.setting.controller_type, choices["controller"],
            on_change=lambda: root.reload_stack(reload_code=False),
        )
        grid.addWidget(self.ctrl, 0, 1, 1, 2)
        self.gauges = {
            "vel_cte": QtWidgets.QLabel("Vel CTE: 0.00"),
            "pos_cte": QtWidgets.QLabel("Pos CTE: 0.00"),
            "accel": QtWidgets.QLabel("Accel: 0.00"),
            "steer": QtWidgets.QLabel("Steer: 0.00"),
        }
        grid.addWidget(self.gauges["vel_cte"], 1, 0)
        grid.addWidget(self.gauges["pos_cte"], 1, 1)
        grid.addWidget(self.gauges["accel"], 2, 0)
        grid.addWidget(self.gauges["steer"], 2, 1)
        btn_row = QtWidgets.QHBoxLayout()
        for label, slot in (
            ("Step", self._step),
            ("Align", self._align),
            ("W", lambda: self._drive(3, 0)),
            ("A", lambda: self._drive(0, 0.7)),
            ("S", lambda: self._drive(-3, 0)),
            ("D", lambda: self._drive(0, -0.7)),
        ):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        grid.addLayout(btn_row, 3, 0, 1, 3)
        root.setting.cte_vel.bind(self._update_gauges)
        root.setting.cte_steer.bind(self._update_gauges)
        root.setting.cmd_accel.bind(self._update_gauges)
        root.setting.cmd_steer.bind(self._update_gauges)

    def _update_gauges(self) -> None:
        s = self.root.setting
        self.gauges["vel_cte"].setText(f"Vel CTE: {float(s.cte_vel.get()):.2f}")
        self.gauges["pos_cte"].setText(f"Pos CTE: {float(s.cte_steer.get()):.2f}")
        self.gauges["accel"].setText(f"Accel: {float(s.cmd_accel.get()):.2f}")
        self.gauges["steer"].setText(f"Steer: {float(s.cmd_steer.get()):.2f}")

    def _drive(self, accel: float, steer: float) -> None:
        if self.root.exec is None:
            return
        self.root.apply_world_control(
            ControlCommand(acceleration=accel, steer=steer),
            dt=float(self.root.setting.sim_dt.get()),
        )
        self.root.update_ui()

    def _step(self) -> None:
        if self.root.exec is None or self.root.exec.controller is None:
            return
        cmd = self.root.exec.controller.control(
            perception_model=self.root.exec.pm,
            sensors=self.root.exec.world.get_sensor_frame(),
        )
        self.root.apply_world_control(cmd, dt=float(self.root.setting.control_dt.get()))
        self.root.update_ui()

    def _align(self) -> None:
        if self.root.exec is None or self.root.exec.controller is None:
            return
        if hasattr(self.root.exec.controller, "align"):
            self.root.exec.controller.align()
        self.root.update_ui()

    def refresh(self) -> None:
        choices = self.root.setting.registry_choices()
        cur = self.ctrl.currentText()
        self.ctrl.blockSignals(True)
        self.ctrl.clear()
        self.ctrl.addItems(choices["controller"])
        idx = self.ctrl.findText(cur)
        self.ctrl.setCurrentIndex(idx if idx >= 0 else 0)
        self.ctrl.blockSignals(False)


class ExecutionPanel(QtWidgets.QGroupBox):
    def __init__(self, root: "VisualizerQtWindow"):
        super().__init__("Execution")
        self.root = root
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._exec_tick)
        v = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        choices = root.setting.registry_choices()
        top.addWidget(QtWidgets.QLabel("Executer"))
        self.executer = _combo(
            self, root.setting.executer_type, choices["executer"],
            on_change=lambda: root.reload_stack(reload_code=False),
        )
        top.addWidget(self.executer)
        top.addWidget(QtWidgets.QLabel("Bridge"))
        self.bridge = _combo(
            self, root.setting.execution_bridge, choices["bridge"],
            on_change=lambda: root.reload_stack(reload_code=False),
        )
        top.addWidget(self.bridge)
        top.addWidget(_check(self, "Perception", root.setting.exec_perceive))
        top.addWidget(_check(self, "Planning", root.setting.exec_plan))
        top.addWidget(_check(self, "Control", root.setting.exec_control))
        top.addWidget(_check(self, "Tasks", root.setting.exec_tasks))
        v.addLayout(top)

        dts = QtWidgets.QHBoxLayout()
        for label, pace_var, dt_var in (
            ("Perception Δt", root.setting.pace_perception, root.setting.perception_dt),
            ("Replan Δt", root.setting.pace_replan, root.setting.replan_dt),
            ("Control Δt", root.setting.pace_control, root.setting.control_dt),
            ("Sim Δt", root.setting.pace_sim, root.setting.sim_dt),
        ):
            dts.addWidget(_check(self, "", pace_var))
            dts.addWidget(QtWidgets.QLabel(label))
            dts.addWidget(_dbl_spin(self, dt_var))
        v.addLayout(dts)

        tasks = QtWidgets.QHBoxLayout()
        tasks.addWidget(QtWidgets.QLabel("Tasks"))
        self.tasks_edit = QtWidgets.QLineEdit(str(root.setting.execution_tasks.get()))
        self.tasks_edit.editingFinished.connect(
            lambda: root.setting.execution_tasks.set(self.tasks_edit.text())
        )
        root.setting.execution_tasks.bind(
            lambda: self.tasks_edit.setText(str(root.setting.execution_tasks.get()))
        )
        tasks.addWidget(self.tasks_edit, 1)
        add_btn = QtWidgets.QPushButton("+")
        add_btn.clicked.connect(self._add_task)
        tasks.addWidget(add_btn)
        v.addLayout(tasks)

        btns = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.start_btn.clicked.connect(self.toggle_exec)
        btns.addWidget(self.start_btn)
        for label, slot in (
            ("Stop", self.stop_exec),
            ("Step", self.step_exec),
            ("Reset", self.reset_exec),
        ):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(slot)
            btns.addWidget(b)
        v.addLayout(btns)

        stats = QtWidgets.QHBoxLayout()
        self.stats_lbl = QtWidgets.QLabel("real=0  sim=0  perc=0  plan=0  ctrl=0  ui=0  plot=0ms")
        stats.addWidget(self.stats_lbl, 1)
        v.addLayout(stats)
        for var in (
            root.setting.elapsed_real_time, root.setting.elapsed_sim_time,
            root.setting.perception_fps, root.setting.replan_fps, root.setting.control_fps,
            root.setting.ui_fps, root.setting.plot_latency_ms,
        ):
            var.bind(self._update_stats)

    def _update_stats(self) -> None:
        s = self.root.setting
        self.stats_lbl.setText(
            f"real={s.elapsed_real_time.get()}  sim={s.elapsed_sim_time.get()}  "
            f"perc={s.perception_fps.get()}  plan={s.replan_fps.get()}  ctrl={s.control_fps.get()}  "
            f"ui={s.ui_fps.get()} Hz  plot={s.plot_latency_ms.get()} ms"
        )

    def _add_task(self) -> None:
        selected = set(self.root.setting.execution_task_names())
        available = sorted(n for n in TaskStrategy.registry if n not in selected)
        if not available:
            QtWidgets.QMessageBox.information(self, "Tasks", "All registered tasks are already selected.")
            return
        name, ok = QtWidgets.QInputDialog.getItem(self, "Add Task", "TaskStrategy:", available, 0, False)
        if ok and name:
            names = self.root.setting.execution_task_names() + [name]
            self.root.setting.execution_tasks.set(",".join(names))
            self.root.reload_stack(reload_code=False)

    def toggle_exec(self) -> None:
        if self.root.setting.exec_running:
            self.stop_exec()
            return
        self.root.setting.exec_running = True
        self.start_btn.setEnabled(False)
        # Prefer executer-reported poll delay; fall back to ~1ms busy poll.
        delay_s = getattr(self.root.exec, "ui_poll_delay", None) if self.root.exec else None
        interval_ms = int((delay_s if delay_s is not None else 0.001) * 1000)
        self._timer.start(max(1, interval_ms))
        self.root.update_ui()

    def _exec_tick(self) -> None:
        if not self.root.setting.exec_running or self.root.exec is None:
            return
        s = self.root.setting
        self.root.exec.step(
            control_dt=float(s.control_dt.get()),
            replan_dt=float(s.replan_dt.get()),
            perception_dt=float(s.perception_dt.get()),
            sim_dt=float(s.sim_dt.get()),
            call_replan=bool(s.exec_plan.get()),
            call_control=bool(s.exec_control.get()),
            call_perceive=bool(s.exec_perceive.get()),
            call_tasks=bool(s.exec_tasks.get()),
            pace_perception=bool(s.pace_perception.get()),
            pace_replan=bool(s.pace_replan.get()),
            pace_control=bool(s.pace_control.get()),
            pace_sim=bool(s.pace_sim.get()),
        )
        # UI redraw is owned by the window's refresh timer (decoupled from sim).
        delay_s = getattr(self.root.exec, "ui_poll_delay", None)
        if delay_s is not None:
            self._timer.setInterval(max(1, int(delay_s * 1000)))

    def stop_exec(self) -> None:
        self.root.setting.exec_running = False
        self._timer.stop()
        if self.root.exec is not None:
            self.root.exec.stop()
        self.start_btn.setEnabled(True)
        self.root.update_ui()

    def step_exec(self) -> None:
        if self.root.exec is None:
            return
        s = self.root.setting
        self.root.exec.step(
            control_dt=float(s.control_dt.get()),
            replan_dt=float(s.replan_dt.get()),
            perception_dt=float(s.perception_dt.get()),
            sim_dt=float(s.sim_dt.get()),
            call_replan=bool(s.exec_plan.get()),
            call_control=bool(s.exec_control.get()),
            call_perceive=bool(s.exec_perceive.get()),
            call_tasks=bool(s.exec_tasks.get()),
            pace_perception=bool(s.pace_perception.get()),
            pace_replan=bool(s.pace_replan.get()),
            pace_control=bool(s.pace_control.get()),
            pace_sim=bool(s.pace_sim.get()),
        )
        self.root.update_ui()

    def reset_exec(self) -> None:
        if self.root.exec is not None:
            self.root.exec.reset()
        self.root.update_ui()

    def refresh(self) -> None:
        choices = self.root.setting.registry_choices()
        for box, key in ((self.executer, "executer"), (self.bridge, "bridge")):
            cur = box.currentText()
            box.blockSignals(True)
            box.clear()
            box.addItems(choices[key])
            idx = box.findText(cur)
            box.setCurrentIndex(idx if idx >= 0 else 0)
            box.blockSignals(False)
