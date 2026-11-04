"""High-performance pyqtgraph plot widgets (no Matplotlib).

Each frame updates PlotDataItem/ScatterPlotItem via ``setData`` — O(n) GPU/CPU
path instead of rebuilding a Matplotlib figure. Suitable for 60+ FPS UI refresh
while AsyncThreadedExecuter / SyncExecuter.step runs off or between UI ticks.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from .qt_binding import import_pyqtgraph, import_qt

if TYPE_CHECKING:
    from .main_window import VisualizerQtWindow

log = logging.getLogger(__name__)

QtCore, QtGui, QtWidgets = import_qt()
pg = import_pyqtgraph()


def _closed_poly(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if corners is None or len(corners) == 0:
        return np.array([]), np.array([])
    xs = np.append(corners[:, 0], corners[0, 0])
    ys = np.append(corners[:, 1], corners[0, 1])
    return xs, ys


def _safe_xy(xs, ys) -> tuple[np.ndarray, np.ndarray]:
    if xs is None or ys is None:
        return np.array([]), np.array([])
    x = np.asarray(xs, dtype=float).ravel()
    y = np.asarray(ys, dtype=float).ravel()
    n = min(len(x), len(y))
    return x[:n], y[:n]


class GlobalPlotWidget(pg.PlotWidget):
    """Global map / raceline / HD-map overview with ego marker."""

    def __init__(self, root: "VisualizerQtWindow", parent=None):
        super().__init__(parent=parent, background=root.setting.p60_bg_color)
        self.root = root
        self.setLabel("bottom", "x (m)")
        self.setLabel("left", "y (m)")
        self.showGrid(x=True, y=True, alpha=0.25)
        self.setAspectLocked(True)
        self._center_delta = np.array([0.0, 0.0])
        self._map_bounds: Optional[tuple[float, float, float, float]] = None
        self._drag_pan = False
        self._drag_teleport = False
        self._last_pos = None
        self._start_goal: list[tuple[float, float]] = []
        self._hover_plan_xy: Optional[np.ndarray] = None
        self._hover_plan_v: Optional[np.ndarray] = None

        self.left_boundary = self.plotItem.plot(pen=pg.mkPen("#ff8c00", width=2))
        self.right_boundary = self.plotItem.plot(pen=pg.mkPen("#d2b48c", width=2))
        self.global_plan = self.plotItem.plot(pen=pg.mkPen("#4fc3f7", width=3))
        self.hdmap_lanes: list[pg.PlotDataItem] = []
        self.ego_box = self.plotItem.plot(pen=pg.mkPen("r", width=2))
        self.ego_marker = self.plotItem.plot(pen=None, symbol="o", symbolBrush="r", symbolSize=10)
        self.past_path = self.plotItem.plot(pen=pg.mkPen("#66bb6a", width=1.5))
        self.start_marker = self.plotItem.plot(pen=None, symbol="o", symbolBrush="b", symbolSize=14)
        self.goal_marker = self.plotItem.plot(pen=None, symbol="o", symbolBrush="g", symbolSize=14)
        self.hover_marker = self.plotItem.plot(pen=None, symbol="o", symbolBrush="w", symbolSize=8)
        self.hover_text = pg.TextItem(color="w", anchor=(0, 1))
        self.addItem(self.hover_text)
        self.hover_text.hide()

        self.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        vb = self.getViewBox()
        vb.mouseDragEvent = self._mouse_drag_event  # type: ignore[method-assign]
        vb.wheelEvent = self._wheel_event  # type: ignore[method-assign]

        self._last_draw_ms = 0.0
        self._fps = 0.0
        self._fps_t0 = time.perf_counter()
        self._fps_frames = 0

    def reset(self) -> None:
        self._map_bounds = None
        self._center_delta[:] = 0.0
        self._start_goal.clear()
        for item in self.hdmap_lanes:
            self.removeItem(item)
        self.hdmap_lanes.clear()
        for curve in (
            self.left_boundary, self.right_boundary, self.global_plan,
            self.ego_box, self.ego_marker, self.past_path,
            self.start_marker, self.goal_marker,
        ):
            curve.setData([], [])

    def update_theme(self, bg: str, fg: str) -> None:
        self.setBackground(bg)
        self.getAxis("bottom").setPen(fg)
        self.getAxis("left").setPen(fg)

    def _wheel_event(self, ev):  # noqa: N802
        delta = ev.delta() if hasattr(ev, "delta") else ev.angleDelta().y()
        factor = 0.9 if delta > 0 else 1.1
        self.root.setting.p66_global_zoom = float(
            np.clip(self.root.setting.p66_global_zoom * factor, 5.0, 5000.0)
        )
        ev.accept()

    def _mouse_drag_event(self, ev, axis=None):  # noqa: N802
        if ev.button() not in (QtCore.Qt.LeftButton, QtCore.Qt.NoButton) and ev.button() != QtCore.Qt.LeftButton:
            return pg.ViewBox.mouseDragEvent(self.getViewBox(), ev, axis=axis)
        pos = self.getViewBox().mapSceneToView(ev.scenePos())
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if ev.isStart():
            self._last_pos = (pos.x(), pos.y())
            self._drag_pan = bool(modifiers & QtCore.Qt.ControlModifier)
            self._drag_teleport = not self._drag_pan
            ev.accept()
            return
        if ev.isFinish():
            if self._drag_teleport and self.root.exec is not None and self._last_pos is not None:
                x0, y0 = self._last_pos
                theta = float(np.arctan2(pos.y() - y0, pos.x() - x0))
                self.root.teleport_ego(pos.x(), pos.y(), theta)
                self.root.update_ui()
            self._drag_pan = self._drag_teleport = False
            self._last_pos = None
            ev.accept()
            return
        if self._drag_pan and self._last_pos is not None:
            dx = pos.x() - self._last_pos[0]
            dy = pos.y() - self._last_pos[1]
            self._center_delta[0] -= dx
            self._center_delta[1] -= dy
            self._last_pos = (pos.x(), pos.y())
        ev.accept()

    def _on_mouse_clicked(self, ev):
        if ev.button() != QtCore.Qt.RightButton or self.root.exec is None:
            return
        pos = self.getViewBox().mapSceneToView(ev.scenePos())
        pt = (float(pos.x()), float(pos.y()))
        self._start_goal.append(pt)
        if len(self._start_goal) == 1:
            self.start_marker.setData([pt[0]], [pt[1]])
            self.goal_marker.setData([], [])
        elif len(self._start_goal) >= 2:
            start, goal = self._start_goal[0], self._start_goal[1]
            self.start_marker.setData([start[0]], [start[1]])
            self.goal_marker.setData([goal[0]], [goal[1]])
            gp = self.root.exec.global_planner
            if gp is not None and hasattr(gp, "set_start_goal"):
                gp.set_start_goal(start, goal)
                self.root.replan_global()
            self._start_goal.clear()

    def _on_mouse_moved(self, pos):
        if self._hover_plan_xy is None or len(self._hover_plan_xy) == 0:
            self.hover_marker.setData([], [])
            self.hover_text.hide()
            return
        mouse = self.getViewBox().mapSceneToView(pos)
        xy = self._hover_plan_xy
        d2 = (xy[:, 0] - mouse.x()) ** 2 + (xy[:, 1] - mouse.y()) ** 2
        i = int(np.argmin(d2))
        if d2[i] > (self.root.setting.p66_global_zoom * 0.05) ** 2:
            self.hover_marker.setData([], [])
            self.hover_text.hide()
            return
        x, y = xy[i]
        v = float(self._hover_plan_v[i]) if self._hover_plan_v is not None else 0.0
        self.hover_marker.setData([x], [y])
        self.hover_text.setText(f"{v:.1f} m/s ({v * 3.6:.0f} km/h)")
        self.hover_text.setPos(x, y)
        self.hover_text.show()

    def redraw(self) -> float:
        """Redraw from executer state. Returns draw latency in ms."""
        t0 = time.perf_counter()
        stack = self.root.exec
        if stack is None:
            return 0.0
        setting = self.root.setting
        ego = stack.ego_state

        # Map / plan geometry
        plan = None
        if stack.global_planner is not None:
            plan = getattr(stack.global_planner, "global_plan", None)
        if (plan is None or getattr(plan, "trajectory", None) is None) and stack.local_planner is not None:
            plan = getattr(stack.local_planner, "global_plan", None)

        if setting.p66_show_global_plan_boundaries.get() and plan is not None:
            self.left_boundary.setData(*_safe_xy(plan.left_boundary_x, plan.left_boundary_y))
            self.right_boundary.setData(*_safe_xy(plan.right_boundary_x, plan.right_boundary_y))
            xs = list(plan.left_boundary_x or []) + list(plan.right_boundary_x or [])
            ys = list(plan.left_boundary_y or []) + list(plan.right_boundary_y or [])
            if xs and ys:
                self._map_bounds = (min(xs), max(xs), min(ys), max(ys))
        else:
            self.left_boundary.setData([], [])
            self.right_boundary.setData([], [])

        # HD map lanes (static-ish; redraw when empty)
        gp = stack.global_planner
        if gp is not None and hasattr(gp, "hdmap") and gp.hdmap is not None and not self.hdmap_lanes:
            for lane in gp.hdmap.lanes:
                color = "#427b58" if lane.type == "driving" else (
                    "#af3a03" if lane.type == "shoulder" else "#777777"
                )
                alpha = 0.7 if lane.type == "driving" else 0.35
                item = self.plotItem.plot(
                    lane.center_line[0], lane.center_line[1],
                    pen=pg.mkPen(pg.mkColor(color), width=1.5),
                )
                item.setOpacity(alpha)
                self.hdmap_lanes.append(item)
            all_x = [c for lane in gp.hdmap.lanes for c in lane.center_line[0]]
            all_y = [c for lane in gp.hdmap.lanes for c in lane.center_line[1]]
            if all_x and all_y:
                self._map_bounds = (min(all_x), max(all_x), min(all_y), max(all_y))

        if setting.p66_show_global_plan.get() and plan is not None and plan.trajectory is not None:
            traj = plan.trajectory
            xs, ys = _safe_xy(traj.path_x, traj.path_y)
            self.global_plan.setData(xs, ys)
            if len(xs):
                self._hover_plan_xy = np.column_stack([xs, ys])
                self._hover_plan_v = np.asarray(traj.velocity, dtype=float).ravel()[: len(xs)]
        else:
            self.global_plan.setData([], [])
            self._hover_plan_xy = None

        # Ego
        if setting.p66_show_state.get():
            bx, by = _closed_poly(ego.get_bb_corners())
            self.ego_box.setData(bx, by)
            self.ego_marker.setData([ego.x], [ego.y])
        else:
            self.ego_box.setData([], [])
            self.ego_marker.setData([], [])

        if setting.p66_show_past_locations.get() and stack.local_planner is not None:
            hist = getattr(stack.local_planner, "past_locations", None) or getattr(
                stack, "past_locations", None
            )
            if hist:
                pts = np.asarray(hist[-100:], dtype=float)
                if pts.ndim == 2 and pts.shape[1] >= 2:
                    self.past_path.setData(pts[:, 0], pts[:, 1])
                else:
                    self.past_path.setData([], [])
            else:
                self.past_path.setData([], [])

        # View window
        zoom = float(setting.p66_global_zoom)
        cx = ego.x + self._center_delta[0]
        cy = ego.y + self._center_delta[1]
        if setting.p66_global_view_follow_planner.get() and stack.local_planner is not None:
            loc = getattr(stack.local_planner, "location_xy", None)
            if loc is not None:
                cx, cy = float(loc[0]), float(loc[1])
        self.setXRange(cx - zoom, cx + zoom, padding=0)
        aspect = max(self.width() / max(self.height(), 1), 0.25)
        self.setYRange(cy - zoom / aspect, cy + zoom / aspect, padding=0)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._last_draw_ms = dt_ms
        self._fps_frames += 1
        elapsed = time.perf_counter() - self._fps_t0
        if elapsed >= 0.5:
            self._fps = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_t0 = time.perf_counter()
        return dt_ms


class LocalPlotWidget(QtWidgets.QWidget):
    """Dual-pane local XY + Frenet view."""

    def __init__(self, root: "VisualizerQtWindow", parent=None):
        super().__init__(parent)
        self.root = root
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.xy_plot = pg.PlotWidget(background=root.setting.p60_bg_color)
        self.frenet_plot = pg.PlotWidget(background=root.setting.p60_bg_color)
        for pw, xlabel, ylabel in (
            (self.xy_plot, "x (m)", "y (m)"),
            (self.frenet_plot, "s (m)", "d (m)"),
        ):
            pw.setLabel("bottom", xlabel)
            pw.setLabel("left", ylabel)
            pw.showGrid(x=True, y=True, alpha=0.25)
            pw.setAspectLocked(True)
        layout.addWidget(self.xy_plot, 1)
        layout.addWidget(self.frenet_plot, 1)

        self._init_curves()
        self._last_draw_ms = 0.0
        self._npc_drag = False
        self._npc_start = None

        vb = self.xy_plot.getViewBox()
        vb.mouseDragEvent = self._xy_drag  # type: ignore[method-assign]
        vb.wheelEvent = self._xy_wheel  # type: ignore[method-assign]
        self.frenet_plot.getViewBox().wheelEvent = self._frenet_wheel  # type: ignore[method-assign]

    def _init_curves(self) -> None:
        self.xy_left = self.xy_plot.plotItem.plot(pen=pg.mkPen("#ff8c00", width=2))
        self.xy_right = self.xy_plot.plotItem.plot(pen=pg.mkPen("#d2b48c", width=2))
        self.xy_ref = self.xy_plot.plotItem.plot(pen=pg.mkPen("#888888", width=2))
        self.xy_local = self.xy_plot.plotItem.plot(pen=pg.mkPen("#ef5350", width=3))
        self.xy_ego = self.xy_plot.plotItem.plot(pen=pg.mkPen("r", width=2))
        self.xy_agents = [
            self.xy_plot.plotItem.plot(pen=pg.mkPen("#1565c0", width=1.5)) for _ in range(12)
        ]
        self.xy_lidar = pg.ScatterPlotItem(size=5, brush=pg.mkBrush(50, 255, 50, 200), pen=None)
        self.xy_plot.addItem(self.xy_lidar)
        self.xy_clusters = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(255, 255, 0, 220), pen=None)
        self.xy_plot.addItem(self.xy_clusters)
        self.xy_pred = self.xy_plot.plotItem.plot(
            pen=pg.mkPen("#ff9800", width=1.5, style=QtCore.Qt.DashLine)
        )
        self.xy_past = self.xy_plot.plotItem.plot(pen=pg.mkPen("#66bb6a", width=1.5))
        self.xy_lattice = self.xy_plot.plotItem.plot(
            pen=pg.mkPen("#90caf9", width=1), connect="finite"
        )

        self.sd_left = self.frenet_plot.plotItem.plot(pen=pg.mkPen("#ff8c00", width=1))
        self.sd_right = self.frenet_plot.plotItem.plot(pen=pg.mkPen("#d2b48c", width=1))
        self.sd_ref = self.frenet_plot.plotItem.plot(pen=pg.mkPen("#888888", width=1))
        self.sd_local = self.frenet_plot.plotItem.plot(pen=pg.mkPen("#ef5350", width=3))
        self.sd_ego = self.frenet_plot.plotItem.plot(pen=pg.mkPen("r", width=2))
        self.sd_agents = [
            self.frenet_plot.plotItem.plot(pen=pg.mkPen("#1565c0", width=1.5)) for _ in range(12)
        ]
        self.sd_lidar = pg.ScatterPlotItem(size=5, brush=pg.mkBrush(50, 255, 50, 200), pen=None)
        self.frenet_plot.addItem(self.sd_lidar)

    def reset(self) -> None:
        for curve in (
            self.xy_left, self.xy_right, self.xy_ref, self.xy_local, self.xy_ego,
            self.xy_pred, self.xy_past, self.xy_lattice,
            self.sd_left, self.sd_right, self.sd_ref, self.sd_local, self.sd_ego,
        ):
            curve.setData([], [])
        for c in self.xy_agents + self.sd_agents:
            c.setData([], [])
        self.xy_lidar.setData([], [])
        self.xy_clusters.setData([], [])
        self.sd_lidar.setData([], [])

    def update_theme(self, bg: str, fg: str) -> None:
        for pw in (self.xy_plot, self.frenet_plot):
            pw.setBackground(bg)
            pw.getAxis("bottom").setPen(fg)
            pw.getAxis("left").setPen(fg)

    def update_visibility(self) -> None:
        s = self.root.setting
        self.xy_plot.setVisible(bool(s.p67_show_local_global_view.get()))
        self.frenet_plot.setVisible(bool(s.p67_show_local_frenet_view.get()))

    def _xy_wheel(self, ev):  # noqa: N802
        delta = ev.delta() if hasattr(ev, "delta") else ev.angleDelta().y()
        factor = 0.9 if delta > 0 else 1.1
        self.root.setting.p66_xy_zoom = float(
            np.clip(self.root.setting.p66_xy_zoom * factor, 5.0, 500.0)
        )
        ev.accept()

    def _frenet_wheel(self, ev):  # noqa: N802
        delta = ev.delta() if hasattr(ev, "delta") else ev.angleDelta().y()
        factor = 0.9 if delta > 0 else 1.1
        self.root.setting.p66_frenet_zoom = float(
            np.clip(self.root.setting.p66_frenet_zoom * factor, 5.0, 500.0)
        )
        ev.accept()

    def _xy_drag(self, ev, axis=None):  # noqa: N802
        if self.root.exec is None:
            return
        pos = self.xy_plot.getViewBox().mapSceneToView(ev.scenePos())
        if ev.button() == QtCore.Qt.LeftButton or (
            ev.button() == QtCore.Qt.NoButton and self._npc_drag is False and getattr(self, "_teleporting", False)
        ):
            if ev.button() == QtCore.Qt.LeftButton or getattr(self, "_teleporting", False):
                if ev.isStart():
                    self._teleporting = True
                    self._teleport_start = (pos.x(), pos.y())
                    ev.accept()
                    return
                if ev.isFinish() and getattr(self, "_teleporting", False):
                    x0, y0 = self._teleport_start
                    theta = float(np.arctan2(pos.y() - y0, pos.x() - x0))
                    self.root.teleport_ego(pos.x(), pos.y(), theta)
                    self.root.update_ui()
                    self._teleporting = False
                    ev.accept()
                    return
                ev.accept()
                return
        if ev.button() == QtCore.Qt.RightButton or self._npc_drag:
            from avlite.c10_perception.c11_perception_model import AgentState

            if ev.isStart():
                self._npc_drag = True
                self._npc_start = (pos.x(), pos.y())
                ev.accept()
                return
            if ev.isFinish() and self._npc_drag and self._npc_start is not None:
                x0, y0 = self._npc_start
                theta = float(np.arctan2(pos.y() - y0, pos.x() - x0))
                agent = AgentState()
                agent.x, agent.y, agent.theta = float(pos.x()), float(pos.y()), theta
                agent.velocity = float(self.root.exec.ego_state.velocity)
                self.root.spawn_agent(agent)
                self.root.update_ui()
                self._npc_drag = False
                self._npc_start = None
                ev.accept()
                return
            ev.accept()
            return
        return pg.ViewBox.mouseDragEvent(self.xy_plot.getViewBox(), ev, axis=axis)

    def redraw(self) -> float:
        t0 = time.perf_counter()
        stack = self.root.exec
        if stack is None:
            return 0.0
        s = self.root.setting
        self.update_visibility()
        ego = stack.ego_state
        lp = stack.local_planner
        gt = getattr(lp, "global_trajectory", None) if lp is not None else None

        # Boundaries / reference
        plan = getattr(lp, "global_plan", None) if lp is not None else None
        if plan is not None and s.p66_show_race_boundary.get():
            self.xy_left.setData(*_safe_xy(plan.left_boundary_x, plan.left_boundary_y))
            self.xy_right.setData(*_safe_xy(plan.right_boundary_x, plan.right_boundary_y))
        else:
            self.xy_left.setData([], [])
            self.xy_right.setData([], [])

        if plan is not None and plan.trajectory is not None and s.p66_show_global_plan.get():
            self.xy_ref.setData(*_safe_xy(plan.trajectory.path_x, plan.trajectory.path_y))
            if gt is not None:
                # Frenet reference: s along path, d≈0
                sx = np.asarray(plan.trajectory.path_s, dtype=float)
                self.sd_ref.setData(sx, np.zeros_like(sx))
                if hasattr(plan, "left_boundary_d") and plan.left_boundary_d is not None:
                    sd_s = np.asarray(getattr(plan, "path_s", plan.trajectory.path_s), dtype=float)
                    n = min(len(sd_s), len(plan.left_boundary_d), len(plan.right_boundary_d))
                    self.sd_left.setData(sd_s[:n], np.asarray(plan.left_boundary_d[:n], dtype=float))
                    self.sd_right.setData(sd_s[:n], np.asarray(plan.right_boundary_d[:n], dtype=float))
        else:
            self.xy_ref.setData([], [])
            self.sd_ref.setData([], [])

        # Local plan
        if lp is not None and s.p66_show_local_plan.get():
            local_plan = getattr(lp, "local_plan", None)
            traj = getattr(local_plan, "trajectory", None) if local_plan is not None else None
            if traj is not None:
                self.xy_local.setData(*_safe_xy(traj.path_x, traj.path_y))
                if gt is not None:
                    try:
                        sd = gt.convert_xy_path_to_sd_path_np(
                            np.column_stack([traj.path_x, traj.path_y])
                        )
                        self.sd_local.setData(sd[:, 0], sd[:, 1])
                    except Exception:
                        self.sd_local.setData([], [])
            else:
                self.xy_local.setData([], [])
                self.sd_local.setData([], [])
        else:
            self.xy_local.setData([], [])
            self.sd_local.setData([], [])

        # Lattice
        if lp is not None and s.p66_show_local_lattice.get():
            edges = getattr(lp, "lattice_edges", None) or getattr(lp, "edges", None) or []
            xs: list[float] = []
            ys: list[float] = []
            for edge in edges[:200]:
                pts = getattr(edge, "points_xy", None) or getattr(edge, "xy", None)
                if pts is None:
                    continue
                arr = np.asarray(pts, dtype=float)
                if arr.ndim != 2 or arr.shape[0] < 2:
                    continue
                xs.extend(arr[:, 0].tolist() + [np.nan])
                ys.extend(arr[:, 1].tolist() + [np.nan])
            self.xy_lattice.setData(xs, ys)
        else:
            self.xy_lattice.setData([], [])

        # Ego + agents
        if s.p66_show_state.get():
            self.xy_ego.setData(*_closed_poly(ego.get_bb_corners()))
            if gt is not None:
                try:
                    sd_c = gt.convert_xy_path_to_sd_path_np(ego.get_bb_corners())
                    self.sd_ego.setData(*_closed_poly(sd_c))
                except Exception:
                    self.sd_ego.setData([], [])
            agents = list(getattr(stack.pm, "agents", []) or [])[:12]
            for i, curve in enumerate(self.xy_agents):
                if i < len(agents):
                    curve.setData(*_closed_poly(agents[i].get_bb_corners()))
                else:
                    curve.setData([], [])
            if gt is not None:
                for i, curve in enumerate(self.sd_agents):
                    if i < len(agents):
                        try:
                            sd_c = gt.convert_xy_path_to_sd_path_np(agents[i].get_bb_corners())
                            curve.setData(*_closed_poly(sd_c))
                        except Exception:
                            curve.setData([], [])
                    else:
                        curve.setData([], [])
        else:
            self.xy_ego.setData([], [])
            self.sd_ego.setData([], [])

        # LiDAR
        lidar_xy = None
        try:
            frame = stack.world.get_sensor_frame()
            lidar = getattr(frame, "lidar", None)
            pts = getattr(lidar, "points", None) if lidar is not None else None
            if pts is not None and len(pts):
                arr = np.asarray(pts, dtype=float)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    lidar_xy = arr[:, :2]
        except Exception:
            lidar_xy = None

        if lidar_xy is not None and s.p66_show_lidar_global.get():
            self.xy_lidar.setData(lidar_xy[:, 0], lidar_xy[:, 1])
        else:
            self.xy_lidar.setData([], [])
        if lidar_xy is not None and s.p66_show_lidar_frenet.get() and gt is not None:
            try:
                sd = gt.convert_xy_path_to_sd_path_np(lidar_xy)
                self.sd_lidar.setData(sd[:, 0], sd[:, 1])
            except Exception:
                self.sd_lidar.setData([], [])
        else:
            self.sd_lidar.setData([], [])

        # Predictions (first mode polylines flattened)
        if s.p67_show_prediction.get() and stack.pm is not None:
            pred_x: list[float] = []
            pred_y: list[float] = []
            for agent in getattr(stack.pm, "agents", []) or []:
                pred = getattr(agent, "prediction", None)
                traj = None
                if pred is not None:
                    if hasattr(pred, "trajectories") and pred.trajectories:
                        traj = pred.trajectories[0]
                    elif hasattr(pred, "trajectory"):
                        traj = pred.trajectory
                if traj is None:
                    continue
                xs = getattr(traj, "path_x", None)
                ys = getattr(traj, "path_y", None)
                if xs is None:
                    continue
                pred_x.extend(list(xs) + [np.nan])
                pred_y.extend(list(ys) + [np.nan])
            self.xy_pred.setData(pred_x, pred_y)
        else:
            self.xy_pred.setData([], [])

        # View windows
        zoom = float(s.p66_xy_zoom)
        cx, cy = ego.x, ego.y
        if s.p66_global_view_follow_planner.get() and lp is not None:
            loc = getattr(lp, "location_xy", None)
            if loc is not None:
                cx, cy = float(loc[0]), float(loc[1])
        self.xy_plot.setXRange(cx - zoom, cx + zoom, padding=0)
        aspect = max(self.xy_plot.width() / max(self.xy_plot.height(), 1), 0.25)
        self.xy_plot.setYRange(cy - zoom / aspect, cy + zoom / aspect, padding=0)

        fzoom = float(s.p66_frenet_zoom)
        if gt is not None:
            try:
                sd_ego = gt.convert_xy_to_sd(ego.x, ego.y)
                cs, cd = float(sd_ego[0]), float(sd_ego[1])
            except Exception:
                cs, cd = 0.0, 0.0
        else:
            cs, cd = 0.0, 0.0
        self.frenet_plot.setXRange(cs - fzoom, cs + fzoom, padding=0)
        self.frenet_plot.setYRange(cd - fzoom / 2, cd + fzoom / 2, padding=0)

        self._last_draw_ms = (time.perf_counter() - t0) * 1000.0
        return self._last_draw_ms
