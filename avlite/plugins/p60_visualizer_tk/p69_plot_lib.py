from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c10_perception.c11_perception_model import (
    AggregatedOccupancyFlow,
    EgoState,
    GMM,
    GP,
    HDMap,
    MultiTrajectory,
    OccupancyFlow,
    OccupancyMap,
    SingleTrajectory,
)
from avlite.c10_perception.c12_perception_strategy import PerceptionModel
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c20_planning.c28_local_lattice_planners import Edge
from avlite.c50_common.c54_trajectory_tracker import TrajectoryTracker
from avlite.c40_execution.c44_sync_executer import SyncExecuter
from avlite.c40_execution.c46_basic_sim import boundary_segments_from_global_plan
from avlite.c20_planning.c24_global_hdmap_planners import HDMapGlobalPlanner
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c30_control.c39_settings import ControlSettings

from typing import cast, Optional
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba

import logging

log = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


class BlitManager:
    """Minimal matplotlib blitting helper for a multi-axes figure.

    Caches each axis' background whenever a real ``draw()`` happens and, on the
    fast path, restores those backgrounds and re-renders only the registered
    ``animated`` artists. The caller decides when a full redraw is required
    (e.g. the axis limits or canvas size changed); everything else can go
    through :meth:`update`.
    """

    def __init__(self, fig, axes):
        self.fig = fig
        self.axes = list(axes)
        self._artists: list = []
        self._bg: dict = {}
        self._connected_canvas = None
        self._drawing = False
        self._pending = False

    def add_artist(self, artist) -> None:
        artist.set_animated(True)
        self._artists.append(artist)

    def _ensure_connected(self) -> None:
        # The Tk canvas is created after the figure (FigureCanvasTkAgg replaces
        # fig.canvas), so (re)connect the draw_event handler lazily.
        canvas = self.fig.canvas
        if canvas is not self._connected_canvas:
            canvas.mpl_connect("draw_event", self._on_draw)
            self._connected_canvas = canvas
            self._bg = {}

    def _on_draw(self, _event) -> None:
        canvas = self.fig.canvas
        self._bg = {ax: canvas.copy_from_bbox(ax.bbox) for ax in self.axes}

    def full_draw(self) -> None:
        """Full software redraw; repopulates the cached backgrounds."""
        self._ensure_connected()
        if self._drawing:
            self._pending = True
            return
        self._drawing = True
        try:
            self.fig.canvas.draw()
            self._blit_animated()
            if self._pending:
                self._pending = False
                self._blit_animated()
        finally:
            self._drawing = False

    def update(self) -> None:
        """Fast path: restore cached backgrounds and blit animated artists."""
        self._ensure_connected()
        if self._drawing:
            self._pending = True
            return
        if not self._bg:
            self.full_draw()
            return
        self._drawing = True
        try:
            self._blit_animated()
            if self._pending:
                self._pending = False
                self._blit_animated()
        finally:
            self._drawing = False

    def _blit_animated(self) -> None:
        # No flush_events here: under TkAgg it re-enters motion handlers and
        # nests blits until RecursionError. canvas.blit() already updates the UI.
        canvas = self.fig.canvas
        for ax in self.axes:
            bg = self._bg.get(ax)
            if bg is not None:
                canvas.restore_region(bg)
        for artist in sorted(self._artists, key=lambda a: a.get_zorder()):
            ax = getattr(artist, "axes", None)
            if ax is None or not ax.get_visible():
                continue
            ax.draw_artist(artist)
        for ax in self.axes:
            if ax.get_visible():
                canvas.blit(ax.bbox)


class GlobalPlot(ABC):
    def __init__(self, figsize=(8, 10), name="Global Plot"):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.name = name
        self.ax.grid(True)
        self.ax.set_aspect('equal') 
        
        self.background = None  # For blitting
        
        # Disable the 'l' shortcut for toggling log scale
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.fig.subplots_adjust(left=0, right=1, top=0.99, bottom=0.1)
        self.start, = self.ax.plot([], [], 'bo', markersize=14, label="Start", zorder=7)
        self.start_text = self.ax.text(1000, 1000, 'S', fontsize=12, color='white', zorder=9, ha='center', va='center')
        self.goal, = self.ax.plot([], [], 'go', markersize=14, label="Goal", zorder=7)
        self.goal_text = self.ax.text(1000, 1000 , 'G', fontsize=12, color='white', zorder=9, ha='center', va='center')
        self.vehicle_location, = self.ax.plot([], [], 'ro', markersize=14, label="Planner Location", zorder=8)
        self.vehicle_location_text = self.ax.text(0, 0, 'L', fontsize=12, color='white', zorder=9, ha='center', va='center')

        self.orientation_arrow = None  # For the vehicle orientation arrow
        for tick in self.ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(8)
        for tick in self.ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(8)

        # self.fig.legend(loc="upper right", fontsize=8, framealpha=0.3)
        
        self.map_min_x = None
        self.map_min_y = None
        self.map_max_x = None
        self.map_max_y = None
        self.view_width = None
        self.view_height = None
        self.map_plotted = False
        self._velocity_scale = "relative"

        # Hover-speed readout: a highlight marker on the nearest trajectory point
        # plus an annotation showing target speed in m/s and km/h. Shared by all
        # global plot types; each feeds its plotted trajectory via _set_hover_data.
        self._hover_xy: Optional[np.ndarray] = None
        self._hover_v: Optional[np.ndarray] = None
        self._hover_marker, = self.ax.plot([], [], 'o', color="white", markersize=8,
                                           markeredgecolor="black", zorder=10)
        self._hover_text = self.ax.annotate(
            "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
            fontsize=9, color="white", zorder=11, visible=False,
            bbox=dict(facecolor="#1d2021", alpha=0.8, pad=2, edgecolor="none",
                      boxstyle="round,pad=0.3"),
        )
        self._hover_visible = False


    def plot(self, exec:SyncExecuter, aspect_ratio=4.0, zoom=None, show_legend=True, follow_vehicle=True, show_plan_boundaries=True, velocity_scale="relative", delta:Optional[tuple[float,float]]=None):
        self._velocity_scale = velocity_scale
        if not self.map_plotted and exec.global_planner is not None:
            self.plot_map(exec.global_planner)

        self.plot_vehicle(exec.ego_state) 
        self.adjust_center_and_zoom(zoom, aspect_ratio, delta=delta)

        # if not show_legend:
            # self.ax.get_legend().remove() if self.ax.get_legend() else None

        # self.ax.set_aspect(aspect_ratio)
        self.fig.canvas.draw()

    def plot_vehicle(self, ego:EgoState):
        """Plot the vehicle location"""
        self.vehicle_x, self.vehicle_y = ego.x, ego.y
        self.vehicle_location.set_data([self.vehicle_x], [self.vehicle_y ])
        self.vehicle_location_text.set_position((self.vehicle_x, self.vehicle_y))

    def _set_hover_data(self, xs, ys, velocity):
        """Cache the plotted trajectory geometry + speed for hover lookups."""
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        n = min(len(xs), len(ys))
        if n < 1:
            self._hover_xy = None
            self._hover_v = None
            return
        self._hover_xy = np.column_stack([xs[:n], ys[:n]])
        v = np.asarray(velocity, dtype=float)
        if len(v) < n:
            v = np.pad(v, (0, n - len(v)), mode="edge") if len(v) else np.zeros(n)
        self._hover_v = v[:n]

    def show_speed_at(self, x: float, y: float) -> bool:
        """Highlight the nearest trajectory point and annotate its target speed.

        Returns True when a nearby trajectory point was found (and drawn), False
        otherwise (annotation hidden). Only redraws when the displayed state
        changes, to keep mouse-move handling cheap.
        """
        if self._hover_xy is None or self._hover_v is None or len(self._hover_xy) == 0:
            return self._hide_speed()

        d = np.hypot(self._hover_xy[:, 0] - x, self._hover_xy[:, 1] - y)
        idx = int(np.argmin(d))
        # Tolerance scales with the current view so it works at any zoom level.
        tol = (self.view_width or (self.map_max_x - self.map_min_x if self.map_max_x else 100)) * 0.03
        if d[idx] > max(tol, 1.0):
            return self._hide_speed()

        wx, wy = self._hover_xy[idx]
        v = float(self._hover_v[idx])
        self._hover_marker.set_data([wx], [wy])
        self._hover_text.xy = (wx, wy)
        self._hover_text.set_text(f"{v:.1f} m/s  ({v * 3.6:.0f} km/h)")
        self._hover_text.set_visible(True)
        self._hover_visible = True
        self.fig.canvas.draw_idle()
        return True

    def _hide_speed(self) -> bool:
        if self._hover_visible:
            self._hover_marker.set_data([], [])
            self._hover_text.set_visible(False)
            self._hover_visible = False
            self.fig.canvas.draw_idle()
        return False

    def clear_tmp_plots(self):
        """Clears temporary plots (e.g., cursor highlights)"""
        pass

    @abstractmethod
    def plot_map(self, exec:SyncExecuter):
        pass
    
    def adjust_center_and_zoom(self, zoom, aspect_ratio,center:Optional[tuple[float,float]]=None, delta:Optional[tuple[float,float]]=None):
        """Adjust the zoom level and aspect ratio of the plot"""
        if self.map_min_x is not None and self.map_min_y is not None and self.map_max_x is not None and self.map_max_y is not None:
            # Set view limits
            mi_x = self.map_min_x - zoom
            mi_y = self.map_min_y - zoom/aspect_ratio
            ma_x = self.map_max_x + zoom
            ma_y = self.map_max_y + zoom/aspect_ratio
            
            center_x, center_y = center if center else (self.vehicle_x, self.vehicle_y)
            if delta:
                center_x += delta[0]
                center_y += delta[1]


            if mi_x < ma_x and mi_y < ma_y:
                pad = 100
                if self.map_min_x - pad < center_x < self.map_max_x + pad and self.map_min_y - pad < center_y < self.map_max_y + pad \
                     and zoom < self.map_max_x - self.map_min_x and zoom/aspect_ratio < self.map_max_y - self.map_min_y:

                    self.ax.set_xlim(center_x - zoom, center_x + zoom)
                    self.ax.set_ylim(center_y - zoom/aspect_ratio, center_y + zoom/aspect_ratio)
                    self.view_width = zoom * 2
                    self.view_height = zoom / aspect_ratio * 2
                else:
                    map_width = self.map_max_x - self.map_min_x
                    map_height = self.map_max_y - self.map_min_y
                    
                    target_width = map_height * aspect_ratio  
                    target_height = map_width / aspect_ratio
                    
                    if target_width > map_width:
                        # Need to expand width
                        x_pad = (target_width - map_width) / 2
                        y_pad = 0
                    else:
                        # Need to expand height  
                        x_pad = 0
                        y_pad = (target_height - map_height) / 2
                    
                    border = max(map_width, map_height) * 0.05
                    self.ax.set_xlim(self.map_min_x - x_pad - border, self.map_max_x + x_pad + border)
                    self.ax.set_ylim(self.map_min_y - y_pad - border, self.map_max_y + y_pad + border)
                    self.view_width = map_width + x_pad * 2 + border * 2
                    self.view_height = map_height + y_pad * 2 + border * 2


    def set_start(self, x, y):
        """Set the start point"""
        self.start.set_data([x], [y])
        # self.ax.draw_artist(self.start)
        # self.fig.canvas.blit(self.ax.bbox)
        self.start_text.set_position((x, y))
        self.start_text.set_text("S")
        self.fig.canvas.draw()

    def set_goal(self, x, y):
        """Set the goal point"""
        self.goal.set_data([x], [y])
        self.goal_text.set_position((x, y))
        self.goal_text.set_text("G")
        # self.ax.draw_artist(self.goal)
        # self.fig.canvas.blit(self.ax.bbox)
        
        self.fig.canvas.draw()

    def show_vehicle_orientation(self, x, y, theta):
        """Show the vehicle orientation on the plot"""
        log.debug(f"Showing vehicle orientation at ({x}, {y}) with theta={theta}")

        if self.view_width is not None:
            length = min(self.view_width, self.view_height) * .15
        else:
            length = np.abs(self.map_max_x - self.map_min_x) * .2

        x2 = x + length * np.cos(theta)
        y2 = y + length * np.sin(theta)
        if self.orientation_arrow:
            self.orientation_arrow.remove()
        self.orientation_arrow = self.ax.annotate('', xy=(x2, y2), xytext=(x,y), arrowprops=dict(arrowstyle='->',
                                                     mutation_scale=20, color="red", lw=5), zorder=9)
        
    def clear_tmp_plots(self):
        self.orientation_arrow.remove() if self.orientation_arrow else None
        self.orientation_arrow = None
        self._hide_speed()


    def set_plot_theme(self, bg_color="white", fg_color="black"):
        """Set the plot theme colors"""
        # Apply background color with no transparency
        self.fig.set_facecolor(bg_color)
        self.ax.set_facecolor(bg_color)
        
        # Set axis, ticks, and label colors
        for spine in self.ax.spines.values():
            spine.set_edgecolor(fg_color)
        
        self.ax.tick_params(axis="both", colors=fg_color)
        self.ax.xaxis.label.set_color(fg_color)
        self.ax.yaxis.label.set_color(fg_color)
        
        # Set grid color with proper alpha for visibility
        self.ax.grid(False)
        # self.ax.set_title(label=self.name, color=fg_color, y=-0.05)
        
        # Apply redraw
        self.fig.canvas.draw()
        
        log.debug(f"Global plot theme set to {bg_color} background and {fg_color} foreground.")
        

    def reset(self):
        self.map_plotted = False

    def close(self) -> None:
        plt.close(self.fig)


class GlobalRacePlot(GlobalPlot):
    def __init__(self, figsize=(8, 10)):
        super().__init__(figsize, name = "Global Race Plot")
        # Create plot elements with empty data - they'll be updated later
        self.left_boundary, = self.ax.plot([], [], 'orange', linewidth=3, label="Left Boundary")
        self.right_boundary, = self.ax.plot([], [], 'tan', linewidth=3, label="Right Boundary")
        self.reference_trajectory = LineCollection([], cmap=_VELOCITY_CMAP, linewidths=5, zorder=4)
        self.ax.add_collection(self.reference_trajectory)

        # self.ax.legend()
        
        # Adjust layout to align with LocalPlot
        self.fig.subplots_adjust(left=0, right=1, top=0.99, bottom=0.1)

    def plot(self, exec: SyncExecuter, aspect_ratio=4.0, zoom=None, show_legend=True,
             follow_vehicle=True, show_plan_boundaries: bool = True,
             velocity_scale: str = "relative",
             delta: Optional[tuple[float, float]] = None):
        self._velocity_scale = velocity_scale
        if exec.global_planner is not None:
            self.plot_map(exec.global_planner, show_plan_boundaries=show_plan_boundaries)
        self.plot_vehicle(exec.ego_state)
        self.adjust_center_and_zoom(zoom, aspect_ratio, delta=delta)
        self.fig.canvas.draw()

    def clear_goal(self):
        """Clear the goal plot (no-op for race plot)."""
        self.goal.set_data([], [])
        self.goal_text.set_text("")
        self.fig.canvas.draw()

    def clear_road_path_plots(self):
        """Clear road path plots (no-op for race plot)."""
        pass

    def set_plot_theme(self, bg_color="white", fg_color="black"):
        super().set_plot_theme(bg_color, fg_color)

        # Use the same colors as LocalPlot, not black/white specific colors
        self.left_boundary.set_color("orange")
        self.right_boundary.set_color("tan")
        self.vehicle_location.set_color("red")
        
        
    def plot_map(self, global_planner: GlobalPlannerStrategy, show_plan_boundaries: bool = True):
        """Update the plot with current data"""

        log.debug("Plotting Race Global Plot")
        plan = global_planner.global_plan
        if show_plan_boundaries:
            self.left_boundary.set_data(plan.left_boundary_x, plan.left_boundary_y)
            self.right_boundary.set_data(plan.right_boundary_x, plan.right_boundary_y)
        else:
            self.left_boundary.set_data([], [])
            self.right_boundary.set_data([], [])
        traj = plan.trajectory
        _update_velocity_colored_line(
            self.reference_trajectory, traj.path_x, traj.path_y, traj.velocity,
            velocity_scale=self._velocity_scale,
        )
        # Cache raceline geometry + speed for the hover-speed readout.
        self._set_hover_data(traj.path_x, traj.path_y, traj.velocity)

        self.map_min_x = min(plan.left_boundary_x)
        self.map_max_x = max(plan.right_boundary_x)
        self.map_min_y = min(plan.left_boundary_y)
        self.map_max_y = max(plan.right_boundary_y)
        self.map_plotted = True


class GlobalHDMapPlot(GlobalPlot):
    def __init__(self, figsize=(10, 10), MAX_ROAD_PATH=20, MAX_SUCCS=10, MAX_PREDS=10):
        super().__init__(figsize, name="HD Map Road Network")
        
        # Gruvbox colors
        orange = "#d65d0e"
        light_orange = "#fe8019"
        light_aqua = "#8ec07c"
        aqua = "#689d6a"
        blue = "#076678"
        yellow = "yellow"
        red = 'red'
        purple='#b16286'
        txt_bg = '#1d2021'

        self.yellow = yellow; self.red = red; self.blue = blue

        self.closest_lane, *_ = self.ax.plot([], [], 'o-', color=yellow,  alpha=.2, label="Closest Lane", zorder=3)
        self.closest_road, *_ = self.ax.plot([], [], 'o-', color=red, alpha=.1,  label="Closest Road", zorder=3)
        self.closest_road_preds = []
        self.closest_road_succs = []
        for i in range(MAX_PREDS):
            p, *_ = self.ax.plot([], [], 'o-', color=orange,  alpha=.1, label="Closest Road Pred", zorder=3)
            self.closest_road_preds.append(p)
        for i in range(MAX_SUCCS):
            s, *_ = self.ax.plot([], [], 'o-', color=aqua, alpha=.1, label="Closest Road Succ", zorder=3)
            self.closest_road_succs.append(s)

        self.closest_lane_preds = []
        self.closest_lane_neighbors = []
        for i in range(MAX_PREDS):
            p, *_ = self.ax.plot([], [], 'o-', color=light_orange,  alpha=.1, label="Closest Lane Pred", zorder=3)
            self.closest_lane_preds.append(p)
        for i in range(MAX_SUCCS):
            s, *_ = self.ax.plot([], [], 'o-', color=light_aqua,  alpha=.1, label="Closest Lane Pred", zorder=3)
            self.closest_lane_neighbors.append(s)

        #################
        # Texts
        self.road_id_text = self.ax.text(0, 0, '', fontsize=12, color=red, zorder=4, ha='center', va='center',
                                              bbox=dict(facecolor=txt_bg, alpha=0.4, pad=1, edgecolor='none', boxstyle='round, pad=0.1'))
        self.road_succ_id_text = self.ax.text(0, 0, '', fontsize=10, color=aqua, alpha=0.8, zorder=4, ha='center', va='center')
        self.road_pred_id_text = self.ax.text(0, 0, '', fontsize=10, color=orange, alpha=0.8, zorder=4, ha='center', va='center')
        self.lane_id_text = self.ax.text(0, 0, '', fontsize=12, color=yellow, zorder=4, ha='center', va='center',
                                              bbox=dict(facecolor=txt_bg, alpha=0.4, pad=1, edgecolor='none', boxstyle='round, pad=0.1'))
        self.junct_id_text = self.ax.text(0, 0, '', fontsize=10, color=red, alpha=0.8, zorder=4, ha='center', va='center')
        
        self.lane_succ_id_text = self.ax.text(0, 0, '', fontsize=8, color=light_aqua, alpha=0.8, zorder=4, ha='center', va='center',
                                              bbox=dict(facecolor=txt_bg, alpha=0.4, pad=1, edgecolor='none', boxstyle='round, pad=0.1'))
        self.lane_pred_id_text = self.ax.text(0, 0, '', fontsize=8, color=orange, alpha=0.8, zorder=4, ha='center', va='center',
                                              bbox=dict(facecolor=txt_bg, alpha=0.4, pad=1, edgecolor='none', boxstyle='round, pad=0.1'))
        #################

        self.__tmp_road, *_ = self.ax.plot([], [], 'o-', color='blue', markersize=5, alpha=.1, label="Closest Road", zorder=3)

        self.lane_path_plots = []
        for i in range(MAX_ROAD_PATH):
           pl , *_ = self.ax.plot([], [], 'o-', color=blue, linewidth=2, alpha=0.5, label="Lane Path", zorder=2)
           self.lane_path_plots.append(pl)

        self.global_plan_path = LineCollection([], cmap=_VELOCITY_CMAP, linewidths=5, zorder=4)
        self.ax.add_collection(self.global_plan_path)

        self.road_arrow = None # for the road direction arrow
        self.lane_arrow = None # for the lane direction arrow
        self._map_lane_artists: list[Line2D] = []

    def reset(self):
        """Clear route overlays only; keep the static HD map layer."""
        self.clear_road_path_plots()

    def _clear_map_lane_artists(self) -> None:
        for ln in self._map_lane_artists:
            ln.remove()
        self._map_lane_artists.clear()

    def plot(self, exec: SyncExecuter, aspect_ratio=4.0, zoom=None, show_legend=True,
             follow_vehicle=True, show_plan_boundaries: bool = True,
             velocity_scale: str = "relative",
             delta: Optional[tuple[float, float]] = None):
        self._velocity_scale = velocity_scale
        if not self.map_plotted and exec.global_planner is not None:
            self.plot_map(exec.global_planner)
        self.plot_vehicle(exec.ego_state)
        self.adjust_center_and_zoom(zoom, aspect_ratio, delta=delta)
        if exec.global_planner is None:
            self.fig.canvas.draw()
            return
        plan = exec.global_planner.global_plan
        if len(plan.path) < 2 and exec.local_planner is not None:
            plan = exec.local_planner.global_plan
        if len(plan.path) >= 2:
            path = np.array(plan.path).T
            velocity = plan.velocity or plan.trajectory.velocity
            _update_velocity_colored_line(
                self.global_plan_path, path[0], path[1], velocity,
                velocity_scale=self._velocity_scale,
            )
            self._set_hover_data(path[0], path[1], velocity)
        self.fig.canvas.draw()

    def show_closest_road_and_lane(self,  x:int, y:int, map:HDMap):
        """Show the closest road and lane to the given coordinates"""
        l = map.find_nearest_lane(x,y)

        if l is not None:
            # log.debug(f"Lane ID: {l.id}, Road: {l.road_id} Lane Type: {l.type}")
            self.__clear_closest_road_and_lane()
            self.closest_lane.set_data(l.center_line[0], l.center_line[1])
            line_xs = l.center_line[0]
            line_ys = l.center_line[1]
            if int(l.id) < 0: 
                lx1, lx2 = line_xs[-2], line_xs[-1]
                ly1, ly2 = line_ys[-2], line_ys[-1]
            else:
                lx1, lx2 = line_xs[1], line_xs[0]
                ly1, ly2 = line_ys[1], line_ys[0]
            self.lane_arrow = self.ax.annotate('', xy=(lx2, ly2), xytext=(lx1, ly1), arrowprops=dict(arrowstyle='->',
                                                     mutation_scale=20, color=self.yellow, lw=2), zorder=5)
            # log.debug(f"lane p1: ({lx1}, {ly1}), p2: ({lx2}, {ly2})")
            center_idx = int(len(l.center_line[0]) / 2.5)
            self.lane_id_text.set_position((l.center_line[0][center_idx], l.center_line[1][center_idx]-5))
            self.lane_id_text.set_text(f"{l.id}")
            self.lane_pred_id_text.set_position((l.center_line[0][0], l.center_line[1][0]-5))
            self.lane_pred_id_text.set_text(f"P: {l.pred_id}")

            self.lane_succ_id_text.set_position((l.center_line[0][-1], l.center_line[1][-1]-5))
            self.lane_succ_id_text.set_text(f"S: {l.succ_id}")
            for line, s in zip(self.closest_lane_neighbors, l.neighbors):
                line.set_data(s.center_line[0], s.center_line[1])

            # if l.type == "driving":
            #     log.debug(f"Neighbors({l.uid}): {[n.uid for n in l.neighbors]}")
            #     for n in l.neighbors:
            #         log.debug(f" Can I go to {n.uid}? {map.can_laneA_access_laneB(l,n)}")


            r:HDMap.Road|None = map.road_by_id.get(l.road_id)
            if r is not None:
                self.closest_road.set_data(r.center_line[0], r.center_line[1])
                x1, x2 = r.center_line[0][-2], r.center_line[0][-1]
                y1, y2 = r.center_line[1][-2], r.center_line[1][-1]
                self.road_arrow = self.ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->',
                                                     mutation_scale=20, color=self.red, lw=2), zorder=5)

                center_idx = int(len(r.center_line[0]) / 2)
                self.road_id_text.set_position((r.center_line[0][center_idx], r.center_line[1][center_idx]))
                self.road_id_text.set_text(r.id)
                self.junct_id_text.set_position((r.center_line[0][center_idx], r.center_line[1][center_idx]-5))
                self.junct_id_text.set_text(f"{r.junction_id}+") if r.junction_id != "-1" else self.junct_id_text.set_text("")
                self.road_pred_id_text.set_position((r.center_line[0][0]-3, r.center_line[1][0]))
                p_txt = f"P: {r.pred_id}" if r.pred_type == "road" else f"P: {r.pred_id}+"
                self.road_pred_id_text.set_text(p_txt)
                self.road_succ_id_text.set_position((r.center_line[0][-1]+3, r.center_line[1][-1]))
                s_txt = f"S: {r.succ_id}" if r.succ_type == "road" else f"S: {r.succ_id}+"
                self.road_succ_id_text.set_text(s_txt)
                for i,s in enumerate(r.successors):
                    self.closest_road_succs[i].set_data(s.center_line[0], s.center_line[1])
                for i,p in enumerate(r.predecessors):
                    self.closest_road_preds[i].set_data(p.center_line[0], p.center_line[1])
                # log.debug(f"getting connecting roads: {[p for p in map._get_connecting_roads_from_junction(map.root, r.road_element, r.pred_id )]}")
                # log.debug(f"Road preds: {[p.id for p in r.predecessors]}, succs: {[s.id for s in r.successors]}")
                # log.debug(f"Road ID: {r.id}, lane sections: {[l[0].id for s,l in r.lane_sections.items()]}")
        
        
        self.fig.canvas.draw()

    def plot_global_plan(self, global_plan: GlobalPlan):
        """Plot the road path"""
        try:
            log.debug("Plotting Road Path: length = %d", len(global_plan.path))
            self.clear_road_path_plots()
            path = np.array(global_plan.path).T
            velocity = global_plan.velocity or global_plan.trajectory.velocity
            _update_velocity_colored_line(
                self.global_plan_path, path[0], path[1], velocity,
                velocity_scale=self._velocity_scale,
            )
            self._set_hover_data(path[0], path[1], velocity)
            self.fig.canvas.draw()
        except Exception as e:
            log.error(f"Error plotting global plan: {e}")
            _update_velocity_colored_line(self.global_plan_path, [], [], [])
            self._set_hover_data([], [], [])


    def clear_road_path_plots(self):
        """Clear the road path plots"""
        for i in range(len(self.lane_path_plots)):
            self.lane_path_plots[i].set_data([], [])
        _update_velocity_colored_line(self.global_plan_path, [], [], [])
        self._set_hover_data([], [], [])
        self.__clear_closest_road_and_lane()

    
    def clear_tmp_plots(self):
        super().clear_tmp_plots()
        self.__clear_closest_road_and_lane()

    def __clear_closest_road_and_lane(self):
        """Clear the closest road and lane plots"""
        self.closest_road.set_data([], [])
        self.closest_lane.set_data([], [])
        self.road_id_text.set_text("")
        self.junct_id_text.set_text("")
        self.road_pred_id_text.set_text("")
        self.road_succ_id_text.set_text("")
        self.lane_id_text.set_text("")
        self.lane_pred_id_text.set_text("")
        self.lane_succ_id_text.set_text("")
        
        for pr in self.closest_road_preds:
            pr.set_data([],[])
        for sr in self.closest_road_succs:
            sr.set_data([],[])
        for pl in self.closest_lane_preds:
            pl.set_data([],[])
        for sl in self.closest_lane_neighbors:
            sl.set_data([],[])

        if self.road_arrow is not None:
            self.road_arrow.remove()
            self.road_arrow = None
        if self.lane_arrow is not None:
            self.lane_arrow.remove()
            self.lane_arrow = None

        self.fig.canvas.draw()

    def clear_goal(self):
        """Clear the goal plot"""
        self.goal.set_data([], [])
        self.goal_text.set_text("")
        self.fig.canvas.draw()
        
    def plot_map(self, global_planner:GlobalPlannerStrategy, show_road_points=False, show_lane_points=False):
        """Implement the abstract method from GlobalPlot"""
        
        if not hasattr(global_planner, "hdmap"):
            log.warning("HDMap not found in the global planner.")
            return

        global_planner = cast(HDMapGlobalPlanner,global_planner)
        hdmap = global_planner.hdmap

        #TODO: remove this later if not needed
        if show_road_points:
            map = global_planner.hdmap
            p = np.array(list(map.point_to_road.keys())).T
            log.debug(f"road points length {len(p)}")
            self.ax.scatter(p[0], p[1], color='blue', s=10, alpha=0.5)
        if show_lane_points:
            map = global_planner.hdmap
            p = np.array(list(map.point_to_lane.keys())).T
            log.debug(f"lane points length {len(p)}")
            self.ax.scatter(p[0], p[1], color='blue', s=10, alpha=0.5)


        all_x_coords = []
        all_y_coords = []

        self._clear_map_lane_artists()

        # for r in hdmap.roads:
            # self.ax.plot(r.center_line[0], r.center_line[1], color='red', linewidth=2, alpha=0.5)

        for l in hdmap.lanes:
            if l.type == "driving":
                color = "#427b58" 
                alpha = 0.7
            elif l.type == "shoulder":
                color = "#af3a03"
                alpha = 0.4
            else:
                color = "gray"
                alpha = 0.3
            
            ln, = self.ax.plot(l.center_line[0], l.center_line[1], color=color, linewidth=2, alpha=alpha)
            self._map_lane_artists.append(ln)
            all_x_coords.extend(l.center_line[0])
            all_y_coords.extend(l.center_line[1])

        self.map_min_x = min(all_x_coords)  
        self.map_min_y = min(all_y_coords)
        self.map_max_x = max(all_x_coords)
        self.map_max_y = max(all_y_coords)
        self.map_plotted = True
            

class LocalPlot:
    _FRENET_EGO_X_FRAC = 0.25
    # Prediction polyline colours: agents ahead of the ego vs. agents behind it.
    PREDICTION_AHEAD_COLOR = "darkorange"
    PREDICTION_BEHIND_COLOR = "mediumpurple"
    # Gaussian contours drawn along a forecast. Opacity of a ring equals its percentile.
    PREDICTION_PERCENTILES = (0.5, 0.95)
    _ELLIPSE_STRIDE_S = 0.5
    _ELLIPSE_SAMPLES = 24
    def __init__(self, max_plan_length=5, max_agent_count=12, controller: Optional[ControlStrategy] = None, coord: str = "xy"):
        self.MAX_PLAN_LENGTH = max_plan_length
        self.MAX_AGENT_COUNT = max_agent_count
        self.controller = controller
        self.coord = coord

        self.fig, self.ax = plt.subplots()
        # Disable the 'l' shortcut for toggling log scale
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.ax.set_aspect("equal")
        self.fig.subplots_adjust(left=0, right=1, top=0.99, bottom=0.1)

        self.lattice_graph_plots = []
        self.lattice_graph_endpoints = []
        self._lattice_legend_added = False
        self.local_plan_plots = []
        self.view_width = None
        self.view_height = None
        self.orientation_arrow = None

        self._window: Optional[tuple[float, float, float, float]] = None
        self._gp_cache = None
        self._tb_cache = None
        self._needs_full_draw = False

        frenet = coord == "frenet"
        bound_lw = 1 if frenet else 2
        ref_lw = 1 if frenet else 2
        ref_alpha = 0.5 if frenet else 1.0
        left_label = "Left Boundary (Ref)" if frenet else "Left Boundary"
        right_label = "Right Boundary (Ref)" if frenet else "Right Boundary"
        ref_label = "Global Trajectory" if frenet else "Reference Trajectory"
        for i in range(self.MAX_PLAN_LENGTH):
            (local_plan,) = self.ax.plot([], [], "r-", label=f"Local Plan {i}", alpha=0.6 / (i + 1), linewidth=8)
            self.local_plan_plots.append(local_plan)

        (self.left_boundry,) = self.ax.plot([], [], color="orange", label=left_label, linewidth=bound_lw)
        (self.right_boundry,) = self.ax.plot([], [], color="tan", label=right_label, linewidth=bound_lw)
        (self.reference_trajectory,) = self.ax.plot(
            [], [], color="gray", label=ref_label, linewidth=ref_lw, alpha=ref_alpha,
        )
        (self.last_locs,) = self.ax.plot([], [], "g-", label="Last 100 Locations", linewidth=2)
        (self.planner_loc,) = self.ax.plot([], [], "ro", markersize=10, label="Planner Location")
        (self.g_wp_current,) = self.ax.plot(
            [], [], "g", markersize=13, label="G WP: Curent", marker="o", fillstyle="none"
        )
        (self.g_wp_next,) = self.ax.plot([], [], "gx", markersize=13, label="G WP: Next")
        (self.current_wp_plot,) = self.ax.plot(
            [], [], "bo", markersize=15, label="L WP: Current", fillstyle="none"
        )
        (self.next_wp_plot,) = self.ax.plot([], [], "bx", markersize=15, label="L WP: Next", fillstyle="none")
        (self.car_heading_plot,) = self.ax.plot(
            [], [], "-", color="darkslategray", label=None if frenet else "Car Heading",
        )
        (self.car_location_plot,) = self.ax.plot(
            [], [], "ko", markersize=7, label=None if frenet else "Car Location",
        )

        self._ruler_line, = self.ax.plot([], [], color="0.7", lw=1, alpha=0.8, zorder=20)
        self._ruler_text = self.ax.annotate(
            "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
            fontsize=9, color="0.85", zorder=21, visible=False,
            bbox=dict(facecolor="#1d2021", alpha=0.75, pad=2, edgecolor="none",
                      boxstyle="round,pad=0.3"),
        )
        self._ruler_visible = False
        self._ruler_last = None

        self.ego_vehicle = Polygon(np.empty((0, 2)), closed=True, edgecolor="r", facecolor="azure", alpha=0.7)
        self.ax.add_patch(self.ego_vehicle)

        self.pm_plots = []
        for _ in range(self.MAX_AGENT_COUNT):
            agent_vehicle = Polygon(
                np.empty((0, 2)), closed=True, edgecolor="darkblue", facecolor="azure", alpha=0.6
            )
            self.ax.add_patch(agent_vehicle)
            self.pm_plots.append(agent_vehicle)

        self.lidar_scatter = self.ax.scatter([], [], s=6, c="lime", alpha=0.9, zorder=6, label="LiDAR")
        self.cluster_scatter = self.ax.scatter([], [], s=8, c="yellow", alpha=0.9, zorder=7, label="Clusters")
        self.track_boundary_collection = LineCollection(
            [], linewidths=1, colors="cyan", alpha=0.5, zorder=2, label="Track Boundary"
        )
        self.ax.add_collection(self.track_boundary_collection)
        self.prediction_lines = LineCollection([], linewidths=1.5, linestyles="solid", zorder=3)
        self.ax.add_collection(self.prediction_lines)
        self.prediction_regions = PolyCollection(
            [], facecolors="none", edgecolors="none", linewidths=1.0, zorder=2,
        )
        self.ax.add_collection(self.prediction_regions)
        self.occupancy_flow_cells = PolyCollection(
            [], facecolors="none", edgecolors="none", linewidths=0.0, zorder=0.5,
        )
        self.ax.add_collection(self.occupancy_flow_cells)

        self.legend_ax = self.fig.add_axes([0.0, -0.013, 1, 0.1])
        self.legend_ax.legend(
            *self.ax.get_legend_handles_labels(), loc="center", ncol=7, borderaxespad=0.0, fontsize=7, framealpha=0.3)
        for tick in self.ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(8)
        for tick in self.ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(8)
        self.legend_ax.axis("off")
        self._init_blit()

    def _init_blit(self) -> None:
        """Register the per-frame (dynamic) artists for blitting.

        Static artists (boundaries, reference trajectory, track boundary, the
        legend and axes frame) are intentionally left non-animated so they end
        up in the cached background and are only re-rendered on a full draw.
        """
        self.blit_manager = BlitManager(self.fig, [self.ax])
        animated = [
            self.ego_vehicle,
            self.car_heading_plot, self.car_location_plot,
            self.last_locs, self.planner_loc,
            self.g_wp_current, self.g_wp_next,
            self.current_wp_plot, self.next_wp_plot,
            self.lidar_scatter, self.cluster_scatter,
            self._ruler_line, self._ruler_text,
        ]
        animated += self.local_plan_plots
        animated += self.pm_plots
        animated += [self.prediction_lines, self.prediction_regions, self.occupancy_flow_cells]
        for art in animated:
            self.blit_manager.add_artist(art)

    def plot(
        self,
        exec: SyncExecuter,
        aspect_ratio=4.0,
        zoom=30,
        show_legend=True,
        plot_last_pts=True,
        plot_global_plan=True,
        plot_local_plan=True,
        plot_local_lattice=True,
        plot_state=True,
        plot_perception_model=True,
        num_plot_last_pts=100,
        follow_planner=False,
        plot_occupancy_map=False,
        plot_predictions=True,
        plot_lidar=False,
        lidar_data=None,
        plot_clusters=True,
        plot_ground_truth=True,
        plot_race_boundary=True,
    ):
        self._needs_full_draw = False
        self.legend_ax.set_visible(show_legend)
        self.fig.subplots_adjust(left=0, right=1, top=0.99, bottom=0.1)

        planner = exec.local_planner
        if self.coord == "frenet":
            if follow_planner and planner is not None:
                center = planner.location_sd
            elif planner is not None:
                center = planner.global_trajectory.convert_xy_to_sd(exec.ego_state.x, exec.ego_state.y)
            else:
                center = (0.0, 0.0)
            if zoom is not None:
                view_w = zoom * 2
                ego_x_frac = self._FRENET_EGO_X_FRAC
                self.ax.set_xlim(center[0] - ego_x_frac * view_w, center[0] + (1 - ego_x_frac) * view_w)
                self.ax.set_ylim(-zoom / aspect_ratio, zoom / aspect_ratio)
                self.view_width = view_w
                self.view_height = zoom / aspect_ratio * 2
            self._window = None
        else:
            center = planner.location_xy if follow_planner and planner is not None else (exec.ego_state.x, exec.ego_state.y)
            if zoom is not None:
                _x0, _x1 = center[0] - zoom, center[0] + zoom
                _y0 = center[1] - zoom / aspect_ratio
                _y1 = center[1] + zoom / aspect_ratio
                self.ax.set_xlim(_x0, _x1)
                self.ax.set_ylim(_y0, _y1)
                self.view_width = zoom * 2
                self.view_height = zoom / aspect_ratio * 2
                self._window = (_x0, _x1, _y0, _y1)
            else:
                self._window = None

        if plot_last_pts and num_plot_last_pts > 0 and planner is not None:
            if self.coord == "frenet":
                self.last_locs.set_data(planner.traversed_s[-num_plot_last_pts:], planner.traversed_d[-num_plot_last_pts:])
                self.planner_loc.set_data([planner.location_sd[0]], [planner.location_sd[1]])
            else:
                self.last_locs.set_data(planner.traversed_x[-num_plot_last_pts:], planner.traversed_y[-num_plot_last_pts:])
                self.planner_loc.set_data([planner.location_xy[0]], [planner.location_xy[1]])
        else:
            self.last_locs.set_data([], [])
            self.planner_loc.set_data([], [])

        self.update_pm_occupancy_map_plots(exec.pm, plot_occupancy_map)
        show_race_boundaries = plot_race_boundary and not isinstance(exec.global_planner, HDMapGlobalPlanner)
        fallback_plan = getattr(exec.global_planner, "global_plan", None) if exec.global_planner else None
        local_tj = planner.global_trajectory if planner else None
        self.update_track_boundary_plot(
            exec.world,
            show_plot=show_race_boundaries,
            global_trajectory=local_tj,
            fallback_plan=fallback_plan,
        )
        if planner is None:
            return
        self.update_global_plan_plots(planner, plot_global_plan)
        self.update_lattice_graph_plots(planner, plot_local_lattice)
        self.update_local_plan_plots(planner, plot_local_plan)
        self.update_state_plots(exec.ego_state, planner.global_trajectory, plot_state)
        world_pm = (
            exec.world.get_ground_truth_perception_model()
            if plot_ground_truth and hasattr(exec.world, "get_ground_truth_perception_model")
            else None
        )
        self.update_perception_model_plots(
            exec.pm,
            planner.global_trajectory,
            plot_perception_model,
            plot_predictions,
            world_pm=world_pm,
        )
        self.update_lidar_plot(lidar_data, plot_lidar, planner.global_trajectory)
        self.update_cluster_plot(getattr(exec.pm, "detection_clusters", None), plot_clusters, planner.global_trajectory)
        self.update_pm_occupancy_flow_plots(
            exec.pm,
            plot_predictions,
            global_trajectory=planner.global_trajectory,
        )

    def update_track_boundary_plot(
        self,
        world_bridge,
        show_plot=True,
        global_trajectory=None,
        fallback_plan=None,
    ):
        if not show_plot or not hasattr(world_bridge, 'boundary_segments'):
            self.track_boundary_collection.set_segments([])
            self._tb_cache = None
            return
        segs = np.asarray(world_bridge.boundary_segments)  # (M, 2, 2) world-frame
        if len(segs) == 0 and fallback_plan is not None:
            segs = np.asarray(boundary_segments_from_global_plan(fallback_plan))

        if self.coord != "frenet":
            win = self._window
            if len(segs) and win is not None:
                mid = segs.mean(axis=1)
                x0, x1, y0, y1 = win
                mx = (x1 - x0) * 0.15
                my = (y1 - y0) * 0.15
                m = (mid[:, 0] >= x0 - mx) & (mid[:, 0] <= x1 + mx) & (mid[:, 1] >= y0 - my) & (mid[:, 1] <= y1 + my)
                segs = segs[m]
            self.track_boundary_collection.set_segments(segs if len(segs) else [])
            return

        if global_trajectory is None or len(segs) == 0:
            self.track_boundary_collection.set_segments([])
            self._tb_cache = None
            return

        # Frenet conversion is expensive and view-independent; cache it and
        # only rebuild when the trajectory or the segment set actually changes.
        key = (
            id(global_trajectory),
            segs.shape,
            float(segs.flat[0]), float(segs.flat[-1]),
        )
        if self._tb_cache is None or self._tb_cache[0] != key:
            pts = segs.reshape(-1, 2)
            sd = global_trajectory.convert_xy_path_to_sd_path_np(pts)
            sd_segs = sd.reshape(-1, 2, 2)
            s_diff = np.abs(sd_segs[:, 1, 0] - sd_segs[:, 0, 0])
            valid = s_diff < (np.median(s_diff) * 20 + 50.0)
            self._tb_cache = (key, sd_segs[valid] if valid.any() else np.empty((0, 2, 2)))
            self.track_boundary_collection.set_segments(
                self._tb_cache[1] if len(self._tb_cache[1]) else []
            )
            self._needs_full_draw = True

    def redraw_plots(self):
        self.blit_manager.full_draw()

    def update_global_plan_plots(self, pl: LocalPlanningStrategy, show_plot=True):
        if not show_plot:
            self.g_wp_current.set_data([], [])
            self.g_wp_next.set_data([], [])
            self.left_boundry.set_data([], [])
            self.right_boundry.set_data([], [])
            self.reference_trajectory.set_data([], [])
            self._gp_cache = None
            return

        gt = pl.global_trajectory
        key = (id(pl.global_plan), id(gt), len(gt.path_s))
        if self._gp_cache is None or self._gp_cache[0] != key:
            self._gp_cache = (key, self._compute_global_plan_geometry(pl))
            self._needs_full_draw = True
        geom = self._gp_cache[1]

        if self.coord == "frenet":
            if geom["refreshed"]:
                if geom["has_boundary_d"]:
                    self.left_boundry.set_data(geom["s2"], geom["ld2"])
                    self.right_boundry.set_data(geom["s2"], geom["rd2"])
                else:
                    self.left_boundry.set_data([], [])
                    self.right_boundry.set_data([], [])
                self.reference_trajectory.set_data(geom["s2"], geom["ref2"])
                geom["refreshed"] = False
            if gt.next_wp is not None:
                self.g_wp_current.set_data([gt.path_s[gt.current_wp]], [gt.path_d[gt.current_wp]])
                self.g_wp_next.set_data([gt.path_s[gt.next_wp]], [gt.path_d[gt.next_wp]])
            return

        win = self._window
        if geom["has_boundary"]:
            self.left_boundry.set_data(*_decimate_polyline(geom["lbx"], geom["lby"], win))
            self.right_boundry.set_data(*_decimate_polyline(geom["rbx"], geom["rby"], win))
        else:
            self.left_boundry.set_data([], [])
            self.right_boundry.set_data([], [])
        self.reference_trajectory.set_data(*_decimate_polyline(geom["ref_x"], geom["ref_y"], win))
        if gt.next_wp is not None:
            self.g_wp_current.set_data([gt.path_x[gt.current_wp]], [gt.path_y[gt.current_wp]])
            self.g_wp_next.set_data([gt.path_x[gt.next_wp]], [gt.path_y[gt.next_wp]])

    def _compute_global_plan_geometry(self, pl: LocalPlanningStrategy) -> dict:
        """Precompute view-independent global-plan geometry once per plan.

        Returns the whole-lap XY arrays (sliced to the window each frame) plus
        the static Frenet arrays. ``refreshed`` signals that the Frenet artists
        still need the new data.
        """
        gp = pl.global_plan
        gt = pl.global_trajectory
        has_boundary = bool(gp.left_boundary_x and gp.right_boundary_x)
        geom = {
            "refreshed": True,
            "has_boundary": has_boundary,
            "lbx": np.asarray(gp.left_boundary_x, dtype=float) if has_boundary else np.empty(0),
            "lby": np.asarray(gp.left_boundary_y, dtype=float) if has_boundary else np.empty(0),
            "rbx": np.asarray(gp.right_boundary_x, dtype=float) if has_boundary else np.empty(0),
            "rby": np.asarray(gp.right_boundary_y, dtype=float) if has_boundary else np.empty(0),
            "ref_x": np.asarray(gt.path_x, dtype=float),
            "ref_y": np.asarray(gt.path_y, dtype=float),
        }
        _s = np.asarray(gt.path_s, dtype=float)
        _gaps = np.where(np.diff(_s) < 0)[0] + 1
        has_boundary_d = bool(gp.left_boundary_d and gp.right_boundary_d)
        geom["has_boundary_d"] = has_boundary_d
        if len(_gaps):
            geom["s2"] = np.insert(_s, _gaps, np.nan)
            geom["ref2"] = np.insert(np.asarray(gt.path_d, dtype=float), _gaps, np.nan)
            if has_boundary_d:
                geom["ld2"] = np.insert(np.asarray(gp.left_boundary_d, dtype=float), _gaps, np.nan)
                geom["rd2"] = np.insert(np.asarray(gp.right_boundary_d, dtype=float), _gaps, np.nan)
        else:
            geom["s2"] = _s
            geom["ref2"] = np.asarray(gt.path_d, dtype=float)
            if has_boundary_d:
                geom["ld2"] = np.asarray(gp.left_boundary_d, dtype=float)
                geom["rd2"] = np.asarray(gp.right_boundary_d, dtype=float)
        return geom

    def update_lattice_graph_plots(self, pl: LocalPlanningStrategy, show_plot=True):
        if not show_plot or not hasattr(pl, "lattice") or len(pl.lattice.edges) == 0:
            for line in self.lattice_graph_plots + self.lattice_graph_endpoints:
                line.set_data([], [])
            return

        curvature_check = getattr(pl, "_is_curvature_feasible", None)
        edge_index = 0
        for edge in pl.lattice.edges:
            if edge_index >= len(self.lattice_graph_plots):
                (line,) = self.ax.plot([], [], "--", color="#8ec07c", alpha=0.6)
                (endpoint,) = self.ax.plot([], [], "bo", alpha=0.6)
                self.lattice_graph_plots.append(line)
                self.lattice_graph_endpoints.append(endpoint)
                self.blit_manager.add_artist(line)
                self.blit_manager.add_artist(endpoint)

            traj = edge.local_trajectory
            if self.coord == "frenet":
                xs, ys = traj.path_s_from_parent, traj.path_d_from_parent
            else:
                xs, ys = traj.path_x, traj.path_y
            self.lattice_graph_plots[edge_index].set_data(xs, ys)
            if edge.collision:
                color = "firebrick"
            elif edge.boundary_violation or (
                curvature_check is not None and not curvature_check(edge)
            ):
                color = "royalblue"
            else:
                color = "#8ec07c"
            self.lattice_graph_plots[edge_index].set_color(color)
            self.lattice_graph_endpoints[edge_index].set_color(color)
            self.lattice_graph_endpoints[edge_index].set_data([xs[-1]], [ys[-1]])
            edge_index += 1

        if not self._lattice_legend_added:
            legend_handles = [
                Line2D([0], [0], color="firebrick", lw=2, label="Collision"),
                Line2D([0], [0], color="royalblue", lw=2, label="Kinematic violation"),
                Line2D([0], [0], color="#8ec07c", lw=2, label="Feasible"),
            ]
            self.ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.5)
            self._lattice_legend_added = True

        for i in range(edge_index, len(self.lattice_graph_plots)):
            self.lattice_graph_plots[i].set_data([], [])
            self.lattice_graph_endpoints[i].set_data([], [])

    def update_local_plan_plots(self, pl: LocalPlanningStrategy, show_plot=True):
        if not hasattr(pl, "selected_local_plan"):
            self.current_wp_plot.set_data([], [])
            self.next_wp_plot.set_data([], [])
            local_plan = pl.get_local_plan() if show_plot else None
            tj = local_plan.as_trajectory() if local_plan is not None else None
            if tj is None:
                self.__clear_local_plan_plots()
            else:
                self._set_local_plan_line(pl, tj.path_x, tj.path_y, 0)
                self.__clear_local_plan_plots(index=1)
            return
        if not show_plot or pl.selected_local_plan is None:
            self.__clear_local_plan_plots()
            self.current_wp_plot.set_data([], [])
            self.next_wp_plot.set_data([], [])
        elif pl.selected_local_plan is not None:
            local_plan = pl.get_local_plan()
            tj = local_plan.as_trajectory()
            x, y = tj.get_current_xy()
            x_n, y_n = tj.get_xy_by_waypoint(tj.next_wp)
            if self.coord == "frenet":
                s, d = pl.global_trajectory.convert_xy_to_sd(x, y)
                s_n, d_n = pl.global_trajectory.convert_xy_to_sd(x_n, y_n)
                self.current_wp_plot.set_data([s], [d])
                self.next_wp_plot.set_data([s_n], [d_n])
            else:
                self.current_wp_plot.set_data([x], [y])
                self.next_wp_plot.set_data([x_n], [y_n])
            self._set_local_plan_line(pl, tj.path_x, tj.path_y, 0)
            self.__clear_local_plan_plots(index=1)

    def __update_local_plan_plots(self, v: Edge, index: int = 0, horizon: int = None):
        if horizon is None:
            horizon = self.MAX_PLAN_LENGTH - 1
        if v is not None:
            traj = v.local_trajectory
            if self.coord == "frenet":
                self.local_plan_plots[index].set_data(traj.path_s_from_parent, traj.path_d_from_parent)
            else:
                self.local_plan_plots[index].set_data(traj.path_x, traj.path_y)
            self.__update_local_plan_plots(v.selected_next_local_plan, index + 1, horizon)
        elif index < horizon - 1:
            self.__clear_local_plan_plots(index=index)

    def _set_local_plan_line(self, pl: LocalPlanningStrategy, path_x, path_y, index: int) -> None:
        if self.coord == "frenet":
            s_plot, d_plot = pl.global_trajectory.convert_xy_path_to_sd_path(list(zip(path_x, path_y)))
            self.local_plan_plots[index].set_data(list(s_plot), list(d_plot))
        else:
            self.local_plan_plots[index].set_data(path_x, path_y)

    def __clear_local_plan_plots(self, index=0):
        for i in range(index, self.MAX_PLAN_LENGTH):
            self.local_plan_plots[i].set_data([], [])

    def update_state_plots(self, state: EgoState, global_trajectory: TrajectoryTracker, show_plot=True):
        if not show_plot:
            self.car_heading_plot.set_data([], [])
            self.car_location_plot.set_data([], [])
            self.ego_vehicle.set_xy(np.empty((0, 2)))
            return

        if self.coord == "frenet":
            self.car_heading_plot.set_data([], [])
            self.car_location_plot.set_data([], [])
            sd_corners = global_trajectory.convert_xy_path_to_sd_path_np(state.get_bb_corners())
            if np.abs(sd_corners[0][0] - sd_corners[1][0]) < 10:
                self.ego_vehicle.set_xy(np.array(sd_corners))
            else:
                self.ego_vehicle.set_xy(np.empty((0, 2)))
            return

        car_L_f = self.controller.ego_distance_front_axle if self.controller is not None else 2.5
        car_L_r = state.length - car_L_f
        car_x_front = state.x + car_L_f * np.cos(state.theta)
        car_y_front = state.y + car_L_f * np.sin(state.theta)
        car_x_rear = state.x - car_L_r * np.cos(state.theta)
        car_y_rear = state.y - car_L_r * np.sin(state.theta)
        self.car_heading_plot.set_data([car_x_front, car_x_rear], [car_y_front, car_y_rear])
        self.car_location_plot.set_data([state.x], [state.y])
        self.ego_vehicle.set_xy(state.get_bb_corners())

    def update_perception_model_plots(
        self,
        exec_pm: PerceptionModel,
        global_trajectory: TrajectoryTracker,
        show_plot=True,
        show_prediction=False,
        world_pm: Optional[PerceptionModel] = None,
    ):
        pm_agents = world_pm if world_pm is not None else exec_pm
        if not show_plot or len(pm_agents.agent_vehicles) == 0:
            for patch in self.pm_plots:
                patch.set_xy(np.empty((0, 2)))
            self._clear_prediction_drawings()
            return

        n = min(len(pm_agents.agent_vehicles), self.MAX_AGENT_COUNT)
        if len(pm_agents.agent_vehicles) > self.MAX_AGENT_COUNT:
            log.warning(f"Exceeded maximum number of agents: {self.MAX_AGENT_COUNT}")
        agents = pm_agents.agent_vehicles[:n]
        all_corners_xy = np.vstack([agent.get_bb_corners() for agent in agents])
        if self.coord == "frenet":
            all_corners = global_trajectory.convert_xy_path_to_sd_path_np(all_corners_xy)
        else:
            all_corners = all_corners_xy
        for i in range(n):
            self.pm_plots[i].set_xy(all_corners[i * 4:(i + 1) * 4])
        for j in range(n, self.MAX_AGENT_COUNT):
            self.pm_plots[j].set_xy(np.empty((0, 2)))

        # Prediction trajectories (always from exec_pm; world_pm has no pipeline outputs).
        # Each mode is one polyline starting at the agent. Opacity is the mode weight.
        # GP and GMM also draw a set of percentile ellipses; a lower probability is fainter.
        pred = exec_pm.prediction if show_prediction else None
        segments: list[np.ndarray] = []
        colors: list[tuple] = []
        regions: list[np.ndarray] = []
        region_colors: list[tuple] = []
        if isinstance(pred, (SingleTrajectory, GP, MultiTrajectory, GMM)):
            ego_hdg = np.array([np.cos(exec_pm.ego_vehicle.theta), np.sin(exec_pm.ego_vehicle.theta)])
            for agent in agents:
                if isinstance(pred, SingleTrajectory):
                    path = pred.trajectories.get(agent.agent_id)
                    modes = None if path is None else path.reshape(1, -1, 2)
                    mode_weights = np.array([1.0])
                elif isinstance(pred, GP):
                    path = pred.means.get(agent.agent_id)
                    modes = None if path is None else path.reshape(1, -1, 2)
                    mode_weights = np.array([1.0])
                else:
                    modes = pred.trajectories.get(agent.agent_id)
                    mode_weights = pred.weights.get(agent.agent_id)
                    if modes is not None and (mode_weights is None or len(mode_weights) != len(modes)):
                        mode_weights = np.ones(len(modes))
                if modes is None or len(modes) == 0:
                    continue
                to_agent = np.array([agent.x - exec_pm.ego_vehicle.x, agent.y - exec_pm.ego_vehicle.y])
                color = self.PREDICTION_BEHIND_COLOR if float(np.dot(ego_hdg, to_agent)) < 0.0 \
                    else self.PREDICTION_AHEAD_COLOR
                pose = np.array([agent.x, agent.y])
                for mode_index, (mode, weight) in enumerate(zip(modes, mode_weights)):
                    path_xy = np.vstack((pose, mode))
                    if self.coord == "frenet":
                        segments.append(global_trajectory.convert_xy_path_to_sd_path_np(path_xy))
                    else:
                        segments.append(path_xy)
                    colors.append(to_rgba(color, alpha=float(np.clip(weight, 0.0, 1.0))))
                    if isinstance(pred, (GP, GMM)):
                        self._append_prediction_ellipses(
                            pred, agent.agent_id, mode_index, mode, float(weight), color,
                            global_trajectory, regions, region_colors,
                        )
        self.prediction_lines.set_segments(segments)
        self.prediction_lines.set_colors(colors)
        self._set_prediction_regions(regions, region_colors)

    def _clear_prediction_drawings(self) -> None:
        self.prediction_lines.set_segments([])
        self._set_prediction_regions([], [])

    def _set_prediction_regions(self, verts: list[np.ndarray], colors: list[tuple]) -> None:
        self.prediction_regions.set_verts(verts)
        self.prediction_regions.set_alpha(None)
        self.prediction_regions.set_facecolors("none")
        self.prediction_regions.set_edgecolors(colors if colors else "none")

    def _append_prediction_ellipses(
        self,
        pred: GP | GMM,
        agent_id: int,
        mode_index: int,
        centers: np.ndarray,
        weight: float,
        color: str,
        global_trajectory: TrajectoryTracker,
        regions: list[np.ndarray],
        region_colors: list[tuple],
    ) -> None:
        """One pair of percentile ellipses every half-second, including the last step."""
        n_steps = len(centers)
        for step in _ellipse_step_indices(n_steps, float(pred.predict_delta_t), self._ELLIPSE_STRIDE_S):
            cov = _prediction_step_covariance(pred, agent_id, mode_index, step)
            if cov is None:
                continue
            center = np.asarray(centers[step], dtype=float)
            for percentile in self.PREDICTION_PERCENTILES:
                polygon = _ellipse_polygon(center, cov, percentile, self._ELLIPSE_SAMPLES)
                if polygon is None:
                    continue
                if self.coord == "frenet":
                    regions.append(global_trajectory.convert_xy_path_to_sd_path_np(polygon))
                else:
                    regions.append(polygon)
                region_colors.append(to_rgba(color, alpha=float(np.clip(weight, 0.0, 1.0)) * percentile))

    def update_lidar_plot(self, lidar_data, show_plot=True, global_trajectory=None):
        """Update the LiDAR scatter in this plot's coordinate frame."""
        if not show_plot or lidar_data is None or len(lidar_data) == 0:
            self.lidar_scatter.set_offsets(np.empty((0, 2)))
            return
        xy = lidar_data[:, :2]
        if self.coord == "frenet":
            if global_trajectory is None:
                self.lidar_scatter.set_offsets(np.empty((0, 2)))
                return
            self.lidar_scatter.set_offsets(global_trajectory.convert_xy_path_to_sd_path_np(xy))
        else:
            self.lidar_scatter.set_offsets(xy)

    def update_cluster_plot(self, clusters, show_plot=True, global_trajectory=None):
        """Highlight the LiDAR points that passed segmentation + range gating."""
        if not show_plot or clusters is None or len(clusters) == 0:
            self.cluster_scatter.set_offsets(np.empty((0, 2)))
            return
        xy = np.asarray(clusters)[:, :2]
        if self.coord == "frenet" and global_trajectory is not None:
            self.cluster_scatter.set_offsets(global_trajectory.convert_xy_path_to_sd_path_np(xy))
        elif self.coord == "frenet":
            self.cluster_scatter.set_offsets(np.empty((0, 2)))
        else:
            self.cluster_scatter.set_offsets(xy)

    _OCCUPANCY_FLOW_MIN = 0.02

    def _clear_occupancy_flow_cells(self) -> None:
        self.occupancy_flow_cells.set_verts([])
        self.occupancy_flow_cells.set_facecolors("none")

    def _view_limits(self) -> tuple[float, float, float, float]:
        if self._window is not None:
            return self._window
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        return (x0, x1, y0, y1)

    def update_pm_occupancy_flow_plots(
        self,
        pm: Optional[PerceptionModel] = None,
        show_plot=True,
        global_trajectory=None,
    ):
        shown = _occupancy_flow_image(pm.prediction) if show_plot and pm is not None else None
        if shown is None:
            self._clear_occupancy_flow_cells()
            return
        image, origin_x, origin_y, resolution = shown
        h, w = image.shape
        if self.coord == "frenet":
            x0, x1 = origin_x, origin_x + w * resolution
            y0, y1 = origin_y, origin_y + h * resolution
        else:
            x0, x1, y0, y1 = self._view_limits()
        c0 = max(0, int(np.floor((min(x0, x1) - origin_x) / resolution)))
        c1 = min(w - 1, int(np.floor((max(x0, x1) - origin_x) / resolution)))
        r0 = max(0, int(np.floor((min(y0, y1) - origin_y) / resolution)))
        r1 = min(h - 1, int(np.floor((max(y0, y1) - origin_y) / resolution)))
        if c1 < c0 or r1 < r0:
            self._clear_occupancy_flow_cells()
            return
        sub = image[r0:r1 + 1, c0:c1 + 1]
        rows, cols = np.nonzero(sub > self._OCCUPANCY_FLOW_MIN)
        if len(rows) == 0:
            self._clear_occupancy_flow_cells()
            return
        probs = sub[rows, cols]
        rgba = np.array(plt.cm.plasma(np.clip(probs, 0.0, 1.0)), copy=True)
        rgba[:, 3] = np.clip(probs, 0.0, 1.0)
        x_left = origin_x + (cols + c0) * resolution
        y_bottom = origin_y + (rows + r0) * resolution
        quads = np.empty((len(rows), 4, 2))
        quads[:, 0, 0] = x_left
        quads[:, 0, 1] = y_bottom
        quads[:, 1, 0] = x_left + resolution
        quads[:, 1, 1] = y_bottom
        quads[:, 2, 0] = x_left + resolution
        quads[:, 2, 1] = y_bottom + resolution
        quads[:, 3, 0] = x_left
        quads[:, 3, 1] = y_bottom + resolution
        if self.coord == "frenet":
            if global_trajectory is None:
                self._clear_occupancy_flow_cells()
                return
            quads = np.asarray(
                global_trajectory.convert_xy_path_to_sd_path_np(quads.reshape(-1, 2)),
                dtype=float,
            ).reshape(quads.shape)
        _paint_occupancy_cells(self.occupancy_flow_cells, quads, rgba)

    def update_pm_occupancy_map_plots(self, pm: Optional[PerceptionModel] = None, show_plot=True):
        if self.coord == "frenet":
            return
        om = getattr(pm, "occupancy_map", None) if pm is not None else None
        if not isinstance(om, OccupancyMap):
            fallback = getattr(pm, "map", None) if pm is not None else None
            om = fallback if isinstance(fallback, OccupancyMap) else None
        if not show_plot or om is None:
            if hasattr(self, "pm_occupancy_map"):
                self.pm_occupancy_map.set_data(np.zeros((1, 1)))
                self.pm_occupancy_map.set_extent([0, 0, 0, 0])
                self.pm_occupancy_map.set_zorder(0)
            return
        if om.grid is None or om.grid.size == 0:
            return
        h, w = om.grid.shape
        extent = [
            om.origin_x,
            om.origin_x + w * om.resolution,
            om.origin_y,
            om.origin_y + h * om.resolution,
        ]
        extent_key = (h, w, float(om.origin_x), float(om.origin_y), float(om.resolution))
        if getattr(self, "_occ_map_extent_key", None) != extent_key:
            self._needs_full_draw = True
            self._occ_map_extent_key = extent_key
        shape = (h, w)
        recreate = (
            not hasattr(self, "pm_occupancy_map")
            or getattr(self, "_occ_map_shape", None) != shape
        )
        if recreate:
            if hasattr(self, "pm_occupancy_map"):
                self.pm_occupancy_map.remove()
            self.pm_occupancy_map = self.ax.imshow(
                om.grid,
                origin="lower",
                extent=extent,
                cmap="Greys",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
                zorder=0,
            )
            self._occ_map_shape = shape
        else:
            self.pm_occupancy_map.set_data(om.grid)
            self.pm_occupancy_map.set_extent(extent)
        self.pm_occupancy_map.set_zorder(0)


    def set_plot_theme(self, bg_color="white", fg_color="black"):
        self.fig.patch.set_facecolor(bg_color)
        self.ax.patch.set_facecolor(bg_color)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(fg_color)
        self.ax.tick_params(axis="both", colors=fg_color)
        self.ax.xaxis.label.set_color(fg_color)
        self.ax.yaxis.label.set_color(fg_color)
        log.debug(f"Plot theme set to {bg_color} background and {fg_color} foreground.")
        self.redraw_plots()

    def show_vehicle_orientation(self, x, y, theta, color="red"):
        """Show the vehicle orientation on the plot."""
        length = .15 * min(self.view_width, self.view_height) if self.view_height is not None or self.view_width is None else 20
        x2 = x + length * np.cos(theta)
        y2 = y + length * np.sin(theta)
        if self.orientation_arrow:
            self.orientation_arrow.remove()
        self.orientation_arrow = self.ax.annotate(
            "", xy=(x2, y2), xytext=(x, y),
            arrowprops=dict(arrowstyle="->", mutation_scale=20, color=color, lw=5), zorder=5,
        )

    def show_distance_ruler(self, x0, y0, x1, y1, dist_m: float) -> None:
        """Draw a light line from (x0, y0) to cursor with world-XY distance label."""
        key = (float(x0), float(y0), float(x1), float(y1), round(float(dist_m), 1))
        if self._ruler_visible and key == self._ruler_last:
            return
        self._ruler_line.set_data([x0, x1], [y0, y1])
        self._ruler_text.xy = (x1, y1)
        self._ruler_text.set_text(f"{dist_m:.1f} m")
        self._ruler_text.set_visible(True)
        self._ruler_visible = True
        self._ruler_last = key
        self.blit_manager.update()

    def hide_distance_ruler(self) -> None:
        if not self._ruler_visible:
            return
        self._ruler_line.set_data([], [])
        self._ruler_text.set_visible(False)
        self._ruler_visible = False
        self._ruler_last = None
        self.blit_manager.update()

    def clear_tmp_plots(self):
        self.orientation_arrow.remove() if self.orientation_arrow else None
        self.orientation_arrow = None
        self.hide_distance_ruler()

    def reset(self):
        self.update_pm_occupancy_flow_plots(None, show_plot=False)
        self.update_pm_occupancy_map_plots(None, show_plot=False)


# --- module helpers ---

def _decimate_polyline(x: np.ndarray, y: np.ndarray, window, margin_frac: float = 0.15):
    """Return only the polyline vertices near the visible window.

    Contiguous visible runs are kept intact (each extended by one vertex so the
    line meets the window edge); disjoint runs -- e.g. a closed lap whose start
    and finish both fall in view -- are joined with a NaN break so the arcs are
    not connected across the off-screen gap. Returns ``(x, y)`` ready for
    ``set_data``.
    """
    if window is None or x.size == 0:
        return x, y
    x0, x1, y0, y1 = window
    mx = (x1 - x0) * margin_frac
    my = (y1 - y0) * margin_frac
    inside = (x >= x0 - mx) & (x <= x1 + mx) & (y >= y0 - my) & (y <= y1 + my)
    if not inside.any():
        return x[:0], y[:0]
    idx = np.flatnonzero(inside)
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    xs_parts, ys_parts = [], []
    nan = np.array([np.nan])
    for k, run in enumerate(runs):
        a = max(int(run[0]) - 1, 0)
        b = min(int(run[-1]) + 2, x.size)
        if k > 0:
            xs_parts.append(nan)
            ys_parts.append(nan)
        xs_parts.append(x[a:b])
        ys_parts.append(y[a:b])
    return np.concatenate(xs_parts), np.concatenate(ys_parts)


# slow → fast : green → yellow → red (distinct from HD lane #427b58)
_VELOCITY_CMAP = LinearSegmentedColormap.from_list(
    "velocity_slow_fast",
    [(0.0, "#39ff14"), (0.5, "#ffd700"), (1.0, "#dc143c")],
)


def _update_velocity_colored_line(collection, x, y, velocity, *, velocity_scale="relative"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        collection.set_segments([])
        collection.set_array(np.array([]))
        return
    v = np.asarray(velocity, float)
    if len(v) == 0:
        v = np.full(len(x), 0.5)
    n = min(len(x), len(v))
    x, y, v = x[:n], y[:n], v[:n]
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
    seg_v = 0.5 * (v[:-1] + v[1:])
    collection.set_segments(segments)
    collection.set_array(seg_v)
    if velocity_scale == "absolute":
        vmax = float(ControlSettings.c32_ego_max_velocity)
        if vmax <= 0.0:
            vmax = 1e-9
        collection.set_norm(Normalize(vmin=0.0, vmax=vmax, clip=True))
    else:
        vmin, vmax = float(seg_v.min()), float(seg_v.max())
        if vmin == vmax:
            vmax = vmin + 1e-9
        collection.set_norm(Normalize(vmin=vmin, vmax=vmax))


def _ellipse_step_indices(n_steps: int, dt: float, stride_s: float) -> list[int]:
    """Forecast steps about every ``stride_s`` seconds, always including the last."""
    if n_steps <= 0:
        return []
    stride = max(1, int(round(stride_s / dt))) if dt > 0.0 else n_steps
    indices = list(range(stride - 1, n_steps, stride))
    if not indices or indices[-1] != n_steps - 1:
        indices.append(n_steps - 1)
    return indices


def _ellipse_polygon(center: np.ndarray, cov: np.ndarray, percentile: float, n: int) -> np.ndarray | None:
    """Closed-enough sample of the 2D Gaussian contour at ``percentile``.

    Semi-axes are ``sqrt(-2 ln(1 - p)) * sqrt(eigenvalues)``. A circular
    covariance is the special case of equal axes.
    """
    if not 0.0 < percentile < 1.0:
        return None
    sym = np.asarray(cov, dtype=float)
    if sym.shape != (2, 2) or not np.all(np.isfinite(sym)):
        return None
    sym = 0.5 * (sym + sym.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, 0.0, None)
    if float(vals[-1]) <= 0.0:
        return None
    radii = np.sqrt(-2.0 * np.log(1.0 - percentile)) * np.sqrt(vals)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    local = np.column_stack((np.cos(theta), np.sin(theta))) * radii
    polygon = np.asarray(center, dtype=float) + local @ vecs.T
    return np.vstack((polygon, polygon[:1]))


def _prediction_step_covariance(pred: GP | GMM, agent_id: int, mode_index: int, step: int) -> np.ndarray | None:
    """Marginal 2×2 at one forecast step. ``GP`` is stored ``[x0, y0, x1, y1, ...]``."""
    if isinstance(pred, GP):
        joint = pred.covariance.get(agent_id)
        if joint is None:
            return None
        i = 2 * step
        joint = np.asarray(joint)
        if joint.ndim != 2 or joint.shape[0] < i + 2 or joint.shape[1] < i + 2:
            return None
        return joint[i:i + 2, i:i + 2]
    covs = pred.covariances.get(agent_id)
    if covs is None:
        return None
    covs = np.asarray(covs)
    if covs.ndim != 4 or mode_index >= covs.shape[0] or step >= covs.shape[1]:
        return None
    return covs[mode_index, step]


def _occupancy_flow_image(pred) -> tuple[np.ndarray, float, float, float] | None:
    """Max-over-horizon occupancy. Per-agent grids combine with ``1 - Π(1 - p)``."""
    if isinstance(pred, AggregatedOccupancyFlow):
        grids = [np.asarray(g, dtype=float) for g in pred.occupancy_flow if np.size(g)]
        if not grids:
            return None
        return np.maximum.reduce(grids), pred.origin_x, pred.origin_y, pred.resolution
    if isinstance(pred, OccupancyFlow):
        series = [steps for steps in pred.occupancy_flow.values() if steps]
        if not series:
            return None
        n_steps = max(len(steps) for steps in series)
        acc = None
        for step in range(n_steps):
            union = None
            for steps in series:
                if step >= len(steps) or np.size(steps[step]) == 0:
                    continue
                grid = np.clip(np.asarray(steps[step], dtype=float), 0.0, 1.0)
                union = grid if union is None else 1.0 - (1.0 - union) * (1.0 - grid)
            if union is None:
                continue
            acc = union if acc is None else np.maximum(acc, union)
        if acc is None:
            return None
        return acc, pred.origin_x, pred.origin_y, pred.resolution
    return None


def _paint_occupancy_cells(collection: PolyCollection, quads: np.ndarray, rgba: np.ndarray) -> None:
    collection.set_alpha(None)
    collection.set_verts(quads)
    collection.set_facecolors(rgba)
    collection.set_edgecolors("none")
