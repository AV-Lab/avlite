from __future__ import annotations
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import logging
import numpy as np

from avlite.c10_perception.c11_perception_model import AgentState
from avlite.c30_control.c39_settings import ControlSettings
from avlite.c60_apps.c61_app_strategy import PlotStrategy
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c24_global_hdmap_planners import HDMapGlobalPlanner
from avlite.c20_planning.c25_global_race_planners import GlobalCenterlineRacePlanner, GlobalRacePlanner
from avlite.plugins.p60_visualizer_tk.p65_ui_lib import DpiScale, SpeedGauge, SteerWheel
from avlite.plugins.p60_visualizer_tk.p69_plot_lib import LocalPlot, GlobalRacePlot, GlobalHDMapPlot
from avlite.c40_execution.c41_world_bridge import (
    is_world_capability_enabled,
    is_world_stack_capability_enabled,
)
from avlite.c50_common.c51_capabilities import StackCapability, WorldCapability

log = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from avlite.plugins.p60_visualizer_tk.p61_visualizer_app import VisualizerApp

_CONTROL_GRACE_S = 0.5  # late Control after click still counts as drag




class LocalCoordPlot(PlotStrategy, abstract=True):
    """Shared canvas and pointer handling for one local coordinate frame."""

    coord = "xy"
    zoom_name = "p66_xy_zoom"
    lidar_name = "p66_show_lidar_global"
    follow_name = "p66_global_view_follow_planner"
    status_fmt = "Spawn {x:.1f},{y:.1f}"

    def __init__(self, root: VisualizerApp):
        super().__init__(root)
        self.local_plot = LocalPlot(coord=self.coord)
        self.fig = self.local_plot.fig
        self.ax = self.local_plot.ax
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("scroll_event", self.on_mouse_scroll)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_click)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self._prev_scroll_time = None
        self.left_mouse_button_pressed = False
        self.right_mouse_button_pressed = False
        self._spawn_agent_state = None
        self.press_x = 0.0
        self.press_y = 0.0
        self.teleport_x = 0.0
        self.teleport_y = 0.0
        self.teleport_orientation = 0.0
        self.spawn_x = 0.0
        self.spawn_y = 0.0
        self.spawn_orientation = 0.0
        self.update_plot_theme()

    def reset(self):
        self.local_plot.reset()

    def update_plot_theme(self):
        self.local_plot.set_plot_theme(self.root.setting.p60_bg_color, self.root.setting.p60_fg_color)

    def to_world(self, x, y):
        return x, y

    def to_world_heading(self, x, y, plot_theta):
        return plot_theta

    def marker_theta(self, x, y, world_theta):
        return world_theta

    def ruler_segment(self, fx, fy, mx, my):
        return fx, fy, mx, my, float(np.hypot(mx - fx, my - fy))

    def zoom_in(self):
        setting = self.root.setting
        zoom = getattr(setting, self.zoom_name)
        setattr(setting, self.zoom_name, zoom - (5 if zoom > 5 else 0))
        self.root.update_ui()

    def zoom_out(self):
        setting = self.root.setting
        setattr(setting, self.zoom_name, getattr(setting, self.zoom_name) + 5)
        self.root.update_ui()

    def on_mouse_scroll(self, event, increment=10):
        setting = self.root.setting
        zoom = getattr(setting, self.zoom_name)
        if event.button == "up":
            setattr(setting, self.zoom_name, zoom - (increment if zoom > increment else 0))
        elif event.button == "down":
            setattr(setting, self.zoom_name, zoom + increment)
        if (
            self._prev_scroll_time is None or time.time() - self._prev_scroll_time > 0.01
        ) and not setting.exec_running:
            self.root.update_ui()
        self._prev_scroll_time = time.time()

    def on_mouse_click(self, event):
        if event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        wx, wy = self.to_world(x, y)
        if event.button == 3:
            self.right_mouse_button_pressed = True
            self.press_x, self.press_y = x, y
            self.spawn_x, self.spawn_y = wx, wy
            self.spawn_orientation = self.root.exec.ego_state.theta
            agent = AgentState(x=wx, y=wy, theta=self.spawn_orientation, velocity=0)
            self.root.spawn_agent(agent)
            self._spawn_agent_state = agent
            self.local_plot.show_vehicle_orientation(
                x, y, self.marker_theta(x, y, self.spawn_orientation), color="blue"
            )
            self.root.update_ui()
        elif event.button == 1:
            self.root.teleport_ego(x=wx, y=wy)
            self.left_mouse_button_pressed = True
            self.press_x, self.press_y = x, y
            self.teleport_x, self.teleport_y = wx, wy
            self.root.exec.local_planner.step(state=self.root.exec.world.get_ego_state())
            self.root.update_ui()

    def on_mouse_move(self, event):
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            self.root.setting.perception_status_text.set(self.status_fmt.format(x=x, y=y))
            if self.left_mouse_button_pressed:
                plot_theta = np.arctan2(y - self.press_y, x - self.press_x)
                self.local_plot.show_vehicle_orientation(self.press_x, self.press_y, plot_theta)
                self.teleport_orientation = self.to_world_heading(x, y, plot_theta)
                self.root.teleport_ego(self.teleport_x, self.teleport_y, self.teleport_orientation)
                self.root.exec.local_planner.step(state=self.root.exec.world.get_ego_state())
                self.root.update_ui()
            elif self.right_mouse_button_pressed:
                plot_theta = np.arctan2(y - self.press_y, x - self.press_x)
                self.local_plot.show_vehicle_orientation(
                    self.press_x, self.press_y, plot_theta, color="blue"
                )
                self.spawn_orientation = self.to_world_heading(self.press_x, self.press_y, plot_theta)
                if self._spawn_agent_state is not None:
                    self._spawn_agent_state.theta = self.spawn_orientation
                self.root.update_ui()
        else:
            self.root.setting.perception_status_text.set("Click plot")

        if (
            self.root.setting.exec_running
            or self.left_mouse_button_pressed
            or self.right_mouse_button_pressed
            or event.inaxes != self.ax
            or self.root.exec is None
        ):
            self.local_plot.hide_distance_ruler()
            return
        ego = self.root.exec.ego_state
        ctrl = self.root.exec.controller
        front = float(getattr(ctrl, "ego_distance_front_axle", 2.5) if ctrl is not None else 2.5)
        fx = float(ego.x) + front * float(np.cos(ego.theta))
        fy = float(ego.y) + front * float(np.sin(ego.theta))
        segment = self.ruler_segment(fx, fy, float(event.xdata), float(event.ydata))
        if segment is None:
            self.local_plot.hide_distance_ruler()
            return
        self.local_plot.show_distance_ruler(*segment)

    def on_mouse_release(self, event):
        if event.button == 1:
            self.left_mouse_button_pressed = False
            self.local_plot.clear_tmp_plots()
            self.root.exec.controller.reset()
            self.root.update_ui()
            log.debug(
                f"Teleport Ego to X: {self.teleport_x:.2f}, Y: {self.teleport_y:.2f}, "
                f"Orientation: {self.teleport_orientation:.2f}"
            )
        elif event.button == 3 and self.right_mouse_button_pressed:
            self.right_mouse_button_pressed = False
            self._spawn_agent_state = None
            self.local_plot.clear_tmp_plots()
            self.root.update_ui()
            log.debug(
                f"Spawn Agent at X: {self.spawn_x:.2f}, Y: {self.spawn_y:.2f}, "
                f"Orientation: {self.spawn_orientation:.2f}"
            )

    def plot(self):
        if self.root.exec is None:
            return
        canvas_widget = self.canvas.get_tk_widget()
        if not _canvas_ready(canvas_widget):
            self.root.after_idle(self.plot)
            return
        if self.root.setting.exec_running:
            self.local_plot.hide_distance_ruler()
        width = canvas_widget.winfo_width()
        height = canvas_widget.winfo_height()
        aspect_ratio = width / height if height else 4.0
        setting = self.root.setting
        bridge_lidar = is_world_capability_enabled(WorldCapability.LIDAR_2D) or is_world_capability_enabled(WorldCapability.LIDAR_3D)
        want_lidar = bridge_lidar and getattr(setting, self.lidar_name).get()
        lidar_data = None
        if want_lidar:
            world = self.root.exec.world
            lidar = world.get_sensor_frame().get_lidar()
            if lidar is not None:
                lidar_data = lidar.to_map(lidar.points, world.get_ego_state())
        t1 = time.time()
        self.local_plot.plot(
            exec=self.root.exec,
            aspect_ratio=aspect_ratio,
            zoom=getattr(setting, self.zoom_name),
            show_legend=setting.p66_show_legend.get(),
            plot_last_pts=setting.p66_show_past_locations.get(),
            plot_global_plan=setting.p66_show_global_plan.get(),
            plot_local_plan=setting.p66_show_local_plan.get(),
            plot_local_lattice=setting.p66_show_local_lattice.get(),
            plot_state=setting.p66_show_state.get(),
            follow_planner=getattr(setting, self.follow_name).get(),
            plot_occupancy_map=setting.p67_show_occupancy_map.get(),
            plot_predictions=setting.p67_show_prediction.get(),
            plot_lidar=want_lidar,
            lidar_data=lidar_data,
            plot_clusters=bridge_lidar and setting.p66_show_lidar_clusters.get(),
            plot_ground_truth=is_world_stack_capability_enabled(StackCapability.DETECTION),
            plot_race_boundary=setting.p66_show_race_boundary.get(),
        )
        self.local_plot.blit_manager.full_draw()
        log.debug(f"Local Plot Time: {(time.time()-t1)*1000:.2f} ms (aspect_ratio: {aspect_ratio:0.2f})")


class LocalGlobalPlot(LocalCoordPlot):
    label = "Local Planner (Global)"
    enabled_field = "p67_show_local_global_view"


class FrenetPlot(LocalCoordPlot):
    label = "Local Planner (Frenet)"
    enabled_field = "p67_show_local_frenet_view"
    coord = "frenet"
    zoom_name = "p66_frenet_zoom"
    lidar_name = "p66_show_lidar_frenet"
    follow_name = "p66_frenet_view_follow_planner"
    status_fmt = "Spawn S:{x:.1f},D:{y:.1f}"

    def to_world(self, x, y):
        return self.root.exec.local_planner.global_plan.trajectory.convert_sd_to_xy(x, y)

    def to_world_heading(self, x, y, plot_theta):
        _, _, theta = (
            self.root.exec.local_planner.global_plan.trajectory
            .convert_sd_orientation_to_xy_orientation(x, y, plot_theta)
        )
        return theta

    def marker_theta(self, x, y, world_theta):
        _, _, path_heading = (
            self.root.exec.local_planner.global_plan.trajectory
            .convert_sd_orientation_to_xy_orientation(x, y, 0.0)
        )
        return world_theta - path_heading

    def ruler_segment(self, fx, fy, mx, my):
        traj = getattr(getattr(self.root.exec, "local_planner", None), "global_trajectory", None)
        if traj is None:
            return None
        fs, fd = traj.convert_xy_to_sd(fx, fy)
        cx, cy = traj.convert_sd_to_xy(mx, my)
        return fs, fd, mx, my, float(np.hypot(cx - fx, cy - fy))


class GlobalPlanPlot(PlotStrategy):
    label = "Global Planner"
    enabled_field = "p67_global_plan_view"

    def __init__(self, root: VisualizerApp):
        super().__init__(root)

        if self.root.setting.global_planner_type.get() in (
            GlobalCenterlineRacePlanner.__name__,
            GlobalRacePlanner.__name__,
        ):
            self.global_plot = GlobalRacePlot()
        elif self.root.setting.global_planner_type.get() == HDMapGlobalPlanner.__name__:
            self.global_plot = GlobalHDMapPlot()

        if not hasattr(self, "global_plot"):
            log.error("Global Plot type not set. Please check the global planner type.")

        self.__config_canvas()

        self.start_point = None
        self._prev_scroll_time = None  # used to throttle the replot
        self._init_drag_mouse_pos = None # used for drag the global map
        self._drag_mode = False  # used to drag the global map
        self._center_delta = (0, 0)  # used to adjust the center of the global map

        self.left_mouse_button_pressed = False  
        self.teleport_x = 0.0
        self.teleport_y = 0.0
        self.teleport_orientation = 0.0
        self._left_press_time = 0.0
        self._ego_orient_active = False
        

    def __config_canvas(self):  
        self.fig = self.global_plot.fig
        self.ax = self.global_plot.ax   

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.global_plot.set_plot_theme(self.root.setting.p60_bg_color, self.root.setting.p60_fg_color)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_click)
        self.canvas.mpl_connect("scroll_event", self.on_mouse_scroll)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        
    def __get_aspect_ratio(self):
        canvas_widget = self.canvas.get_tk_widget()
        width = canvas_widget.winfo_width()
        height = canvas_widget.winfo_height()
        aspect_ratio = width / height if height > 0 else 4.0
        return aspect_ratio

    def _clamp_center_delta(self):
        gp = self.global_plot
        if gp.map_min_x is None:
            return
        pad = 100
        cx = self.root.exec.ego_state.x + self._center_delta[0]
        cy = self.root.exec.ego_state.y + self._center_delta[1]
        cx = np.clip(cx, gp.map_min_x - pad, gp.map_max_x + pad)
        cy = np.clip(cy, gp.map_min_y - pad, gp.map_max_y + pad)
        self._center_delta = (cx - self.root.exec.ego_state.x, cy - self.root.exec.ego_state.y)

    def _max_zoom_out(self, aspect_ratio: float) -> float | None:
        gp = self.global_plot
        if gp.map_min_x is None:
            return None
        map_width = gp.map_max_x - gp.map_min_x
        map_height = gp.map_max_y - gp.map_min_y
        return min(map_width, map_height * aspect_ratio)

    def _control_held(self, event) -> bool:
        gui = getattr(event, "guiEvent", None)
        if gui is not None and getattr(gui, "state", 0) & 0x0004:
            return True
        return bool(event.key and "control" in event.key)

    def plot(self):
        canvas_widget = self.canvas.get_tk_widget()
        if not _canvas_ready(canvas_widget):
            self.root.after_idle(self.plot)
            return
        t1 = time.time()
        try:
            aspect_ratio = self.__get_aspect_ratio()
            max_zoom = self._max_zoom_out(aspect_ratio)
            if max_zoom is not None and self.root.setting.p66_global_zoom >= max_zoom:
                self._center_delta = (0, 0)
            self._clamp_center_delta()

            self.global_plot.plot(
                exec=self.root.exec,
                aspect_ratio=aspect_ratio,
                zoom=self.root.setting.p66_global_zoom,
                show_legend=self.root.setting.p66_show_legend.get(),
                follow_vehicle=self.root.setting.p66_global_view_follow_planner.get(),
                show_plan_boundaries=self.root.setting.p66_show_global_plan_boundaries.get(),
                velocity_scale=self.root.setting.p66_global_plan_velocity_scale.get(),
                delta=self._center_delta,
            )
            log.debug(f"Global Plot Time: {(time.time()-t1)*1000:.2f} ms (aspect_ratio: {aspect_ratio:0.2f})")
        except Exception as e:
            log.error(f"Error in Global Plot: {e}")
            self.update_plot_type()
        
    
    def update_plot_theme(self):
        self.global_plot.set_plot_theme(self.root.setting.p60_bg_color, self.root.setting.p60_fg_color)
        

    def update_plot_type(self):
        """Update the plot type based on the selected global planner"""

        planner_type = self.root.setting.global_planner_type.get()
        log.debug(f"Updating Global Plot type {planner_type}...")
        if hasattr(self, "global_plot"):
            self.global_plot.close()
        if hasattr(self, "canvas"):
            self.canvas.get_tk_widget().destroy()

        if planner_type not in GlobalPlannerStrategy.registry.keys():
            log.error(f"Global planner type '{planner_type}' not found in registry. Defaulting to race plot.")
            self.global_plot = GlobalRacePlot()
        elif planner_type in (GlobalCenterlineRacePlanner.__name__, GlobalRacePlanner.__name__):
            self.global_plot = GlobalRacePlot()
            log.debug("Global Plot type changed to Race Plot.")
        elif planner_type == HDMapGlobalPlanner.__name__:
            self.global_plot = GlobalHDMapPlot()
            log.debug("Global Plot type changed to HD Map Plot.")
        else:
            log.error(f"No plot view defined for planner type '{planner_type}'. Defaulting to race plot.")
            self.global_plot = GlobalRacePlot()

        self.__config_canvas()


    def on_mouse_move(self, event):
        try:
            if event.inaxes == self.ax:
                x, y = event.xdata, event.ydata

                self.root.setting.perception_status_text.set(f"Teleport {x:.1f},{y:.1f}")

                # Hover speed readout on the colored raceline (idle only).
                if (not self.left_mouse_button_pressed
                        and not self._drag_mode
                        and hasattr(self.global_plot, "show_speed_at")):
                    self.global_plot.show_speed_at(x, y)

                if self.root.setting.global_planner_type.get() == HDMapGlobalPlanner.__name__:
                    if not self.left_mouse_button_pressed:
                        self.global_plot.show_closest_road_and_lane(x=int(x), y=int(y), map=self.root.exec.global_planner.hdmap)   
                
                if self._control_held(event) and self._drag_mode:
                    dx =-(x - self._init_drag_mouse_pos[0])*self.root.setting.p69_mouse_drag_slowdown_factor
                    dy =-(y - self._init_drag_mouse_pos[1])*self.root.setting.p69_mouse_drag_slowdown_factor
                    self._center_delta = (self._center_delta[0]+dx, self._center_delta[1]+dy)
                    self._init_drag_mouse_pos = (x, y)
                    self.plot()

                if self.left_mouse_button_pressed and not self._drag_mode:
                    if self._control_held(event):
                        if (not self._ego_orient_active
                                or time.time() - self._left_press_time < _CONTROL_GRACE_S):
                            self._drag_mode = True
                            self._init_drag_mouse_pos = (x, y)
                            self.global_plot.clear_tmp_plots()
                            return
                    self._ego_orient_active = True
                    self.teleport_orientation = np.arctan2(y - self.teleport_y, x - self.teleport_x)
                    self.global_plot.show_vehicle_orientation(self.teleport_x, self.teleport_y, self.teleport_orientation) 
                    self.root.teleport_ego(x=self.teleport_x, y=self.teleport_y, theta=self.teleport_orientation)
                    self.root.exec.local_planner.step(state=self.root.exec.world.get_ego_state())       
                    self.root.update_ui()
            else:
                self.root.setting.perception_status_text.set("Click plot")
                self.global_plot.clear_tmp_plots()
                self._drag_mode = False
                self._init_drag_mouse_pos = None
                self._ego_orient_active = False
        except Exception as e:
            log.error(f"Error in mouse move event: {e}", exc_info=True)


    def on_mouse_click(self, event):
        if event.inaxes == self.ax:
            if event.button == 1:  # Left click
                self.left_mouse_button_pressed = True
                if self._control_held(event):  # for dragging
                    self._init_drag_mouse_pos = (event.xdata, event.ydata)
                    self._drag_mode = True
                    self.global_plot.clear_tmp_plots()
                else:
                    x, y = event.xdata, event.ydata
                    self.teleport_x = x
                    self.teleport_y = y
                    self._left_press_time = time.time()
                    self._ego_orient_active = False
                    self.global_plot.clear_tmp_plots()
            elif event.button == 3: # Right click
                if self.start_point:
                    self.global_plot.set_goal(event.xdata, event.ydata)
                    self.root.exec.global_planner.set_start_goal(start_point=self.start_point, goal_point=(event.xdata, event.ydata))
                    log.info(f"Set start: {self.start_point}, goal: {(event.xdata, event.ydata)}")

                    self.root.exec.global_planner.plan(
                        perception_model=self.root.exec.pm,
                        sensors=self.root.exec.world.get_sensor_frame(),
                    )
                    if len(self.root.exec.global_planner.global_plan.path) == 0:
                        log.warning("No global plan found. Please check the start and goal points.")
                        return

                    if self.root.setting.global_planner_type.get() == HDMapGlobalPlanner.__name__:
                        self.global_plot.plot_global_plan(self.root.exec.global_planner.global_plan)
                        self.root.apply_global_plan(
                            self.root.exec.global_planner.global_plan,
                            ego_xy=(self.root.exec.ego_state.x, self.root.exec.ego_state.y),
                        )
                        for view in self.root.plots.values():
                            if isinstance(view, LocalCoordPlot):
                                view.reset()
                        self.root.update_ui()

                    self.start_point = None
                else:
                    self.global_plot.set_start(event.xdata, event.ydata)
                    self.global_plot.clear_goal()
                    self.global_plot.clear_road_path_plots()
                    self.pending_goal_set = True
                    self.start_point = (event.xdata, event.ydata)

    def on_mouse_release(self, event):
        if event.inaxes == self.ax:
            if event.button == 1:
                if not self._drag_mode and not self._ego_orient_active:
                    self.root.teleport_ego(x=self.teleport_x, y=self.teleport_y)
                    self.root.exec.local_planner.step(state=self.root.exec.world.get_ego_state())
                self.left_mouse_button_pressed = False
                self._drag_mode = False
                self._init_drag_mouse_pos = None
                self._ego_orient_active = False
                self.global_plot.clear_tmp_plots()
                self.root.exec.controller.reset()
                self.root.update_ui()
                log.debug(f"Teleport Ego to X: {self.teleport_x:.2f}, Y: {self.teleport_y:.2f}, Orientation: {self.teleport_orientation:.2f}")
    
    def on_mouse_scroll(self, event, increment=10):
        log.debug(f"Scroll Event in global coordinate. Zoom: {self.root.setting.p66_global_zoom}")
        if event.button == "up":
            self.root.setting.p66_global_zoom -= increment if self.root.setting.p66_global_zoom > increment else 0
        elif event.button == "down":
            max_zoom = self._max_zoom_out(self.__get_aspect_ratio())
            if max_zoom is None or self.root.setting.p66_global_zoom < max_zoom:
                self.root.setting.p66_global_zoom += increment
                if max_zoom is not None:
                    self.root.setting.p66_global_zoom = min(self.root.setting.p66_global_zoom, max_zoom)
        threshold = 0.01
        if (self._prev_scroll_time is None or time.time() - self._prev_scroll_time > threshold) and not self.root.setting.exec_running:
            # self.root.update_ui()
            # center = None
            # if event.key and 'control' in event.key:
            #     center = (event.xdata, event.ydata)
            self.plot()


        self._prev_scroll_time = time.time()

    def reset(self):
        self.update_plot_type()


class ControlPlot(PlotStrategy):
    """Speed gauge, steering wheel, and a short g-g trail."""

    label = "Control"
    enabled_field = "p67_show_control_view"

    def __init__(self, root: VisualizerApp):
        super().__init__(root)
        scale = DpiScale.for_widget(self)
        self.gg = tk.Canvas(self, highlightthickness=0, bd=0)
        self.speed_gauge = SpeedGauge(
            self,
            accel_min=ControlSettings.c32_ego_min_acceleration,
            accel_max=ControlSettings.c32_ego_max_acceleration,
            dpi_scale=scale,
        )
        self.steer_wheel = SteerWheel(
            self,
            limit=max(abs(ControlSettings.c32_ego_min_steering), abs(ControlSettings.c32_ego_max_steering)),
            dpi_scale=scale,
        )
        self.trail = []
        self.previous = None
        self.face = None
        self.dial = None
        font = tkfont.Font(root=self, font=("Courier", 8))
        # Measured once. The live readouts sit in a margin of this size outside the rings.
        self.readout = (font.measure("-0.0"), font.metrics("linespace"))

    def reset(self) -> None:
        self.trail.clear()
        self.previous = None
        for tag in ("trail", "now"):
            if self.gg.find_withtag(tag):
                self.gg.coords(tag, 0, 0, 0, 0)
                self.gg.itemconfigure(tag, state="hidden")

    def plot(self) -> None:
        ego = self.root.exec.ego_state
        now = self.root.exec.elapsed_sim_time
        if self.previous is not None:
            then, speed, heading = self.previous
            dt = now - then
            if 0 < dt <= 1.0:
                turn = (ego.theta - heading + np.pi) % (2 * np.pi) - np.pi
                self.trail.append((ego.velocity * turn / dt, (ego.velocity - speed) / dt))
                del self.trail[:-80]
        self.previous = (now, ego.velocity, ego.theta)
        plot_w, plot_h = self.winfo_width(), self.winfo_height()
        margin = 8
        inner_w = plot_w - 2 * margin
        inner_h = plot_h - 2 * margin
        speed_w = self.speed_gauge.caption_w
        steer_w = self.steer_wheel.caption_w
        speed_h = self.speed_gauge.caption_h
        steer_h = self.steer_wheel.caption_h
        # The rings match the dials. The readouts are extra, like the captions under the dials.
        side_m = self.readout[0] + 12
        top_m = self.readout[1] + 4

        def largest(allow_wrap):
            found = 1
            lo, hi = 1, max(1, min(inner_w, inner_h))
            while lo <= hi:
                mid = (lo + hi) // 2
                sizes = (
                    (max(mid, speed_w), mid + speed_h),
                    (max(mid, steer_w), mid + steer_h),
                    (mid + 2 * side_m, mid + top_m),
                )
                used = 0
                column_h = column_w = 0
                open_column = False
                fits = inner_w > 0 and inner_h > 0
                for w, h in sizes:
                    if not fits or w > inner_w or h > inner_h:
                        fits = False
                        break
                    if open_column and column_h + h > inner_h:
                        if not allow_wrap:
                            fits = False
                            break
                        used += column_w
                        column_h = column_w = 0
                        open_column = False
                    open_column = True
                    column_w = max(column_w, w)
                    column_h += h
                if fits:
                    used += column_w
                    fits = used <= inner_w
                if fits:
                    found = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            return found

        # Keep the diagram under the wheel. A second column is only for a pane so short that it pays off clearly.
        stacked, wrapped = largest(False), largest(True)
        wanted = wrapped if wrapped > 1.5 * stacked else stacked
        # A one-pixel jitter would recreate both canvases every frame.
        if plot_w > 48 and plot_h > 48 and wanted > 48 and (self.dial is None or abs(wanted - self.dial) > 4):
            scale = DpiScale.for_widget(self)
            self.speed_gauge.destroy()
            self.steer_wheel.destroy()
            self.speed_gauge = SpeedGauge(
                self,
                accel_min=ControlSettings.c32_ego_min_acceleration,
                accel_max=ControlSettings.c32_ego_max_acceleration,
                dpi_scale=scale,
                dial=wanted,
            )
            self.steer_wheel = SteerWheel(
                self,
                limit=max(abs(ControlSettings.c32_ego_min_steering), abs(ControlSettings.c32_ego_max_steering)),
                dpi_scale=scale,
                dial=wanted,
            )
            self.dial = wanted
        if plot_w > 48 and plot_h > 48 and wanted > 48:
            # One size for the circles and the diagram. Stack downward, and start a column on the right when the next one does not fit.
            gg_w, gg_h = wanted + 2 * side_m, wanted + top_m
            items = (
                (self.speed_gauge, self.speed_gauge.winfo_reqwidth(), self.speed_gauge.winfo_reqheight()),
                (self.steer_wheel, self.steer_wheel.winfo_reqwidth(), self.steer_wheel.winfo_reqheight()),
                (self.gg, gg_w, gg_h),
            )
            columns = []
            column = []
            y = col_w = 0
            for widget, w, h in items:
                if column and y + h > inner_h:
                    columns.append((column, col_w, y))
                    column = []
                    y = col_w = 0
                column.append((widget, w, h, y))
                col_w = max(col_w, w)
                y += h
            if column:
                columns.append((column, col_w, y))
            used_w = sum(width for _, width, _ in columns)
            x = max(0, (plot_w - used_w) // 2) if len(columns) == 1 else margin
            for column, width, height in columns:
                y0 = max(0, (plot_h - height) // 2) if len(columns) == 1 else margin
                for widget, w, h, top in column:
                    px = x + max(0, (width - w) // 2)
                    py = y0 + top
                    info = widget.place_info()
                    if (
                        not info
                        or int(info["x"]) != px
                        or int(info["y"]) != py
                        or int(info["width"]) != w
                        or int(info["height"]) != h
                    ):
                        widget.place(x=px, y=py, width=w, height=h)
                x += width
            if self.gg.winfo_width() != gg_w or self.gg.winfo_height() != gg_h:
                self.update_idletasks()
        controller = getattr(self.root.exec, "controller", None)
        cmd = controller.cmd if controller is not None else getattr(self.root, "_applied_cmd", None)
        self.speed_gauge.set(ego.velocity, 0.0 if cmd is None else cmd.acceleration)
        self.steer_wheel.set(0.0 if cmd is None else cmd.steer)

        width, height = self.gg.winfo_width(), self.gg.winfo_height()
        radius = min(width - 2 * side_m, height - top_m) / 2.0 - 4.0
        if radius < 12:
            return
        cx = width / 2.0
        cy = (height + top_m) / 2.0
        style = ttk.Style(self)
        background = style.lookup("TFrame", "background")
        foreground = style.lookup("TLabel", "foreground") or "black"
        if background:
            self.gg.configure(bg=background)
        limit = max(
            abs(ControlSettings.c32_ego_min_acceleration),
            abs(ControlSettings.c32_ego_max_acceleration),
            1e-3,
        )
        outer = max(1.0, float(np.ceil(limit / 9.80665)))
        spec = ("Courier", 8)
        ground = self.gg.cget("bg")
        if self.face != (width, height, outer, foreground, ground):
            self.gg.delete("ring", "tick", "label", "plate")
            slant = 2 ** -0.5
            # Rings closer than a text line would stack the numbers on each other; the outer one is enough then.
            roomy = radius / 3.0 * slant >= self.readout[1] + 1
            for step in range(1, 4):
                ring = radius * step / 3.0
                self.gg.create_oval(
                    cx - ring, cy - ring, cx + ring, cy + ring,
                    outline=foreground, width=1, tags="ring",
                )
                for dx, dy in ((0.0, -ring), (ring, 0.0), (0.0, ring), (-ring, 0.0)):
                    self.gg.create_oval(
                        cx + dx - 2, cy + dy - 2, cx + dx + 2, cy + dy + 2,
                        fill=foreground, outline="", tags="tick",
                    )
                if roomy or step == 3:
                    # On its own ring, on the upper-left diagonal. The plate breaks the stroke behind the number.
                    item = self.gg.create_text(
                        cx - ring * slant, cy - ring * slant, text=f"{outer * step / 3.0:.1f}",
                        fill="#888888", font=spec, tags="label",
                    )
                    x0, y0, x1, y1 = self.gg.bbox(item)
                    self.gg.create_rectangle(x0 - 1, y0, x1 + 1, y1, fill=ground, outline="", tags="plate")
                    self.gg.tag_raise(item)
            self.gg.create_text(cx, cy, text="0.0", fill="#888888", font=spec, tags="label")
            self.gg.tag_raise("trail")
            self.gg.tag_raise("now")
            self.face = (width, height, outer, foreground, ground)
        if self.trail:
            lateral, longitudinal = self.trail[-1]
            long_text = f"{longitudinal / 9.80665:+.1f}"
            lat_text = f"{lateral / 9.80665:+.1f}"
        else:
            long_text = lat_text = "0.0"
        for tag, x, y, text, anchor in (
            ("along", cx, cy - radius - 3, long_text, tk.S),
            ("alat", cx + radius + 10, cy, lat_text, tk.W),
        ):
            if self.gg.find_withtag(tag):
                self.gg.coords(tag, x, y)
                self.gg.itemconfigure(tag, text=text, fill=foreground, font=spec)
            else:
                self.gg.create_text(x, y, text=text, anchor=anchor, fill=foreground, font=spec, tags=tag)
        span = outer * 9.80665
        flat = []
        for lateral, longitudinal in self.trail:
            flat.extend((cx + lateral / span * radius, cy - longitudinal / span * radius))
        if len(flat) >= 4:
            if self.gg.find_withtag("trail"):
                self.gg.coords("trail", *flat)
                self.gg.itemconfigure("trail", state="normal", fill=foreground)
            else:
                self.gg.create_line(*flat, fill=foreground, width=1, tags="trail")
        elif self.gg.find_withtag("trail"):
            self.gg.itemconfigure("trail", state="hidden")
        if flat:
            x, y = flat[-2], flat[-1]
            dot = (x - 5, y - 5, x + 5, y + 5)
            if self.gg.find_withtag("now"):
                self.gg.coords("now", *dot)
                self.gg.itemconfigure("now", state="normal", fill="#3d8bfd", outline="")
            else:
                self.gg.create_oval(*dot, fill="#3d8bfd", outline="", tags="now")
            self.gg.tag_raise("now")


def _canvas_ready(widget, *, min_px: int = 80) -> bool:
    w, h = widget.winfo_width(), widget.winfo_height()
    if w < min_px or h < min_px:
        return False
    parent = widget.master
    pw, ph = parent.winfo_width(), parent.winfo_height()
    if pw > min_px and w < int(pw * 0.5):
        return False
    if ph > min_px and h < int(ph * 0.5):
        return False
    return True
