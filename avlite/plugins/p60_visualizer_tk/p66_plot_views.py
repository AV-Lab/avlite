from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import logging
import numpy as np

from avlite.c10_perception.c11_perception_model import AgentState
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c24_global_hdmap_planners import HDMapGlobalPlanner
from avlite.c20_planning.c25_global_race_planners import GlobalCenterlineRacePlanner, GlobalRacePlanner
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




class GlobalPlanPlotView(ttk.Frame):
    def __init__(self, root: VisualizerApp):
        super().__init__(root)
        self.root = root

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
                        self.root.local_plan_plot_view.reset()
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

class LocalPlanPlotView(ttk.Frame):

    def __init__(self, root: VisualizerApp):
        super().__init__(root)
        self.root = root


        self.local_plot = LocalPlot()
        self.fig = self.local_plot.fig
        self.ax1 = self.local_plot.ax1
        self.ax2 = self.local_plot.ax2

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)  # A tk.DrawingArea.
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect("scroll_event", self.on_mouse_scroll)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_click)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self._prev_scroll_time = None  # used to throttle the replot

        self.left_mouse_button_pressed = False
        self.teleport_x = 0.0
        self.teleport_y = 0.0
        self.teleport_s = 0.0
        self.teleport_d = 0.0
        self.teleport_orientation = 0.0

        self.right_mouse_button_pressed = False
        self.spawn_in_ax1 = True
        self.spawn_x = 0.0
        self.spawn_y = 0.0
        self.spawn_s = 0.0
        self.spawn_d = 0.0
        self.spawn_orientation = 0.0
        self._spawn_agent_state = None


    def reset(self):
        self.local_plot.reset()


    def on_mouse_move(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            if event.inaxes == self.ax1:
                self.root.setting.perception_status_text.set(f"Spawn {x:.1f},{y:.1f}")
                if self.left_mouse_button_pressed:
                    self.teleport_orientation = np.arctan2(y - self.teleport_y, x - self.teleport_x)
                    self.local_plot.show_vehicle_orientation_ax1(self.teleport_x, self.teleport_y, self.teleport_orientation) 
                    self.root.teleport_ego(x=self.teleport_x, y=self.teleport_y, theta=self.teleport_orientation)
                    self.root.exec.local_planner.step(state=self.root.exec.world.get_ego_state())       
                    self.root.update_ui()
                elif self.right_mouse_button_pressed and self.spawn_in_ax1:
                    self.spawn_orientation = np.arctan2(y - self.spawn_y, x - self.spawn_x)
                    if self._spawn_agent_state is not None:
                        self._spawn_agent_state.theta = self.spawn_orientation
                    self.local_plot.show_vehicle_orientation_ax1(
                        self.spawn_x, self.spawn_y, self.spawn_orientation, color="blue"
                    )
                    self.root.update_ui()
            elif event.inaxes == self.ax2:
                if self.left_mouse_button_pressed:
                    teleport_orientation = np.arctan2(y - self.teleport_d, x - self.teleport_s)
                    self.local_plot.show_vehicle_orientation_ax2(s=self.teleport_s, d=self.teleport_d, theta=teleport_orientation) 
                    # Use the click pose (teleport_s/d), not the drag-tip cursor (x,y),
                    # so world heading is path-tangent at the ego, not at the mouse tip.
                    _, _, theta = self.root.exec.local_planner.global_plan.trajectory.convert_sd_orientation_to_xy_orientation(
                        self.teleport_s, self.teleport_d, teleport_orientation
                    )
                    self.root.teleport_ego(self.teleport_x, self.teleport_y, theta)
                    self.root.exec.local_planner.step(state=self.root.exec.world.get_ego_state())       
                    self.root.update_ui()
                elif self.right_mouse_button_pressed and not self.spawn_in_ax1:
                    plot_theta = np.arctan2(y - self.spawn_d, x - self.spawn_s)
                    self.local_plot.show_vehicle_orientation_ax2(
                        s=self.spawn_s, d=self.spawn_d, theta=plot_theta, color="blue"
                    )
                    _, _, self.spawn_orientation = (
                        self.root.exec.local_planner.global_plan.trajectory
                        .convert_sd_orientation_to_xy_orientation(self.spawn_s, self.spawn_d, plot_theta)
                    )
                    if self._spawn_agent_state is not None:
                        self._spawn_agent_state.theta = self.spawn_orientation
                    self.root.update_ui()
                self.root.setting.perception_status_text.set(f"Spawn S:{x:.1f},D:{y:.1f}")
        else:
            self.root.setting.perception_status_text.set("Click plot")

        # Paused-only distance ruler from ego front center to cursor.
        if (
            self.root.setting.exec_running
            or self.left_mouse_button_pressed
            or self.right_mouse_button_pressed
            or event.inaxes not in (self.ax1, self.ax2)
            or self.root.exec is None
        ):
            self.local_plot.hide_distance_ruler()
            return
        ego = self.root.exec.ego_state
        ctrl = self.root.exec.controller
        L_f = float(getattr(ctrl, "ego_distance_front_axle", 2.5) if ctrl is not None else 2.5)
        fx = float(ego.x) + L_f * float(np.cos(ego.theta))
        fy = float(ego.y) + L_f * float(np.sin(ego.theta))
        mx, my = float(event.xdata), float(event.ydata)
        if event.inaxes == self.ax1:
            dist = float(np.hypot(mx - fx, my - fy))
            self.local_plot.show_distance_ruler(self.ax1, fx, fy, mx, my, dist)
            return
        # Frenet: draw in (s, d); distance in world XY meters.
        traj = getattr(getattr(self.root.exec, "local_planner", None), "global_trajectory", None)
        if traj is None:
            self.local_plot.hide_distance_ruler()
            return
        fs, fd = traj.convert_xy_to_sd(fx, fy)
        cx, cy = traj.convert_sd_to_xy(mx, my)
        dist = float(np.hypot(cx - fx, cy - fy))
        self.local_plot.show_distance_ruler(self.ax2, fs, fd, mx, my, dist)

    def on_mouse_click(self, event):
        if event.button == 3:
            if event.inaxes == self.ax1:
                x, y = event.xdata, event.ydata
                self.right_mouse_button_pressed = True
                self.spawn_in_ax1 = True
                self.spawn_x = x
                self.spawn_y = y
                self.spawn_orientation = self.root.exec.ego_state.theta
                self._spawn_agent_state = self.__spawn_agent(
                    x=x, y=y, theta=self.spawn_orientation
                )
                self.local_plot.show_vehicle_orientation_ax1(
                    x, y, self.spawn_orientation, color="blue"
                )
                self.root.update_ui()
            elif event.inaxes == self.ax2:
                s, d = event.xdata, event.ydata
                traj = self.root.exec.local_planner.global_plan.trajectory
                x, y = traj.convert_sd_to_xy(s, d)
                self.right_mouse_button_pressed = True
                self.spawn_in_ax1 = False
                self.spawn_s = s
                self.spawn_d = d
                self.spawn_x = x
                self.spawn_y = y
                self.spawn_orientation = self.root.exec.ego_state.theta
                self._spawn_agent_state = self.__spawn_agent(
                    s=s, d=d, theta=self.spawn_orientation
                )
                _, _, path_heading = traj.convert_sd_orientation_to_xy_orientation(s, d, 0.0)
                self.local_plot.show_vehicle_orientation_ax2(
                    s=s, d=d, theta=self.spawn_orientation - path_heading, color="blue"
                )
                self.root.update_ui()

        elif event.button == 1:
            if event.inaxes == self.ax1:
                x, y = event.xdata, event.ydata
                self.root.teleport_ego(x=x, y=y)
                self.left_mouse_button_pressed = True
                self.teleport_x = x
                self.teleport_y = y
            elif event.inaxes == self.ax2:
                s, d = event.xdata, event.ydata
                x,y = self.root.exec.local_planner.global_plan.trajectory.convert_sd_to_xy(s,d)
                self.root.teleport_ego(x,y)
                self.left_mouse_button_pressed = True
                self.teleport_s = s
                self.teleport_d = d
                self.teleport_x = x
                self.teleport_y = y

            self.root.exec.local_planner.step(state=self.root.exec.world.get_ego_state())       
            self.root.update_ui()
    
    def __spawn_agent(self, x=None, y=None, s=None, d=None, theta=None):
        if x is not None and y is not None:
            t = self.root.exec.ego_state.theta if theta is None else theta
            agent = AgentState(x=x, y=y, theta=t, velocity=0)
            self.root.spawn_agent(agent)
        elif s is not None and d is not None:
            # Convert (s, d) to (x, y) using some transformation logic
            x, y = self.root.exec.local_planner.global_trajectory.convert_sd_to_xy(s, d)
            log.info(f"Spawning agent at (x, y) = ({x}, {y}) from (s, d) = ({s}, {d})")
            t = self.root.exec.ego_state.theta if theta is None else theta
            agent = AgentState(x=x, y=y, theta=t, velocity=0)
            self.root.spawn_agent(agent)
        else:
            raise ValueError("Either (x, y) or (s, d) must be provided")
        return agent

    def on_mouse_release(self, event):
        # if event.inaxes == self.ax1:
        if event.button == 1:
            self.left_mouse_button_pressed = False
            self.local_plot.clear_tmp_plots()
            self.root.exec.controller.reset()
            self.root.update_ui()
            log.debug(f"Teleport Ego to X: {self.teleport_x:.2f}, Y: {self.teleport_y:.2f}, Orientation: {self.teleport_orientation:.2f}")
        elif event.button == 3 and self.right_mouse_button_pressed:
            self.right_mouse_button_pressed = False
            self._spawn_agent_state = None
            self.local_plot.clear_tmp_plots()
            self.root.update_ui()
            log.debug(
                f"Spawn Agent at X: {self.spawn_x:.2f}, Y: {self.spawn_y:.2f}, "
                f"Orientation: {self.spawn_orientation:.2f}"
            )


    def on_mouse_scroll(self, event, increment=10):
        if event.inaxes == self.ax1:
            log.debug(f"Scroll Event in real coordinate: {event.button}")
            if event.button == "up":
                self.root.setting.p66_xy_zoom -= increment if self.root.setting.p66_xy_zoom > increment else 0
            elif event.button == "down":
                self.root.setting.p66_xy_zoom += increment
        elif event.inaxes == self.ax2:
            log.debug(f"Scroll Event in frenet: {event.button}")
            if event.button == "up":
                self.root.setting.p66_frenet_zoom -= increment if self.root.setting.p66_frenet_zoom > increment else 0
            elif event.button == "down":
                self.root.setting.p66_frenet_zoom += increment

        threshold = 0.01
        if (
            self._prev_scroll_time is None or time.time() - self._prev_scroll_time > threshold
        ) and not self.root.setting.exec_running:
            self.root.update_ui()

        self._prev_scroll_time = time.time()

    def zoom_in(self):
        self.root.setting.p66_xy_zoom -= 5 if self.root.setting.p66_xy_zoom > 5 else 0
        self.root.update_ui()

    def zoom_out(self):
        self.root.setting.p66_xy_zoom += 5
        self.root.update_ui()

    def zoom_in_frenet(self):
        self.root.setting.p66_frenet_zoom -= 5 if self.root.setting.p66_frenet_zoom > 5 else 0
        self.root.update_ui()

    def zoom_out_frenet(self):
        self.root.setting.p66_frenet_zoom += 5
        self.root.update_ui()

    def update_plot_theme(self):
        self.local_plot.set_plot_theme(self.root.setting.p60_bg_color, self.root.setting.p60_fg_color)

    def plot(self):
        """Plot the local plan and update the canvas."""
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
        aspect_ratio = width / height
        # ylim formulas assume two stacked axes each occupying half the height;
        # when only one axis is shown at full height, halving the ratio doubles
        # the y-range so the content fills the available space.
        show_gv = self.root.setting.p67_show_local_global_view.get()
        show_fv = self.root.setting.p67_show_local_frenet_view.get()
        if show_gv != show_fv:
            aspect_ratio /= 2

        bridge_lidar = is_world_capability_enabled(WorldCapability.LIDAR_2D) or is_world_capability_enabled(WorldCapability.LIDAR_3D)
        want_lidar = bridge_lidar and (
            self.root.setting.p66_show_lidar_global.get()
            or self.root.setting.p66_show_lidar_frenet.get()
        )

        t1 = time.time()
        # self.canvas.restore_region(self.plt_background)
        self.local_plot.plot(
            exec=self.root.exec,
            aspect_ratio=aspect_ratio,
            xy_zoom=self.root.setting.p66_xy_zoom,
            frenet_zoom=self.root.setting.p66_frenet_zoom,
            show_legend=self.root.setting.p66_show_legend.get(),
            plot_last_pts=self.root.setting.p66_show_past_locations.get(),
            plot_global_plan=self.root.setting.p66_show_global_plan.get(),
            plot_local_plan=self.root.setting.p66_show_local_plan.get(),
            plot_local_lattice=self.root.setting.p66_show_local_lattice.get(),
            plot_state=self.root.setting.p66_show_state.get(),
            global_follow_planner=self.root.setting.p66_global_view_follow_planner.get(),
            frenet_follow_planner=self.root.setting.p66_frenet_view_follow_planner.get(),
            plot_occupancy_flow=self.root.setting.p67_show_occupancy_flow.get(),
            plot_predictions=self.root.setting.p67_show_prediction.get(),
            plot_lidar=want_lidar,
            lidar_data=self.root.exec.world.get_lidar_data() if want_lidar else None,
            plot_lidar_global=self.root.setting.p66_show_lidar_global.get(),
            plot_lidar_frenet=self.root.setting.p66_show_lidar_frenet.get(),
            plot_clusters=bridge_lidar and self.root.setting.p66_show_lidar_clusters.get(),
            plot_ground_truth=is_world_stack_capability_enabled(StackCapability.DETECTION),
            plot_race_boundary=self.root.setting.p66_show_race_boundary.get(),
            show_global_view=self.root.setting.p67_show_local_global_view.get(),
            show_frenet_view=self.root.setting.p67_show_local_frenet_view.get(),
        )
        # Always full_draw: TkAgg blit update drops animated scatter/markers on the 2nd frame.
        self.local_plot.blit_manager.full_draw()
        log.debug(f"Local Plot Time: {(time.time()-t1)*1000:.2f} ms (aspect_ratio: {aspect_ratio:0.2f})")


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
