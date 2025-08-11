from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import logging
import numpy as np

from c10_perception.c11_perception_model import AgentState
from c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from c20_planning.c24_global_planners import RaceGlobalPlanner
from c20_planning.c24_global_planners import HDMapGlobalPlanner
from c50_visualization.c57_plot_lib import LocalPlot, GlobalRacePlot, GlobalHDMapPlot

log = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c50_visualization.c51_visualizer_app import VisualizerApp

class GlobalPlanPlotView(ttk.Frame):
    def __init__(self, root: VisualizerApp):
        super().__init__(root)
        self.root = root

        if self.root.setting.global_planner_type.get() == RaceGlobalPlanner.__name__:
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
        

    def __config_canvas(self):  
        self.fig = self.global_plot.fig
        self.ax = self.global_plot.ax   

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.global_plot.set_plot_theme(self.root.setting.bg_color, self.root.setting.fg_color)
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

    def plot(self):
        t1 = time.time()
        try:
            aspect_ratio = self.__get_aspect_ratio()

            self.global_plot.plot(
                exec=self.root.exec,
                aspect_ratio=aspect_ratio,
                zoom=self.root.setting.global_zoom,
                show_legend=self.root.setting.show_legend.get(),
                follow_vehicle=self.root.setting.global_view_follow_planner.get(),
                delta=self._center_delta,
            )
            log.debug(f"Global Plot Time: {(time.time()-t1)*1000:.2f} ms (aspect_ratio: {aspect_ratio:0.2f})")
        except Exception as e:
            log.error(f"Error in Global Plot: {e}")
        
    
    def update_plot_theme(self):
        self.global_plot.set_plot_theme(self.root.setting.bg_color, self.root.setting.fg_color)
        

    def update_plot_type(self):
        """Update the plot type based on the selected global planner"""

        log.debug(f"Updating Global Plot type {self.root.setting.global_planner_type.get()}...")
        self.canvas.get_tk_widget().destroy()  
        assert self.root.setting.global_planner_type.get() in GlobalPlannerStrategy.registry.keys(), \
                f"Global planner type {self.root.setting.global_planner_type.get()} not found in registry."

        if self.root.setting.global_planner_type.get() == RaceGlobalPlanner.__name__:
            self.global_plot = GlobalRacePlot()
            log.debug("Global Plot type changed to Race Plot.")
        elif self.root.setting.global_planner_type.get() == HDMapGlobalPlanner.__name__:
            self.global_plot = GlobalHDMapPlot()
            log.debug("Global Plot type changed to HD Map Plot.")
        else:
            raise ValueError(f"Global planner type {self.root.setting.global_planner_type.get()} not found in registry.")

        self.__config_canvas()


    def on_mouse_move(self, event):
        try:
            if event.inaxes == self.ax:
                x, y = event.xdata, event.ydata

                self.root.setting.perception_status_text.set(f"Teleport Ego: X: {x:.2f}, Y: {y:.2f}")

                if self.root.setting.global_planner_type.get() == HDMapGlobalPlanner.__name__:
                    if not self.left_mouse_button_pressed:
                        self.global_plot.show_closest_road_and_lane(x=int(x), y=int(y), map=self.root.exec.global_planner.hdmap)   
                
                if event.key and 'control' in event.key and self._drag_mode:
                    dx =-(x - self._init_drag_mouse_pos[0])*self.root.setting.mouse_drag_slowdown_factor
                    dy =-(y - self._init_drag_mouse_pos[1])*self.root.setting.mouse_drag_slowdown_factor
                    self._center_delta = (self._center_delta[0]+dx, self._center_delta[1]+dy)
                    self.plot()

                if self.left_mouse_button_pressed and not self._drag_mode:
                    self.teleport_orientation = np.arctan2(y - self.teleport_y, x - self.teleport_x)
                    self.global_plot.show_vehicle_orientation(self.teleport_x, self.teleport_y, self.teleport_orientation) 
                    self.root.exec.world.teleport_ego(x=self.teleport_x, y=self.teleport_y, theta=self.teleport_orientation)
                    self.root.exec.local_planner.step(state=self.root.exec.world.get_ego_state())       
                    self.root.update_ui()
            else:
                self.root.setting.perception_status_text.set("Click on the plot.")
                self.global_plot.clear_tmp_plots()
                self._drag_mode = False
                self._init_drag_mouse_pos = None
        except Exception as e:
            log.error(f"Error in mouse move event: {e}", exc_info=True)


    def on_mouse_click(self, event):
        if event.inaxes == self.ax:
            if event.button == 1:  # Left click
                self.left_mouse_button_pressed = True
                if event.key and 'control' in event.key: # for dragging
                    self._init_drag_mouse_pos = (event.xdata, event.ydata)
                    self._drag_mode = True
                    self.global_plot.clear_tmp_plots()
                else:
                    x, y = event.xdata, event.ydata
                    self.root.exec.world.teleport_ego(x=x, y=y)
                    self.teleport_x = x
                    self.teleport_y = y
                    self.global_plot.clear_tmp_plots()

                    self.root.update_ui()
            elif event.button == 3: # Right click
                if self.start_point:
                    self.global_plot.set_goal(event.xdata, event.ydata)
                    self.root.exec.global_planner.set_start_goal(start_point=self.start_point, goal_point=(event.xdata, event.ydata))
                    log.info(f"Set start: {self.start_point}, goal: {(event.xdata, event.ydata)}")

                    self.root.exec.global_planner.plan()
                    if len(self.root.exec.global_planner.global_plan.path) == 0:
                        log.warning("No global plan found. Please check the start and goal points.")
                        return

                    if self.root.setting.global_planner_type.get() == HDMapGlobalPlanner.__name__:
                        self.global_plot.plot_global_plan(self.root.exec.global_planner.global_plan)
                        self.root.exec.local_planner.set_global_plan(self.root.exec.global_planner.global_plan)
                        self.root.exec.controller.reset()
                        self.root.local_plan_plot_view.reset()
                        self.root.update_ui()
                        self.root.exec.controller.reset()

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
                self.left_mouse_button_pressed = False
                self._drag_mode = False
                self._init_drag_mouse_pos = None
                self.global_plot.clear_tmp_plots()
                self.root.exec.controller.reset()
                self.root.update_ui()
                log.debug(f"Teleport Ego to X: {self.teleport_x:.2f}, Y: {self.teleport_y:.2f}, Orientation: {self.teleport_orientation:.2f}")
    
    def on_mouse_scroll(self, event, increment=10):
        log.debug(f"Scroll Event in global coordinate. Zoom: {self.root.setting.global_zoom}")
        if event.button == "up":
            self.root.setting.global_zoom -= increment if self.root.setting.global_zoom > increment else 0
        elif event.button == "down":
            self.root.setting.global_zoom += increment
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


        self.local_plot = LocalPlot(max_lattice_size=self.root.exec.local_planner.lattice.targetted_num_edges)
        self.fig = self.local_plot.fig
        self.ax1 = self.local_plot.ax1
        self.ax2 = self.local_plot.ax2

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        root.after(300, self.root.update_ui)

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
        

    def reset(self):
        self.local_plot.reset()


    def on_mouse_move(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            if event.inaxes == self.ax1:
                self.root.setting.perception_status_text.set(f"Spawn Agent: X: {x:.2f}, Y: {y:.2f}")
                if self.left_mouse_button_pressed:
                    self.teleport_orientation = np.arctan2(y - self.teleport_y, x - self.teleport_x)
                    self.local_plot.show_vehicle_orientation_ax1(self.teleport_x, self.teleport_y, self.teleport_orientation) 
                    self.root.exec.world.teleport_ego(x=self.teleport_x, y=self.teleport_y, theta=self.teleport_orientation)
                    self.root.exec.local_planner.step(state=self.root.exec.world.get_ego_state())       
                    self.root.update_ui()
            elif event.inaxes == self.ax2:
                if self.left_mouse_button_pressed:
                    teleport_orientation = np.arctan2(y - self.teleport_d, x - self.teleport_s)
                    self.local_plot.show_vehicle_orientation_ax2(s=self.teleport_s, d=self.teleport_d, theta=teleport_orientation) 
                    x_,y_, theta = self.root.exec.local_planner.global_plan.trajectory.convert_sd_orientation_to_xy_orientation(x,y,teleport_orientation)
                    self.root.exec.world.teleport_ego(self.teleport_x, self.teleport_y,theta)
                    self.root.exec.local_planner.step(state=self.root.exec.world.get_ego_state())       
                    self.root.update_ui()
                self.root.setting.perception_status_text.set(f"Spawn Agent: S: {x:.2f}, D: {y:.2f}")
        else:
            self.root.setting.perception_status_text.set("Click on the plot.")

    def on_mouse_click(self, event):
        if event.button == 3:
            if event.inaxes == self.ax1:
                x, y = event.xdata, event.ydata
                self.__spawn_agent(x=x, y=y)
                self.root.update_ui()
            elif event.inaxes == self.ax2:
                s, d = event.xdata, event.ydata
                self.__spawn_agent(d=d, s=s)
                self.root.update_ui()

        elif event.button == 1:
            if event.inaxes == self.ax1:
                x, y = event.xdata, event.ydata
                self.root.exec.world.teleport_ego(x=x, y=y)
                self.left_mouse_button_pressed = True
                self.teleport_x = x
                self.teleport_y = y
            elif event.inaxes == self.ax2:
                s, d = event.xdata, event.ydata
                x,y = self.root.exec.local_planner.global_plan.trajectory.convert_sd_to_xy(s,d)
                self.root.exec.world.teleport_ego(x,y)
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
            self.root.exec.world.spawn_agent(agent)
        elif s is not None and d is not None:
            # Convert (s, d) to (x, y) using some transformation logic
            x, y = self.root.exec.local_planner.global_trajectory.convert_sd_to_xy(s, d)
            log.info(f"Spawning agent at (x, y) = ({x}, {y}) from (s, d) = ({s}, {d})")
            t = self.root.exec.ego_state.theta if theta is None else theta
            agent = AgentState(x=x, y=y, theta=t, velocity=0)
            self.root.exec.world.spawn_agent(agent)
        else:
            raise ValueError("Either (x, y) or (s, d) must be provided")

        # self.pm.add_agent_vehicle(agent)
    
    def on_mouse_release(self, event):
        # if event.inaxes == self.ax1:
        if event.button == 1:
            self.left_mouse_button_pressed = False
            self.local_plot.clear_tmp_plots()
            self.root.exec.controller.reset()
            self.root.update_ui()
            log.debug(f"Teleport Ego to X: {self.teleport_x:.2f}, Y: {self.teleport_y:.2f}, Orientation: {self.teleport_orientation:.2f}")


    def on_mouse_scroll(self, event, increment=10):
        if event.inaxes == self.ax1:
            log.debug(f"Scroll Event in real coordinate: {event.button}")
            if event.button == "up":
                self.root.setting.xy_zoom -= increment if self.root.setting.xy_zoom > increment else 0
            elif event.button == "down":
                self.root.setting.xy_zoom += increment
        elif event.inaxes == self.ax2:
            log.debug(f"Scroll Event in frenet: {event.button}")
            if event.button == "up":
                self.root.setting.frenet_zoom -= increment if self.root.setting.frenet_zoom > increment else 0
            elif event.button == "down":
                self.root.setting.frenet_zoom += increment

        threshold = 0.01
        if (
            self._prev_scroll_time is None or time.time() - self._prev_scroll_time > threshold
        ) and not self.root.setting.exec_running:
            self.root.update_ui()

        self._prev_scroll_time = time.time()

    def zoom_in(self):
        self.root.setting.xy_zoom -= 5 if self.root.setting.xy_zoom > 5 else 0
        self.root.update_ui()

    def zoom_out(self):
        self.root.setting.xy_zoom += 5
        self.root.update_ui()

    def zoom_in_frenet(self):
        self.root.setting.frenet_zoom -= 5 if self.root.setting.frenet_zoom > 5 else 0
        self.root.update_ui()

    def zoom_out_frenet(self):
        self.root.setting.frenet_zoom += 5
        self.root.update_ui()

    def update_plot_theme(self):
        self.local_plot.set_plot_theme(self.root.setting.bg_color, self.root.setting.fg_color)

    def plot(self):
        """Plot the local plan and update the canvas."""
        canvas_widget = self.canvas.get_tk_widget()
        width = canvas_widget.winfo_width()
        height = canvas_widget.winfo_height()
        aspect_ratio = width / height

        t1 = time.time()
        # self.canvas.restore_region(self.plt_background)
        self.local_plot.plot(
            exec=self.root.exec,
            aspect_ratio=aspect_ratio,
            xy_zoom=self.root.setting.xy_zoom,
            frenet_zoom=self.root.setting.frenet_zoom,
            show_legend=self.root.setting.show_legend.get(),
            plot_last_pts=self.root.setting.show_past_locations.get(),
            plot_global_plan=self.root.setting.show_global_plan.get(),
            plot_local_plan=self.root.setting.show_local_plan.get(),
            plot_local_lattice=self.root.setting.show_local_lattice.get(),
            plot_state=self.root.setting.show_state.get(),
            global_follow_planner=self.root.setting.global_view_follow_planner.get(),
            frenet_follow_planner=self.root.setting.frenet_view_follow_planner.get(),
        )
        self.canvas.draw()
        log.debug(f"Local Plot Time: {(time.time()-t1)*1000:.2f} ms (aspect_ratio: {aspect_ratio:0.2f})")
