from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import logging

from c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from c20_planning.c24_race_global_planner import RaceGlobalPlanner
from c20_planning.c25_hdmap_global_planner import HDMapGlobalPlanner
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
        self.bind("<Configure>",lambda x: self.plot())

        self.start_point = None
        self._prev_scroll_time = None  # used to throttle the replot

        self.initialized = False
        
        self.current_highlighted_road_id = "-1" # used to track the current road id
         
    def __config_canvas(self):  
        self.fig = self.global_plot.fig
        self.ax = self.global_plot.ax   

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.global_plot.set_plot_theme(self.root.setting.bg_color, self.root.setting.fg_color)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_click)
        self.canvas.mpl_connect("scroll_event", self.on_mouse_scroll)
        

    def plot(self):
        t1 = time.time()
        try:
            canvas_widget = self.canvas.get_tk_widget()
            width = canvas_widget.winfo_width()
            height = canvas_widget.winfo_height()
            aspect_ratio = width / height if height > 0 else 4.0

            self.global_plot.plot(
                exec=self.root.exec,
                aspect_ratio=aspect_ratio,
                zoom=self.root.setting.global_zoom,
                show_legend=self.root.setting.show_legend.get(),
                follow_vehicle=self.root.setting.global_view_follow_planner.get()
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
            if event.inaxes:
                x, y = event.xdata, event.ydata
                if event.inaxes == self.ax:
                    self.root.setting.perception_status_text.set(f"Teleport Ego: X: {x:.2f}, Y: {y:.2f}")

                    if self.root.setting.global_planner_type.get() == HDMapGlobalPlanner.__name__:
                        r  = self.root.exec.global_planner.hdmap.find_nearest_road(x=x, y=y)
                        l = self.root.exec.global_planner.hdmap.find_nearest_lane(x=x, y=y)
                        if r is not None: # and r.id != self.current_highlighted_road_id:
                            self.global_plot.show_closest_road_and_lane(x=int(x), y=int(y), map=self.root.exec.global_planner.hdmap)   
                            # log.debug(f"road id: {r.id:4s}             | pred_id: {r.pred_id:4s} ({r.pred_type:^5.5}) | succ_id: {r.succ_id:4s} ({r.succ_type:.5})")
                            # log.debug(f"lane id: {l.id:4s} (road {l.road_id:4s}) | pred_id: {l.pred_id:4s}         | succ_id: {l.succ_id:4s} | lane_type: {l.type:.5}")
                            self.current_highlighted_road_id = r.id
            else:
                self.root.setting.perception_status_text.set("Click on the plot.")
                self.global_plot.clear_tmp_plots()
        except Exception as e:
            log.error(f"Error in mouse move event: {e}", exc_info=True)


    def on_mouse_click(self, event):
        if event.inaxes == self.ax:
            if event.button == 1:  # Left click
                x, y = event.xdata, event.ydata
                self.root.exec.world.teleport_ego(x=x, y=y)
                self.root.update_ui()
            elif event.button == 3: # Right click
                if self.start_point:
                    self.global_plot.set_goal(event.xdata, event.ydata)
                    self.root.exec.global_planner.set_start_goal(start_point=self.start_point, goal_point=(event.xdata, event.ydata))
                    log.info(f"Set start: {self.start_point}, goal: {(event.xdata, event.ydata)}")
                    self.root.exec.global_planner.plan()

                    if self.root.setting.global_planner_type.get() == HDMapGlobalPlanner.__name__:
                        self.global_plot.plot_road_path(self.root.exec.global_planner.road_path)

                    self.start_point = None
                else:
                    self.global_plot.set_start(event.xdata, event.ydata)
                    self.global_plot.clear_goal()
                    self.global_plot.clear_road_path_plots()
                    self.pending_goal_set = True
                    self.start_point = (event.xdata, event.ydata)
    
    def on_mouse_scroll(self, event, increment=10):
        log.debug(f"Scroll Event in global coordinate. Zoom: {self.root.setting.global_zoom}")
        if event.button == "up":
            self.root.setting.global_zoom -= increment if self.root.setting.global_zoom > increment else 0
        elif event.button == "down":
            self.root.setting.global_zoom += increment
        threshold = 0.01
        if (
            self._prev_scroll_time is None or time.time() - self._prev_scroll_time > threshold
        ) and not self.root.setting.exec_running:
            self.root.update_ui()

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
        self._prev_scroll_time = None  # used to throttle the replot
        
        self.bind("<Configure>",lambda x: self.plot())

    def on_mouse_move(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            if event.inaxes == self.ax1:
                self.root.setting.perception_status_text.set(f"Spawn Agent: X: {x:.2f}, Y: {y:.2f}")
            elif event.inaxes == self.ax2:
                self.root.setting.perception_status_text.set(f"Spawn Agent: S: {x:.2f}, D: {y:.2f}")
        else:
            self.root.setting.perception_status_text.set("Click on the plot.")

    def on_mouse_click(self, event):
        if event.button == 1:
            if event.inaxes == self.ax1:
                x, y = event.xdata, event.ydata
                self.root.exec.spawn_agent(x=x, y=y)
                self.root.update_ui()
            elif event.inaxes == self.ax2:
                s, d = event.xdata, event.ydata
                self.root.exec.spawn_agent(d=d, s=s)
                self.root.update_ui()

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
