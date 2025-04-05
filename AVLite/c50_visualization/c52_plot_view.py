from __future__ import annotations

from c20_planning.c23_race_global_planner import RaceGlobalPlanner
from c20_planning.c22_hdmap_global_planner import GlobalHDMapPlanner
from c20_planning.c21_base_global_planner import PlannerType
from c50_visualization.c57_plot_lib import LocalPlot, GlobalRacePlot, GlobalHDMapPlot

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import logging
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
        elif self.root.setting.global_planner_type.get() == GlobalHDMapPlanner.__name__:
            self.global_plot = GlobalHDMapPlot()

        self.fig = self.global_plot.fig

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        # Pack the canvas widget to make it visible
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Apply current theme
        if self.root.setting.dark_mode.get():
            bg_color = "#333333" if self.root.setting.dark_mode.get() else "white"
            fg_color = "white" if self.root.setting.dark_mode.get() else "black"
            self.set_plot_theme(bg_color, fg_color)
            
        self.plot()  # Initialize the view


    def plot(self):
        t1 = time.time()
        canvas_widget = self.canvas.get_tk_widget()
        width = canvas_widget.winfo_width()
        height = canvas_widget.winfo_height()
        aspect_ratio = width / height if height > 0 else 4.0

        # log.info(f"Global planner type: {self.root.exec.global_planner.__class__.__name__}")

        
        self.global_plot.plot(
            exec=self.root.exec,
            aspect_ratio=aspect_ratio,
            zoom=self.root.setting.xy_zoom,
            show_legend=self.root.setting.show_legend.get(),
            follow_vehicle=self.root.setting.global_view_follow_planner.get()
        )
        
        log.debug(f"Global Plot Time: {(time.time()-t1)*1000:.2f} ms (aspect_ratio: {aspect_ratio:0.2f})")
        
    def set_plot_theme(self, bg_color="white", fg_color="black"):
        """Apply theme to the global plot"""
        self.global_plot.set_plot_theme(bg_color, fg_color)
        # Force a complete redraw after theme change
        self.plot()

    def update_plot_type(self):
        """Update the plot type based on the selected global planner"""
        if self.root.setting.global_planner_type.get() == PlannerType.RACE_PLANNER.value:
            self.global_plot = GlobalRacePlot()
            log.debug("Global Plot type changed to Race Plot.")
        elif self.root.setting.global_planner_type.get() == PlannerType.HD_MAP_PLANNER.value:
            self.global_plot = GlobalHDMapPlot(self.root.exec)
            log.debug("Global Plot type changed to HD Map Plot.")
        self.plot()

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

    def on_mouse_move(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            if event.inaxes == self.ax1:
                self.root.perceive_plan_control_view.coordinates_label.config(
                    text=f"Spawn Agent: X: {x:.2f}, Y: {y:.2f}"
                )
            elif event.inaxes == self.ax2:
                self.root.perceive_plan_control_view.coordinates_label.config(
                    text=f"Spawn Agent: S: {x:.2f}, D: {y:.2f}"
                )
        else:
            # Optionally, clear the coordinates display when the mouse is not over the axes
            self.root.perceive_plan_control_view.coordinates_label.config(
                text="Spawn Agent: Click on the plot."
            )

    def on_mouse_click(self, event):
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

    def set_plot_theme(self, bg_color="white", fg_color="black"):
        # Ensure we're using the same background color as GlobalPlanPlotView
        if bg_color == "#000000":
            bg_color = "#333333"  # Use dark gray instead of pure black
        self.local_plot.set_plot_theme(bg_color, fg_color)

    def plot(self):
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
