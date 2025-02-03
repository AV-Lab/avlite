from __future__ import annotations
from c50_visualize.c57_plot_lib import PlotLib

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import logging
log = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c50_visualize.c51_visualizer_app import VisualizerApp

class PlotView(ttk.Frame):

    def __init__(self, root: VisualizerApp):
        super().__init__(root)
        self.root = root


        self.plot_lib = PlotLib()
        self.fig = self.plot_lib.fig
        self.ax1 = self.plot_lib.ax1
        self.ax2 = self.plot_lib.ax2

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
                self.root.data.xy_zoom -= increment if self.root.data.xy_zoom > increment else 0
            elif event.button == "down":
                self.root.data.xy_zoom += increment
        elif event.inaxes == self.ax2:
            log.debug(f"Scroll Event in frenet: {event.button}")
            if event.button == "up":
                self.root.data.frenet_zoom -= increment if self.root.data.frenet_zoom > increment else 0
            elif event.button == "down":
                self.root.data.frenet_zoom += increment

        threshold = 0.01
        if (
            self._prev_scroll_time is None or time.time() - self._prev_scroll_time > threshold
        ) and not self.root.data.exec_running:
            self.root.update_ui()

        self._prev_scroll_time = time.time()

    def zoom_in(self):
        self.root.data.xy_zoom -= 5 if self.root.data.xy_zoom > 5 else 0
        self.root.update_ui()

    def zoom_out(self):
        self.root.data.xy_zoom += 5
        self.root.update_ui()

    def zoom_in_frenet(self):
        self.root.data.frenet_zoom -= 5 if self.root.data.frenet_zoom > 5 else 0
        self.root.update_ui()

    def zoom_out_frenet(self):
        self.root.data.frenet_zoom += 5
        self.root.update_ui()

    def set_plot_theme(self, bg_color="white", fg_color="black"):
        self.plot_lib.set_plot_theme(bg_color, fg_color)

    def plot(self):
        canvas_widget = self.canvas.get_tk_widget()
        width = canvas_widget.winfo_width()
        height = canvas_widget.winfo_height()
        aspect_ratio = width / height

        t1 = time.time()
        # self.canvas.restore_region(self.plt_background)
        self.plot_lib.plot(
            exec=self.root.exec,
            aspect_ratio=aspect_ratio,
            xy_zoom=self.root.data.xy_zoom,
            frenet_zoom=self.root.data.frenet_zoom,
            show_legend=self.root.data.show_legend.get(),
            plot_last_pts=self.root.data.show_past_locations.get(),
            plot_global_plan=self.root.data.show_global_plan.get(),
            plot_local_plan=self.root.data.show_local_plan.get(),
            plot_local_lattice=self.root.data.show_local_lattice.get(),
            plot_state=self.root.data.show_state.get(),
            global_follow_planner=self.root.data.global_view_follow_planner.get(),
            frenet_follow_planner=self.root.data.frenet_view_follow_planner.get(),
        )
        self.canvas.draw()
        log.debug(f"Plot Time: {(time.time()-t1)*1000:.2f} ms (aspect_ratio: {aspect_ratio:0.2f})")
