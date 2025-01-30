from __future__ import annotations
import c50_visualize.c52_plotlib as c52_plotlib


import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import logging

log = logging.getLogger(__name__)


class PlotView(tk.Frame):

    def __init__(self, root: VisualizerApp):
        super().__init__(root)
        # self.plot_frame = ttk.Frame(root)
        # self.plot_frame.pack(fill=tk.BOTH, expand=True)
        # self.plot_frame.grid(row=0, column=0, sticky="nsew")

        self.xy_zoom = 30
        self.frenet_zoom = 30

        self.fig = c52_plotlib.fig
        self.ax1 = c52_plotlib.ax1
        self.ax2 = c52_plotlib.ax2
        self.set_plot_theme = c52_plotlib.set_plot_theme

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        root.after(300, self.replot)

        self.canvas.mpl_connect("scroll_event", self.on_mouse_scroll)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_click)
        self._prev_scroll_time = None  # used to throttle the replot
        self.root = root
        # c52_plotlib.initialize_plots()

    def on_mouse_move(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            if event.inaxes == self.ax1:
                self.root.perceive_plan_control_view.coordinates_label.config(text=f"Spawn Agent: X: {x:.2f}, Y: {y:.2f}")
            elif event.inaxes == self.ax2:
                self.root.perceive_plan_control_view.coordinates_label.config(text=f"Spawn Agent: S: {x:.2f}, D: {y:.2f}")
        else:
            # Optionally, clear the coordinates display when the mouse is not over the axes
            self.root.perceive_plan_control_view.coordinates_label.config(text="Spawn Agent: Click on the plot.")

    def on_mouse_click(self, event):
        if event.inaxes == self.ax1:
            x, y = event.xdata, event.ydata
            self.root.exec.spawn_agent(x=x, y=y)

            self.replot()
        elif event.inaxes == self.ax2:
            s, d = event.xdata, event.ydata
            self.root.exec.spawn_agent(d=d, s=s)
            self.replot()

    def on_mouse_scroll(self, event, increment=10):
        if event.inaxes == self.ax1:
            log.debug(f"Scroll Event in real coordinate: {event.button}")
            if event.button == "up":
                self.xy_zoom -= increment if self.xy_zoom > increment else 0
            elif event.button == "down":
                self.xy_zoom += increment
        elif event.inaxes == self.ax2:
            log.debug(f"Scroll Event in frenet: {event.button}")
            if event.button == "up":
                self.frenet_zoom -= increment if self.frenet_zoom > increment else 0
            elif event.button == "down":
                self.frenet_zoom += increment

        threshold = 0.01
        if (
            self._prev_scroll_time is None or time.time() - self._prev_scroll_time > threshold
        ) and not self.root.data.animation_running:
            self.replot()

        self._prev_scroll_time = time.time()

    def zoom_in(self):
        self.xy_zoom -= 5 if self.xy_zoom > 5 else 0
        self.replot()

    def zoom_out(self):
        self.xy_zoom += 5
        self.replot()

    def zoom_in_frenet(self):
        self.frenet_zoom -= 5 if self.frenet_zoom > 5 else 0
        self.replot()

    def zoom_out_frenet(self):
        self.frenet_zoom += 5
        self.replot()

    def replot(self):
        canvas_widget = self.canvas.get_tk_widget()
        width = canvas_widget.winfo_width()
        height = canvas_widget.winfo_height()
        aspect_ratio = width / height

        t1 = time.time()
        # self.canvas.restore_region(self.plt_background)
        c52_plotlib.plot(
            exec=self.root.exec,
            aspect_ratio=aspect_ratio,
            xy_zoom=self.xy_zoom,
            frenet_zoom=self.frenet_zoom,
            show_legend=self.root.data.show_legend.get(),
            plot_last_pts=self.root.data.show_past_locations.get(),
            plot_global_plan=self.root.data.show_global_plan.get(),
            plot_local_plan=self.root.data.show_local_plan.get(),
            plot_local_lattice=self.root.data.show_local_lattice.get(),
            plot_state=self.root.data.show_state.get(),
        )
        self.canvas.draw()
        log.debug(f"Plot Time: {(time.time()-t1)*1000:.2f} ms")



        self.root.perceive_plan_control_view.vehicle_state_label.config(
            text=f"Ego State: X: {self.root.exec.ego_state.x:+.2f}, Y: {self.root.exec.ego_state.y:+.2f}, v: {self.root.exec.ego_state.speed:+.2f}, θ: {self.root.exec.ego_state.theta:+.2f}"
        )

        self.root.perceive_plan_control_view.global_tj_wp_entry.delete(0, tk.END)
        self.root.perceive_plan_control_view.global_tj_wp_entry.insert(
            0, str(self.root.exec.planner.global_trajectory.current_wp)
        )
