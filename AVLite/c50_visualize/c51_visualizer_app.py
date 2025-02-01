from c40_execute.c41_executer import Executer
from c50_visualize.c53_plot_view import PlotView
from c50_visualize.c54_perceive_plan_control_view import PerceivePlanControlView
from c50_visualize.c55_exec_visualize_view import ExecVisualizeView
from c50_visualize.c59_data import VisualizerData
from c50_visualize.c56_log_view import LogView
from c50_visualize.c57_config_shortcut_view import ConfigShortcutView


import tkinter as tk
from tkinter import ttk
import logging


log = logging.getLogger(__name__)



class VisualizerApp(tk.Tk):
    exe: Executer

    def __init__(self, executer: Executer, code_reload_function=None, only_visualize=False):
        super().__init__()

        self.exec = executer
        self.code_reload_function = code_reload_function

        self.title("AVlite Visualizer")
        # self.geometry("1200x1100")
        self.small_font = ("Courier", 10)

        # ----------------------------------------------------------------------
        # Variables
        # ----------------------------------------------------------------------
        self.data = VisualizerData(only_visualize=only_visualize)
        # ----------------------------------------------------------------------
        # UI Views 
        # ----------------------------------------------------------------------
        self.plot_view = PlotView(self)
        self.config_shortcut_view = ConfigShortcutView(self)
        self.perceive_plan_control_view = PerceivePlanControlView(self)
        self.visualize_exec_view = ExecVisualizeView(self)
        self.log_view = LogView(self)
        # ----------------------------------------------------------------------

        self.plot_view.grid(row=0, column=0, sticky="nswe")
        self.config_shortcut_view.grid(row=1, column=0, sticky="ew")
        self.perceive_plan_control_view.grid(row=2, column=0, sticky="ew")
        self.visualize_exec_view.grid(row=3, column=0, sticky="ew")
        self.log_view.grid(row=4, column=0, sticky="nsew")

        # Configure grid weights
        self.grid_rowconfigure(0, weight=1) # make the plot view expand 
        self.grid_columnconfigure(0, weight=1)

        # need otherwise matplotlib plt acts funny        
        self.after(50, self.config_shortcut_view.set_dark_mode)

    def disable_frame(self, frame: ttk.Frame):
        for child in frame.winfo_children():
            if isinstance(child, (tk.Entry, tk.Button, ttk.Entry, ttk.Button)):
                child.configure(state='disabled')
            elif isinstance(child, (ttk.LabelFrame, ttk.Frame, tk.Frame)):
                self.disable_frame(child)

    def enable_frame(self, frame: ttk.Frame):
        for child in frame.winfo_children():
            if isinstance(child, (tk.Entry, tk.Button, ttk.Entry, ttk.Button)):
                child.configure(state='normal')
            elif isinstance(child, (ttk.LabelFrame, ttk.Frame, tk.Frame)):
                self.enable_frame(child)

    def update_ui(self):
        self.plot_view.plot()
        self.perceive_plan_control_view.vehicle_state_label.config(
            text=f"Ego: ({self.exec.ego_state.x:+7.2f}, {self.exec.ego_state.y:+7.2f}), v: {self.exec.ego_state.velocity:5.2f} ({self.exec.ego_state.velocity*3.6:6.2f} km/h), θ: {self.exec.ego_state.theta:+4.1f}"
        )

        self.perceive_plan_control_view.global_tj_wp_entry.delete(0, tk.END)
        self.perceive_plan_control_view.global_tj_wp_entry.insert(
            0, str(self.exec.planner.global_trajectory.current_wp)
        )

        acc =self.exec.controller.cmd.acceleration  
        steer = self.exec.controller.cmd.steer
        state = self.exec.ego_state

        self.perceive_plan_control_view.gauge_cte_vel.set_value(self.exec.controller.cte_velocity)
        self.perceive_plan_control_view.gauge_cte_steer.set_value(self.exec.controller.cte_steer)
        self.perceive_plan_control_view.progressbar_acc.set_value(acc)
        self.perceive_plan_control_view.progressbar_steer.set_value(steer)
