from c40_execute.c41_executer import Executer
from c50_visualize.c53_plot_view import PlotView
from c50_visualize.c54_perceive_plan_control_view import PerceivePlanControlView
from c50_visualize.c55_exec_visualize_view import ExecVisualizeView
from c50_visualize.c59_data import VisualizerData
from c50_visualize.c56_log_view import LogView
from c50_visualize.c57_config_shortcut_view import ConfigShortcutView

import tkinter as tk

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

        # ----------------------------------------------------------------------
        # Variables
        # ----------------------------------------------------------------------
        self.data = VisualizerData(only_visualize=only_visualize)
        # ----------------------------------------------------------------------
        # UI Views 
        # ----------------------------------------------------------------------
        self.plot_view = PlotView(self)
        # self.plot_view.pack(fill=tk.BOTH, expand=True)

        self.config_shortcut_view = ConfigShortcutView(self)
        # self.config_view.pack(fill=tk.X, side=tk.TOP)

        self.perceive_plan_control_view = PerceivePlanControlView(self)
        # self.perceive_plan_control_view.pack(fill=tk.X)

        self.visualize_exec_view = ExecVisualizeView(self)
        # self.visualize_exec_view.pack(fill=tk.X)

        self.log_view = LogView(self)
        # self.log_view.pack(fill=tk.X)
        # ----------------------------------------------------------------------

        self.plot_view.grid(row=0, column=0, sticky="nswe")
        self.config_shortcut_view.grid(row=1, column=0, sticky="ew")
        self.perceive_plan_control_view.grid(row=2, column=0, sticky="ew")
        self.visualize_exec_view.grid(row=3, column=0, sticky="ew")
        self.log_view.grid(row=4, column=0, sticky="nsew")

        # Configure grid weights
        self.grid_rowconfigure(0, weight=1) # make the plot view expand 
        self.grid_columnconfigure(0, weight=1)

        self.config_shortcut_view.set_dark_mode()

        self.config_shortcut_view.update_UI() #its confusing why I need to do this


