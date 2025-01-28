from c40_execute.c41_executer import Executer
from c50_visualize.c53_plot_view import PlotView
from c50_visualize.c54_perceive_plan_control_view import PerceivePlanControlView
from c50_visualize.c55_visualize_exec_view import VisualizeExecView
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
        self.geometry("1200x1100")

        # ----------------------------------------------------------------------
        # Variables
        # ----------------------------------------------------------------------
        self.data = VisualizerData(only_visualize=only_visualize)
        # ----------------------------------------------------------------------
        # UI Views 
        # ----------------------------------------------------------------------
        self.plot_view = PlotView(self)
        self.config_view = ConfigShortcutView(self)
        self.perceive_plan_control_view = PerceivePlanControlView(self)
        self.visualize_exec_view = VisualizeExecView(self)
        self.log_view = LogView(self)
        # ----------------------------------------------------------------------


        self.config_view.set_dark_mode()
        self.config_view.update_UI() #its confusing why I need to do this


