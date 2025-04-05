from c40_execution.c41_base_executer import BaseExecuter
from c50_visualization.c52_plot_view import LocalPlanPlotView, GlobalPlanPlotView
from c50_visualization.c53_perceive_plan_control_view import PerceivePlanControlView
from c50_visualization.c54_exec_visualize_view import ExecVisualizeView
from c50_visualization.c59_settings import VisualizationSettings
from c50_visualization.c55_log_view import LogView
from c50_visualization.c56_config_shortcut_view import ConfigShortcutView
from c60_tools.c61_utils import load_setting
    
import threading
import time
import tkinter as tk
from tkinter import ttk
import logging

log = logging.getLogger(__name__)


class VisualizerApp(tk.Tk):
    exec: BaseExecuter

    def __init__(self, executer: BaseExecuter, code_reload_function=None, only_visualize=False):
        super().__init__()

        self.exec = executer
        self.code_reload_function = code_reload_function
        self.is_loading = False    

        self.title("AVlite Visualizer")
        # self.geometry("1200x1100")
        self.small_font = ("Courier", 10)

        # ----------------------------------------------------------------------
        # Variables
        # ----------------------------------------------------------------------
        self.setting = VisualizationSettings(only_visualize=only_visualize)
        load_setting(self.setting)
        # ----------------------------------------------------------------------
        # UI Views
        # ---------------------------------------------------------------------
        self.local_plan_plot_view = LocalPlanPlotView(self)
        self.global_plan_plot_view = GlobalPlanPlotView(self)
        self.config_shortcut_view = ConfigShortcutView(self)
        self.perceive_plan_control_view = PerceivePlanControlView(self)
        self.exec_visualize_view = ExecVisualizeView(self)
        self.log_view = LogView(self)
        # ----------------------------------------------------------------------
        if self.setting.global_plan_view.get():
            self._update_two_plots_layout()
        else: # Default to local view
            self._update_one_plot_layout()

        self.reload_stack()
        # Bind to window resize to maintain ratio
        self.toggle_shortcut_mode()
        # self.config_shortcut_view.toggle_dark_mode()  
        self.after(0, self.config_shortcut_view.toggle_dark_mode)


    def __forget_all(self):
        self.local_plan_plot_view.grid_forget()
        self.global_plan_plot_view.grid_forget()
        self.config_shortcut_view.grid_forget()
        self.perceive_plan_control_view.grid_forget()
        self.exec_visualize_view.grid_forget()
        self.log_view.grid_forget()

    def _update_two_plots_layout(self):
        def __update_column_sizes(event=None):
            """Update column sizes when window is resized to maintain 3:1 ratio."""
            if event and event.widget == self:
                width = event.width
                if width > 10:  # Avoid division by zero or tiny windows
                    local_width = int(width * 0.75)
                    global_width = int(width * 0.25)
                    self.grid_columnconfigure(0, minsize=local_width)
                    self.grid_columnconfigure(1, minsize=global_width)
            self.global_plan_plot_view.plot()
        self.__forget_all()
        self.local_plan_plot_view.grid(row=0, column=0, sticky="nswe")
        self.global_plan_plot_view.grid(row=0, column=1, sticky="nswe")
        self.config_shortcut_view.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.perceive_plan_control_view.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.exec_visualize_view.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.log_view.grid(row=4, column=0, columnspan=2, sticky="nsew")
        # Configure grid weights for the 3:1 ratio
        self.grid_rowconfigure(0, weight=1)  # make the plot views expand
        self.grid_columnconfigure(0, weight=3)  # local view gets 3x weight
        self.grid_columnconfigure(1, weight=1)  # global view gets 1x weight
        
        # Set minimum sizes to help enforce the ratio
        self.update_idletasks()
        total_width = self.winfo_width()
        if total_width > 0:
            # Set minimum sizes to maintain approximate ratio
            self.grid_columnconfigure(0, minsize=int(total_width * 0.75))
            self.grid_columnconfigure(1, minsize=int(total_width * 0.25))
        
        self.bind("<Configure>", __update_column_sizes)
    
    def reload_stack(self):
        # self.__reload_stack_async()
        # return
        if not self.is_loading:
            log.info(f"Reloading the code with async_mode: {self.setting.async_exec.get()}")
            thread = threading.Thread(target=self.__reload_stack_async)
            thread.daemon = True
            thread.start()
            self.is_loading = True
            self.disable_frame(self.exec_visualize_view.execution_frame)
            
        

    def __reload_stack_async(self):
        try:
            self.exec_visualize_view.stop_exec()
            if self.code_reload_function is not None:
                self.exec = self.code_reload_function(
                    async_mode=self.setting.async_exec.get(),
                    bridge=self.setting.execution_bridge.get(),
                    global_planner=self.setting.global_planner_type.get(),
                    replan_dt=self.setting.replan_dt.get(),
                    control_dt=self.setting.control_dt.get(),
                )

                self.update_ui()
            else:
                log.warning("No code reload function provided.")
        except Exception as e:
            log.error(f"Error reloading stack: {e}", exc_info=True)

        finally:
            self.is_loading = False
            self.enable_frame(self.exec_visualize_view.execution_frame)
            

    def _update_one_plot_layout(self):
        self.__forget_all()
        self.local_plan_plot_view.grid(row=0, column=0, sticky="nswe")
        self.config_shortcut_view.grid(row=1, column=0, sticky="ew")
        self.perceive_plan_control_view.grid(row=2, column=0, sticky="ew")
        self.exec_visualize_view.grid(row=3, column=0, sticky="ew")
        self.log_view.grid(row=4, column=0, sticky="nsew")
        
        # Reset column configuration
        self.grid_columnconfigure(0, weight=1, minsize=0)  # Full width
        self.grid_columnconfigure(1, weight=0, minsize=0)  # Reset column 1
        self.grid_rowconfigure(0, weight=1)  # make the plot views expand
        
        # Unbind Configure event to prevent ratio maintenance
        self.unbind("<Configure>")
    
    def toggle_plan_view(self):
        if self.setting.global_plan_view.get():
            self._update_two_plots_layout()
            # self.local_plan_plot_view.grid(row=0, column=0)
            # self.global_plan_plot_view.grid(row=0, column=1)
            # # Ensure the global view is updated when switching to it
            self.global_plan_plot_view.plot()
            self.local_plan_plot_view.plot()
        else:
            self._update_one_plot_layout()
            # self.global_plan_plot_view.grid_forget()
            # self.local_plan_plot_view.grid(row=0, column=0, sticky="nsew")
            # # Ensure the local view is pdated when switching to it
            self.local_plan_plot_view.plot()

    def toggle_shortcut_mode(self):
        if self.setting.shortcut_mode.get():
            self.perceive_plan_control_view.grid_forget()
            self.exec_visualize_view.grid_forget()

            column_count = 2 if self.setting.global_plan_view.get() else 1
            self.config_shortcut_view.grid(row=1, column=0, columnspan=column_count, sticky="ew")
            self.config_shortcut_view.shortcut_frame.grid(row=2, column=0, columnspan=column_count, sticky="ew")
        else:
            if self.setting.global_plan_view.get():
                self._update_two_plots_layout()
            else:
                self._update_one_plot_layout()
            self.config_shortcut_view.shortcut_frame.grid_forget()

        # max_height = int(self.winfo_height() * 0.4)
        # self.log_frame.config(height=max_height)
        self.update_ui()
    def disable_frame(self, frame: ttk.Frame):
        for child in frame.winfo_children():
            if isinstance(child, (tk.Entry, tk.Button, ttk.Entry, ttk.Button, ttk.Checkbutton, ttk.Radiobutton)):
                child.configure(state="disabled")
            elif isinstance(child, (ttk.LabelFrame, ttk.Frame, tk.Frame)):
                self.disable_frame(child)

    def enable_frame(self, frame: ttk.Frame):
        for child in frame.winfo_children():
            if isinstance(child, (tk.Entry, tk.Button, ttk.Entry, ttk.Button, ttk.Checkbutton, ttk.Radiobutton)):
                child.configure(state="normal")
            elif isinstance(child, (ttk.LabelFrame, ttk.Frame, tk.Frame)):
                self.enable_frame(child)

    def validate_float_input(self, user_input):
        if user_input == "" or user_input == "-":
            return True
        try:
            float(user_input)
            return True
        except ValueError:
            log.error("Please enter a valid float number")
            return False


    
    def update_ui(self):
        t1 = time.time()
        self.local_plan_plot_view.plot() 
        if self.setting.global_plan_view.get():
            self.global_plan_plot_view.plot()

        if not self.setting.shortcut_mode.get():
            self.setting.vehicle_state.set(
                f"Loc: ({self.exec.ego_state.x:+7.2f}, {self.exec.ego_state.y:+7.2f}),\nVel: {self.exec.ego_state.velocity:5.2f} ({self.exec.ego_state.velocity*3.6:6.2f} km/h),\nθ: {self.exec.ego_state.theta:+5.1f}"
            )
            self.setting.current_wp.set(str(self.exec.local_planner.global_trajectory.current_wp))


            self.perceive_plan_control_view.gauge_cte_vel.set_value(self.exec.controller.cte_velocity)
            self.perceive_plan_control_view.gauge_cte_steer.set_value(self.exec.controller.cte_steer)
            self.perceive_plan_control_view.gauge_acc.set_value(self.exec.controller.cmd.acceleration)
            self.perceive_plan_control_view.gauge_steer.set_value(self.exec.controller.cmd.steer)

            self.setting.elapsed_real_time.set(f"{self.exec.elapsed_real_time:6.2f}")
            self.setting.elapsed_sim_time.set(f"{self.exec.elapsed_sim_time:6.2f}")
            self.setting.replan_fps.set(f"{self.exec.planner_fps:6.1f}")
            self.setting.control_fps.set(f"{self.exec.control_fps:6.1f}")
            self.setting.lap.set(f"{self.exec.local_planner.lap:5d}")

        if self.setting.async_exec.get():
            if self.setting.control_dt.get() < 0.1 or self.setting.replan_dt.get() < 0.1:
                if not self.setting.disable_log.get():
                    self.setting.disable_log.set(True)
                    log.warning(
                        f"Logs removed due to low control_dt: {self.setting.control_dt} or replan_dt: {self.setting.replan_dt}."
                    )
                    self.setting.log_level.set("STDOUT")
                    self.log_view.update_log_level()
            else:
                if self.setting.disable_log.get():
                    self.setting.disable_log.set(False)
                    self.setting.log_level.set("INFO")
                    self.log_view.update_log_level()
        else:
            if self.setting.disable_log.get():
                self.setting.disable_log.set(False)
                self.setting.log_level.set("INFO")
                self.log_view.update_log_level()

        log.debug(f"UI Update Time: {(time.time()-t1)*1000:.2f} ms")
