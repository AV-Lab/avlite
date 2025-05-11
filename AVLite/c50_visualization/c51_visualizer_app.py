from c40_execution.c42_sync_executer import SyncExecuter
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
    exec: SyncExecuter

    def __init__(self, executer: SyncExecuter, code_reload_function=None):
        super().__init__()
        # self.set_dark_mode()
        self.set_dark_mode_themed()

        self.exec = executer
        self.code_reload_function = code_reload_function
        self.is_loading = False    

        self.title("AVlite Visualizer")
        # self.geometry("1200x1100")
        self.small_font = ("Courier", 10)

        # ----------------------------------------------------------------------
        # Variables
        # ----------------------------------------------------------------------
        self.setting = VisualizationSettings()
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
       
        if self.setting.global_plan_view.get() and self.setting.local_plan_view.get():
            self._update_two_plots_layout()
        else:
            self._update_one_plot_layout()

        self.config_shortcut_view.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.perceive_plan_control_view.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.exec_visualize_view.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.log_view.grid(row=4, column=0, columnspan=2, sticky="nsew")
        # Configure grid weights for the 3:1 ratio
        self.grid_rowconfigure(0, weight=1)  # make the plot views expand
        self.grid_columnconfigure(0, weight=1)  # local view gets 3x weight
        self.grid_columnconfigure(1, weight=1)  # global view gets 1x weight
        self.update_idletasks()
        
        self.bind("<Configure>", self.__update_grid_column_sizes)
        
        

        log.info("Reloading stack to ensure configuration is applied.")
        self.reload_stack()
        # Bind to window resize to maintain ratio
        self.toggle_shortcut_mode()
        self.config_shortcut_view.toggle_dark_mode()  
        # self.after(0, self.config_shortcut_view.toggle_dark_mode)

        # self.after(10, self.toggle_plan_view)



    def __update_grid_column_sizes(self,event=None, update_plot=False):
        """Update column sizes when window is resized to maintain 3:1 ratio."""
        if event and event.widget == self:
            width = event.width
            if width > 10:  # Avoid division by zero or tiny windows
                local_width = int(width * 0.5)
                global_width = int(width * 0.5)
                self.grid_columnconfigure(0, minsize=local_width)
                self.grid_columnconfigure(1, minsize=global_width)

        if update_plot:
            if self.setting.global_plan_view.get():
                self.global_plan_plot_view.plot()
            if self.setting.local_plan_view.get():
                self.local_plan_plot_view.plot()
    

    def _update_two_plots_layout(self):
        log.debug(f"Updating two plots layout: global_plan_view: {self.setting.global_plan_view.get()}, local_plan_view: {self.setting.local_plan_view.get()}")
        self.local_plan_plot_view.grid_forget()
        self.global_plan_plot_view.grid_forget()
        self.local_plan_plot_view.grid(row=0, column=0, sticky="nswe")
        self.global_plan_plot_view.grid(row=0, column=1, sticky="nswe")

    def _update_one_plot_layout(self):
        log.debug(f"Updating one plot layout: global_plan_view: {self.setting.global_plan_view.get()}, local_plan_view: {self.setting.local_plan_view.get()}")
        self.local_plan_plot_view.grid_forget()
        self.global_plan_plot_view.grid_forget()
        if self.setting.global_plan_view.get() and not self.setting.local_plan_view.get():
            self.global_plan_plot_view.grid(row=0, column=0, columnspan=2, sticky="nswe")
        elif self.setting.local_plan_view.get() and not self.setting.global_plan_view.get():
            self.local_plan_plot_view.grid(row=0, column=0, columnspan=2, sticky="nswe")
    
    def toggle_plan_view(self):
        if self.setting.global_plan_view.get() and self.setting.local_plan_view.get():
            self._update_two_plots_layout()
            self.global_plan_plot_view.plot()
            self.local_plan_plot_view.plot()
        else:
            self._update_one_plot_layout()
            self.local_plan_plot_view.plot()

    def toggle_shortcut_mode(self):
        if self.setting.shortcut_mode.get():
            self.perceive_plan_control_view.grid_forget()
            self.exec_visualize_view.grid_forget()

            self.config_shortcut_view.grid(row=1, column=0, columnspan=2, sticky="ew")
            self.config_shortcut_view.shortcut_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        else:
            self.config_shortcut_view.shortcut_frame.grid_forget()
            self.perceive_plan_control_view.grid(row=2, column=0, columnspan=2, sticky="ew")
            self.exec_visualize_view.grid(row=3, column=0, columnspan=2, sticky="ew")
            self.log_view.grid(row=4, column=0, columnspan=2, sticky="nsew")


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

    def validate_float_input(self, user_input:str):
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
        if self.setting.global_plan_view.get():
            self.global_plan_plot_view.plot()
        if self.setting.local_plan_view.get():
            self.local_plan_plot_view.plot()

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

        # if self.setting.async_exec.get():
        #     if self.setting.control_dt.get() < 0.1 or self.setting.replan_dt.get() < 0.1:
        #         if not self.setting.disable_log.get():
        #             self.setting.disable_log.set(True)
        #             log.warning(
        #                 f"Logs removed due to low control_dt: {self.setting.control_dt} or replan_dt: {self.setting.replan_dt}."
        #             )
        #             self.setting.log_level.set("STDOUT")
        #             self.log_view.update_log_level()
        #     else:
        #         if self.setting.disable_log.get():
        #             self.setting.disable_log.set(False)
        #             self.setting.log_level.set("INFO")
        #             self.log_view.update_log_level()
        # else:
        #     if self.setting.disable_log.get():
        #         self.setting.disable_log.set(False)
        #         self.setting.log_level.set("INFO")
        #         self.log_view.update_log_level()

        log.debug(f"UI Update Time: {(time.time()-t1)*1000:.2f} ms")
    


    def reload_stack(self):
        # self.__reload_stack_async()
        self.exec_visualize_view.stop_exec()
        if not self.is_loading:
            log.info(f"Reloading the code with async_mode: {self.setting.async_exec.get()}")
            thread = threading.Thread(target=self.__reload_stack_async)
            thread.daemon = True
            thread.start()
            self.is_loading = True
            self.disable_frame(self.exec_visualize_view.execution_frame)
            
        self.global_plan_plot_view.reset()

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

    def set_dark_mode(self):
        # self.local_plan_plot_view.update_plot_theme()
        # self.global_plan_plot_view.toggle_plot_theme()
        # self.configure(bg="gray14")
        # self.log_view.log_area.config(bg="gray14", fg="white", highlightbackground="black")
        # self.config_shortcut_view.help_text.config(bg="gray14", fg="white", highlightbackground="black")

        # self.setting.bg_color = "#333333"
        # self.setting.fg_color = "white"

        try:
            from ttkbootstrap import Style

            # style = Style("darkly")  # Or "cyborg", etc.
            style = Style()
            style.load_user_themes("data/avlite-theme.json")
            style.theme_use("avlitetheme")
            style.configure("TButton",borderwidth=5, bordercolor="black", padding=(5,2))

            style.master = self
            gruvbox_green = "#b8bb26"
            gruvbox_light_green = "#fe8019"
            gruvbox_red = "#9d0006"
            gruvbox_orange = "#d65d0e"
            
            style.configure(
                "Start.TButton",
                background=gruvbox_orange,
                foreground="white",
            )
            style.configure(
                "Stop.TButton",
                background=gruvbox_red,
                foreground="white",
            )

        except ImportError: 
            log.error("Please install ttkbootstrap to use dark mode.")
            self.set_set_light_mode_darker()


    def set_dark_mode_themed(self):

        self.configure(bg="gray14")

        if hasattr(self, "setting"):
            self.setting.bg_color = "#333333"
            self.setting.fg_color = "white"
        if hasattr(self, "local_plan_plot_view") and hasattr(self, "global_plan_plot_view"):
            self.local_plan_plot_view.update_plot_theme()
            self.global_plan_plot_view.update_plot_theme()
        
        if hasattr(self, "log_view") and hasattr(self, "config_shortcut_view"):
            self.log_view.log_area.config(bg="gray14", fg="white", highlightbackground="black")
            self.config_shortcut_view.help_text.config(bg="gray14", fg="white", highlightbackground="black")

        try:
            from ttkthemes import ThemedStyle
            style = ThemedStyle(self)
            style.set_theme("equilux")
            gruvbox_red = "#9d0006"
            gruvbox_orange = "#d65d0e"
            style.layout(
                "Start.TButton",
                [("Button.border", {"sticky": "nswe", "children": [
                    ("Button.padding", {"sticky": "nswe", "children": [
                        ("Button.label", {"sticky": "nswe"})
                    ]})
                ]})]
            )
            style.layout(
                "Stop.TButton",
                [("Button.border", {"sticky": "nswe", "children": [
                    ("Button.padding", {"sticky": "nswe", "children": [
                        ("Button.label", {"sticky": "nswe"})
                    ]})
                ]})]
            )
            
            style.configure(
                "Start.TButton",
                background=gruvbox_orange,
                foreground="white",
            )
            style.configure(
                "Stop.TButton",
                background=gruvbox_red,
                foreground="white",
            )
            style.map(
                "Start.TButton",
                background=[("active", "#ff8800")],  # Lighter orange on click/hover
                foreground=[("active", "white")],
            )
            style.map(
                "Stop.TButton",
                background=[("active", "#ff4444")],  # Lighter red on click/hover
                foreground=[("active", "white")],
            )

        except ImportError:
            log.error("Please install ttkthemes to use dark mode.")
            # self.set_set_light_mode_darker()
        
        log.info("Dark mode enabled.")

    def set_light_mode(self):
        self.configure(bg="white")
        self.log_view.log_area.config(bg="white", fg="black")
        self.config_shortcut_view.help_text.config(bg="white", fg="black")

        self.setting.bg_color = "white"
        self.setting.fg_color = "black"
        self.local_plan_plot_view.update_plot_theme()
        self.global_plan_plot_view.update_plot_theme()
        log.info("Light mode enabled.")
        style = ttk.Style(self)
        style.theme_use('default')  # Reset to default theme

    def set_set_light_mode_darker(self):
        self.configure(bg="gray14")
        self.log_view.log_area.config(bg="gray14", fg="white", highlightbackground="black")
        self.config_shortcut_view.help_text.config(bg="gray14", fg="white", highlightbackground="black")

        self.setting.bg_color = "#333333"
        self.setting.fg_color = "white"
        self.local_plan_plot_view.update_plot_theme()
        self.global_plan_plot_view.update_plot_theme()
        
        style = ttk.Style(self)
        style.theme_use('default')  # Reset to default theme
