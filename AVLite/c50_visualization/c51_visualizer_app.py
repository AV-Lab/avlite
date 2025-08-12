import time
import tkinter as tk
from tkinter import ttk
import logging

from yaml import load_all

from c10_perception.c19_settings import PerceptionSettings
from c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from c20_planning.c29_settings import PlanningSettings
from c30_control.c39_settings import ControlSettings
from c40_execution.c41_execution_model import Executer
from c40_execution.c42_sync_executer import SyncExecuter
from c40_execution.c43_async_threaded_executer import AsyncThreadedExecuter
from c40_execution.c49_settings import ExecutionSettings
from c50_visualization.c52_plot_views import LocalPlanPlotView, GlobalPlanPlotView
from c50_visualization.c53_perceive_plan_control_views import PerceivePlanControlView
from c50_visualization.c54_exec_views import ExecView
from c50_visualization.c59_settings import VisualizationSettings
from c50_visualization.c55_log_view import LogView
from c50_visualization.c56_config_views import ConfigShortcutView
from c60_tools.c61_utils import load_setting, list_profiles, list_extensions
    

log = logging.getLogger(__name__)
logging.getLogger("PIL").setLevel(logging.WARNING)


class VisualizerApp(tk.Tk):
    exec: SyncExecuter

    def __init__(self):
        super().__init__()
        self.exec = Executer.executor_factory()
        self.loading_overlay = None
        self.ui_initialized = False
        self.show_loading_overlay()
        self.update_idletasks()  # Force GUI to update and show the overlay
        self.update()            # Process all pending events
        # self.after(500, self.__initialize_ui)  
        self.__initialize_ui()
        self.hide_loading_overlay()
        

    def __initialize_ui(self):
        self.set_dark_mode_themed()

        self.title("AVlite Visualizer")
        # self.geometry("1200x1100")
        self.small_font = ("Courier", 10)

        # self.set_dark_mode()
        # ----------------------------------------------------------------------
        # Variables
        # ----------------------------------------------------------------------
        self.setting = VisualizationSettings()
        self.setting.profile_list = list_profiles(self.setting)
        self.setting.extension_list = list_extensions()
        
        # ----------------------------------------------------------------------
        # UI Views
        # ---------------------------------------------------------------------
        self.local_plan_plot_view = LocalPlanPlotView(self)
        self.global_plan_plot_view = GlobalPlanPlotView(self)
        self.config_shortcut_view = ConfigShortcutView(self)
        self.perceive_plan_control_view = PerceivePlanControlView(self)
        self.exec_visualize_view = ExecView(self)
        self.log_view = LogView(self)

        self.config_shortcut_view.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.perceive_plan_control_view.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.exec_visualize_view.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.log_view.grid(row=4, column=0, columnspan=2, sticky="nsew")
        # Configure grid weights for the 3:1 ratio
        self.grid_rowconfigure(0, weight=1)  # make the plot views expand
        self.grid_columnconfigure(0, weight=1)  # local view gets xx weight
        self.grid_columnconfigure(1, weight=1)  # global view gets 1x weight
        self.update_idletasks()
        
        log.info("Reloading stack to ensure configuration is applied.")
        self.load_configs()
        self.reload_stack()
        # Bind to window resize to maintain ratio
        self.toggle_shortcut_mode()
        self.config_shortcut_view.toggle_dark_mode()  

        self.bind("<Configure>", self.__update_grid_column_sizes)
        self.after(500, self.update_ui)
        self.last_resize_time = time.time()
        self.ui_initialized = True
        
        log.info(f"Available profiles: {self.setting.profile_list}")

    def __update_grid_column_sizes(self,event=None):
        """Update column sizes when window is resized to maintain 3:1 ratio."""

        if event and event.widget == self:
            width = event.width
            if width > 10:  # Avoid division by zero or tiny windows
                local_width = int(width * 0.5)
                global_width = int(width * 0.5)
                self.grid_columnconfigure(0, minsize=local_width)
                self.grid_columnconfigure(1, minsize=global_width)

        if time.time() - self.last_resize_time > 1:

            log.debug(f"Updating UI after resize in 500 ms")
            self.after(500, self.update_ui)
            # self.update_ui()
            self.last_resize_time = time.time()
    

    def __update_two_plots_layout(self):
        log.debug(f"Updating two plots layout: global_plan_view: {self.setting.global_plan_view.get()}, local_plan_view: {self.setting.local_plan_view.get()}")
        self.local_plan_plot_view.grid_forget()
        self.global_plan_plot_view.grid_forget()
        self.local_plan_plot_view.grid(row=0, column=0, sticky="nswe")
        self.global_plan_plot_view.grid(row=0, column=1, sticky="nswe")

    def __update_one_plot_layout(self):
        log.debug(f"Updating one plot layout: global_plan_view: {self.setting.global_plan_view.get()}, local_plan_view: {self.setting.local_plan_view.get()}")
        self.local_plan_plot_view.grid_forget()
        self.global_plan_plot_view.grid_forget()
        if self.setting.global_plan_view.get() and not self.setting.local_plan_view.get():
            self.global_plan_plot_view.grid(row=0, column=0, columnspan=2, sticky="nswe")
        elif self.setting.local_plan_view.get() and not self.setting.global_plan_view.get():
            self.local_plan_plot_view.grid(row=0, column=0, columnspan=2, sticky="nswe")
        
    
    def toggle_plan_view(self):
        if self.setting.global_plan_view.get() and self.setting.local_plan_view.get():
            self.__update_two_plots_layout()
        else:
            self.__update_one_plot_layout()

        self.after(500, self.update_ui)  

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
            self.setting.vehicle_state.set( f"Loc: ({self.exec.ego_state.x:+7.2f}, {self.exec.ego_state.y:+7.2f}),\nVel: {self.exec.ego_state.velocity:5.2f} ({self.exec.ego_state.velocity*3.6:6.2f} km/h),\nθ: {self.exec.ego_state.theta:+5.1f}")
            self.setting.current_wp.set(str(self.exec.local_planner.global_trajectory.current_wp))

            # TODO: need to connect to a tkinter variable instead
            self.perceive_plan_control_view.control_frame.gauge_cte_vel.set_value(self.exec.controller.cte_velocity)
            self.perceive_plan_control_view.control_frame.gauge_cte_steer.set_value(self.exec.controller.cte_steer)
            self.perceive_plan_control_view.control_frame.gauge_acc.set_value(self.exec.controller.cmd.acceleration)
            self.perceive_plan_control_view.control_frame.gauge_steer.set_value(self.exec.controller.cmd.steer)

            self.setting.elapsed_real_time.set(f"{self.exec.elapsed_real_time:6.2f}")
            self.setting.elapsed_sim_time.set(f"{self.exec.elapsed_sim_time:6.2f}")
            self.setting.replan_fps.set(f"{self.exec.planner_fps:6.1f}")
            self.setting.control_fps.set(f"{self.exec.control_fps:6.1f}")
            self.setting.lap.set(f"{self.exec.local_planner.lap:5d}")


        log.debug(f"UI Update Time: {(time.time()-t1)*1000:.2f} ms")
    

    def load_configs(self, only_stack=False):
        profile = self.setting.selected_profile.get()
        load_setting(PerceptionSettings, profile=profile)
        load_setting(PlanningSettings, profile=profile)
        load_setting(ControlSettings, profile=profile)
        load_setting(ExecutionSettings, profile=profile)
        if not only_stack:
            load_setting(self.setting, profile=profile)
        log.info(f"Loaded settings from profile: {profile}")
        log.info(f"map is {ExecutionSettings.hd_map}")

        self.log_view.reset()

    def reload_stack(self, reload_code:bool = True):
        if reload_code:
            self.show_loading_overlay("Reloading stack...")
        else:
            self.show_loading_overlay("Reinitializing stack...")

        self.exec_visualize_view.stop_exec()
        self.disable_frame(self)
        self.local_plan_plot_view.grid_forget()
        self.global_plan_plot_view.grid_forget()

        try:
            # if reload_code:
                # self.load_configs()
            self.exec = Executer.executor_factory(
                async_mode=self.setting.async_exec.get(),
                bridge=self.setting.execution_bridge.get(),
                perception=self.setting.perception_type.get(),
                global_planner=self.setting.global_planner_type.get(),
                local_planner=self.setting.local_planner_type.get(),
                controller=self.setting.controller_type.get(),
                replan_dt=self.setting.replan_dt.get(),
                control_dt=self.setting.control_dt.get(),
                hd_map=ExecutionSettings.hd_map,
                reload_code=reload_code,
                exclude_reload_settings=True,
                load_extensions=self.setting.load_extensions.get(),
            )

        except Exception as e:
            log.error(f"Error reloading stack: {e}", exc_info=True)


        self.local_plan_plot_view.reset()
        self.global_plan_plot_view.reset()
        self.toggle_plan_view()
        self.update_ui()
        self.enable_frame(self)
        self.hide_loading_overlay()
            



    def show_loading_overlay(self, message="Loading..."):
        if hasattr(self, 'loading_window') and self.loading_window is not None:
            return
        
        try:
            self.loading_window = tk.Toplevel(self)
            self.loading_window.overrideredirect(True)  # No window decorations
            self.loading_window.attributes("-topmost", True)  # Keep on top
        except Exception as e:
            log.error(f"Error in creating loading overlay {e}")

    
        # Center the loading window on the screen using xrandr output
        width = 450
        height = 350
        try:
            import subprocess
            output = subprocess.check_output(['xrandr']).decode('utf-8')
            import re
            current = re.search(r'(\d+)x(\d+)\+(\d+)\+(\d+)', output)
            if current:
                mon_w, mon_h, mon_x, mon_y = map(int, current.groups())
                x = mon_x + (mon_w - width) // 2
                y = mon_y + (mon_h - height) // 2
            else:
                raise Exception("Couldn't parse xrandr output")
        except Exception:
            x = (self.winfo_screenwidth() - width) // 2
            y = (self.winfo_screenheight() - height) // 2
        
        
        try:
            self.loading_window.geometry(f"{width}x{height}+{x}+{y}")
        except Exception as e:
            log.error(f"unable to set window geometry {e}")
        
        
        # Black background that matches the logo
        frame = tk.Frame(self.loading_window, bg="#000707", bd=1)
        frame.place(relwidth=1, relheight=1)
        
        # Try to load and display logo
        try:
            from PIL import Image, ImageTk
            logo_img = Image.open("data/imgs/logo.png")
            logo_img = logo_img.resize((256, 256), Image.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            logo_label = tk.Label(frame, image=self.logo_photo, bg="black")
            logo_label.pack(pady=(15, 5))
        except Exception:
            log.error("Failed to load logo image.")
            
        # Add loading message
        tk.Label(frame, text=message, fg="#10bfe8", bg="black", font=("Arial", 12)).pack(pady=10)
        
        # Update the window to make it visible
        self.loading_window.update_idletasks()


    def hide_loading_overlay(self):
        if hasattr(self, 'loading_window') and self.loading_window is not None:
            self.loading_window.destroy()
            self.loading_window = None
            if hasattr(self, 'logo_photo'):
                del self.logo_photo

    def set_dark_mode_themed(self):

        # self.configure(bg="gray14")

        if hasattr(self, "setting"):
            self.setting.bg_color = "#333333"
            self.setting.fg_color = "white"
        if hasattr(self, "local_plan_plot_view") and hasattr(self, "global_plan_plot_view"):
            self.local_plan_plot_view.update_plot_theme()
            self.global_plan_plot_view.update_plot_theme()
        
        if hasattr(self, "log_view") and hasattr(self, "config_shortcut_view"):
            self.log_view.log_area.config(bg="gray14", fg="white", highlightbackground="black")
            self.config_shortcut_view.help_text.config(bg="gray14", fg="white", highlightbackground="black")
    
        if hasattr(self, 'menubar'):
            bg = "#333333"; fg = "white"; activebg = "#555555"; activefg = "white"
            self.menubar.configure(bg=bg, fg=fg, activebackground=activebg, activeforeground=activefg)
            for menu in getattr(self, "menus", []):
                menu.configure(bg=bg, fg=fg, activebackground=activebg, activeforeground=activefg)

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
            
            style.configure( "Start.TButton", background=gruvbox_orange, foreground="white",)
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
            self.option_add('*Listbox.background', '#222222')
            self.option_add('*Listbox.foreground', '#ffffff')
            self.option_add('*Listbox.selectBackground', '#444444')
            self.option_add('*Listbox.selectForeground', '#dddddd')
            self.option_add('*Listbox.highlightBackground', '#1a1a1a')
            self.option_add('*Listbox.highlightColor', '#333333')
            self.option_add('*Listbox.borderWidth', 1)

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
        if hasattr(self, 'menubar'):
            bg = "white"; fg = "black"; activebg = "#ececec"; activefg = "black"
            self.menubar.configure(bg=bg, fg=fg, activebackground=activebg, activeforeground=activefg)
            for menu in getattr(self, "menus", []):
                menu.configure(bg=bg, fg=fg, activebackground=activebg, activeforeground=activefg)

        log.info("Light mode enabled.")
        style = ttk.Style(self)
        style.theme_use('default')  # Reset to default theme
        self.option_add('*Listbox.background', 'white')
        self.option_add('*Listbox.foreground', 'black')
        self.option_add('*Listbox.selectBackground', '#0078d7')  # Or 'lightblue' for a more neutral color
        self.option_add('*Listbox.selectForeground', 'white')
        self.option_add('*Listbox.highlightBackground', 'white')
        self.option_add('*Listbox.highlightColor', '#0078d7')  # Or 'black' for a simple border
        self.option_add('*Listbox.borderWidth', 2)

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

