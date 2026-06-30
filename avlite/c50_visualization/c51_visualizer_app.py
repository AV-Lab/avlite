import time
import tkinter as tk
from tkinter import ttk, messagebox

import logging

from avlite.c40_execution.c42_executer import Executer
from avlite.c40_execution.c43_factory import executor_factory
from avlite.c40_execution.c44_sync_executer import SyncExecuter
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c50_visualization.c52_plot_views import LocalPlanPlotView, GlobalPlanPlotView
from avlite.c50_visualization.c53_stack_views import PerceivePlanControlView, ExecView
from avlite.c40_execution.c43_factory import load_stack_settings
from avlite.c50_visualization.c59_settings import VisualizationSettings
from avlite.c50_visualization.c58_ui_lib import (
    TkSettingsBinder,
    UiAssets,
    apply_ttk_theme,
    get_dpi_scale,
    scaled,
    setup_dpi,
)
from avlite.c50_visualization.c58_ui_lib import DataPicker
from avlite.c50_visualization.c55_log_view import LogView
from avlite.c50_visualization.c56_config_views import ConfigShortcutView
from avlite.c60_common.c66_plugins import reload_lib
from avlite.c60_common.c67_paths import ConfigPaths
from avlite.c60_common.c69_setting_utils import load_setting, list_profiles
from avlite import __version__
    

log = logging.getLogger(__name__)
logging.getLogger("PIL").setLevel(logging.WARNING)


class VisualizerApp(tk.Tk):
    exec: SyncExecuter | None

    def __init__(self):
        setup_dpi()
        super().__init__()
        apply_ttk_theme(self, dark=True)
        self._dpi_scale: float = get_dpi_scale(self)
        self.exec = None
        self.loading_overlay = None
        self.ui_initialized = False
        self.show_loading_overlay()
        self.update_idletasks()  # Force GUI to update and show the overlay
        self.update()            # Process all pending events
        try:
            self.__initialize_ui()
        except Exception as e:
            log.error("Startup failed: %s", e, exc_info=True)
            messagebox.showerror(
                "Startup failed",
                f"Failed to start AVLite.\n\n{e}",
                parent=self,
            )
        finally:
            self.hide_loading_overlay()


    def __initialize_ui(self):
        self.title("AVlite Visualizer")
        s = self._dpi_scale
        self.geometry(f"{scaled(1200, s)}x{scaled(900, s)}")
        self.small_font = ("Courier", 10)

        # self.set_dark_mode()
        # ----------------------------------------------------------------------
        # Variables
        # ----------------------------------------------------------------------
        self.setting = VisualizationSettings()
        self.setting.profile_list = list_profiles(self.setting)
        startup = ConfigPaths.startup_profile()
        if startup and startup in self.setting.profile_list:
            self.setting.selected_profile.set(startup)

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
        self.exec_visualize_view.grid(row=4, column=0, columnspan=2, sticky="ew")
        self.log_view.grid(row=5, column=0, columnspan=2, sticky="nsew")
        # Configure grid weights for the 3:1 ratio
        self.grid_columnconfigure(0, weight=1)  # local view gets xx weight
        self.grid_columnconfigure(1, weight=1)  # global view gets 1x weight
        self.grid_rowconfigure(0, weight=1)  # make the plot views expand
        self.update_idletasks()
        
        log.info("Reloading stack to ensure configuration is applied.")
        self.load_configs()
        log.warning(f"map is {ExecutionSettings.c40_hd_map}")
        self.reload_stack(reload_code=False)
        log.warning(f"map after is {ExecutionSettings.c40_hd_map}")

        # Bind to window resize to maintain ratio
        self.update_shortcut_mode()

        self.validate_cmd = (self.register(self.validate_float_input), "%P")
        self.bind("<Configure>", self.__update_grid_column_sizes)
        self.after(500, self.update_ui)
        self.last_resize_time = time.time()
        self._create_menubar()
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
        
    
    def update_views(self):
        if self.setting.global_plan_view.get() and self.setting.local_plan_view.get():
            self.__update_two_plots_layout()
        else:
            self.__update_one_plot_layout()

        self.log_view.update_log_view_height()
        self.update_shortcut_mode()
        # self.after(500, self.update_ui)  

    def update_shortcut_mode(self, reverse=False):
        if reverse:
            self.setting.shortcut_mode.set(not self.setting.shortcut_mode.get())

        if self.setting.shortcut_mode.get():
            self.perceive_plan_control_view.grid_forget()
            self.exec_visualize_view.grid_forget()

            self.config_shortcut_view.grid(row=1, column=0, columnspan=2, sticky="ew")
            self.config_shortcut_view.shortcut_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        else:
            self.config_shortcut_view.shortcut_frame.grid_forget()
            self.perceive_plan_control_view.grid(row=2, column=0, columnspan=2, sticky="ew")
            self.exec_visualize_view.grid(row=4, column=0, columnspan=2, sticky="ew")
            self.log_view.grid(row=5, column=0, columnspan=2, sticky="nsew")



    def disable_frame(self, frame: ttk.Frame):
        for child in frame.winfo_children():
            if isinstance(child, ttk.Combobox):
                child.configure(state="disabled")
            elif isinstance(child, (tk.Entry, tk.Button, ttk.Entry, ttk.Button, ttk.Checkbutton, ttk.Radiobutton)):
                child.configure(state="disabled")
            elif isinstance(child, (ttk.LabelFrame, ttk.Frame, tk.Frame)):
                self.disable_frame(child)

    def enable_frame(self, frame: ttk.Frame):
        for child in frame.winfo_children():
            if isinstance(child, ttk.Combobox):
                child.configure(state="readonly")
            elif isinstance(child, (tk.Entry, tk.Button, ttk.Entry, ttk.Button, ttk.Checkbutton, ttk.Radiobutton)):
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
        s = getattr(self, '_dpi_scale', 1.0)
        width = round(450 * s)
        height = round(350 * s)
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
            logo_img = Image.open(UiAssets.resolve("logo.png"))
            logo_img = logo_img.resize((round(256 * s), round(256 * s)), Image.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            logo_label = tk.Label(frame, image=self.logo_photo, bg="black")
            logo_label.pack(pady=(15, 5))
        except Exception:
            log.error("Failed to load logo image.")
            
        # Add loading message
        tk.Label(
            frame, text=message, fg="#10bfe8", bg="black",
            font=("Arial", 12),
        ).pack(pady=10)
        
        # Update the window to make it visible
        self.loading_window.update_idletasks()


    def hide_loading_overlay(self):
        if hasattr(self, 'loading_window') and self.loading_window is not None:
            self.loading_window.destroy()
            self.loading_window = None
            if hasattr(self, 'logo_photo'):
                del self.logo_photo

    def set_dark_mode_themed(self):
        apply_ttk_theme(self, dark=True)

        if hasattr(self, "setting"):
            self.setting.bg_color = "#333333"
            self.setting.fg_color = "white"
        if hasattr(self, "local_plan_plot_view") and hasattr(self, "global_plan_plot_view"):
            self.local_plan_plot_view.update_plot_theme()
            self.global_plan_plot_view.update_plot_theme()

        if hasattr(self, "log_view") and hasattr(self, "config_shortcut_view"):
            self.log_view.log_area.config(bg="gray14", fg="white", highlightbackground="black")
            self.config_shortcut_view.help_text.config(bg="gray14", fg="white", highlightbackground="black")

        if hasattr(self, "menubar"):
            bg = "#333333"
            fg = "#bbbbbb"
            activebg = "#555555"
            activefg = "#bbbbbb"
            self.menubar.configure(bg=bg, fg=fg, activebackground=activebg, activeforeground=activefg)
            for menu in getattr(self, "menus", []):
                menu.configure(bg=bg, fg=fg, activebackground=activebg, activeforeground=activefg)

        log.info("Dark mode enabled.")

    def set_light_mode(self):
        apply_ttk_theme(self, dark=False)

        if hasattr(self, "log_view") and hasattr(self, "config_shortcut_view"):
            self.log_view.log_area.config(bg="white", fg="black")
            self.config_shortcut_view.help_text.config(bg="white", fg="black")

        if hasattr(self, "setting"):
            self.setting.bg_color = "white"
            self.setting.fg_color = "black"
        if hasattr(self, "local_plan_plot_view") and hasattr(self, "global_plan_plot_view"):
            self.local_plan_plot_view.update_plot_theme()
            self.global_plan_plot_view.update_plot_theme()
        if hasattr(self, "menubar"):
            bg = "white"
            fg = "black"
            activebg = "#ececec"
            activefg = "black"
            self.menubar.configure(bg=bg, fg=fg, activebackground=activebg, activeforeground=activefg)
            for menu in getattr(self, "menus", []):
                menu.configure(bg=bg, fg=fg, activebackground=activebg, activeforeground=activefg)

        log.info("Light mode enabled.")

    def _create_menubar(self):
        self.menubar = tk.Menu(self)
        self.menus = []

        file_menu = tk.Menu(self.menubar, tearoff=0)
        file_menu.add_command(label="Settings", command=self.config_shortcut_view.open_settings_window)
        file_menu.add_command(label="Community Plugins", command=self.config_shortcut_view.open_plugins_window)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        self.menubar.add_cascade(label="File", menu=file_menu)
        self.menus.append(file_menu)

        help_menu = tk.Menu(self.menubar, tearoff=0)
        help_menu.add_command(label="About AVLite", command=self._show_about)
        self.menubar.add_cascade(label="Help", menu=help_menu)
        self.menus.append(help_menu)

        # Apply current theme colours
        if self.setting.dark_mode.get():
            bg, fg, activebg, activefg = "#333333", "#bbbbbb", "#555555", "#bbbbbb"
        else:
            bg, fg, activebg, activefg = "white", "black", "#ececec", "black"
        self.menubar.configure(bg=bg, fg=fg, activebackground=activebg, activeforeground=activefg)
        for menu in self.menus:
            menu.configure(bg=bg, fg=fg, activebackground=activebg, activeforeground=activefg)

        self._apply_menubar_visibility()
        self.setting.hide_menubar.trace_add("write", lambda *_: self._apply_menubar_visibility())

    def _show_about(self):
        win = tk.Toplevel(self)
        win.title("About AVLite")
        win.resizable(False, False)
        win.configure(bg="black")
        s = self._dpi_scale

        inner = tk.Frame(win, bg="black", padx=scaled(40, s), pady=0)
        inner.pack(fill="both", expand=True)

        try:
            from PIL import Image, ImageTk
            logo_img = Image.open(UiAssets.resolve("logo.png"))
            logo_size = scaled(200, s)
            logo_img = logo_img.resize((logo_size, logo_size), Image.LANCZOS)
            win._logo_photo = ImageTk.PhotoImage(logo_img)
            tk.Label(inner, image=win._logo_photo, bg="black").pack(pady=(scaled(24, s), scaled(8, s)))
        except Exception:
            log.warning("Failed to load logo for About dialog.")

        tk.Label(inner, text="AVLite", fg="#10bfe8", bg="black",
                 font=("Arial", 16, "bold")).pack()
        tk.Label(inner, text=f"Version {__version__}", fg="#10bfe8", bg="black",
                 font=("Arial", 11)).pack(pady=(scaled(4, s), 0))
        tk.Label(inner, text="A lightweight autonomous driving software stack.",
                 fg="#10bfe8", bg="black", font=("Arial", 10)).pack(pady=(scaled(6, s), scaled(24, s)))

        ttk.Button(inner, text="OK", command=win.destroy).pack(pady=(0, scaled(20, s)))
        win.grab_set()
        win.focus_set()

    def _apply_menubar_visibility(self):
        if self.setting.hide_menubar.get():
            self.config(menu="")
        else:
            self.config(menu=self.menubar)

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

    
    def update_ui(self):
        if self.exec is None:
            return
        t1 = time.time()
        _plot_dt = t1 - getattr(self, '_last_plot_time', 0)
        _do_plot = _plot_dt >= 0.033  # cap plot redraws at ~30 Hz
        if _do_plot:
            self._last_plot_time = t1
            if self.setting.global_plan_view.get():
                self.global_plan_plot_view.plot()
            if self.setting.local_plan_view.get():
                self.local_plan_plot_view.plot()

        if not self.setting.shortcut_mode.get():
            self.setting.vehicle_state.set( f"Loc: ({self.exec.ego_state.x:+7.2f}, {self.exec.ego_state.y:+7.2f}), Vel: {self.exec.ego_state.velocity:5.2f} ({self.exec.ego_state.velocity*3.6:6.2f} km/h), θ: {self.exec.ego_state.theta:+5.1f}")
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
            self.setting.perception_fps.set(f"{self.exec.perception_fps:6.1f}")
            self.setting.lap.set(f"{self.exec.local_planner.lap:5d}")


        log.debug("UI Update Time: %.2f ms", (time.time() - t1) * 1000)
    

    def load_configs(self, only_stack=False, profile=None):
        """ Load settings from a profile or the current settings. Uses c69_setting_utils files plus UI house keeping """
        
        if profile:
            self.setting.selected_profile.set(profile)
        else:
            profile = self.setting.selected_profile.get() 
        # load_setting(PerceptionSettings, profile=profile)
        # load_setting(PlanningSettings, profile=profile)
        # load_setting(ControlSettings, profile=profile)
        # load_setting(ExecutionSettings, profile=profile)
        binder = TkSettingsBinder()
        if not only_stack:
            load_setting(self.setting, profile=profile, binder=binder)
        load_stack_settings(profile=profile, load_plugins=self.setting.load_plugins.get())
        self.setting.default_map_file.set(DataPicker.default_map_display_path())
        self.setting.default_global_plan_file.set(DataPicker.default_global_plan_display_path())
        self.exec_visualize_view.refresh_default_map_tooltips()
        self.config_shortcut_view.update_setting_window()
        log.info(f"Loaded settings from profile: {profile}")

        self.log_view.reset()
        ConfigPaths.set_startup_profile(profile)
        if hasattr(self, "config_shortcut_view"):
            self.config_shortcut_view.toggle_dark_mode()

    def reload_stack(self, reload_code:bool = True):
        if reload_code:
            self.show_loading_overlay("Reloading stack...")
        else:
            self.show_loading_overlay("Reinitializing stack...")

        self.exec_visualize_view.stop_exec()
        self.disable_frame(self)
        self.local_plan_plot_view.grid_forget()
        self.global_plan_plot_view.grid_forget()

        error = None
        try:
            if reload_code:
                reload_lib(exclude_settings=True, reload_plugins=self.setting.load_plugins.get())
            self.exec = executor_factory(
                executer_type=self.setting.executer_type.get(),
                bridge=self.setting.execution_bridge.get(),
                perception_strategy_name=self.setting.perception_type.get(),
                localization_strategy_name=self.setting.localization_type.get(),
                global_planner_strategy_name=self.setting.global_planner_type.get(),
                local_planner_strategy_name=self.setting.local_planner_type.get(),
                controller_strategy_name=self.setting.controller_type.get(),
                perception_dt=self.setting.perception_dt.get(),
                localization_dt=self.setting.localization_dt.get(),
                replan_dt=self.setting.replan_dt.get(),
                control_dt=self.setting.control_dt.get(),
                hd_map=ExecutionSettings.c40_hd_map,
                default_global_trajectory_file=self.setting.default_global_plan_file.get(),
                load_plugins=self.setting.load_plugins.get(),
                async_combined_perception_planning=ExecutionSettings.c40_async_combined_perception_planning,
            )

            self.setting.default_map_file.set(DataPicker.default_map_display_path())
            self.exec_visualize_view.refresh_default_map_tooltips()

        except Exception as e:
            error = e
            log.error(f"Error reloading stack: {e}", exc_info=True)
        finally:
            self.local_plan_plot_view.reset()
            self.global_plan_plot_view.reset()
            self.perceive_plan_control_view.reset()
            self.exec_visualize_view.update_data()
            self.update_views()
            self.update_ui()
            self.enable_frame(self)
            if self.exec is not None:
                self.exec_visualize_view.bridge_frame.update_for_bridge(self.exec.world.capabilities)
            self.focus_set()
            self.hide_loading_overlay()

        if error is not None:
            messagebox.showerror(
                "Reload failed",
                f"Failed to rebuild the stack.\n\n{error}",
                parent=self,
            )
            

    def switch_profile(self):
        self.load_configs(profile=self.setting.next_profile.get(), only_stack=False)
        # self.reload_stack(reload_code=False)
        self.update_views()
        self.update_ui()
