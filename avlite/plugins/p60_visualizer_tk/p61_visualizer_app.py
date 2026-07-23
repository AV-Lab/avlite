import logging
import re
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = ImageTk = None  # type: ignore[misc, assignment]

_PIL_AVAILABLE = Image is not None

from avlite.c60_apps.c62_factory import executor_factory
from avlite.c10_perception.c11_perception_model import AgentState
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c40_execution.c44_sync_executer import SyncExecuter
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c60_apps.c61_app_strategy import AppStrategy
from avlite.c60_apps.c69_settings import AppSettings
from avlite.plugins.p60_visualizer_tk.p66_plot_views import (
    LocalPlanPlotView,
    GlobalPlanPlotView,
    _canvas_ready,
)
from avlite.plugins.p60_visualizer_tk.p67_stack_views import PerceivePlanControlView, ExecView
from avlite.c60_apps.c62_factory import load_stack_settings
from avlite.plugins.p60_visualizer_tk.settings import VisualizationSettings, sync_stack_settings_to_ui
from avlite.plugins.p60_visualizer_tk.p65_ui_lib import (
    DpiScale,
    TkSettingsBinder,
    UiAssets,
    apply_ttk_theme,
)
from avlite.plugins.p60_visualizer_tk.p65_ui_lib import DataPicker
from avlite.plugins.p60_visualizer_tk.p68_log_view import LogView
from avlite.plugins.p60_visualizer_tk.p64_setting_views import SettingShortcutView
from avlite.c60_apps.c63_plugins import reload_lib
from avlite.c60_apps.c68_paths import ConfigPaths
from avlite.c60_apps.c65_setting_utils import load_setting, list_profiles
from avlite.c60_apps.c66_app_update import AppUpdater
from avlite import __version__


log = logging.getLogger(__name__)
logging.getLogger("PIL").setLevel(logging.WARNING)


class VisualizerApp(tk.Tk):
    exec: SyncExecuter | None
    hosting_plugin_name = "p60_visualizer_tk"

    def __init__(self):
        DpiScale.setup()
        super().__init__()
        apply_ttk_theme(self, dark=True)
        self._dpi_scale: float = DpiScale.for_widget(self)
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
        self.geometry(f"{DpiScale.scaled(1200, s)}x{DpiScale.scaled(900, s)}")
        self.withdraw()
        self.small_font = ("Courier", 10)

        # self.set_dark_mode()
        # ----------------------------------------------------------------------
        # Variables
        # ----------------------------------------------------------------------
        self.setting = VisualizationSettings()
        self.setting.profile_list = list_profiles(AppSettings)
        startup = ConfigPaths.startup_profile()
        if startup and startup in self.setting.profile_list:
            self.setting.c60_selected_profile.set(startup)

        # ----------------------------------------------------------------------
        # UI Views
        # ---------------------------------------------------------------------
        self.local_plan_plot_view = LocalPlanPlotView(self)
        self.global_plan_plot_view = GlobalPlanPlotView(self)
        self.setting_shortcut_view = SettingShortcutView(self)
        self.perceive_plan_control_view = PerceivePlanControlView(self)
        self.exec_visualize_view = ExecView(self)
        self.log_view = LogView(self)

        self.setting_shortcut_view.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.perceive_plan_control_view.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.exec_visualize_view.grid(row=4, column=0, columnspan=2, sticky="ew")
        self.log_view.grid(row=5, column=0, columnspan=2, sticky="nsew")
        # Configure grid weights for the 3:1 ratio
        self.grid_columnconfigure(0, weight=1)  # local view gets xx weight
        self.grid_columnconfigure(1, weight=1)  # global view gets 1x weight
        self.grid_rowconfigure(0, weight=1)  # make the plot views expand
        self.update_views()

        log.info("Reloading stack to ensure configuration is applied.")
        self.load_settings()
        self.reload_stack(reload_code=False, preserve_plot_layout=True)

        # Bind to window resize to maintain ratio
        self.update_shortcut_mode()

        self.validate_cmd = (self.register(self.validate_float_input), "%P")
        self.bind("<Configure>", self.__update_grid_column_sizes)
        self.last_resize_time = time.time()
        self._finalize_startup_layout()
        self._create_menubar()
        self.ui_initialized = True
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(4000, self._maybe_offer_update)

        log.info(f"Available profiles: {self.setting.profile_list}")

    def _on_close(self):
        if hasattr(self, "exec_visualize_view"):
            self.exec_visualize_view.stop_exec()
        if hasattr(self, "log_view"):
            self.log_view.shutdown()
        self.quit()
        self.destroy()

    def _plots_canvas_ready(self) -> bool:
        local_ok = (
            not self.setting.p67_local_plan_view.get()
            or _canvas_ready(self.local_plan_plot_view.canvas.get_tk_widget())
        )
        global_ok = (
            not self.setting.p67_global_plan_view.get()
            or _canvas_ready(self.global_plan_plot_view.canvas.get_tk_widget())
        )
        return local_ok and global_ok

    def _finalize_startup_layout(self, attempt: int = 0) -> None:
        """Reveal the main window and plot only after canvases have final geometry."""
        self.update_views()
        if not self.winfo_viewable():
            self.deiconify()
        self.update_idletasks()
        if self.exec is not None and self._plots_canvas_ready():
            self.update_ui()
            self.focus_set()
            return
        if attempt < 5:
            self.after_idle(lambda: self._finalize_startup_layout(attempt + 1))
            return
        if self.exec is not None:
            self.update_ui()
        self.focus_set()

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
        log.debug(f"Updating two plots layout: global_plan_view: {self.setting.p67_global_plan_view.get()}, local_plan_view: {self.setting.p67_local_plan_view.get()}")
        self.local_plan_plot_view.grid_forget()
        self.global_plan_plot_view.grid_forget()
        self.local_plan_plot_view.grid(row=0, column=0, sticky="nswe")
        self.global_plan_plot_view.grid(row=0, column=1, sticky="nswe")

    def __update_one_plot_layout(self):
        log.debug(f"Updating one plot layout: global_plan_view: {self.setting.p67_global_plan_view.get()}, local_plan_view: {self.setting.p67_local_plan_view.get()}")
        self.local_plan_plot_view.grid_forget()
        self.global_plan_plot_view.grid_forget()
        if self.setting.p67_global_plan_view.get() and not self.setting.p67_local_plan_view.get():
            self.global_plan_plot_view.grid(row=0, column=0, columnspan=2, sticky="nswe")
        elif self.setting.p67_local_plan_view.get() and not self.setting.p67_global_plan_view.get():
            self.local_plan_plot_view.grid(row=0, column=0, columnspan=2, sticky="nswe")
        
    
    def update_views(self):
        if self.setting.p67_global_plan_view.get() and self.setting.p67_local_plan_view.get():
            self.__update_two_plots_layout()
        else:
            self.__update_one_plot_layout()

        self.log_view.update_log_view_height()
        self.update_shortcut_mode()
        # self.after(500, self.update_ui)  

    def update_shortcut_mode(self, reverse=False):
        if reverse:
            self.setting.p60_shortcut_mode.set(not self.setting.p60_shortcut_mode.get())

        if self.setting.p60_shortcut_mode.get():
            self.perceive_plan_control_view.grid_forget()
            self.exec_visualize_view.grid_forget()

            self.setting_shortcut_view.grid(row=1, column=0, columnspan=2, sticky="ew")
            self.setting_shortcut_view.shortcut_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        else:
            self.setting_shortcut_view.shortcut_frame.grid_forget()
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
            output = subprocess.check_output(['xrandr']).decode('utf-8')
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
            if not _PIL_AVAILABLE:
                raise ImportError("PIL not available")
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
        tk.Label(
            frame, text=f"v{__version__}", fg="#0a7a96", bg="black",
            font=("Arial", 10),
        ).place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-8)
        
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
            self.setting.p60_bg_color = "#333333"
            self.setting.p60_fg_color = "white"
        if hasattr(self, "local_plan_plot_view") and hasattr(self, "global_plan_plot_view"):
            self.local_plan_plot_view.update_plot_theme()
            self.global_plan_plot_view.update_plot_theme()
        if hasattr(self, "exec_visualize_view"):
            self.exec_visualize_view.bridge_frame.update_canvas_theme()

        if hasattr(self, "log_view") and hasattr(self, "setting_shortcut_view"):
            self.log_view.log_area.config(bg="gray14", fg="white", highlightbackground="black")
            self.setting_shortcut_view.help_text.config(bg="gray14", fg="white", highlightbackground="black")

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

        if hasattr(self, "log_view") and hasattr(self, "setting_shortcut_view"):
            self.log_view.log_area.config(bg="white", fg="black")
            self.setting_shortcut_view.help_text.config(bg="white", fg="black")

        if hasattr(self, "setting"):
            self.setting.p60_bg_color = "white"
            self.setting.p60_fg_color = "black"
        if hasattr(self, "local_plan_plot_view") and hasattr(self, "global_plan_plot_view"):
            self.local_plan_plot_view.update_plot_theme()
            self.global_plan_plot_view.update_plot_theme()
        if hasattr(self, "exec_visualize_view"):
            self.exec_visualize_view.bridge_frame.update_canvas_theme()
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
        file_menu.add_command(label="Settings", command=self.setting_shortcut_view.open_settings_window)
        file_menu.add_command(label="Community Plugins", command=self.setting_shortcut_view.open_plugins_window)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        self.menubar.add_cascade(label="File", menu=file_menu)
        self.menus.append(file_menu)

        help_menu = tk.Menu(self.menubar, tearoff=0)
        help_menu.add_command(label="Update…", command=self._on_check_update)
        help_menu.add_command(label="About AVLite", command=self._show_about)
        self.menubar.add_cascade(label="Help", menu=help_menu)
        self.menus.append(help_menu)

        # Apply current theme colours
        if self.setting.p60_dark_mode.get():
            bg, fg, activebg, activefg = "#333333", "#bbbbbb", "#555555", "#bbbbbb"
        else:
            bg, fg, activebg, activefg = "white", "black", "#ececec", "black"
        self.menubar.configure(bg=bg, fg=fg, activebackground=activebg, activeforeground=activefg)
        for menu in self.menus:
            menu.configure(bg=bg, fg=fg, activebackground=activebg, activeforeground=activefg)

        self._apply_menubar_visibility()
        self.setting.p60_hide_menubar.trace_add("write", lambda *_: self._apply_menubar_visibility())

    def _show_about(self):
        win = tk.Toplevel(self)
        win.title("About AVLite")
        win.resizable(False, False)
        win.configure(bg="black")
        s = self._dpi_scale

        inner = tk.Frame(win, bg="black", padx=DpiScale.scaled(40, s), pady=0)
        inner.pack(fill="both", expand=True)

        try:
            if not _PIL_AVAILABLE:
                raise ImportError("PIL not available")
            logo_img = Image.open(UiAssets.resolve("logo.png"))
            logo_size = DpiScale.scaled(200, s)
            logo_img = logo_img.resize((logo_size, logo_size), Image.LANCZOS)
            win._logo_photo = ImageTk.PhotoImage(logo_img)
            tk.Label(inner, image=win._logo_photo, bg="black").pack(pady=(DpiScale.scaled(24, s), DpiScale.scaled(8, s)))
        except Exception:
            log.warning("Failed to load logo for About dialog.")

        tk.Label(inner, text="AVLite", fg="#10bfe8", bg="black",
                 font=("Arial", 16, "bold")).pack()
        tk.Label(inner, text=f"Version {__version__}", fg="#10bfe8", bg="black",
                 font=("Arial", 11)).pack(pady=(DpiScale.scaled(4, s), 0))
        tk.Label(inner, text="A lightweight autonomous vehicle software stack.",
                 fg="#10bfe8", bg="black", font=("Arial", 10)).pack(pady=(DpiScale.scaled(6, s), DpiScale.scaled(24, s)))

        ttk.Button(inner, text="OK", command=win.destroy).pack(pady=(0, DpiScale.scaled(20, s)))
        win.grab_set()
        win.focus_set()

    def _on_check_update(self):
        if getattr(self, "_update_busy", False):
            return
        self._update_busy = True

        def work():
            try:
                latest = AppUpdater.latest()
                newer = AppUpdater.is_newer(latest)
                err = None
            except Exception as e:
                latest, newer, err = None, False, e

            def done():
                self._update_busy = False
                if err is not None:
                    log.warning("Update check failed: %s", err)
                    messagebox.showerror(
                        "Update",
                        f"Could not check for updates.\n\n{err}",
                        parent=self,
                    )
                    return
                if not newer:
                    log.info("AVLite is up to date (%s)", __version__)
                    messagebox.showinfo(
                        "Update",
                        f"AVLite is up to date ({__version__}).",
                        parent=self,
                    )
                    return
                log.info("Update available: %s → %s", __version__, latest)
                if not messagebox.askyesno(
                    "Update",
                    f"Update available: {__version__} → {latest}\n\n"
                    "Install with pip now?",
                    parent=self,
                ):
                    log.info("User declined avlite upgrade")
                    return
                self._update_busy = True

                def upgrade_work():
                    try:
                        AppUpdater.upgrade()
                        up_err = None
                    except Exception as e:
                        up_err = e

                    def upgrade_done():
                        self._update_busy = False
                        if up_err is not None:
                            log.warning("Upgrade failed: %s", up_err)
                            messagebox.showerror(
                                "Update",
                                f"Upgrade failed.\n\n{up_err}",
                                parent=self,
                            )
                            return
                        log.info("Updated to %s; restart required", latest)
                        messagebox.showinfo(
                            "Update",
                            f"Updated to {latest}.\n\nRestart AVLite to use the new version.",
                            parent=self,
                        )

                    self.after(0, upgrade_done)

                threading.Thread(target=upgrade_work, daemon=True).start()

            self.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    def _maybe_offer_update(self):
        if self.setting.exec_running:
            return

        def work():
            try:
                latest = AppUpdater.latest()
                if not AppUpdater.is_newer(latest):
                    log.info("Startup update check: up to date (%s)", __version__)
                    return
            except Exception as e:
                log.info("Startup update check skipped: %s", e)
                return

            def show_toast():
                if self.setting.exec_running:
                    return
                if getattr(self, "_update_toast", None) is not None:
                    return
                s = self._dpi_scale
                toast = tk.Frame(self, bg="#1a1a1a", highlightbackground="#10bfe8", highlightthickness=1)
                toast.place(relx=1.0, rely=0.0, anchor="ne", x=-DpiScale.scaled(12, s), y=DpiScale.scaled(12, s))
                self._update_toast = toast
                log.info("Showing update toast: %s → %s", __version__, latest)
                msg = tk.Label(
                    toast,
                    text=f"Update available: {__version__} → {latest}",
                    fg="#10bfe8",
                    bg="#1a1a1a",
                    font=("Arial", 10),
                    padx=DpiScale.scaled(12, s),
                    pady=DpiScale.scaled(8, s),
                )
                msg.pack(side="left")

                def dismiss():
                    if getattr(self, "_update_toast", None) is toast:
                        toast.destroy()
                        self._update_toast = None

                def on_update():
                    dismiss()
                    self._on_check_update()

                ttk.Button(toast, text="Update", command=on_update).pack(
                    side="left", padx=(0, DpiScale.scaled(6, s)), pady=DpiScale.scaled(4, s)
                )
                ttk.Button(toast, text="✕", width=2, command=dismiss).pack(
                    side="left", padx=(0, DpiScale.scaled(6, s)), pady=DpiScale.scaled(4, s)
                )
                self.after(12000, dismiss)

            self.after(0, show_toast)

        threading.Thread(target=work, daemon=True).start()

    def _apply_menubar_visibility(self):
        if self.setting.p60_hide_menubar.get():
            self.config(menu="")
        else:
            self.config(menu=self.menubar)

    def set_set_light_mode_darker(self):
        self.configure(bg="gray14")
        self.log_view.log_area.config(bg="gray14", fg="white", highlightbackground="black")
        self.setting_shortcut_view.help_text.config(bg="gray14", fg="white", highlightbackground="black")

        self.setting.p60_bg_color = "#333333"
        self.setting.p60_fg_color = "white"
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
            if self.setting.p67_global_plan_view.get():
                self.global_plan_plot_view.plot()
            if self.setting.p67_local_plan_view.get():
                self.local_plan_plot_view.plot()

        if not self.setting.p60_shortcut_mode.get():
            self.setting.vehicle_state.set( f"Loc: ({self.exec.ego_state.x:+7.2f}, {self.exec.ego_state.y:+7.2f}), Vel: {self.exec.ego_state.velocity:5.2f} ({self.exec.ego_state.velocity*3.6:6.2f} km/h), θ: {self.exec.ego_state.theta:+5.1f}")
            if self.exec.local_planner is not None:
                self.setting.current_wp.set(str(self.exec.local_planner.global_trajectory.current_wp))
                self.setting.lap.set(f"{self.exec.local_planner.lap:5d}")
            else:
                self.setting.current_wp.set("0")
                self.setting.lap.set("0")

            # TODO: need to connect to a tkinter variable instead
            if self.exec.controller is not None:
                self.perceive_plan_control_view.control_frame.gauge_cte_vel.set_value(self.exec.controller.cte_velocity)
                self.perceive_plan_control_view.control_frame.gauge_cte_steer.set_value(self.exec.controller.cte_steer)
                self.perceive_plan_control_view.control_frame.gauge_acc.set_value(self.exec.controller.cmd.acceleration)
                self.perceive_plan_control_view.control_frame.gauge_steer.set_value(self.exec.controller.cmd.steer)

            self.setting.elapsed_real_time.set(f"{self.exec.elapsed_real_time:6.2f}")
            self.setting.elapsed_sim_time.set(f"{self.exec.elapsed_sim_time:6.2f}")
            self.setting.replan_fps.set(f"{self.exec.planner_fps:6.1f}")
            self.setting.control_fps.set(f"{self.exec.control_fps:6.1f}")
            self.setting.perception_fps.set(f"{self.exec.perception_fps:6.1f}")


        log.debug("UI Update Time: %.2f ms", (time.time() - t1) * 1000)

    def apply_global_plan(
        self,
        global_plan: GlobalPlan,
        ego_xy: Optional[tuple[float, float]] = None,
    ) -> None:
        """Push a global plan to the local planner and controller."""
        if self.exec is None:
            return
        # ROSExecuter (and similar) own proxy update + worker publish.
        if "apply_global_plan" in vars(type(self.exec)):
            self.exec.apply_global_plan(global_plan, ego_xy=ego_xy)
            return
        if global_plan is None or global_plan.trajectory is None:
            log.error("apply_global_plan: plan or trajectory is None")
            return
        if len(global_plan.trajectory.path_s) == 0:
            log.error("apply_global_plan: trajectory is empty")
            return
        ego_xy = ego_xy if ego_xy is not None else (self.exec.ego_state.x, self.exec.ego_state.y)
        if self.exec.local_planner:
            self.exec.local_planner.set_global_plan(global_plan, ego_xy=ego_xy)
        if self.exec.controller:
            self.exec.controller.set_trajectory_tracker(global_plan.trajectory)
            self.exec.controller.reset()

    def replan_global(self) -> None:
        """Recompute the global plan from the current ego pose and hand it to the local planner.

        Keeps the existing goal, moves the start to the ego's current position,
        re-runs the global planner, then pushes the resulting plan to the local
        planner and controller so subsequent ticks follow the new route.
        """
        if self.exec is None:
            return
        if not self.exec.global_planner:
            log.error("Global replan failed: no global planner.")
            return
        goal = self.exec.global_planner.goal_point
        if goal is not None:
            self.exec.global_planner.set_start_goal(
                (self.exec.ego_state.x, self.exec.ego_state.y), goal
            )

        new_plan = self.exec.global_planner.plan(
            perception_model=self.exec.pm,
            sensors=self.exec.world.get_sensor_frame(),
        )
        if new_plan is None or new_plan.trajectory is None:
            log.error("Global replan failed: planner returned no valid plan.")
            return

        self.apply_global_plan(new_plan)
        log.info(f"Global replan complete; {len(new_plan.left_boundary_d)} boundary pts")

    def spawn_agent(self, agent_state: AgentState) -> None:
        """Spawn an agent in the world using the ego's current global plan."""
        if self.exec is None:
            return
        global_plan = (
            self.exec.local_planner.global_plan if self.exec.local_planner else None
        )
        self.exec.world.spawn_agent(agent_state, global_plan=global_plan)

    def load_settings(self, only_stack=False, profile=None):
        """Load settings from a profile or the current settings. Uses c55_setting_utils files plus UI housekeeping."""
        
        if profile:
            self.setting.c60_selected_profile.set(profile)
        else:
            profile = self.setting.c60_selected_profile.get() 
        # load_setting(PerceptionSettings, profile=profile)
        # load_setting(PlanningSettings, profile=profile)
        # load_setting(ControlSettings, profile=profile)
        # load_setting(ExecutionSettings, profile=profile)
        binder = TkSettingsBinder()
        load_setting(AppSettings, profile=profile)
        self.setting.sync_app_from_singleton()
        if not only_stack:
            load_setting(self.setting, profile=profile, binder=binder)
        load_stack_settings(profile=profile)
        sync_stack_settings_to_ui(self.setting)
        self.setting.default_map_file.set(DataPicker.default_map_display_path())
        self.setting.default_global_plan_file.set(DataPicker.default_global_plan_display_path())
        self.perceive_plan_control_view.perceive_frame.refresh_default_map_tooltips()
        self.setting_shortcut_view.update_setting_window()
        log.info(f"Loaded settings from profile: {profile}")

        self.log_view.reset()
        ConfigPaths.set_startup_profile(profile)
        if hasattr(self, "setting_shortcut_view"):
            self.setting_shortcut_view.toggle_dark_mode()
        if hasattr(self, "perceive_plan_control_view"):
            self.perceive_plan_control_view.reset()
        self.update_views()

    def on_community_plugins_changed(self) -> None:
        """Reload profile stack settings and refresh UI after plugin install/uninstall."""
        self.load_settings(only_stack=True)
        if self.setting.c62_load_plugins.get():
            self.reload_stack(reload_code=True)
        else:
            self.perceive_plan_control_view.reset()
            self.exec_visualize_view.update_data()

    def refresh_pipeline(self):
        """Reconstruct active pipeline strategies from current settings (no full stack reload)."""
        if self.exec is None:
            return
        # Re-import so construction uses post-reload_lib classes; match by name
        # because isinstance fails across importlib.reload class identities.
        from avlite.c10_perception.c12_perception_strategy import PerceptionPipeline as PercPipe
        from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningPipeline as LocPipe

        if type(self.exec.perception).__name__ == "PerceptionPipeline":
            self.exec.perception = PercPipe(self.exec.pm)
        if self.exec.local_planner is not None and type(self.exec.local_planner).__name__ == "LocalPlanningPipeline":
            lp = self.exec.local_planner
            self.exec.local_planner = LocPipe(global_plan=lp.global_plan, env=self.exec.pm)
        try:
            self.exec._validate_stack()
        except ValueError as e:
            log.error(f"Pipeline refresh failed: {e}")

        self.update_ui()

    def reload_stack(self, reload_code: bool = True, preserve_plot_layout: bool = False):
        if reload_code:
            self.show_loading_overlay("Reloading stack...")
        else:
            self.show_loading_overlay("Reinitializing stack...")

        self.exec_visualize_view.stop_exec()
        self.disable_frame(self)
        if not preserve_plot_layout:
            self.local_plan_plot_view.grid_forget()
            self.global_plan_plot_view.grid_forget()

        error = None
        try:
            if reload_code:
                reload_lib(exclude_settings=True, reload_plugins=self.setting.c62_load_plugins.get())
            self.exec = executor_factory(
                executer_type=self.setting.executer_type.get(),
                bridge=self.setting.execution_bridge.get(),
                perception_strategy_name=self.setting.perception_type.get(),
                localization_strategy_name=self.setting.localization_type.get(),
                mapping_strategy_name=self.setting.mapping_type.get(),
                global_planner_strategy_name=self.setting.global_planner_type.get(),
                local_planner_strategy_name=self.setting.local_planner_type.get(),
                controller_strategy_name=self.setting.controller_type.get(),
                execution_task_names=self.setting.execution_task_names(),
                perception_dt=self.setting.perception_dt.get(),
                localization_dt=self.setting.localization_dt.get(),
                replan_dt=self.setting.replan_dt.get(),
                control_dt=self.setting.control_dt.get(),
                map_file=ExecutionSettings.c40_map,
                default_global_trajectory_file=self.setting.default_global_plan_file.get(),
                load_plugins=self.setting.c62_load_plugins.get(),
                async_combined_perception_planning=ExecutionSettings.c40_async_combined_perception_planning,
            )

            self.setting.default_map_file.set(DataPicker.default_map_display_path())
            self.perceive_plan_control_view.perceive_frame.refresh_default_map_tooltips()

        except Exception as e:
            error = e
            log.error(f"Error reloading stack: {e}", exc_info=True)
        finally:
            self.local_plan_plot_view.reset()
            self.global_plan_plot_view.reset()
            self.perceive_plan_control_view.reset()
            self.exec_visualize_view.update_data()
            self.update_views()
            if not preserve_plot_layout and self.winfo_viewable():
                self.update_ui()
            self.enable_frame(self)
            if self.exec is not None:
                self.exec_visualize_view.bridge_frame.update_for_bridge(
                    self.exec.world.world_capabilities, self.exec.world.stack_capabilities
                )
            self.focus_set()
            self.hide_loading_overlay()

        if error is not None:
            messagebox.showerror(
                "Reload failed",
                f"Failed to rebuild the stack.\n\n{error}",
                parent=self,
            )

    def switch_profile(self):
        self.exec_visualize_view.stop_exec()
        self.load_settings(profile=self.setting.p60_next_profile.get(), only_stack=False)
        # self.reload_stack(reload_code=False)
        self.update_views()
        self.update_ui()

    def on_stack_settings_changed(self) -> None:
        if hasattr(self, "perceive_plan_control_view"):
            self.perceive_plan_control_view.reset()
        if hasattr(self, "exec_visualize_view"):
            self.exec_visualize_view.update_data()


class VisualizationApp(AppStrategy):
    """Default app: the AVLite visualizer GUI (runs when no subcommand is given)."""

    cli_name = None
    help = "Real-time visualizer GUI (default)"

    def run(self, args, unknown):
        app = VisualizerApp()
        app.mainloop()
