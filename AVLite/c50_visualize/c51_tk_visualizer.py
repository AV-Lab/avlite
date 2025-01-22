from c10_perceive.c12_state import AgentState
import c50_visualize.c52_plot as c52_plot
from c40_execute.c41_executer import Executer

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import logging
from tkinter.scrolledtext import ScrolledText
import sys


log = logging.getLogger(__name__)
log_blacklist = set()  # used to filter 'excute', 'plan', 'control' subpackage logs

# import TKinterModernThemes as TKMT


class VisualizerApp(tk.Tk):
    exe: Executer

    def __init__(self, executer: Executer, code_reload_function=None, only_visualize=False):
        super().__init__()

        self.exec = executer
        self.code_reload_function = code_reload_function

        self.title("Path Planning Visualization")
        self.geometry("1200x1100")

        # ----------------------------------------------------------------------
        # Variables for checkboxes --------------------------------------------
        # ----------------------------------------------------------------------
        self.plot_only_UI = tk.BooleanVar(value=only_visualize)
        self.dark_mode = tk.BooleanVar(value=True)

        self.show_legend = tk.BooleanVar(value=True)
        self.show_past_locations = tk.BooleanVar(value=True)
        self.show_global_plan = tk.BooleanVar(value=True)
        self.show_local_plan = tk.BooleanVar(value=True)
        self.show_local_lattice = tk.BooleanVar(value=True)
        self.show_state = tk.BooleanVar(value=True)

        # Exec Options
        self.exec_plan = tk.BooleanVar(value=True)
        self.exec_control = tk.BooleanVar(value=True)
        self.exec_perceive = tk.BooleanVar(value=True)

        self.animation_running = False

        self.exec_option = tk.StringVar(value="Basic")
        self.debug_option = tk.StringVar(value="INFO")

        self.show_perceive_logs = tk.BooleanVar(value=True)
        self.show_plan_logs = tk.BooleanVar(value=True)
        self.show_control_logs = tk.BooleanVar(value=True)
        self.show_execute_logs = tk.BooleanVar(value=True)
        self.show_vis_logs = tk.BooleanVar(value=True)

        # ----------------------------------------------------------------------
        # Key Bindings --------------------------------------------------------
        # ----------------------------------------------------------------------
        self.bind("Q", lambda e: self.quit())
        self.bind("R", lambda e: self._reload_stack())
        self.bind("D", lambda e: self._toggle_dark_mode_shortcut())
        self.bind("S", lambda e: self._shortcut_mode())

        self.bind("x", lambda e: self.toggle_exec())
        self.bind("c", lambda e: self.step_exec())
        self.bind("t", lambda e: self.reset_exec())

        self.bind("n", lambda e: self.step_plan())
        self.bind("r", lambda e: self.replan())

        self.bind("h", lambda e: self.step_control())
        self.bind("g", lambda e: self.align_control())

        self.bind("a", lambda e: self.step_steer_left())
        self.bind("d", lambda e: self.step_steer_right())
        self.bind("w", lambda e: self.step_acc())
        self.bind("s", lambda e: self.step_dec())

        self.bind("<Control-plus>", lambda e: self.zoom_in_frenet())
        self.bind("<Control-minus>", lambda e: self.zoom_out_frenet())
        self.bind("<plus>", lambda e: self.zoom_in())
        self.bind("<minus>", lambda e: self.zoom_out())

        # ----------------------------------------------------------------------
        # -Plot Frame ----------------------------------------------------------
        # ----------------------------------------------------------------------

        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        self.xy_zoom = 30
        self.frenet_zoom = 30

        self.fig = c52_plot.fig
        self.ax1 = c52_plot.ax1
        self.ax2 = c52_plot.ax2
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.after(300, self._replot)

        self.canvas.mpl_connect("scroll_event", self.on_mouse_scroll)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_click)
        self._prev_scroll_time = None  # used to throttle the replot

        # ----------------------------------------------------------------------
        # Config frame
        # ------------------------------------------------------
        config_frame = ttk.LabelFrame(self, text="Config")
        config_frame.pack(fill=tk.X, side=tk.TOP)
        ttk.Button(config_frame, text="Reload Code", command=self._reload_stack).pack(side=tk.RIGHT)
        ttk.Checkbutton(
            config_frame,
            text="Shortcut Mode",
            variable=self.plot_only_UI,
            command=self._update_UI,
        ).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton(
            config_frame,
            text="Dark Mode",
            variable=self.dark_mode,
            command=self._toggle_dark_mode,
        ).pack(anchor=tk.W, side=tk.LEFT)

        # ----------------------------------------------------------------------
        # Shortcut frame
        # ------------------------------------------------------
        self.shortcut_frame = ttk.LabelFrame(self, text="Shortcuts")
        self.help_text = tk.Text(self.shortcut_frame, wrap=tk.WORD, width=50, height=7)
        key_binding_info = """
Perceive: 
Plan:     n - Step plan        r - Replan            
Control:  h - Control Step     g - Re-align control         w - Accelerate 
          a - Steer left       d - Steer right              s - Deccelerate
Visalize: Q - Quit             S - Toggle shortcut          D - Toggle Dark Mode        R - Reload imports     
          + - Zoom In          - - Zoom Out           <Ctrl+> - Zoom In F         <Ctrl-> - Zoom Out F
Execute:  c - Step Execution   t - Reset execution          x - Toggle execution
         """.strip()
        self.help_text.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.help_text.insert(tk.END, key_binding_info)
        self.help_text.config(state=tk.DISABLED)  # Make the text area read-only


        # ----------------------------------------------------------------------
        # Percieve Plan Control Frame -----------------------------------------
        # ----------------------------------------------------------------------

        self.perceive_plan_control_frame = ttk.Frame(self)
        self.perceive_plan_control_frame.pack(fill=tk.X)
        # ----------------------------------------------------------------------
        ## Perceive Frame
        # ----------------------------------------------------------------------
        self.perceive_frame = ttk.LabelFrame(self.perceive_plan_control_frame, text="Perceive")
        self.perceive_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.vehicle_state_label = ttk.Label(self.perceive_frame, text="")
        self.vehicle_state_label.pack(side=tk.TOP, expand=True, fill=tk.X, pady=5)

        self.coordinates_label = ttk.Label(self.perceive_frame, text="Spawn Agent: Click on the plot.")
        self.coordinates_label.pack(side=tk.LEFT, pady=5)
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        ## Plan frame
        # ----------------------------------------------------------------------
        self.plan_frame = ttk.LabelFrame(self.perceive_plan_control_frame, text="Plan (Manual)")
        self.plan_frame.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=5, pady=5)

        wp_frame = ttk.Frame(self.plan_frame)
        wp_frame.pack(fill=tk.X)

        ttk.Button(wp_frame, text="Set Waypoint", command=self.set_waypoint).pack(side=tk.LEFT)
        self.global_tj_wp_entry = ttk.Entry(wp_frame, width=6)
        self.global_tj_wp_entry.insert(0, "0")
        self.global_tj_wp_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(wp_frame, text=f"{len(self.exec.planner.global_trajectory.path_x)-1}").pack(side=tk.LEFT, padx=5)

        ttk.Button(self.plan_frame, text="Replan", command=self.replan).pack(side=tk.LEFT)
        ttk.Button(self.plan_frame, text="Step", command=self.step_plan).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ----------------------------------------------------------------------
        ## Control Frame
        # ----------------------------------------------------------------------
        self.control_frame = ttk.LabelFrame(self.perceive_plan_control_frame, text="Control (Manual)")
        self.control_frame.pack(fill=tk.X, expand=True, side=tk.LEFT)
        dt_frame = ttk.Frame(self.control_frame)
        dt_frame.pack(fill=tk.X)
        ttk.Label(dt_frame, text="Δt ").pack(side=tk.LEFT, padx=5, pady=5)
        self.dt_entry = ttk.Entry(dt_frame, width=5)
        self.dt_entry.insert(2, "0.1")
        self.dt_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(dt_frame, text="Control Step", command=self.step_control).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dt_frame, text="Re-align", command=self.align_control).pack(side=tk.LEFT)  # Re-alignes with plan

        ttk.Button(self.control_frame, text="Steer Left", command=self.step_steer_left).pack(side=tk.LEFT)
        ttk.Button(self.control_frame, text="Steer Right", command=self.step_steer_right).pack(side=tk.LEFT)
        ttk.Button(self.control_frame, text="Accelerate", command=self.step_acc).pack(side=tk.LEFT)
        ttk.Button(self.control_frame, text="Deccelerate", command=self.step_dec).pack(side=tk.LEFT)

        # ----------------------------------------------------------------------
        # -End of Perceive Plan Contorl Frame --------------------------------------
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # Visualize + Exec ------------------------------------------------
        # ----------------------------------------------------------------------
        self.vis_exec_frame = ttk.Frame(self)
        self.vis_exec_frame.pack(fill=tk.X)

        # ----------------------------------------------------------------------
        ## Execute Frame
        # ----------------------------------------------------------------------
        self.execution_frame = ttk.LabelFrame(self.vis_exec_frame, text="Execute (Auto)")
        self.execution_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        exec_first_frame = ttk.Frame(self.execution_frame)
        exec_first_frame.pack(fill=tk.X)
        exec_second_frame = ttk.Frame(self.execution_frame)
        exec_second_frame.pack(fill=tk.X)
        exec_third_frame = ttk.Frame(self.execution_frame)
        exec_third_frame.pack(fill=tk.X)

        ttk.Label(exec_first_frame, text="Control Δt ").pack(side=tk.LEFT, padx=5, pady=5)
        self.dt_exec_cn_entry = ttk.Entry(exec_first_frame, width=5)
        self.dt_exec_cn_entry.insert(0, "0.02")
        self.dt_exec_cn_entry.pack(side=tk.LEFT)

        ttk.Label(exec_first_frame, text="Replan Δt ").pack(side=tk.LEFT, padx=5, pady=5)
        self.dt_exec_pl_entry = ttk.Entry(exec_first_frame, width=5)
        self.dt_exec_pl_entry.insert(0, "1.7")
        self.dt_exec_pl_entry.pack(side=tk.LEFT)

        gruvbox_green = "#b8bb26"
        gruvbox_light_green = "#fe8019"
        gruvbox_orange = "#d65d0e"
        self.start_exec_button = tk.Button(
            exec_second_frame,
            text="Start",
            command=self.toggle_exec,
            bg=gruvbox_orange,
            fg="white",
            borderwidth=0,
            highlightthickness=0,
            width=10,
        )
        self.start_exec_button.pack(fill=tk.X, side=tk.LEFT)

        ttk.Button(exec_second_frame, text="Stop", command=self.stop_exec).pack(side=tk.LEFT)
        ttk.Button(exec_second_frame, text="Step", command=self.step_exec).pack(side=tk.LEFT)
        ttk.Button(exec_second_frame, text="Reset", command=self.reset_exec).pack(side=tk.LEFT)

        ttk.Label(exec_third_frame, text="Bridge:").pack(side=tk.LEFT)
        ttk.Radiobutton(
            exec_third_frame,
            text="Basic",
            variable=self.exec_option,
            value="Basic",
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(exec_third_frame, text="ROS", variable=self.exec_option, value="ROS").pack(side=tk.LEFT)
        ttk.Radiobutton(exec_third_frame, text="Carla", variable=self.exec_option, value="Carla").pack(side=tk.LEFT)

        # ttk.Checkbutton(exec_third_frame, text="Control", variable=self.exec_control).pack(side=tk.RIGHT)
        # ttk.Checkbutton(exec_third_frame, text="Plan", variable=self.exec_plan).pack(side=tk.RIGHT)
        # ttk.Checkbutton(exec_third_frame, text="Percieve", variable=self.exec_perceive).pack(side=tk.RIGHT)

        # ----------------------------------------------------------------------
        # Visualize frame setup
        # ----------------------------------------------------------------------
        self.visualize_frame = ttk.LabelFrame(self.vis_exec_frame, text="Visualize")
        self.visualize_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ## UI Elements for Visualize - Checkboxes
        checkboxes_frame = ttk.Frame(self.visualize_frame)
        checkboxes_frame.pack(fill=tk.X)
        ttk.Checkbutton(
            checkboxes_frame,
            text="Legend",
            variable=self.show_legend,
            state="selected",
            command=self._replot,
        ).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton(
            checkboxes_frame,
            text="Locations",
            variable=self.show_past_locations,
            state="selected",
            command=self._replot,
        ).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton(
            checkboxes_frame,
            text="Global Plan",
            variable=self.show_global_plan,
            state="selected",
            command=self._replot,
        ).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton(
            checkboxes_frame,
            text="Local Plan",
            variable=self.show_local_plan,
            state="selected",
            command=self._replot,
        ).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton(
            checkboxes_frame,
            text="Local Lattice",
            variable=self.show_local_lattice,
            state="selected",
            command=self._replot,
        ).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton(
            checkboxes_frame,
            text="State",
            variable=self.show_state,
            state="selected",
            command=self._replot,
        ).pack(anchor=tk.W, side=tk.LEFT)

        ## UI Elements for Visualize - Buttons
        zoom_global_frame = ttk.Frame(self.visualize_frame)
        zoom_global_frame.pack(fill=tk.X, padx=5)

        ttk.Label(zoom_global_frame, text="Global Coordinate").pack(anchor=tk.W, side=tk.LEFT)
        ttk.Button(zoom_global_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT)
        ttk.Button(zoom_global_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT)
        zoom_frenet_frame = ttk.Frame(self.visualize_frame)
        zoom_frenet_frame.pack(fill=tk.X, padx=5)
        ttk.Label(zoom_frenet_frame, text="Frenet Coordinate").pack(anchor=tk.W, side=tk.LEFT)
        ttk.Button(zoom_frenet_frame, text="Zoom In", command=self.zoom_in_frenet).pack(side=tk.LEFT)
        ttk.Button(zoom_frenet_frame, text="Zoom Out", command=self.zoom_out_frenet).pack(side=tk.LEFT)

        # ----------------------------------------------------------------------
        # - Log Frame
        # ----------------------------------------------------------------------
        self.log_frame = ttk.LabelFrame(self, text="Log")
        self.log_frame.pack(fill=tk.X)

        log_cb_frame = ttk.Frame(self.log_frame)
        log_cb_frame.pack(fill=tk.X)

        self.ck_perceive = ttk.Checkbutton(
            log_cb_frame,
            text="Perceive",
            state = "selected",
            variable=self.show_perceive_logs,
            command=self.update_log_filter,
        )
        self.ck_perceive.pack(side=tk.LEFT)
        self.ck_perceive.state(["!alternate"])
        self.ck_perceive.state(["selected"])

        self.ck_plan = ttk.Checkbutton(
            log_cb_frame,
            text="Plan",
            state = "selected",
            variable=self.show_plan_logs,
            command=self.update_log_filter,
        )
        self.ck_plan.pack(side=tk.LEFT)
        self.ck_plan.state(["!alternate"])
        self.ck_plan.state(["selected"])
        self.ck_control = ttk.Checkbutton(
            log_cb_frame,
            text="Control",
            state = "selected",
            variable=self.show_control_logs,
            command=self.update_log_filter,
        )
        self.ck_control.pack(side=tk.LEFT)
        self.ck_control.state(["!alternate"])
        self.ck_control.state(["selected"])
        self.ck_exec = ttk.Checkbutton(
            log_cb_frame,
            text="Execute",
            state = "selected",
            variable=self.show_execute_logs,
            command=self.update_log_filter,
        )
        self.ck_exec.pack(side=tk.LEFT)
        self.ck_exec.state(["!alternate"])
        self.ck_exec.state(["selected"])

        self.ck_vis = ttk.Checkbutton(
            log_cb_frame,
            text="Visualize",
            state = "selected",
            variable=self.show_vis_logs,
            command=self.update_log_filter,
        )
        self.ck_vis.pack(side=tk.LEFT)
        self.ck_vis.state(["!alternate"])
        self.ck_vis.state(["selected"])

        self.rb_db_stdout = ttk.Radiobutton(
            log_cb_frame,
            text="STDOUT",
            variable=self.debug_option,
            value="STDOUT",
            command=self.update_log_level,
        )
        self.rb_db_stdout.pack(side=tk.RIGHT)
        self.rb_db_warn = ttk.Radiobutton(
            log_cb_frame,
            text="WARN",
            variable=self.debug_option,
            value="WARN",
            command=self.update_log_level,
        )
        self.rb_db_warn.pack(side=tk.RIGHT)
        self.rb_db_info = ttk.Radiobutton(
            log_cb_frame,
            text="INFO",
            variable=self.debug_option,
            value="INFO",
            command=self.update_log_level,
        )
        self.rb_db_info.pack(side=tk.RIGHT)
        self.rb_db_debug = ttk.Radiobutton(
            log_cb_frame,
            text="DEBUG",
            variable=self.debug_option,
            value="DEBUG",
            command=self.update_log_level,
        )
        self.rb_db_debug.pack(side=tk.RIGHT)

        ttk.Label(log_cb_frame, text="Log Level:").pack(side=tk.RIGHT)

        self.log_area = ScrolledText(self.log_frame, state="disabled", height=8)
        self.log_area.pack(fill=tk.BOTH, expand=True)

        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # -End of UI Elements---------------------------------------------------
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------

        self.__dark_mode()

        # -------------------------------------------
        # -Configure logging-------------------------
        # -------------------------------------------
        logger = logging.getLogger()
        text_handler = VisualizerApp.LogTextHandler(self.log_area)
        formatter = logging.Formatter("[%(levelname).4s] %(name)-33s (L: %(lineno)3d): %(message)s")
        text_handler.setFormatter(formatter)
        # remove other handlers to avoid duplicate logs
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        logger.addHandler(text_handler)
        logger.setLevel(logging.INFO)
        log.info("Log initialized.")

    def _shortcut_mode(self):
        if self.plot_only_UI.get():
            self.plot_only_UI.set(False)
        else:
            self.plot_only_UI.set(True)
        self._update_UI()

    def _update_UI(self):
        if self.plot_only_UI.get():
            self.vis_exec_frame.pack_forget()
            self.perceive_plan_control_frame.pack_forget()
            self.log_frame.pack_forget()

            self.shortcut_frame.pack(fill=tk.X, side=tk.TOP)
            self.log_frame.pack(fill=tk.X)
        else:
            self.shortcut_frame.pack_forget()
            self.log_frame.pack_forget()

            self.vis_exec_frame.pack(fill=tk.X, side=tk.TOP)
            self.perceive_plan_control_frame.pack(fill=tk.X)
            self.log_area.pack(fill=tk.BOTH, expand=True)
            self.log_frame.pack(fill=tk.X)
        self._replot()

    def _toggle_dark_mode(self):
        self.__dark_mode() if self.dark_mode.get() else self.__light_mode()

    def _toggle_dark_mode_shortcut(self):
        if self.dark_mode.get():
            self.dark_mode.set(False)
        else:
            self.dark_mode.set(True)
        self._toggle_dark_mode()

    def __dark_mode(self):
        self.configure(bg="black")
        self.log_area.config(bg="gray14", fg="white", highlightbackground="black")
        self.help_text.config(bg="gray14", fg="white", highlightbackground="black")
        c52_plot.set_plot_theme(bg_color="#2d2d2d", fg_color="white")

        try:
            from ttkthemes import ThemedStyle

            style = ThemedStyle(self)
            style.set_theme("equilux")

        except ImportError:
            log.error("Please install ttkthemes to use dark mode.")

    def __light_mode(self):
        self.configure(bg="white")
        self.log_area.config(bg="white", fg="black")
        self.help_text.config(bg="white", fg="black")
        c52_plot.set_plot_theme(bg_color="white", fg_color="black")
        # reset the theme
        try:
            from ttkthemes import ThemedStyle

            style = ThemedStyle(self)
            style.set_theme("yaru")
        except ImportError:
            log.error("Please install ttkthemes to use dark mode.")

    def _replot(self):
        canvas_widget = self.canvas.get_tk_widget()
        width = canvas_widget.winfo_width()
        height = canvas_widget.winfo_height()
        aspect_ratio = width / height

        t1 = time.time()
        # self.canvas.restore_region(self.plt_background)
        c52_plot.plot(
            exec=self.exec,
            aspect_ratio=aspect_ratio,
            xy_zoom=self.xy_zoom,
            frenet_zoom=self.frenet_zoom,
            show_legend=self.show_legend.get(),
            plot_last_pts=self.show_past_locations.get(),
            plot_global_plan=self.show_global_plan.get(),
            plot_local_plan=self.show_local_plan.get(),
            plot_local_lattice=self.show_local_lattice.get(),
            plot_state=self.show_state.get(),
        )
        self.canvas.draw()
        log.debug(f"Plot Time: {(time.time()-t1)*1000:.2f} ms")
        self.vehicle_state_label.config(
            text=f"Ego State: X: {self.exec.ego_state.x:+.2f}, Y: {self.exec.ego_state.y:+.2f}, v: {self.exec.ego_state.speed:+.2f}, θ: {self.exec.ego_state.theta:+.2f}"
        )

        self.global_tj_wp_entry.delete(0, tk.END)
        self.global_tj_wp_entry.insert(0, str(self.exec.planner.global_trajectory.current_wp))

    def _reload_stack(self):
        if self.code_reload_function is not None:
            self.exec = self.code_reload_function()
            self._replot()
        else:
            log.warning("No code reload function provided.")

    def on_mouse_move(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            if event.inaxes == self.ax1:
                self.coordinates_label.config(text=f"Spawn Agent: X: {x:.2f}, Y: {y:.2f}")
            elif event.inaxes == self.ax2:
                self.coordinates_label.config(text=f"Spawn Agent: S: {x:.2f}, D: {y:.2f}")
        else:
            # Optionally, clear the coordinates display when the mouse is not over the axes
            self.coordinates_label.config(text="Spawn Agent: Click on the plot.")


    def on_mouse_click(self, event):
        if event.inaxes == self.ax1:
            x, y = event.xdata, event.ydata
            self.exec.spawn_agent(x=x,y=y)

            self._replot()
        elif event.inaxes == self.ax2:
            s, d = event.xdata, event.ydata
            self.exec.spawn_agent(d=d, s=s)
            self._replot()

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
        if (self._prev_scroll_time is None or time.time() - self._prev_scroll_time > threshold) and not self.animation_running:
            self._replot()

        self._prev_scroll_time = time.time()

    def zoom_in(self):
        self.xy_zoom -= 5 if self.xy_zoom > 5 else 0
        self._replot()

    def zoom_out(self):
        self.xy_zoom += 5
        self._replot()

    def zoom_in_frenet(self):
        self.frenet_zoom -= 5 if self.frenet_zoom > 5 else 0
        self._replot()

    def zoom_out_frenet(self):
        self.frenet_zoom += 5
        self._replot()

    # --------------------------------------------------------------------------------------------
    # -Plan---------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------

    def set_waypoint(self):
        timestep_value = int(self.global_tj_wp_entry.get())
        self.exec.planner.reset(wp=timestep_value)
        self._replot()

    def replan(self):
        t1 = time.time()
        self.exec.planner.replan()
        t2 = time.time()
        log.info(f"Re-plan Time: {(t2-t1)*1000:.2f} ms")
        self._replot()

    def step_plan(self):
        # Placeholder for the method to step to the next waypoint
        t1 = time.time()
        self.exec.planner.step_wp()
        log.info(f"Plan Step Time: {(time.time()-t1)*1000:.2f} ms")
        self.global_tj_wp_entry.delete(0, tk.END)
        self.global_tj_wp_entry.insert(0, str(self.exec.planner.global_trajectory.next_wp - 1))
        self._replot()

    # --------------------------------------------------------------------------------------------
    # -Control------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------

    def step_control(self):
        d = self.exec.planner.traversed_d[-1]
        steer = self.exec.controller.control(d)

        dt = float(self.dt_entry.get())
        self.exec.update_ego_state(steering_angle=steer, dt=dt)
        self._replot()

    def align_control(self):
        self.exec.ego_state.x = self.exec.planner.global_trajectory.path_x[self.exec.planner.global_trajectory.next_wp - 1]
        self.exec.ego_state.y = self.exec.planner.global_trajectory.path_y[self.exec.planner.global_trajectory.next_wp - 1]

        self._replot()

    def step_steer_left(self):
        dt = float(self.dt_entry.get())
        self.exec.update_ego_state(dt=dt, steering_angle=0.1)
        self._replot()

    def step_steer_right(self):
        dt = float(self.dt_entry.get())
        self.exec.update_ego_state(dt=dt, steering_angle=-0.1)
        self._replot()

    def step_acc(self):
        dt = float(self.dt_entry.get())
        self.exec.update_ego_state(dt=dt, acceleration=8)
        self._replot()

    def step_dec(self):
        dt = float(self.dt_entry.get())
        self.exec.update_ego_state(dt=dt, acceleration=-8)
        self._replot()

    # --------------------------------------------------------------------------------------------
    # -SIM----------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------

    def toggle_exec(self):
        if self.animation_running:
            self.stop_exec()
            return
        self.animation_running = True
        self.start_exec_button.config(state=tk.DISABLED)
        self._exec_loop()

    def _exec_loop(self):
        if self.animation_running:
            cn_dt = float(self.dt_exec_cn_entry.get())
            pl_dt = float(self.dt_exec_pl_entry.get())

            self.exec.step(control_dt=cn_dt, replan_dt=pl_dt)
            self._replot()
            self.global_tj_wp_entry.delete(0, tk.END)
            self.global_tj_wp_entry.insert(0, str(self.exec.planner.global_trajectory.next_wp - 1))
            self.after(int(cn_dt * 1000), self._exec_loop)

    def stop_exec(self):
        self.animation_running = False
        self.start_exec_button.config(state=tk.NORMAL)

    def step_exec(self):
        cn_dt = float(self.dt_exec_cn_entry.get())
        pl_dt = float(self.dt_exec_pl_entry.get())
        self.exec.step(control_dt=cn_dt, replan_dt=pl_dt)
        self._replot()

    def reset_exec(self):
        self.exec.reset()
        self._replot()

    def update_log_filter(self):
        log.info("Log filter updated.")
        # based on blacklist, LogTextHandler will filter out the logs
        (log_blacklist.discard("c10_perceive") if self.show_perceive_logs.get() else log_blacklist.add("c10_perceive"))
        (log_blacklist.discard("c20_plan") if self.show_plan_logs.get() else log_blacklist.add("c20_plan"))
        (log_blacklist.discard("c30_control") if self.show_control_logs.get() else log_blacklist.add("c30_control"))
        (log_blacklist.discard("c40_execute") if self.show_execute_logs.get() else log_blacklist.add("c40_execute"))
        (log_blacklist.discard("c50_visualize") if self.show_vis_logs.get() else log_blacklist.add("c50_visualize"))
        


    def update_log_level(self):
        if self.rb_db_debug.instate(["selected"]):
            logging.getLogger().setLevel(logging.DEBUG)
            log.debug("Log setting updated to DEBUG.")
        elif self.rb_db_info.instate(["selected"]):
            logging.getLogger().setLevel(logging.INFO)
            log.info("Log setting updated to INFO.")
        elif self.rb_db_warn.instate(["selected"]):
            logging.getLogger().setLevel(logging.WARNING)
            log.warn("Log setting updated to WARNING.")

        if self.rb_db_stdout.instate(["selected"]):
            logging.getLogger().setLevel(logging.CRITICAL)
            sys.stdout = VisualizerApp.TextRedirector(self.log_area)
        else:
            sys.stdout = sys.__stdout__
        print("Log setting updated: routing CRITICAL and stdout.")

    class LogTextHandler(logging.Handler):
        def __init__(self, text_widget):
            super().__init__()
            self.text_widget = text_widget

        def emit(self, record):
            for bl in log_blacklist:
                if bl + "." in record.name:
                    return
            msg = self.format(record)
            self.text_widget.configure(state="normal")
            self.text_widget.insert(tk.END, msg + "\n")
            self.text_widget.configure(state="disabled")
            self.text_widget.yview(tk.END)

    class TextRedirector(object):
        def __init__(self, widget):
            self.widget = widget

        def write(self, str):
            self.widget.configure(state="normal")
            self.widget.insert(tk.END, str)
            self.widget.configure(state="disabled")
            self.widget.see(tk.END)

        def flush(self):
            pass
