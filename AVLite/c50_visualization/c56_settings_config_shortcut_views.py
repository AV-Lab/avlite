from __future__ import annotations
from os import wait
from typing import TYPE_CHECKING
from c60_tools.c61_utils import save_config, load_setting
from c50_visualization.c58_ui_lib import ThemedInputDialog
if TYPE_CHECKING:
    from c50_visualization.c51_visualizer_app import VisualizerApp
import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog


import logging

log = logging.getLogger(__name__)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from c50_visualization.c51_visualizer_app import VisualizerApp


class ConfigShortcutView(ttk.LabelFrame):
    def __init__(self, root: VisualizerApp):
        super().__init__(root, text="Config")


        self.root: VisualizerApp = root
        # ----------------------------------------------------------------------
        # Key Bindings --------------------------------------------------------
        # ----------------------------------------------------------------------
        self.root.bind("T", lambda e: self.root.open_settings_window())

        self.root.bind("Q", lambda e: self.root.quit())
        self.root.bind("R", lambda e: self.root.reload_stack())
        self.root.bind("D", lambda e: self.toggle_dark_mode_shortcut())
        self.root.bind("S", lambda e: self.root.toggle_shortcut_mode())

        self.root.bind("x", lambda e: self.root.exec_visualize_view.toggle_exec())
        self.root.bind("c", lambda e: self.root.exec_visualize_view.step_exec())
        self.root.bind("t", lambda e: self.root.exec_visualize_view.reset_exec())

        self.root.bind("n", lambda e: self.root.perceive_plan_control_view.step_plan())
        self.root.bind("b", lambda e: self.root.perceive_plan_control_view.step_waypoint_back())
        self.root.bind("r", lambda e: self.root.perceive_plan_control_view.replan())

        self.root.bind("h", lambda e: self.root.perceive_plan_control_view.step_control())
        self.root.bind("g", lambda e: self.root.perceive_plan_control_view.align_control())

        self.root.bind("<KeyPress-a>", lambda e: self.root.perceive_plan_control_view.step_steer_left())
        self.root.bind("<KeyPress-d>", lambda e: self.root.perceive_plan_control_view.step_steer_right())
        self.root.bind("<KeyRelease-a>", lambda e: self.root.perceive_plan_control_view.reset_steer())
        self.root.bind("<KeyRelease-d>", lambda e: self.root.perceive_plan_control_view.reset_steer())
        self.root.bind("w", lambda e: self.root.perceive_plan_control_view.step_acc())
        self.root.bind("s", lambda e: self.root.perceive_plan_control_view.step_dec())

        self.root.bind("<Control-plus>", lambda e: self.root.local_plan_plot_view.zoom_in_frenet())
        self.root.bind("<Control-minus>", lambda e: self.root.local_plan_plot_view.zoom_out_frenet())
        self.root.bind("<plus>", lambda e: self.root.local_plan_plot_view.zoom_in())
        self.root.bind("<minus>", lambda e: self.root.local_plan_plot_view.zoom_out())

        ttk.Button(self, text="Reload Code", command=self.root.reload_stack).pack(side=tk.RIGHT)
        ttk.Button(self, text="Load Config", command=self.load_config).pack(side=tk.RIGHT)
        ttk.Button(self, text="Save Config", command=self.save_config).pack(side=tk.RIGHT)
        ttk.Checkbutton(
            self,
            text="Shortcut Mode",
            variable=self.root.setting.shortcut_mode,
            command=self.root.toggle_shortcut_mode,
        ).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton(
            self,
            text="Dark Mode",
            variable=self.root.setting.dark_mode,
            command=self.toggle_dark_mode,
        ).pack(anchor=tk.W, side=tk.LEFT)

        # ----------------------------------------------------------------------
        # Shortcut frame
        # ------------------------------------------------------
        # TODO: this is a dirty trick because its parent is root
        self.shortcut_frame = ttk.LabelFrame(root, text="Shortcuts")
        self.help_text = tk.Text(self.shortcut_frame, wrap=tk.WORD, width=50, height=7)
        key_binding_info = """
Perceive: 
Plan:     n - Step plan        b - Step Back                r - Replan            
Control:  h - Control Step     g - Re-align control         w - Accelerate 
          a - Steer left       d - Steer right              s - Deccelerate
Visalize: Q - Quit             S - Toggle shortcut          D - Toggle Dark Mode        R - Reload imports     
          + - Zoom In          - - Zoom Out           <Ctrl+> - Zoom In F         <Ctrl-> - Zoom Out F
Execute:  c - Step Execution   t - Reset execution          x - Toggle execution
         """.strip()
        self.help_text.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.help_text.insert(tk.END, key_binding_info)
        self.help_text.config(state=tk.DISABLED)  # Make the text area read-only


    def toggle_dark_mode(self):
        self.root.set_dark_mode_themed() if self.root.setting.dark_mode.get() else self.root.set_light_mode()

    def toggle_dark_mode_shortcut(self):
        # if self.root.setting.dark_mode.get():
        #     self.root.setting.dark_mode.set(False)
        # else:
        #     self.root.setting.dark_mode.set(True)

        self.toggle_dark_mode()


    def save_config(self):
        save_config(self.root.setting)
    def load_config(self):
        load_setting(self.root.setting)

class SettingView:
    """
    A view to display and edit settings.
    """
    def __init__(self, root: VisualizerApp):
        self.root = root
        self.setting = root.setting
        settings_window = tk.Toplevel(root)
        # settings_window.title("Settings")
        settings_window.geometry("400x300")
        # self.root.bind("Q", lambda e: settings_window.destroy())

        self.frame = ttk.Frame(settings_window)
        self.frame.pack(fill=tk.BOTH, expand=True)

        
        ##########
        # Profiles
        ##########
        profile_frame = ttk.Frame(self.frame)
        profile_frame.grid(row=0, column=0, sticky="nswe", padx=10, pady=10)

        ttk.Label(profile_frame, text="Load Profile:").grid(row=0, column=0, padx=5, pady=5)
        self.global_planner_dropdown_menu = ttk.Combobox(profile_frame, width=10)
        self.global_planner_dropdown_menu["values"] = ("default", "profile1")
        self.global_planner_dropdown_menu.state(["readonly"])
        self.global_planner_dropdown_menu.bind("<<ComboboxSelected>>", self.load_profile)

        self.global_planner_dropdown_menu.grid(row=0, column=1, columnspan=2, padx=5, pady=5)

        ttk.Button(profile_frame, text="New", width=5, command=lambda: self.create_profile()).grid(row=1, column=0, padx=5, pady=5)
        
        ttk.Button(profile_frame, text="Delete",width=5, command=lambda: self.delete_profile(
            self.global_planner_dropdown_menu.get())).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Button(profile_frame, text="Save",width=5, command=lambda: self.delete_profile(
            self.global_planner_dropdown_menu.get())).grid(row=1, column=2, padx=5, pady=5)

        ##########
        # settings
        ##########
        settings_container = ttk.Frame(self.frame)
        settings_container.grid(row=0, column=2, rowspan=3, sticky="nsew", padx=10, pady=10)
        settings_container.columnconfigure(0, weight=1)
        settings_container.rowconfigure(0, weight=1)

        self.settings_frame= ttk.Frame(self.frame)
        self.settings_frame.grid(row=0, column=2, rowspan=3, sticky="nswe", padx=10, pady=10)
        from c10_perception.c19_settings import PerceptionSettings
        self.create_widgets(PerceptionSettings, "Perception Settings")
        from c20_planning.c29_settings import PlanningSettings
        self.create_widgets(PlanningSettings, "Planning Settings")
        from c30_control.c39_settings import ControlSettings
        self.create_widgets(ControlSettings, "Control Settings")
        from c40_execution.c49_settings import ExecutionSettings
        self.create_widgets(ExecutionSettings, "Execution Settings")
        



    def create_profile(self):
        """
        Load a profile from the settings.
        """
        
        # text = simpledialog.askstring("Profile", "Enter profile Name")
        dialog = ThemedInputDialog(self.root, "Profile", "Name")
        text =  dialog.result
        log.info(f"Creating profile: {text}")


    def delete_profile(self, profile_name: str):
        """
        Delete a profile from the settings.
        """
        log.info(f"Deleting profile: {profile_name}")

    def save_profile(self, profile_name: str):
        pass

    def load_profile(self, event):
        """
        Load a profile from the settings.
        """
        log.info(f"loading profile: {event.widget.get()}")

    
    def create_widgets(self, setting, setting_name="Settings"):
        """
        Create widgets for the settings view.
        """

        frame = ttk.Labelframe(self.settings_frame, text=setting_name)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.entries = {}
        row = 0
        for field in dir(setting):
            if field.startswith("__") or callable(getattr(setting, field)) or field == "filepath":
                continue
            value = getattr(setting, field)
            if isinstance(value, (str, int, float)):
                ttk.Label(frame, text=field).grid(row=row, column=0, sticky="w", padx=5, pady=2)
                entry = ttk.Entry(frame)
                entry.insert(0, str(value))
                entry.grid(row=row, column=1, padx=5, pady=2)
                self.entries[field] = entry
                row += 1
        def save():
            for field, entry in self.entries.items():
                val = entry.get()
                orig = getattr(setting, field)
                if isinstance(orig, int):
                    setattr(setting, field, int(val))
                elif isinstance(orig, float):
                    setattr(setting, field, float(val))
                else:
                    setattr(setting, field, val)




