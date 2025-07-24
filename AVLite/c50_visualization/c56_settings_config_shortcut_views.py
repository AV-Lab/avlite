from __future__ import annotations
from os import wait
from typing import TYPE_CHECKING

from numpy import delete
from c60_tools.c61_utils import save_setting, load_setting, delete_profile 
from c50_visualization.c58_ui_lib import ThemedInputDialog
if TYPE_CHECKING:
    from c50_visualization.c51_visualizer_app import VisualizerApp
import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog

from c10_perception.c19_settings import PerceptionSettings
from c20_planning.c29_settings import PlanningSettings
from c30_control.c39_settings import ControlSettings
from c40_execution.c49_settings import ExecutionSettings


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

        ttk.Button(self, text="⚙" , command=self.open_settings_window, width=2).pack(side=tk.RIGHT)
        ttk.Button(self, text="Reload Stack", command=self.root.reload_stack).pack(side=tk.RIGHT)
        ttk.Button(self, text="Reset Config", command=self.load_config).pack(side=tk.RIGHT)
        ttk.Button(self, text="Save Config", command=self.save_config).pack(side=tk.RIGHT)
        self.global_planner_dropdown_menu = ttk.Combobox(self, width=10, textvariable=self.root.setting.selected_profile, state="readonly",
            justify=tk.CENTER, font=("Arial", 10, "bold"))
        self.global_planner_dropdown_menu["values"] = self.root.setting.profile_list
        # self.global_planner_dropdown_menu.current(0)  
        self.global_planner_dropdown_menu.state(["readonly"])
        self.global_planner_dropdown_menu.bind("<<ComboboxSelected>>", self.__on_dropdown_change)
        self.global_planner_dropdown_menu.pack(side=tk.RIGHT)    
        

        ttk.Checkbutton( self, text="Shortcut Mode", variable=self.root.setting.shortcut_mode,
            command=self.root.toggle_shortcut_mode,).pack(anchor=tk.W, side=tk.LEFT)

        ttk.Checkbutton( self, text="Dark Mode", variable=self.root.setting.dark_mode, command=self.toggle_dark_mode,
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

    def __on_dropdown_change(self, event):
        log.info(f"Selected profile: {event.widget.get()}")
        self.load_config()
        self.root.reinitialize_stack()


    def toggle_dark_mode(self):
        self.root.set_dark_mode_themed() if self.root.setting.dark_mode.get() else self.root.set_light_mode()

    def toggle_dark_mode_shortcut(self):
        # if self.root.setting.dark_mode.get():
        #     self.root.setting.dark_mode.set(False)
        # else:
        #     self.root.setting.dark_mode.set(True)

        self.toggle_dark_mode()


    def save_config(self):
        save_setting(self.root.setting, profile=self.root.setting.selected_profile.get())

    def load_config(self):
        profile = self.root.setting.selected_profile.get()
        load_setting(self.root.setting, profile=profile)
        load_setting(PerceptionSettings, profile=profile)
        load_setting(PlanningSettings, profile=profile)
        load_setting(ControlSettings, profile=profile)
        load_setting(ExecutionSettings, profile=profile)
    
    def open_settings_window(self):
        self.setting_window = SettingView(self.root)

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

        self.frame.columnconfigure(2, weight=3)  # Give column 2 more weight
        self.frame.rowconfigure(0, weight=1)
        
        ##########
        # Profiles
        ##########
        profile_frame = ttk.Frame(self.frame)
        profile_frame.grid(row=0, column=0, sticky="nswe", padx=10, pady=10)

        ttk.Label(profile_frame, text="Load Profile:").grid(row=0, column=0, padx=5, pady=5)
        self.global_planner_dropdown_menu = ttk.Combobox(profile_frame, width=10, textvariable=self.root.setting.selected_profile, state="readonly",)
        self.global_planner_dropdown_menu["values"] = self.root.setting.profile_list
        # self.global_planner_dropdown_menu.current(0)  
        self.global_planner_dropdown_menu.state(["readonly"])
        self.global_planner_dropdown_menu.bind("<<ComboboxSelected>>", self.__on_dropdown_change)

        self.global_planner_dropdown_menu.grid(row=0, column=1, columnspan=2, padx=5, pady=5)

        ttk.Button(profile_frame, text="New", width=5, command=self.create_profile).grid(row=1, column=0, padx=5, pady=5)
        
        ttk.Button(profile_frame, text="Delete",width=5, command=self.delete_profile).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Button(profile_frame, text="Save",width=5, command=self.save_profile).grid(row=1, column=2, padx=5, pady=5)

        ##########
        # settings
        ##########
        settings_container = ttk.Frame(self.frame)
        settings_container.grid(row=0, column=2, rowspan=3, sticky="nsew", padx=10, pady=10)
        settings_container.columnconfigure(0, weight=1)
        settings_container.rowconfigure(0, weight=1)
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        settings_container.bind_all("<MouseWheel>", _on_mousewheel)  # Windows/macOS
        settings_container.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))  # Linux
        settings_container.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))   # Linux
        def unbind_mousewheel_events(event=None):
            settings_container.unbind_all("<MouseWheel>")
            settings_container.unbind_all("<Button-4>")
            settings_container.unbind_all("<Button-5>")
        settings_container.bind("<Destroy>", unbind_mousewheel_events)

        style = ttk.Style()
        bg_color = style.lookup("TFrame", "background")
        canvas = tk.Canvas(settings_container, highlightthickness=0, bd=0, background=bg_color)
        scrollbar = ttk.Scrollbar(settings_container, orient="vertical", command=canvas.yview)
        self.settings_frame = ttk.Frame(canvas)

        # Configure scrolling
        canvas.configure(yscrollcommand=scrollbar.set)
        self.settings_frame.bind( "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # Create window with fixed width that matches canvas
        canvas.create_window((0, 0), window=self.settings_frame, anchor="nw")

        # Grid layout with proper weights
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Set minimum width/height for the container
        settings_window.update_idletasks()  # Force layout update
        settings_window.minsize(500, 400)  # Set minimum window size

        # to keep track of all widgets
        self.widget_entries = {}

        self.create_widgets(PerceptionSettings, "Perception Settings")
        self.create_widgets(PlanningSettings, "Planning Settings")
        self.create_widgets(ControlSettings, "Control Settings")
        self.create_widgets(ExecutionSettings, "Execution Settings")




    def create_profile(self):
        """ Load a profile from the settings. """
        
        # text = simpledialog.askstring("Profile", "Enter profile Name")
        dialog = ThemedInputDialog(self.root, "Profile", "Name")
        text =  dialog.result.strip() if dialog.result else None
        if not text:
            return
        log.info(f"Creating profile: {text}")
        self.root.setting.selected_profile.set(text)
        self.root.setting.profile_list.append(text)
        self.global_planner_dropdown_menu["values"] = self.root.setting.profile_list
        self.root.config_shortcut_view.global_planner_dropdown_menu["values"] = self.root.setting.profile_list


    def delete_profile(self):
        """ Delete a profile from the settings. """

        from tkinter import messagebox
        result = messagebox.askyesno("Confirmation", f"Are you sure you want to delete {self.root.setting.selected_profile.get()}?")
        if result:
            log.info(f"Deleting profile: {self.root.setting.selected_profile.get()}")
            delete_profile(PerceptionSettings, profile=self.root.setting.selected_profile.get())
            delete_profile(PlanningSettings, profile=self.root.setting.selected_profile.get())
            delete_profile(ControlSettings, profile=self.root.setting.selected_profile.get())
            delete_profile(ExecutionSettings, profile=self.root.setting.selected_profile.get())
            delete_profile(self.root.setting, profile=self.root.setting.selected_profile.get())
            self.root.setting.profile_list.remove(self.root.setting.selected_profile.get())
            self.global_planner_dropdown_menu["values"] = self.root.setting.profile_list
            self.root.config_shortcut_view.global_planner_dropdown_menu["values"] = self.root.setting.profile_list
            self.root.setting.selected_profile.set("default")  
            self.load_profile("default")



    def save_profile(self):
        """ Save the current settings to the selected profile. """

        log.info(f"Saving profile: {self.root.setting.selected_profile.get()}")
        self.save_widgets(PerceptionSettings)
        save_setting(PerceptionSettings, profile=self.root.setting.selected_profile.get())
        self.save_widgets(PlanningSettings) 
        save_setting(PlanningSettings, profile=self.root.setting.selected_profile.get())
        self.save_widgets(ControlSettings)
        save_setting(ControlSettings, profile=self.root.setting.selected_profile.get())
        self.save_widgets(ExecutionSettings)
        save_setting(ExecutionSettings, profile=self.root.setting.selected_profile.get())
        
        # just to save the profile 
        save_setting(self.root.setting, profile=self.root.setting.selected_profile.get())

    def __on_dropdown_change(self, event):
        log.info(f"Selected profile: {event.widget.get()}")
        self.load_profile(event.widget.get())

    def load_profile(self, profile="default"):
        """ Load a profile from the settings. """

        log.info(f"loading profile: {profile}")
        load_setting(PerceptionSettings, profile=profile)
        load_setting(PlanningSettings, profile=profile)
        load_setting(ControlSettings, profile=profile)
        load_setting(ExecutionSettings, profile=profile)
        load_setting(self.root.setting, profile=profile)



        # delete previous widgets in settings_frame
        # for widget in self.settings_frame.winfo_children():
        #     widget.destroy()
        #
        # self.create_widgets(PerceptionSettings, "Perception Settings")
        # self.create_widgets(PlanningSettings, "Planning Settings")
        # self.create_widgets(ControlSettings, "Control Settings")
        # self.create_widgets(ExecutionSettings, "Execution Settings")

        self.update_widgets(PerceptionSettings)
        self.update_widgets(PlanningSettings)
        self.update_widgets(ControlSettings)
        self.update_widgets(ExecutionSettings)


    
    def create_widgets(self, setting, setting_name="Settings"):
        """ Create widgets for the settings view. """

        frame = ttk.Labelframe(self.settings_frame, text=setting_name)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.widget_entries[setting.__name__] = {}
        row = 0
        for field in dir(setting):
            if field.startswith("__") or callable(getattr(setting, field)) or field == "filepath":
                continue
            value = getattr(setting, field)
            if isinstance(value, (str, int, float)):
                ttk.Label(frame, text=field).grid(row=row, column=0, sticky="w", padx=5, pady=2)
                entry = ttk.Entry(frame)
                entry.insert(0, str(value))
                entry.grid(row=row, column=1, padx=5, pady=2, sticky="ew")
                self.widget_entries[setting.__name__][field] = entry
                row += 1

    def save_widgets(self, setting):
        """ Save the settings from the widgets to the setting class. """

        if setting.__name__ not in self.widget_entries:
            log.warning(f"No widgets found for setting: {setting.__name__}")
            return

        for field, entry in self.widget_entries[setting.__name__].items():
            if field.startswith("__") or callable(getattr(setting, field)) or field == "filepath":
                continue
            
            if not hasattr(setting, field):  # Changed from app.data to data
                log.warning(f"Skipping unknown attribute: {field}")
                continue

            val = entry.get()
            orig = getattr(setting, field)
            log.debug(f"Saving {field} with value {val} of type {type(val)} to setting {setting.__name__}")
            if isinstance(orig, bool):
                if val.lower() in ["true", "1", "yes"]:
                    setattr(setting, field, True)
                elif val.lower() in ["false", "0", "no"]:
                    setattr(setting, field, False)
                else:
                    log.warning(f"Invalid boolean value for {field}: {val}. Keeping original value: {orig}")
            elif isinstance(orig, int):
                setattr(setting, field, int(val))
            elif isinstance(orig, float):
                setattr(setting, field, float(val))
            else:
                setattr(setting, field, val)

    def update_widgets(self, setting):
        """ Update the widgets with the current settings. """
        if setting.__name__ not in self.widget_entries:
            log.warning(f"No widgets found for setting: {setting.__name__}")
            return

        for field, entry in self.widget_entries[setting.__name__].items():
            if field.startswith("__") or callable(getattr(setting, field)) or field == "filepath":
                continue
            
            if not hasattr(setting, field):
                log.error(f"Skipping unknown attribute: {field}")
                continue
            value = getattr(setting, field)
            log.debug(f"Updating {field} with value {value} of type {type(value)} in setting {setting.__name__}")
            if isinstance(value, bool):
                entry.delete(0, tk.END)
                entry.insert(0, "True" if value else "False")
            elif isinstance(value, (int, float)):
                entry.delete(0, tk.END)
                entry.insert(0, str(value))
            else:
                entry.delete(0, tk.END)
                entry.insert(0, value)



