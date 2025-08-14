from __future__ import annotations
from os import wait
from typing import TYPE_CHECKING
import importlib
import tkinter as tk
from tkinter import ttk

from networkx import reverse


from c60_common.c61_setting_utils import save_setting, load_setting, delete_setting_profile, reload_lib, load_all_stack_settings
from c50_visualization.c58_ui_lib import ThemedInputDialog
if TYPE_CHECKING:
    from c50_visualization.c51_visualizer_app import VisualizerApp

from c10_perception.c19_settings import PerceptionSettings
from c20_planning.c29_settings import PlanningSettings
from c30_control.c39_settings import ControlSettings
from c40_execution.c49_settings import ExecutionSettings
from c60_common.c61_setting_utils import list_extensions


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
        self.root.bind("T", lambda e: self.open_settings_window())

        self.root.bind_all("Q", lambda e: self.root.quit())
        self.root.bind_all("R", lambda e: self.root.reload_stack())
        self.root.bind_all("F", lambda e: self.root.switch_profile() )
        self.root.bind("S", lambda e: self.root.update_shortcut_mode(reverse=True))

        self.root.bind("x", lambda e: self.root.exec_visualize_view.toggle_exec())
        self.root.bind("c", lambda e: self.root.exec_visualize_view.step_exec())
        self.root.bind("t", lambda e: self.root.exec_visualize_view.reset_exec())

        self.root.bind("n", lambda e: self.root.perceive_plan_control_view.plan_frame.step_plan())
        self.root.bind("b", lambda e: self.root.perceive_plan_control_view.plan_frame.step_waypoint_back())
        self.root.bind("r", lambda e: self.root.perceive_plan_control_view.plan_frame.replan())

        self.root.bind("h", lambda e: self.root.perceive_plan_control_view.control_frame.step_control())
        self.root.bind("i", lambda e: self.root.perceive_plan_control_view.control_frame.align_control())

        self.root.bind("<KeyPress-a>", lambda e: self.root.perceive_plan_control_view.control_frame.step_steer_left())
        self.root.bind("<KeyPress-d>", lambda e: self.root.perceive_plan_control_view.control_frame.step_steer_right())
        self.root.bind("<KeyRelease-a>", lambda e: self.root.perceive_plan_control_view.control_frame.reset_steer())
        self.root.bind("<KeyRelease-d>", lambda e: self.root.perceive_plan_control_view.control_frame.reset_steer())
        self.root.bind("w", lambda e: self.root.perceive_plan_control_view.control_frame.step_acc())
        self.root.bind("s", lambda e: self.root.perceive_plan_control_view.control_frame.step_dec())

        self.root.bind("<Control-plus>", lambda e: self.root.local_plan_plot_view.zoom_in_frenet())
        self.root.bind("<Control-minus>", lambda e: self.root.local_plan_plot_view.zoom_out_frenet())
        self.root.bind("<plus>", lambda e: self.root.local_plan_plot_view.zoom_in())
        self.root.bind("<minus>", lambda e: self.root.local_plan_plot_view.zoom_out())

        # scroll log text using vim motion
        self.root.bind("k", lambda e: self.root.log_view.log_area.yview_scroll(-1, "units")) 
        self.root.bind("j", lambda e: self.root.log_view.log_area.yview_scroll(1, "units"))
        self.root.bind("<Control-u>",  lambda e: self.root.log_view.log_area.yview_scroll(-5, "units"))
        self.root.bind("<Control-d>",  lambda e: self.root.log_view.log_area.yview_scroll(int(0.5*self.root.setting.log_view_default_height.get()), "units"))
        self.root.bind("G", lambda e: self.root.log_view.log_area.yview_moveto(1.0))
        self.root.bind("g", lambda e: self.root.log_view.log_area.yview_moveto(0.0))

        self.root.bind("E", lambda e: self.root.log_view.update_log_view_height(reverse=True))  # Toggle log view height
    
        # ----------------------------------------------------------------------

        ttk.Button(self, text="⚙" , command=self.open_settings_window, width=2).pack(side=tk.RIGHT)
        ttk.Button(self, text="Reload Stack", command=self.root.reload_stack).pack(side=tk.RIGHT)
        ttk.Button(self, text="Reset Config", command=self.root.load_configs).pack(side=tk.RIGHT)
        ttk.Button(self, text="Save Config", command=self.save_config).pack(side=tk.RIGHT)


        self.profile_dropdown_menu = ttk.Combobox(self, width=10, textvariable=self.root.setting.selected_profile, state="readonly",
            justify=tk.CENTER, font=("Arial", 10, "bold"))
        self.profile_dropdown_menu["values"] = self.root.setting.profile_list
        self.profile_dropdown_menu.state(["readonly"])
        self.profile_dropdown_menu.bind("<<ComboboxSelected>>", self.__on_profile_dropdown_change)
        self.profile_dropdown_menu.pack(side=tk.RIGHT)    
        

        ttk.Checkbutton(self, text="Shortcut Mode", variable=self.root.setting.shortcut_mode,
            command=self.root.update_shortcut_mode,).pack(anchor=tk.W, side=tk.LEFT)

        ttk.Checkbutton(self, text="Dark Mode", variable=self.root.setting.dark_mode, command=self.toggle_dark_mode,
        ).pack(anchor=tk.W, side=tk.LEFT)
        
        ttk.Label(self, textvariable=self.root.setting.perception_status_text, width=30).pack(side=tk.LEFT, padx=(25,5), pady=5)

        # ----------------------------------------------------------------------
        # Shortcut frame
        # ------------------------------------------------------
        # TODO: this is a dirty trick because its parent is root
        self.shortcut_frame = ttk.LabelFrame(root, text="Shortcuts")
        self.help_text = tk.Text(self.shortcut_frame, wrap=tk.WORD, width=50, height=7)
        key_binding_info = """
App:      Q - Quit             S - Toggle shortcut          F - Switch to next Profile  R - Reload imports     
          T - Open Settings    E - Expand/collapse log    vim - use vim motion to scroll log   
Plan:     n - Step plan        b - Step Back                r - Replan            
          + - Zoom In          - - Zoom Out           <Ctrl+> - Zoom In F         <Ctrl-> - Zoom Out F
Control:  h - Control Step     i - Re-align control         w - Accelerate 
          a - Steer left       d - Steer right              s - Deccelerate
Execute:  c - Step Execution   t - Reset execution          x - Toggle execution
         """.strip()
        self.help_text.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.help_text.insert(tk.END, key_binding_info)
        self.help_text.config(state=tk.DISABLED)  # Make the text area read-only



    def __on_profile_dropdown_change(self, event):
        log.info(f"Selected profile: {event.widget.get()}")
        self.root.load_configs()
        self.root.reload_stack(reload_code=False)


    def toggle_dark_mode(self):
        self.root.set_dark_mode_themed() if self.root.setting.dark_mode.get() else self.root.set_light_mode()

    def save_config(self):
        save_setting(self.root.setting, profile=self.root.setting.selected_profile.get())
        save_setting(ExecutionSettings, profile=self.root.setting.selected_profile.get())


    def open_settings_window(self):
        if hasattr(self, "setting_view") and hasattr(self.setting_view, "window") and self.setting_view.window.winfo_exists():
            # Show existing window
            self.root.load_configs(only_stack=True)
            self.setting_view.show()
            log.info("Showing existing settings window")
        else:
            self.root.load_configs(only_stack=True)
            self.setting_view = SettingWindow(self.root)
            log.info("Creating new settings window")

    def update_setting_window(self):
        """Update the settings window with the latest settings."""
        if hasattr(self, "setting_view") and hasattr(self.setting_view, "window") and self.setting_view.window.winfo_exists():
            self.setting_view.update_core_widgets()
            self.setting_view.update_extensions_widgets()
            log.info("Updated existing settings window")


class SettingWindow:
    """
    A view to display and edit settings.
    """
    def __init__(self, root: VisualizerApp):
        self.root = root
        self.setting = root.setting
        self.window = tk.Toplevel(root)
        # settings_window.title("Settings")
        self.window.geometry("400x300")
        # self.root.bind("Q", lambda e: settings_window.destroy())

        self.frame = ttk.Frame(self.window)
        self.frame.pack(fill=tk.BOTH, expand=True)

        
        self.window.bind("<Control-s>", lambda e: self.save_profile())
        self.window.bind("k", lambda e: self.canvas.yview_scroll(-1, "units")) 
        self.window.bind("j", lambda e: self.canvas.yview_scroll(1, "units"))
        self.window.bind("<Control-u>",  lambda e: self.canvas.yview_scroll(-5, "units"))
        self.window.bind("<Control-d>",  lambda e: self.canvas.yview_scroll(int(0.5*self.root.setting.log_view_default_height.get()), "units"))
        self.window.bind("G", lambda e: self.canvas.yview_moveto(1.0))
        self.window.bind("g", lambda e: self.canvas.yview_moveto(0.0))
        
        #########
        # Main Layout
        #########
        self.frame.rowconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=1)  # Settings frame
        
        profile_ext_frame = ttk.Frame(self.frame)
        profile_ext_frame.grid(row=0, column=0, sticky="nswe", padx=10, pady=10)

        settings_frame = ttk.Frame(self.frame)
        settings_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=10, pady=10)

        additional_setting_frame = ttk.LabelFrame(self.frame, text="Additional Settings")
        additional_setting_frame.grid(row=1, column=0, columnspan=2, sticky="sew", padx=5, pady=5)

        ##########
        # Profiles & Extensions
        ##########
        profile_ext_frame.rowconfigure(5,weight=1)


        ttk.Label(profile_ext_frame, text="Execution Profiles",style="Big.TLabel").grid(row=0, column=0, sticky="w", columnspan=3, padx=10, pady=5)
        ttk.Label(profile_ext_frame, text="Load Profile").grid(row=1, column=0, padx=5, pady=5)
        self.profile_dropdown_menu = ttk.Combobox(profile_ext_frame, textvariable=self.root.setting.selected_profile, state="readonly",)
        self.profile_dropdown_menu["values"] = self.root.setting.profile_list
        # self.global_planner_dropdown_menu.current(0)  
        self.profile_dropdown_menu.state(["readonly"])
        self.profile_dropdown_menu.bind("<<ComboboxSelected>>", self.__on_profile_dropdown_change)
        self.profile_dropdown_menu.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky="we")

        ttk.Button(profile_ext_frame, text="New", width=5, command=self.create_profile).grid(row=2, column=0, padx=5, pady=5, sticky="we")
        ttk.Button(profile_ext_frame, text="Delete",width=5, command=self.delete_profile).grid(row=2, column=1, padx=5, pady=5, sticky="we")
        ttk.Button(profile_ext_frame, text="Save",width=5, underline=0, command=self.save_profile).grid(row=2, column=2, padx=5, pady=5, sticky="we")
        ttk.Button(
            profile_ext_frame, text="Reset to source code defaults", command=self.reset_to_to_source_stack_values
        ).grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="we")
        
        ttk.Label(profile_ext_frame, text="Cycle Next (Shortcut F)").grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        next_profile_dropdown_menu = ttk.Combobox(profile_ext_frame, width=10, textvariable=self.root.setting.next_profile, state="readonly",)
        next_profile_dropdown_menu["values"] = self.root.setting.profile_list
        next_profile_dropdown_menu.state(["readonly"])
        # next_profile_dropdown_menu.bind("<<ComboboxSelected>>", self.__on_dropdown_change)
        next_profile_dropdown_menu.grid(row=4, column=2, padx=5, pady=5, sticky="we")


        ##############################################
        ## Extensions
        ##############################################
        extension_frame = ttk.LabelFrame(profile_ext_frame, text="Extensions")
        extension_frame.grid(row=5, column=0, columnspan=3, sticky="sew", padx=5, pady=5)
        ttk.Checkbutton(extension_frame, text="Load Extensions" , variable=self.root.setting.load_extensions,
            command=self.recreate_extensions_widgets).grid(row=0, column=0, sticky="w", padx=5, pady=5)

        listbox = tk.Listbox(extension_frame, height=5, selectmode=tk.SINGLE, exportselection=False, width=30,)
        listbox.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        # Convert comma-separated string to list items

        for ext in self.root.setting.extension_list:
            listbox.insert(tk.END, ext)



        #############################################
        # settings
        #############################################
        settings_frame.columnconfigure(0, weight=1)
        settings_frame.rowconfigure(0, weight=1)
        
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        settings_frame.bind_all("<MouseWheel>", _on_mousewheel)  # Windows/macOS
        settings_frame.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))  # Linux
        settings_frame.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))   # Linux

        def unbind_mousewheel_events(event=None):
            settings_frame.unbind_all("<MouseWheel>")
            settings_frame.unbind_all("<Button-4>")
            settings_frame.unbind_all("<Button-5>")
        settings_frame.bind("<Destroy>", unbind_mousewheel_events)

        style = ttk.Style()
        bg_color = style.lookup("TFrame", "background")
        self.canvas = tk.Canvas(settings_frame, highlightthickness=0, bd=0, background=bg_color)
        self.scrollbar = ttk.Scrollbar(settings_frame, orient="vertical", command=self.canvas.yview)
        self.settings_frame = ttk.Frame(self.canvas)

        # Configure scrolling
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.settings_frame.bind( "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        # Create window with fixed width that matches self.canvas
        self.canvas.create_window((0, 0), window=self.settings_frame, anchor="nw")

        # Grid layout with proper weights
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        
        ########
        ########
        # Set minimum width/height for the container
        self.window.update_idletasks()  # Force layout update
        self.window.minsize(500, 400)  # Set minimum window size
        # to prevent killing the window afte close
        self.window.protocol("WM_DELETE_WINDOW", self.hide) 

        ######################
        ## Stack widgetes
        ######################
        ttk.Label(self.settings_frame, text="Core Stack Settings",style="Big.TLabel").pack(anchor=tk.W, padx=5, pady=5)
        # to keep track of all widgets
        self.widget_entries = {}
        self.create_widgets(PerceptionSettings, "Perception Settings")
        self.create_widgets(PlanningSettings, "Planning Settings")
        self.create_widgets(ControlSettings, "Control Settings")
        self.create_widgets(ExecutionSettings, "Execution Settings")
        #######
        #######
        if self.root.setting.load_extensions.get():
            ttk.Separator(self.settings_frame, orient='horizontal').pack(fill='x', pady=10)
            ttk.Label(self.settings_frame, text="Extensions Settings",style="Big.TLabel").pack(anchor=tk.W, padx=5, pady=5)
            self.create_ext_widgets()


        ################
        # Additional settings
        ################

        ## UI Elements for Visualize - Checkboxes
        additional_setting_row_1 = ttk.Frame(additional_setting_frame)
        ttk.Label(additional_setting_row_1, text="Local Plan Plot View:").pack(anchor=tk.W, side=tk.LEFT, padx=5)
        additional_setting_row_1.pack(fill=tk.X)
        ttk.Checkbutton( additional_setting_row_1, text="Legend", variable=self.root.setting.show_legend, command=self.root.update_ui,).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton( additional_setting_row_1, text="Locations", variable=self.root.setting.show_past_locations, command=self.root.update_ui,).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton( additional_setting_row_1, text="Global Plan", variable=self.root.setting.show_global_plan, command=self.root.update_ui,).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton( additional_setting_row_1, text="Local Plan", variable=self.root.setting.show_local_plan, command=self.root.update_ui,).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton( additional_setting_row_1, text="Local Lattice", variable=self.root.setting.show_local_lattice, command=self.root.update_ui,).pack(anchor=tk.W, side=tk.LEFT)
        ttk.Checkbutton( additional_setting_row_1, text="State", variable=self.root.setting.show_state, command=self.root.update_ui,).pack(anchor=tk.W, side=tk.LEFT)

        ttk.Checkbutton( additional_setting_row_1, text="Follow Planner in Global", variable=self.root.setting.global_view_follow_planner).pack(side=tk.LEFT)
        ttk.Checkbutton( additional_setting_row_1, text="Follow Planner in Frenet", variable=self.root.setting.frenet_view_follow_planner).pack(side=tk.LEFT)

        additional_setting_row_2 = ttk.Frame(additional_setting_frame)
        additional_setting_row_2.pack(fill=tk.X, padx=5)

        ttk.Label(additional_setting_row_2, text="Log View:").pack(anchor=tk.W, side=tk.LEFT, padx=5)
        ttk.Checkbutton(additional_setting_row_2, text="Expand Log View", variable=self.root.setting.log_view_expanded).pack(side=tk.LEFT)
        
        ttk.Label(additional_setting_row_2, text="Default Height:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(additional_setting_row_2, textvariable=self.root.setting.log_view_default_height, width=5,
                  validatecommand=self.root.validate_cmd).pack(side=tk.LEFT, padx=5)

        ttk.Label(additional_setting_row_2, text="Expanded Height").pack(side=tk.LEFT, padx=5)
        ttk.Entry(additional_setting_row_2, textvariable=self.root.setting.log_view_expended_height, width=5,
                  validatecommand=self.root.validate_cmd).pack(side=tk.LEFT, padx=5)
        
        # additional_setting_row_3 = ttk.Frame(additional_setting_frame)
        # additional_setting_row_3.pack(fill=tk.X, padx=5)

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
        self.profile_dropdown_menu["values"] = self.root.setting.profile_list
        self.root.config_shortcut_view.profile_dropdown_menu["values"] = self.root.setting.profile_list
        self.save_profile()


    def delete_profile(self):
        """ Delete a profile from the settings. """

        from tkinter import messagebox
        result = messagebox.askyesno("Confirmation", f"Are you sure you want to delete {self.root.setting.selected_profile.get()}?")
        if result:
            log.info(f"Deleting profile: {self.root.setting.selected_profile.get()}")
            delete_setting_profile(PerceptionSettings, profile=self.root.setting.selected_profile.get())
            delete_setting_profile(PlanningSettings, profile=self.root.setting.selected_profile.get())
            delete_setting_profile(ControlSettings, profile=self.root.setting.selected_profile.get())
            delete_setting_profile(ExecutionSettings, profile=self.root.setting.selected_profile.get())
            delete_setting_profile(self.root.setting, profile=self.root.setting.selected_profile.get())
            if self.root.setting.load_extensions.get():
                for ext in self.root.setting.extension_list:
                    try:
                        module = importlib.import_module(f"extensions.{ext}.settings")
                        ExtensionSettings = getattr(module, "ExtensionSettings")
                        delete_setting_profile(ExtensionSettings, profile=self.root.setting.selected_profile.get())
                    except Exception as e:
                        log.error(f"Failed to delete extension settings for {ext}: {e}")

            self.root.setting.profile_list.remove(self.root.setting.selected_profile.get())
            self.profile_dropdown_menu["values"] = self.root.setting.profile_list
            self.root.config_shortcut_view.profile_dropdown_menu["values"] = self.root.setting.profile_list
            self.root.setting.selected_profile.set("default")  
            self.load_profile("default")



    def save_profile(self):
        """ Save the current settings to the selected profile. """

        log.info(f"Saving profile: {self.root.setting.selected_profile.get()}")
        self.save_from_widgets(PerceptionSettings)
        save_setting(PerceptionSettings, profile=self.root.setting.selected_profile.get())
        self.save_from_widgets(PlanningSettings) 
        save_setting(PlanningSettings, profile=self.root.setting.selected_profile.get())
        self.save_from_widgets(ControlSettings)
        save_setting(ControlSettings, profile=self.root.setting.selected_profile.get())
        self.save_from_widgets(ExecutionSettings)
        save_setting(ExecutionSettings, profile=self.root.setting.selected_profile.get())

        save_setting(self.root.setting, profile=self.root.setting.selected_profile.get())

        if self.root.setting.load_extensions.get():
            for ext in self.root.setting.extension_list:
                try:
                    module = importlib.import_module(f"extensions.{ext}.settings")
                    ExtensionSettings = getattr(module, "ExtensionSettings")
                    self.save_from_widgets(ExtensionSettings, extension_name=ext)
                    save_setting(ExtensionSettings, profile=self.root.setting.selected_profile.get())
                except Exception as e:
                    log.error(f"Failed to save extension settings for {ext}: {e}", stack_info=True)

        
        # just to save the profile 
        save_setting(self.root.setting, profile=self.root.setting.selected_profile.get())
    

    def load_profile(self, profile="default"):
        """ Load a profile from the settings. """

        log.info(f"loading profile: {profile}")
        load_all_stack_settings(profile=profile, load_extensions=self.root.setting.load_extensions.get(),
                                all_extension_dirs=ExecutionSettings.extension_directories)
        load_setting(self.root.setting, profile=profile)

        self.update_core_widgets()
        self.update_extensions_widgets()

    def update_core_widgets(self):
        self.update_widgets(PerceptionSettings)
        self.update_widgets(PlanningSettings)
        self.update_widgets(ControlSettings)
        self.update_widgets(ExecutionSettings)
        
        if self.root.setting.load_extensions.get():
            for ext in self.root.setting.extension_list:
                try:
                    module = importlib.import_module(f"extensions.{ext}.settings")
                    ExtensionSettings = getattr(module, "ExtensionSettings")
                    load_setting(ExtensionSettings, profile=self.root.setting.selected_profile.get())
                    self.update_widgets(ExtensionSettings, extension_name=ext)
                    log.debug(f"loaded extension settings for {ext}")
                except Exception as e:
                    log.error(f"Failed to save extension settings for {ext}: {e}")

    def update_extensions_widgets(self):
        """ Update the extension widgets with the current settings. """ 

        if self.root.setting.load_extensions.get():
            for ext in self.root.setting.extension_list:
                try:
                    module = importlib.import_module(f"extensions.{ext}.settings")
                    ExtensionSettings = getattr(module, "ExtensionSettings")
                    self.update_widgets(ExtensionSettings, extension_name=ext)
                except Exception as e:
                    log.error(f"Failed to save extension settings for {ext}: {e}")

    def __on_profile_dropdown_change(self, event):
        log.info(f"Selected profile: {event.widget.get()}")
        self.load_profile(event.widget.get())

    def reset_to_to_source_stack_values(self):
        """ Reset the stack settings to the source code defaults, except for the UI as it is using some 
            some instant variables for tkinter.
        """
        reload_lib(exclude_stack=True, reload_extensions=True)
        from c10_perception.c19_settings import PerceptionSettings
        from c20_planning.c29_settings import PlanningSettings
        from c30_control.c39_settings import ControlSettings
        from c40_execution.c49_settings import ExecutionSettings

        self.update_widgets(PerceptionSettings)
        self.update_widgets(PlanningSettings)
        self.update_widgets(ControlSettings)
        self.update_widgets(ExecutionSettings)

        self.update_extensions_widgets()
        

    def recreate_extensions_widgets(self):
        """ Recreate the extension widgets based on the current setting."""

        if self.root.setting.load_extensions.get():
            self.create_ext_widgets()
        else:
            # Clear existing extension widgets
            for ext in self.root.setting.extension_list:
                ext_key = f"ExtensionSettings{ext}"
                if ext_key in self.widget_entries:
                    entry_dict = self.widget_entries[ext_key]
                    if entry_dict:
                        first_widget = next(iter(entry_dict.values()))
                        parent_frame = first_widget.master
                        parent_frame.pack_forget()  # or grid_forget() if using grid
                    del self.widget_entries[ext_key]
                    log.debug(f"Removed widgets for {ext_key}")

            self.ext_widget_created = False


    def create_ext_widgets(self):
        if hasattr(self, "ext_widget_created") and self.ext_widget_created:
            log.warning("Extension widgets already created, skipping.")
            return

        for ext in self.root.setting.extension_list:
            try:
                module = importlib.import_module(f"extensions.{ext}.settings")
                ExtensionSettings = getattr(module, "ExtensionSettings")
                self.create_widgets(ExtensionSettings, f"Extension {ext} Settings", extension_name=ext)
                self.ext_widget_created = True
            except Exception as e:
                log.error(f"Failed to load extension settings for {ext}: {e}")

    
    # TODO: need add list to possible values
    def create_widgets(self, setting, setting_name="Settings", extension_name=""):
        """ Create widgets for the settings view. """

        frame = ttk.Labelframe(self.settings_frame, text=setting_name)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.widget_entries[setting.__name__+extension_name] = {}
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
                self.widget_entries[setting.__name__+extension_name][field] = entry
                row += 1



    def save_from_widgets(self, setting, extension_name=""):
        """ Save the settings from the widgets to the setting class. """

        if setting.__name__+extension_name not in self.widget_entries:
            log.warning(f"No widgets found for setting: {setting.__name__}+{extension_name}")
            return

        # log.warning(f"keys in widget_entries: {self.widget_entries.keys()}")
        for field, entry in self.widget_entries[setting.__name__+extension_name].items():
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

    def update_widgets(self, setting, extension_name=""):
        """ Update the widgets with the current settings. """
        if setting.__name__+extension_name not in self.widget_entries:
            log.warning(f"No widgets found for setting: {extension_name} {setting.__name__}")
            return

        for field, entry in self.widget_entries[setting.__name__+extension_name].items():
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

    def hide(self):
        """Hide the window instead of destroying it"""
        self.window.withdraw()
        
    def show(self):
        """Show the hidden window"""
        self.window.deiconify()
        self.window.lift()
        self.window.focus_set()
        # Update widgets with latest settings
        self.update_widgets(PerceptionSettings)
        self.update_widgets(PlanningSettings)
        self.update_widgets(ControlSettings)
        self.update_widgets(ExecutionSettings)
