from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import logging
import queue
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from avlite.c50_visualization.c58_ui_lib import attach_schema_tooltip, attach_tooltip, BUTTON_TOOLTIPS
from avlite.c50_visualization.c59_settings import VisualizationSettings
from avlite.c60_common.c68_settings_schema import field_tooltip_text
from avlite.c60_common.c60_plugins import (
    layer_key_for_plugin_log_record,
    plugin_package_from_logger,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from avlite.c50_visualization.c51_visualizer_app import VisualizerApp


class LogView(ttk.LabelFrame):
    _LAYER_TOGGLES = (
        ("perception", "avlite.c10_perception", "show_perceive_logs"),
        ("planning", "avlite.c20_planning", "show_plan_logs"),
        ("control", "avlite.c30_control", "show_control_logs"),
        ("execution", "avlite.c40_execution", "show_execute_logs"),
        ("visualization", "avlite.c50_visualization", "show_vis_logs"),
        ("common", "avlite.c60_common", "show_common_logs"),
    )

    _DEFAULT_FILTER_STATE: dict[str, bool] = {
        "show_perceive_logs": True,
        "show_plan_logs": True,
        "show_control_logs": True,
        "show_execute_logs": True,
        "show_vis_logs": True,
        "show_common_logs": True,
        "show_core_logs": True,
        "show_plugins_logs": True,
        "disable_log": False,
        "log_to_file": False,
    }

    def __init__(self, root: VisualizerApp):
        super().__init__(root, text="Log")
        self.root = root
        self.max_log_lines = self.root.setting.max_log_lines

        # self.log_queue = queue.Queue()
        # self.after(50, self.process_log_queue)

        self.log_blacklist = set()
        self._filter_state = dict(self._DEFAULT_FILTER_STATE)

        self.controls_frame = ttk.Frame(self)
        self.controls_frame.pack(fill=tk.X, side=tk.TOP)

        cb_core = ttk.Checkbutton(
            self.controls_frame, text="Core", variable=self.root.setting.show_core_logs, command=self.update_log_filter,
        )
        cb_core.pack(side=tk.LEFT)
        attach_schema_tooltip(cb_core, VisualizationSettings, "show_core_logs")
        cb_plugins = ttk.Checkbutton(
            self.controls_frame, text="Plugins", variable=self.root.setting.show_plugins_logs, command=self.update_log_filter,
        )
        cb_plugins.pack(side=tk.LEFT)
        attach_schema_tooltip(cb_plugins, VisualizationSettings, "show_plugins_logs")
        self.cb_perceive = ttk.Checkbutton( self.controls_frame, text="Perception", variable=self.root.setting.show_perceive_logs, command=self.update_log_filter,)
        self.cb_perceive.pack(side=tk.LEFT)
        attach_schema_tooltip(self.cb_perceive, VisualizationSettings, "show_perceive_logs")
        self.cb_plan = ttk.Checkbutton( self.controls_frame, text="Planning", variable=self.root.setting.show_plan_logs, command=self.update_log_filter,)
        self.cb_plan.pack(side=tk.LEFT)
        attach_schema_tooltip(self.cb_plan, VisualizationSettings, "show_plan_logs")
        self.cb_control = ttk.Checkbutton( self.controls_frame, text="Control", variable=self.root.setting.show_control_logs, command=self.update_log_filter,)
        self.cb_control.pack(side=tk.LEFT)
        attach_schema_tooltip(self.cb_control, VisualizationSettings, "show_control_logs")
        self.cb_execute = ttk.Checkbutton( self.controls_frame, text="Execution", variable=self.root.setting.show_execute_logs, command=self.update_log_filter,)
        self.cb_execute.pack(side=tk.LEFT)
        attach_schema_tooltip(self.cb_execute, VisualizationSettings, "show_execute_logs")
        self.cb_vis = ttk.Checkbutton( self.controls_frame, text="Visualization", variable=self.root.setting.show_vis_logs, command=self.update_log_filter,)
        self.cb_vis.pack(side=tk.LEFT)
        attach_schema_tooltip(self.cb_vis, VisualizationSettings, "show_vis_logs")
        self.cb_common = ttk.Checkbutton( self.controls_frame, text="Common", variable=self.root.setting.show_common_logs, command=self.update_log_filter,)
        self.cb_common.pack(side=tk.LEFT)
        attach_schema_tooltip(self.cb_common, VisualizationSettings, "show_common_logs")

        cb_file = ttk.Checkbutton(self.controls_frame, text="File", variable=self.root.setting.log_to_file, command=self.update_log_to_file)
        cb_file.pack(side=tk.RIGHT)
        attach_schema_tooltip(cb_file, VisualizationSettings, "log_to_file")

        self.rb_db_stdout = ttk.Radiobutton( self.controls_frame, text="STDOUT", variable=self.root.setting.log_level, value="STDOUT", command=self.update_log_level,)
        self.rb_db_stdout.pack(side=tk.RIGHT)

        self.rb_db_warn = ttk.Radiobutton( self.controls_frame, text="WARN", variable=self.root.setting.log_level, value="WARN", command=self.update_log_level,)
        self.rb_db_warn.pack(side=tk.RIGHT)

        self.rb_db_info = ttk.Radiobutton( self.controls_frame, text="INFO", variable=self.root.setting.log_level, value="INFO", command=self.update_log_level,)
        self.rb_db_info.pack(side=tk.RIGHT)

        self.rb_db_debug = ttk.Radiobutton( self.controls_frame, text="DEBUG", variable=self.root.setting.log_level, value="DEBUG", command=self.update_log_level,)
        self.rb_db_debug.pack(side=tk.RIGHT)

        _log_level_tip = field_tooltip_text(VisualizationSettings, "log_level") or ""
        for rb, value, hint in (
            (self.rb_db_stdout, "STDOUT", "Mirror log output to the terminal."),
            (self.rb_db_warn, "WARN", "Show warnings and errors only."),
            (self.rb_db_info, "INFO", "Show info, warnings, and errors."),
            (self.rb_db_debug, "DEBUG", "Show all messages including debug."),
        ):
            attach_tooltip(rb, f"{_log_level_tip} Selects {value}. {hint}")
        
        btn_clear = ttk.Button(self.controls_frame, text="Clear", command=self.clear_log, width=4)
        btn_clear.pack(side=tk.RIGHT)
        attach_tooltip(btn_clear, BUTTON_TOOLTIPS["log_clear"])
        self.root.setting.log_view_expanded.trace_add("write", lambda *_: self._sync_expand_button())
        btn_copy = ttk.Button(self.controls_frame, text="Copy", command=self.copy_log, width=4)
        btn_copy.pack(side=tk.RIGHT)
        attach_tooltip(btn_copy, BUTTON_TOOLTIPS["log_copy"])
        self._btn_expand = ttk.Button( self.controls_frame, text="▾", width=3,
            command=lambda: self.update_log_view_height(reverse=True),
        )
        self._btn_expand.pack(side=tk.RIGHT, padx=(0, 2))
        attach_tooltip(self._btn_expand, BUTTON_TOOLTIPS["log_toggle_height"])
        self._sync_expand_button()

        
        self.rb_db_debug.pack(side=tk.RIGHT)
        self.log_area = ScrolledText(self, state="disabled", height=self.root.setting.log_view_default_height.get(), wrap=tk.WORD)
        self.log_area.pack(fill=tk.BOTH, side=tk.BOTTOM, expand=True)

        self._file_handler: logging.FileHandler | None = None

        self.after(100, self.update_log_level)
        self.after(100, self.update_log_filter)

        # -------------------------------------------
        # -------------------------------------------
        # -Configure logging-------------------------
        # -------------------------------------------
        logger = logging.getLogger()
        self.log_handler = LogView.LogTextHandler(self.log_area, self)
        # remove other handlers to avoid duplicate logs
        for handler in logger.handlers:
            logger.removeHandler(handler)
        logger.addHandler(self.log_handler)
        logger.setLevel(logging.INFO)
        # self.poll_log_queue()

        ## Redirect stdout and stderr to the log area
        sys.stderr = LogView.StreamToLogger(logger, logging.ERROR)
        self._sync_filter_state()
        log.info("Log initialized.")
        self.after(self.root.setting.log_pull_time, self.poll_log_queue)

    def reset(self):
        self.update_log_filter()
        self.update_log_level()
    
    def clear_log(self):
        """ Clear the log area """
        self.log_area.config(state="normal")
        self.log_area.delete("1.0", "end")
        self.log_area.config(state="disabled")
        # while not self.log_queue.empty():
            # self.log_queue.get_nowait()
    
    def copy_log(self):
        """ Clear the log area """
        self.log_area.config(state="normal")
        self.log_area.clipboard_clear()
        self.log_area.clipboard_append(self.log_area.get("1.0", "end"))
        self.log_area.config(state="disabled")
        log.info("Log copied to clipboard.")


    def update_log_to_file(self):
        log_dir = Path.cwd() / "logs"
        self._sync_filter_state()

        if self.root.setting.log_to_file.get():
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"avlite_{timestamp}.log"
            self._file_handler = logging.FileHandler(log_path, encoding="utf-8")
            formatter = logging.Formatter(
                "%(asctime)s %(lineno)-4d [%(levelname).4s] %(name)-36s: %(message)s"
            )
            self._file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(self._file_handler)
            self.log_area.pack_forget()
            log.info(f"Logging to file: {log_path}")
        else:
            if self._file_handler is not None:
                logging.getLogger().removeHandler(self._file_handler)
                self._file_handler.close()
                self._file_handler = None
                log.info("File logging stopped.")
            self.log_area.pack(fill=tk.BOTH, side=tk.BOTTOM, expand=True)
        self._sync_filter_state()

    def update_log_view_height(self, reverse: bool = False):
        """ update the log view height based on the setting """

        if reverse:
            self.root.setting.log_view_expanded.set(not self.root.setting.log_view_expanded.get())

        if self.root.setting.log_view_expanded.get():
            self.log_area.configure(height=self.root.setting.log_view_expended_height.get())
            log.debug("Log view expanded.")
        else:
            self.log_area.configure(height=self.root.setting.log_view_default_height.get())
            log.debug("Log view collapsed.")
        self._sync_expand_button()

    def _sync_expand_button(self) -> None:
        if not hasattr(self, "_btn_expand"):
            return
        if self.root.setting.log_view_expanded.get():
            self._btn_expand.configure(text="▾")
        else:
            self._btn_expand.configure(text="▴")


    def _sync_filter_state(self) -> None:
        """Copy Tk checkbox state into plain bools (main thread only)."""
        s = self.root.setting
        self._filter_state = {
            "show_perceive_logs": bool(s.show_perceive_logs.get()),
            "show_plan_logs": bool(s.show_plan_logs.get()),
            "show_control_logs": bool(s.show_control_logs.get()),
            "show_execute_logs": bool(s.show_execute_logs.get()),
            "show_vis_logs": bool(s.show_vis_logs.get()),
            "show_common_logs": bool(s.show_common_logs.get()),
            "show_core_logs": bool(s.show_core_logs.get()),
            "show_plugins_logs": bool(s.show_plugins_logs.get()),
            "disable_log": bool(s.disable_log.get()),
            "log_to_file": bool(s.log_to_file.get()),
        }

    def update_log_filter(self):
        log.info("Log filter updated.")
        self._sync_filter_state()
        self.log_blacklist.clear()
        if not self._filter_state["show_perceive_logs"]:
            self.log_blacklist.add("avlite.c10_perception")
        if not self._filter_state["show_plan_logs"]:
            self.log_blacklist.add("avlite.c20_planning")
        if not self._filter_state["show_control_logs"]:
            self.log_blacklist.add("avlite.c30_control")
        if not self._filter_state["show_execute_logs"]:
            self.log_blacklist.add("avlite.c40_execution")
        if not self._filter_state["show_vis_logs"]:
            self.log_blacklist.add("avlite.c50_visualization")
        if not self._filter_state["show_common_logs"]:
            self.log_blacklist.add("avlite.c60_common")

    def update_log_level(self):
        logger = logging.getLogger()
        if self.root.setting.log_level.get() == "DEBUG":
            logging.getLogger().setLevel(logging.DEBUG)
            log.debug("Log setting updated to DEBUG.")
        elif self.root.setting.log_level.get() == "INFO":
            logging.getLogger().setLevel(logging.INFO)
            log.info("Log setting updated to INFO.")
        elif self.root.setting.log_level.get() == "WARN":
            logging.getLogger().setLevel(logging.WARNING)
            log.warning("Log setting updated to WARNING.")

        if self.root.setting.log_level.get() == "STDOUT":
            logging.getLogger().setLevel(logging.CRITICAL)
            sys.stdout = LogView.TextRedirector(self.log_area)
        else:
            sys.stdout = sys.__stdout__
        
        # Log after setting update is complete
        log.info(f"Log level updated to: {self.root.setting.log_level.get()}")
    

    def process_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_area.config(state="normal")
                self.log_area.insert("end", msg)
                self.log_area.config(state="disabled")
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(50, self.process_log_queue)

    def poll_log_queue(self, max_per_poll: int = 20):
        messages = []
        try:
            while len(messages) < max_per_poll:
                record, levelno = self.log_handler.log_queue.get_nowait()
                # Format lazily on the UI thread (avoids blocking planner/controller threads in emit)
                msg = self.log_handler.format(record)
                _first_dot = msg.find('.')
                _second_dot = msg.find('.', _first_dot + 1)
                code = msg[_second_dot+1 : msg.find('_', _second_dot)]
                msg = code[:4] + ':' + msg
                if levelno >= logging.ERROR:
                    tag = "error"
                elif levelno >= logging.WARNING:
                    tag = "warn"
                elif levelno >= logging.INFO:
                    tag = "info"
                else:
                    tag = "debug"
                messages.append((msg + "\n", tag))
        except queue.Empty:
            pass

        if messages and not self._filter_state.get("log_to_file") and not self._filter_state.get("disable_log"):
            self.log_area.configure(state="normal")
            # Batch consecutive same-tag records into a single insert call
            # to minimise the number of Tkinter operations per poll.
            i = 0
            while i < len(messages):
                tag = messages[i][1]
                batch = ""
                while i < len(messages) and messages[i][1] == tag:
                    batch += messages[i][0]
                    i += 1
                self.log_area.insert(tk.END, batch, tag)
            if self.max_log_lines > 0:
                line_count = int(self.log_area.index("end-1c").split(".")[0])
                if line_count > self.max_log_lines:
                    self.log_area.delete("1.0", f"{line_count - self.max_log_lines}.0")
            self.log_area.configure(state="disabled")
            self.log_area.yview(tk.END)

        if self.winfo_exists():
            self.after(self.root.setting.log_pull_time, self.poll_log_queue)


    class TextRedirector:
        """ Redirects stdout to a Tkinter Text widget """
        def __init__(self, log_area_widget):
            self.log_area_widget = log_area_widget

        def write(self, str):
            self.log_area_widget.configure(state="normal")
            self.log_area_widget.insert(tk.END, str)
            self.log_area_widget.configure(state="disabled")
            self.log_area_widget.see(tk.END)

        def flush(self):
            pass

    class StreamToLogger:
        """ Redirects stdout/stderr to a logger """
        def __init__(self, logger, log_level=logging.ERROR):
            self.logger = logger
            self.log_level = log_level
            # self.linebuf = ""

        def write(self, buf):
            for line in buf.rstrip().splitlines():
                self.logger.log(self.log_level, line)

        def flush(self):
            pass
    
    class LogTextHandler(logging.Handler):
        def __init__(self, text_widget, log_view: LogView):
            super().__init__()
            self.text_widget = text_widget
            self.log_view = log_view
            self.text_widget.tag_configure("error", foreground="red")
            self.text_widget.tag_configure("warn", foreground="#FFFF00")  # bright yellow
            formatter = logging.Formatter("%(lineno)-4d [%(levelname).4s] %(name)-36s: %(message)s")
            self.setFormatter(formatter)
            self.text_widget.tag_configure("error", foreground="red", lmargin2=82)
            # self.text_widget.tag_configure("error", foreground="black", background="#470E00", lmargin2=82)
            self.text_widget.tag_configure("warning", foreground="yellow", lmargin2=82)
            self.text_widget.tag_configure("info", foreground="lightgreen", lmargin2=82)
            self.text_widget.tag_configure("debug", lmargin2=82)
            self.log_queue = queue.Queue()


        def emit(self, record):
            """Emit a log record to the text widget."""
            for bl in self.log_view.log_blacklist:
                if record.name.startswith(bl + "."):
                    return
            if not LogView.should_show_log(record.name, self.log_view._filter_state):
                return
            self.log_queue.put((record, record.levelno))

    @staticmethod
    def should_show_log(record_name: str, filter_state: dict[str, bool]) -> bool:
        """Return whether *record_name* should be shown (thread-safe: plain bools only)."""
        for _key, prefix, attr in LogView._LAYER_TOGGLES:
            if record_name.startswith(prefix + "."):
                if not filter_state.get("show_core_logs", True):
                    return False
                return filter_state.get(attr, True)

        pkg = plugin_package_from_logger(record_name)
        if pkg is not None:
            if not filter_state.get("show_plugins_logs", True):
                return False
            layer = layer_key_for_plugin_log_record(record_name)
            if layer is None:
                return True
            for key, _prefix, attr in LogView._LAYER_TOGGLES:
                if key == layer:
                    return filter_state.get(attr, True)
            return False

        return True

    #     def emit(self, record):
    #         """ Emit a log record to the text widget """
    #         
    #         for bl in self.log_view.log_blacklist:
    #             if record.name.startswith(bl + "."):
    #                 return
    #
    #         msg = self.format(record)
    #         _first_dot = msg.find('.')
    #         _second_dot = msg.find('.', _first_dot + 1)
    #         code = msg[_second_dot+1 : msg.find('_', _second_dot)] + ":" 
    #         msg = code + msg
    #
    #         # Put the formatted message and level in the queue
    #         self.log_queue.put((msg, record.levelno))
    #
    #
    # def poll_log_queue(self, max_per_poll: int = 10):
    #     processed = 0
    #     messages = []
    #     tag = "debug"
    #    
    #     current_log_level = logging._nameToLevel[self.root.setting.log_level.get()]
    #     
    #     # Collect messages first without modifying the widget
    #     try:
    #         while processed < max_per_poll:
    #             msg, levelno = self.log_handler.log_queue.get_nowait()
    #             if levelno < current_log_level:
    #                 continue
    #
    #             if levelno >= logging.ERROR:
    #                 tag = "error"
    #             elif levelno >= logging.WARNING:
    #                 tag = "warning"
    #             elif levelno >= logging.INFO:
    #                 tag = "info"
    #
    #             
    #             messages.append((msg + "\n", tag))
    #             processed += 1
    #     except queue.Empty:
    #         pass
    #     
    #     # modify the widget once with all collected messages
    #     if messages:
    #         self.log_area.configure(state="normal")
    #         for msg, tag in messages:
    #             self.log_area.insert(tk.END, msg, tag)
    #         
    #         # Limit total lines 
    #         if self.max_log_lines > 0:
    #             line_count = int(self.log_area.index('end-1c').split('.')[0])
    #             if line_count > self.max_log_lines:
    #                 self.log_area.delete('1.0', f'{line_count - self.max_log_lines}.0')
    #         
    #         self.log_area.configure(state="disabled")
    #         self.log_area.yview(tk.END)  # Only scroll once at the end
    #     
    #     # Schedule next poll
    #     self.after(self.root.setting.log_pull_time, self.poll_log_queue)  # Slightly longer interval
    #             


            
               
