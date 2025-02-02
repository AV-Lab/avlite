from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import sys

import logging
log = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from c50_visualize.c51_visualizer_app import VisualizerApp


class LogView(ttk.LabelFrame):
    def __init__(self, root: VisualizerApp):
        super().__init__(root, text="Log")
        self.root = root
        self.log_blacklist = set()  # used to filter 'excute', 'plan', 'control' subpackage logs

        self.controls_frame = ttk.Frame(self)
        self.controls_frame.pack(fill=tk.X, side=tk.TOP)

        self.ck_perceive = ttk.Checkbutton(
            self.controls_frame,
            text="Perceive",
            variable=self.root.data.show_perceive_logs,
            command=self.update_log_filter,
        )
        self.ck_perceive.pack(side=tk.LEFT)

        self.ck_plan = ttk.Checkbutton(
            self.controls_frame,
            text="Plan",
            variable=self.root.data.show_plan_logs,
            command=self.update_log_filter,
        )
        self.ck_plan.pack(side=tk.LEFT)

        self.ck_control = ttk.Checkbutton(
            self.controls_frame,
            text="Control",
            variable=self.root.data.show_control_logs,
            command=self.update_log_filter,
        )
        self.ck_control.pack(side=tk.LEFT)

        self.ck_exec = ttk.Checkbutton(
            self.controls_frame,
            text="Execute",
            variable=self.root.data.show_execute_logs,
            command=self.update_log_filter,
        )
        self.ck_exec.pack(side=tk.LEFT)

        self.ck_vis = ttk.Checkbutton(
            self.controls_frame,
            text="Visualize",
            variable=self.root.data.show_vis_logs,
            command=self.update_log_filter,
        )
        self.ck_vis.pack(side=tk.LEFT)

        self.rb_db_stdout = ttk.Radiobutton(
            self.controls_frame,
            text="STDOUT",
            variable=self.root.data.debug_option,
            value="STDOUT",
            command=self.update_log_level,
        )
        self.rb_db_stdout.pack(side=tk.RIGHT)
        
        self.rb_db_warn = ttk.Radiobutton(
            self.controls_frame,
            text="WARN",
            variable=self.root.data.debug_option,
            value="WARN",
            command=self.update_log_level,
        )
        self.rb_db_warn.pack(side=tk.RIGHT)

        self.rb_db_info = ttk.Radiobutton(
            self.controls_frame,
            text="INFO",
            variable=self.root.data.debug_option,
            value="INFO",
            command=self.update_log_level,
        )
        self.rb_db_info.pack(side=tk.RIGHT)

        self.rb_db_debug = ttk.Radiobutton(
            self.controls_frame,
            text="DEBUG",
            variable=self.root.data.debug_option,
            value="DEBUG",
            command=self.update_log_level,
        )
        self.rb_db_debug.pack(side=tk.RIGHT)


        self.log_area = ScrolledText(self, state="disabled", height=12)
        self.log_area.pack(fill=tk.BOTH, side=tk.BOTTOM, expand=True)


        # -------------------------------------------
        # -------------------------------------------
        # -Configure logging-------------------------
        # -------------------------------------------
        logger = logging.getLogger()
        text_handler = LogView.LogTextHandler(self.log_area, self)
        formatter = logging.Formatter("[%(levelname).4s] %(name)-30s (L: %(lineno)3d): %(message)s")
        text_handler.setFormatter(formatter)
        # remove other handlers to avoid duplicate logs
        for handler in logger.handlers:
            logger.removeHandler(handler)
        logger.addHandler(text_handler)
        logger.setLevel(logging.INFO)
        sys.stderr = LogView.StreamToLogger(logger, logging.ERROR)
        log.info("Log initialized.")



    def update_log_filter(self):
        log.warning(f"show_perceive_logs: {self.root.data.show_perceive_logs.get()}")
        log.info("Log filter updated.")
        # based on blacklist, LogTextHandler will filter out the logs
        (self.log_blacklist.discard("c10_perceive") if self.root.data.show_perceive_logs.get() else self.log_blacklist.add("c10_perceive"))
        (self.log_blacklist.discard("c20_plan") if self.root.data.show_plan_logs.get() else self.log_blacklist.add("c20_plan"))
        (self.log_blacklist.discard("c30_control") if self.root.data.show_control_logs.get() else self.log_blacklist.add("c30_control"))
        (self.log_blacklist.discard("c40_execute") if self.root.data.show_execute_logs.get() else self.log_blacklist.add("c40_execute"))
        (self.log_blacklist.discard("c50_visualize") if self.root.data.show_vis_logs.get() else self.log_blacklist.add("c50_visualize"))
        


    def update_log_level(self):
        if self.rb_db_debug.instate(["selected"]):
            logging.getLogger().setLevel(logging.DEBUG)
            log.debug("Log setting updated to DEBUG.")
        elif self.rb_db_info.instate(["selected"]):
            logging.getLogger().setLevel(logging.INFO)
            log.info("Log setting updated to INFO.")
        elif self.rb_db_warn.instate(["selected"]):
            logging.getLogger().setLevel(logging.WARNING)
            log.warning("Log setting updated to WARNING.")
        if self.rb_db_stdout.instate(["selected"]):
            logging.getLogger().setLevel(logging.CRITICAL)
            sys.stdout = LogView.TextRedirector(self.log_area)
        else:
            sys.stdout = sys.__stdout__
        print("Log setting updated: routing CRITICAL and stdout.")

    class LogTextHandler(logging.Handler):
        def __init__(self, text_widget, log_view: LogView):
            super().__init__()
            self.text_widget = text_widget
            self.log_view = log_view
            self.text_widget.tag_configure("error", foreground="red")
            # self.text_widget.tag_configure("warning", foreground="yellow") 
            self.text_widget.tag_configure("warning", foreground="#FFFF00")  # bright yellow

        def emit(self, record):
            for bl in self.log_view.log_blacklist:
                if bl + "." in record.name:
                    return
            msg = self.format(record)

            self.text_widget.configure(state="normal")
            if record.levelno >= logging.ERROR:
                self.text_widget.tag_configure("error", foreground="red")
                self.text_widget.insert(tk.END, msg + "\n", "error")
            else:
                self.text_widget.insert(tk.END, msg + "\n")
            self.text_widget.configure(state="disabled")
            self.text_widget.yview(tk.END)

    class TextRedirector:
        def __init__(self, widget):
            self.widget = widget

        def write(self, str):
            self.widget.configure(state="normal")
            self.widget.insert(tk.END, str)
            self.widget.configure(state="disabled")
            self.widget.see(tk.END)

        def flush(self):
            pass

    class StreamToLogger:
       def __init__(self, logger, log_level=logging.ERROR):
           self.logger = logger
           self.log_level = log_level
           self.linebuf = ''

       def write(self, buf):
           for line in buf.rstrip().splitlines():
               self.logger.log(self.log_level, line)

       def flush(self):
           pass
