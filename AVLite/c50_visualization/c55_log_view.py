from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import sys
import queue

import logging

log = logging.getLogger(__name__)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from c50_visualization.c51_visualizer_app import VisualizerApp

class LogView(ttk.LabelFrame):
    def __init__(self, root: VisualizerApp):
        super().__init__(root, text="Log")
        self.root = root
        self.max_log_lines = 1000  # Maximum number of log lines to keep
        # self.log_queue = queue.Queue()
        # self.after(50, self.process_log_queue)

        self.log_blacklist = set()  # used to filter 'excute', 'plan', 'control' subpackage logs

        self.controls_frame = ttk.Frame(self)
        self.controls_frame.pack(fill=tk.X, side=tk.TOP)

        ttk.Checkbutton(
            self.controls_frame,
            text="Perception",
            variable=self.root.setting.show_perceive_logs,
            command=self.update_log_filter,
        ).pack(side=tk.LEFT)

        ttk.Checkbutton(
            self.controls_frame,
            text="Planning",
            variable=self.root.setting.show_plan_logs,
            command=self.update_log_filter,
        ).pack(side=tk.LEFT)

        ttk.Checkbutton(
            self.controls_frame,
            text="Control",
            variable=self.root.setting.show_control_logs,
            command=self.update_log_filter,
        ).pack(side=tk.LEFT)

        ttk.Checkbutton(
            self.controls_frame,
            text="Execution",
            variable=self.root.setting.show_execute_logs,
            command=self.update_log_filter,
        ).pack(side=tk.LEFT)

        ttk.Checkbutton(
            self.controls_frame,
            text="Visualization",
            variable=self.root.setting.show_vis_logs,
            command=self.update_log_filter,
        ).pack(side=tk.LEFT)
        
        ttk.Checkbutton(
            self.controls_frame,
            text="Tools",
            variable=self.root.setting.show_tools_logs,
            command=self.update_log_filter,
        ).pack(side=tk.LEFT)

        self.rb_db_stdout = ttk.Radiobutton(
            self.controls_frame,
            text="STDOUT",
            variable=self.root.setting.log_level,
            value="STDOUT",
            command=self.update_log_level,
        )
        self.rb_db_stdout.pack(side=tk.RIGHT)

        self.rb_db_warn = ttk.Radiobutton(
            self.controls_frame,
            text="WARN",
            variable=self.root.setting.log_level,
            value="WARN",
            command=self.update_log_level,
        )
        self.rb_db_warn.pack(side=tk.RIGHT)

        self.rb_db_info = ttk.Radiobutton(
            self.controls_frame,
            text="INFO",
            variable=self.root.setting.log_level,
            value="INFO",
            command=self.update_log_level,
        )
        self.rb_db_info.pack(side=tk.RIGHT)

        self.rb_db_debug = ttk.Radiobutton(
            self.controls_frame,
            text="DEBUG",
            variable=self.root.setting.log_level,
            value="DEBUG",
            command=self.update_log_level,
        )
        self.rb_db_debug.pack(side=tk.RIGHT)

        self.log_area = ScrolledText(self, state="disabled", height=12)
        self.log_area.pack(fill=tk.BOTH, side=tk.BOTTOM, expand=True)

        self.after(100, self.update_log_level)
        self.after(100, self.update_log_filter)

        # -------------------------------------------
        # -------------------------------------------
        # -Configure logging-------------------------
        # -------------------------------------------
        logger = logging.getLogger()
        text_handler = LogView.LogTextHandler(self.log_area, self)
        formatter = logging.Formatter("[%(levelname).4s] %(name)-35s (L: %(lineno)3d): %(message)s")
        text_handler.setFormatter(formatter)
        # remove other handlers to avoid duplicate logs
        for handler in logger.handlers:
            logger.removeHandler(handler)
        logger.addHandler(text_handler)
        logger.setLevel(logging.INFO)
        sys.stderr = LogView.StreamToLogger(logger, logging.ERROR)
        log.info("Log initialized.")

    def update_log_filter(self):
        log.warning(f"show_perceive_logs: {self.root.setting.show_perceive_logs.get()}")
        log.info("Log filter updated.")
        # based on blacklist, LogTextHandler will filter out the logs
        (
            self.log_blacklist.discard("c10_perception")
            if self.root.setting.show_perceive_logs.get()
            else self.log_blacklist.add("c10_perception")
        )
        (
            self.log_blacklist.discard("c20_planning")
            if self.root.setting.show_plan_logs.get()
            else self.log_blacklist.add("c20_planning")
        )
        (
            self.log_blacklist.discard("c30_control")
            if self.root.setting.show_control_logs.get()
            else self.log_blacklist.add("c30_control")
        )
        (
            self.log_blacklist.discard("c40_execution")
            if self.root.setting.show_execute_logs.get()
            else self.log_blacklist.add("c40_execution")
        )
        (
            self.log_blacklist.discard("c50_visualization")
            if self.root.setting.show_vis_logs.get()
            else self.log_blacklist.add("c50_visualization")
        )
        (
            self.log_blacklist.discard("c60_tools")
            if self.root.setting.show_tools_logs.get()
            else self.log_blacklist.add("c60_tools")
        )

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
        print("Log setting updated: showing only CRITICAL and STDOUT messages.")
    

    def process_log_queue(self):
        processed = 0
        self.log_area.configure(state="normal")
        try:
            while processed < 50:
                msg, tag = self.log_queue.get_nowait()
                self.log_area.insert(tk.END, msg + "\n", tag if tag else "")
                processed += 1
        except queue.Empty:
            pass

        # Trim log if too long
        line_count = float(self.log_area.index('end-1c').split('.')[0])
        if line_count > self.max_log_lines:
            self.log_area.delete('1.0', f'{int(line_count - self.max_log_lines)}.0')

        self.log_area.configure(state="disabled")
        self.log_area.yview(tk.END)
        self.after(50, self.process_log_queue)

    class LogTextHandler(logging.Handler):
        def __init__(self, text_widget, log_view: LogView):
            super().__init__()
            self.text_widget = text_widget
            self.log_view = log_view
            self.text_widget.tag_configure("error", foreground="red")
            self.text_widget.tag_configure("warn", foreground="#FFFF00")  # bright yellow
            
            # self.message_buffer = []
            # self.buffer_size = 20  # Process messages in batches
            # self.after_id = None
        #
        def emit(self, record):
            for bl in self.log_view.log_blacklist:
                if bl + "." in record.name:
                    return
            msg = self.format(record)

            self.text_widget.configure(state="normal")
            if record.levelno >= logging.ERROR:
                self.text_widget.insert(tk.END, msg + "\n", "error")
            elif record.levelno >= logging.WARNING:
                self.text_widget.insert(tk.END, msg + "\n", "warn")
            else:
                self.text_widget.insert(tk.END, msg + "\n")
            self.text_widget.configure(state="disabled")
            self.text_widget.yview(tk.END)
        # 
        # def emit(self, record):
        #     for bl in self.log_view.log_blacklist:
        #         if bl + "." in record.name:
        #             return
        #             
        #     msg = self.format(record)
        #     tag = None
        #     if record.levelno >= logging.ERROR:
        #         tag = "error"
        #     elif record.levelno >= logging.WARNING:
        #         tag = "warn"
        #         
        #     # self.message_buffer.append((msg, tag))
        #     
        #     # Schedule an update if not already scheduled
        #     # if self.after_id is None:
        #         # self.after_id = self.text_widget.after(50, self.process_buffer)
        #     self.process_buffer()
        #
        #     # self.log_view.log_queue.put((msg, tag))
    
        
        # def process_buffer(self):
        #     # self.after_id = None
        #     # if not self.message_buffer:
        #         # return
        #         
        #     self.text_widget.configure(state="normal")
        #     
        #     # Process the buffer
        #     for msg, tag in self.message_buffer[:self.buffer_size]:
        #         self.text_widget.insert(tk.END, msg + "\n", tag if tag else "")
        #     
        #     # Clear the processed messages
        #     self.message_buffer = self.message_buffer[self.buffer_size:] if len(self.message_buffer) > self.buffer_size else []
        #     
        #     # Trim log if too long
        #     line_count = float(self.text_widget.index('end-1c').split('.')[0])
        #     if line_count > self.log_view.max_log_lines:
        #         self.text_widget.delete('1.0', f'{int(line_count - self.log_view.max_log_lines)}.0')
        #         
        #     self.text_widget.configure(state="disabled")
        #     self.text_widget.yview(tk.END)
        #     
        #     # If there are more messages to process, schedule another update
        #     if self.message_buffer:
        #         self.after_id = self.text_widget.after(50, self.process_buffer)

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
            self.linebuf = ""

        def write(self, buf):
            for line in buf.rstrip().splitlines():
                self.logger.log(self.log_level, line)

        def flush(self):
            pass
