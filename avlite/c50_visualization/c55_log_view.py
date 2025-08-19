from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import sys
import queue

import logging

from textual import widget

log = logging.getLogger(__name__)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from avlite.c50_visualization.c51_visualizer_app import VisualizerApp

class LogView(ttk.LabelFrame):
    def __init__(self, root: VisualizerApp):
        super().__init__(root, text="Log")
        self.root = root
        self.max_log_lines = self.root.setting.max_log_lines

        # self.log_queue = queue.Queue()
        # self.after(50, self.process_log_queue)

        self.log_blacklist = set()  # used to filter 'excute', 'plan', 'control' subpackage logs

        self.controls_frame = ttk.Frame(self)
        self.controls_frame.pack(fill=tk.X, side=tk.TOP)

        ttk.Checkbutton( self.controls_frame, text="Perception", variable=self.root.setting.show_perceive_logs, command=self.update_log_filter,).pack(side=tk.LEFT)
        ttk.Checkbutton( self.controls_frame, text="Planning", variable=self.root.setting.show_plan_logs, command=self.update_log_filter,).pack(side=tk.LEFT)
        ttk.Checkbutton( self.controls_frame, text="Control", variable=self.root.setting.show_control_logs, command=self.update_log_filter,).pack(side=tk.LEFT)
        ttk.Checkbutton( self.controls_frame, text="Execution", variable=self.root.setting.show_execute_logs, command=self.update_log_filter,).pack(side=tk.LEFT)
        ttk.Checkbutton( self.controls_frame, text="Visualization", variable=self.root.setting.show_vis_logs, command=self.update_log_filter,).pack(side=tk.LEFT)
        ttk.Checkbutton( self.controls_frame, text="Common", variable=self.root.setting.show_common_logs, command=self.update_log_filter,).pack(side=tk.LEFT)
        ttk.Checkbutton( self.controls_frame, text="Extensions", variable=self.root.setting.show_extensions_logs, command=self.update_log_filter,).pack(side=tk.LEFT)


        ttk.Checkbutton(self.controls_frame, text="File", variable=self.root.setting.log_to_file, command=self.update_log_to_file).pack(side=tk.RIGHT)

        self.rb_db_stdout = ttk.Radiobutton( self.controls_frame, text="STDOUT", variable=self.root.setting.log_level, value="STDOUT", command=self.update_log_level,)
        self.rb_db_stdout.pack(side=tk.RIGHT)

        self.rb_db_warn = ttk.Radiobutton( self.controls_frame, text="WARN", variable=self.root.setting.log_level, value="WARN", command=self.update_log_level,)
        self.rb_db_warn.pack(side=tk.RIGHT)

        self.rb_db_info = ttk.Radiobutton( self.controls_frame, text="INFO", variable=self.root.setting.log_level, value="INFO", command=self.update_log_level,)
        self.rb_db_info.pack(side=tk.RIGHT)

        self.rb_db_debug = ttk.Radiobutton( self.controls_frame, text="DEBUG", variable=self.root.setting.log_level, value="DEBUG", command=self.update_log_level,)
        self.rb_db_debug.pack(side=tk.RIGHT)
        
        ttk.Button(self.controls_frame, text="Clear", command=self.clear_log, width=4).pack(side=tk.RIGHT)
        ttk.Button(self.controls_frame, text="Copy", command=self.copy_log, width=4).pack(side=tk.RIGHT)

        
        self.rb_db_debug.pack(side=tk.RIGHT)
        self.log_area = ScrolledText(self, state="disabled", height=self.root.setting.log_view_default_height.get(), wrap=tk.WORD)
        self.log_area.pack(fill=tk.BOTH, side=tk.BOTTOM, expand=True)

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
        log.info("Log initialized.")

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
        raise NotImplementedError("Log to file functionality is not implemented yet.")

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


    def update_log_filter(self):
        log.info("Log filter updated.")
        # based on blacklist, LogTextHandler will filter out the logs
        (
            self.log_blacklist.discard("avlite.c10_perception")
            if self.root.setting.show_perceive_logs.get()
            else self.log_blacklist.add("avlite.c10_perception")
        )
        (
            self.log_blacklist.discard("avlite.c20_planning")
            if self.root.setting.show_plan_logs.get()
            else self.log_blacklist.add("avlite.c20_planning")
        )
        (
            self.log_blacklist.discard("avlite.c30_control")
            if self.root.setting.show_control_logs.get()
            else self.log_blacklist.add("avlite.c30_control")
        )
        (
            self.log_blacklist.discard("avlite.c40_execution")
            if self.root.setting.show_execute_logs.get()
            else self.log_blacklist.add("avlite.c40_execution")
        )
        (
            self.log_blacklist.discard("avlite.c50_visualization")
            if self.root.setting.show_vis_logs.get()
            else self.log_blacklist.add("avlite.c50_visualization")
        )
        (
            self.log_blacklist.discard("avlite.c60_common")
            if self.root.setting.show_common_logs.get()
            else self.log_blacklist.add("avlite.c60_common")
        )
        (
            self.log_blacklist.discard("avlite.extensions")
            if self.root.setting.show_extensions_logs.get()
            else self.log_blacklist.add("avlite.extensions")
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
            """ Emit a log record to the text widget """

            for bl in self.log_view.log_blacklist:
                if record.name.startswith(bl + "."):
                    return

            msg = self.format(record)
            _first_dot = msg.find('.')
            _second_dot = msg.find('.', _first_dot + 1)
            code = msg[_second_dot+1 : msg.find('_', _second_dot)] + ":" 
            msg = code + msg

            self.text_widget.configure(state="normal")
            if record.levelno >= logging.ERROR:
                self.text_widget.insert(tk.END, msg + "\n", "error")
            elif record.levelno >= logging.WARNING:
                self.text_widget.insert(tk.END, msg + "\n", "warn")
            elif record.levelno >= logging.INFO:
                self.text_widget.insert(tk.END, msg + "\n", "info")
            else:
                self.text_widget.insert(tk.END, msg + "\n", "debug")
            self.text_widget.configure(state="disabled")
            self.text_widget.yview(tk.END)
 

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


            
               
