from tkinter import ttk
import tkinter as tk

class ValueGauge(ttk.Frame):
    def __init__(self, parent, name:str = "", min_value:float=0, max_value:float=100, variable=None, height=12, **kwargs):
        super().__init__(parent, **kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.current_value = 0
        self.marker_value = 0
        self.variable = variable
        self.config(height=height)
        self.pack_propagate(False)
        self.font = ("Helvetica", 7, "bold")
        self._old_value = 0 # used to not draw if value is same

        if name != "":
            tk.Label(self, text=name, font=self.font, bg="#2f2f2f", fg="gray").pack(side=tk.LEFT, padx=0)
        self.min_label = tk.Label(self, text=f"{min_value:+.2f}", font=self.font, bg="#2f2f2f", fg="gray")
        self.min_label.pack(side=tk.LEFT, padx=0)
        
        self.max_label = tk.Label(self, text=f"{max_value:+.2f}", font=self.font, bg="#2f2f2f", fg="gray")
        self.max_label.pack(side=tk.RIGHT, padx=0)

        self.canvas = tk.Canvas(self, height=height, bg="gray", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
        self.bind("<Configure>", lambda e: self.after_idle(self._draw))

        self.bind("<Configure>", lambda e: self._draw())

        if self.variable is not None:
            self.variable.trace_add("write", self._variable_changed)

        
        # Schedule a delayed draw to ensure widget dimensions are set
        self.after(100, self._draw)
    
    def _variable_changed(self, *args):
        self.set_value(self.variable.get())
    def set_value(self, value):
        if value == self.current_value:
            return  # Skip if no change
        self.current_value = value
        self.after_idle(self._draw)  # Schedule redraw for next idle time

    def set_marker(self, value):
        if value == self.marker_value:
            return  # Skip if no change
        self.marker_value = value
        self.after_idle(self._draw)  # Schedule redraw for next idle time
        self._draw()
    def _draw(self):
        # Avoid unnecessary redraws
        if not self.winfo_ismapped():
            return
            
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        # Skip drawing if dimensions are too small
        if width <= 1 or height <= 1:
            return
            
        self.canvas.delete("all")

        # Bound values to prevent drawing errors
        bounded_marker = max(self.min_value, min(self.max_value, self.marker_value))
        bounded_current = max(self.min_value, min(self.max_value, self.current_value))
        
        # Calculate marker position
        marker_x = ((bounded_marker - self.min_value) / (self.max_value - self.min_value)) * width
        self.canvas.create_line(marker_x, 0, marker_x, height, fill="red", width=2, tags="marker")

        # Draw value text with black background highlight
        text = f"{bounded_current:+4.2f}"
        
        # Calculate text position
        x = ((bounded_current - self.min_value) / (self.max_value - self.min_value)) * width
        y = height / 2

        # Create rectangle first (will be behind text)
        text_id = self.canvas.create_text(x, y, text=text, fill="white", font=self.font, anchor="center", tags="value")
        bbox = self.canvas.bbox(text_id)
        self.canvas.create_rectangle(bbox, fill="black", tags="bg")
        self.canvas.tag_raise("value")
        self.canvas.tag_raise(text_id)
