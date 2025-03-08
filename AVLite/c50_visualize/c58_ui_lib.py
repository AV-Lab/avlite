from tkinter import ttk
import tkinter as tk

class ValueGauge(ttk.Frame):
    def __init__(self, parent, name:str = "hello", min_value:float=0, max_value:float=100, height=10, **kwargs):
        super().__init__(parent, **kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.current_value = 0
        self.marker_value = 0
        self.config(height=height)
        self.pack_propagate(False)
        self.font = ("Helvetica", 8, "bold")

        self.min_label = tk.Label(self, text=f"{min_value:+.2f}", font=self.font, bg="#2f2f2f", fg="gray")
        self.min_label.pack(side=tk.LEFT, padx=0)
        
        self.max_label = tk.Label(self, text=f"{max_value:+.2f}", font=self.font, bg="#2f2f2f", fg="gray")
        self.max_label.pack(side=tk.RIGHT, padx=0)

        # Create canvas between labels
        self.canvas = tk.Canvas(self, height=22, bg="gray", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)

        # Update the binding to use a lambda function that calls self._draw()
        self.bind("<Configure>", lambda e: self._draw())
        
        # Schedule a delayed draw to ensure widget dimensions are set
        self.after(100, self._draw)

    def set_value(self, value):
        self.current_value = value
        self._draw()

    def set_marker(self, value):
        self.marker_value = value
        self._draw()

    def _draw(self):
        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        # Skip drawing if dimensions are too small
        if width <= 1 or height <= 1:
            return

        # Calculate marker position
        marker_x = ((self.marker_value - self.min_value) / (self.max_value - self.min_value)) * width
        self.canvas.create_line(marker_x, 0, marker_x, height, fill="red", width=2)

        # Draw value text with black background highlight
        text = f"{self.current_value:+4.2f}"
        
        # Calculate text position
        x = ((self.current_value - self.min_value) / (self.max_value - self.min_value)) * width
        y = height / 2

        text_id = self.canvas.create_text(x, y, text=text, fill="white", font=self.font, anchor="center")
        bbox = self.canvas.bbox(text_id)
        self.canvas.create_rectangle(bbox, fill="black")
        self.canvas.tag_raise(text_id)
