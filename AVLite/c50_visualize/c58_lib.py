from tkinter import ttk
import tkinter as tk
import numpy as np



class ValueGauge(ttk.Frame):
    def __init__(self, parent, min_value:float=0, max_value:float=100, height=10, **kwargs):
        super().__init__(parent, **kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.current_value = min_value
        self.marker_value = 0
        self.config(height=height)
        self.pack_propagate(False)

        self.canvas = tk.Canvas(self, height=20, bg="gray", highlightthickness=0)
        self.canvas.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.bind("<Configure>", lambda e: self._draw())

    def set_value(self, value):
        self.current_value = value
        self._draw()

    def set_marker(self, value):
        self.marker_value = value
        self._draw()

    def _draw(self):
        self.canvas.delete("all")
        width = self.winfo_width()
        height = self.winfo_height()

        # Draw marker
        if self.marker_value is not None:
            marker_x = (
                np.abs(self.marker_value - self.min_value)
                / (self.max_value - self.min_value)
                * width
            )
            self.canvas.create_line(marker_x, 0, marker_x, height, fill="red", width=4)

        # Draw value text with black background highlight
        text = f"{self.current_value:+4.2f}"
        font = ("Helvetica", 8, "bold")
        buffer = 0
        x = np.abs(np.clip((self.current_value - self.min_value) / (self.max_value - self.min_value), self.min_value + buffer, self.max_value - buffer)) * width
        y = height / 2

        # Measure text size
        text_id = self.canvas.create_text(x, y, text=text, fill="white", font=font, anchor="center")
        bbox = self.canvas.bbox(text_id)
        self.canvas.create_rectangle(bbox, fill="black")
        self.canvas.tag_raise(text_id)
