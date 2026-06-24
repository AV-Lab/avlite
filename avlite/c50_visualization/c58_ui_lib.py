import os
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

class ValueGauge(ttk.Frame):
    def __init__(
        self,
        parent,
        name: str = "",
        min_value: float = 0,
        max_value: float = 100,
        variable=None,
        height=12,
        dpi_scale: float | None = None,
        **kwargs,
    ):
        super().__init__(parent, **kwargs)
        if dpi_scale is None:
            dpi_scale = getattr(parent, "_dpi_scale", None)
        if dpi_scale is None:
            dpi_scale = get_dpi_scale(parent)

        self.min_value = min_value
        self.max_value = max_value
        self.current_value = 0
        self.marker_value = 0
        self.variable = variable
        self.pack_propagate(False)
        self.font = ("Helvetica", 7, "bold")
        self._old_value = 0  # used to not draw if value is same
        gauge_height = scaled(height, dpi_scale)

        if name != "":
            tk.Label(self, text=name, font=self.font).pack(side=tk.LEFT, padx=0)

        min_label = tk.Label(self, text=f"{min_value:+.2f}", font=self.font)
        min_label.pack(side=tk.LEFT, padx=0)

        max_label = tk.Label(self, text=f"{max_value:+.2f}", font=self.font)
        max_label.pack(side=tk.RIGHT, padx=0)

        self.canvas = tk.Canvas(self, height=gauge_height, bg="gray", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)

        total_height = gauge_height
        self.config(height=total_height)

        self.bind("<Configure>", lambda e: self.__draw())

        if self.variable is not None:
            self.variable.trace_add("write", self.__variable_changed)

        
        # Schedule a delayed draw to ensure widget dimensions are set
        self.after(100, self.__draw)
    
    def __variable_changed(self, *args):
        self.set_value(self.variable.get())

    def set_value(self, value):
        if value == self.current_value:
            return  # Skip if no change
        self.current_value = value
        self.after_idle(self.__draw)  # Schedule redraw for next idle time

    def set_marker(self, value):
        if value == self.marker_value:
            return  # Skip if no change
        self.marker_value = value
        self.after_idle(self.__draw)  # Schedule redraw for next idle time
        # self.__draw()

    def __draw(self):
        # Avoid unnecessary redraws
        if not self.winfo_ismapped():
            return
            
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
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
        # self.canvas.coords(text_id, x, y + 2)  # Move text down within the box
        self.canvas.create_rectangle(bbox, fill="black", tags="bg")
        self.canvas.tag_raise("value")
        self.canvas.tag_raise(text_id)


class ThemedInputDialog:
    def __init__(self, parent, title, prompt, initial=""):
        self.result = None

        # Create the dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.transient(parent)
        s = get_dpi_scale(self.dialog, parent=parent)
        self.dialog.minsize(scaled(300, s), scaled(100, s))
        
        self.dialog.bind("<Escape>", lambda e: self.on_cancel())  # Bind Escape key to cancel

        
        
        frame = ttk.Frame(self.dialog)
        frame.pack(expand=True, fill=tk.BOTH)

        top_frame = ttk.Frame(frame)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Add prompt and entry field
        ttk.Label(top_frame, text=prompt).pack(side=tk.LEFT, padx=10)
        self.entry = ttk.Entry(top_frame, width=max(10, scaled(20, s)))
        self.entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10, pady=5)
        if initial:
            self.entry.insert(0, initial)
            self.entry.select_range(0, tk.END)
        self.entry.focus_set()
        
        # Add button frame
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Button(btn_frame, text="OK", command=self.on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.on_cancel).pack(side=tk.RIGHT , padx=5)
        
        # Make dialog modal
        self.dialog.grab_set()
        self.dialog.wait_window()
    
    def on_ok(self):
        self.result = self.entry.get()
        self.dialog.destroy()
    
    def on_cancel(self):
        self.dialog.destroy()

class ThemedTwoInputDialog:
    def __init__(self, parent, title, prompt1="", prompt2="", initial1="", initial2=""):
        self.result = None

        # Create the dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.transient(parent)
        s = get_dpi_scale(self.dialog, parent=parent)
        self.dialog.minsize(scaled(360, s), scaled(140, s))
        
        self.dialog.bind("<Escape>", lambda e: self.on_cancel())  # Bind Escape key to cancel
        
        frame = ttk.Frame(self.dialog)
        # frame.pack(expand=True, fill=tk.BOTH)
        frame.grid(row=0, column=0, sticky="nsew")
        self.dialog.grid_rowconfigure(0, weight=1)
        self.dialog.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=0)  # Entry row doesn't expand
        frame.grid_rowconfigure(1, weight=0)  # Entry row doesn't expand
        frame.grid_rowconfigure(2, weight=1)  # Button row expands to push buttons down
        frame.grid_columnconfigure(1, weight=1)


        # Add prompt and entry field
        ttk.Label(frame, text=prompt1).grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        self.first_entry = ttk.Entry(frame)
        self.first_entry.grid(row=0, column=1, padx=10, pady=5, sticky=tk.EW)
        
        self.first_entry.insert(0, initial1)
        
        ttk.Label(frame, text=prompt2).grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        self.second_entry = ttk.Entry(frame, width=max(20, scaled(30, s)))
        self.second_entry.grid(row=1, column=1, padx=10, pady=5, sticky=tk.EW)
        self.second_entry.insert(0, initial2)
        
        
        ttk.Button(frame, text="OK", command=self.on_ok).grid(row=2, column=0, padx=5, pady=5, sticky="sw")
        ttk.Button(frame, text="Cancel", command=self.on_cancel).grid(row=2, column=1, padx=5, pady=5, sticky="se")
        
        self.dialog.transient(parent)
        self.dialog.update_idletasks()
        self.dialog.deiconify()  # Ensure window is visible
        self.dialog.wait_visibility()  # Wait until window is visible
        self.dialog.grab_set()
        self.dialog.wait_window()
    
    def on_ok(self):
        self.result = (self.first_entry.get(), self.second_entry.get())
        self.dialog.destroy()
    
    def on_cancel(self):
        self.dialog.destroy()


class HoverTooltip:
    """Show *text* in a small popup while the pointer is over *widget*."""

    def __init__(self, widget: tk.Widget, text: str, *, delay_ms: int = 400) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._tip: tk.Toplevel | None = None
        self._after_id: str | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self) -> None:
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self) -> None:
        self._after_id = None
        if self._tip is not None:
            return

        s = get_dpi_scale(self.widget)
        x = self.widget.winfo_rootx() + scaled(20, s)
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + scaled(4, s)
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tip,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            wraplength=scaled(360, s),
            padx=scaled(6, s),
            pady=scaled(4, s),
        )
        label.pack()
        self._tip = tip

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


def attach_schema_tooltip(widget, settings_cls, field_name: str) -> None:
    from avlite.c60_common.c68_settings_schema import field_tooltip_text

    tip = field_tooltip_text(settings_cls, field_name)
    if tip:
        HoverTooltip(widget, tip)


class TkSettingsBinder:
    """Read/write settings backed by ``tk.Variable`` attributes."""

    def get_value(self, setting, attr_name: str):
        from avlite.c60_common.c68_settings_schema import PlainBinder

        attr_value = getattr(setting if not isinstance(setting, type) else setting, attr_name)
        if isinstance(attr_value, tk.Variable):
            return attr_value.get()
        return PlainBinder().get_value(setting, attr_name)

    def set_value(self, setting, attr_name: str, value) -> None:
        attr_value = getattr(setting if not isinstance(setting, type) else setting, attr_name)
        if isinstance(attr_value, tk.BooleanVar):
            attr_value.set(bool(value))
        elif isinstance(attr_value, tk.IntVar):
            attr_value.set(int(value))
        elif isinstance(attr_value, tk.DoubleVar):
            attr_value.set(float(value))
        elif isinstance(attr_value, tk.Variable):
            attr_value.set(value)
        else:
            from avlite.c60_common.c68_settings_schema import PlainBinder

            PlainBinder().set_value(setting, attr_name, value)


_DPI_MIN = 1.0
_DPI_MAX = 3.0


def scaled(n: float, scale: float) -> int:
    """Scale a pixel value and round to int."""
    return round(n * scale)


def scaled_font(scale: float, family: str, size: int, **kwargs) -> tuple:
    """Return a font tuple scaled for geometry scale (plugin README rendering only)."""
    font_size = max(1, scaled(size, scale))
    parts: list = [family, font_size]
    if "weight" in kwargs:
        parts.append(kwargs["weight"])
    if "slant" in kwargs:
        parts.append(kwargs["slant"])
    return tuple(parts)


def _font_scale_from_default() -> float:
    """Estimate scale from TkDefaultFont when PPI alone is insufficient (common on Linux)."""
    try:
        font = tkfont.nametofont("TkDefaultFont")
        size = font.cget("size")
        if isinstance(size, str):
            size = int(size)
        size = abs(int(size))
        if size <= 0:
            return _DPI_MIN
        # Negative size is pixels; positive is points. Baseline ~10 pt / 12 px.
        baseline = 12 if font.cget("size") < 0 else 10
        return max(_DPI_MIN, min(_DPI_MAX, size / baseline))
    except (tk.TclError, ValueError, TypeError):
        return _DPI_MIN


def get_dpi_scale(widget: tk.Misc, parent: tk.Misc | None = None) -> float:
    """Pixels-per-inch normalised to 96 dpi, with font and GDK fallbacks."""
    if parent is not None:
        inherited = getattr(parent, "_dpi_scale", None)
        if inherited is not None:
            return float(inherited)

    ppi_scale = _DPI_MIN
    try:
        widget.update_idletasks()
        ppi_scale = max(_DPI_MIN, min(_DPI_MAX, widget.winfo_fpixels("1i") / 96.0))
    except tk.TclError:
        pass

    gdk_scale = os.environ.get("GDK_SCALE")
    env_scale = _DPI_MIN
    if gdk_scale:
        try:
            env_scale = max(_DPI_MIN, min(_DPI_MAX, float(gdk_scale)))
        except ValueError:
            pass

    font_scale = _font_scale_from_default()
    return max(ppi_scale, env_scale, font_scale)


def configure_treeview_style(style: ttk.Style, name: str, scale: float = 1.0) -> None:
    """Set Treeview row height and heading font to match the default UI font."""
    font = tkfont.nametofont("TkDefaultFont")
    rowheight = font.metrics("linespace") + scaled(4, scale)
    style.configure(f"{name}.Treeview", rowheight=rowheight)
    style.configure(f"{name}.Treeview.Heading", font=font)

