"""Control dials stay undrawn while unmapped and keep their static face across value changes."""

from __future__ import annotations

import tkinter as tk

from avlite.plugins.p60_visualizer_tk.p65_ui_lib import SpeedGauge, SteerWheel


def _map(root: tk.Tk, holder: tk.Frame) -> None:
    # A withdrawn window can return iconic, so <Map> never arrives.
    root.overrideredirect(True)
    root.geometry("400x300+0+0")
    holder.pack()
    root.deiconify()
    root.update()


def test_speed_gauge_skips_draw_until_mapped_and_keeps_face():
    root = tk.Tk()
    root.withdraw()
    try:
        holder = tk.Frame(root)
        gauge = SpeedGauge(holder, dpi_scale=1.0)
        gauge.pack()
        root.update_idletasks()
        assert not gauge.winfo_ismapped()
        assert gauge.find_all() == ()
        gauge.set(12.0, 1.0)
        assert gauge.find_all() == ()

        _map(root, holder)
        assert gauge.winfo_ismapped()
        caption = " ".join(gauge.itemcget(item, "text") for item in gauge.find_withtag("caption"))
        assert "43.2 km/h" in caption
        assert "12.0 m/s" in caption
        x0, _y0, x1, _y1 = gauge.coords("speed")
        assert x1 - x0 >= 0.42 * 96 - 1e-6
        face = gauge.find_withtag("face")
        assert len(face) == 1
        gauge.set(20.0, -2.0)
        assert gauge.find_withtag("face") == face
        caption = " ".join(gauge.itemcget(item, "text") for item in gauge.find_withtag("caption"))
        assert "72.0 km/h" in caption
        assert "20.0 m/s" in caption
        assert len(gauge.find_withtag("speed")) == 1
        assert len(gauge.find_withtag("needle")) == 1
        assert gauge.winfo_reqheight() < 148

        gauge.pack_forget()
        root.update()
        assert not gauge.winfo_ismapped()
        drawn = gauge.find_all()
        gauge.set(30.0, 2.0)
        assert gauge.find_all() == drawn
        gauge.pack()
        root.update()
        caption = " ".join(gauge.itemcget(item, "text") for item in gauge.find_withtag("caption"))
        assert "108.0 km/h" in caption
        assert "30.0 m/s" in caption
        assert gauge.find_withtag("face") == face
    finally:
        root.destroy()


def test_steer_wheel_skips_draw_until_mapped_and_keeps_face():
    root = tk.Tk()
    root.withdraw()
    try:
        holder = tk.Frame(root)
        wheel = SteerWheel(holder, dpi_scale=1.0)
        wheel.pack()
        root.update_idletasks()
        assert not wheel.winfo_ismapped()
        assert wheel.find_all() == ()
        wheel.set(0.3)
        assert wheel.find_all() == ()

        _map(root, holder)
        assert wheel.winfo_ismapped()
        face = wheel.find_withtag("face")
        assert len(face) == 1
        before = wheel.coords("spoke0")
        wheel.set(-0.4)
        assert wheel.find_withtag("face") == face
        assert wheel.coords("spoke0") != before
        assert len(wheel.find_withtag("accent")) == 1
        assert wheel.winfo_reqheight() > 96
        assert "-0.40 rad" in wheel.itemcget("caption", "text")
    finally:
        root.destroy()


def test_theme_event_on_a_sibling_does_not_rebuild_the_face():
    root = tk.Tk()
    root.withdraw()
    try:
        holder = tk.Frame(root)
        gauge = SpeedGauge(holder, dpi_scale=1.0)
        gauge.pack()
        sibling = tk.Frame(holder)
        sibling.pack()
        _map(root, holder)
        gauge.set(10.0, 1.0)
        face = gauge.find_withtag("face")
        assert len(face) == 1
        sibling.event_generate("<<ThemeChanged>>")
        assert gauge.find_withtag("face") == face
        gauge.event_generate("<<ThemeChanged>>")
        assert gauge.find_withtag("face") == face
    finally:
        root.destroy()
