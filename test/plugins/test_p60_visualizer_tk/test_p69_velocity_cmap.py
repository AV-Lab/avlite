import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from avlite.c30_control.c39_settings import ControlSettings
from avlite.plugins.p60_visualizer_tk.p69_plot_lib import _VELOCITY_CMAP, _update_velocity_colored_line


def test_slow_is_greener_than_fast():
    slow = _VELOCITY_CMAP(0.0)[:3]
    fast = _VELOCITY_CMAP(1.0)[:3]
    assert slow[1] > fast[1]  # green channel higher at slow end
    assert fast[0] > slow[0]  # red channel higher at fast end


def test_relative_scale_maps_path_min_max_to_colormap_ends():
    fig, ax = plt.subplots()
    lc = LineCollection([], cmap=_VELOCITY_CMAP, linewidths=5)
    ax.add_collection(lc)
    x = np.linspace(0, 10, 20)
    v = np.linspace(5.0, 11.0, len(x))
    _update_velocity_colored_line(lc, x, np.zeros(len(x)), v, velocity_scale="relative")
    fig.canvas.draw()
    colors = lc.get_colors()
    assert colors[0][1] > colors[-1][0]  # first greener, last redder


def test_absolute_scale_11_mps_is_not_red():
    vmax = float(ControlSettings.c32_ego_max_velocity)
    norm = Normalize(vmin=0.0, vmax=vmax, clip=True)
    rgb = _VELOCITY_CMAP(norm(11.0))[:3]
    assert norm(11.0) < 0.5
    assert rgb[1] > rgb[0]


def test_absolute_scale_ego_max_is_red():
    vmax = float(ControlSettings.c32_ego_max_velocity)
    norm = Normalize(vmin=0.0, vmax=vmax, clip=True)
    rgb = _VELOCITY_CMAP(norm(vmax))[:3]
    assert rgb[0] > rgb[1]
