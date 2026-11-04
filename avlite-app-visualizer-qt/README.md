# avlite-app-visualizer-qt

High-performance **Qt5** visualizer for [AVLite](https://github.com/AV-Lab/avlite) — an alternative to the built-in Tk / Matplotlib app (`p60_visualizer_tk`).

**Plugin name:** `avlite-app-visualizer-qt`  
**CLI:** `avlite qt`

## Why Qt5 + PyQt5 + pyqtgraph

| Choice | Rationale |
|--------|-----------|
| **Qt5** (not Qt6) | Matches the requested Qt5 app strategy; keeps ABI aligned with common robotics/desktop stacks. |
| **PyQt5** (not PySide2) | PySide2 has no Python 3.12 wheels (last public builds are 5.13.x). PyQt5 5.15.x ships `cp38-abi3` manylinux wheels that work on 3.10–3.12. |
| **pyqtgraph** | Incremental `PlotDataItem.setData` / OpenGL path instead of full Matplotlib figure redraws — keeps the UI thread responsive at 60 Hz while `AsyncThreadedExecuter` / `exec.step` runs sim & planning. |

## Layout (split-out ready)

This directory follows the same community-plugin conventions as
[`avlite-bridge-carla`](https://github.com/AV-Lab/avlite-bridge-carla) and
[`avlite-controller-joystick`](https://github.com/AV-Lab/avlite-controller-joystick):

```
avlite-app-visualizer-qt/
├── __init__.py
├── app_strategy.py          # AppStrategy → `avlite qt`
├── main_window.py
├── plot_widgets.py          # pyqtgraph global + local views
├── stack_panels.py          # perception / plan / control / execution
├── log_panel.py
├── setting_model.py
├── settings.py
├── shortcuts.py
├── qt_binding.py
├── requirements.txt
├── avlite-app-visualizer-qt.yaml
├── README.md
└── test/
```

While developed inside the AVLite monorepo, a thin shim
`avlite/plugins/p60_visualizer_qt/` loads this package at `bootstrap_apps()` so
the subcommand is registered. To publish as a standalone repo later: move this
folder to `github.com/AV-Lab/avlite-app-visualizer-qt`, keep the shim or drop it
and install via the Plugins browser + `c62_community_plugins`.

## Install

```bash
# From an AVLite checkout (monorepo / this PR)
pip install -e .
pip install -r avlite-app-visualizer-qt/requirements.txt

# Or as a community plugin
git clone https://github.com/AV-Lab/avlite-app-visualizer-qt \
  ~/.local/share/avlite/plugins/avlite-app-visualizer-qt
pip install -r ~/.local/share/avlite/plugins/avlite-app-visualizer-qt/requirements.txt
```

Profile snippet (optional — the built-in shim already finds the monorepo path):

```yaml
c69_apps:
  c62_community_plugins:
    avlite-app-visualizer-qt: avlite-app-visualizer-qt
```

## Usage

```bash
# GUI (BasicSim via default / selected profile)
avlite qt

# Headless / CI
QT_QPA_PLATFORM=offscreen avlite qt --offscreen

# Explicit profile
avlite qt --profile default
```

### Feature map (parity with Tk visualizer)

- **Plots:** global map/raceline/HD lanes, local XY + Frenet, ego/NPC boxes, global & local plans, lattice, LiDAR, predictions, past path
- **Interaction:** LMB drag teleport (+ heading), Ctrl+LMB pan (global), scroll zoom, RMB start/goal replan (global), RMB drag spawn NPC (local)
- **Stack UI:** perception / detect-track-predict / map / loc, global & local planners, pipeline sub-strategies, controller gauges + WASD, execution Start/Stop/Step/Reset, Δt + pace, bridge & executer switching, execution tasks
- **Runtime:** profile load (File → Settings / Profiles, shortcut `F`), stack reload (`R`), shortcut mode (`S`)
- **Logs:** filterable log panel
- **Perf:** UI refresh timer decoupled from `exec.step`; plot path is pyqtgraph-only

### Shortcuts (defaults)

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `X` / `C` / `T` | Toggle / step / reset execution |
| `R` | Reload stack |
| `F` | Switch to next profile |
| `S` | Shortcut panel mode |
| `W` `A` `S` `D` | Drive |
| `+` / `-` | Zoom |

## Tests

```bash
QT_QPA_PLATFORM=offscreen python -m pytest avlite-app-visualizer-qt/test -q
```

## License

MIT (same as AVLite). PyQt5 itself is under the Riverbank GPL/commercial dual license — use a GPL-compatible distribution or a commercial PyQt license if your product needs it; the plugin code here remains MIT.
