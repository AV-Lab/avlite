# Quick Start

Get AVLite running in a few minutes: install the package, launch the visualizer, drive the built-in simulator, then run the same profile headless.

!!! note "Requirements"
    - Python **3.10+**
    - Linux, macOS, or Windows with a working Tkinter (bundled with most CPython builds)

## 1. Install

=== "PyPI (recommended)"

    ```bash
    pip install avlite
    ```

=== "From requirements"

    ```bash
    git clone https://github.com/AV-Lab/avlite.git
    cd avlite
    pip install -r requirements.txt
    ```

=== "Editable (with dev tools)"

    ```bash
    git clone https://github.com/AV-Lab/avlite.git
    cd avlite
    pip install -e ".[dev]"
    ```

The core stack depends only on NumPy, Matplotlib, PyYAML, Shapely, NetworkX, SciPy, Pydantic, and ttkthemes. The optional headless dashboard needs `rich` (`pip install rich`).

## 2. Launch the visualizer

```bash
avlite
```

<figure markdown="span">
  ![AVLite Tk visualizer](imgs/tk_visualizer.png){ width="720" }
  <figcaption>The Tk visualizer with real-time plots and configuration panels.</figcaption>
</figure>

## 3. Drive the built-in simulator

The default profile uses **BasicSim**, a dependency-free 2D simulator, so there is nothing else to install.

- [ ] **Config tab** — pick a profile (start with `default`).
- [ ] **Start/Stop Stack** — start the simulation loop.
- [ ] **Stop, then Step** — advance one tick at a time for frame-by-frame inspection.
- [ ] **Left-click-drag the plot** — teleport ego and set heading.
- [ ] **Right-click-drag the plot** — spawn an NPC and set its heading.
- [ ] **Tune parameters** — adjust perception, planning, and control settings live in the GUI panels.
- [ ] **Save Global Plan** — use the ⬇ button in the Planning panel to export the current plan as JSON.

!!! tip "Slow down or free-run"
    With Sim Δt pacing on, a larger period runs slower than real time (practical slow motion). Uncheck pacing for best-effort / wall-clock integration.

!!! tip "Vim-style shortcuts"
    The visualizer is fully keyboard-drivable with vim motions for a fast, mouse-free workflow: `j`/`k` (or `↑`/`↓`) scroll the log, `g`/`G` jump to top/bottom, and `Ctrl+u`/`Ctrl+d` half-page scroll. Single-key actions cover planning (`n`/`b`/`r`), control (`w`/`a`/`s`/`d`), and execution (`x`/`c`/`t`). The on-screen **Shortcuts** panel lists every binding.

!!! tip "Optional plugins (CARLA, Gazebo, ROS2, …)"
    Install world-bridge and other community plugins from the **Plugins** browser (`avlite plugins`). Installing or updating needs `git` on your PATH (and `pip` if the plugin has extra dependencies). See [Community Plugins](overview.md#community-plugins) and [Plugin Development](plugin-development.md).

## 4. Save a profile and run headless

Once you have a configuration you like, save it as a named profile from the Config tab, then run it without a GUI — ideal for a robot, server, or CI runner.

```bash
# Default profile
avlite headless

# A saved profile
avlite headless -p my_robot_profile
avlite headless my_robot_profile        # positional shortcut

# Tune log noise / loop rates
avlite headless -p my_robot_profile \
    --log-level WARNING \
    --control-dt 0.01 --replan-dt 0.5 --perceive
```

A live `rich` dashboard shows FPS, ego state, lap counter, and recent log lines. Press ++ctrl+c++ to stop.

!!! tip "Recommended workflow"
    1. **Configure** with `avlite` — pick the bridge and strategies, and tune parameters in the GUI.
    2. **Save** the result as a named profile from the Config tab.
    3. **Deploy** with `avlite headless -p <profile>`. The same YAML profile drives both modes, so what you see in the visualizer is what the robot runs.

## Next steps

- [Architecture](architecture.md) — layers, capability system, and data flow.
- [Algorithms](algorithms.md) — planning algorithms and lattice parameters.
- [Settings Naming](settings-naming.md) — YAML key prefixes and validation.
- [Plugin Development](plugin-development.md) — build custom perception, planning, control, and world-bridge strategies.
