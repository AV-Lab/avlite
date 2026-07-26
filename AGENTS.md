# AGENTS.md

## Cursor Cloud specific instructions

AVLite is a single, self-contained Python (3.10+) autonomous-vehicle stack — a library
plus a Tkinter/matplotlib desktop visualizer and CLI. There are no servers, databases, or
external services required; everything runs in-process against the built-in `BasicSim`
world bridge.

### Environment
- Python deps install to the user site (`pip install -e ".[dev]"`), no virtualenv is used.
- The GUI requires the `python3-tk` system package. It is preinstalled in the VM snapshot
  and is NOT part of the pip update script; if `import tkinter` ever fails, reinstall it
  with `sudo apt-get install -y python3-tk`.
- `rich` is an optional dependency needed only for headless mode; the update script installs it.
- A desktop display is available at `DISPLAY=:1` for the GUI.

### Running the app (see README "Quick Start" / "Headless Mode" for full details)
- GUI visualizer: `python -m avlite` — pick a profile in the Config/Settings tab, click
  **Start** to run the perceive→plan→control loop, right-click a plot to spawn an NPC.
- Headless dashboard: `python -m avlite headless -p default` (add `--perceive` to run the
  full perception loop). The `rich` Live dashboard only renders on a real TTY — when
  capturing output, run it inside `tmux`, not through a plain pipe (a pipe produces no
  visible frames even though the loop is running fine).

### Non-obvious gotcha: profiles for headless mode
Config profiles ship in `avlite/configs/` but are read from the user config dir
(`~/.config/avlite/`). The GUI copies the bundled profiles into the user dir automatically
on startup (via `list_profiles()`), so it works out of the box. Headless mode does NOT do
this copy first, so a fresh machine reports `Profile 'default' not found`. Trigger the copy
once (e.g. launch the GUI, or run
`python -c "from avlite.c60_apps.c65_setting_utils import list_profiles; list_profiles()"`)
before using `python -m avlite headless`.

### Testing
- Fast suite: `pytest` (excludes `slow` and `requires_data` markers; see `pyproject.toml`).
- Full suite: `pytest -m ""`.
- Known expected failure: `test/plugins/test_ros_step_pace_kwargs.py` needs the optional,
  git-ignored sibling repo `avlite-community-plugins/` (the ROS2 community plugin) checked
  out next to the package. It is not present by default and this failure is unrelated to
  core AVLite. Everything else passes.
