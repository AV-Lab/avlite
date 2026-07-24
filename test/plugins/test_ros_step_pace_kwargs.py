"""ROSExecuter.step must accept UI pace_* kwargs (AST only — no rclpy)."""

from __future__ import annotations

import ast
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_LAUNCHER = (
    _REPO
    / "avlite-community-plugins"
    / "avlite-executer-ROS2"
    / "p41_ros_launcher.py"
)


def test_ros_executer_step_accepts_pace_kwargs():
    tree = ast.parse(_LAUNCHER.read_text())
    step_args = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ROSExecuter":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "step":
                    step_args = {a.arg for a in item.args.args}
                    break
    assert step_args is not None
    assert {"pace_perception", "pace_replan", "pace_control", "pace_sim"} <= step_args
