"""Tests for LogView filtering in p68_log_view."""

import logging

from avlite.plugins.p60_visualizer_tk.p68_log_view import LogView


def _filters(**overrides: bool) -> dict[str, bool]:
    base = {
        "p68_show_perceive_logs": True,
        "p68_show_plan_logs": True,
        "p68_show_control_logs": True,
        "p68_show_execute_logs": True,
        "p68_show_vis_logs": True,
        "p68_show_common_logs": True,
        "p68_show_core_logs": True,
        "p68_show_plugins_logs": True,
        "p68_disable_log": False,
        "log_to_file": False,
    }
    base.update(overrides)
    return base


def test_should_show_log_core_layer():
    name = "avlite.c30_control.c32_control_strategy"
    assert LogView.should_show_log(name, _filters(p68_show_control_logs=True))
    assert not LogView.should_show_log(name, _filters(p68_show_control_logs=False))
    assert not LogView.should_show_log(name, _filters(p68_show_core_logs=False, p68_show_control_logs=True))


def test_should_show_log_plugins_master_toggle():
    name = "avlite.plugins.avlite_controller_joystick.p30_joystick_controller"
    assert LogView.should_show_log(name, _filters(p68_show_plugins_logs=True))
    assert not LogView.should_show_log(name, _filters(p68_show_plugins_logs=False))


def test_should_show_log_pnx_routes_to_layer():
    name = "avlite.plugins.avlite_controller_joystick.p30_joystick_controller"
    assert LogView.should_show_log(name, _filters(p68_show_plugins_logs=True, p68_show_control_logs=True))
    assert not LogView.should_show_log(name, _filters(p68_show_plugins_logs=True, p68_show_control_logs=False))


def test_should_show_log_community_plugin_bucket():
    name = "avlite.plugins.sample_avlite_plugin.handler"
    assert LogView.should_show_log(name, _filters(p68_show_plugins_logs=True))
    assert not LogView.should_show_log(name, _filters(p68_show_plugins_logs=False))


def test_record_code_prefix():
    cases = {
        "avlite.c10_perception.c15_perception_algs": "c15",
        "avlite.c20_planning": "c20",
        "avlite.c30_control.c34_stanley": "c34",
        "avlite.plugins.p40_executer_ROS2.p42_perception_node": "p42",
        "avlite.plugins.p40_bridge_carla.carla_bridge": "p40",
        "avlite.plugins.p30_controller_joystick.p31_joystick_controller": "p31",
        "avlite.plugins.sample_avlite_plugin.test_plugin": "sample_avlite_plugin",
        "avlite.plugins.avlite_bridge_carla.carla_bridge": "avlite_bridge_carla",
        "avlite.plugins.avlite_controller_joystick.p30_joystick_controller": "p30",
    }
    for record_name, expected in cases.items():
        assert LogView.record_code_prefix(record_name) == expected, record_name


def test_log_view_prefix_from_record_name():
    record = logging.LogRecord(
        name="avlite.c20_planning",
        level=logging.INFO,
        pathname="",
        lineno=76,
        msg="Global plan set: ego Frenet s=106.86 d=-607.58",
        args=(),
        exc_info=None,
    )
    assert LogView.record_code_prefix(record.name) == "c20"
