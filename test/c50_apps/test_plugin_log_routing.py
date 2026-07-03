"""Tests for plugin log routing helpers."""

import logging

from avlite.plugins.p50_visualizer_tk.p57_log_view import LogView
from avlite.c50_apps.c53_plugins import (
    layer_key_for_plugin_log_record,
    layer_key_for_plugin_package,
    plugin_module_from_logger,
    plugin_package_from_logger,
)


def _filters(**overrides: bool) -> dict[str, bool]:
    base = {
        "p57_show_perceive_logs": True,
        "p57_show_plan_logs": True,
        "p57_show_control_logs": True,
        "p57_show_execute_logs": True,
        "p57_show_vis_logs": True,
        "p57_show_common_logs": True,
        "p57_show_core_logs": True,
        "p57_show_plugins_logs": True,
        "p57_disable_log": False,
        "log_to_file": False,
    }
    base.update(overrides)
    return base


def test_layer_key_built_in_plugins():
    assert layer_key_for_plugin_package("p10_perception_MO_prediction") == "perception"
    assert layer_key_for_plugin_package("avlite-bridge-carla") == "execution"
    assert layer_key_for_plugin_package("avlite-controller-joystick") == "control"
    assert layer_key_for_plugin_package("avlite_bridge_carla") == "execution"
    assert layer_key_for_plugin_package("p100_future") == "perception"
    assert layer_key_for_plugin_package("sample_avlite_plugin") is None


def test_plugin_package_from_logger():
    assert (
        plugin_package_from_logger("avlite.plugins.avlite_bridge_carla.carla_bridge")
        == "avlite_bridge_carla"
    )
    assert plugin_package_from_logger("avlite.c50_apps.c52_factory") is None


def test_plugin_module_from_logger():
    assert (
        plugin_module_from_logger(
            "avlite.plugins.avlite_controller_joystick.p30_joystick_controller"
        )
        == "p30_joystick_controller"
    )
    assert (
        plugin_module_from_logger(
            "avlite.plugins.p10_perception_MO_prediction.p10_perception.prediction_utils"
        )
        == "p10_perception"
    )
    assert plugin_module_from_logger("avlite.plugins.avlite_bridge_carla") is None


def test_layer_key_for_plugin_log_record_module_first():
    name = "avlite.plugins.avlite_controller_joystick.p30_joystick_controller"
    assert layer_key_for_plugin_log_record(name) == "control"


def test_layer_key_for_plugin_log_record_package_fallback():
    name = "avlite.plugins.avlite_bridge_carla.carla_bridge"
    assert layer_key_for_plugin_log_record(name) == "execution"
    assert layer_key_for_plugin_log_record("avlite.plugins.sample_avlite_plugin.test_plugin") is None


def test_should_show_log_core_layer():
    name = "avlite.c30_control.c32_control_strategy"
    assert LogView.should_show_log(name, _filters(p57_show_control_logs=True))
    assert not LogView.should_show_log(name, _filters(p57_show_control_logs=False))
    assert not LogView.should_show_log(name, _filters(p57_show_core_logs=False, p57_show_control_logs=True))


def test_should_show_log_plugins_master_toggle():
    name = "avlite.plugins.avlite_controller_joystick.p30_joystick_controller"
    assert LogView.should_show_log(name, _filters(p57_show_plugins_logs=True))
    assert not LogView.should_show_log(name, _filters(p57_show_plugins_logs=False))


def test_should_show_log_pnx_routes_to_layer():
    name = "avlite.plugins.avlite_controller_joystick.p30_joystick_controller"
    assert LogView.should_show_log(name, _filters(p57_show_plugins_logs=True, p57_show_control_logs=True))
    assert not LogView.should_show_log(name, _filters(p57_show_plugins_logs=True, p57_show_control_logs=False))


def test_should_show_log_community_plugin_bucket():
    name = "avlite.plugins.sample_avlite_plugin.handler"
    assert LogView.should_show_log(name, _filters(p57_show_plugins_logs=True))
    assert not LogView.should_show_log(name, _filters(p57_show_plugins_logs=False))


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
