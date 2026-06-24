"""Tests for plugin log routing helpers."""

from avlite.c50_visualization.c55_log_view import should_show_log
from avlite.c60_common.c60_plugins import (
    layer_key_for_plugin_log_record,
    layer_key_for_plugin_package,
    plugin_module_from_logger,
    plugin_package_from_logger,
)


def _filters(**overrides: bool) -> dict[str, bool]:
    base = {
        "show_perceive_logs": True,
        "show_plan_logs": True,
        "show_control_logs": True,
        "show_execute_logs": True,
        "show_vis_logs": True,
        "show_common_logs": True,
        "show_core_logs": True,
        "show_plugins_logs": True,
        "disable_log": False,
        "log_to_file": False,
    }
    base.update(overrides)
    return base


def test_layer_key_built_in_plugins():
    assert layer_key_for_plugin_package("p10_perception_MO_prediction") == "perception"
    assert layer_key_for_plugin_package("p40_bridge_carla") == "execution"
    assert layer_key_for_plugin_package("p100_future") == "perception"
    assert layer_key_for_plugin_package("sample_avlite_plugin") is None


def test_plugin_package_from_logger():
    assert (
        plugin_package_from_logger("avlite.plugins.p40_bridge_carla.carla_bridge")
        == "p40_bridge_carla"
    )
    assert plugin_package_from_logger("avlite.c40_execution.c43_factory") is None


def test_plugin_module_from_logger():
    assert (
        plugin_module_from_logger(
            "avlite.plugins.p30_controller_joystick.p31_joystick_controller"
        )
        == "p31_joystick_controller"
    )
    assert (
        plugin_module_from_logger(
            "avlite.plugins.p10_perception_MO_prediction.p10_perception.prediction_utils"
        )
        == "p10_perception"
    )
    assert plugin_module_from_logger("avlite.plugins.p40_bridge_carla") is None


def test_layer_key_for_plugin_log_record_module_first():
    name = "avlite.plugins.p30_controller_joystick.p31_joystick_controller"
    assert layer_key_for_plugin_log_record(name) == "control"


def test_layer_key_for_plugin_log_record_package_fallback():
    name = "avlite.plugins.p40_bridge_carla.carla_bridge"
    assert layer_key_for_plugin_log_record(name) == "execution"
    assert layer_key_for_plugin_log_record("avlite.plugins.sample_avlite_plugin.test_plugin") is None


def test_should_show_log_core_layer():
    name = "avlite.c30_control.c32_control_strategy"
    assert should_show_log(name, _filters(show_control_logs=True))
    assert not should_show_log(name, _filters(show_control_logs=False))
    assert not should_show_log(name, _filters(show_core_logs=False, show_control_logs=True))


def test_should_show_log_plugins_master_toggle():
    name = "avlite.plugins.p30_controller_joystick.p31_joystick_controller"
    assert should_show_log(name, _filters(show_plugins_logs=True))
    assert not should_show_log(name, _filters(show_plugins_logs=False))


def test_should_show_log_pnx_routes_to_layer():
    name = "avlite.plugins.p30_controller_joystick.p31_joystick_controller"
    assert should_show_log(name, _filters(show_plugins_logs=True, show_control_logs=True))
    assert not should_show_log(name, _filters(show_plugins_logs=True, show_control_logs=False))


def test_should_show_log_community_plugin_bucket():
    name = "avlite.plugins.sample_avlite_plugin.handler"
    assert should_show_log(name, _filters(show_plugins_logs=True))
    assert not should_show_log(name, _filters(show_plugins_logs=False))
