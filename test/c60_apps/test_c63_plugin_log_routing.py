"""Tests for plugin log routing helpers in c63_plugins."""

from avlite.c60_apps.c63_plugins import (
    layer_key_for_plugin_log_record,
    layer_key_for_plugin_package,
    plugin_module_from_logger,
    plugin_package_from_logger,
)


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
    assert plugin_package_from_logger("avlite.c60_apps.c62_factory") is None


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
