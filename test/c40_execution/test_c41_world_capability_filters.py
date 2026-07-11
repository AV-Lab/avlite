"""Tests for Bridge Setting world / stack capability enablement filters."""

from __future__ import annotations

from avlite.c40_execution.c41_world_bridge import (
    is_world_capability_enabled,
    is_world_stack_capability_enabled,
)
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c50_common.c51_capabilities import StackCapability, WorldCapability


def test_world_capability_none_means_all_enabled():
    ExecutionSettings.c41_world_capabilities = None
    assert is_world_capability_enabled(WorldCapability.LIDAR_2D)
    assert is_world_capability_enabled(WorldCapability.GNSS)


def test_world_capability_explicit_list():
    ExecutionSettings.c41_world_capabilities = ["LIDAR_2D"]
    assert is_world_capability_enabled(WorldCapability.LIDAR_2D)
    assert not is_world_capability_enabled(WorldCapability.GNSS)
    ExecutionSettings.c41_world_capabilities = None


def test_world_stack_capability_none_means_all_enabled():
    ExecutionSettings.c41_world_stack_capabilities = None
    assert is_world_stack_capability_enabled(StackCapability.DETECTION)
    assert is_world_stack_capability_enabled(StackCapability.LOCALIZATION)


def test_world_stack_capability_explicit_list():
    ExecutionSettings.c41_world_stack_capabilities = ["LOCALIZATION"]
    assert is_world_stack_capability_enabled(StackCapability.LOCALIZATION)
    assert not is_world_stack_capability_enabled(StackCapability.DETECTION)
    ExecutionSettings.c41_world_stack_capabilities = None


def test_settings_schema_has_split_fields_not_c41_provided():
    assert hasattr(ExecutionSettings, "c41_world_capabilities")
    assert hasattr(ExecutionSettings, "c41_world_stack_capabilities")
    assert not hasattr(ExecutionSettings, "c41_provided")
