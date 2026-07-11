"""Tests for ExecutionStrategy stack capability validation."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from avlite.c10_perception.c11_perception_model import EgoState, HDMap, PerceptionModel, RaceMap
from avlite.c10_perception.c14_mapping_strategy import MapReader
from avlite.c20_planning.c25_global_race_planners import GlobalCenterlineRacePlanner
from avlite.c40_execution.c42_execution_strategy import ExecutionStrategy
from avlite.c40_execution.c46_basic_sim import BasicSim
from avlite.c50_common.c51_capabilities import StackCapability


class _StubExecuter(ExecutionStrategy):
    def step(self, **kwargs):
        return None


def _race_map() -> RaceMap:
    left = np.array([[0.0, 1.0], [10.0, 1.0]])
    right = np.array([[0.0, -1.0], [10.0, -1.0]])
    return RaceMap(source_path="synthetic", left_bound=left, right_bound=right)


def test_available_stack_capabilities_includes_map_from_mapping():
    ego = EgoState()
    pm = PerceptionModel(ego_vehicle=ego)
    race_map = _race_map()
    world = BasicSim(ego_state=ego, pm=pm, map=race_map)
    mapping = MapReader(race_map)
    exec_ = _StubExecuter(
        perception_model=pm,
        perception=None,
        global_planner=None,
        local_planner=None,
        controller=None,
        world=world,
        mapping=mapping,
    )
    assert StackCapability.MAP_RACE_TRACK in exec_.available_stack_capabilities()
    assert StackCapability.MAP_RACE_TRACK not in world.stack_capabilities
    assert StackCapability.MAP_HD not in exec_.available_stack_capabilities()


def test_validate_stack_raises_unmet_map():
    ego = EgoState()
    pm = PerceptionModel(ego_vehicle=ego)
    world = BasicSim(ego_state=ego, pm=pm, map=None)
    gp = GlobalCenterlineRacePlanner(_race_map())
    with pytest.raises(ValueError, match="stack_requirements not satisfied"):
        _StubExecuter(
            perception_model=pm,
            perception=None,
            global_planner=gp,
            local_planner=None,
            controller=None,
            world=world,
            mapping=None,
        )


def test_validate_stack_raises_typed_map_mismatch(minimal_opendrive_path):
    """Race planner needs MAP_RACE_TRACK; MapReader(HDMap) only provides MAP_HD."""
    ego = EgoState()
    pm = PerceptionModel(ego_vehicle=ego)
    world = BasicSim(ego_state=ego, pm=pm, map=None)
    hd_map = HDMap.from_path(minimal_opendrive_path)
    gp = GlobalCenterlineRacePlanner(_race_map())
    with pytest.raises(ValueError, match="stack_requirements not satisfied"):
        _StubExecuter(
            perception_model=pm,
            perception=None,
            global_planner=gp,
            local_planner=None,
            controller=None,
            world=world,
            mapping=MapReader(hd_map),
        )


def test_validate_stack_warns_unmet_world_control():
    """BasicSim requires CONTROL; warn when no controller is assembled."""
    ego = EgoState()
    pm = PerceptionModel(ego_vehicle=ego)
    world = BasicSim(ego_state=ego, pm=pm, map=None)
    with patch("avlite.c40_execution.c42_execution_strategy.log.warning") as warn:
        _StubExecuter(
            perception_model=pm,
            perception=None,
            global_planner=None,
            local_planner=None,
            controller=None,
            world=world,
        )
    assert any(
        "world bridge" in str(c) and "CONTROL" in str(c) and "stack_requirements not satisfied" in str(c)
        for c in warn.call_args_list
    )


def test_validate_stack_warns_duplicate_localization():
    """World GT LOCALIZATION + a localization module both advertise the same cap."""
    from avlite.c10_perception.c13_localization_strategy import LocalizationStrategy

    class _Loc(LocalizationStrategy):
        stack_capabilities = frozenset({StackCapability.LOCALIZATION})

        def localize(self, perception_model=None, sensors=None):
            return perception_model

    ego = EgoState()
    pm = PerceptionModel(ego_vehicle=ego)
    world = BasicSim(ego_state=ego, pm=pm, map=None)
    with patch("avlite.c40_execution.c42_execution_strategy.log.warning") as warn:
        _StubExecuter(
            perception_model=pm,
            perception=None,
            global_planner=None,
            local_planner=None,
            controller=None,
            world=world,
            localization=_Loc(perception_model=pm),
        )
    assert any(
        "LOCALIZATION" in str(c) and "multiple sources" in str(c)
        for c in warn.call_args_list
    )
