"""Tests for capability helpers in c51_capabilities."""

from __future__ import annotations

from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c20_planning.c26_local_path_planners import ReferencePathPlanner
from avlite.c20_planning.c27_local_behavioral_and_velocity_planners import (
    CruiseBehavioralPlanner,
    VelocityLocalPlanner,
)
from avlite.c20_planning.c28_local_lattice_planners import GreedyLatticePlanner
from avlite.c50_common.c51_capabilities import (
    AnyOf,
    MayUse,
    StackCapability,
    is_any_of,
    is_may_use,
    is_requirement_wrapper,
    required_stack_capabilities,
    satisfies_requirements,
    used_stack_capabilities,
)
from avlite.c50_common.c54_trajectory_tracker import TrajectoryTracker


class _Mod:
    def __init__(self, reqs):
        self.stack_requirements = reqs


def test_required_stack_capabilities_flattens_any_of():
    mods = [
        _Mod({AnyOf(StackCapability.GLOBAL_PLAN, StackCapability.LOCAL_PLAN), StackCapability.LOCALIZATION}),
        _Mod({StackCapability.DETECTION}),
        None,
    ]
    needed = required_stack_capabilities(mods)
    assert StackCapability.GLOBAL_PLAN in needed
    assert StackCapability.LOCAL_PLAN in needed
    assert StackCapability.LOCALIZATION in needed
    assert StackCapability.DETECTION in needed


def test_may_use_never_blocks_satisfies_requirements():
    reqs = {
        StackCapability.LOCALIZATION,
        MayUse(StackCapability.DETECTION, StackCapability.TRACKING),
    }
    assert satisfies_requirements(reqs, {StackCapability.LOCALIZATION})
    assert satisfies_requirements(reqs, {StackCapability.LOCALIZATION, StackCapability.DETECTION})
    assert not satisfies_requirements(reqs, set())


def test_required_stack_capabilities_ignores_may_use():
    mods = [
        _Mod({
            StackCapability.LOCALIZATION,
            MayUse(StackCapability.DETECTION, StackCapability.TRACKING),
        }),
    ]
    needed = required_stack_capabilities(mods)
    assert StackCapability.LOCALIZATION in needed
    assert StackCapability.DETECTION not in needed
    assert StackCapability.TRACKING not in needed


def test_used_stack_capabilities_includes_may_use():
    mods = [
        _Mod({
            StackCapability.LOCALIZATION,
            MayUse(StackCapability.DETECTION, StackCapability.PREDICTION),
        }),
        _Mod({AnyOf(StackCapability.GLOBAL_PLAN, StackCapability.LOCAL_PLAN)}),
    ]
    used = used_stack_capabilities(mods)
    assert StackCapability.LOCALIZATION in used
    assert StackCapability.DETECTION in used
    assert StackCapability.PREDICTION in used
    assert StackCapability.GLOBAL_PLAN in used
    assert StackCapability.LOCAL_PLAN in used
    assert StackCapability.TRACKING not in used


def test_concrete_local_planners_declare_contracts():
    pm = PerceptionModel(ego_vehicle=EgoState())
    path = [(0.0, 0.0), (10.0, 0.0)]
    vel = [5.0, 5.0]
    tj = TrajectoryTracker(path=path, velocity=vel)
    plan = GlobalPlan(start_point=path[0], goal_point=path[-1], path=path, velocity=vel, trajectory=tj)

    cruise = CruiseBehavioralPlanner()
    assert cruise.world_requirements == set()
    assert cruise.stack_requirements == set()
    assert cruise.stack_capabilities == set()

    ref = ReferencePathPlanner(plan, pm)
    assert StackCapability.GLOBAL_PLAN in ref.stack_requirements
    assert StackCapability.LOCALIZATION in ref.stack_requirements
    assert ref.stack_capabilities == {StackCapability.LOCAL_PLAN}

    vel_pl = VelocityLocalPlanner(plan, pm)
    assert MayUse(StackCapability.DETECTION, StackCapability.PREDICTION) in vel_pl.stack_requirements
    assert vel_pl.stack_capabilities == {StackCapability.LOCAL_PLAN}

    greedy = GreedyLatticePlanner(plan, pm)
    assert MayUse(StackCapability.DETECTION, StackCapability.PREDICTION) in greedy.stack_requirements
    assert greedy.stack_capabilities == {StackCapability.LOCAL_PLAN}


def test_concrete_perception_modules_declare_contracts():
    from avlite.c10_perception.c15_perception_algs import (
        ConstantVelocityPrediction,
        FastBEVLidarDetection,
        KalmanTracker,
    )
    from avlite.c10_perception.c16_localization_algs import LidarLocalization

    det = FastBEVLidarDetection()
    assert det.stack_requirements == set()
    assert det.stack_capabilities == {StackCapability.DETECTION}

    trk = KalmanTracker()
    assert StackCapability.DETECTION in trk.stack_requirements
    assert trk.stack_capabilities == {StackCapability.TRACKING}

    pred = ConstantVelocityPrediction()
    assert MayUse(StackCapability.DETECTION, StackCapability.TRACKING) in pred.stack_requirements
    assert pred.stack_capabilities == {StackCapability.PREDICTION}

    loc = LidarLocalization(PerceptionModel(ego_vehicle=EgoState()))
    assert loc.stack_requirements == set()
    assert loc.stack_capabilities == {StackCapability.LOCALIZATION}


def test_perception_pipeline_stack_capabilities_follow_stages():
    from avlite.c10_perception import c15_perception_algs  # noqa: F401 — register stages
    from avlite.c10_perception.c12_perception_strategy import PerceptionPipeline
    from avlite.c10_perception.c19_settings import PerceptionSettingsSchema

    pm = PerceptionModel(ego_vehicle=EgoState())

    empty = PerceptionPipeline(pm, PerceptionSettingsSchema(
        c12_detection_strategy="",
        c12_tracking_strategy="",
        c12_prediction_strategy="",
    ))
    assert empty.stack_capabilities == set()
    assert empty.stack_requirements == set()
    assert StackCapability.PREDICTION not in empty.stack_capabilities

    # Empty detect/track + predictor: soft MayUse only (no hard GT reqs).
    gt_plus_pred = PerceptionPipeline(pm, PerceptionSettingsSchema(
        c12_detection_strategy="",
        c12_tracking_strategy="",
        c12_prediction_strategy="ConstantVelocityPrediction",
    ))
    assert gt_plus_pred.stack_capabilities == {StackCapability.PREDICTION}
    assert StackCapability.DETECTION not in gt_plus_pred.stack_requirements
    assert StackCapability.TRACKING not in gt_plus_pred.stack_requirements
    assert MayUse(StackCapability.DETECTION, StackCapability.TRACKING) in gt_plus_pred.stack_requirements

    with_pred = PerceptionPipeline(pm, PerceptionSettingsSchema(
        c12_detection_strategy="FastBEVLidarDetection",
        c12_tracking_strategy="KalmanTracker",
        c12_prediction_strategy="ConstantVelocityPrediction",
    ))
    assert with_pred.stack_capabilities == {
        StackCapability.DETECTION,
        StackCapability.TRACKING,
        StackCapability.PREDICTION,
    }
    # Tracker hard-requires DETECTION; predictor MayUse shrinks to TRACKING only.
    assert MayUse(StackCapability.TRACKING) in with_pred.stack_requirements
    assert not any(
        is_may_use(r) and StackCapability.DETECTION in r.capabilities
        for r in with_pred.stack_requirements
    )


def test_reload_safe_wrapper_helpers_and_eq():
    """is_may_use / __eq__ key off type name, not class identity."""
    # Local classes with the same names simulate post-importlib.reload identities.
    class AnyOf:
        def __init__(self, *caps):
            self.capabilities = frozenset(caps)

    class MayUse:
        def __init__(self, *caps):
            self.capabilities = frozenset(caps)

    stale_may = MayUse(StackCapability.DETECTION, StackCapability.TRACKING)
    stale_any = AnyOf(StackCapability.GLOBAL_PLAN, StackCapability.LOCAL_PLAN)
    fresh_may = __import__(
        "avlite.c50_common.c51_capabilities", fromlist=["MayUse"]
    ).MayUse(StackCapability.DETECTION, StackCapability.TRACKING)
    fresh_any = __import__(
        "avlite.c50_common.c51_capabilities", fromlist=["AnyOf"]
    ).AnyOf(StackCapability.GLOBAL_PLAN, StackCapability.LOCAL_PLAN)

    assert type(stale_may) is not type(fresh_may)
    assert is_may_use(stale_may)
    assert is_may_use(fresh_may)
    assert is_any_of(stale_any)
    assert is_any_of(fresh_any)
    assert is_requirement_wrapper(stale_may)
    assert stale_may == fresh_may
    assert fresh_may == stale_may
    assert stale_any == fresh_any
    assert satisfies_requirements({stale_may, StackCapability.LOCALIZATION}, {StackCapability.LOCALIZATION})
    assert StackCapability.DETECTION in used_stack_capabilities([_Mod({stale_may})])
    assert StackCapability.DETECTION not in required_stack_capabilities([_Mod({stale_may})])


def test_pack_requirement_rows_survives_mayuse_reload():
    """Visualizer packing must not crash when MayUse class identity diverged."""
    import tkinter as tk
    from tkinter import ttk

    from avlite.c50_common.c51_capabilities import MayUse as RealMayUse
    from avlite.plugins.p60_visualizer_tk import p65_ui_lib as ui

    class MayUse:
        def __init__(self, *caps):
            self.capabilities = frozenset(caps)

    reqs = {
        StackCapability.GLOBAL_PLAN,
        StackCapability.LOCALIZATION,
        MayUse(StackCapability.DETECTION, StackCapability.PREDICTION),
    }
    soft = next(r for r in reqs if is_may_use(r))
    assert not isinstance(soft, RealMayUse)

    root = tk.Tk()
    root.withdraw()
    try:
        frame = ttk.Frame(root)
        ui._pack_requirement_rows(frame, reqs, set())
    finally:
        root.destroy()


def test_live_strategy_from_exec_matches_by_name_across_reload():
    import importlib

    from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
    from avlite.c10_perception import c12_perception_strategy as c12
    from avlite.c10_perception import c15_perception_algs as c15
    from avlite.c10_perception.c19_settings import PerceptionSettingsSchema
    from avlite.plugins.p60_visualizer_tk.p65_ui_lib import _live_strategy_from_exec

    pm = PerceptionModel(ego_vehicle=EgoState())
    pipe = c12.PerceptionPipeline(pm, PerceptionSettingsSchema(
        c12_detection_strategy="",
        c12_tracking_strategy="KalmanTracker",
        c12_prediction_strategy="ConstantVelocityPrediction",
    ))
    assert pipe.stack_capabilities == {StackCapability.TRACKING, StackCapability.PREDICTION}
    assert StackCapability.DETECTION in pipe.stack_requirements

    class _Exec:
        perception = pipe
        localization = mapping = global_planner = local_planner = controller = world = None

    # Reload strategy modules only (not c51) so StackCapability identity stays stable.
    importlib.reload(c12)
    importlib.reload(c15)
    new_cls = c12.PerceptionPipeline
    assert type(pipe) is not new_cls
    assert _live_strategy_from_exec(_Exec(), new_cls) is pipe

    assert not isinstance(pipe, new_cls)
    rebuilt = new_cls(pm, PerceptionSettingsSchema(
        c12_detection_strategy="",
        c12_tracking_strategy="KalmanTracker",
        c12_prediction_strategy="ConstantVelocityPrediction",
    ))
    assert rebuilt.stack_capabilities == {StackCapability.TRACKING, StackCapability.PREDICTION}
    assert MayUse(StackCapability.TRACKING) in rebuilt.stack_requirements


def test_leaf_contracts_readable_without_init():
    from avlite.c10_perception.c15_perception_algs import (
        ConstantVelocityPrediction,
        FastBEVLidarDetection,
        KalmanTracker,
    )
    from avlite.c20_planning.c24_global_hdmap_planners import HDMapGlobalPlanner
    from avlite.c20_planning.c25_global_race_planners import (
        GlobalCenterlineRacePlanner,
        GlobalRacePlanner,
    )

    assert FastBEVLidarDetection.stack_capabilities == frozenset({StackCapability.DETECTION})
    assert StackCapability.DETECTION in KalmanTracker.stack_requirements
    assert ConstantVelocityPrediction.stack_capabilities == frozenset({StackCapability.PREDICTION})
    assert MayUse(StackCapability.DETECTION, StackCapability.PREDICTION) in (
        VelocityLocalPlanner.stack_requirements
    )
    assert GreedyLatticePlanner.stack_capabilities == frozenset({StackCapability.LOCAL_PLAN})

    for cls in (HDMapGlobalPlanner, GlobalCenterlineRacePlanner, GlobalRacePlanner):
        assert StackCapability.LOCALIZATION in cls.stack_requirements
        assert cls.stack_capabilities == frozenset({StackCapability.GLOBAL_PLAN})
        assert cls.world_requirements == frozenset()
    assert StackCapability.MAP_HD in HDMapGlobalPlanner.stack_requirements
    assert StackCapability.MAP_RACE_TRACK in GlobalCenterlineRacePlanner.stack_requirements
    assert StackCapability.MAP_RACE_TRACK in GlobalRacePlanner.stack_requirements

    from avlite.c10_perception.c11_perception_model import RaceMap
    from avlite.c10_perception.c14_mapping_strategy import MapReader, MappingStrategy
    import numpy as np

    assert MapReader.__name__ in MappingStrategy.registry
    assert MappingStrategy.stack_capabilities == frozenset()
    race_map = RaceMap(
        source_path="synthetic",
        left_bound=np.array([[0.0, 1.0], [10.0, 1.0]]),
        right_bound=np.array([[0.0, -1.0], [10.0, -1.0]]),
    )
    assert MapReader(race_map).stack_capabilities == frozenset({StackCapability.MAP_RACE_TRACK})
    assert not hasattr(MapReader, "from_path")


def test_strategy_capability_defaults_are_not_abstract():
    """Capability attrs are soft defaults; only algorithm methods stay abstract."""
    from avlite.c10_perception.c12_perception_strategy import (
        DetectionStrategy,
        PredictionStrategy,
    )
    from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
    from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy

    assert "world_requirements" not in PredictionStrategy.__abstractmethods__
    assert "stack_requirements" not in PredictionStrategy.__abstractmethods__
    assert "stack_capabilities" not in PredictionStrategy.__abstractmethods__
    assert PredictionStrategy.__abstractmethods__ == frozenset({"predict"})

    assert "world_requirements" not in LocalPlanningStrategy.__abstractmethods__
    assert "stack_requirements" not in LocalPlanningStrategy.__abstractmethods__
    assert "stack_capabilities" not in LocalPlanningStrategy.__abstractmethods__
    assert LocalPlanningStrategy.__abstractmethods__ == frozenset({"replan"})

    class _MinimalDetection(DetectionStrategy):
        def detect(
            self,
            perception_model=None,
            sensors=None,
            rgb_img=None,
            depth_img=None,
            lidar_data=None,
        ):
            return perception_model

    assert not _MinimalDetection.__abstractmethods__
    assert _MinimalDetection.stack_capabilities == frozenset({StackCapability.DETECTION})
    _MinimalDetection()  # instantiable without declaring capability attrs

    class _MinimalGlobal(GlobalPlannerStrategy):
        def plan(self, perception_model=None, sensors=None):
            return GlobalPlan()

    assert not _MinimalGlobal.__abstractmethods__
    assert _MinimalGlobal.stack_capabilities == frozenset({StackCapability.GLOBAL_PLAN})
    _MinimalGlobal()


def test_control_strategy_defaults_unchanged():
    from avlite.c30_control.c32_control_strategy import ControlStrategy

    assert ControlStrategy.stack_capabilities == frozenset({StackCapability.CONTROL})
    assert "control" in ControlStrategy.__abstractmethods__
    assert "reset" in ControlStrategy.__abstractmethods__


def test_pipeline_contract_via_class_is_property_descriptor():
    from avlite.c10_perception.c12_perception_strategy import PerceptionPipeline

    assert isinstance(PerceptionPipeline.stack_capabilities, property)
