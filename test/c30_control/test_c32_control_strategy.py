from avlite.c20_planning.c21_planning_model import GlobalPlan, LocalPlan
from avlite.c30_control.c31_control_model import ControlCommand
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c50_common.c51_capabilities import AnyOf, StackCapability, satisfies_requirements
from avlite.c50_common.c53_trajectory_tracker import TrajectoryTracker


class _StubController(ControlStrategy, abstract=True):
    def control(self, ego, plan=None, control_dt=None) -> ControlCommand:
        return ControlCommand()

    def reset(self):
        pass


def test_global_plan_as_trajectory():
    tj = TrajectoryTracker(path=[(0.0, 0.0), (1.0, 0.0)], velocity=[1.0, 1.0])
    plan = GlobalPlan(path=[(0.0, 0.0), (1.0, 0.0)], velocity=[1.0, 1.0], trajectory=tj)
    assert plan.as_trajectory() is tj


def test_control_stack_requirements_any_of():
    ctrl = _StubController()
    reqs = ctrl.stack_requirements
    assert AnyOf(StackCapability.GLOBAL_PLAN, StackCapability.LOCAL_PLAN) in reqs
    assert satisfies_requirements(reqs, {StackCapability.GLOBAL_PLAN})
    assert satisfies_requirements(reqs, {StackCapability.LOCAL_PLAN})
    assert not satisfies_requirements(reqs, {StackCapability.CONTROL})


def test_set_plan_global_plan():
    tj = TrajectoryTracker(path=[(0.0, 0.0), (2.0, 0.0)], velocity=[1.0, 1.0])
    plan = GlobalPlan(path=[(0.0, 0.0), (2.0, 0.0)], velocity=[1.0, 1.0], trajectory=tj)
    ctrl = _StubController()
    ctrl.set_plan(plan)
    assert ctrl.tj is tj


def test_set_plan_local_plan():
    tj = TrajectoryTracker(path=[(0.0, 0.0), (4.0, 0.0)], velocity=[1.0, 1.0])
    plan = LocalPlan.from_trajectory(tj)
    ctrl = _StubController()
    ctrl.set_plan(plan)
    assert ctrl.tj is tj
