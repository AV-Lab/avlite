"""Reload-failed dialog text for an unmet stack capability."""

from avlite.c40_execution.c42_execution_strategy import UnmetRequirements
from avlite.c50_common.c51_capabilities import AnyOf, MayUse, StackCapability
from avlite.plugins.p60_visualizer_tk.p65_ui_lib import stack_build_failure_text


def test_stack_build_failure_dialog_lists_names():
    err = UnmetRequirements(
        "global planner GlobalCenterlineRacePlanner",
        {StackCapability.LOCALIZATION, StackCapability.MAP_RACE_TRACK},
        {StackCapability.LOCALIZATION, StackCapability.MAP_HD},
    )
    assert stack_build_failure_text(err) == (
        "global planner GlobalCenterlineRacePlanner\n"
        "\n"
        "Missing\n"
        "  MAP_RACE_TRACK\n"
        "\n"
        "Required\n"
        "  LOCALIZATION\n"
        "  MAP_RACE_TRACK\n"
        "\n"
        "Available\n"
        "  LOCALIZATION\n"
        "  MAP_HD"
    )
    assert "frozenset" not in stack_build_failure_text(err)


def test_stack_build_failure_dialog_any_of():
    err = UnmetRequirements(
        "local planner Lattice",
        {AnyOf(StackCapability.PREDICTION_TRAJECTORY, StackCapability.PREDICTION_GP)},
        set(),
    )
    assert stack_build_failure_text(err) == (
        "local planner Lattice\n"
        "\n"
        "Missing\n"
        "  any of\n"
        "    PREDICTION_GP\n"
        "    PREDICTION_TRAJECTORY\n"
        "\n"
        "Required\n"
        "  any of\n"
        "    PREDICTION_GP\n"
        "    PREDICTION_TRAJECTORY\n"
        "\n"
        "Available\n"
        "  (none)"
    )
    assert "frozenset" not in stack_build_failure_text(err)


def test_stack_build_failure_dialog_optional():
    err = UnmetRequirements(
        "controller Stanley",
        {
            StackCapability.LOCALIZATION,
            MayUse(StackCapability.DETECTION),
        },
        {StackCapability.LOCALIZATION},
    )
    text = stack_build_failure_text(err)
    assert "Missing\n  (none)" in text
    assert "Required\n  LOCALIZATION\n  optional\n    DETECTION" in text


def test_stack_build_failure_dialog_other_errors():
    assert stack_build_failure_text(ValueError("nope")) == "Failed to rebuild the stack.\n\nnope"
