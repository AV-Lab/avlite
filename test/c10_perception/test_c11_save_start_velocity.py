"""Save Start must snapshot velocity 0 so Reset matches a cold profile start."""

from avlite.c10_perception.c11_perception_model import EgoState


def test_save_start_snapshot_zeros_velocity_while_preserving_live_speed():
    """Mirrors ExecView.set_start: capture pose with v=0, keep live velocity."""
    ego = EgoState(x=10.0, y=20.0, theta=0.5, velocity=0.0)
    ego.velocity = 12.5

    live_v = ego.velocity
    ego.velocity = 0.0
    ego.set_start()
    ego.velocity = live_v

    assert ego.velocity == 12.5
    ego.x, ego.y, ego.theta, ego.velocity = 99.0, 99.0, 0.0, 0.0
    ego.reset()
    assert ego.x == 10.0
    assert ego.y == 20.0
    assert ego.theta == 0.5
    assert ego.velocity == 0.0
