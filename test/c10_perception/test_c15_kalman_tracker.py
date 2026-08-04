"""Regression tests for KalmanTracker association and track lifecycle."""

from avlite.c10_perception.c11_perception_model import AgentState, EgoState, PerceptionModel
from avlite.c10_perception.c15_perception_algs import KalmanTracker


def _pm_with_detections(*xy: tuple[float, float]) -> PerceptionModel:
    agents = [
        AgentState(x=x, y=y, theta=0.0, velocity=0.0, agent_id=i, length=4.0, width=2.0)
        for i, (x, y) in enumerate(xy)
    ]
    return PerceptionModel(
        ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=0.0),
        agent_vehicles=agents,
    )


class TestKalmanTrackerLifecycle:
    def test_persists_id_and_estimates_velocity_across_frames(self):
        tracker = KalmanTracker(dt=1.0, gate_distance=2.0, max_missed=2, min_speed=0.1)

        out1 = tracker.track(_pm_with_detections((0.0, 0.0)))
        assert len(out1.agent_vehicles) == 1
        track_id = out1.agent_vehicles[0].agent_id
        assert out1.agent_vehicles[0].velocity == 0.0

        # Same detection id is reassigned each frame; tracker must keep track_id.
        out2 = tracker.track(_pm_with_detections((1.0, 0.0)))
        assert len(out2.agent_vehicles) == 1
        assert out2.agent_vehicles[0].agent_id == track_id
        assert out2.agent_vehicles[0].velocity > 0.5

    def test_far_detection_spawns_new_track(self):
        tracker = KalmanTracker(dt=1.0, gate_distance=2.0, max_missed=2, min_speed=0.1)

        out1 = tracker.track(_pm_with_detections((0.0, 0.0)))
        first_id = out1.agent_vehicles[0].agent_id

        # Outside gate → new track rather than hijacking the existing one.
        out2 = tracker.track(_pm_with_detections((10.0, 0.0)))
        assert len(out2.agent_vehicles) == 1
        assert out2.agent_vehicles[0].agent_id != first_id

    def test_missed_tracks_are_pruned_after_max_missed(self):
        tracker = KalmanTracker(dt=1.0, gate_distance=2.0, max_missed=1, min_speed=0.1)

        tracker.track(_pm_with_detections((0.0, 0.0)))
        # One unmatched frame increments missed; still kept internally but not emitted.
        empty = tracker.track(
            PerceptionModel(
                ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=0.0),
                agent_vehicles=[],
            )
        )
        assert empty.agent_vehicles == []
        assert len(tracker._tracks) == 1

        # Second miss exceeds max_missed and drops the track.
        tracker.track(
            PerceptionModel(
                ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=0.0),
                agent_vehicles=[],
            )
        )
        assert tracker._tracks == []

    def test_requires_perception_model(self):
        tracker = KalmanTracker()
        try:
            tracker.track(None)
        except ValueError as exc:
            assert "perception_model" in str(exc)
        else:
            raise AssertionError("expected ValueError when perception_model is None")
