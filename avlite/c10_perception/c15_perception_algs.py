import numpy as np

from avlite.c10_perception.c11_perception_model import AgentState, PerceptionModel, PredictionMode
from avlite.c10_perception.c12_perception_strategy import (
    DetectionStrategy,
    PredictionStrategy,
    TrackingStrategy,
)
from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c60_common.c61_capabilities import AnyOf, WorldCapability

import logging 

log = logging.getLogger(__name__)


class ConstantVelocityPrediction(PredictionStrategy):
    """Predict each agent's future positions assuming constant velocity.

    Writes results into ``pm.trajectories`` (shape ``[n_agents, n_steps, 2]``)
    and sets ``pm.prediction_mode = PredictionMode.TRAJECTORY``.
    """

    @property
    def requirements(self):
        return set()

    def predict(self, perception_model: PerceptionModel) -> PerceptionModel:
        agents = perception_model.agent_vehicles
        if not agents:
            perception_model.prediction_mode = PredictionMode.TRAJECTORY
            perception_model.trajectories = np.empty((0, 0, 2))
            return perception_model

        dt = perception_model.predict_delta_t
        horizon = PerceptionSettings.c15_prediction_horizon
        n_steps = max(1, int(round(horizon / dt)))

        trajectories = np.empty((len(agents), n_steps, 2))
        for i, agent in enumerate(agents):
            for t in range(n_steps):
                time = (t + 1) * dt
                trajectories[i, t, 0] = agent.x + agent.velocity * np.cos(agent.theta) * time
                trajectories[i, t, 1] = agent.y + agent.velocity * np.sin(agent.theta) * time

        perception_model.trajectories = trajectories
        perception_model.prediction_mode = PredictionMode.TRAJECTORY
        log.debug("Predicted trajectories for %d agents over %d steps", len(agents), n_steps)
        return perception_model


class _Track:
    """A single constant-velocity Kalman filter track.

    State vector ``[x, y, vx, vy]``; measurement ``[x, y]``.
    """

    def __init__(self, track_id: int, agent: AgentState, init_vel_var: float):
        self.track_id = track_id
        self.missed = 0
        self.agent = agent
        self.x = np.array([agent.x, agent.y, 0.0, 0.0])
        self.P = np.diag([1.0, 1.0, init_vel_var, init_vel_var])

    def predict(self, dt: float, q: float) -> None:
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        # Constant-acceleration process noise (discrete white-noise model).
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        Q = q * np.array([
            [dt4 / 4, 0, dt3 / 2, 0],
            [0, dt4 / 4, 0, dt3 / 2],
            [dt3 / 2, 0, dt2, 0],
            [0, dt3 / 2, 0, dt2],
        ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray, r: float) -> None:
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        R = r * np.eye(2)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P


class KalmanTracker(TrackingStrategy):
    """Constant-velocity Kalman filter tracker with greedy data association.

    Detection sub-strategies output per-frame boxes with ``velocity = 0`` and
    re-assign ``agent_id`` every frame.  This tracker links detections across
    frames into persistent tracks, estimating each agent's velocity and
    smoothing its position.  Unmatched detections spawn new tracks; tracks that
    go unmatched for ``max_missed`` frames are removed.
    """

    def __init__(
        self,
        dt: float = PerceptionSettings.c15_tracking_dt,
        process_noise: float = PerceptionSettings.c15_tracking_process_noise,
        measurement_noise: float = PerceptionSettings.c15_tracking_measurement_noise,
        init_velocity_var: float = PerceptionSettings.c15_tracking_init_velocity_var,
        gate_distance: float = PerceptionSettings.c15_tracking_gate_distance,
        max_missed: int = PerceptionSettings.c15_tracking_max_missed,
        min_speed: float = PerceptionSettings.c15_tracking_min_speed,
    ):
        self._dt = dt
        self._q = process_noise ** 2
        self._r = measurement_noise ** 2
        self._init_vel_var = init_velocity_var
        self._gate = gate_distance
        self._max_missed = max_missed
        self._min_speed = min_speed
        self._tracks: list[_Track] = []
        self._next_id = 0

    @property
    def requirements(self) -> set:
        return set()

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 0

    def track(self, perception_model: PerceptionModel) -> PerceptionModel:
        dt = perception_model.predict_delta_t or self._dt
        for trk in self._tracks:
            trk.predict(dt, self._q)

        detections = perception_model.agent_vehicles
        matches, unmatched = self._associate(detections)

        matched_tracks = {trk_idx for trk_idx, _ in matches}
        for trk_idx, det_idx in matches:
            trk = self._tracks[trk_idx]
            det = detections[det_idx]
            trk.update(np.array([det.x, det.y]), self._r)
            trk.agent = det
            trk.missed = 0

        for i, trk in enumerate(self._tracks):
            if i not in matched_tracks:
                trk.missed += 1
        self._tracks = [t for t in self._tracks if t.missed <= self._max_missed]

        for det_idx in unmatched:
            det = detections[det_idx]
            self._tracks.append(_Track(self._next_id, det, self._init_vel_var))
            self._next_id += 1

        perception_model.agent_vehicles = [
            self._to_agent(t) for t in self._tracks if t.missed == 0
        ]

        log.debug("Tracking updated: %d tracks, %d detections, %d matches", len(self._tracks), len(detections), len(matches))
        return perception_model

    def _associate(self, detections: list[AgentState]) -> tuple[list[tuple[int, int]], list[int]]:
        """Greedy nearest-neighbour association within the gating distance."""
        if not self._tracks or not detections:
            return [], list(range(len(detections)))

        track_xy = np.array([[t.x[0], t.x[1]] for t in self._tracks])
        det_xy = np.array([[d.x, d.y] for d in detections])
        cost = np.linalg.norm(track_xy[:, None, :] - det_xy[None, :, :], axis=2)

        matches: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
        for trk_idx, det_idx in order:
            if trk_idx in used_tracks or det_idx in used_dets:
                continue
            if cost[trk_idx, det_idx] > self._gate:
                break
            matches.append((int(trk_idx), int(det_idx)))
            used_tracks.add(trk_idx)
            used_dets.add(det_idx)

        unmatched = [i for i in range(len(detections)) if i not in used_dets]
        return matches, unmatched

    def _to_agent(self, trk: _Track) -> AgentState:
        x, y, vx, vy = trk.x
        speed = float(np.hypot(vx, vy))
        theta = float(np.arctan2(vy, vx)) if speed >= self._min_speed else trk.agent.theta
        return AgentState(
            x=float(x),
            y=float(y),
            theta=theta,
            length=trk.agent.length,
            width=trk.agent.width,
            velocity=speed,
            agent_id=trk.track_id,
        )


class FastBEVLidarDetection(DetectionStrategy):
    """LiDAR object detection via BEV segmentation and rotating-calipers MBR.

    Accepts both 2D scans ``(N, 2)`` and 3D point clouds ``(N, 3+)``.  For 3D
    input, points outside ``[z_min, z_max]`` are discarded before the BEV
    pipeline, removing the ground plane and rooftop returns.

    Algorithm (Khonji et al., e-Energy '17, §4):

    1. **Segmentation** – split the ordered scan on consecutive gap > ``mu``.
       O(n), exploiting angular scan order instead of an EMST.
    2. **Bounding rectangle** – minimum-area bounding rectangle via rotating
       calipers.  Extracts centre, heading, length and width per cluster.

    Parameters
    ----------
    z_min, z_max : float
        Z-band filter for 3D input [m].  Ignored for 2D input.
    delta_min, delta_max : float
        Accepted cluster bounding-box diagonal range [m].
    mu : float
        Max consecutive-point gap [m] before starting a new cluster.
    """

    def __init__(
        self,
        z_min: float = PerceptionSettings.c15_detection_z_min,
        z_max: float = PerceptionSettings.c15_detection_z_max,
        delta_min: float = PerceptionSettings.c15_detection_delta_min,
        delta_max: float = PerceptionSettings.c15_detection_delta_max,
        mu: float = PerceptionSettings.c15_detection_mu,
        min_length: float = PerceptionSettings.c15_detection_min_length,
        min_width: float = PerceptionSettings.c15_detection_min_width,
        default_length: float = PerceptionSettings.c15_detection_default_length,
        default_width: float = PerceptionSettings.c15_detection_default_width,
    ):
        self._z_min = z_min
        self._z_max = z_max
        self._delta_min = delta_min
        self._delta_max = delta_max
        self._mu = mu
        self._min_length = min_length
        self._min_width = min_width
        self._default_length = default_length
        self._default_width = default_width

    @property
    def requirements(self) -> set:
        return {AnyOf(WorldCapability.LIDAR_2D, WorldCapability.LIDAR_3D)}

    def detect(
        self,
        perception_model: PerceptionModel,
        rgb_img=None,
        depth_img=None,
        lidar_data=None,
    ) -> PerceptionModel:
        if lidar_data is None or len(lidar_data) == 0:
            perception_model.detection_clusters = None
            return perception_model
        pts = np.asarray(lidar_data, dtype=float)
        if pts.shape[1] >= 3:
            mask = (pts[:, 2] >= self._z_min) & (pts[:, 2] <= self._z_max)
            pts = pts[mask]
            if len(pts) == 0:
                perception_model.detection_clusters = None
                return perception_model
        pts_xy = pts[:, :2]
        ego = perception_model.ego_vehicle
        ego_xytheta = (ego.x, ego.y, ego.theta) if ego else (0.0, 0.0, 0.0)
        perception_model.agent_vehicles = []
        accepted: list[np.ndarray] = []
        for cluster in self._segment(pts_xy):
            agent = self._fit_rectangle(cluster, ego_xytheta)
            if agent is not None:
                perception_model.add_agent_vehicle(agent)
                accepted.append(cluster)
        perception_model.detection_clusters = np.vstack(accepted) if accepted else None
        log.debug("Detected %d clusters from %d points", len(accepted), len(pts))
        return perception_model

    # ------------------------------------------------------------------
    # Stage 1 – sequential scan segmentation (§4.1)
    # ------------------------------------------------------------------

    def _segment(self, pts: np.ndarray) -> list[np.ndarray]:
        """Split ordered scan on gap > mu; keep clusters within diagonal range."""
        if len(pts) == 0:
            return []
        clusters: list[np.ndarray] = []
        start = 0
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - pts[i - 1]) > self._mu:
                clusters.append(pts[start:i])
                start = i
        clusters.append(pts[start:])
        return [c for c in clusters if self._in_range(c)]

    def _in_range(self, cluster: np.ndarray) -> bool:
        diag = np.linalg.norm(cluster.max(axis=0) - cluster.min(axis=0))
        return self._delta_min <= diag <= self._delta_max

    # ------------------------------------------------------------------
    # Stage 2 – minimum bounding rectangle via rotating calipers (§4.2)
    # Pure numpy implementation (no shapely dep; robust with numpy >= 2.0)
    # ------------------------------------------------------------------

    def _fit_rectangle(self, cluster: np.ndarray, ego: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> AgentState | None:
        if len(cluster) < 2:
            return None
        hull = self._convex_hull(cluster)
        cx, cy, theta, length, width = self._min_bounding_rect(hull)
        # An edge-on / collinear cluster (only one face of the car is visible)
        # collapses a rectangle dimension into an invisible sliver. When the fitted
        # shape is degenerate, fall back to a default car-sized box aligned with ego
        # heading (track traffic moves roughly parallel to ego). The visible face is
        # the near side, so push the box centre away from ego along the ego axis by
        # length/2: the seen line becomes the car's back (obstacle ahead of ego) or
        # front (obstacle behind ego).
        if length < self._min_length or width < self._min_width:
            ego_x, ego_y, theta = ego[0], ego[1], ego[2]
            length = self._default_length
            width = self._default_width
            heading = np.array([np.cos(theta), np.sin(theta)])
            to_cluster = np.array([cx - ego_x, cy - ego_y])
            sign = 1.0 if float(np.dot(heading, to_cluster)) >= 0.0 else -1.0
            cx += sign * (length / 2.0) * heading[0]
            cy += sign * (length / 2.0) * heading[1]
        return AgentState(x=cx, y=cy, theta=theta, length=length, width=width)

    @staticmethod
    def _convex_hull(pts: np.ndarray) -> np.ndarray:
        """Andrew's monotone chain – O(n log n), returns CCW hull vertices."""
        pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
        if len(pts) <= 1:
            return pts

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower: list = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper: list = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return np.array(lower[:-1] + upper[:-1])

    @staticmethod
    def _min_bounding_rect(hull: np.ndarray) -> tuple[float, float, float, float, float]:
        """Rotating calipers MBR – returns (cx, cy, theta, length, width)."""
        n = len(hull)
        if n < 2:
            x, y = hull[0]
            return float(x), float(y), 0.0, 0.0, 0.0

        best_area = np.inf
        best = (0.0, 0.0, 0.0, 0.0, 0.0)
        for i in range(n):
            edge = hull[(i + 1) % n] - hull[i]
            angle = np.arctan2(edge[1], edge[0])
            c, s = np.cos(-angle), np.sin(-angle)
            rot = np.array([[c, -s], [s, c]])
            rotated = hull @ rot.T
            mn, mx = rotated.min(axis=0), rotated.max(axis=0)
            w, h = mx[0] - mn[0], mx[1] - mn[1]
            area = w * h
            if area < best_area:
                best_area = area
                # Centre in rotated frame → un-rotate
                cr = np.array([(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2])
                cx, cy = np.array([[c, s], [-s, c]]) @ cr
                if w >= h:
                    best = (float(cx), float(cy), float(angle), float(w), float(h))
                else:
                    best = (float(cx), float(cy), float(angle + np.pi / 2), float(h), float(w))
        return best



