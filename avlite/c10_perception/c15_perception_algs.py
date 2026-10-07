import numpy as np
from scipy.special import erf

from avlite.c10_perception.c11_perception_model import (
    AgentState,
    GMM,
    GP,
    MultiTrajectory,
    OccupancyFlow,
    PerceptionModel,
    SingleTrajectory,
    State,
)
from avlite.c10_perception.c12_perception_strategy import (
    DetectionStrategy,
    PredictionStrategy,
    TrackingStrategy,
)
from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c20_planning.c29_settings import PlanningSettings
from avlite.c50_common.c51_capabilities import AnyOf, MayUse, StackCapability, WorldCapability
from avlite.c50_common.c52_world_sensor_datatypes import Lidar, SensorFrame

import logging 

log = logging.getLogger(__name__)

class ConstantVelocityForecast:
    """Shared constant-velocity forecast. Not a strategy, so it is not in the dropdown."""

    # Position std along / across the heading, times the step time (m/s).
    along_sigma = 2.0
    cross_sigma = 0.8
    # Half-width of the heading fan for GMM and MultiTrajectory modes (rad).
    mode_heading_span = 0.35

    def horizon(self) -> tuple[float, int, np.ndarray]:
        """``dt``, step count, and the time of each step. Step ``k`` is at ``(k + 1) * dt``."""
        dt = PerceptionSettings.c11_predict_delta_t
        n_steps = max(1, int(round(PerceptionSettings.c15_prediction_horizon / dt)))
        return dt, n_steps, (np.arange(n_steps) + 1) * dt

    def polyline(self, agent: AgentState, theta: float, dt: float, n_steps: int) -> np.ndarray:
        """Constant-velocity ``[n_steps, 2]`` path."""
        time = (np.arange(n_steps) + 1) * dt
        return np.column_stack((
            agent.x + agent.velocity * np.cos(theta) * time,
            agent.y + agent.velocity * np.sin(theta) * time,
        ))

    def marginals(self, theta: float, time: np.ndarray) -> np.ndarray:
        """``[n_steps, 2, 2]`` world covariance, longer along ``theta`` than across it."""
        along = (self.along_sigma * time) ** 2
        cross = (self.cross_sigma * time) ** 2
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        out = np.zeros((time.shape[0], 2, 2))
        out[:, 0, 0] = c * c * along + s * s * cross
        out[:, 1, 1] = s * s * along + c * c * cross
        out[:, 0, 1] = out[:, 1, 0] = c * s * (along - cross)
        return out

    def mode_fan(self, theta: float, n_modes: int) -> tuple[np.ndarray, np.ndarray]:
        """Headings across ±``mode_heading_span``, and weights that sum to 1.

        An odd count places the middle mode on ``theta``. The weight Gaussian
        uses the full fan half-width as its std, so an edge mode stays a large
        fraction of the middle mode.
        """
        n = max(1, int(n_modes))
        if n == 1:
            return np.array([theta]), np.array([1.0])
        offsets = np.linspace(-self.mode_heading_span, self.mode_heading_span, n)
        weights = np.exp(-0.5 * (offsets / self.mode_heading_span) ** 2)
        return theta + offsets, weights / weights.sum()


class ConstantVelocityPrediction(ConstantVelocityForecast, PredictionStrategy):
    """Predict each agent's future positions assuming constant velocity.

    Writes results into ``pm.prediction`` as ``SingleTrajectory``.
    """

    world_requirements = frozenset()
    stack_requirements = frozenset({StackCapability.DETECTION, StackCapability.TRACKING})
    stack_capabilities = frozenset({StackCapability.PREDICTION_TRAJECTORY})

    def predict(
        self,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
    ) -> PerceptionModel:
        if perception_model is None:
            raise ValueError("perception_model is required for prediction")
        dt, n_steps, _time = self.horizon()
        agents = perception_model.agent_vehicles
        trajectories = {
            agent.agent_id: self.polyline(agent, agent.theta, dt, n_steps)
            for agent in agents
        }
        perception_model.prediction = SingleTrajectory(predict_delta_t=dt, trajectories=trajectories)
        log.debug("Predicted trajectories for %d agents over %d steps", len(agents), n_steps)
        return perception_model


class ConstantVelocityGP(ConstantVelocityForecast, PredictionStrategy):
    """Constant-velocity mean with a heading-aligned covariance that grows with time.

    Writes ``GP``. Along the heading the position std is ``2 * time``; across
    it, ``0.8 * time``.
    """

    world_requirements = frozenset()
    stack_requirements = frozenset({StackCapability.DETECTION, StackCapability.TRACKING})
    stack_capabilities = frozenset({StackCapability.PREDICTION_GP})

    def predict(
        self,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
    ) -> PerceptionModel:
        if perception_model is None:
            raise ValueError("perception_model is required for prediction")
        dt, n_steps, time = self.horizon()
        agents = perception_model.agent_vehicles
        means = {}
        covariance = {}
        for agent in agents:
            means[agent.agent_id] = self.polyline(agent, agent.theta, dt, n_steps)
            blocks = self.marginals(agent.theta, time)
            joint = np.zeros((2 * n_steps, 2 * n_steps))
            for k in range(n_steps):
                joint[2 * k:2 * k + 2, 2 * k:2 * k + 2] = blocks[k]
            covariance[agent.agent_id] = joint
        perception_model.prediction = GP(
            predict_delta_t=dt,
            means=means,
            covariance=covariance,
        )
        return perception_model


class ConstantVelocityOccupancyFlow(ConstantVelocityForecast, PredictionStrategy):
    """Constant-velocity Gaussian, the same one as :class:`ConstantVelocityGP`, as grids.

    Each cell is the probability that the inflated vehicle rectangle covers that
    cell. Writes per-agent ``OccupancyFlow``.
    """

    # Metres per cell. Coarser than the lidar map so a spread-out scene stays small.
    cell_size = 1.0
    stamp_sigmas = 3.0

    world_requirements = frozenset()
    stack_requirements = frozenset({StackCapability.DETECTION, StackCapability.TRACKING})
    stack_capabilities = frozenset({StackCapability.PREDICTION_OCCUPANCY})

    def window(
        self,
        means: list[np.ndarray],
        horizon: float,
        body_pad: float = 0.0,
    ) -> tuple[float, float, int, int]:
        """Shared window around every forecast mean, padded by ``3 σ_along`` plus the body."""
        pts = np.vstack(means)
        pad = self.stamp_sigmas * self.along_sigma * horizon + body_pad
        resolution = self.cell_size
        origin_x = float(np.floor((pts[:, 0].min() - pad) / resolution) * resolution)
        origin_y = float(np.floor((pts[:, 1].min() - pad) / resolution) * resolution)
        max_x = float(np.ceil((pts[:, 0].max() + pad) / resolution) * resolution)
        max_y = float(np.ceil((pts[:, 1].max() + pad) / resolution) * resolution)
        width = max(1, int(np.round((max_x - origin_x) / resolution)))
        height = max(1, int(np.round((max_y - origin_y) / resolution)))
        return origin_x, origin_y, height, width

    def stamp(
        self,
        grid: np.ndarray,
        origin_x: float,
        origin_y: float,
        mean: np.ndarray,
        cov: np.ndarray,
        length: float,
        width: float,
        heading: float,
        margin: float,
    ) -> None:
        """Write the probability that the inflated body covers each cell center.

        ``heading`` is the body axis, the same frame as ``cov``. Half-extents are
        ``length / 2 + margin`` along that axis and ``width / 2 + margin`` across
        it. The covariance is diagonal in that frame, so the probability is the
        product of two Gaussian CDF spans. Cells under the body can each be near
        1; the grid is not renormalized.
        """
        resolution = self.cell_size
        half_along = max(0.0, 0.5 * float(length) + float(margin))
        half_cross = max(0.0, 0.5 * float(width) + float(margin))
        c = float(np.cos(heading))
        s = float(np.sin(heading))
        sym = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
        along_axis = np.array([c, s])
        cross_axis = np.array([-s, c])
        sig_along = float(np.sqrt(max(0.0, float(along_axis @ sym @ along_axis))))
        sig_cross = float(np.sqrt(max(0.0, float(cross_axis @ sym @ cross_axis))))
        reach = self.stamp_sigmas * max(sig_along, sig_cross) + float(np.hypot(half_along, half_cross))
        height, width_cells = grid.shape
        c0 = max(0, int(np.floor((float(mean[0]) - reach - origin_x) / resolution)))
        c1 = min(width_cells - 1, int(np.floor((float(mean[0]) + reach - origin_x) / resolution)))
        r0 = max(0, int(np.floor((float(mean[1]) - reach - origin_y) / resolution)))
        r1 = min(height - 1, int(np.floor((float(mean[1]) + reach - origin_y) / resolution)))
        if c1 < c0 or r1 < r0:
            return
        cx = origin_x + (np.arange(c0, c1 + 1) + 0.5) * resolution
        cy = origin_y + (np.arange(r0, r1 + 1) + 0.5) * resolution
        xx, yy = np.meshgrid(cx, cy, indexing="xy")
        dx = xx - float(mean[0])
        dy = yy - float(mean[1])
        off_along = dx * c + dy * s
        off_cross = -dx * s + dy * c
        grid[r0:r1 + 1, c0:c1 + 1] = np.clip(
            self._gaussian_span(off_along, half_along, sig_along)
            * self._gaussian_span(off_cross, half_cross, sig_cross),
            0.0,
            1.0,
        )

    @staticmethod
    def _gaussian_span(offset: np.ndarray, half: float, sigma: float) -> np.ndarray:
        """Probability mass of ``N(0, sigma^2)`` inside ``[offset - half, offset + half]``."""
        sigma = max(float(sigma), 1e-12)
        scale = 1.0 / (sigma * np.sqrt(2.0))
        return 0.5 * (erf((half - offset) * scale) - erf((-half - offset) * scale))

    def predict(
        self,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
    ) -> PerceptionModel:
        if perception_model is None:
            raise ValueError("perception_model is required for prediction")
        dt, n_steps, time = self.horizon()
        agents = perception_model.agent_vehicles
        if not agents:
            perception_model.prediction = OccupancyFlow(predict_delta_t=dt, resolution=self.cell_size)
            return perception_model
        margin = float(PlanningSettings.c20_obstacle_inflation_margin)
        means = {
            agent.agent_id: self.polyline(agent, agent.theta, dt, n_steps)
            for agent in agents
        }
        marginals = {
            agent.agent_id: self.marginals(agent.theta, time)
            for agent in agents
        }
        body_pad = max(
            float(np.hypot(agent.length / 2.0 + margin, agent.width / 2.0 + margin))
            for agent in agents
        )
        origin_x, origin_y, height, width = self.window(list(means.values()), float(time[-1]), body_pad)
        occupancy: dict[int, list[np.ndarray]] = {}
        for agent in agents:
            grids = [np.zeros((height, width)) for _ in range(n_steps)]
            mean = means[agent.agent_id]
            covs = marginals[agent.agent_id]
            for k in range(n_steps):
                self.stamp(
                    grids[k], origin_x, origin_y, mean[k], covs[k],
                    agent.length, agent.width, agent.theta, margin,
                )
            occupancy[agent.agent_id] = grids
        perception_model.prediction = OccupancyFlow(
            predict_delta_t=dt,
            occupancy_flow=occupancy,
            origin_x=origin_x,
            origin_y=origin_y,
            resolution=self.cell_size,
        )
        return perception_model


class ConstantVelocityGMM(ConstantVelocityForecast, PredictionStrategy):
    """Constant-velocity heading fan as a Gaussian mixture.

    Mode count is ``c15_gmm_n_modes``. Each mode's position covariance is the
    heading-aligned ellipse from :class:`ConstantVelocityGP`, turned to that
    mode's heading. Writes ``GMM``.
    """

    world_requirements = frozenset()
    stack_requirements = frozenset({StackCapability.DETECTION, StackCapability.TRACKING})
    stack_capabilities = frozenset({StackCapability.PREDICTION_GMM})

    def predict(
        self,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
    ) -> PerceptionModel:
        if perception_model is None:
            raise ValueError("perception_model is required for prediction")
        dt, n_steps, time = self.horizon()
        agents = perception_model.agent_vehicles
        n_modes = max(1, int(PerceptionSettings.c15_gmm_n_modes))
        trajectories: dict[int, np.ndarray] = {}
        weights: dict[int, np.ndarray] = {}
        covariances: dict[int, np.ndarray] = {}
        for agent in agents:
            headings, mode_weights = self.mode_fan(agent.theta, n_modes)
            trajectories[agent.agent_id] = np.stack([
                self.polyline(agent, heading, dt, n_steps) for heading in headings
            ])
            weights[agent.agent_id] = mode_weights
            covariances[agent.agent_id] = np.stack([
                self.marginals(float(heading), time) for heading in headings
            ])
        perception_model.prediction = GMM(
            predict_delta_t=dt,
            trajectories=trajectories,
            weights=weights,
            covariances=covariances,
        )
        return perception_model


class ConstantVelocityMultiTrajectory(ConstantVelocityForecast, PredictionStrategy):
    """Constant-velocity heading fan as weighted polylines, without covariance.

    Mode count is ``c15_multitrajectory_n_modes``. Writes ``MultiTrajectory``.
    """

    world_requirements = frozenset()
    stack_requirements = frozenset({StackCapability.DETECTION, StackCapability.TRACKING})
    stack_capabilities = frozenset({StackCapability.PREDICTION_MULTI_TRAJECTORY})

    def predict(
        self,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
    ) -> PerceptionModel:
        if perception_model is None:
            raise ValueError("perception_model is required for prediction")
        dt, n_steps, _time = self.horizon()
        agents = perception_model.agent_vehicles
        n_modes = max(1, int(PerceptionSettings.c15_multitrajectory_n_modes))
        trajectories: dict[int, np.ndarray] = {}
        weights: dict[int, np.ndarray] = {}
        for agent in agents:
            headings, mode_weights = self.mode_fan(agent.theta, n_modes)
            trajectories[agent.agent_id] = np.stack([
                self.polyline(agent, float(heading), dt, n_steps) for heading in headings
            ])
            weights[agent.agent_id] = mode_weights
        perception_model.prediction = MultiTrajectory(
            predict_delta_t=dt,
            trajectories=trajectories,
            weights=weights,
        )
        return perception_model


class KalmanTracker(TrackingStrategy):
    """Constant-velocity Kalman filter tracker with greedy data association.

    Detection sub-strategies output per-frame boxes with ``velocity = 0`` and
    re-assign ``agent_id`` every frame.  This tracker links detections across
    frames into persistent tracks, estimating each agent's velocity and
    smoothing its position.  Unmatched detections spawn new tracks; tracks that
    go unmatched for ``max_missed`` frames are removed.
    """

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
        self._tracks: list["KalmanTracker._Track"] = []
        self._next_id = 0

    world_requirements = frozenset()
    stack_requirements = frozenset({StackCapability.DETECTION})
    stack_capabilities = frozenset({StackCapability.TRACKING})

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 0

    def track(
        self,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
    ) -> PerceptionModel:
        if perception_model is None:
            raise ValueError("perception_model is required for tracking")
        dt = self._dt
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
            self._tracks.append(self._Track(self._next_id, det, self._init_vel_var))
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

    def _to_agent(self, trk: "KalmanTracker._Track") -> AgentState:
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

    Accepts both 2D scans ``(N, 2)`` and 3D point clouds ``(N, 3+)`` in the
    lidar's own coordinate frame; ``sensors.lidar.to_map`` places them in the map
    frame with ``perception_model.ego_vehicle`` before clustering, so detected
    agents come out in map coordinates.  For 3D
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

    world_requirements = frozenset({AnyOf(WorldCapability.LIDAR_2D, WorldCapability.LIDAR_3D)})
    stack_requirements = frozenset()
    stack_capabilities = frozenset({StackCapability.DETECTION})

    def detect(
        self,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
        rgb_img=None,
        depth_img=None,
        lidar_data=None,
    ) -> PerceptionModel:
        if perception_model is None:
            raise ValueError("perception_model is required for detection")
        lidar_sensor = sensors.get_lidar() if sensors is not None else None
        if lidar_data is None and lidar_sensor is not None:
            lidar_data = lidar_sensor.points
        if lidar_data is None or len(lidar_data) == 0:
            perception_model.detection_clusters = None
            return perception_model
        ego = perception_model.ego_vehicle if perception_model.ego_vehicle is not None else State(theta=0.0)
        ego_xytheta = (ego.x, ego.y, ego.theta)
        # Sensor frame → map frame via the stack's own pose estimate.
        lidar_sensor = lidar_sensor if lidar_sensor is not None else Lidar()
        pts = np.asarray(lidar_sensor.to_map(lidar_data, ego), dtype=float)
        if pts.shape[1] >= 3:
            mask = (pts[:, 2] >= self._z_min) & (pts[:, 2] <= self._z_max)
            pts = pts[mask]
            if len(pts) == 0:
                perception_model.detection_clusters = None
                return perception_model
        pts_xy = pts[:, :2]
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

