import logging
import time

import numpy as np
import scipy.sparse as sp
import shapely
from scipy.optimize import Bounds, minimize
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

from avlite.c10_perception.c11_perception_model import RaceMap
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c29_settings import PlanningSettings
from avlite.c50_common.c53_trajectory_tracker import TrajectoryTracker
from avlite.c60_apps.c68_paths import DataPaths

log = logging.getLogger(__name__)


class GlobalCenterlineRacePlanner(GlobalPlannerStrategy):
    """A global planner that reads a race-line JSON file with left/right
    boundary coordinates and produces a centre-line path with curvature-adapted
    target velocities.

    Expected JSON format::

        {
            "LeftBound":      [[x, y, z], ...],
            "RightBound":     [[x, y, z], ...],
            "ReferencePoint": [lat, lon, alt]   # required WGS84 degrees
        }

    The path is the corridor centre between the left and right boundary polylines,
    refined from an index-wise midpoint so tight corners stay equidistant to both sides.
    Target speed at each waypoint is capped by the lateral-acceleration limit:

        a_lat = v² · κ  →  v = min(v_max, sqrt(a_lat / κ))
    """

    def __init__(
        self,
        filepath: str | RaceMap,
        max_velocity: float = 10.0,
        max_lateral_accel: float = 5.0,
        margin: float | None = None,
    ):
        super().__init__()
        if isinstance(filepath, RaceMap):
            self._race_map = filepath
            self.filepath = filepath.source_path
        else:
            self._race_map = None
            self.filepath = filepath
        self.max_velocity = max_velocity
        self.max_lateral_accel = max_lateral_accel
        self.margin = margin

    def plan(self) -> GlobalPlan:
        margin = (
            self.margin
            if self.margin is not None
            else PlanningSettings.c20_boundary_margin
        )

        if self._race_map is not None:
            left = self._race_map.left_bound
            right = self._race_map.right_bound
        else:
            race_map = RaceMap.from_path(DataPaths.resolve_stored(self.filepath))
            left = race_map.left_bound
            right = race_map.right_bound

        if len(left) != len(right):
            raise ValueError(
                f"LeftBound ({len(left)}) and RightBound ({len(right)}) "
                "arrays must have equal length."
            )

        # Apply inward margin: shift each boundary toward the centreline.
        eps = 1e-6
        diff = right - left
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        dir_unit = diff / np.maximum(norms, eps)
        eff_left = left + margin * dir_unit
        eff_right = right - margin * dir_unit

        path_np = (eff_left + eff_right) / 2.0
        path_np = self._refine_centerline_to_corridor(path_np, eff_left, eff_right)
        path = [tuple(p) for p in path_np]
        velocity = self._curvature_velocity(path_np)
        trajectory = TrajectoryTracker(path=path, velocity=velocity)

        self.global_plan = GlobalPlan(
            start_point=path[0],
            goal_point=path[-1],
            path=path,
            velocity=velocity,
            left_boundary_d=[trajectory.convert_xy_to_sd(x, y)[1] for x, y in eff_left],
            right_boundary_d=[trajectory.convert_xy_to_sd(x, y)[1] for x, y in eff_right],
            left_boundary_x=eff_left[:, 0].tolist(),
            left_boundary_y=eff_left[:, 1].tolist(),
            right_boundary_x=eff_right[:, 0].tolist(),
            right_boundary_y=eff_right[:, 1].tolist(),
            trajectory=trajectory,
            race_mode=True,
        )
        log.debug(f"GlobalCenterlineRacePlanner: planned {len(path)} waypoints from {self.filepath}")
        return self.global_plan

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _refine_centerline_to_corridor(
        path_np: np.ndarray, left_np: np.ndarray, right_np: np.ndarray
    ) -> np.ndarray:
        """Re-center path so each point is midway between nearest boundary points."""
        left_ls = LineString(left_np)
        right_ls = LineString(right_np)
        refined = np.empty_like(path_np)
        for i, p in enumerate(path_np):
            pl = nearest_points(Point(p), left_ls)[1]
            pr = nearest_points(Point(p), right_ls)[1]
            refined[i] = [(pl.x + pr.x) / 2.0, (pl.y + pr.y) / 2.0]
        return refined

    def _curvature_velocity(self, path_np: np.ndarray) -> list[float]:
        """Compute per-waypoint speed limited by lateral acceleration.

        From circular motion: a_lat = v² · κ  →  v = sqrt(a_lat / κ).
        Speed is then capped by max_velocity on straights (κ ≈ 0).
        """
        eps = 1e-6
        x, y = path_np[:, 0], path_np[:, 1]
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        kappa = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2) ** 1.5
        kappa = np.maximum(kappa, eps)
        v = np.minimum(self.max_velocity, np.sqrt(self.max_lateral_accel / kappa))
        return v.tolist()


class GlobalRacePlanner(GlobalPlannerStrategy):
    """A global planner that computes an optimized raceline inside the race
    corridor instead of following the centerline.

    The raceline is parametrized by a lateral offset α_i along the corridor
    normal at each reference waypoint and solved as a bounded linear
    least-squares problem blending two quadratic objectives:

    - **minimum curvature**: ‖D₂ x(α)‖² + ‖D₂ y(α)‖²  (second differences)
    - **shortest path**:     ‖D₁ x(α)‖² + ‖D₁ y(α)‖²  (first differences)

    subject to box bounds keeping the raceline inside the track boundaries
    minus a safety margin. The linearization (normals, bounds) is refreshed
    over a few outer iterations. The velocity profile respects the lateral
    acceleration limit (v = √(a_lat/κ)) plus longitudinal acceleration and
    braking limits via forward/backward passes over v².

    Accepts the same race-boundary JSON (or ``RaceMap``) as
    :class:`GlobalCenterlineRacePlanner`.

    Formulation based on Braghin et al., "Race driver model" (Computers &
    Structures, 2008) for the curvature/length blend, and Heilmeier et al.,
    "Minimum curvature trajectory planning and control for an autonomous race
    car" (Vehicle System Dynamics, 2020) for the iteratively re-linearized
    lateral-offset QP and forward-backward velocity solver; see
    docs/algorithms.md for full citations.
    """

    #: Uniform arc-length spacing (m) of the optimization reference.
    RESAMPLE_STEP = 3.0

    def __init__(
        self,
        filepath: str | RaceMap,
        max_velocity: float | None = None,
        max_lateral_accel: float | None = None,
        max_longitudinal_accel: float | None = None,
        max_braking_decel: float | None = None,
        curvature_weight: float | None = None,
        optimization_iterations: int | None = None,
        margin: float | None = None,
    ):
        super().__init__()
        if isinstance(filepath, RaceMap):
            self._race_map = filepath
            self.filepath = filepath.source_path
        else:
            self._race_map = None
            self.filepath = filepath
        self.max_velocity = max_velocity
        self.max_lateral_accel = max_lateral_accel
        self.max_longitudinal_accel = max_longitudinal_accel
        self.max_braking_decel = max_braking_decel
        self.curvature_weight = curvature_weight
        self.optimization_iterations = optimization_iterations
        self.margin = margin

    def plan(self) -> GlobalPlan:
        s = PlanningSettings
        max_velocity = self.max_velocity if self.max_velocity is not None else s.c25_max_velocity
        max_lat_accel = self.max_lateral_accel if self.max_lateral_accel is not None else s.c25_max_lateral_accel
        max_lon_accel = (
            self.max_longitudinal_accel if self.max_longitudinal_accel is not None else s.c25_max_longitudinal_accel
        )
        max_brake = self.max_braking_decel if self.max_braking_decel is not None else s.c25_max_braking_decel
        curv_weight = self.curvature_weight if self.curvature_weight is not None else s.c25_curvature_weight
        iterations = (
            self.optimization_iterations if self.optimization_iterations is not None else s.c25_optimization_iterations
        )
        margin = self.margin if self.margin is not None else s.c20_boundary_margin
        curv_weight = float(np.clip(curv_weight, 0.0, 1.0))

        t_start = time.perf_counter()
        race_map = self._race_map
        if race_map is None:
            resolved = DataPaths.resolve_stored(self.filepath)
            log.info(f"GlobalRacePlanner: loading race boundaries from {resolved}")
            race_map = RaceMap.from_path(resolved)
        left = np.asarray(race_map.left_bound, dtype=float)
        right = np.asarray(race_map.right_bound, dtype=float)
        if len(left) < 2 or len(right) < 2:
            raise ValueError("RaceMap boundaries must contain at least two points each.")

        left_ls = LineString(left)
        right_ls = LineString(right)

        # Initial reference: index-wise midpoint of the corridor (boundaries may
        # have different lengths — interpolate the shorter one).
        n_mid = max(len(left), len(right))
        mid = (self._resample_by_count(left, n_mid) + self._resample_by_count(right, n_mid)) / 2.0
        closed = self._is_closed(mid)
        ref = self._resample(mid, self.RESAMPLE_STEP, closed)
        log.info(
            f"GlobalRacePlanner: {len(left)}/{len(right)} boundary pts → "
            f"{len(ref)} reference pts at {self.RESAMPLE_STEP} m spacing "
            f"(closed={closed}, margin={margin} m, curvature_weight={curv_weight})"
        )

        # Outer iterations: re-linearize normals/bounds around the previous solution.
        raceline = ref
        for it in range(max(1, int(iterations))):
            t_it = time.perf_counter()
            normals = self._left_normals(ref, closed)
            d_a = self._signed_offsets(ref, normals, left_ls)
            d_b = self._signed_offsets(ref, normals, right_ls)
            upper = np.maximum(d_a, d_b) - margin
            lower = np.minimum(d_a, d_b) + margin
            # Degenerate (corridor narrower than 2·margin): pin to corridor center.
            narrow = lower >= upper
            center = (upper + lower) / 2.0
            eps_b = 1e-6
            lb = np.where(narrow, center - eps_b, lower)
            ub = np.where(narrow, center + eps_b, upper)

            alpha = self._solve_offsets(ref, normals, lb, ub, curv_weight, closed)
            # Keep the exact solved points (each satisfies its own bound); the
            # uniform resample is only used to linearize the next iteration —
            # resampling the final solution could cut inside the margin where
            # the raceline rides an apex bound.
            raceline = ref + alpha[:, None] * normals
            ref = self._resample(raceline, self.RESAMPLE_STEP, closed)
            log.info(
                f"GlobalRacePlanner: iteration {it + 1}/{iterations} solved "
                f"({len(alpha)} offsets, max |α|={float(np.max(np.abs(alpha))):.2f} m) "
                f"in {time.perf_counter() - t_it:.2f} s"
            )

        normals = self._left_normals(raceline, closed)
        kappa = self._curvature(raceline, closed)
        velocity = self._velocity_profile(
            raceline, kappa, max_velocity, max_lat_accel, max_lon_accel, max_brake, closed
        )
        log.info(
            f"GlobalRacePlanner: velocity profile "
            f"min={min(velocity):.1f} max={max(velocity):.1f} m/s "
            f"(v_max={max_velocity}, a_lat={max_lat_accel}, "
            f"a_accel={max_lon_accel}, a_brake={max_brake} m/s²)"
        )

        # Per-waypoint boundary offsets relative to the raceline (positive d = left),
        # inset by the margin so the local planner corridor respects it too.
        d_a = self._signed_offsets(raceline, normals, left_ls)
        d_b = self._signed_offsets(raceline, normals, right_ls)
        left_d = np.maximum(d_a, d_b) - margin
        right_d = np.minimum(d_a, d_b) + margin
        # At apexes the raceline rides the margin-inset boundary, and the final
        # resampling can leave it marginally on it; keep a small strictly
        # positive corridor on both sides so downstream lateral sampling never
        # sees an inverted band.
        min_side = 0.05
        left_d = np.maximum(left_d, min_side)
        right_d = np.minimum(right_d, -min_side)
        left_xy = raceline + left_d[:, None] * normals
        right_xy = raceline + right_d[:, None] * normals

        path = [tuple(p) for p in raceline]
        trajectory = TrajectoryTracker(path=path, velocity=velocity)

        self.global_plan = GlobalPlan(
            start_point=path[0],
            goal_point=path[-1],
            path=path,
            velocity=velocity,
            left_boundary_d=left_d.tolist(),
            right_boundary_d=right_d.tolist(),
            left_boundary_x=left_xy[:, 0].tolist(),
            left_boundary_y=left_xy[:, 1].tolist(),
            right_boundary_x=right_xy[:, 0].tolist(),
            right_boundary_y=right_xy[:, 1].tolist(),
            trajectory=trajectory,
            race_mode=True,
        )
        seg = np.linalg.norm(np.diff(raceline, axis=0), axis=1)
        lap_time = float(np.sum(seg / np.maximum((np.array(velocity)[:-1] + np.array(velocity)[1:]) / 2.0, 1e-3)))
        log.info(
            f"GlobalRacePlanner: done — {len(path)} waypoints, "
            f"length={float(np.sum(seg)):.0f} m, est. time={lap_time:.1f} s, "
            f"total {time.perf_counter() - t_start:.2f} s"
        )
        return self.global_plan

    # ------------------------------------------------------------------
    # geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_closed(pts: np.ndarray) -> bool:
        """A track is closed when its endpoints nearly coincide."""
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        threshold = max(3.0 * float(np.median(seg)), 1.0)
        return float(np.linalg.norm(pts[0] - pts[-1])) < threshold

    @staticmethod
    def _resample_by_count(pts: np.ndarray, n: int) -> np.ndarray:
        """Resample a polyline to *n* points, uniform in arc length."""
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        s_new = np.linspace(0.0, s[-1], n)
        return np.column_stack([np.interp(s_new, s, pts[:, 0]), np.interp(s_new, s, pts[:, 1])])

    @staticmethod
    def _resample(pts: np.ndarray, step: float, closed: bool) -> np.ndarray:
        """Resample to uniform arc-length spacing ≈ *step*.

        Closed tracks are resampled over the wrap segment and returned without
        a duplicated endpoint.
        """
        pts_ext = np.vstack([pts, pts[:1]]) if closed else pts
        seg = np.linalg.norm(np.diff(pts_ext, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(s[-1])
        n = max(int(round(total / step)), 8)
        if closed:
            s_new = np.linspace(0.0, total, n, endpoint=False)
        else:
            s_new = np.linspace(0.0, total, n + 1)
        return np.column_stack(
            [np.interp(s_new, s, pts_ext[:, 0]), np.interp(s_new, s, pts_ext[:, 1])]
        )

    @staticmethod
    def _left_normals(pts: np.ndarray, closed: bool) -> np.ndarray:
        """Unit left-hand normals (positive d side), matching TrajectoryTracker."""
        if closed:
            tangent = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
        else:
            tangent = np.gradient(pts, axis=0)
        norms = np.linalg.norm(tangent, axis=1, keepdims=True)
        tangent = tangent / np.maximum(norms, 1e-9)
        return np.column_stack([-tangent[:, 1], tangent[:, 0]])

    @staticmethod
    def _signed_offsets(pts: np.ndarray, normals: np.ndarray, boundary_ls: LineString) -> np.ndarray:
        """Signed lateral offset from each point to its nearest boundary point.

        Positive means the boundary lies on the left (positive-d) side.
        Uses shapely's vectorized line projection (much faster than per-point
        ``nearest_points`` calls).
        """
        pts_geom = shapely.points(pts)
        nearest = shapely.line_interpolate_point(boundary_ls, shapely.line_locate_point(boundary_ls, pts_geom))
        qx = shapely.get_x(nearest)
        qy = shapely.get_y(nearest)
        return (qx - pts[:, 0]) * normals[:, 0] + (qy - pts[:, 1]) * normals[:, 1]

    # ------------------------------------------------------------------
    # optimization
    # ------------------------------------------------------------------

    @staticmethod
    def _difference_matrices(n: int, ds: float, closed: bool) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """Arc-length-normalized first/second difference operators (periodic when closed)."""
        if closed:
            d1 = sp.diags([-np.ones(n), np.ones(n - 1), [1.0]], [0, 1, -(n - 1)], format="csr")
            d2 = sp.diags(
                [np.ones(n - 1), -2.0 * np.ones(n), np.ones(n - 1), [1.0], [1.0]],
                [-1, 0, 1, n - 1, -(n - 1)],
                format="csr",
            )
        else:
            d1 = sp.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n), format="csr")
            d2 = sp.diags(
                [np.ones(n - 2), -2.0 * np.ones(n - 2), np.ones(n - 2)],
                [0, 1, 2],
                shape=(n - 2, n),
                format="csr",
            )
        return d1 / ds, d2 / ds**2

    def _solve_offsets(
        self,
        ref: np.ndarray,
        normals: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        curv_weight: float,
        closed: bool,
    ) -> np.ndarray:
        """Solve the bounded least-squares problem for the lateral offsets α.

        Raceline points are p_i = c_i + α_i·n_i, so both x(α) and y(α) are
        affine in α and the blended curvature/length objective is a linear
        least-squares residual ‖A α − b‖² with box bounds.

        Each term is normalized by its residual at α = 0 (the current
        reference): the raw length residual (‖tangent‖ ≈ 1 per row) is orders
        of magnitude larger than the curvature residual (≈ κ per row), so
        without normalization the shortest-path term would dominate any blend
        and pin the raceline to the inside of every corner.
        """
        n = len(ref)
        seg = np.linalg.norm(np.diff(ref, axis=0), axis=1)
        ds = float(np.mean(seg))
        d1, d2 = self._difference_matrices(n, ds, closed)

        nx = sp.diags(normals[:, 0])
        ny = sp.diags(normals[:, 1])
        cx, cy = ref[:, 0], ref[:, 1]

        eps = 1e-9
        norm_curv = float(np.sqrt(np.sum((d2 @ cx) ** 2) + np.sum((d2 @ cy) ** 2)))
        norm_len = float(np.sqrt(np.sum((d1 @ cx) ** 2) + np.sum((d1 @ cy) ** 2)))
        w_curv = np.sqrt(curv_weight) / max(norm_curv, eps)
        w_len = np.sqrt(1.0 - curv_weight) / max(norm_len, eps)
        # Joint rescale (argmin-invariant) to keep A entries well conditioned.
        scale = max(w_curv, w_len)
        w_curv /= scale
        w_len /= scale

        blocks_a: list[sp.spmatrix] = []
        blocks_b: list[np.ndarray] = []
        if w_curv > 0.0:
            blocks_a += [w_curv * (d2 @ nx), w_curv * (d2 @ ny)]
            blocks_b += [-w_curv * (d2 @ cx), -w_curv * (d2 @ cy)]
        if w_len > 0.0:
            blocks_a += [w_len * (d1 @ nx), w_len * (d1 @ ny)]
            blocks_b += [-w_len * (d1 @ cx), -w_len * (d1 @ cy)]

        a_mat = sp.vstack(blocks_a, format="csr")
        b_vec = np.concatenate(blocks_b)

        # Minimize the box-bounded quadratic 0.5·αᵀHα − fᵀα directly with
        # L-BFGS-B on the (banded, sparse) normal equations — orders of
        # magnitude faster than lsq_linear on the stacked system.
        h_mat = (a_mat.T @ a_mat).tocsr()
        f_vec = a_mat.T @ b_vec

        def fun_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
            hx = h_mat @ x
            return 0.5 * float(x @ hx) - float(f_vec @ x), hx - f_vec

        x0 = np.clip(np.zeros(n), lb, ub)
        result = minimize(
            fun_grad,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=Bounds(lb, ub),
            options={"maxiter": 1000, "maxcor": 20},
        )
        return np.clip(result.x, lb, ub)

    # ------------------------------------------------------------------
    # velocity profile
    # ------------------------------------------------------------------

    @staticmethod
    def _curvature(pts: np.ndarray, closed: bool) -> np.ndarray:
        """Finite-difference curvature magnitude (wrap-padded for closed tracks)."""
        pad = 3 if closed else 0
        ext = np.vstack([pts[-pad:], pts, pts[:pad]]) if closed else pts
        x, y = ext[:, 0], ext[:, 1]
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        kappa = np.abs(dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2) ** 1.5, 1e-12)
        return kappa[pad : len(kappa) - pad] if closed else kappa

    @staticmethod
    def _velocity_profile(
        pts: np.ndarray,
        kappa: np.ndarray,
        max_velocity: float,
        max_lat_accel: float,
        max_lon_accel: float,
        max_brake: float,
        closed: bool,
    ) -> list[float]:
        """Lateral-limit velocity cap plus forward/backward longitudinal passes.

        Forward pass enforces v_{i+1}² ≤ v_i² + 2·a_accel·Δs, backward pass
        enforces v_i² ≤ v_{i+1}² + 2·a_brake·Δs. Closed tracks iterate the
        passes with wrap-around so the profile is consistent across the seam.
        """
        v = np.minimum(max_velocity, np.sqrt(max_lat_accel / np.maximum(kappa, 1e-6)))
        ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        ds_wrap = float(np.linalg.norm(pts[0] - pts[-1])) if closed else 0.0

        n_pass = 3 if closed else 1
        for _ in range(n_pass):
            for i in range(len(v) - 1):
                v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2.0 * max_lon_accel * ds[i]))
            if closed:
                v[0] = min(v[0], np.sqrt(v[-1] ** 2 + 2.0 * max_lon_accel * ds_wrap))
            for i in range(len(v) - 2, -1, -1):
                v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * max_brake * ds[i]))
            if closed:
                v[-1] = min(v[-1], np.sqrt(v[0] ** 2 + 2.0 * max_brake * ds_wrap))
        return v.tolist()
