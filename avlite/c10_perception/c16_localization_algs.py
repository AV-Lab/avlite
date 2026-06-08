"""LiDAR-based localization strategies for AVLite.

This module implements :class:`LidarLocalization`, a lightweight scan-to-map
localization strategy that estimates the ego pose by aligning each incoming
LiDAR scan against a reference map built from the first scan, using Iterative
Closest Point (ICP).  It works with 2D scans ``(N, 2)`` and 3D point clouds
``(N, 3+)``; 3D input is "squashed" to 2D by keeping only points whose height
lies within ``[z_min, z_max]`` and then dropping the z (and intensity) columns.
"""
from typing import Optional, Type

import logging

import numpy as np

from avlite.c10_perception.c11_perception_model import PerceptionModel
from avlite.c10_perception.c13_localization_strategy import LocalizationStrategy
from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c60_common.c62_capabilities import (
    AnyOf,
    LocalizationCapability,
    WorldCapability,
)

log = logging.getLogger(__name__)


class LidarLocalization(LocalizationStrategy):
    """Estimate the ego pose by ICP scan-to-map alignment of LiDAR scans.

    On the first scan a reference map is built and the running pose estimate is
    seeded from the current ``ego_vehicle`` pose.  On every subsequent scan,
    ICP aligns the new scan to the reference map (initialised from the previous
    estimate) and the resulting 2D rigid transform updates the running pose,
    which is written back into ``perception_model.ego_vehicle`` in-place.

    3D point clouds are squashed to 2D: points are kept only when their height
    is within ``[z_min, z_max]``, then the x/y columns are used.

    Parameters
    ----------
    z_min, z_max : float
        Z-band filter for 3D input [m].  Ignored for 2D input.
    max_iterations : int
        Maximum ICP iterations per :meth:`localize` call.
    tolerance : float
        Convergence tolerance on the mean per-iteration shift [m].
    max_correspondence_dist : float
        Correspondences farther apart than this are rejected [m].
    map_subsample : int
        Keep every ``map_subsample``-th map point (1 keeps all).
    """

    def __init__(
        self,
        perception_model: PerceptionModel,
        setting: Type[PerceptionSettings] = PerceptionSettings,
        z_min: float = PerceptionSettings.c16_localization_lidar_z_min,
        z_max: float = PerceptionSettings.c16_localization_lidar_z_max,
        max_iterations: int = PerceptionSettings.c16_localization_icp_max_iterations,
        tolerance: float = PerceptionSettings.c16_localization_icp_tolerance,
        max_correspondence_dist: float = PerceptionSettings.c16_localization_icp_max_correspondence_dist,
        map_subsample: int = PerceptionSettings.c16_localization_map_subsample,
    ):
        super().__init__(perception_model, setting)
        self._z_min = z_min
        self._z_max = z_max
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._max_correspondence_dist = max_correspondence_dist
        self._map_subsample = max(1, int(map_subsample))

        # Running pose estimate (x, y, theta); reference map of 2D points.
        self._x: Optional[float] = None
        self._y: Optional[float] = None
        self._theta: Optional[float] = None
        self._map: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Capability declaration
    # ------------------------------------------------------------------

    @property
    def requirements(self) -> set:
        return {AnyOf(WorldCapability.LIDAR_2D, WorldCapability.LIDAR_3D)}

    @property
    def capabilities(self) -> set[LocalizationCapability]:
        return {
            LocalizationCapability.LOCALIZATION_2D,
            LocalizationCapability.LOCALIZATION_HEADING,
        }

    # ------------------------------------------------------------------
    # Main estimation step
    # ------------------------------------------------------------------

    def localize(
        self,
        imu: Optional[np.ndarray] = None,
        lidar: Optional[np.ndarray] = None,
        rgb_img: Optional[np.ndarray] = None,
    ) -> None:
        scan = self._squash(lidar)
        if scan is None or len(scan) < 3:
            return

        ego = self.perception_model.ego_vehicle

        # First valid scan: seed the estimate from the current ego pose and
        # store the scan as the reference map. Nothing to align against yet.
        if self._map is None:
            self._x = float(ego.x)
            self._y = float(ego.y)
            self._theta = float(ego.theta)
            self._map = scan[:: self._map_subsample]
            return

        # Align the new scan to the reference map starting from the previous
        # estimate, then commit the refined pose.
        x, y, theta = self._icp(scan, self._map, self._x, self._y, self._theta)
        self._x, self._y, self._theta = x, y, theta
        ego.x = x
        ego.y = y
        ego.theta = theta

    def reset(self):
        """Forget the reference map and running pose estimate."""
        self._x = None
        self._y = None
        self._theta = None
        self._map = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _squash(self, lidar: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Reduce a LiDAR scan to ordered 2D points ``(N, 2)``.

        2D scans are returned as-is. 3D point clouds are filtered to the
        ``[z_min, z_max]`` height band, then the z/intensity columns are
        dropped.
        """
        if lidar is None or len(lidar) == 0:
            return None
        pts = np.asarray(lidar, dtype=float)
        if pts.ndim != 2 or pts.shape[1] < 2:
            return None
        if pts.shape[1] >= 3:
            mask = (pts[:, 2] >= self._z_min) & (pts[:, 2] <= self._z_max)
            pts = pts[mask]
            if len(pts) == 0:
                return None
        return pts[:, :2]

    def _icp(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        x0: float,
        y0: float,
        theta0: float,
    ) -> tuple[float, float, float]:
        """Align ``src`` onto ``dst`` via point-to-point ICP.

        The transform is initialised from the previous pose estimate
        ``(x0, y0, theta0)`` and refined by alternating nearest-neighbour
        correspondence and rigid (Procrustes) fitting.  Returns the refined
        absolute pose ``(x, y, theta)``.
        """
        x, y, theta = x0, y0, theta0
        prev_error = np.inf

        for _ in range(self._max_iterations):
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            transformed = src @ rot.T + np.array([x, y])

            idx, dists = self._nearest_neighbours(transformed, dst)
            keep = dists <= self._max_correspondence_dist
            if keep.sum() < 3:
                break
            matched_src = transformed[keep]
            matched_dst = dst[idx[keep]]

            d_cos, d_sin, dtx, dty = self._best_fit_transform(matched_src, matched_dst)

            # Compose the incremental world-frame transform
            #   new = R_delta @ (R(theta) p + (x, y)) + (dtx, dty)
            # with the running pose, updating x, y and theta together.
            new_x = d_cos * x - d_sin * y + dtx
            new_y = d_sin * x + d_cos * y + dty
            x, y = new_x, new_y
            theta = theta + np.arctan2(d_sin, d_cos)

            mean_error = float(dists[keep].mean())
            if abs(prev_error - mean_error) < self._tolerance:
                break
            prev_error = mean_error

        return x, y, theta

    @staticmethod
    def _nearest_neighbours(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Brute-force nearest neighbour from each ``src`` point to ``dst``.

        Returns the index into ``dst`` and the distance for each ``src`` point.
        """
        diff = src[:, None, :] - dst[None, :, :]
        sq = np.einsum("ijk,ijk->ij", diff, diff)
        idx = np.argmin(sq, axis=1)
        dists = np.sqrt(sq[np.arange(len(src)), idx])
        return idx, dists

    @staticmethod
    def _best_fit_transform(src: np.ndarray, dst: np.ndarray) -> tuple[float, float, float, float]:
        """Least-squares rigid 2D transform mapping ``src`` onto ``dst``.

        Returns the incremental transform as ``(cos, sin, tx, ty)`` such that
        ``dst ≈ R(cos, sin) @ src + (tx, ty)``.
        """
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)
        src_c = src - src_mean
        dst_c = dst - dst_mean

        h = src_c.T @ dst_c
        u, _, vt = np.linalg.svd(h)
        r = vt.T @ u.T
        # Reflection guard.
        if np.linalg.det(r) < 0:
            vt[-1, :] *= -1
            r = vt.T @ u.T

        t = dst_mean - r @ src_mean
        return float(r[0, 0]), float(r[1, 0]), float(t[0]), float(t[1])
