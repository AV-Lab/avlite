"""LiDAR occupancy-grid mapping.

``OccupancyMapper`` is the ``MappingStrategy`` that integrates the tick
``SensorFrame`` into a log-odds window and writes ``PerceptionModel.occupancy_map``.
"""
from __future__ import annotations

import logging

import numpy as np

from avlite.c10_perception.c11_perception_model import (
    HDMap,
    Map,
    OccupancyMap,
    PerceptionModel,
    RaceMap,
)
from avlite.c10_perception.c14_mapping_strategy import MappingStrategy
from avlite.c10_perception.c19_settings import PerceptionSettings, PerceptionSettingsSchema
from avlite.c50_common.c51_capabilities import AnyOf, StackCapability, WorldCapability
from avlite.c50_common.c52_world_sensor_datatypes import Lidar, SensorFrame

log = logging.getLogger(__name__)

# Inverse-sensor-model log-odds increments (p_occ=0.7, p_free=0.3) and clamp.
_L_OCC = float(np.log(0.7 / 0.3))
_L_FREE = float(np.log(0.3 / 0.7))
_L_MIN = -2.0
_L_MAX = 2.0


class OccupancyMapper(MappingStrategy):
    """Online occupancy grid from LiDAR; optionally holds a static map too."""

    world_requirements = frozenset({AnyOf(WorldCapability.LIDAR_2D, WorldCapability.LIDAR_3D)})
    stack_requirements = frozenset({StackCapability.LOCALIZATION})
    stack_capabilities = frozenset({StackCapability.MAP_OCCUPANCY})

    def __init__(
        self,
        map: Map | None = None,
        setting: PerceptionSettingsSchema = PerceptionSettings,
    ):
        super().__init__(setting=setting)
        self.map = map
        caps = {StackCapability.MAP_OCCUPANCY}
        if isinstance(map, HDMap):
            caps.add(StackCapability.MAP_HD)
        elif isinstance(map, RaceMap):
            caps.add(StackCapability.MAP_RACE_TRACK)
        self.stack_capabilities = frozenset(caps)
        self._z_min = setting.c17_z_min
        self._z_max = setting.c17_z_max
        self._init_resolution = float(setting.c17_resolution)
        self._init_n = max(1, int(round(float(setting.c17_size) / self._init_resolution)))
        self._ref_point: tuple[float, float] | None = None
        self.resolution = self._init_resolution
        self.n = self._init_n
        self._rows = self._init_n
        self._cols = self._init_n
        self.size = self.n * self.resolution
        self._logodds = np.zeros((self._rows, self._cols), dtype=np.float32)
        self._prob = np.full((self._rows, self._cols), 0.5, dtype=np.float32)
        self._origin_x: float | None = None
        self._origin_y: float | None = None
        self._published = OccupancyMap(
            grid=self._prob,
            origin_x=0.0,
            origin_y=0.0,
            resolution=self.resolution,
        )
        if isinstance(map, OccupancyMap):
            grid = np.asarray(map.grid, dtype=np.float32)
            if grid.ndim == 2 and grid.size > 0:
                self.resolution = float(map.resolution)
                self._rows, self._cols = int(grid.shape[0]), int(grid.shape[1])
                self.n = self._rows
                self.size = self._cols * self.resolution
                eps = 1e-6
                self._prob = np.clip(grid.copy(), 0.0, 1.0)
                p = np.clip(self._prob, eps, 1.0 - eps)
                self._logodds = np.log(p / (1.0 - p)).astype(np.float32)
                np.clip(self._logodds, _L_MIN, _L_MAX, out=self._logodds)
                self._origin_x = float(map.origin_x)
                self._origin_y = float(map.origin_y)
                self._ref_point = map.reference_point
                self._published = OccupancyMap(
                    grid=self._prob,
                    origin_x=self._origin_x,
                    origin_y=self._origin_y,
                    resolution=self.resolution,
                    source_path=map.source_path,
                    _reference_point=self._ref_point,
                )

    def update(
        self,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
        ego=None,
    ) -> None:
        if perception_model is None:
            return
        pose = ego if ego is not None else perception_model.ego_vehicle
        lidar_sensor = sensors.lidar_sensor if sensors is not None and sensors.lidar_sensor is not None else Lidar()
        lidar = None if sensors is None else sensors.lidar
        pts = np.empty((0, 2), dtype=np.float64)
        if lidar is not None and len(lidar) > 0:
            raw = np.asarray(lidar_sensor.to_map(lidar, pose), dtype=float)
            if raw.size:
                if raw.ndim == 1:
                    raw = raw.reshape(1, -1)
                if raw.shape[1] >= 3:
                    raw = raw[(raw[:, 2] >= self._z_min) & (raw[:, 2] <= self._z_max)]
                pts = np.asarray(raw[:, :2], dtype=np.float64)

        # Rays start at the sensor's own map-frame position, not the vehicle
        # origin — matters whenever the lidar mount has a translational offset.
        # Derived from the static mount alone, independent of whether this tick
        # produced any points.
        origin = np.asarray(lidar_sensor.to_map(np.zeros((1, 3)), pose), dtype=float)
        ray_x, ray_y = float(origin[0, 0]), float(origin[0, 1])

        ego_x, ego_y = float(pose.x), float(pose.y)
        if self._origin_x is None:
            half = self._init_n * self.resolution / 2.0
            self._origin_x = ego_x - half
            self._origin_y = ego_y - half
        cover_x = np.array([ego_x, ray_x], dtype=np.float64)
        cover_y = np.array([ego_y, ray_y], dtype=np.float64)
        if pts.size:
            if pts.ndim == 1:
                pts = pts.reshape(1, -1)
            cover_x = np.concatenate([cover_x, pts[:, 0]])
            cover_y = np.concatenate([cover_y, pts[:, 1]])
        self._expand_to_cover(cover_x, cover_y)

        if pts.size:
            if pts.ndim == 1:
                pts = pts.reshape(1, -1)
            rdx = pts[:, 0] - ray_x
            rdy = pts[:, 1] - ray_y
            dist = np.hypot(rdx, rdy)
            valid = dist > 1e-6
            if np.any(valid):
                pts = pts[valid]
                rdx = rdx[valid]
                rdy = rdy[valid]
                dist = dist[valid]
                n_samp = np.maximum(1, np.ceil(dist / self.resolution).astype(np.int32))
                free_counts = np.maximum(0, n_samp - 1)
                n_free = int(free_counts.sum())
                if n_free:
                    ray_idx = np.repeat(np.arange(len(pts)), free_counts)
                    starts = np.empty_like(free_counts)
                    starts[0] = 0
                    if len(free_counts) > 1:
                        np.cumsum(free_counts[:-1], out=starts[1:])
                    k = np.arange(n_free, dtype=np.float64) - np.repeat(starts, free_counts)
                    t = (k + 0.5) / n_samp[ray_idx]
                    fx = ray_x + t * rdx[ray_idx]
                    fy = ray_y + t * rdy[ray_idx]
                    fc = np.floor((fx - self._origin_x) / self.resolution).astype(np.int32)
                    fr = np.floor((fy - self._origin_y) / self.resolution).astype(np.int32)
                    inside = (fr >= 0) & (fr < self._rows) & (fc >= 0) & (fc < self._cols)
                    np.add.at(self._logodds, (fr[inside], fc[inside]), _L_FREE)
                hc = np.floor((pts[:, 0] - self._origin_x) / self.resolution).astype(np.int32)
                hr = np.floor((pts[:, 1] - self._origin_y) / self.resolution).astype(np.int32)
                inside = (hr >= 0) & (hr < self._rows) & (hc >= 0) & (hc < self._cols)
                np.add.at(self._logodds, (hr[inside], hc[inside]), _L_OCC)
                np.clip(self._logodds, _L_MIN, _L_MAX, out=self._logodds)

        if self._prob.shape != self._logodds.shape:
            self._prob = np.empty_like(self._logodds)
        np.negative(self._logodds, out=self._prob)
        np.exp(self._prob, out=self._prob)
        self._prob += 1.0
        np.reciprocal(self._prob, out=self._prob)
        self._published.grid = self._prob
        self._published.origin_x = float(self._origin_x)
        self._published.origin_y = float(self._origin_y)
        self._published.resolution = self.resolution
        self._published._reference_point = self._ref_point
        perception_model.occupancy_map = self._published

    def _chunk_pad(self, needed: int) -> int:
        if needed <= 0:
            return 0
        chunk = self._init_n
        return ((needed + chunk - 1) // chunk) * chunk

    def _expand_to_cover(self, xs: np.ndarray, ys: np.ndarray) -> None:
        """Grow the grid so every world point in *xs*/*ys* falls inside a cell."""
        res = self.resolution
        x0, y0 = float(self._origin_x), float(self._origin_y)
        x1 = x0 + self._cols * res
        y1 = y0 + self._rows * res
        xmin, xmax = float(np.min(xs)), float(np.max(xs))
        ymin, ymax = float(np.min(ys)), float(np.max(ys))
        pad_left = int(np.ceil((x0 - xmin) / res)) if xmin < x0 else 0
        pad_right = int(np.floor((xmax - x1) / res)) + 1 if xmax >= x1 else 0
        pad_bottom = int(np.ceil((y0 - ymin) / res)) if ymin < y0 else 0
        pad_top = int(np.floor((ymax - y1) / res)) + 1 if ymax >= y1 else 0
        pad_left = self._chunk_pad(pad_left)
        pad_right = self._chunk_pad(pad_right)
        pad_bottom = self._chunk_pad(pad_bottom)
        pad_top = self._chunk_pad(pad_top)
        if not (pad_left or pad_right or pad_bottom or pad_top):
            return
        new_rows = self._rows + pad_bottom + pad_top
        new_cols = self._cols + pad_left + pad_right
        new_l = np.zeros((new_rows, new_cols), dtype=np.float32)
        new_l[pad_bottom:pad_bottom + self._rows, pad_left:pad_left + self._cols] = self._logodds
        self._logodds = new_l
        self._prob = np.empty((new_rows, new_cols), dtype=np.float32)
        self._origin_x = x0 - pad_left * res
        self._origin_y = y0 - pad_bottom * res
        self._rows = new_rows
        self._cols = new_cols
        self.n = max(new_rows, new_cols)
        self.size = self.n * res

    def reset(self) -> None:
        self._ref_point = None
        self.resolution = self._init_resolution
        self.n = self._init_n
        self._rows = self._init_n
        self._cols = self._init_n
        self.size = self.n * self.resolution
        self._logodds = np.zeros((self._rows, self._cols), dtype=np.float32)
        self._prob = np.full((self._rows, self._cols), 0.5, dtype=np.float32)
        self._origin_x = None
        self._origin_y = None
        self._published = OccupancyMap(
            grid=self._prob,
            origin_x=0.0,
            origin_y=0.0,
            resolution=self.resolution,
        )
