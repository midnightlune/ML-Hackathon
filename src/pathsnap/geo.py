"""Per-trip metric projection and vectorised segment geometry."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from pyproj import CRS, Transformer

from .io import Trip

GAP_S = 30.0  # time gap that starts a new segment


def utm_crs(lon: float, lat: float) -> CRS:
    zone = int(np.floor((lon + 180) / 6) % 60) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


@dataclass
class ProjTrip:
    trip_id: str
    crs: CRS
    t: np.ndarray        # (N,)
    pxy: np.ndarray      # (N,2) projected track, metres
    segment_id: np.ndarray  # (N,) increments at gaps > GAP_S
    nxy: np.ndarray      # (M,2) projected nodes
    arcs: np.ndarray     # (A,2)
    a_xy: np.ndarray     # (A,2) arc start xy
    b_xy: np.ndarray     # (A,2) arc end xy
    arc_len: np.ndarray  # (A,)
    arc_bearing: np.ndarray  # (A,) radians, atan2(dy, dx)
    valid: np.ndarray    # (N,) bool: finite + plausible lon/lat

    _tree: shapely.STRtree | None = None

    @property
    def arc_tree(self) -> shapely.STRtree:
        if self._tree is None:
            lines = shapely.linestrings(np.stack([self.a_xy, self.b_xy], axis=1))
            self._tree = shapely.STRtree(lines)
        return self._tree


def valid_lonlat(lonlat: np.ndarray) -> np.ndarray:
    lon, lat = lonlat[:, 0], lonlat[:, 1]
    return np.isfinite(lon) & np.isfinite(lat) & (np.abs(lon) <= 180) & (np.abs(lat) <= 90) & ~((lon == 0) & (lat == 0))


def segment_ids(t: np.ndarray, gap: float = GAP_S) -> np.ndarray:
    if len(t) == 0:
        return np.zeros(0, dtype=np.int32)
    return np.concatenate([[0], np.cumsum(np.diff(t) > gap)]).astype(np.int32)


def project_trip(trip: Trip) -> ProjTrip:
    valid = valid_lonlat(trip.track[:, :2])
    c = trip.track[valid, :2].mean(axis=0)
    crs = utm_crs(float(c[0]), float(c[1]))
    tr = Transformer.from_crs(4326, crs, always_xy=True)
    px, py = tr.transform(trip.track[:, 0], trip.track[:, 1])
    nx, ny = tr.transform(trip.nodes[:, 0], trip.nodes[:, 1])
    nxy = np.column_stack([nx, ny])
    a, b = nxy[trip.arcs[:, 0]], nxy[trip.arcs[:, 1]]
    d = b - a
    return ProjTrip(
        trip_id=trip.trip_id, crs=crs, t=trip.track[:, 2],
        pxy=np.column_stack([px, py]), segment_id=segment_ids(trip.track[:, 2]),
        nxy=nxy, arcs=trip.arcs, a_xy=a, b_xy=b,
        arc_len=np.hypot(d[:, 0], d[:, 1]), arc_bearing=np.arctan2(d[:, 1], d[:, 0]), valid=valid,
    )


def point_seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Broadcasting point-to-segment distance. p, a, b: (...,2). Returns (dist, t in [0,1])."""
    ab = b - a
    ap = p - a
    den = (ab * ab).sum(-1)
    t = np.where(den > 0, (ap * ab).sum(-1) / np.where(den > 0, den, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[..., None] * ab
    return np.linalg.norm(p - proj, axis=-1), t
