"""Raw trip loading. Deliberately does NOT read .route (ground truth lives in pathsnap.labels)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PARTITIONS = ("train", "student_test")


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray  # (N,3) lon, lat, t
    nodes: np.ndarray  # (M,2) lon, lat
    arcs: np.ndarray   # (A,2) int from_node, to_node


def trip_dirs(partition: str, raw: Path = RAW) -> list[Path]:
    return sorted(p for p in (raw / partition).iterdir() if p.is_dir())


def _load(path: Path, ncol: int, dtype) -> np.ndarray:
    a = np.loadtxt(path, dtype=dtype, ndmin=2)
    if a.size == 0:
        return np.empty((0, ncol), dtype=dtype)
    if a.shape[1] != ncol:
        raise ValueError(f"{path.name}: expected {ncol} columns, got {a.shape[1]}")
    return a


def load_trip(trip_dir: Path) -> Trip:
    tid = trip_dir.name
    return Trip(
        trip_id=tid,
        track=_load(trip_dir / f"{tid}.track", 3, np.float64),
        nodes=_load(trip_dir / f"{tid}.nodes", 2, np.float64),
        arcs=_load(trip_dir / f"{tid}.arcs", 2, np.int64),
    )
