"""
Generate fake trips so the two-laptop test can happen before the real data lands.

Creates the same four-file layout as the real dataset (.track .nodes .arcs .route)
with a synthetic wiggly path, one folder per trip.

    python tools/make_fake_trips.py --out fake_test --trips 4 --points 900

Delete the folder once `student_test` is unzipped; the API reads whichever
--eval-dir you give it.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

# A few start points around the world, to mirror the real dataset being worldwide.
ORIGINS = [
    ("berlin", 13.4050, 52.5200),
    ("hanoi", 105.8342, 21.0278),
    ("london", -0.1276, 51.5072),
    ("austin", -97.7431, 30.2672),
    ("moscow", 37.6173, 55.7558),
]


def make_track(lon0: float, lat0: float, n: int, rng: random.Random) -> list[tuple[float, float, int]]:
    """A path that drifts and turns, at 1 Hz, with a little GPS noise."""
    points: list[tuple[float, float, int]] = []
    lon, lat = lon0, lat0
    heading = rng.uniform(0, 2 * math.pi)
    step = 0.00012  # ~13 m per second

    for t in range(n):
        if t % 120 == 0:
            heading += rng.uniform(-0.8, 0.8)
        heading += rng.uniform(-0.02, 0.02)
        lon += step * math.cos(heading) / max(math.cos(math.radians(lat)), 0.1)
        lat += step * math.sin(heading)
        jitter_lon = rng.gauss(0, 0.00002)
        jitter_lat = rng.gauss(0, 0.00002)
        points.append((lon + jitter_lon, lat + jitter_lat, 1700000000 + t))

    return points


def write_trip(folder: Path, name: str, lon0: float, lat0: float, n: int, rng: random.Random) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    points = make_track(lon0, lat0, n, rng)

    # .track : lon lat t, whitespace separated, no header
    with (folder / f"{name}.track").open("w", encoding="utf-8") as fh:
        for lon, lat, t in points:
            fh.write(f"{lon:.7f} {lat:.7f} {t}\n")

    # .nodes : every 5th point becomes a road node -> "id lon lat"
    nodes = points[::5]
    with (folder / f"{name}.nodes").open("w", encoding="utf-8") as fh:
        for i, (lon, lat, _) in enumerate(nodes):
            fh.write(f"{i} {lon:.7f} {lat:.7f}\n")

    # .arcs : directed segments "arc_id from_node to_node", with reverse twins
    arcs: list[tuple[int, int, int]] = []
    arc_id = 0
    for i in range(len(nodes) - 1):
        arcs.append((arc_id, i, i + 1))
        arc_id += 1
        if rng.random() < 0.8:  # ~80% of roads exist in both directions
            arcs.append((arc_id, i + 1, i))
            arc_id += 1
    with (folder / f"{name}.arcs").open("w", encoding="utf-8") as fh:
        for aid, a, b in arcs:
            fh.write(f"{aid} {a} {b}\n")

    # .route : the forward arcs in order. NOTE this is ground truth - the API
    # never reads it, and it must never be sent to the UI.
    with (folder / f"{name}.route").open("w", encoding="utf-8") as fh:
        for aid, a, b in arcs:
            if b == a + 1:
                fh.write(f"{aid}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fake trips for the demo skeleton")
    parser.add_argument("--out", default="fake_test", help="output folder")
    parser.add_argument("--trips", type=int, default=4)
    parser.add_argument("--points", type=int, default=900)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out).expanduser().resolve()

    for i in range(args.trips):
        city, lon0, lat0 = ORIGINS[i % len(ORIGINS)]
        name = f"trip_{i:03d}_{city}"
        write_trip(out / name, name, lon0, lat0, args.points, rng)
        print(f"wrote {out / name} ({args.points} points)")

    print(f"\n{args.trips} fake trips in {out}")
    print(f"start the API with:  python api/main.py --eval-dir {out} --port 8000")


if __name__ == "__main__":
    main()
