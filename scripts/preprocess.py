"""Step 1: load + project every trip, quality checks -> results/eda/inventory.csv,
projected points -> data/processed/points_<partition>.parquet, Folium map of trip 00000001.

Run: python -m scripts.preprocess
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):  # also allow `python scripts/<name>.py` from any folder
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import logging
import traceback

import folium
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap.geo import GAP_S, project_trip
from pathsnap.io import PARTITIONS, ROOT, load_trip, trip_dirs
from pathsnap.labels.align import load_route  # inventory only: route counts / connectivity checks

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("preprocess")
OUT_EDA = ROOT / "results" / "eda"
OUT_PROC = ROOT / "data" / "processed"
MAP_TRIP = "00000001"


def inventory_row(part, d, trip, pt, route):
    t = trip.track[:, 2]
    dt = np.diff(t)
    M, A = len(trip.nodes), len(trip.arcs)
    g = coo_matrix((np.ones(A), (trip.arcs[:, 0], trip.arcs[:, 1])), shape=(M, M))
    n_weak, lab = connected_components(g, directed=True, connection="weak")
    deg = np.bincount(trip.arcs.ravel(), minlength=M)
    route_ok = (route.min() >= 0 and route.max() < A) if len(route) else False
    route_conn = bool(route_ok and np.all(trip.arcs[route[:-1], 1] == trip.arcs[route[1:], 0]))
    ext = pt.pxy[pt.valid].max(0) - pt.pxy[pt.valid].min(0)
    return dict(
        partition=part, trip_id=trip.trip_id, epsg=pt.crs.to_epsg(),
        n_points=len(t), n_nodes=M, n_arcs=A, n_route_arcs=len(route),
        track_bytes=(d / f"{trip.trip_id}.track").stat().st_size,
        arcs_bytes=(d / f"{trip.trip_id}.arcs").stat().st_size,
        invalid_coords=int((~pt.valid).sum()),
        non_monotonic_ts=int((dt <= 0).sum()), median_dt_s=float(np.median(dt)) if len(dt) else np.nan,
        frac_dt_1s=float(np.mean(dt == 1)) if len(dt) else np.nan,
        max_gap_s=float(dt.max()) if len(dt) else 0.0, n_gaps_gt30s=int((dt > GAP_S).sum()),
        n_segments=int(pt.segment_id.max() + 1) if len(t) else 0,
        extent_x_km=ext[0] / 1000, extent_y_km=ext[1] / 1000,
        isolated_nodes=int((deg == 0).sum()), weak_components=int(n_weak),
        largest_component_frac=float(np.bincount(lab).max() / M),
        frac_arcs_with_reverse=float(np.isin(trip.arcs[:, 0] * M + trip.arcs[:, 1], trip.arcs[:, 1] * M + trip.arcs[:, 0]).mean()),
        route_ids_in_range=bool(route_ok), route_connected=route_conn,
        route_repeats=int(len(route) - len(np.unique(route))),
    )


def make_map(trip, pt, route, path):
    ll = trip.track[pt.valid, :2]
    m = folium.Map(location=[ll[:, 1].mean(), ll[:, 0].mean()], zoom_start=14, tiles="OpenStreetMap")
    near = np.unique(pt.arc_tree.query(shapely.points(pt.pxy[pt.valid]), predicate="dwithin", distance=150)[1])
    roads = folium.FeatureGroup(name=f"Road network (arcs within 150 m, n={len(near)})")
    for a in near:
        u, v = trip.arcs[a]
        folium.PolyLine([[trip.nodes[u, 1], trip.nodes[u, 0]], [trip.nodes[v, 1], trip.nodes[v, 0]]],
                        color="#999999", weight=2, opacity=0.6).add_to(roads)
    ref = folium.FeatureGroup(name=f"Reference route ({len(route)} arcs)")
    coords = [[trip.nodes[trip.arcs[route[0], 0], 1], trip.nodes[trip.arcs[route[0], 0], 0]]] + \
             [[trip.nodes[trip.arcs[a, 1], 1], trip.nodes[trip.arcs[a, 1], 0]] for a in route]
    folium.PolyLine(coords, color="#1a9850", weight=5, opacity=0.8).add_to(ref)
    gps = folium.FeatureGroup(name=f"Raw GPS ({len(ll)} points)")
    for lon, lat in ll:
        folium.CircleMarker([lat, lon], radius=2, color="#2166ac", fill=True, fill_opacity=0.8, weight=0).add_to(gps)
    for fg in (roads, ref, gps):
        fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[ll[:, 1].min(), ll[:, 0].min()], [ll[:, 1].max(), ll[:, 0].max()]])
    m.save(str(path))


def main():
    OUT_EDA.mkdir(parents=True, exist_ok=True)
    OUT_PROC.mkdir(parents=True, exist_ok=True)
    (ROOT / "maps").mkdir(exist_ok=True)
    rows, fails = [], []
    for part in PARTITIONS:
        pts = []
        for d in trip_dirs(part):
            try:
                trip = load_trip(d)
                pt = project_trip(trip)
                route = load_route(d)
                rows.append(inventory_row(part, d, trip, pt, route))
                pts.append(pd.DataFrame(dict(
                    trip_id=trip.trip_id, point_idx=np.arange(len(pt.t), dtype=np.int32), t=pt.t,
                    x=pt.pxy[:, 0], y=pt.pxy[:, 1], segment_id=pt.segment_id, valid=pt.valid)))
                if trip.trip_id == MAP_TRIP and part == "train":
                    make_map(trip, pt, route, ROOT / "maps" / f"trip_{MAP_TRIP}_raw.html")
            except Exception:
                log.error("trip %s/%s failed:\n%s", part, d.name, traceback.format_exc())
                fails.append((part, d.name))
        pd.concat(pts, ignore_index=True).to_parquet(OUT_PROC / f"points_{part}.parquet", index=False)
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT_EDA / "inventory.csv", index=False)
    g = inv.groupby("partition")
    print(g.agg(trips=("trip_id", "size"), points=("n_points", "sum"), route_arcs=("n_route_arcs", "sum"),
                max_gap_s=("max_gap_s", "max"), trips_with_gap30=("n_gaps_gt30s", lambda s: int((s > 0).sum())),
                invalid=("invalid_coords", "sum"), nonmono=("non_monotonic_ts", "sum"),
                route_disconnected=("route_connected", lambda s: int((~s).sum())),
                route_repeats=("route_repeats", "sum")).to_string())
    print(f"UTM zones: {inv.epsg.nunique()} distinct EPSG codes:", ", ".join(str(int(e)) for e in sorted(inv.epsg.unique())))
    print("failures:", fails or "none")


if __name__ == "__main__":
    main()
