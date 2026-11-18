"""Round 2 / Step 5: Folium maps of student_test trips (good / median / worst by HMM Top-1, plus the trip
where the HMM hurt most if it hurt anywhere). Layers: raw GPS, nearby road network, reference route,
classifier-only route, HMM route; legend + LayerControl. Uses the deployed predictor (artifacts/r2).
Writes maps/r2/*.html and results/r2/maps_index.csv.
Run: python scripts\\r2_make_maps.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import folium
import numpy as np
import pandas as pd
import shapely

from pathsnap.eval.server_eval import evaluate_trip
from pathsnap.inference import Predictor
from pathsnap.io import RAW, ROOT
from pathsnap.labels.align import load_route   # reference layer only (evaluation)

OUT, RES = ROOT / "maps" / "r2", ROOT / "results" / "r2"
COL = dict(gps="#2563eb", net="#9ca3af", ref="#16a34a", clf="#0ea5e9", hmm="#7c3aed")


def arcs_layer(trip, arc_ids, name, color, weight, dash=None, show=True):
    g = folium.FeatureGroup(name=name, show=show)
    for a in arc_ids:
        u, v = trip.arcs[a]
        folium.PolyLine([[trip.nodes[u, 1], trip.nodes[u, 0]], [trip.nodes[v, 1], trip.nodes[v, 0]]],
                        color=color, weight=weight, opacity=0.85, dash_array=dash).add_to(g)
    return g


def legend(title, m):
    items = "".join(f'<div><span style="display:inline-block;width:22px;height:4px;background:{c};margin-right:6px;'
                    f'vertical-align:middle"></span>{n}</div>' for n, c in
                    (("raw GPS", COL["gps"]), ("road network (within 150 m)", COL["net"]), ("reference route", COL["ref"]),
                     ("classifier-only route", COL["clf"]), ("HMM route", COL["hmm"])))
    html = (f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 10px;'
            f'border:1px solid #ccc;border-radius:6px;font:12px sans-serif"><b>{title}</b>{items}</div>')
    m.get_root().html.add_child(folium.Element(html))


def make_map(pr, tid, tag):
    d = RAW / "student_test" / tid
    pred = pr.predict_trip(d)
    met = evaluate_trip(d, pred)
    trip, pt = pred["trip"], pred["proj"]
    ll = trip.track[pred["point_idx"], :2]
    m = folium.Map(location=[ll[:, 1].mean(), ll[:, 0].mean()], zoom_start=14, tiles="OpenStreetMap")
    near = np.unique(pt.arc_tree.query(shapely.points(pt.pxy[pred["point_idx"]]), predicate="dwithin", distance=150)[1])
    arcs_layer(trip, near, "road network (within 150 m)", COL["net"], 2, show=False).add_to(m)
    arcs_layer(trip, load_route(d), "reference route", COL["ref"], 7).add_to(m)
    clf = pred["classifier_only_arcs"]
    arcs_layer(trip, np.unique(clf), "classifier-only route", COL["clf"], 3, dash="6,6", show=False).add_to(m)
    arcs_layer(trip, np.unique(pred["point_arcs"]), "HMM route", COL["hmm"], 4).add_to(m)
    g = folium.FeatureGroup(name="raw GPS")
    step = max(1, len(ll) // 3000)
    for lon, lat in ll[::step]:
        folium.CircleMarker([lat, lon], radius=2, color=COL["gps"], fill=True, weight=0, fill_opacity=0.8).add_to(g)
    g.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[ll[:, 1].min(), ll[:, 0].min()], [ll[:, 1].max(), ll[:, 0].max()]])
    t = (f"student_test {tid} ({tag}): Top-1 classifier {met.get('classifier_only_top1_reliable', float('nan')):.3f} "
         f"-> HMM {met.get('hmm_top1_reliable', float('nan')):.3f}")
    legend(t, m)
    path = OUT / f"student_test_{tid}_{tag}.html"
    m.save(str(path))
    return dict(trip_id=tid, tag=tag, file=str(path.relative_to(ROOT)), **{k: met.get(k) for k in (
        "classifier_only_top1_reliable", "hmm_top1_reliable", "hmm_route_f1", "hmm_continuity", "label_quality")})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(RES / "hmm_per_trip.csv", dtype={"trip_id": str})
    sel = __import__("json").loads((RES / "hmm_selection.json").read_text())["selected"]
    q = P[(P.partition == "student_test") & (P.model == sel)].sort_values("hmm_top1_eligible")
    q = q[q.eligible_points > 0]
    picks = {"worst": q.trip_id.iloc[0], "median": q.trip_id.iloc[len(q) // 2], "best": q.trip_id.iloc[-1]}
    hurt = q.sort_values("hmm_minus_clf_eligible").iloc[0]
    if hurt.hmm_minus_clf_eligible < 0 and hurt.trip_id not in picks.values():
        picks["hmm_hurt"] = hurt.trip_id
    pr = Predictor.load()
    rows = [make_map(pr, tid, tag) for tag, tid in picks.items()]
    df = pd.DataFrame(rows); df.to_csv(RES / "maps_index.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
