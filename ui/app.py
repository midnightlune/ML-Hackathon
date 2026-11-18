"""
Map-matching demo UI (two-laptop setup).

Runs on the UI laptop. Point it at the API laptop's LAN IP in the sidebar.

Run:
    streamlit run ui/app.py

Note on coordinates: the .track files and the GeoJSON use [lon, lat].
folium wants [lat, lon]. Everything is swapped in one place (latlon()) -- if the
route shows up in the sea off West Africa, that swap is what went wrong.
"""

from __future__ import annotations

import time
from typing import Any

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

DEFAULT_API = "http://localhost:8000"
TIMEOUT = 30

st.set_page_config(page_title="Map matching demo", layout="wide")


# ----------------------------------------------------------------- helpers ---
def latlon(coords: list[list[float]]) -> list[list[float]]:
    """[[lon, lat], ...] -> [[lat, lon], ...] for folium."""
    return [[c[1], c[0]] for c in coords]


def geometry(fc: dict[str, Any] | None, geom_type: str) -> list[list[float]]:
    """Pull the first geometry of a given type out of a FeatureCollection."""
    if not fc:
        return []
    for feature in fc.get("features", []):
        geom = feature.get("geometry") or {}
        if geom.get("type") == geom_type:
            return geom.get("coordinates") or []
    return []


def connection_help(base: str, err: Exception) -> None:
    st.error(f"Could not reach the API at {base}")
    st.caption(f"{type(err).__name__}: {err}")
    st.markdown(
        "Work through these in order, on the **API laptop** unless stated:\n\n"
        "1. Is the server running, and does its log say `0.0.0.0:8000` (not `127.0.0.1`)?\n"
        "2. Is the URL above the API laptop's **hotspot** IPv4 from `ipconfig`, not its Wi-Fi/Ethernet one?\n"
        "3. Is inbound TCP 8000 allowed in Windows Firewall? (`profile=any` — the hotspot often "
        "counts as a Public network.)\n"
        "4. From **this** laptop: `ping <api-laptop-ip>`. If that fails, the hotspot has client "
        "isolation on — turn it off in the phone's hotspot settings, or use a different hotspot.\n"
        "5. Then: `python tools/smoke_test.py http://<api-laptop-ip>:8000`"
    )


# ----------------------------------------------------------------- sidebar ---
st.sidebar.header("Connection")
api_base = st.sidebar.text_input("API base URL", value=st.session_state.get("api_base", DEFAULT_API))
st.session_state["api_base"] = api_base.rstrip("/")
base = st.session_state["api_base"]

col_a, col_b = st.sidebar.columns(2)
check = col_a.button("Check API")
reload_trips = col_b.button("Reload trips")

if check:
    try:
        r = requests.get(f"{base}/health", timeout=5)
        r.raise_for_status()
        info = r.json()
        st.sidebar.success(f"OK - {info.get('trips', 0)} trips")
        st.sidebar.caption(f"eval_dir: {info.get('eval_dir')}")
        st.sidebar.caption(f"layout: {info.get('layout')}")
    except Exception as exc:  # noqa: BLE001 - demo UI, show anything that goes wrong
        st.sidebar.error(f"{type(exc).__name__}")
        st.sidebar.caption(str(exc)[:300])

if reload_trips:
    st.session_state.pop("trips", None)


# ------------------------------------------------------------------- trips ---
st.title("PathSnap - GPS map matching")
st.caption("Classifier (LightGBM) + HMM/Viterbi. Ground truth stays on the API server; only aggregate metrics come back.")

if "trips" not in st.session_state:
    try:
        r = requests.get(f"{base}/trips", timeout=10)
        r.raise_for_status()
        payload = r.json()
        # Accept either a bare list or {"trips": [...]} so a later API change can't break the UI.
        st.session_state["trips"] = payload if isinstance(payload, list) else payload.get("trips", [])
    except Exception as exc:  # noqa: BLE001
        connection_help(base, exc)
        st.stop()

trip_ids: list[str] = st.session_state["trips"]
if not trip_ids:
    st.warning("The API is reachable but reports no trips. Check its --eval-dir, then press Reload trips.")
    st.stop()

left, right = st.columns([3, 1])

with left:
    trip_id = st.selectbox("Trip", trip_ids, index=0)
    run = st.button("Run prediction", type="primary")

if run:
    with st.spinner(f"Predicting {trip_id}..."):
        started = time.perf_counter()
        try:
            r = requests.post(f"{base}/predict", json={"trip_id": trip_id}, timeout=TIMEOUT)
            r.raise_for_status()
            st.session_state["result"] = r.json()
            st.session_state["round_trip_ms"] = (time.perf_counter() - started) * 1000.0
        except Exception as exc:  # noqa: BLE001
            connection_help(base, exc)
            st.stop()

result: dict[str, Any] | None = st.session_state.get("result")


# -------------------------------------------------------------------- view ---
def lines(fc: dict[str, Any] | None) -> list[list[list[float]]]:
    """All line pieces ([lon, lat] lists) of a FeatureCollection (LineString or MultiLineString)."""
    out = []
    for f in (fc or {}).get("features", []):
        g = f.get("geometry") or {}
        if g.get("type") == "LineString":
            out.append(g["coordinates"])
        elif g.get("type") == "MultiLineString":
            out.extend(g["coordinates"])
    return out


def conf_colour(c: float) -> str:
    return "#16a34a" if c >= 0.8 else ("#f59e0b" if c >= 0.5 else "#dc2626")


def fmt(v) -> str:
    return "-" if v is None else f"{v:.3f}"


if result is None:
    with left:
        st.info("Pick a trip and press **Run prediction**.")
else:
    gps_coords = geometry(result.get("gps"), "LineString")
    pts_feature = next((f for f in result["gps"]["features"] if f["geometry"]["type"] == "MultiPoint"), None)
    metrics = result.get("metrics") or {}
    hist = st.session_state.setdefault("history", {})
    hist[result["trip_id"]] = {
        "trip": result["trip_id"], "points": result.get("n_points"),
        "Top-1 classifier": metrics.get("classifier_only_top1_reliable"), "Top-1 HMM": metrics.get("hmm_top1_reliable"),
        "route F1 HMM": metrics.get("hmm_route_f1"), "continuity HMM": metrics.get("hmm_continuity"),
        "breaks": result.get("breaks"), "server ms": (result.get("timings_ms") or {}).get("total")}

    with left:
        if gps_coords:
            gps_ll = latlon(gps_coords)
            fmap = folium.Map(location=gps_ll[len(gps_ll) // 2], zoom_start=14, tiles="OpenStreetMap")
            g_raw = folium.FeatureGroup(name="Raw GPS (coloured by confidence)")
            if pts_feature:
                conf = pts_feature["properties"].get("confidence") or []
                step = max(1, len(conf) // 3000)      # keep the browser fast on long trips
                for (lon, lat), c in list(zip(pts_feature["geometry"]["coordinates"], conf))[::step]:
                    folium.CircleMarker([lat, lon], radius=2, color=conf_colour(c), fill=True, weight=0,
                                        fill_opacity=0.9, tooltip=f"confidence {c:.2f}").add_to(g_raw)
            g_hmm = folium.FeatureGroup(name="Predicted route (HMM)")
            for seg in lines(result.get("predicted_route")):
                folium.PolyLine(latlon(seg), color="#7c3aed", weight=6, opacity=0.85).add_to(g_hmm)
            g_clf = folium.FeatureGroup(name="Classifier-only route", show=False)
            for seg in lines(result.get("classifier_only_route")):
                folium.PolyLine(latlon(seg), color="#0ea5e9", weight=3, opacity=0.9, dash_array="6,6").add_to(g_clf)
            for g in (g_hmm, g_clf, g_raw):
                g.add_to(fmap)
            folium.CircleMarker(gps_ll[0], radius=6, color="#16a34a", fill=True, tooltip="start").add_to(fmap)
            folium.CircleMarker(gps_ll[-1], radius=6, color="#111827", fill=True, tooltip="end").add_to(fmap)
            folium.LayerControl(collapsed=False).add_to(fmap)
            lats = [p[0] for p in gps_ll]; lons = [p[1] for p in gps_ll]
            fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
            st_folium(fmap, width=None, height=560, returned_objects=[])
            st.caption("Point colour = classifier confidence of the HMM-chosen arc: green >= 0.8, amber >= 0.5, red < 0.5.")
        else:
            st.warning("The response contained no GPS LineString to draw.")

    with right:
        st.subheader("Classifier vs HMM")
        c1, c2 = st.columns(2)
        c1.metric("Top-1 classifier", fmt(metrics.get("classifier_only_top1_reliable")))
        c2.metric("Top-1 HMM", fmt(metrics.get("hmm_top1_reliable")),
                  None if metrics.get("hmm_top1_reliable") is None else
                  f"{metrics['hmm_top1_reliable'] - metrics['classifier_only_top1_reliable']:+.3f}")
        c1.metric("Route F1 classifier", fmt(metrics.get("classifier_only_route_f1")))
        c2.metric("Route F1 HMM", fmt(metrics.get("hmm_route_f1")))
        c1.metric("Continuity classifier", fmt(metrics.get("classifier_only_continuity")))
        c2.metric("Continuity HMM", fmt(metrics.get("hmm_continuity")))
        st.caption(f"Top-1 on reliable points ({metrics.get('reliable_points', '-')}); "
                   f"all points: classifier {fmt(metrics.get('classifier_only_top1_all'))}, HMM {fmt(metrics.get('hmm_top1_all'))}. "
                   f"Label quality: {metrics.get('label_quality', '-')}. Breaks: {result.get('breaks')}.")
        timings = result.get("timings_ms") or {}
        st.metric("Latency (server)", f"{(timings.get('total') or 0) / 1000:.2f} s",
                  f"round trip {st.session_state.get('round_trip_ms', 0.0) / 1000:.2f} s", delta_color="off")
        with st.expander("Timings and raw response"):
            st.write(timings)
            st.json({k: v for k, v in result.items() if k not in ("gps", "predicted_route", "classifier_only_route")})

    st.subheader("Trips run this session")
    st.dataframe(list(hist.values()), use_container_width=True)
