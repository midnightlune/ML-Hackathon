"""
Prove the API is reachable, before blaming Streamlit.

Run this from the UI laptop with the API laptop's URL:

    python tools/smoke_test.py http://192.168.43.12:8000

It hits /health, /trips and /predict and prints what came back. If this passes
and the UI still fails, the problem is in the UI. If this fails, the problem is
the network or the server, and the output says which.

Only needs `requests`.
"""

from __future__ import annotations

import json
import sys
import time

import requests

DEFAULT = "http://localhost:8000"


def step(label: str) -> None:
    print(f"\n--- {label} " + "-" * max(0, 60 - len(label)))


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT).rstrip("/")
    print(f"target: {base}")

    step("GET /health")
    try:
        started = time.perf_counter()
        r = requests.get(f"{base}/health", timeout=5)
        elapsed = (time.perf_counter() - started) * 1000
        print(f"{r.status_code} in {elapsed:.0f}ms")
        print(json.dumps(r.json(), indent=2))
        r.raise_for_status()
    except requests.exceptions.ConnectTimeout:
        print("FAIL connect timeout -> packets are not getting there.")
        print("     Likely Windows Firewall on the API laptop, or hotspot client isolation.")
        print("     Test with: ping <api-laptop-ip>")
        return 1
    except requests.exceptions.ConnectionError as exc:
        print(f"FAIL connection refused / unreachable: {exc}")
        print("     Is the server running? Does its log say 0.0.0.0:8000 and not 127.0.0.1?")
        print("     Is the IP in the URL the API laptop's hotspot IPv4 from ipconfig?")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {type(exc).__name__}: {exc}")
        return 1

    step("GET /trips")
    r = requests.get(f"{base}/trips", timeout=10)
    trips = r.json()
    print(f"{r.status_code} -> {len(trips)} trips")
    print(trips[:10])
    if not trips:
        print("FAIL no trips. Check the API's --eval-dir.")
        return 1

    step(f"POST /predict {{'trip_id': '{trips[0]}'}}")
    started = time.perf_counter()
    r = requests.post(f"{base}/predict", json={"trip_id": trips[0]}, timeout=60)
    elapsed = (time.perf_counter() - started) * 1000
    print(f"{r.status_code} in {elapsed:.0f}ms")
    r.raise_for_status()
    payload = r.json()
    summary = {k: v for k, v in payload.items() if k not in ("gps", "predicted_route")}
    print(json.dumps(summary, indent=2))

    gps_features = len((payload.get("gps") or {}).get("features", []))
    route_features = len((payload.get("predicted_route") or {}).get("features", []))
    print(f"gps features: {gps_features}, route features: {route_features}")

    leaked = [k for k in payload if "ground" in k.lower() or k == "route"]
    if leaked:
        print(f"WARNING response contains suspicious keys: {leaked}")
        print("        Ground truth must never be sent to the UI.")
        return 1

    step("result")
    print("PASS - the API is reachable from this machine and returns a prediction.")
    print("Now check the API laptop's terminal: there should be one log line per request.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
