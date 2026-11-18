"""API smoke test on one student_test trip (skipped if the data or artifacts are not present)."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "raw" / "student_test"
pytestmark = pytest.mark.skipif(not (EVAL.is_dir() and (ROOT / "artifacts" / "r2" / "feature_spec.json").exists()),
                                reason="needs data/raw/student_test and artifacts/r2")


def test_health_trips_predict():
    os.environ["EVAL_DIR"] = str(EVAL)
    sys.path.insert(0, str(ROOT / "api"))
    from fastapi.testclient import TestClient
    import main as api   # api/main.py

    c = TestClient(api.app)
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["trips"] == 10
    trips = c.get("/trips").json()
    smallest = min(trips, key=lambda t: (EVAL / t / f"{t}.track").stat().st_size)
    r = c.post("/predict", json={"trip_id": smallest})
    assert r.status_code == 200, r.text
    js = r.json()
    for key in ("gps", "predicted_route", "classifier_only_route", "metrics", "timings_ms"):
        assert key in js
    assert js["metrics"]["hmm_top1_reliable"] is not None
    text = r.text
    assert ".route" not in text and "label_arc" not in text   # no ground truth in the response
    assert c.post("/predict", json={"trip_id": "../../etc"}).status_code == 404
