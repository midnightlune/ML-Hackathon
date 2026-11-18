"""Inference code must never import the ground-truth modules (pathsnap.labels, pathsnap.eval)."""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFERENCE_MODULES = ["io.py", "geo.py", "candidates.py", "features.py", "features_v2.py", "inference.py",
                     "hmm/transitions.py", "hmm/viterbi.py"]
FORBIDDEN = ("pathsnap.labels", "pathsnap.eval", ".labels", ".eval", "labels", "eval")


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            yield from (a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            mod = ("." * n.level) + (n.module or "")
            yield mod
            if n.module is None:
                yield from ("." * n.level + a.name for a in n.names)


def test_static_imports():
    for rel in INFERENCE_MODULES:
        for mod in _imports(ROOT / "src" / "pathsnap" / rel):
            tail = mod.lstrip(".")
            assert not (tail.startswith("labels") or tail.startswith("eval") or "pathsnap.labels" in mod
                        or "pathsnap.eval" in mod), f"{rel} imports {mod}"


def test_runtime_imports():
    """Importing the inference module must not load the label/eval packages."""
    code = ("import sys; sys.path.insert(0, r'%s'); import pathsnap.inference; "
            "bad=[m for m in sys.modules if m.startswith(('pathsnap.labels','pathsnap.eval'))]; print(bad); "
            "sys.exit(1 if bad else 0)") % (ROOT / "src")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
