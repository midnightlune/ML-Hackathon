import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.hmm.transitions import TripLattice  # noqa: E402
from pathsnap.hmm.viterbi import HMMParams, decode  # noqa: E402


def _lattice(blocks, cands, d_gps, dt):
    T = len(cands)
    off = np.r_[0, np.cumsum([len(c) for c in cands])]
    pair_off, flat, pos = np.full(T, -1), [], 0
    for t, b in enumerate(blocks):
        if b is not None:
            pair_off[t] = pos; flat.append(np.asarray(b, np.float32).ravel()); pos += np.size(b)
    return TripLattice(point_idx=np.arange(T), cand_off=off, cand_arc=np.concatenate(cands), pair_off=pair_off,
                       dnet=np.concatenate(flat), d_gps=np.asarray(d_gps, float), dt=np.asarray(dt, float))


def test_hmm_overrides_noisy_emission():
    # 3 points, candidates {10, 20}; arc 20 is unreachable from 10 and vice versa (inf), 10->10 consistent.
    inf = np.inf
    blk = [[10, inf], [inf, 10]]
    lat = _lattice([blk, blk, None], [np.array([10, 20])] * 3, [10, 10, np.nan], [1, 1, np.nan])
    prob = np.array([0.9, 0.1, 0.4, 0.6, 0.9, 0.1])     # classifier-only would pick 20 at t=1
    arcs, breaks = decode(lat, prob, HMMParams(tau=1, beta=10, v_max=100))
    assert arcs.tolist() == [10, 10, 10] and breaks == 0


def test_break_and_restart():
    inf = np.inf
    # t0 -> t1: every transition impossible (break); t1 -> t2 normal
    lat = _lattice([[[inf, inf], [inf, inf]], [[5, inf], [inf, 5]], None], [np.array([1, 2])] * 3,
                   [5, 5, np.nan], [1, 1, np.nan])
    prob = np.array([0.2, 0.8, 0.7, 0.3, 0.6, 0.4])
    arcs, breaks = decode(lat, prob, HMMParams(tau=1, beta=10, v_max=100))
    assert breaks == 1
    assert arcs[0] == 2            # first piece decoded on its own
    assert arcs.tolist()[1:] == [1, 1]
    assert len(arcs) == 3          # never an empty route
