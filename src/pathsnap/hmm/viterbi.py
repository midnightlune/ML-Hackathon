"""Viterbi decoding over a TripLattice in log space.

emission:   log e_t(c) = log( p_t(c) / sum_c' p_t(c') ) / tau        (classifier probability, renormalised)
transition: log T(a->b) = -|d_net(a,b) - d_gps| / beta ; -inf if d_net is inf (unreachable / backwards)
            or if d_net / dt > v_max (implausible speed)
break:      if t starts a new gap-free segment, or every transition into t is -inf, Viterbi restarts at t
            from the emissions alone (a break is counted only in the second case). Every point gets an arc.
Complexity O(T * K^2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transitions import TripLattice


@dataclass
class HMMParams:
    tau: float = 1.0
    beta: float = 10.0
    v_max: float = 60.0   # m/s


def decode(lat: TripLattice, prob: np.ndarray, prm: HMMParams) -> tuple[np.ndarray, int]:
    """prob: classifier probability per candidate, aligned with lat.cand_arc.
    Returns (chosen arc per point in lat.point_idx order, number of breaks)."""
    T = len(lat.point_idx)
    off = lat.cand_off
    p = np.clip(prob.astype(np.float64), 1e-12, None)
    back: list = [None] * T          # back[t][j] = best predecessor index; None = piece starts at t
    piece_end: dict[int, np.ndarray] = {}   # final scores of the piece ending at t (before a restart)
    breaks = 0

    def emis(t):
        q = p[off[t]:off[t + 1]]
        return np.log(q / q.sum()) / prm.tau

    score = emis(0)
    for t in range(1, T):
        e = emis(t)
        D = lat.block(t - 1)
        if D is None:                                    # gap -> new piece
            piece_end[t - 1] = score; score = e; continue
        bad = ~np.isfinite(D) | ((D / max(float(lat.dt[t - 1]), 1e-6)) > prm.v_max)
        trans = np.where(bad, -np.inf, -np.abs(D - lat.d_gps[t - 1]) / prm.beta)
        tot = score[:, None] + trans
        best = tot.argmax(0)
        val = tot[best, np.arange(tot.shape[1])]
        if not np.isfinite(val).any():                   # no feasible transition -> break
            breaks += 1; piece_end[t - 1] = score; score = e; continue
        back[t] = best
        score = np.where(np.isfinite(val), val, -np.inf) + e
    out = np.empty(T, np.int64)
    k = int(np.argmax(score))
    for t in range(T - 1, -1, -1):
        out[t] = lat.cand_arc[off[t] + k]
        if t == 0:
            break
        k = int(np.argmax(piece_end[t - 1])) if back[t] is None else int(back[t][k])
    return out, breaks
