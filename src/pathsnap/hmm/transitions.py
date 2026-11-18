"""Network distances between the candidates of consecutive GPS points (computed once per trip, cached).

For consecutive points t, t+1 in the same gap-free segment and every candidate pair (a, b):
  a == b : d_net = (s_b - s_a) * len(a)   (s = projection fraction); allowed only if >= -BACK_EPS_M
  a != b : d_net = (1 - s_a) * len(a) + SP(to(a) -> from(b)) + s_b * len(b)
SP = shortest directed path length on the trip's road graph restricted to arcs within SUBGRAPH_M of the
track, bounded by CUTOFF_M (unreachable -> inf). d_gps = straight-line distance between the two GPS fixes.
Uses only the track and road graph (no labels).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from ..geo import ProjTrip

SUBGRAPH_M = 400.0
CUTOFF_M = 2000.0
BACK_EPS_M = 15.0
CHUNK = 64


@dataclass
class TripLattice:
    point_idx: np.ndarray    # (T,) points in time order (valid points with candidates)
    cand_off: np.ndarray     # (T+1,) offsets into cand_arc
    cand_arc: np.ndarray     # (sum K,) arc ids
    pair_off: np.ndarray     # (T,) offset into dnet for pair (t, t+1); -1 if t+1 starts a new segment
    dnet: np.ndarray         # flat float32 Ka*Kb blocks, row-major (a over rows)
    d_gps: np.ndarray        # (T,) distance to next point (nan at segment ends)
    dt: np.ndarray           # (T,) time to next point

    def cands(self, t):
        return self.cand_arc[self.cand_off[t]:self.cand_off[t + 1]]

    def block(self, t):
        o = self.pair_off[t]
        if o < 0:
            return None
        ka = self.cand_off[t + 1] - self.cand_off[t]
        kb = self.cand_off[t + 2] - self.cand_off[t + 1]
        return self.dnet[o:o + ka * kb].reshape(ka, kb)

    def save(self, path):
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(**{k: z[k] for k in cls.__dataclass_fields__})


def build_lattice(pt: ProjTrip, point_idx: np.ndarray, arc_id: np.ndarray, proj_fraction: np.ndarray) -> TripLattice:
    """point_idx/arc_id/proj_fraction: candidate rows of one trip (any order)."""
    o = np.lexsort((arc_id, point_idx))
    point_idx, arc_id, proj_fraction = point_idx[o], arc_id[o], proj_fraction[o]
    pts, start = np.unique(point_idx, return_index=True)
    off = np.r_[start, len(point_idx)].astype(np.int64)
    T = len(pts)
    seg = pt.segment_id[pts]
    nxt_ok = np.r_[(np.diff(pts) >= 1) & (seg[1:] == seg[:-1]), False]
    d_gps = np.r_[np.hypot(*(pt.pxy[pts[1:]] - pt.pxy[pts[:-1]]).T), np.nan]
    dt = np.r_[np.diff(pt.t[pts]), np.nan]

    # restricted road subgraph around the track
    near = np.unique(pt.arc_tree.query(shapely.points(pt.pxy[pts]), predicate="dwithin", distance=SUBGRAPH_M)[1])
    near = np.union1d(near, np.unique(arc_id))
    fr, to = pt.arcs[near, 0], pt.arcs[near, 1]
    nodes, inv = np.unique(np.r_[fr, to], return_inverse=True)
    lf, lt = inv[:len(near)], inv[len(near):]
    n = len(nodes)
    w = np.maximum(pt.arc_len[near], 1e-3)
    G = csr_matrix((w, (lf, lt)), shape=(n, n))   # duplicate edges keep the sum; fine-tune: use min
    G.sum_duplicates()
    node_local = {int(v): i for i, v in enumerate(nodes)}
    to_loc = np.array([node_local[int(v)] for v in pt.arcs[arc_id, 1]])
    fr_loc = np.array([node_local[int(v)] for v in pt.arcs[arc_id, 0]])

    # shortest paths from every needed source (end node of a candidate at t that has a successor pair)
    need_src = np.unique(np.concatenate([to_loc[off[t]:off[t + 1]] for t in range(T) if nxt_ok[t]] or [np.empty(0, int)]))
    need_tgt = np.unique(fr_loc)
    src_row = {int(s): i for i, s in enumerate(need_src)}
    tgt_col = np.full(n, -1); tgt_col[need_tgt] = np.arange(len(need_tgt))
    SP = np.full((len(need_src), len(need_tgt)), np.inf, dtype=np.float32)
    for i in range(0, len(need_src), CHUNK):
        d = dijkstra(G, directed=True, indices=need_src[i:i + CHUNK], limit=CUTOFF_M)
        SP[i:i + CHUNK] = d[:, need_tgt]

    alen = pt.arc_len[arc_id]
    blocks, pair_off, pos = [], np.full(T, -1, np.int64), 0
    for t in range(T):
        if not nxt_ok[t]:
            continue
        ia = np.arange(off[t], off[t + 1]); ib = np.arange(off[t + 1], off[t + 2])
        a, b = arc_id[ia], arc_id[ib]
        sp = SP[[src_row[int(s)] for s in to_loc[ia]]][:, tgt_col[fr_loc[ib]]]
        d = (1 - proj_fraction[ia])[:, None] * alen[ia][:, None] + sp + (proj_fraction[ib] * alen[ib])[None, :]
        same = a[:, None] == b[None, :]
        along = (proj_fraction[ib][None, :] - proj_fraction[ia][:, None]) * alen[ia][:, None]
        d = np.where(same, np.where(along >= -BACK_EPS_M, np.maximum(along, 0), np.inf), d)
        blocks.append(d.astype(np.float32).ravel()); pair_off[t] = pos; pos += d.size
    dnet = np.concatenate(blocks) if blocks else np.empty(0, np.float32)
    return TripLattice(point_idx=pts, cand_off=off, cand_arc=arc_id, pair_off=pair_off, dnet=dnet, d_gps=d_gps, dt=dt)
