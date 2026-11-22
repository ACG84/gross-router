"""Cross-basis prior update: use the other check type's syndrome to sharpen the priors.

The kit decodes X memory from the 936 X-check detectors and ignores the 792 Z-check
detectors. Many circuit faults (Y errors, two-qubit faults) flip detectors of both types.
Pass 1 decodes the Z-check syndrome. Pass 2 decodes the X-check syndrome with priors
conditioned on what pass 1 found.
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kit"))
from circuit import detector_ids  # noqa: E402


class JointModel:
    """The full detector error model, split into a main side and a side-information side."""

    def __init__(self, circuit, basis):
        other = "Z" if basis == "X" else "X"
        main_ids, side_ids = detector_ids(circuit, basis), detector_ids(circuit, other)
        main_row = {int(d): i for i, d in enumerate(main_ids)}
        side_row = {int(d): i for i, d in enumerate(side_ids)}
        self.main_ids, self.side_ids = main_ids, side_ids
        main_cols, side_cols, pairs = {}, {}, {}
        for inst in circuit.detector_error_model().flattened():
            if inst.type != "error":
                continue
            p = inst.args_copy()[0]
            t = inst.targets_copy()
            dm = tuple(sorted(main_row[x.val] for x in t if x.is_relative_detector_id() and x.val in main_row))
            ds = tuple(sorted(side_row[x.val] for x in t if x.is_relative_detector_id() and x.val in side_row))
            ob = tuple(sorted(x.val for x in t if x.is_logical_observable_id()))
            if (dm, ob) == ((), ()):
                mk = None
            else:
                mk = main_cols.setdefault((dm, ob), len(main_cols))
            sk = side_cols.setdefault(ds, len(side_cols)) if ds else None
            key = (mk, sk)
            pairs[key] = pairs.get(key, 0.0) * (1 - p) + p * (1 - pairs.get(key, 0.0))
        self.main_keys = list(main_cols)            # same order as kit's dem_matrices (insertion order)
        self.n_main, self.n_side = len(main_cols), len(side_cols)
        side_keys = list(side_cols)
        self.Hs = sp.csc_matrix((np.ones(sum(len(k) for k in side_keys), np.uint8),
                                 ([r for k in side_keys for r in k],
                                  [c for c, k in enumerate(side_keys) for _ in k])), shape=(len(side_ids), self.n_side))
        # side priors: probability that a side column fires (xor-combination of its mechanisms)
        ps = np.zeros(self.n_side)
        for (mk, sk), p in pairs.items():
            if sk is not None:
                ps[sk] = ps[sk] * (1 - p) + p * (1 - ps[sk])
        self.side_priors = ps
        # sparse lists: mechanisms with a main effect
        mk_, sk_, p_ = zip(*[(mk, -1 if sk is None else sk, p) for (mk, sk), p in pairs.items() if mk is not None])
        self.m_main, self.m_side, self.m_p = np.array(mk_), np.array(sk_), np.array(p_)

    def main_priors(self, side_fired_prob=None):
        """Priors of the main columns. side_fired_prob[k] = belief that side column k fired (None = unconditioned)."""
        if side_fired_prob is None:
            w = self.m_p
        else:
            has = self.m_side >= 0
            ratio = np.ones(len(self.m_p))
            ratio[has] = side_fired_prob[self.m_side[has]] / self.side_priors[self.m_side[has]]
            w = np.clip(self.m_p * ratio, 0.0, 0.499)
        # combine mechanisms that share a main column: 1 - prod(1 - 2w) trick for xor
        logt = np.log1p(-2 * w)
        s = np.bincount(self.m_main, weights=logt, minlength=self.n_main)
        return np.clip((1 - np.exp(s)) / 2, 1e-9, 0.499)
