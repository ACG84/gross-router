"""Learned calibration of the side decoder's beliefs.

The routed decoder conditions the main priors on the side decoder's hard answer, and only when
the side decoder converged. Here one cheap min-sum leg runs on the side syndrome and a small
logistic model turns its per-column posterior into a calibrated probability that the column
fired. The model is trained on shots sampled from the detector error model, where the true
set of fired mechanisms is known. Calibrated beliefs can be used whether or not the leg converged.
"""
import numpy as np

from .correlated import JointModel


def side_truth_map(circuit, jm: JointModel):
    """For every error instruction of the flattened detector error model: its side column, or -1."""
    row = {int(d): i for i, d in enumerate(jm.side_ids)}
    cols, out = {}, []
    for inst in circuit.detector_error_model().flattened():
        if inst.type != "error":
            continue
        ds = tuple(sorted(row[x.val] for x in inst.targets_copy() if x.is_relative_detector_id() and x.val in row))
        out.append(cols.setdefault(ds, len(cols)) if ds else -1)
    assert len(cols) == jm.n_side
    return np.array(out)


def side_truth(errors, inst_side, n_side):
    """Which side columns truly fired in one shot (parity of the fired instructions per column)."""
    fired = np.nonzero(errors)[0]
    fired = fired[inst_side[fired] >= 0]
    return (np.bincount(inst_side[fired], minlength=n_side) % 2).astype(np.uint8)


def features(posterior, converged, prior, weight_frac):
    """Per-column features. posterior: relay-BP posterior ratios of one leg (negative = fired)."""
    r = np.clip(np.asarray(posterior) / 40.0, -8.0, 8.0)
    conv = float(converged)
    lp = np.log(prior / (1 - prior))
    return np.stack([np.ones_like(r), r, r * conv, np.full_like(r, conv), lp, np.full_like(r, weight_frac),
                     (r < 0).astype(float), np.minimum(r, 0.0) ** 2], axis=1)


class Calibrator:
    """Logistic regression on the features above (weights stored as a plain array)."""

    def __init__(self, w=None):
        self.w = None if w is None else np.asarray(w, float)

    def fit(self, X, y, sample_weight, iters=60, ridge=1e-3):
        w = np.zeros(X.shape[1])
        for _ in range(iters):                       # Newton's method
            p = 1 / (1 + np.exp(-X @ w))
            g = X.T @ (sample_weight * (p - y)) + ridge * w
            Hm = (X * (sample_weight * p * (1 - p))[:, None]).T @ X + ridge * np.eye(len(w))
            step = np.linalg.solve(Hm, g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.w = w
        return self

    def predict(self, X):
        return 1 / (1 + np.exp(-np.clip(X @ self.w, -40, 40)))

    def beliefs(self, posterior, converged, prior, weight_frac):
        return self.predict(features(posterior, converged, prior, weight_frac))
