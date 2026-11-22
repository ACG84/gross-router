"""The learned gate: logistic regression on first-stage soft information.

It predicts the probability that the cheap answer is logically wrong. The router
spends the expensive stage only on shots whose risk is above a threshold.
"""
import json

import numpy as np


class Gate:
    def __init__(self, names):
        self.names, self.mu, self.sd, self.w, self.b = list(names), None, None, None, 0.0

    @staticmethod
    def _squash(X):
        return np.log1p(np.abs(X)) * np.sign(X)      # features span orders of magnitude

    def fit(self, X, y, steps=3000, lr=0.3, l2=1e-2):
        Z = self._squash(np.asarray(X, float))
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-9
        Z = (Z - self.mu) / self.sd
        y = np.asarray(y, float)
        pos = max(y.mean(), 1e-6)
        weight = np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos))   # balance the rare positives
        self.w, self.b = np.zeros(Z.shape[1]), 0.0
        for _ in range(steps):
            q = 1 / (1 + np.exp(-(Z @ self.w + self.b)))
            g = weight * (q - y)
            self.w -= lr * (Z.T @ g / len(y) + l2 * self.w)
            self.b -= lr * g.mean()
        return self

    def risk(self, X):
        Z = (self._squash(np.atleast_2d(np.asarray(X, float))) - self.mu) / self.sd
        return 1 / (1 + np.exp(-(Z @ self.w + self.b)))

    def save(self, path):
        json.dump({"names": self.names, "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                   "w": self.w.tolist(), "b": float(self.b)}, open(path, "w"), indent=1)

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        g = cls(d["names"])
        g.mu, g.sd, g.w, g.b = np.array(d["mu"]), np.array(d["sd"]), np.array(d["w"]), d["b"]
        return g


def auc(score, y):
    """Area under the ROC curve (probability a positive outranks a negative)."""
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    return (ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
