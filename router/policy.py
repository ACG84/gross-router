"""Routing policies, replayed on a table made by scripts/make_data.py.

Every policy returns (failed, seconds) per shot. Times are the measured stage times.
Stages B and C restart from the syndrome, so a policy that runs A, then B, then C is
charged for all three. An integrated relay-BP would resume instead and be faster.
"""
import numpy as np

from .stages import FEATURES_A, FEATURES_B


def columns(npz):
    names = [str(n) for n in npz["columns"]]
    return {n: npz["table"][:, i] for i, n in enumerate(names)}


def feats(col, names):
    return np.column_stack([col[n] for n in names])


def only_b(col):
    return col["b_fail"].astype(bool), col["b_sec"]


def only_c(col):
    return col["c_fail"].astype(bool), col["c_sec"]


def rule_a_then(col, heavy):
    """No learning: accept stage A when it converges, otherwise run the heavy stage."""
    ok = col["a_ok"] == 1
    failed = np.where(ok, col["a_fail"], col[f"{heavy}_fail"]).astype(bool)
    return failed, col["a_sec"] + np.where(ok, 0.0, col[f"{heavy}_sec"])


def learned(col, gate, tau):
    """A, then B, then the gate decides from A and B features whether to pay for C."""
    ok = col["a_ok"] == 1
    risk = gate.risk(feats(col, FEATURES_A + FEATURES_B))
    escalate = ~ok & ((risk > tau) | (col["b_ok"] == 0))
    failed = np.where(ok, col["a_fail"], np.where(escalate, col["c_fail"], col["b_fail"])).astype(bool)
    seconds = col["a_sec"] + np.where(ok, 0.0, col["b_sec"] + np.where(escalate, col["c_sec"], 0.0))
    return failed, seconds, escalate.mean()
