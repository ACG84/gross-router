"""The three decoding stages and the features the gate reads.

Stage A  one min-sum leg with memory (80 iterations). Cheap. Most shots end here.
Stage B  relay-BP that stops at its first solution.
Stage C  relay-BP at the kit's setting: keeps the lightest of 5 solutions.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kit"))
from circuit import GROSS_CIRCUITS, dem_matrices, detector_ids, load_circuit, sample  # noqa: E402,F401
from decoders import relay_decoder  # noqa: E402

FEATURES_A = ["syndrome_weight", "a_unsatisfied", "a_weight", "a_cost", "a_low1", "a_low5", "a_q10", "a_q50"]
FEATURES_C = ["c_iterations_f", "c_weight", "c_cost", "c_low1", "c_low5", "c_min_flipped", "c_q10"]
FEATURES_B = ["b_iterations", "b_weight", "b_cost", "b_low1", "b_low5", "b_min_flipped", "b_q10"]


class Problem:
    """One memory experiment: matrices, priors and the three stage decoders."""

    def __init__(self, basis, p, circuit_path=None):
        self.basis, self.p = basis, p
        self.circuit = load_circuit(circuit_path or GROSS_CIRCUITS[basis], p)
        self.H, self.L, self.priors = dem_matrices(self.circuit, basis)
        self.Hr = self.H.tocsr()
        self.llr = np.log((1 - self.priors) / self.priors)
        self.det_ids = detector_ids(self.circuit, basis)
        self.stage_a = relay_decoder(self.H, self.priors, num_sets=0, stop_nconv=1)
        self.stage_b = relay_decoder(self.H, self.priors, stop_nconv=1)
        self.stage_c = relay_decoder(self.H, self.priors)

    def sample(self, shots, seed):
        det, obs = sample(self.circuit, shots, seed)
        return det[:, self.det_ids], obs

    def logical(self, error):
        return (self.L @ error) % 2


def _soft(posterior):
    """Summary of the posterior log-ratios: how many bits the decoder is unsure about."""
    a = np.abs(posterior)
    scale = np.median(a) + 1e-9          # the ratios grow with iteration count, so normalise
    r = a / scale
    return (r < 0.01).sum(), (r < 0.05).sum(), np.quantile(r, 0.10), np.quantile(r, 0.50)


def features_a(problem, syndrome, result):
    """Features after stage A (used when stage A did not converge)."""
    error = np.asarray(result.decoding, np.uint8)
    post = np.asarray(result.posterior_ratios)
    unsatisfied = int(((problem.Hr @ error) % 2 != syndrome).sum())
    low1, low5, q10, q50 = _soft(post)
    return [syndrome.sum(), unsatisfied, error.sum(), float(problem.llr @ error), low1, low5, q10, q50]


def features_b(problem, result):
    """Features of the first relay-BP solution."""
    error = np.asarray(result.decoding, np.uint8)
    post = np.asarray(result.posterior_ratios)
    low1, low5, q10, _ = _soft(post)
    flipped = np.abs(post[error == 1])
    scale = np.median(np.abs(post)) + 1e-9
    min_flipped = flipped.min() / scale if len(flipped) else 0.0
    return [result.iterations, error.sum(), float(problem.llr @ error), low1, low5, min_flipped, q10]
