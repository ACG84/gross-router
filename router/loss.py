"""Heralded qubit loss for the kit's memory circuits, and priors conditioned on the heralds.

Loss model: before every syndrome cycle each data qubit is lost with probability p_loss. A lost
qubit is detected (heralded) and replaced by a fresh one in the maximally mixed state, which is
Stim's HERALDED_ERASE channel. This is the usual erasure picture of atom loss with replacement;
it does not model a lost atom staying lost through later gates.

A decoder that ignores the heralds sees extra Pauli noise of unknown location. A decoder that
reads them knows where each erasure happened: the mechanisms of a heralded location become
likely (probability about 1/4 each) and the mechanisms of every location that did not herald
are switched off.
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import stim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kit"))
from circuit import detector_ids  # noqa: E402

HERALD_COORDINATE = 2      # the kit uses 0 for X-check detectors and 1 for Z-check detectors


def _flat(circuit):
    return [inst for inst in circuit.flattened()]


def add_loss(circuit, p_loss):
    """Copy of a kit memory circuit with heralded loss on the data qubits before every cycle.

    Every herald becomes a detector with coordinate HERALD_COORDINATE, appended after the kit's
    detectors would be ambiguous, so heralds are emitted in place and the kit's detectors keep
    their coordinates; use detector_ids / herald_ids to find each group.
    """
    flat = _flat(circuit)
    measures = [i for i, inst in enumerate(flat) if stim.gate_data(inst.name).produces_measurements]
    data = [t.value for t in flat[measures[-1]].targets_copy()]          # final data readout
    resets = [i for i, inst in enumerate(flat) if inst.name in ("R", "RX")]
    data_set = set(data)
    # a cycle starts at an ancilla reset that follows a measurement or the data preparation
    starts, seen_meas = [], True
    for i, inst in enumerate(flat):
        if stim.gate_data(inst.name).produces_measurements:
            seen_meas = True
        elif inst.name in ("R", "RX") and seen_meas and not (set(t.value for t in inst.targets_copy()) & data_set):
            starts.append(i)
            seen_meas = False
    del resets
    out = stim.Circuit()
    new_index, new_count, cycle = [], 0, 0          # new_index[k] = position of original record k
    starts = set(starts)
    for i, inst in enumerate(flat):
        if i in starts:
            out.append("HERALDED_ERASE", data, p_loss)
            new_count += len(data)
            for j, q in enumerate(data):
                out.append("DETECTOR", [stim.target_rec(j - len(data))], [HERALD_COORDINATE, cycle, q])
            cycle += 1
        name = inst.name
        if stim.gate_data(name).produces_measurements:
            out.append(inst)
            for _ in inst.targets_copy():
                new_index.append(new_count)
                new_count += 1
        elif name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            targets = []
            for t in inst.targets_copy():
                if t.is_measurement_record_target:
                    targets.append(stim.target_rec(new_index[len(new_index) + t.value] - new_count))
                else:
                    targets.append(t)
            out.append(name, targets, inst.gate_args_copy())
        else:
            out.append(inst)
    return out


def herald_ids(circuit):
    coordinates = circuit.get_detector_coordinates()
    return np.array([d for d in range(circuit.num_detectors) if coordinates[d] and coordinates[d][0] == HERALD_COORDINATE])


def _xor_combine(index, p, n):
    """Probability that an odd number of the independent mechanisms sharing an index fire."""
    s = np.bincount(index, weights=np.log1p(-2 * np.clip(p, 0.0, 0.4999999)), minlength=n)
    return (1 - np.exp(s)) / 2


class LossModel:
    """Detector error model of a lossy circuit, split into main checks, side checks and heralds."""

    def __init__(self, circuit, basis):
        other = "Z" if basis == "X" else "X"
        self.main_ids, self.side_ids, self.herald_ids = detector_ids(circuit, basis), detector_ids(circuit, other), herald_ids(circuit)
        row = [{int(d): i for i, d in enumerate(ids)} for ids in (self.main_ids, self.side_ids, self.herald_ids)]
        main_cols, side_cols, her_cols, trip = {}, {}, {}, {}
        # HERALDED_ERASE is a disjoint channel; Stim needs this flag to turn it into independent mechanisms
        for inst in circuit.detector_error_model(approximate_disjoint_errors=True).flattened():
            if inst.type != "error":
                continue
            p, t = inst.args_copy()[0], inst.targets_copy()
            dets = [x.val for x in t if x.is_relative_detector_id()]
            dm, ds, dh = (tuple(sorted(r[d] for d in dets if d in r)) for r in row)
            ob = tuple(sorted(x.val for x in t if x.is_logical_observable_id()))
            mk = main_cols.setdefault((dm, ob), len(main_cols)) if (dm, ob) != ((), ()) else -1
            sk = side_cols.setdefault(ds, len(side_cols)) if ds else -1
            hk = her_cols.setdefault(dh, len(her_cols)) if dh else -1
            key = (mk, sk, hk)
            q = trip.get(key, 0.0)
            trip[key] = q * (1 - p) + p * (1 - q)
        keys = np.array(list(trip), int)
        self.t_main, self.t_side, self.t_her, self.t_p = keys[:, 0], keys[:, 1], keys[:, 2], np.array(list(trip.values()))
        self.n_main, self.n_side, self.n_her = len(main_cols), len(side_cols), len(her_cols)
        self.H = self._matrix([k[0] for k in main_cols], len(self.main_ids))
        self.L = self._matrix([k[1] for k in main_cols], circuit.num_observables).tocsr()
        self.Hs = self._matrix(list(side_cols), len(self.side_ids))
        self.her_keys = list(her_cols)
        has = self.t_her >= 0
        self.her_prior = _xor_combine(self.t_her[has], self.t_p[has], self.n_her) if self.n_her else np.zeros(0)
        # (main, side) pairs, for the cross-basis conditioning
        pair = {}
        for m, s in zip(self.t_main, self.t_side):
            pair.setdefault((int(m), int(s)), len(pair))
        self.t_pair = np.array([pair[(int(m), int(s))] for m, s in zip(self.t_main, self.t_side)])
        pk = np.array(list(pair), int)
        self.p_main, self.p_side, self.n_pair = pk[:, 0], pk[:, 1], len(pair)

    @staticmethod
    def _matrix(id_lists, n_rows):
        rows = [i for ids in id_lists for i in ids]
        cols = [c for c, ids in enumerate(id_lists) for _ in ids]
        return sp.csc_matrix((np.ones(len(rows), np.uint8), (rows, cols)), shape=(n_rows, len(id_lists)))

    def mechanism_probs(self, heralds=None):
        """Probability of every mechanism. heralds: the herald bits of one shot (None = heralds unread)."""
        if heralds is None or not self.n_her:
            return self.t_p
        fired = np.array([all(heralds[d] for d in key) for key in self.her_keys], float)
        p = self.t_p.copy()
        has = self.t_her >= 0
        p[has] = np.clip(self.t_p[has] * fired[self.t_her[has]] / self.her_prior[self.t_her[has]], 0.0, 0.4999)
        return p

    def priors(self, heralds=None, side_fired=None):
        """(main priors, side priors) of one shot.

        heralds: herald bits, or None for a loss-blind decoder.
        side_fired: belief that each side column fired (the side decoder's answer), or None.
        """
        p = self.mechanism_probs(heralds)
        pair_p = _xor_combine(self.t_pair, p, self.n_pair)
        has_s = self.p_side >= 0
        side = np.clip(_xor_combine(self.p_side[has_s], pair_p[has_s], self.n_side), 1e-9, 0.499)
        w = pair_p.copy()
        if side_fired is not None:
            w[has_s] = np.clip(pair_p[has_s] * side_fired[self.p_side[has_s]] / side[self.p_side[has_s]], 0.0, 0.4999)
        has_m = self.p_main >= 0
        main = np.clip(_xor_combine(self.p_main[has_m], w[has_m], self.n_main), 1e-9, 0.499)
        return main, side
