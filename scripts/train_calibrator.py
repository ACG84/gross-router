"""Train the belief calibrator on shots sampled from the detector error model (true errors known).

  python scripts/train_calibrator.py --p 0.005 --shots 1200 --seed 101 --out results/calibrator.npz

Only shots where stage A does not converge are used: those are the shots the calibrator sees in
deployment. Both memory bases are pooled. Negatives far from the decision boundary are
subsampled and reweighted. Prints a reliability table on held-out shots.
"""
import argparse, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "kit"))
from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit
from decoders import relay_decoder
from router.correlated import JointModel
from router.learned import Calibrator, features, side_truth, side_truth_map


def collect(basis, p, shots, seed, rng):
    c = load_circuit(GROSS_CIRCUITS[basis], p)
    H, L, pri = dem_matrices(c, basis)
    jm = JointModel(c, basis)
    inst_side = side_truth_map(c, jm)
    det, _, err = c.detector_error_model().flattened().compile_sampler(seed=seed).sample(shots, return_errors=True)
    stage_a = relay_decoder(H, pri, num_sets=0, stop_nconv=1)
    leg = relay_decoder(jm.Hs, jm.side_priors, num_sets=0, stop_nconv=1)
    Xs, ys, ws, used = [], [], [], 0
    for k in range(shots):
        if stage_a.decode_detailed(det[k][jm.main_ids].astype(np.uint8)).success:
            continue
        used += 1
        sy = det[k][jm.side_ids].astype(np.uint8)
        r = leg.decode_detailed(sy)
        post = np.asarray(r.posterior_ratios)
        y = side_truth(err[k], inst_side, jm.n_side)
        X = features(post, r.success, jm.side_priors, sy.mean())
        easy = (post > 120) & (y == 0)                       # confident, correct negatives: keep 3%
        keep = ~easy | (rng.random(len(y)) < 0.03)
        Xs.append(X[keep]); ys.append(y[keep]); ws.append(np.where(easy[keep], 1 / 0.03, 1.0))
    return np.concatenate(Xs), np.concatenate(ys).astype(float), np.concatenate(ws), used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, nargs="+", default=[0.004, 0.005]); ap.add_argument("--shots", type=int, default=800)
    ap.add_argument("--seed", type=int, default=101); ap.add_argument("--out", default=str(ROOT / "results/calibrator.npz"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    parts, held = [], []
    for p in a.p:
        for i, b in enumerate("XZ"):
            parts.append(collect(b, p, a.shots, a.seed + i, rng))
            held.append(collect(b, p, max(a.shots // 4, 100), a.seed + 50 + i, rng))
            print(f"p={p} {b}: {parts[-1][3]} training shots reached stage S, {len(parts[-1][1])} rows, {int(parts[-1][1].sum())} fired", flush=True)
    X, y, w = (np.concatenate([q[j] for q in parts]) for j in range(3))
    cal = Calibrator().fit(X, y, w)
    Xh, yh, wh = (np.concatenate([q[j] for q in held]) for j in range(3))
    ph = cal.predict(Xh)
    raw = 1 / (1 + np.exp(np.clip(Xh[:, 1] * 40.0, -40, 40)))          # the leg's own posterior read as a probability
    ll = lambda q: float(-(wh * (yh * np.log(np.clip(q, 1e-12, 1)) + (1 - yh) * np.log(np.clip(1 - q, 1e-12, 1)))).sum() / wh.sum())
    print("weights", np.round(cal.w, 3))
    print(f"held-out log loss per column: calibrated {ll(ph):.5f}, raw BP posterior {ll(raw):.5f}, prior only {ll(np.exp(Xh[:, 4]) / (1 + np.exp(Xh[:, 4]))):.5f}")
    print("reliability (held-out): predicted bin -> mean predicted, observed rate, weighted count")
    edges = [0, 1e-4, 1e-3, 1e-2, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0000001]
    table = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ph >= lo) & (ph < hi)
        if m.any():
            row = (lo, hi, float((wh[m] * ph[m]).sum() / wh[m].sum()), float((wh[m] * yh[m]).sum() / wh[m].sum()), float(wh[m].sum()))
            table.append(row); print(f"  [{lo:g}, {hi:g}): {row[2]:.4f}  {row[3]:.4f}  {row[4]:.0f}")
    np.savez(a.out, w=cal.w, reliability=np.array(table), logloss=np.array([ll(ph), ll(raw)]), p=np.array(a.p), shots=a.shots)


if __name__ == "__main__":
    main()
