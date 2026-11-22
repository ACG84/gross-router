"""Does the learned belief calibrator help the conditioned decode? Same shots for every variant.

  python scripts/eval_soft.py --basis X --p 0.005 --shots 300 --seed 61 --out results/soft/X_0.005_s61.npz

Stage A (one min-sum leg, plain priors) runs first, as in the routed decoder. For every shot it
does not settle, four ways of finishing are timed:
  plain   relay-BP (first solution) with the plain priors: no cross-basis information
  hard    relay-BP on the side syndrome, then condition on its hard answer (the routed decoder)
  raw     one min-sum leg on the side syndrome, condition on its posterior read as a probability
  learned one min-sum leg on the side syndrome, condition on the calibrator's beliefs
Each conditioned variant falls back to the plain priors if its main decode does not converge.
Columns per shot: 0 reached stage S, 1 stage-A fail flag, 2 stage-A seconds,
  3-6 fail (plain, hard, raw, learned), 7-10 seconds (same order, stage A not included),
  11-13 fell back to plain (hard, raw, learned)
"""
import argparse, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "kit"))
from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit, sample
from decoders import relay_decoder
from router.correlated import JointModel
from router.learned import Calibrator


def work(job):
    basis, p, det, obs, w, skip = job
    c = load_circuit(GROSS_CIRCUITS[basis], p)
    H, L, pri = dem_matrices(c, basis)
    jm = JointModel(c, basis)
    cal = Calibrator(w)
    stage_a = relay_decoder(H, pri, num_sets=0, stop_nconv=1)
    plain = relay_decoder(H, pri, stop_nconv=1)
    side_relay = relay_decoder(jm.Hs, jm.side_priors, stop_nconv=1)
    side_leg = relay_decoder(jm.Hs, jm.side_priors, num_sets=0, stop_nconv=1)
    fail = lambda e, k: float(((L @ np.asarray(e, np.uint8)) % 2 != obs[k]).any())

    def finish(main, belief):
        """Conditioned main decode with fallback. Returns (decoding, fell_back)."""
        if belief is not None:
            m = relay_decoder(H, jm.main_priors(belief), stop_nconv=1).decode_detailed(main)
            if m.success:
                return m.decoding, 0.0
        return plain.decode_detailed(main).decoding, 1.0

    rows = []
    for k in range(len(det)):
        main, side = det[k, jm.main_ids], det[k, jm.side_ids]
        t = time.perf_counter(); a = stage_a.decode_detailed(main); ta = time.perf_counter() - t
        if a.success:
            rows.append([0, fail(a.decoding, k), ta] + [0.0] * 11); continue
        t0 = time.perf_counter()
        e_plain = a.decoding if "plain" in skip else plain.decode_detailed(main).decoding
        t1 = time.perf_counter()
        s = side_relay.decode_detailed(side)
        e_hard, fb_hard = finish(main, (np.asarray(s.decoding) > 0).astype(float) if s.success else None)
        t2 = time.perf_counter()
        e_raw, fb_raw = a.decoding, 0.0
        if "raw" not in skip:
            leg = side_leg.decode_detailed(side); post = np.asarray(leg.posterior_ratios)
            e_raw, fb_raw = finish(main, 1 / (1 + np.exp(np.clip(post, -40, 40))))
        t3 = time.perf_counter()
        leg = side_leg.decode_detailed(side); post = np.asarray(leg.posterior_ratios)
        e_cal, fb_cal = finish(main, cal.beliefs(post, leg.success, jm.side_priors, side.mean()))
        t4 = time.perf_counter()
        rows.append([1, fail(a.decoding, k), ta, fail(e_plain, k), fail(e_hard, k), fail(e_raw, k), fail(e_cal, k),
                     t1 - t0, t2 - t1, t3 - t2, t4 - t3, fb_hard, fb_raw, fb_cal])
    return np.array(rows, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default="X"); ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--shots", type=int, default=300); ap.add_argument("--seed", type=int, default=61)
    ap.add_argument("--calibrator", default=str(ROOT / "results/calibrator.npz"))
    ap.add_argument("--workers", type=int, default=os.cpu_count()); ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--stop", type=int, default=None, help="decode shots [start, stop) of the sample")
    ap.add_argument("--skip", default="", help="comma list of variants not to run: plain, raw (their columns are then meaningless)")
    a = ap.parse_args()
    w = np.load(a.calibrator)["w"]
    det, obs = sample(load_circuit(GROSS_CIRCUITS[a.basis], a.p), a.shots, a.seed)     # always the full sample, so slices line up
    det, obs = det[a.start:a.stop], obs[a.start:a.stop]
    skip = set(filter(None, a.skip.split(",")))
    order = np.concatenate([np.arange(len(det))[k::a.workers] for k in range(a.workers)])
    with Pool(a.workers) as pool:
        parts = np.concatenate(pool.map(work, [(a.basis, a.p, det[k::a.workers], obs[k::a.workers], w, skip) for k in range(a.workers)]))
    R = np.empty_like(parts); R[order] = parts                                            # back to the order of the sample
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, table=R, p=a.p, basis=a.basis, seed=a.seed, start=a.start, skip=sorted(skip))
    S = R[R[:, 0] == 1]
    print(f"{a.basis} p={a.p}: {len(R)} shots, {len(S)} reached stage S")
    for i, name in enumerate(("plain", "hard", "raw", "learned")):
        ms = 1e3 * S[:, 7 + i]
        fb = "" if i == 0 else f"  fell back {S[:, 10 + i].mean():.3f}"
        print(f"  {name:8s} fails {int(S[:, 3 + i].sum()):3d}  mean {ms.mean():8.1f} ms  median {np.median(ms):7.1f}  p99 {np.percentile(ms, 99):8.0f}{fb}")


if __name__ == "__main__":
    main()
