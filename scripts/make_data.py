"""Generate training or test data: run all three stages on every shot and record features, outcomes, times.

  python scripts/make_data.py --basis X --p 0.004 --shots 4000 --seed 1 --out data/X_0.004_s1.npz

Use different seeds for training and test data. --workers splits the shots over processes.
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from router.stages import FEATURES_A, FEATURES_B, FEATURES_C, Problem, features_a, features_b


def run_slice(job):
    basis, p, det, obs = job
    prob = Problem(basis, p)
    rows = []
    for syndrome, truth in zip(det, obs):
        t0 = time.perf_counter()
        a = prob.stage_a.decode_detailed(syndrome)
        t1 = time.perf_counter()
        fa = features_a(prob, syndrome, a)
        b = prob.stage_b.decode_detailed(syndrome)
        t2 = time.perf_counter()
        fb = features_b(prob, b)
        c = prob.stage_c.decode_detailed(syndrome)
        t3 = time.perf_counter()
        fc = features_b(prob, c)
        ea, eb, ec = (np.asarray(r.decoding, np.uint8) for r in (a, b, c))
        la, lb, lc = prob.logical(ea), prob.logical(eb), prob.logical(ec)
        rows.append(fa + fb + fc + [a.success, (la != truth).any(), t1 - t0,
                               b.success, (lb != truth).any(), t2 - t1,
                               c.success, (lc != truth).any(), t3 - t2, c.iterations, (lb != lc).any()])
    return np.array(rows, float)


COLUMNS = FEATURES_A + FEATURES_B + FEATURES_C + ["a_ok", "a_fail", "a_sec", "b_ok", "b_fail", "b_sec",
                                     "c_ok", "c_fail", "c_sec", "c_iterations", "bc_differ"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basis", choices=["X", "Z"], default="X")
    ap.add_argument("--p", type=float, default=0.004)
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    det, obs = Problem(args.basis, args.p).sample(args.shots, args.seed)
    jobs = [(args.basis, args.p, det[k::args.workers], obs[k::args.workers]) for k in range(args.workers)]
    with Pool(args.workers) as pool:
        table = np.concatenate(pool.map(run_slice, jobs))
    np.savez_compressed(args.out, table=table, columns=COLUMNS, p=args.p, basis=args.basis, seed=args.seed)
    col = {n: table[:, i] for i, n in enumerate(COLUMNS)}
    print(args.out, len(table), "shots | stage A converged", col["a_ok"].mean().round(3),
          "| failures A(conv) B C:", int(col["a_fail"][col["a_ok"] == 1].sum()), int(col["b_fail"].sum()),
          int(col["c_fail"].sum()), "| mean ms A B C:", *(1e3 * col[k].mean().round(4) for k in ("a_sec", "b_sec", "c_sec")))


if __name__ == "__main__":
    main()
