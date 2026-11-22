"""Compare the cross-basis decoder with the kit's relay-BP on the same shots.

  python scripts/eval_corr.py --p 0.005 --shots 600 --basis X --workers 2 --out results/corr_X_0.005.npz
Columns: corr_fail, base_fail, side_sec, corr_sec, base_sec, side_ok, corr_iter, base_iter
"""
import argparse, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "kit"))
from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit, sample
from decoders import relay_decoder, per_cycle_rate
from router.correlated import JointModel


def work(job):
    basis, p, det, obs, floor, nconv, side_sets, skip_base = job
    c = load_circuit(GROSS_CIRCUITS[basis], p)
    H, L, pri = dem_matrices(c, basis)
    jm = JointModel(c, basis)
    side = relay_decoder(jm.Hs, jm.side_priors, stop_nconv=1, **({} if side_sets < 0 else {'num_sets': side_sets}))
    base = relay_decoder(H, pri)
    stage_a = relay_decoder(H, pri, num_sets=0, stop_nconv=1)
    Hr = H.tocsr()
    dm, ds = det[:, jm.main_ids], det[:, jm.side_ids]
    rows = []
    for k in range(len(det)):
        ta = time.perf_counter()
        A = stage_a.decode_detailed(dm[k]); eA = np.asarray(A.decoding, np.uint8); pa = np.abs(np.asarray(A.posterior_ratios)); sc = np.median(pa) + 1e-9
        afeat = [dm[k].sum(), ds[k].sum(), A.iterations, eA.sum(), (pa / sc < 0.05).sum(), np.quantile(pa / sc, 0.1), (pa[eA == 1] / sc).min() if eA.any() else 0.0]
        t0 = time.perf_counter()
        s = side.decode_detailed(ds[k])
        q = np.where(np.asarray(s.decoding) > 0, 1.0, floor * jm.side_priors) if s.success else None
        t1 = time.perf_counter()
        a = relay_decoder(H, jm.main_priors(q), stop_nconv=nconv).decode_detailed(dm[k])
        if not a.success and q is not None:                     # conditioning was wrong: fall back
            a = relay_decoder(H, pri, stop_nconv=nconv).decode_detailed(dm[k])
        t2 = time.perf_counter()
        b = a if skip_base else base.decode_detailed(dm[k])
        t3 = time.perf_counter()
        f = lambda r: bool(((L @ np.asarray(r.decoding, np.uint8)) % 2 != obs[k]).any())
        rows.append([f(a), f(b), t1 - t0, t2 - t1, t3 - t2, s.success, a.iterations, b.iterations, A.success, f(A), t0 - ta] + [float(x) for x in afeat])
    return np.array(rows, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default="X"); ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--shots", type=int, default=400); ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--floor", type=float, default=0.0); ap.add_argument("--nconv", type=int, default=1)
    ap.add_argument("--side_sets", type=int, default=-1); ap.add_argument("--skip_base", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2)); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = load_circuit(GROSS_CIRCUITS[a.basis], a.p)
    det, obs = sample(c, a.shots, a.seed)
    jobs = [(a.basis, a.p, det[k::a.workers], obs[k::a.workers], a.floor, a.nconv, a.side_sets, a.skip_base) for k in range(a.workers)]
    with Pool(a.workers) as pool:
        R = np.concatenate(pool.map(work, jobs))
    np.savez_compressed(a.out, table=R, p=a.p, basis=a.basis, floor=a.floor, nconv=a.nconv)
    print(f"{a.basis} p={a.p} shots={len(R)} floor={a.floor} nconv={a.nconv}")
    print(f"  cross-basis  fails {int(R[:,0].sum()):4d}  mean {1e3*(R[:,2]+R[:,3]).mean():7.1f} ms (side {1e3*R[:,2].mean():.1f} + main {1e3*R[:,3].mean():.1f})  p99 {1e3*np.percentile(R[:,2]+R[:,3],99):.0f} ms")
    print(f"  kit relay-BP fails {int(R[:,1].sum()):4d}  mean {1e3*R[:,4].mean():7.1f} ms  p99 {1e3*np.percentile(R[:,4],99):.0f} ms")
    print(f"  side decode converged {R[:,5].mean():.3f}")


if __name__ == "__main__":
    main()
