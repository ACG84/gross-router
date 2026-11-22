"""Decoding with heralded qubit loss: does reading the heralds help, and does routing still pay?

  python scripts/eval_loss.py --basis X --p 0.004 --p_loss 0.004 --shots 400 --seed 31 --out results/loss_X_0.004_s31.npz

Four decoders on the same shots:
  blind-kit      the kit's relay-BP, heralds unread (loss looks like extra Pauli noise)
  aware-kit      the kit's relay-BP with priors conditioned on the heralds of the shot
  blind-routed   our routed cross-basis decoder, heralds unread
  aware-routed   our routed cross-basis decoder with herald-conditioned priors
Columns of the saved table, per shot:
  0-3 fail (blind-kit, aware-kit, blind-routed, aware-routed), 4-7 seconds (same order),
  8 heralds fired, 9 stage of aware-routed (0 A, 1 M-conditioned, 2 M-plain), 10 stage of blind-routed
Decoder construction is counted in the time of every decoder that is rebuilt per shot.
"""
import argparse, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "kit"))
from circuit import GROSS_CIRCUITS, load_circuit, sample
from decoders import relay_decoder
from router.loss import LossModel, add_loss


def routed(lm, main, side, heralds, fixed=None):
    """Stages A, S, M with the priors of this shot. fixed = prebuilt (stage A, side, plain) decoders for the blind case."""
    pm, ps = lm.priors(heralds)
    a_dec = fixed[0] if fixed else relay_decoder(lm.H, pm, num_sets=0, stop_nconv=1)
    a = a_dec.decode_detailed(main)
    if a.success:
        return np.asarray(a.decoding, np.uint8), 0
    s = (fixed[1] if fixed else relay_decoder(lm.Hs, ps, stop_nconv=1)).decode_detailed(side)
    if s.success:
        cond, _ = lm.priors(heralds, side_fired=(np.asarray(s.decoding) > 0).astype(float))
        m = relay_decoder(lm.H, cond, stop_nconv=1).decode_detailed(main)
        if m.success:
            return np.asarray(m.decoding, np.uint8), 1
    m = (fixed[2] if fixed else relay_decoder(lm.H, pm, stop_nconv=1)).decode_detailed(main)
    return np.asarray(m.decoding, np.uint8), 2


def work(job):
    basis, p, p_loss, det, obs = job
    lm = LossModel(add_loss(load_circuit(GROSS_CIRCUITS[basis], p), p_loss), basis)
    pm, ps = lm.priors()
    kit = relay_decoder(lm.H, pm)
    fixed = (relay_decoder(lm.H, pm, num_sets=0, stop_nconv=1), relay_decoder(lm.Hs, ps, stop_nconv=1), relay_decoder(lm.H, pm, stop_nconv=1))
    fail = lambda e, k: bool(((lm.L @ e) % 2 != obs[k]).any())
    rows = []
    for k in range(len(det)):
        main, side, her = det[k, lm.main_ids], det[k, lm.side_ids], det[k, lm.herald_ids]
        t0 = time.perf_counter()
        e0 = np.asarray(kit.decode(main), np.uint8)
        t1 = time.perf_counter()
        e1 = np.asarray(relay_decoder(lm.H, lm.priors(her)[0]).decode(main), np.uint8)
        t2 = time.perf_counter()
        e2, st2 = routed(lm, main, side, None, fixed)
        t3 = time.perf_counter()
        e3, st3 = routed(lm, main, side, her)
        t4 = time.perf_counter()
        rows.append([fail(e0, k), fail(e1, k), fail(e2, k), fail(e3, k), t1 - t0, t2 - t1, t3 - t2, t4 - t3, her.sum(), st3, st2])
    return np.array(rows, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default="X"); ap.add_argument("--p", type=float, default=0.004)
    ap.add_argument("--p_loss", type=float, default=None, help="loss probability per data qubit per cycle (default: p)")
    ap.add_argument("--shots", type=int, default=200); ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--workers", type=int, default=os.cpu_count()); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p_loss = a.p if a.p_loss is None else a.p_loss
    det, obs = sample(add_loss(load_circuit(GROSS_CIRCUITS[a.basis], a.p), p_loss), a.shots, a.seed)
    with Pool(a.workers) as pool:
        R = np.concatenate(pool.map(work, [(a.basis, a.p, p_loss, det[k::a.workers], obs[k::a.workers]) for k in range(a.workers)]))
    np.savez_compressed(a.out, table=R, p=a.p, p_loss=p_loss, basis=a.basis, seed=a.seed)
    print(f"{a.basis} p={a.p} p_loss={p_loss} shots={len(R)} heralds/shot {R[:, 8].mean():.1f}")
    for i, name in enumerate(("blind-kit", "aware-kit", "blind-routed", "aware-routed")):
        ms = 1e3 * R[:, 4 + i]
        print(f"  {name:13s} fails {int(R[:, i].sum()):4d}  mean {ms.mean():7.1f} ms  median {np.median(ms):6.1f}  p99 {np.percentile(ms, 99):7.0f}")
    print(f"  aware-routed stages A/M-cond/M-plain: {[(R[:, 9] == s).mean().round(3) for s in (0, 1, 2)]}")


if __name__ == "__main__":
    main()
