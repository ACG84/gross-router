"""Paired X+Z summary of results/corr_{X,Z}_{p}.npz and the comparison plot.

A pair of shots fails when either memory basis fails (the kit's convention).
Rate per cycle = 1 - (1 - P)^(1/12).
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kit"))
from decoders import per_cycle_rate

rows = []
for p in ("0.003", "0.004", "0.005"):
    per = {}
    for b in "XZ":
        B = np.load(ROOT / f"results/corr_{b}_{p}.npz")["table"]; R = np.load(ROOT / f"results/afirst_{b}_{p}.npz")["table"]
        aok = R[:, 8] == 1; t = R[:, 2] + R[:, 3]
        per[b] = {"kit": (B[:, 1], B[:, 4]), "cross": (R[:, 0], t), "routed": (np.where(aok, R[:, 9], R[:, 0]), R[:, 10] + np.where(aok, 0, t))}
    out = {"p": float(p), "pairs": len(per["X"]["kit"][0])}
    for name in ("kit", "cross", "routed"):
        (fx, tx), (fz, tz) = per["X"][name], per["Z"][name]
        fail = (fx + fz) > 0; ms = 1e3 * (tx + tz); rate, err = per_cycle_rate(fail, 12)
        out[name] = dict(fx=int(fx.sum()), fz=int(fz.sum()), rate=rate, err=err, mean=ms.mean(), p50=np.median(ms), p99=np.percentile(ms, 99))
    rows.append(out)
print("| p | pairs | decoder | failures X, Z | rate per cycle | mean ms | median ms | p99 ms |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    for name, label in (("kit", "kit relay-BP"), ("cross", "cross-basis, always"), ("routed", "cross-basis, routed")):
        d = r[name]
        print(f"| {r['p']} | {r['pairs']} | {label} | {d['fx']}, {d['fz']} | {d['rate']:.1e} +- {d['err']:.0e} | {d['mean']:.0f} | {d['p50']:.0f} | {d['p99']:.0f} |")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ps = [r["p"] for r in rows]
for name, label, c in (("kit", "kit relay-BP", "#888"), ("cross", "cross-basis, always (ours)", "#7fb3e0"), ("routed", "cross-basis, routed (ours)", "#1f6fb2")):
    rate = np.array([max(r[name]["rate"], 5e-6) for r in rows]); err = np.array([r[name]["err"] for r in rows])
    ax[0].errorbar(ps, rate, yerr=err, marker="o", label=label, color=c, capsize=3)
    ax[1].plot(ps, [r[name]["mean"] for r in rows], marker="o", label=label, color=c)
ax[0].set(yscale="log", xlabel="physical error rate p", ylabel="logical error rate per cycle", title="Gross code [[144,12,12]], 12 cycles")
ax[1].set(yscale="log", xlabel="physical error rate p", ylabel="mean decode time per X+Z pair (ms)", title="Latency")
for a in ax: a.set_xticks(ps); a.legend(); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(ROOT / "results/cross_basis_vs_kit.png", dpi=150)
print("zero-failure points are drawn at 5e-6")
