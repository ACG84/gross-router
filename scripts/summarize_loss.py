"""Paired X+Z summary of the heralded-loss runs (results/loss) and a plot.

  python scripts/summarize_loss.py [p_loss]          # default 0.02
A pair of shots fails when either memory basis fails. Rate per cycle = 1 - (1 - P)^(1/12).
"""
import glob, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kit"))
from decoders import per_cycle_rate

loss = sys.argv[1] if len(sys.argv) > 1 else "0.02"
NAMES = ("kit relay-BP, heralds unread", "kit relay-BP, heralds read", "routed, heralds unread", "routed, heralds read")
rows = []
for p in ("0.003", "0.004", "0.005"):
    T = {}
    for b in "XZ":
        files = sorted(glob.glob(str(ROOT / f"results/loss/{b}_{p}_l{loss}_s*.npz")))
        T[b] = np.concatenate([np.load(f)["table"] for f in files]) if files else np.zeros((0, 11))
    n = min(len(T["X"]), len(T["Z"]))
    if not n:
        continue
    X, Z = T["X"][:n], T["Z"][:n]
    out = {"p": float(p), "pairs": n, "heralds": float((X[:, 8].mean() + Z[:, 8].mean()) / 2), "stage_a": float(((X[:, 9] == 0).mean() + (Z[:, 9] == 0).mean()) / 2)}
    for i, name in enumerate(NAMES):
        fail = (X[:, i] + Z[:, i]) > 0; ms = 1e3 * (X[:, 4 + i] + Z[:, 4 + i]); rate, err = per_cycle_rate(fail, 12)
        out[name] = dict(fx=int(X[:, i].sum()), fz=int(Z[:, i].sum()), rate=rate, err=err, mean=ms.mean(), p50=np.median(ms), p99=np.percentile(ms, 99))
    rows.append(out)

print(f"Heralded loss, {float(loss):.0%} per data qubit per cycle.")
print("| p | pairs | heralds per shot | decoder | failures X, Z | rate per cycle | mean ms | median ms | p99 ms |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    for name in NAMES:
        d = r[name]
        print(f"| {r['p']} | {r['pairs']} | {r['heralds']:.0f} | {name} | {d['fx']}, {d['fz']} | {d['rate']:.1e} +- {d['err']:.0e} | {d['mean']:.0f} | {d['p50']:.0f} | {d['p99']:.0f} |")
print("\nShare of basis-shots the herald-aware routed decoder ends in stage A:", {r["p"]: round(r["stage_a"], 3) for r in rows})

fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
ps = [r["p"] for r in rows]
for name, c, ls in zip(NAMES, ("#888888", "#888888", "#1f6fb2", "#1f6fb2"), (":", "-", ":", "-")):
    rate = np.array([max(r[name]["rate"], 2e-5) for r in rows]); err = np.array([r[name]["err"] for r in rows])
    ax[0].errorbar(ps, rate, yerr=err, marker="o", label=name, color=c, ls=ls, capsize=3)
    ax[1].plot(ps, [r[name]["mean"] for r in rows], marker="o", label=name, color=c, ls=ls)
ax[0].set(yscale="log", xlabel="physical error rate p", ylabel="logical error rate per cycle", title=f"Gross code with {float(loss):.0%} heralded loss per cycle")
ax[1].set(yscale="log", xlabel="physical error rate p", ylabel="mean decode time per X+Z pair (ms)", title="Latency")
for a in ax: a.set_xticks(ps); a.legend(fontsize=7.5); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(ROOT / "results/loss_heralds.png", dpi=150)
print("zero-failure points are drawn at 2e-5; plot: results/loss_heralds.png")
