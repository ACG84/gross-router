"""Paired X+Z summary of the learned-calibration runs (results/soft) next to the kit baseline.

The slices of eval_soft.py use the same sample (seed 21) as results/corr_{X,Z}_{p}.npz, so every
decoder is scored on the same shots. A pair fails when either basis fails.
Timing note: the kit baseline was timed in an earlier run; hard, raw and learned are timed
together inside eval_soft.py, so their comparison with each other is like for like.
"""
import glob, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kit"))
from decoders import per_cycle_rate

VARIANTS = (("hard", 4, 8, "routed, hard side answer"), ("raw", 5, 9, "routed, soft side beliefs (raw BP)"), ("learned", 6, 10, "routed, learned calibration"))
rows = []
for p in ("0.003", "0.004", "0.005"):
    per = {}
    for b in "XZ":
        files = sorted(glob.glob(str(ROOT / f"results/soft/{b}_{p}_s21_*.npz")), key=lambda f: int(np.load(f)["start"]))
        R = np.concatenate([np.load(f)["table"] for f in files])
        R = R[np.concatenate([np.arange(len(R))[k::2] for k in range(2)])]      # the baseline files were written by two workers: evens, then odds
        B = np.load(ROOT / f"results/corr_{b}_{p}.npz")["table"]; A = np.load(ROOT / f"results/afirst_{b}_{p}.npz")["table"]
        assert len(R) == len(B) and ((A[:, 8] == 0) == (R[:, 0] == 1)).all(), "slices do not line up with the baseline shots"
        s = R[:, 0] == 1
        per[b] = {"kit": (B[:, 1], B[:, 4]), "S": s}
        for name, fc, tc, _ in VARIANTS:
            per[b][name] = (np.where(s, R[:, fc], R[:, 1]), R[:, 2] + np.where(s, R[:, tc], 0.0), R[s][:, tc], R[s][:, fc])
    out = {"p": float(p), "pairs": len(per["X"]["kit"][0]), "stage_s": float(np.mean(np.concatenate([per["X"]["S"], per["Z"]["S"]])))}
    for name in ("kit", "hard", "raw", "learned"):
        (fx, tx, *_), (fz, tz, *_) = per["X"][name], per["Z"][name]
        fail = (fx + fz) > 0; ms = 1e3 * (tx + tz); rate, err = per_cycle_rate(fail, 12)
        out[name] = dict(fx=int(fx.sum()), fz=int(fz.sum()), rate=rate, err=err, mean=ms.mean(), p50=np.median(ms), p99=np.percentile(ms, 99))
        if name != "kit":
            st = 1e3 * np.concatenate([per["X"][name][2], per["Z"][name][2]])
            out[name].update(stage_mean=st.mean(), stage_p99=np.percentile(st, 99), stage_fail=int(per["X"][name][3].sum() + per["Z"][name][3].sum()))
    rows.append(out)

print("| p | pairs | decoder | failures X, Z | rate per cycle | mean ms | median ms | p99 ms |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    for name, label in (("kit", "kit relay-BP"),) + tuple((v[0], v[3]) for v in VARIANTS):
        d = r[name]
        print(f"| {r['p']} | {r['pairs']} | {label} | {d['fx']}, {d['fz']} | {d['rate']:.1e} +- {d['err']:.0e} | {d['mean']:.0f} | {d['p50']:.0f} | {d['p99']:.0f} |")
print()
print("Shots that stage A does not settle (the only ones the side information touches):")
print("| p | share of basis-shots | variant | failures | mean ms | p99 ms |")
print("|---|---|---|---|---|---|")
for r in rows:
    for name, _, _, label in VARIANTS:
        d = r[name]
        print(f"| {r['p']} | {r['stage_s']:.1%} | {label} | {d['stage_fail']} | {d['stage_mean']:.0f} | {d['stage_p99']:.0f} |")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
ps = [r["p"] for r in rows]
style = (("kit", "kit relay-BP", "#888888"), ("hard", "routed, hard side answer", "#7fb3e0"), ("learned", "routed, learned calibration", "#1f6fb2"))
for name, label, c in style:
    rate = np.array([max(r[name]["rate"], 5e-6) for r in rows]); err = np.array([r[name]["err"] for r in rows])
    ax[0].errorbar(ps, rate, yerr=err, marker="o", label=label, color=c, capsize=3)
    ax[1].plot(ps, [r[name]["mean"] for r in rows], marker="o", label=label, color=c)
    ax[2].plot(ps, [r[name]["p99"] for r in rows], marker="o", label=label, color=c)
ax[0].set(yscale="log", xlabel="physical error rate p", ylabel="logical error rate per cycle", title="Gross code [[144,12,12]], 12 cycles")
ax[1].set(yscale="log", xlabel="physical error rate p", ylabel="ms per X+Z pair", title="Mean latency")
ax[2].set(yscale="log", xlabel="physical error rate p", ylabel="ms per X+Z pair", title="99th-percentile latency")
for a in ax: a.set_xticks(ps); a.legend(fontsize=8); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(ROOT / "results/learned_calibration.png", dpi=150)
print("\nzero-failure points are drawn at 5e-6; plot: results/learned_calibration.png")
