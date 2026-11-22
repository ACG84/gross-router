import sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kit"))
from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit, sample
from decoders import relay_decoder
from router.correlated import JointModel
p, shots, floor = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
c = load_circuit(GROSS_CIRCUITS["X"], p)
H, L, pri = dem_matrices(c, "X")
jm = JointModel(c, "X")
assert jm.n_main == H.shape[1], (jm.n_main, H.shape)
print("main cols", jm.n_main, "side cols", jm.n_side, "prior check", np.abs(jm.main_priors() - pri).max(), flush=True)
det, obs = sample(c, shots, 5)
dm, ds = det[:, jm.main_ids], det[:, jm.side_ids]
side = relay_decoder(jm.Hs, jm.side_priors, stop_nconv=1)
base1, base5 = relay_decoder(H, pri, stop_nconv=1), relay_decoder(H, pri)
R = []
for k in range(shots):
    t0 = time.perf_counter(); s = side.decode_detailed(ds[k]); es = np.asarray(s.decoding, float)
    q = np.where(es > 0, 1.0, floor * jm.side_priors) if s.success else jm.side_priors
    pr2 = jm.main_priors(q); t1 = time.perf_counter()
    d1 = relay_decoder(H, pr2, stop_nconv=1); a = d1.decode_detailed(dm[k]); t2 = time.perf_counter()
    d5 = relay_decoder(H, pr2); a5 = d5.decode_detailed(dm[k]); t3 = time.perf_counter()
    b = base1.decode_detailed(dm[k]); t4 = time.perf_counter(); b5 = base5.decode_detailed(dm[k]); t5 = time.perf_counter()
    f = lambda r: ((L @ np.asarray(r.decoding, np.uint8)) % 2 != obs[k]).any()
    R.append([f(a), f(a5), f(b), f(b5), t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, a.iterations, b.iterations, s.success])
    if k % 100 == 99 or k == shots - 1:
        A = np.array(R, float)
        print(k + 1, "fails corr n1/n5", A[:, 0].sum(), A[:, 1].sum(), "| base n1/n5", A[:, 2].sum(), A[:, 3].sum(),
              "| ms side %.0f corr1 %.0f corr5 %.0f base1 %.0f base5 %.0f" % tuple(1e3 * A[:, 4:9].mean(0)),
              "| first-leg conv corr %.3f base %.3f" % ((A[:, 9] <= 80).mean(), (A[:, 10] <= 80).mean()), flush=True)
