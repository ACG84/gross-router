import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "kit"))
from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit, sample
from router.correlated import JointModel
from router.decoder import RoutedDecoder


def test_joint_model_reproduces_kit_priors():
    c = load_circuit(GROSS_CIRCUITS["X"], 0.004)
    H, L, pri = dem_matrices(c, "X")
    jm = JointModel(c, "X")
    assert jm.n_main == H.shape[1]
    assert np.abs(jm.main_priors() - pri).max() < 1e-12


def test_conditioning_keeps_probabilities_valid():
    c = load_circuit(GROSS_CIRCUITS["Z"], 0.004)
    jm = JointModel(c, "Z")
    belief = np.zeros(jm.n_side); belief[:50] = 1.0
    p = jm.main_priors(belief)
    assert (p > 0).all() and (p < 0.5).all()


def test_decoder_returns_valid_corrections():
    c = load_circuit(GROSS_CIRCUITS["X"], 0.003)
    dec = RoutedDecoder(c, "X")
    det, obs = sample(c, 20, 3)
    ok = 0
    for d, o in zip(det, obs):
        e, info = dec.decode(d)
        assert ((dec.H @ e) % 2 == d[dec.jm.main_ids]).all() or info["stage"] != "A"
        ok += int(((dec.L @ e) % 2 == o).all())
    assert ok >= 19


# ---- heralded loss ----
from router.loss import LossModel, add_loss, herald_ids  # noqa: E402


def _columns(H, L, p):
    H, L = H.tocsc(), L.tocsc()
    return {(tuple(H.indices[H.indptr[k]:H.indptr[k + 1]]), tuple(L.indices[L.indptr[k]:L.indptr[k + 1]])): p[k] for k in range(H.shape[1])}


def test_zero_loss_circuit_has_the_kit_error_model():
    c = load_circuit(GROSS_CIRCUITS["X"], 0.004)
    lm = LossModel(add_loss(c, 0.0), "X")
    kit, ours = _columns(*dem_matrices(c, "X")), _columns(lm.H, lm.L, lm.priors()[0])
    assert set(kit) == set(ours)
    assert max(abs(kit[k] - ours[k]) for k in kit) < 1e-9


def test_one_herald_per_data_qubit_per_cycle_and_the_right_rate():
    c = add_loss(load_circuit(GROSS_CIRCUITS["Z"], 0.003), 0.02)
    ids = herald_ids(c)
    assert len(ids) == 144 * 12
    det, _ = sample(c, 400, 5)
    assert abs(det[:, ids].mean() - 0.02) < 0.002


def test_heralds_sharpen_the_priors():
    c = add_loss(load_circuit(GROSS_CIRCUITS["X"], 0.003), 0.02)
    lm = LossModel(c, "X")
    blind, _ = lm.priors()
    none_fired, _ = lm.priors(np.zeros(len(lm.herald_ids), np.uint8))
    one = np.zeros(len(lm.herald_ids), np.uint8); one[700] = 1
    one_fired, _ = lm.priors(one)
    assert (none_fired <= blind + 1e-12).all() and none_fired.sum() < blind.sum()      # unheralded locations are switched off
    assert (one_fired > 0.2).sum() >= 1 and (one_fired > 0.2).sum() <= 4                # the heralded location becomes likely
    assert ((one_fired > 0) & (one_fired < 0.5)).all()


def test_no_herald_priors_equal_the_lossless_circuit():
    base = load_circuit(GROSS_CIRCUITS["X"], 0.003)
    lm = LossModel(add_loss(base, 0.02), "X")
    quiet = _columns(lm.H, lm.L, lm.priors(np.zeros(len(lm.herald_ids), np.uint8))[0])
    kit = _columns(*dem_matrices(base, "X"))
    assert max(abs(kit[k] - quiet[k]) for k in kit) < 1e-6


# ---- learned calibration ----
from router.learned import Calibrator, features, side_truth, side_truth_map  # noqa: E402


def test_true_side_columns_reproduce_the_side_syndrome():
    c = load_circuit(GROSS_CIRCUITS["X"], 0.004)
    jm = JointModel(c, "X")
    inst_side = side_truth_map(c, jm)
    det, _, err = c.detector_error_model().flattened().compile_sampler(seed=7).sample(5, return_errors=True)
    for k in range(5):
        truth = side_truth(err[k], inst_side, jm.n_side)
        assert ((jm.Hs @ truth) % 2 == det[k][jm.side_ids]).all()


def test_calibrator_recovers_a_logistic_model():
    rng = np.random.default_rng(0)
    post = rng.normal(40, 80, 20000)
    X = features(post, True, np.full(20000, 0.003), 0.05)
    p = 1 / (1 + np.exp(-(-1.0 - 1.5 * X[:, 1])))
    y = (rng.random(20000) < p).astype(float)
    cal = Calibrator().fit(X, y, np.ones(20000))
    q = cal.predict(X)
    assert np.abs(q - p).mean() < 0.02
    assert ((q > 0) & (q < 1)).all()


def test_routed_decoder_with_calibrator_decodes():
    weights = np.load(ROOT / "results" / "calibrator.npz")["w"]
    c = load_circuit(GROSS_CIRCUITS["Z"], 0.004)
    dec = RoutedDecoder(c, "Z", calibrator=weights)
    det, obs = sample(c, 30, 4)
    ok = sum(int(((dec.L @ dec.decode(d)[0]) % 2 == o).all()) for d, o in zip(det, obs))
    assert ok >= 28
