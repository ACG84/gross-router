"""The routed cross-basis decoder as one object.

  dec = RoutedDecoder(circuit, basis)          # circuit: a kit memory circuit at some noise strength
  error, info = dec.decode(detectors)          # detectors: all detector bits of one shot (1728 for the Gross code)
  logical = (dec.L @ error) % 2

Stages
  A  one min-sum leg on the main-basis syndrome with the plain priors. Converged -> accept.
  S  relay-BP (first solution) on the other check type's syndrome.
  M  relay-BP (first solution) on the main-basis syndrome with priors conditioned on S.
     If S did not converge, or M does not converge with the conditioned priors, M runs with the plain priors.

With calibrator=<weights from scripts/train_calibrator.py> stage S is one min-sum leg and its
posteriors go through the learned calibrator (router/learned.py); the soft beliefs are used
whether or not the leg converged. This is the fastest variant in our tests.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kit"))
from circuit import dem_matrices  # noqa: E402
from decoders import relay_decoder  # noqa: E402

from .correlated import JointModel  # noqa: E402
from .learned import Calibrator  # noqa: E402


class RoutedDecoder:
    def __init__(self, circuit, basis, route=True, floor=0.0, side_settings=None, calibrator=None):
        self.H, self.L, self.priors = dem_matrices(circuit, basis)
        self.jm = JointModel(circuit, basis)
        self.route, self.floor = route, floor
        self.stage_a = relay_decoder(self.H, self.priors, num_sets=0, stop_nconv=1)
        self.side = relay_decoder(self.jm.Hs, self.jm.side_priors, stop_nconv=1, **(side_settings or {}))
        self.plain = relay_decoder(self.H, self.priors, stop_nconv=1)
        self.calibrator = None if calibrator is None else Calibrator(calibrator)
        self.side_leg = relay_decoder(self.jm.Hs, self.jm.side_priors, num_sets=0, stop_nconv=1)

    def decode(self, detectors):
        main, side = detectors[self.jm.main_ids], detectors[self.jm.side_ids]
        info, t0 = {}, time.perf_counter()
        if self.route:
            a = self.stage_a.decode_detailed(main)
            info["a_sec"] = time.perf_counter() - t0
            if a.success:
                info["stage"] = "A"
                return np.asarray(a.decoding, np.uint8), info
        t1 = time.perf_counter()
        if self.calibrator is not None:
            s = self.side_leg.decode_detailed(side)
            belief = self.calibrator.beliefs(np.asarray(s.posterior_ratios), s.success, self.jm.side_priors, side.mean())
        else:
            s = self.side.decode_detailed(side)
            belief = np.where(np.asarray(s.decoding) > 0, 1.0, self.floor * self.jm.side_priors) if s.success else None
        info["side_sec"], info["side_ok"] = time.perf_counter() - t1, bool(s.success)
        t2 = time.perf_counter()
        m = None
        if belief is not None:
            m = relay_decoder(self.H, self.jm.main_priors(belief), stop_nconv=1).decode_detailed(main)
        if m is None or not m.success:
            m = self.plain.decode_detailed(main)
            info["stage"] = "M-plain"
        else:
            info["stage"] = "M-conditioned"
        info["main_sec"] = time.perf_counter() - t2
        return np.asarray(m.decoding, np.uint8), info
