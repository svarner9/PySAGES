"""The mlmd backend: bias weight, method update, callbacks, observers, resume.

Runs on the CPU with analytic potentials; needs `mlmd` importable.
"""
import numpy as np
import pytest

mlmd = pytest.importorskip("mlmd")

import pysages  # noqa: E402
from pysages.colvars import Component  # noqa: E402
from pysages.methods import HarmonicBias, SpectralABF  # noqa: E402

from mlmd import Langevin, System, units  # noqa: E402
from mlmd.calculators import Harmonic  # noqa: E402

mlmd.enable_float64()


def harmonic_particles(n, k, mass, dt=1.0, temperature=300.0, friction=0.05, seed=0):
    system = System(np.zeros((n, 3)), [2] * n, Harmonic(k, np.zeros((1, 3)), dtype="float64"),
                    masses=[mass] * n)
    return Langevin(system, timestep=dt, temperature=temperature, friction=friction, seed=seed)


class CVLog:
    def __init__(self):
        self.steps, self.xi = [], []

    def __call__(self, snapshot, state, step):
        self.steps.append(step)
        self.xi.append(np.asarray(state.xi).ravel())


def test_harmonic_bias_acts_with_full_weight():
    """A harmonic bias k_b on x of every particle in a harmonic well k: the biased distribution is
    Gaussian with variance kT / (k + k_b). A half-weight bias would give kT / (k + k_b / 2)."""
    k, k_bias, mass, T = 2.0, 6.0, 4.0, 300.0
    n = 300
    kT = units.kB * T
    cvs = [Component([i], 0) for i in range(n)]
    method = HarmonicBias(cvs, kspring=k_bias, center=np.zeros(n))
    log = CVLog()
    pysages.run(method, lambda: harmonic_particles(n, k, mass), 30000, callback=log, stride=50)
    x = np.array(log.xi[len(log.xi) // 5:])
    variance = np.mean(x**2)
    assert abs(variance / (kT / (k + k_bias)) - 1) < 0.03
    assert abs(variance / (kT / (k + 0.5 * k_bias)) - 1) > 0.3      # rules out half weight


def test_callback_cadence_observers_and_state():
    k, mass = 2.0, 4.0
    dyn = harmonic_particles(50, k, mass)
    frames = []
    dyn.attach(lambda frame: frames.append(frame.step), interval=25)
    log = CVLog()
    method = HarmonicBias([Component([0], 0)], kspring=1.0, center=[0.0])
    result = pysages.run(method, lambda: dyn, 1000, callback=log)      # stride defaults to gcd = 25
    assert log.steps == list(range(25, 1001, 25))
    assert frames == list(range(0, 1001, 25))
    assert dyn.nsteps == 1000
    assert np.allclose(np.asarray(result.snapshots[0].positions), dyn.system.positions)


def test_spectral_abf_runs_and_resumes(tmp_path):
    def context():
        return harmonic_particles(1, 1.0, 1.0, friction=0.5)

    cvs = [Component([0], 0), Component([0], 1)]
    grid = pysages.Grid(lower=(-1.0, -1.0), upper=(1.0, 1.0), shape=(16, 16), periodic=False)
    method = SpectralABF(cvs, grid)
    log = CVLog()
    result = pysages.run(method, context, 4000, callback=log, stride=40)
    assert len(log.xi) == 100
    pysages.save(result, tmp_path / "result.pickle")
    resumed = pysages.run(pysages.load(tmp_path / "result.pickle"), context, 2000, stride=40)
    fes = np.asarray(pysages.analyze(resumed)["free_energy"])
    assert np.isfinite(fes).any()
    assert int(np.asarray(resumed.states[0].hist).sum()) == 6000      # samples from both segments
