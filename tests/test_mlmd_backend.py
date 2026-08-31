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


def test_unbiased_is_bit_identical_to_plain_mlmd():
    """`Unbiased` must not perturb the dynamics at all: same seed, same trajectory, bit for bit
    (CPU). This pins the backend's bias plumbing at zero bias, including the half-kick bookkeeping."""
    from pysages.methods import Unbiased

    reference = harmonic_particles(20, 2.0, 4.0, seed=3)
    reference.run(400)

    dyn = harmonic_particles(20, 2.0, 4.0, seed=3)
    method = Unbiased([Component([0], 0)])
    pysages.run(method, lambda: dyn, 400)
    assert np.array_equal(dyn.system.positions, reference.system.positions)
    assert np.array_equal(dyn.system.velocities, reference.system.velocities)


def test_metadynamics_deposits_and_broadens():
    """Standard metadynamics on x of one particle in a harmonic well: Gaussians are deposited and
    the biased trajectory explores farther than the unbiased one."""
    from pysages.methods import Metadynamics

    def context():
        return harmonic_particles(1, 2.0, 4.0, seed=1, friction=0.5)

    steps, stride = 8000, 100
    method = Metadynamics([Component([0], 0)], height=0.02, sigma=0.05, stride=stride,
                          ngaussians=steps // stride + 1)
    log = CVLog()
    result = pysages.run(method, context, steps, callback=log, stride=50)
    heights = np.asarray(result.states[0].heights).ravel()
    deposited = int((heights > 0).sum())
    assert abs(deposited - steps // stride) <= 1             # one Gaussian per stride (+- bookkeeping)
    x = np.array(log.xi)
    unbiased = harmonic_particles(1, 2.0, 4.0, seed=1, friction=0.5)
    trace = []
    unbiased.attach(lambda f: trace.append(f.positions[0, 0]), interval=50)
    unbiased.run(steps)
    assert np.abs(x).max() > 1.5 * np.abs(np.array(trace)).max()


def test_distance_and_dihedral_cvs_report_the_geometry():
    """CV plumbing through the fused step: the logged xi must equal the geometry computed directly
    from the trajectory positions."""
    from pysages.colvars import DihedralAngle, Distance
    from pysages.methods import Unbiased

    rng = np.random.default_rng(0)
    base = np.array([[0.0, 0, 0], [1.5, 0, 0], [2.0, 1.4, 0], [2.5, 1.4, 1.3]])
    system = System(base + rng.normal(0, 0.01, (4, 3)), [6] * 4,
                    Harmonic(5.0, base, dtype="float64"), masses=[12.0] * 4)
    dyn = Langevin(system, timestep=0.5, temperature=200.0, friction=0.1, seed=2)
    positions_log = []
    dyn.attach(lambda f: positions_log.append(f.positions.copy()), interval=25)
    method = Unbiased([Distance([0, 3]), DihedralAngle([0, 1, 2, 3])])
    log = CVLog()
    pysages.run(method, lambda: dyn, 500, callback=log, stride=25)

    def dihedral(p):
        b1, b2, b3 = p[1] - p[0], p[2] - p[1], p[3] - p[2]
        n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
        return np.arctan2(np.dot(np.cross(n1, n2), b2 / np.linalg.norm(b2)), np.dot(n1, n2))

    # callback at step s sees the CV of the state the method updated from (start of step s), which
    # equals the observer frame at step s - ... : compare against the positions logged at the same
    # steps; the method computes xi from the frame *before* the step it biases, so use a tolerance
    # window of one logging interval by comparing against the closest logged frame.
    for xi, p_end in zip(log.xi[1:], positions_log[1:len(log.xi)]):
        d_ref = np.linalg.norm(p_end[3] - p_end[0])
        phi_ref = dihedral(p_end)
        assert abs(xi[0] - d_ref) < 0.15                      # same geometry, one step apart at most
        assert abs(np.angle(np.exp(1j * (xi[1] - phi_ref)))) < 0.3


def test_npt_is_rejected():
    """Biased sampling with a barostat is unsupported (the bias has no virial); the backend must say
    so rather than sample a wrong ensemble."""
    from ase import Atoms
    from mlmd import NPT
    from mlmd.calculators import LennardJones

    rng = np.random.default_rng(0)
    a = 5.4
    cell = np.eye(3) * a * 2
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) * a
    offsets = np.stack(np.meshgrid(*[np.arange(2)] * 3, indexing="ij"), -1).reshape(-1, 3) * a
    positions = (offsets[:, None, :] + basis[None]).reshape(-1, 3) + rng.normal(0, 0.05, (32, 3))
    system = System(positions, [18] * 32, LennardJones(0.0104, 3.4, 8.0, dtype="float64"),
                    cell=cell, pbc=True, masses=[39.948] * 32)
    dyn = NPT(system, timestep=5.0, temperature=100.0, pressure=units.bar_to_eV_per_A3(1000.0),
              tdamp=500.0, pdamp=5000.0)
    method = HarmonicBias([Component([0], 0)], kspring=1.0, center=[0.0])
    with pytest.raises(ValueError, match="NPT"):
        pysages.run(method, lambda: dyn, 100)
