
# -- Tests for the delta-LES-SPH closure in the solver -- #

'''

The closure of Meringolo et al. (2019) as `pysurf.sphSolver` carries it (`turbulenceClosure =
'deltaLes'`), against known answers. The coefficients against their formulas and caps; the viscous
term's dissipation per unit area for a simple shear against the continuum's mu gamma^2, mu =
alpha h c0 rho0 / K; the term against conserving linear and angular momentum, against any pair
feeding energy into the flow, and against acting on a rigid rotation; the diffusive term with a
coefficient per pair against the scalar form; and the array solver with the closure on, its energy
budget closing and a still tank staying at rest.

Sean Bowman [11/27/2026]

'''

import numpy as np
import pytest

from pysurf import constants
from pysurf.sphKernels import createKernel
from pysurf.sphNeighbors import neighborPairs
from pysurf.sphSolver import (SimulationConfig, WcsphSolver, closureCap, closureCoefficients, closureViscosity,
                              densityDiffusionRate, harmonicMean, pairAccelerations, velocityGradients)

rho0, c0, s = 1000.0, 20.0, 0.01
h = 3.0 * s

def lattice(columns: int, rows: int, jitter: float = 0.0, seed: int = 3) -> np.ndarray:

    '''A square lattice of spacing s from the origin, each point moved by up to `jitter` spacings.'''

    x, y = np.meshgrid((np.arange(columns) + 0.5) * s, (np.arange(rows) + 0.5) * s)
    positions = np.column_stack([x.ravel(), y.ravel()])
    if jitter:
        positions += np.random.default_rng(seed).uniform(-jitter, jitter, size = positions.shape) * s

    return positions

def pairsOf(positions: np.ndarray) -> dict:

    '''The pairs of a point set with their separations, distances and Wendland C2 gradients at h.'''

    i, j = neighborPairs(positions, 2.0 * h)
    separation = positions[i] - positions[j]
    distance = np.linalg.norm(separation, axis = 1)
    kernel = createKernel('wendlandC2', positions.shape[1])

    return dict(i = i, j = j, separation = separation, distance = distance,
                gradients = kernel.gradientBatch(separation, distance, h))

def closurePi(pairs: dict, velocities: np.ndarray, densities: np.ndarray, alphas: np.ndarray,
              kinematicViscosity: float = 0.0) -> np.ndarray:

    '''Each pair's closure Pi_ij from the given alphas.'''

    i, j = pairs['i'], pairs['j']
    return closureViscosity(velocities[i] - velocities[j], pairs['separation'], pairs['distance'], densities[i],
                            densities[j], h, c0, rho0, alphas[i], alphas[j], kinematicViscosity)

def pairDissipation(pairs: dict, masses: np.ndarray, velocities: np.ndarray, viscosity: np.ndarray) -> np.ndarray:

    '''The power each pair's viscous forces remove from the flow, m_i m_j Pi_ij v_ij . grad_i W_ij [W].'''

    i, j = pairs['i'], pairs['j']
    return masses[i] * masses[j] * viscosity * np.sum((velocities[i] - velocities[j]) * pairs['gradients'], axis = 1)

@pytest.mark.parametrize('dimensions', [2, 3])
def testTheCoefficientsFollowTheClosureAndItsCap(dimensions):

    rng = np.random.default_rng(1)
    gradients = rng.normal(size = (5, dimensions, dimensions))
    gradients[3] *= 1e4
    fluid = np.array([True, True, True, True, False])
    alphas, deltas = closureCoefficients(gradients, fluid, h, c0)
    strain = 0.5 * (gradients + np.swapaxes(gradients, 1, 2))
    norms = np.sqrt(2.0 * np.sum(strain**2, axis = (1, 2)))
    width = 2.0 * h

    assert alphas[:3] == pytest.approx(2.0 * (dimensions + 2) * (0.12 * width)**2 * norms[:3] / (c0 * h), rel = 1e-12)
    assert deltas[:3] == pytest.approx((1.5 * width)**2 * norms[:3] / (c0 * h), rel = 1e-12)
    assert alphas[3] == deltas[3] == closureCap
    assert alphas[4] == deltas[4] == 0.0

def testTheViscosityIsTheContinuumDissipationOfASimpleShear():

    # u = (gamma y, 0) dissipates mu gamma^2 per unit area, mu = alpha h c0 rho0 / K, K = 8 in 2D
    gamma, alpha = 3.0, 0.02
    positions = lattice(40, 40)
    count = len(positions)
    velocities = np.column_stack([gamma * positions[:, 1], np.zeros(count)])
    densities, masses = np.full(count, rho0), np.full(count, rho0 * s * s)
    pairs = pairsOf(positions)
    removed = pairDissipation(pairs, masses, velocities, closurePi(pairs, velocities, densities, np.full(count, alpha)))
    # Each pair's dissipation shared between its two particles, read in the interior
    perParticle = np.zeros(count)
    np.add.at(perParticle, pairs['i'], 0.5 * removed)
    np.add.at(perParticle, pairs['j'], 0.5 * removed)
    reach = 2.0 * h
    interior = np.all((positions > reach) & (positions < 40 * s - reach), axis = 1)
    mu = alpha * h * c0 * rho0 / 8.0

    assert np.mean(perParticle[interior]) / (s * s) == pytest.approx(mu * gamma**2, rel = 0.02)

@pytest.mark.parametrize('dimensions', [2, 3])
def testTheViscosityConservesMomentumAndNeverFeedsTheFlow(dimensions):

    rng = np.random.default_rng(7)
    count = 400 if dimensions == 2 else 500
    positions = rng.uniform(0.0, 1.0, size = (count, dimensions)) * s * count**(1.0 / dimensions)
    velocities = rng.normal(size = (count, dimensions))
    densities = rho0 * (1.0 + rng.uniform(-0.01, 0.01, count))
    masses = rho0 * s**dimensions * (1.0 + rng.uniform(-0.05, 0.05, count))
    alphas = rng.uniform(0.0, 0.2, count)
    pairs = pairsOf(positions)
    viscosity = closurePi(pairs, velocities, densities, alphas, constants.freshwaterKinematicViscosity)
    accelerations = pairAccelerations(masses, densities, np.zeros(count), pairs['i'], pairs['j'], pairs['gradients'],
                                      viscosity, count)
    forces = masses[:, None] * accelerations
    scale = np.abs(forces).max()

    assert np.abs(forces.sum(axis = 0)).max() < 1e-12 * count * scale
    if dimensions == 2:
        torque = positions[:, 0] * forces[:, 1] - positions[:, 1] * forces[:, 0]
        assert abs(torque.sum()) < 1e-12 * count * scale
    else:
        assert np.abs(np.cross(positions, forces).sum(axis = 0)).max() < 1e-12 * count * scale
    assert np.all(pairDissipation(pairs, masses, velocities, viscosity) >= 0.0)

def testARigidRotationFeelsNoClosure():

    # The rotation's strain is zero, so its coefficients are; and given any, its pairs never approach
    positions = lattice(24, 24, jitter = 0.15)
    count = len(positions)
    omega = 2.5
    centred = positions - positions.mean(axis = 0)
    velocities = omega * np.column_stack([-centred[:, 1], centred[:, 0]])
    fluid = np.ones(count, dtype = bool)
    densities, masses = np.full(count, rho0), np.full(count, rho0 * s * s)
    pairs = pairsOf(positions)
    gradient = velocityGradients(masses, densities, velocities, pairs['i'], pairs['j'], pairs['separation'],
                                 pairs['gradients'], fluid)
    alphas, deltas = closureCoefficients(gradient, fluid, h, c0)
    # A simple shear of the same rate, ||D|| = omega, for scale
    shearAlpha, shearDelta = closureCoefficients(np.array([[[0.0, omega], [0.0, 0.0]]]), np.array([True]), h, c0)
    viscosity = closurePi(pairs, velocities, densities, np.full(count, 0.1))
    shear = closurePi(pairs, np.column_stack([omega * positions[:, 1], np.zeros(count)]), densities, np.full(count, 0.1))

    assert np.max(alphas) < 1e-9 * shearAlpha[0]
    assert np.max(deltas) < 1e-9 * shearDelta[0]
    assert np.max(np.abs(viscosity)) < 1e-9 * np.max(np.abs(shear))

def testTheDiffusionTakesEachPairsCoefficient():

    positions = lattice(20, 12, jitter = 0.15)
    count = len(positions)
    fluid = np.ones(count, dtype = bool)
    fluid[:5] = False
    rng = np.random.default_rng(2)
    densities = rho0 * (1.0 + rng.uniform(-0.01, 0.01, count))
    masses = np.full(count, rho0 * s * s)
    pairs = pairsOf(positions)
    i, j = pairs['i'], pairs['j']
    active = fluid[i] & fluid[j]
    arguments = (masses, densities, i, j, pairs['separation'], pairs['distance'], pairs['gradients'], count, h, c0)
    scalar = densityDiffusionRate(*arguments, 0.1, rho0, None, active)
    uniform = densityDiffusionRate(*arguments, np.full(len(i), 0.1), rho0, None, active)
    deltas = rng.uniform(0.0, 0.2, count)
    perPair = densityDiffusionRate(*arguments, harmonicMean(deltas[i], deltas[j]), rho0, None, active)
    # The same sum written out pair by pair
    weight = 2.0 * harmonicMean(deltas[i], deltas[j]) * h * c0 * (densities[j] - densities[i]) \
        * -np.sum(pairs['separation'] * pairs['gradients'], axis = 1) / (pairs['distance']**2 + 0.01 * h * h)
    expected = np.zeros(count)
    volumes = masses / densities
    np.add.at(expected, i[active], (volumes[j] * weight)[active])
    np.add.at(expected, j[active], -(volumes[i] * weight)[active])

    assert uniform == pytest.approx(scalar, rel = 1e-12, abs = 1e-12 * np.abs(scalar).max())
    assert perPair == pytest.approx(expected, rel = 1e-10, abs = 1e-12 * np.abs(expected).max())
    assert harmonicMean(np.array([0.0, 0.1]), np.array([0.0, 0.3])) == pytest.approx([0.0, 0.15])

def testTheClosureNeedsADiffusiveTermAndAValidName():

    base = dict(domainMin = np.zeros(2), domainMax = np.ones(2))
    with pytest.raises(ValueError):
        WcsphSolver(SimulationConfig(**base, turbulenceClosure = 'deltaLes'))
    with pytest.raises(ValueError):
        WcsphSolver(SimulationConfig(**base, turbulenceClosure = 'smagorinsky', densityDiffusion = 'antuono'))
    with pytest.raises(ValueError):
        WcsphSolver(SimulationConfig(**base, turbulenceClosure = 'deltaLes', densityDiffusion = 'antuono',
                                     backend = 'compiled'))

# Meringolo et al.'s model: the closure alone; the hybrid keeps the limited artificial viscosity
forms = {'published': dict(alphaViscosity = 0.0, viscosityLimiter = False),
         'hybrid': dict(alphaViscosity = 0.01, viscosityLimiter = True)}

def stillTank(backend: str, form: str, closure: str = 'deltaLes'):

    '''A deltaSph still tank with the closure in the given form, initialized hydrostatically.'''

    from pysurf.WaterSim.scenarios.sloshingTank import SloshingTankConfig, createSloshingTank

    tank = SloshingTankConfig(tankWidth = 0.2, tankHeight = 0.15, fillRatio = 0.5, initialTiltDeg = 0.0,
                              particleSpacing = 0.005)
    config, particles, boundary = createSloshingTank(tank)
    options = dict(forms[form], backend = backend, maxTimeStep = 1e-3, densityFilterInterval = 0, xsphEpsilon = 0.0,
                   wallPressure = 'extrapolated', continuityCorrection = True, densityFloor = 0.9,
                   densityDiffusion = 'antuono', turbulenceClosure = closure)
    for name, value in options.items():
        setattr(config, name, value)
    solver = WcsphSolver(config, boundary = boundary)
    solver.initialize(particles)

    return solver, particles, tank

@pytest.mark.parametrize('form', ['published', 'hybrid'])
def testAStillTankStaysAtRestWithTheClosure(form):

    # The still tank's criteria after 0.2 s, as tests/testSphGpu.py reads them. Near 0.1 s this
    # cubic spline tank peaks above them at alpha 0.01 or 0, with the closure or without it
    solver, particles, tank = stillTank('numpy', form)
    start = particles.positions[particles.isFluid].copy()
    while solver.time < 0.2:
        state = solver.step()
    fillHeight = tank.fillRatio * tank.tankHeight

    assert state.maxVelocity < 0.01 * np.sqrt(constants.gravity * fillHeight)
    assert np.max(np.abs(particles.positions[particles.isFluid] - start)) < 0.25 * tank.particleSpacing
