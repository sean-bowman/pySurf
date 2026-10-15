
# -- pySurf: WCSPH Solver Tests -- #

'''

Checks of the pysurf.sphSolver equations against closed forms and conservation laws.

The Tait equation of state is checked against its value, its slope c0^2 at the reference density
and its inverse. The pair forces are checked against Newton's third law: in a gravity-free cloud
with random pressures and velocities, total linear and angular momentum change by rounding only.
The artificial viscosity is checked for sign, the XSPH correction for leaving momentum untouched,
the continuity rate against zero for a rigid translation and against -rho div(v) for a uniform
expansion, and the Shepard filter against returning a uniform density unchanged at the free
surface. A still tank initialized hydrostatically is run and must stay at rest.

Sean Bowman [10/11/2026]

'''

import math

import numpy as np
import pytest

from pysurf import constants
from pysurf.sphKernels import CubicSplineKernel
from pysurf.sphNeighbors import neighborPairs
from pysurf.sphParticles import ParticleSystem, latticePoints
from pysurf.sphSolver import (artificialViscosity, cflTimeStep, continuityRate, hydrostaticDensity, pairAccelerations,
                              scatterAdd, shepardDensity, speedOfSound, summationDensity, taitDensity,
                              taitEquationOfState, xsphVelocity, SimulationConfig, WcsphSolver)
from pysurf.WaterSim.scenarios.sloshingTank import SloshingTankConfig, createSloshingTank

rho0, c0 = 1000.0, 20.0
s = 0.01
h = 1.3 * s
kernel = CubicSplineKernel(2)

def pairSetup(positions: np.ndarray):

    '''Pairs, separations, distances, kernel values and gradients for a point set.'''

    i, j = neighborPairs(positions, 2.0 * h)
    separation = positions[i] - positions[j]
    distance = np.linalg.norm(separation, axis = 1)

    return i, j, separation, distance, kernel.evaluateBatch(distance, h), kernel.gradientBatch(separation, distance, h)

#----------------------------------------------------------------------#
# -- Equation of State -- #
#----------------------------------------------------------------------#

def testTaitIsZeroAtRestAndHasSlopeCSquared():

    step = 1e-4
    slope = (taitEquationOfState(rho0 + step, rho0, c0) - taitEquationOfState(rho0 - step, rho0, c0, clampNegative = False)) / (2.0 * step)

    assert taitEquationOfState(rho0, rho0, c0) == 0.0
    assert slope == pytest.approx(c0**2, rel = 1e-6)

def testTaitMatchesItsClosedForm():

    assert taitEquationOfState(1010.0, rho0, c0) == pytest.approx(rho0 * c0**2 / 7.0 * (1.01**7 - 1.0))

def testNegativePressureIsClampedByDefault():

    assert taitEquationOfState(990.0, rho0, c0) == 0.0
    assert taitEquationOfState(990.0, rho0, c0, clampNegative = False) < 0.0

def testTaitDensityInvertsTheEquationOfState():

    density = np.array([995.0, 1000.0, 1003.0, 1020.0])
    pressure = taitEquationOfState(density, rho0, c0, clampNegative = False)

    assert taitDensity(pressure, rho0, c0) == pytest.approx(density, rel = 1e-12)

def testHydrostaticDensityCarriesRhoGZ():

    depth = np.array([-0.1, 0.0, 0.2, 1.0])
    pressure = taitEquationOfState(hydrostaticDensity(depth, rho0, c0), rho0, c0)

    assert pressure == pytest.approx(rho0 * constants.gravity * np.maximum(depth, 0.0), rel = 1e-10, abs = 1e-9)

def testSpeedOfSoundIsTenLongWaveSpeeds():

    assert speedOfSound(0.5) == pytest.approx(10.0 * math.sqrt(constants.gravity * 0.5))

#----------------------------------------------------------------------#
# -- Pair Sums -- #
#----------------------------------------------------------------------#

def testScatterAddMatchesAddAt():

    rng = np.random.default_rng(1)
    index = rng.integers(0, 10, size = 200)
    scalars, vectors = rng.normal(size = 200), rng.normal(size = (200, 2))
    expectedScalars, expectedVectors = np.zeros(10), np.zeros((10, 2))
    np.add.at(expectedScalars, index, scalars)
    np.add.at(expectedVectors, index, vectors)

    assert scatterAdd(np.zeros(10), index, scalars) == pytest.approx(expectedScalars)
    assert scatterAdd(np.zeros((10, 2)), index, vectors) == pytest.approx(expectedVectors)

def testSummationDensityIsTheReferenceInsideALattice():

    '''An interior lattice particle sums to rho0 to within the cubic spline's 1% lattice error.'''

    positions = latticePoints([0.0, 0.0], [0.2, 0.2], s)
    i, j, _, _, values, _ = pairSetup(positions)
    density = summationDensity(rho0 * s**2, i, j, values, kernel.evaluate(0.0, h), len(positions))
    center = np.argmin(np.linalg.norm(positions - 0.1, axis = 1))

    assert density[center] == pytest.approx(rho0, rel = 0.01)
    # Array masses and the uniform-mass fast path agree
    assert density == pytest.approx(summationDensity(np.full(len(positions), rho0 * s**2), i, j, values,
                                                     kernel.evaluate(0.0, h)))

def testPairForcesConserveMomentum():

    rng = np.random.default_rng(7)
    positions = latticePoints([0.0, 0.0], [0.1, 0.1], s) + rng.uniform(-0.2 * s, 0.2 * s, size = (100, 2))
    n = len(positions)
    masses = rho0 * s**2 * rng.uniform(0.9, 1.1, size = n)
    densities = rho0 * rng.uniform(0.98, 1.02, size = n)
    pressures = taitEquationOfState(densities, rho0, c0, clampNegative = False)
    velocities = rng.normal(scale = 0.2, size = (n, 2))
    i, j, separation, distance, _, gradients = pairSetup(positions)
    viscosity = artificialViscosity(velocities[i] - velocities[j], separation, distance,
                                    0.5 * (densities[i] + densities[j]), h, c0, 0.1)

    accelerations = pairAccelerations(masses, densities, pressures, i, j, gradients, viscosity, n)
    forces = masses[:, None] * accelerations
    scale = np.max(np.abs(forces))

    assert np.abs(forces.sum(axis = 0)).max() < 1e-12 * scale * n
    torque = np.sum(positions[:, 0] * forces[:, 1] - positions[:, 1] * forces[:, 0])
    assert abs(torque) < 1e-12 * scale * n

def testViscosityActsOnlyOnApproachingPairs():

    separation = np.array([[s, 0.0], [s, 0.0]])
    distance = np.array([s, s])
    relativeVelocity = np.array([[-0.5, 0.0], [0.5, 0.0]])
    viscosity = artificialViscosity(relativeVelocity, separation, distance, np.full(2, rho0), h, c0, 0.1)

    assert viscosity[0] > 0.0
    assert viscosity[1] == 0.0

def testXsphLeavesMomentumAndUniformFlowUntouched():

    rng = np.random.default_rng(2)
    positions = latticePoints([0.0, 0.0], [0.1, 0.1], s)
    n = len(positions)
    masses = np.full(n, rho0 * s**2)
    densities = np.full(n, rho0)
    i, j, _, _, values, _ = pairSetup(positions)

    velocities = rng.normal(size = (n, 2))
    transport = xsphVelocity(velocities, masses, densities, i, j, values, 0.5)
    assert np.sum(masses[:, None] * (transport - velocities), axis = 0) == pytest.approx([0.0, 0.0], abs = 1e-12)

    uniform = np.tile([0.3, -0.2], (n, 1))
    assert xsphVelocity(uniform, masses, densities, i, j, values, 0.5) == pytest.approx(uniform)

def testContinuityIsZeroForARigidTranslation():

    positions = latticePoints([0.0, 0.0], [0.1, 0.1], s)
    i, j, _, _, _, gradients = pairSetup(positions)
    velocities = np.tile([1.5, -0.7], (len(positions), 1))

    assert continuityRate(rho0 * s**2, velocities, i, j, gradients, len(positions)) == pytest.approx(np.zeros(len(positions)), abs = 1e-9)

def testContinuityGivesMinusRhoDivergenceForAnExpansion():

    '''

    v = a x in 2D has div v = 2a, so d rho / dt = -2 a rho0 for an interior particle, to within the
    cubic spline's lattice error at h = 1.3 s (1.05 percent here, the same error the summation
    density carries).

    '''

    positions = latticePoints([0.0, 0.0], [0.2, 0.2], s)
    i, j, _, _, _, gradients = pairSetup(positions)
    rate = 0.5
    rateOfDensity = continuityRate(rho0 * s**2, rate * positions, i, j, gradients, len(positions))
    center = np.argmin(np.linalg.norm(positions - 0.1, axis = 1))

    assert rateOfDensity[center] == pytest.approx(-2.0 * rate * rho0, rel = 0.02)

def testShepardReturnsAUniformDensityUnchangedEvenAtTheSurface():

    positions = latticePoints([0.0, 0.0], [0.1, 0.05], s)
    i, j, _, _, values, _ = pairSetup(positions)
    densities = np.full(len(positions), 1003.0)
    masses = densities * s**2

    filtered = shepardDensity(masses, densities, i, j, values, kernel.evaluate(0.0, h))

    assert filtered == pytest.approx(densities, rel = 1e-12)

def testCflTakesTheTighterLimit():

    acoustic = 0.25 * h / (c0 + 1.0)

    assert cflTimeStep(h, c0, 1.0, 0.0, 0.25, 1.0) == pytest.approx(acoustic)
    assert cflTimeStep(h, c0, 1.0, 1e6, 0.25, 1.0) == pytest.approx(0.25 * math.sqrt(h / 1e6))
    assert cflTimeStep(h, c0, 1.0, 0.0, 0.25, 1e-6) == 1e-6

def accelerationOfTheFirstParticle(otherIsFluid: bool, boundaryFactor: float) -> np.ndarray:

    '''Acceleration of a fluid particle approaching a second particle, fluid or boundary, without gravity.'''

    from pysurf.sphParticles import ParticleSystem
    from pysurf.sphSolver import SimulationConfig

    first, second = np.array([[0.0, s]]), np.array([[0.0, 0.0]])
    velocities = np.array([[0.3, -0.5]])
    if otherIsFluid:
        particles = ParticleSystem.assemble(np.vstack([first, second]), np.zeros((0, 2)), s, rho0,
                                            np.vstack([velocities, [[0.0, 0.0]]]))
    else:
        particles = ParticleSystem.assemble(first, second, s, rho0, velocities)
    config = SimulationConfig(domainMin = np.array([-0.1, -0.1]), domainMax = np.array([0.1, 0.1]), particleSpacing = s,
                              referenceDensity = rho0, gravity = np.zeros(2), speedOfSound = c0, alphaViscosity = 0.1,
                              boundaryViscosityFactor = boundaryFactor)
    solver = WcsphSolver(config)
    solver.initialize(particles)
    solver._updatePairs()
    solver._computeRates()

    return particles.accelerations[0].copy()

def testTheBoundaryViscosityFactorScalesOnlyWallPairs():

    '''At rest density the pressure is zero, so the only force is the viscosity of the approach.'''

    assert accelerationOfTheFirstParticle(False, 0.0) == pytest.approx([0.0, 0.0], abs = 1e-12)
    assert accelerationOfTheFirstParticle(False, 1.0)[1] > 0.0
    assert accelerationOfTheFirstParticle(False, 0.5) == pytest.approx(0.5 * accelerationOfTheFirstParticle(False, 1.0))
    assert accelerationOfTheFirstParticle(True, 0.0) == pytest.approx(accelerationOfTheFirstParticle(True, 1.0))
    assert accelerationOfTheFirstParticle(True, 0.0)[1] > 0.0

#----------------------------------------------------------------------#
# -- Still Tank -- #
#----------------------------------------------------------------------#

def testAHydrostaticStillTankStaysAtRest():

    '''

    Water at rest in a tank, initialized at hydrostatic density, must stay at rest. After 0.2 s
    the largest particle speed stays under 1 percent of the long-wave speed sqrt(g d) (it reads
    0.1 percent) and no particle drifts a quarter spacing (0.02 spacings).

    The spacing divides the tank width and the fill height. A lattice that does not fit stops short
    of one wall by more than the one-spacing gap the boundary expects, and the water slumps into the
    extra gap: at 0.0075 m in this tank the particles drift 2.5 spacings in 0.6 s.

    '''

    tank = SloshingTankConfig(tankWidth = 0.2, tankHeight = 0.15, fillRatio = 0.5, initialTiltDeg = 0.0,
                              particleSpacing = 0.005)
    config, particles, boundary = createSloshingTank(tank)
    config.maxTimeStep = 1e-3
    solver = WcsphSolver(config, boundary = boundary)
    solver.initialize(particles)
    fillHeight = tank.fillRatio * tank.tankHeight
    fluidStart = particles.positions[particles.isFluid].copy()

    while solver.time < 0.2:
        state = solver.step()

    assert state.maxVelocity < 0.01 * math.sqrt(constants.gravity * fillHeight)
    assert np.max(np.abs(particles.positions[particles.isFluid] - fluidStart)) < 0.25 * tank.particleSpacing

@pytest.mark.parametrize('floor', [0.0, 0.9])
def testADensityFloorHoldsADropletWhoseNeighborsFlyApart(floor):

    # Two particles of a droplet flying apart at 20 m/s in a twentieth of a second lose most of
    # their density without a floor, and keep 0.9 rho0 with one
    spacing = 0.01
    particles = ParticleSystem.assemble(np.array([[0.0, 0.0], [spacing, 0.0]]), np.empty((0, 2)), spacing, 1000.0,
                                        np.array([[-10.0, 0.0], [10.0, 0.0]]))
    config = SimulationConfig(domainMin = np.array([-1.0, -1.0]), domainMax = np.array([1.0, 1.0]),
                              particleSpacing = spacing, gravity = np.zeros(2), speedOfSound = 50.0,
                              xsphEpsilon = 0.0, densityFilterInterval = 0, densityFloor = floor)
    solver = WcsphSolver(config)
    solver.initialize(particles)
    for _ in range(20):
        solver.step()

    if floor:
        assert np.min(particles.densities) == pytest.approx(900.0)
    else:
        assert np.min(particles.densities) < 900.0
