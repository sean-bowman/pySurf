
# -- pySurf: WCSPH Solver Tests -- #

'''

Checks of the pysurf.sphSolver equations against closed forms and conservation laws.

The Tait equation of state is checked against its value, its slope c0^2 at the reference density
and its inverse. The pair forces are checked against Newton's third law: in a gravity-free cloud
with random pressures and velocities, total linear and angular momentum change by rounding only.
The artificial viscosity is checked for sign; its limited form for leaving a linear flow alone
while it still damps noise, for momentum, and for never feeding energy in; the velocity gradient it
reconstructs with for exactness on a linear flow out to a free surface. The XSPH correction is
checked for leaving momentum untouched,
the continuity rate against zero for a rigid translation and against -rho div(v) for a uniform
expansion, and the Shepard filter against returning a uniform density unchanged at the free
surface. A still tank initialized hydrostatically is run and must stay at rest.

Sean Bowman [10/11/2026]

'''

import math

import numpy as np
import pytest

from pysurf import constants
from pysurf.sphKernels import CubicSplineKernel, createKernel
from pysurf.sphNeighbors import neighborPairs
from pysurf.sphParticles import ParticleSystem, latticePoints
from pysurf.sphSolver import (artificialViscosity, cflTimeStep, continuityRate, correctionMatrices, densityDiffusionRate, extrapolatedWallPressure, hydrostaticDensity,
                              hydrostaticDensityDifference, limitedVelocityDifference, pairAccelerations, scatterAdd, shepardDensity,
                              freeSurface, shiftedVelocityChange, shiftingDisplacement, shiftingFields, speedOfSound, summationDensity, taitDensity,
                              taitEquationOfState, velocityGradients, xsphVelocity, SimulationConfig, WcsphSolver)
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

#----------------------------------------------------------------------#
# -- Limited Artificial Viscosity -- #
#----------------------------------------------------------------------#

def waterBlock(jitter: float = 0.0, seed: int = 4) -> dict:

    '''Water 40 spacings wide and 20 deep with free surfaces all round, at h = 3 spacings with the Wendland C2.'''

    rng = np.random.default_rng(seed)
    positions = latticePoints([0.0, 0.0], [40 * s, 20 * s], s)
    positions = positions + rng.uniform(-jitter * s, jitter * s, size = positions.shape)
    length = 3.0 * s
    i, j = neighborPairs(positions, 2.0 * length)
    separation = positions[i] - positions[j]
    distance = np.linalg.norm(separation, axis = 1)
    count = len(positions)

    return dict(positions = positions, h = length, i = i, j = j, separation = separation, distance = distance,
                gradients = createKernel('wendlandC2', 2).gradientBatch(separation, distance, length),
                masses = np.full(count, rho0 * s**2), densities = np.full(count, rho0), isFluid = np.ones(count, dtype = bool))

def limitedTerm(block: dict, velocities: np.ndarray, limited: bool, alpha: float = 0.01) -> tuple:

    '''Pi_ij for each pair, plain or limited, and the term's total dissipation sum_pairs m_i m_j Pi_ij v_ij . grad W_ij [W/m].'''

    i, j = block['i'], block['j']
    relative = velocities[i] - velocities[j]
    reconstructed = None
    if limited:
        gradient = velocityGradients(block['masses'], block['densities'], velocities, i, j, block['separation'],
                                     block['gradients'], block['isFluid'])
        reconstructed = limitedVelocityDifference(relative, block['separation'], block['distance'], block['h'],
                                                  gradient[i], gradient[j])
    viscosity = artificialViscosity(relative, block['separation'], block['distance'],
                                    0.5 * (block['densities'][i] + block['densities'][j]), block['h'], c0, alpha,
                                    reconstructed)
    power = block['masses'][i] * block['masses'][j] * viscosity * np.sum(relative * block['gradients'], axis = 1)

    return viscosity, power

def testVelocityGradientsAreExactForALinearFlowOutToTheFreeSurface():

    block = waterBlock(jitter = 0.2)
    field = np.array([[0.7, -1.3], [0.4, -0.7]])
    velocities = block['positions'] @ field.T + np.array([0.3, -0.2])
    result = velocityGradients(block['masses'], block['densities'], velocities, block['i'], block['j'],
                               block['separation'], block['gradients'], block['isFluid'])

    assert np.all(np.any(result != 0.0, axis = (1, 2)))
    assert result == pytest.approx(np.broadcast_to(field, result.shape), abs = 1e-10)

def testTheLimitedViscosityLeavesALinearFlowAlone():

    # In pure strain the plain term acts on every pair the strain closes; the limited one on none
    block = waterBlock()
    strain = block['positions'] @ np.array([[1.0, 0.0], [0.0, -1.0]]).T
    plain, plainPower = limitedTerm(block, strain, False)
    limited, limitedPower = limitedTerm(block, strain, True)

    assert np.count_nonzero(plain) > 0.4 * len(plain)
    assert np.abs(limited).max() < 1e-12 * np.abs(plain).max()
    assert limitedPower.sum() < 1e-12 * plainPower.sum()

def testTheLimitedViscosityStillDampsNoise():

    # Noise of 0.2 spacings per second over a strain of one per second: the limited term dissipates at
    # least half what the plain term takes from the noise alone, and a tenth of what it takes from both
    block = waterBlock()
    strain = block['positions'] @ np.array([[1.0, 0.0], [0.0, -1.0]]).T
    noise = np.random.default_rng(9).normal(scale = 0.2 * s, size = strain.shape)
    _, noisePower = limitedTerm(block, noise, False)
    _, plainPower = limitedTerm(block, strain + noise, False)
    _, limitedPower = limitedTerm(block, strain + noise, True)

    assert limitedPower.sum() > 0.5 * noisePower.sum()
    assert limitedPower.sum() < 0.1 * plainPower.sum()

def testTheLimitedViscosityConservesMomentumAndNeverFeedsEnergy():

    block = waterBlock(jitter = 0.3, seed = 5)
    rng = np.random.default_rng(6)
    positions, i, j = block['positions'], block['i'], block['j']
    velocities = positions @ rng.normal(size = (2, 2)).T + rng.normal(scale = 0.05, size = positions.shape)
    viscosity, power = limitedTerm(block, velocities, True, alpha = 0.1)
    count = len(positions)
    forces = block['masses'][:, None] * pairAccelerations(block['masses'], block['densities'], np.zeros(count), i, j,
                                                          block['gradients'], viscosity, count)
    scale = np.max(np.abs(forces))
    gradient = velocityGradients(block['masses'], block['densities'], velocities, i, j, block['separation'],
                                 block['gradients'], block['isFluid'])
    relative = velocities[i] - velocities[j]
    forward = limitedVelocityDifference(relative, block['separation'], block['distance'], block['h'], gradient[i], gradient[j])
    backward = limitedVelocityDifference(-relative, -block['separation'], block['distance'], block['h'], gradient[j], gradient[i])

    assert np.count_nonzero(viscosity) > 0
    assert np.abs(forces.sum(axis = 0)).max() < 1e-12 * scale * count
    torque = np.sum(positions[:, 0] * forces[:, 1] - positions[:, 1] * forces[:, 0])
    assert abs(torque) < 1e-12 * scale * count
    assert power.min() >= 0.0
    assert backward == pytest.approx(-forward, abs = 1e-15)

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

def equilibriumColumn(surface: float = 0.1) -> tuple:

    '''

    A lattice of water below a free surface at `surface`, at the density water at rest holds:
    dp/dz = -rho g with the Tait equation gives rho = rho0 (1 + 6 g (surface - z) / c0^2)^(1/6).

    '''

    positions = latticePoints([0.0, 0.0], [0.2, surface], s)
    densities = rho0 * (1.0 + 6.0 * constants.gravity * (surface - positions[:, 1]) / c0**2)**(1.0 / 6.0)
    i, j, separation, distance, _, gradients = pairSetup(positions)

    return positions, densities, i, j, separation, distance, gradients

def testDiffusionLeavesAUniformDensityAlone():

    positions = latticePoints([0.0, 0.0], [0.1, 0.05], s)
    i, j, separation, distance, _, gradients = pairSetup(positions)
    densities = np.full(len(positions), 1003.0)

    rate = densityDiffusionRate(rho0 * s**2, densities, i, j, separation, distance, gradients, len(positions), h, c0,
                                0.1, rho0)

    assert rate == pytest.approx(np.zeros(len(positions)), abs = 1e-9)

def testDiffusionFlattensADensityBump():

    # Each pair moves its two densities toward each other: the bump falls and its neighbors rise
    positions = latticePoints([0.0, 0.0], [0.1, 0.1], s)
    i, j, separation, distance, _, gradients = pairSetup(positions)
    center = int(np.argmin(np.linalg.norm(positions - 0.05, axis = 1)))
    densities = np.full(len(positions), rho0)
    densities[center] += 1.0

    rate = densityDiffusionRate(rho0 * s**2, densities, i, j, separation, distance, gradients, len(positions), h, c0,
                                0.1, rho0)
    near = (np.linalg.norm(positions - positions[center], axis = 1) < 2.0 * h) & (np.arange(len(positions)) != center)

    assert rate[center] < 0.0
    assert np.all(rate[near] > 0.0)

def testHydrostaticDiffusionLeavesWaterAtRestAlone():

    '''

    Water at rest holds a density gradient. Molteni and Colagrossi's term diffuses it wherever the
    kernel support is cut: at the free surface its rate reaches 0.057 rho0 per second here, against
    1e-4 rho0 per second inside the water. With the hydrostatic part taken out, the largest rate
    left is 7e-8 of that.

    '''

    positions, densities, i, j, separation, distance, gradients = equilibriumColumn()
    arguments = (rho0 * s**2, densities, i, j, separation, distance, gradients, len(positions), h, c0, 0.1, rho0)

    molteni = densityDiffusionRate(*arguments)
    hydrostatic = densityDiffusionRate(*arguments, gravity = constants.gravity)

    assert np.abs(molteni).max() > 1e-3 * rho0
    assert np.abs(hydrostatic).max() < 1e-6 * np.abs(molteni).max()

def testHydrostaticDifferenceIsTheEquilibriumProfile():

    # Across one spacing the pair formula matches the closed-form profile to second order
    positions, densities, *_ = equilibriumColumn()
    column = np.flatnonzero(np.isclose(positions[:, 0], positions[0, 0]))
    upper, lower = column[-1], column[-2]
    difference = hydrostaticDensityDifference(positions[upper, 1] - positions[lower, 1],
                                              0.5 * (densities[upper] + densities[lower]), rho0, c0, constants.gravity)

    assert difference == pytest.approx(densities[lower] - densities[upper], rel = 1e-6)

def shiftsOf(positions: np.ndarray, isFluid: np.ndarray | None = None, size: float = 1e-5) -> tuple:

    '''The shift of every particle of a point set at rest density, with its fields.'''

    isFluid = np.ones(len(positions), dtype = bool) if isFluid is None else isFluid
    i, j, separation, distance, values, gradients = pairSetup(positions)
    densities = np.full(len(positions), rho0)
    fields = shiftingFields(rho0 * s**2, densities, i, j, separation, values, gradients, len(positions),
                            kernel.evaluate(s, h))

    shift, _ = shiftingDisplacement(*fields, i, j, values, isFluid, size, h)

    return shift, fields, (i, j)

def testShiftingLeavesALatticeInsideTheWaterAlone():

    positions = latticePoints([0.0, 0.0], [0.2, 0.1], s)
    shift, (_, _, divergence, _), _ = shiftsOf(positions)
    top = positions[:, 1] > positions[:, 1].max() - 0.5 * s
    inside = np.all((positions > 3.0 * h) & (positions < np.array([0.2, 0.1]) - 3.0 * h), axis = 1)

    assert np.all(divergence[top] < 1.5)
    assert np.all(shift[top] == 0.0)
    assert np.abs(shift[inside]).max() < 1e-12

def testASurfaceParticleBesideAWallIsOnTheSurface():

    # Wall particles beside the top corner lift its div r over 1.5; its lopsided support still marks it
    water = latticePoints([0.0, 0.0], [0.1, 0.05], s)
    wall = np.vstack([latticePoints([-3.0 * s, -3.0 * s], [0.0, 0.08], s),
                      latticePoints([0.0, -3.0 * s], [0.1, 0.0], s)])
    positions = np.vstack([water, wall])
    isFluid = np.r_[np.ones(len(water), dtype = bool), np.zeros(len(wall), dtype = bool)]
    shift, (_, plain, divergence, _), _ = shiftsOf(positions, isFluid)
    corner = int(np.argmin(np.linalg.norm(water - [0.5 * s, 0.05 - 0.5 * s], axis = 1)))
    below = int(np.argmin(np.linalg.norm(water - [0.5 * s, 0.025], axis = 1)))

    assert divergence[corner] > 1.5
    assert freeSurface(plain, divergence, isFluid, h)[corner]
    assert np.all(shift[corner] == 0.0)
    assert not freeSurface(plain, divergence, isFluid, h)[below]

def testShiftingPartsACrowdedPair():

    positions = latticePoints([0.0, 0.0], [0.2, 0.2], s)
    center = int(np.argmin(np.linalg.norm(positions - 0.1, axis = 1)))
    partner = int(np.argmin(np.linalg.norm(positions - (positions[center] + [s, 0.0]), axis = 1)))
    positions[partner, 0] -= 0.4 * s

    shift, _, _ = shiftsOf(positions)

    assert shift[center, 0] < 0.0 < shift[partner, 0]

def testShiftingNextToTheSurfaceRunsAlongIt():

    # Under the jittered flat top of a block of water the normal of a particle near it, the
    # kernel-weighted mean of its surface neighbors' normals, stays within 10 degrees of vertical,
    # and its shift has no component along it; its own grad C0 two layers down points anywhere
    rng = np.random.default_rng(2)
    positions = latticePoints([0.0, 0.0], [0.3, 0.1], s) + rng.uniform(-0.1, 0.1, size = (300, 2)) * s
    shift, (_, plain, divergence, _), (i, j) = shiftsOf(positions)
    values = kernel.evaluateBatch(np.linalg.norm(positions[i] - positions[j], axis = 1), h)
    surface = freeSurface(plain, divergence, np.ones(len(positions), dtype = bool), h)
    surfaceNormal = np.zeros_like(plain)
    surfaceNormal[surface] = -plain[surface] / np.linalg.norm(plain[surface], axis = 1)[:, None]
    normal = np.zeros_like(plain)
    scatterAdd(normal, i, values[:, None] * surfaceNormal[j])
    scatterAdd(normal, j, values[:, None] * surfaceNormal[i])
    vicinity = np.zeros(len(positions), dtype = bool)
    vicinity[i[surface[j]]] = True
    vicinity[j[surface[i]]] = True
    vicinity &= ~surface
    # Particles with no surface neighbor have no normal
    normal[vicinity] /= np.linalg.norm(normal[vicinity], axis = 1)[:, None]
    middle = vicinity & (positions[:, 0] > 3.0 * h) & (positions[:, 0] < 0.3 - 3.0 * h) & (positions[:, 1] > 0.05)
    ownDirection = -plain / np.linalg.norm(plain, axis = 1)[:, None]

    assert vicinity.sum() > 50
    assert np.abs(np.sum(shift[vicinity] * normal[vicinity], axis = 1)).max() < 1e-15
    assert np.abs(shift[vicinity]).max() > 0.0
    assert np.all(normal[middle, 1] > np.cos(np.radians(10.0)))
    assert np.any(ownDirection[middle, 1] < np.cos(np.radians(45.0)))

def testAShiftCarriesTheDensityOfWhereItLands():

    # Inside the water the difference gradient of a linear density matches it to the cubic spline's
    # lattice error (1.05 percent at h = 1.3 s), and a shifted particle takes rho_i + delta r . grad rho
    positions = latticePoints([0.0, 0.0], [0.2, 0.2], s)
    densities = rho0 + 3.0 * positions[:, 0] - 2.0 * positions[:, 1]
    center = int(np.argmin(np.linalg.norm(positions - 0.1, axis = 1)))
    positions[center, 0] += 0.3 * s
    i, j, separation, distance, values, gradients = pairSetup(positions)
    fields = shiftingFields(rho0 * s**2, densities, i, j, separation, values, gradients, len(positions),
                            kernel.evaluate(s, h))
    shift, change = shiftingDisplacement(*fields, i, j, values, np.ones(len(positions), dtype = bool), 1e-5, h)
    clear = int(np.argmin(np.linalg.norm(positions - (positions[center] + [5.0 * s, 0.0]), axis = 1)))

    assert fields[3][clear] == pytest.approx([3.0, -2.0], rel = 0.02)
    assert change[center] == pytest.approx(np.dot(shift[center], fields[3][center]))
    assert shift[center, 0] < 0.0

def testAShiftCarriesTheVelocityOfWhereItLands():

    # Inside the water (delta r . grad) v of a linear velocity field matches it to the cubic spline's
    # lattice error, and pairs with a wall particle add nothing
    positions = latticePoints([0.0, 0.0], [0.2, 0.2], s)
    i, j, separation, distance, values, gradients = pairSetup(positions)
    gradient = np.array([[0.4, -1.1], [0.7, 0.2]])
    velocities = positions @ gradient.T
    shift = np.zeros_like(positions)
    center = int(np.argmin(np.linalg.norm(positions - 0.1, axis = 1)))
    shift[center] = [0.03 * s, -0.02 * s]
    densities = np.full(len(positions), rho0)
    change = shiftedVelocityChange(rho0 * s**2, densities, velocities, i, j, gradients, shift, np.ones(len(i), dtype = bool))
    wallOnly = shiftedVelocityChange(rho0 * s**2, densities, velocities, i, j, gradients, shift, np.zeros(len(i), dtype = bool))

    assert change[center] == pytest.approx(gradient @ shift[center], rel = 0.02)
    assert np.all(wallOnly == 0.0)

def waterOnABed(surface: float = 0.1) -> tuple:

    '''Water at rest to `surface` on three layers of bed particles, at the density water at rest holds.'''

    water = latticePoints([0.0, 0.0], [0.3, surface], s)
    bed = latticePoints([0.0, -3.0 * s], [0.3, 0.0], s)
    positions = np.vstack([water, bed])
    isFluid = np.arange(len(positions)) < len(water)
    densities = rho0 * (1.0 + 6.0 * constants.gravity * (surface - positions[:, 1]) / c0**2)**(1.0 / 6.0)

    return positions, isFluid, densities

def testCorrectedContinuityIsExactForAStretchingFlowUnderAFreeSurface():

    '''

    u = -a (x - x0), w = a z has no divergence. Within 2h of the free surface the plain continuity
    sum reads it as compression, a quarter of the strain rate one spacing down; with the kernel
    gradient correction it reads zero to rounding at every particle, corners included.

    '''

    water = latticePoints([0.0, 0.0], [0.3, 0.1], s)
    strain = 2.0
    velocities = np.column_stack([-strain * (water[:, 0] - 0.15), strain * water[:, 1]])
    i, j, separation, _, _, gradients = pairSetup(water)
    masses, densities, fluid = rho0 * s * s, np.full(len(water), rho0), np.ones(len(water), dtype = bool)
    plain = continuityRate(masses, velocities, i, j, gradients, len(water)) / rho0
    corrections = correctionMatrices(masses, densities, i, j, separation, gradients, fluid)
    corrected = continuityRate(masses, velocities, i, j, gradients, len(water), corrections = corrections) / rho0
    belowSurface = (water[:, 1] > 0.1 - s) & (np.abs(water[:, 0] - 0.15) < 0.05)

    assert np.all(plain[belowSurface] > 0.2 * strain)
    assert np.max(np.abs(corrected)) < 1e-9 * strain
    # Deep in the water the support is full and the correction is the lattice sum's own error, 1 percent
    deep = (np.abs(water[:, 0] - 0.15) < 0.05) & (np.abs(water[:, 1] - 0.05) < 0.5 * s)
    assert corrections[deep] == pytest.approx(np.broadcast_to(np.eye(2), corrections[deep].shape), abs = 2e-2)

def testTheCorrectionLeavesTheBoundaryAndALoneDropletAlone():

    water = latticePoints([0.0, 0.0], [0.2, 0.1], s)
    droplet = np.array([[0.1, 0.3]])
    bed = latticePoints([0.0, -3.0 * s], [0.2, 0.0], s)
    positions = np.vstack([water, droplet, bed])
    fluid = np.arange(len(positions)) < len(water) + 1
    i, j, separation, _, _, gradients = pairSetup(positions)
    corrections = correctionMatrices(rho0 * s * s, np.full(len(positions), rho0), i, j, separation, gradients, fluid)

    assert np.all(corrections[~fluid] == np.eye(2))
    assert np.all(corrections[len(water)] == np.eye(2))

def testExtrapolatedWallPressureIsHydrostaticUnderWaterAtRest():

    # Each bed particle within reach of the water gets the pressure water at rest holds at its own depth
    positions, isFluid, densities = waterOnABed()
    i, j, separation, _, values, _ = pairSetup(positions)
    pressures = taitEquationOfState(densities, rho0, c0)
    result = extrapolatedWallPressure(densities, pressures, i, j, separation, values, isFluid,
                                      np.array([0.0, -constants.gravity]))
    inner = ~isFluid & (positions[:, 1] > -2.0 * s) & (positions[:, 0] > 2.0 * h) & (positions[:, 0] < 0.3 - 2.0 * h)

    assert result[inner] == pytest.approx(pressures[inner], rel = 2e-3)
    assert np.all(result[~isFluid & (positions[:, 1] < -2.0 * s)] == 0.0)     # beyond the water's reach
    assert np.array_equal(result[isFluid], pressures[isFluid])

def testExtrapolatedWallPressureFollowsTheWaterAlongTheBed():

    # Without gravity, a pressure rising linearly along the water comes through to the bed under it,
    # where the continuity equation would leave the bed at whatever it started with
    positions, isFluid, densities = waterOnABed()
    i, j, separation, _, values, _ = pairSetup(positions)
    pressures = np.where(isFluid, 1000.0 + 5000.0 * positions[:, 0], 0.0)
    result = extrapolatedWallPressure(densities, pressures, i, j, separation, values, isFluid, np.zeros(2))
    inner = ~isFluid & (positions[:, 1] > -2.0 * s) & (positions[:, 0] > 2.0 * h) & (positions[:, 0] < 0.3 - 2.0 * h)

    assert result[inner] == pytest.approx(1000.0 + 5000.0 * positions[inner, 0], rel = 1e-9)

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

def testAPeriodicWidthWithinTwoSupportsIsRejected():

    config = SimulationConfig(domainMin = np.zeros(3), domainMax = np.ones(3), particleSpacing = s, dimensions = 3,
                              gravity = np.zeros(3), speedOfSound = c0, periodicWidth = 4.0 * s)

    with pytest.raises(ValueError):
        WcsphSolver(config)

def testAnExtrudedFlowInAPeriodicChannelStaysPlanar():

    # A random flow in (x, z), the same in every plane across a periodic y axis, must keep zero
    # spanwise velocity to rounding and wrap its particles back into the period
    rng = np.random.default_rng(5)
    planes, period = 8, 8 * s
    section = np.array([(x, z) for x in np.arange(s / 2, 0.1, s) for z in np.arange(s / 2, 0.05, s)])
    sectionVelocity = rng.normal(0.0, 0.5, size = section.shape)
    across = (np.arange(planes) + 0.5) * s
    positions = np.column_stack([np.repeat(section[:, 0], planes), np.tile(across, len(section)),
                                 np.repeat(section[:, 1], planes)])
    velocity = np.column_stack([np.repeat(sectionVelocity[:, 0], planes), np.zeros(len(positions)),
                                np.repeat(sectionVelocity[:, 1], planes)])
    particles = ParticleSystem.assemble(positions, np.empty((0, 3)), s, rho0, velocity)
    config = SimulationConfig(domainMin = np.array([-1.0, 0.0, -1.0]), domainMax = np.array([1.0, period, 1.0]),
                              particleSpacing = s, dimensions = 3, gravity = np.zeros(3), speedOfSound = c0,
                              periodicWidth = period)
    solver = WcsphSolver(config)
    solver.initialize(particles)
    for _ in range(20):
        solver.step()

    assert np.max(np.abs(particles.velocities[:, 1])) < 1e-12
    assert np.all((particles.positions[:, 1] >= 0.0) & (particles.positions[:, 1] < period))
