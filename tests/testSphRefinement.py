# -- pySurf: SPH Particle Refinement Tests -- #

'''

Checks of pysurf.sphRefinement and of the solver with particles of mixed smoothing lengths: a split
conserves mass, momentum, kinetic energy and the centre of mass exactly and leaves every other
particle's index alone; the device sums over a cloud of mixed lengths agree with the array
formulation to float32 rounding; the numpy and GPU solvers step a refined tank alike; and a still
tank refined over half its width stays at rest under the uniform tank's criteria.

Sean Bowman [11/07/2026]

'''

import math

import numpy as np
import pytest

from pysurf import constants
from pysurf.bathymetry import Bathymetry
from pysurf.sphBoundaries import BedBoundary
from pysurf.sphKernels import createKernel
from pysurf.sphNeighbors import neighborPairs
from pysurf.sphParticles import ParticleSystem, latticePoints
from pysurf.sphRefinement import refinementLevels, splitParticles
from pysurf.sphSolver import (SimulationConfig, WcsphSolver, artificialViscosity, continuityRate, correctionMatrices,
                              densityDiffusionRate, extrapolatedWallPressure, pairAccelerations, shepardDensity,
                              taitEquationOfState, xsphVelocity)

rho0, c0, alpha, epsilon = 1000.0, 20.0, 0.1, 0.5

def lattice(dimensions: int, spacing: float = 0.02, seed: int = 5) -> ParticleSystem:

    '''A block of fluid on a lattice with a row of boundary particles under it, all moving at random.'''

    rng = np.random.default_rng(seed)
    upper = [0.3, 0.2] if dimensions == 2 else [0.2, 0.12, 0.12]
    fluid = latticePoints(np.zeros(dimensions), np.array(upper), spacing)
    floor = latticePoints(np.r_[np.zeros(dimensions - 1), -spacing], np.r_[upper[:-1], 0.0], spacing)
    particles = ParticleSystem.assemble(fluid, floor, spacing, rho0, rng.normal(0.0, 1.0, size = fluid.shape))
    particles.densities = rho0 * (1.0 + rng.uniform(-0.01, 0.01, size = particles.nParticles))
    particles.pressures = taitEquationOfState(particles.densities, rho0, c0)
    particles.accelerations = rng.normal(0.0, 1.0, size = particles.positions.shape)

    return particles

@pytest.mark.parametrize('dimensions', [2, 3])
def testASplitConservesMassMomentumEnergyAndTheCentreOfMass(dimensions):

    particles = lattice(dimensions)
    fluid = particles.isFluid
    before = (particles.masses[fluid].sum(), (particles.masses[fluid, None] * particles.velocities[fluid]).sum(axis = 0),
              particles.kineticEnergy(), (particles.masses[fluid, None] * particles.positions[fluid]).sum(axis = 0))
    mask = particles.positions[:, 0] > 0.1                   # boundary particles in it are left alone
    chosen = int(np.sum(mask & fluid))
    count = particles.nParticles
    split = splitParticles(particles, mask, rho0, 0.05)
    fluid = particles.isFluid
    after = (particles.masses[fluid].sum(), (particles.masses[fluid, None] * particles.velocities[fluid]).sum(axis = 0),
             particles.kineticEnergy(), (particles.masses[fluid, None] * particles.positions[fluid]).sum(axis = 0))

    assert split == chosen > 0
    assert particles.nParticles == count + chosen * (2**dimensions - 1)
    assert after[0] == pytest.approx(before[0], rel = 1e-14)
    assert after[1] == pytest.approx(before[1], rel = 1e-12, abs = 1e-12)
    assert after[2] == pytest.approx(before[2], rel = 1e-13)
    assert after[3] == pytest.approx(before[3], rel = 1e-13)

@pytest.mark.parametrize('dimensions', [2, 3])
def testDaughtersSitOnTheMothersLatticeRefinedByTwo(dimensions):

    spacing, length = 0.02, 0.06
    particles = lattice(dimensions, spacing)
    count = particles.nParticles
    mother = int(np.flatnonzero(particles.isFluid)[7])
    centre, velocity, density = particles.positions[mother].copy(), particles.velocities[mother].copy(), particles.densities[mother]
    mask = np.zeros(count, dtype = bool)
    mask[mother] = True
    splitParticles(particles, mask, rho0, length)
    daughters = np.r_[mother, np.arange(count, particles.nParticles)]

    assert len(daughters) == 2**dimensions
    assert particles.masses[daughters] == pytest.approx(rho0 * (spacing / 2.0)**dimensions, rel = 1e-14)
    assert particles.lengths[daughters] == pytest.approx(length)          # h is kept by default
    # The corners of a square (cube) half a spacing across around the mother
    offsets = particles.positions[daughters] - centre
    assert np.abs(offsets) == pytest.approx(np.full(offsets.shape, spacing / 4.0), rel = 1e-12)
    assert len({tuple(np.sign(row)) for row in offsets}) == 2**dimensions
    assert particles.velocities[daughters] == pytest.approx(np.tile(velocity, (len(daughters), 1)))
    assert particles.densities[daughters] == pytest.approx(density)
    assert np.all(particles.isFluid[daughters])

def testOtherParticlesKeepTheirIndicesAndSplitsCount():

    particles = lattice(2)
    count = particles.nParticles
    keep = particles.positions[:, 0] < 0.15
    positions = particles.positions.copy()
    splitParticles(particles, ~keep, rho0, 0.05, lengthRatio = 0.5)
    # A daughter of a daughter is a second split
    second = np.zeros(particles.nParticles, dtype = bool)
    second[count:count + 5] = True
    splitParticles(particles, second, rho0, 0.05, lengthRatio = 0.5)
    levels = refinementLevels(particles, rho0 * 0.02**2)

    assert particles.positions[:count][keep] == pytest.approx(positions[keep])
    assert np.all(levels[:count][keep] == 0)
    assert np.all(levels[count:count + 5] == 2)
    assert set(np.unique(levels[particles.isFluid])) == {0, 1, 2}
    assert np.all(levels[~particles.isFluid] == 0)

#----------------------------------------------------------------------#
# -- Mixed smoothing lengths: the device sums against the array sums -- #
#----------------------------------------------------------------------#

def mixedCloud(dimensions: int, seed: int = 11, kernelType: str = 'wendlandC2') -> dict:

    '''A random cloud of fluid and boundary particles, part of it at half the smoothing length, with its pair terms.'''

    rng = np.random.default_rng(seed)
    count, spacing = (700, 0.04) if dimensions == 2 else (900, 0.045)
    extent = np.full(dimensions, spacing * count**(1.0 / dimensions))
    positions = 25.0 + rng.uniform(0.0, 1.0, size = (count, dimensions)) * extent
    h = (2.0 if kernelType == 'wendlandC2' else 1.3) * spacing
    fine = positions[:, 0] > 25.0 + 0.5 * extent[0]
    lengths = np.where(fine, 0.5 * h, h)
    masses = rho0 * spacing**dimensions * np.where(fine, 0.5**dimensions, 1.0) * (1.0 + rng.uniform(-0.05, 0.05, size = count))
    velocities = rng.normal(0.0, 1.0, size = (count, dimensions))
    densities = rho0 * (1.0 + rng.uniform(-0.01, 0.01, size = count))
    isFluid = rng.uniform(size = count) < 0.8
    pressures = taitEquationOfState(densities, rho0, c0)

    i, j = neighborPairs(positions, 2.0 * h)
    separation = positions[i] - positions[j]
    distance = np.linalg.norm(separation, axis = 1)
    pairLength = 0.5 * (lengths[i] + lengths[j])
    keep = distance < 2.0 * pairLength
    i, j, separation, distance, pairLength = i[keep], j[keep], separation[keep], distance[keep], pairLength[keep]
    kernel = createKernel(kernelType, dimensions)

    return dict(kernelType = kernelType, positions = positions, h = h, lengths = lengths, pairLength = pairLength,
                velocities = velocities, densities = densities, masses = masses, isFluid = isFluid, pressures = pressures,
                i = i, j = j, separation = separation, distance = distance, kernel = kernel,
                values = kernel.evaluateBatch(distance, pairLength), gradients = kernel.gradientBatch(separation, distance, pairLength))

def mixedSums(case: dict):

    '''Device pair sums over a mixed cloud, updated to its positions and lengths.'''

    wp = pytest.importorskip('warp')
    from pysurf import sphGpu
    device = 'cuda:0' if wp.get_cuda_device_count() > 0 else 'cpu'
    positions = case['positions']
    result = sphGpu.PairSums(positions.shape[1], case['h'], positions.min(axis = 0), positions.max(axis = 0), 0.0,
                             'float64', device, case['kernelType'])
    result.update(positions, case['masses'], case['isFluid'], case['lengths'])

    return result

@pytest.mark.parametrize('dimensions', [2, 3])
@pytest.mark.parametrize('kernelType', ['cubicSpline', 'wendlandC2'])
def testMixedLengthAccelerationsMatchTheArrayFormulation(dimensions, kernelType):

    case = mixedCloud(dimensions, kernelType = kernelType)
    i, j, fluid = case['i'], case['j'], case['isFluid']
    gravity = np.zeros(dimensions)
    gravity[-1] = -9.81
    viscosity = artificialViscosity(case['velocities'][i] - case['velocities'][j], case['separation'], case['distance'],
                                    0.5 * (case['densities'][i] + case['densities'][j]), case['pairLength'], c0, alpha)
    viscosity[~(fluid[i] & fluid[j])] *= 0.3
    expected = pairAccelerations(case['masses'], case['densities'], case['pressures'], i, j, case['gradients'], viscosity,
                                 len(fluid))
    expected[fluid] += gravity
    expected[~fluid] = 0.0
    result = mixedSums(case).pairAccelerations(case['velocities'], case['densities'], case['pressures'], c0, alpha, 0.3,
                                               gravity)

    assert np.sum(case['lengths'] < case['h']) > 0.3 * len(fluid)
    assert result == pytest.approx(expected, abs = 1e-5 * np.abs(expected).max())

@pytest.mark.parametrize('dimensions', [2, 3])
def testMixedLengthCorrectedDiffusiveContinuityMatchesTheArrayFormulation(dimensions):

    case = mixedCloud(dimensions)
    i, j, fluid = case['i'], case['j'], case['isFluid']
    corrections = correctionMatrices(case['masses'], case['densities'], i, j, case['separation'], case['gradients'], fluid)
    expected = continuityRate(case['masses'], case['velocities'], i, j, case['gradients'], len(fluid), fluid[i] | fluid[j],
                              corrections)
    expected += densityDiffusionRate(case['masses'], case['densities'], i, j, case['separation'], case['distance'],
                                     case['gradients'], len(fluid), case['pairLength'], c0, 0.1, rho0, 9.81,
                                     fluid[i] & fluid[j])
    pairSums = mixedSums(case)
    pairSums.setDiffusion('hydrostatic', 0.1, c0, rho0, 9.81)
    pairSums.setContinuityCorrection(True)
    matrices = pairSums.correctionMatrices(case['densities'])
    result = pairSums.continuityRate(case['velocities'], case['densities'])

    assert matrices == pytest.approx(corrections, abs = 2e-5 * np.abs(corrections).max())
    assert result == pytest.approx(expected, abs = 1e-5 * np.abs(expected).max())

@pytest.mark.parametrize('dimensions', [2, 3])
def testMixedLengthWallPressureXsphAndShepardMatchTheArrayFormulation(dimensions):

    case = mixedCloud(dimensions)
    i, j, fluid = case['i'], case['j'], case['isFluid']
    gravity = np.zeros(dimensions)
    gravity[-1] = -9.81
    pairSums = mixedSums(case)
    wall = extrapolatedWallPressure(case['densities'], case['pressures'], i, j, case['separation'], case['values'], fluid,
                                    gravity)
    pairs = fluid[i] & fluid[j]
    xsph = xsphVelocity(case['velocities'], case['masses'], case['densities'], i[pairs], j[pairs], case['values'][pairs],
                        epsilon)
    shepard = shepardDensity(case['masses'], case['densities'], i, j, case['values'],
                             case['kernel'].evaluateBatch(np.zeros(len(fluid)), case['lengths']))

    assert pairSums.wallPressure(case['densities'], case['pressures'], gravity, rho0, c0) == pytest.approx(
        wall, abs = 1e-5 * np.abs(wall).max())
    assert pairSums.xsphVelocity(case['velocities'], case['densities'], epsilon)[fluid] == pytest.approx(
        xsph[fluid], abs = 1e-5 * np.abs(xsph).max())
    assert pairSums.shepardDensity(case['densities']) == pytest.approx(shepard, rel = 2e-6)

def testTheDeviceRefusesALengthLongerThanItsGrid():

    case = mixedCloud(2)
    pairSums = mixedSums(case)

    with pytest.raises(ValueError):
        pairSums.update(case['positions'], case['masses'], case['isFluid'], np.full(len(case['masses']), 2.0 * case['h']))

#----------------------------------------------------------------------#
# -- The solver with refined particles -- #
#----------------------------------------------------------------------#

def tankSolver(backend: str, tilt: float = 10.0, lengthRatio: float = 0.5) -> tuple:

    '''The deltaSph scheme's sloshing tank on a backend, initialized, with its right half split once.'''

    from pysurf.WaterSim.scenarios import sloshingTank
    from pysurf.WaterSim.scenarios.sloshingTank import SloshingTankConfig, createSloshingTank

    layers = sloshingTank.boundaryLayers
    sloshingTank.boundaryLayers = 4
    try:
        tank = SloshingTankConfig(tankWidth = 0.2, tankHeight = 0.15, fillRatio = 0.5, initialTiltDeg = tilt,
                                  particleSpacing = 0.005)
        config, particles, boundary = createSloshingTank(tank)
    finally:
        sloshingTank.boundaryLayers = layers
    config.backend, config.maxTimeStep = backend, 1e-3
    config.kernelType, config.smoothingLengthRatio = 'wendlandC2', 2.0
    config.densityFilterInterval, config.densityDiffusion, config.xsphEpsilon = 0, 'hydrostatic', 0.0
    config.wallPressure, config.continuityCorrection, config.densityFloor = 'extrapolated', True, 0.9
    solver = WcsphSolver(config, boundary = boundary)
    solver.initialize(particles)
    middle = 0.5 * tank.tankWidth
    split = solver.refine(particles.positions[:, 0] > middle, lengthRatio)

    return solver, particles, tank, split

def testTheNumpyAndGpuSolversStepARefinedTankAlike():

    pytest.importorskip('warp')
    host, hostParticles, _, split = tankSolver('numpy')
    device, deviceParticles, _, _ = tankSolver('gpu')
    for _ in range(40):
        host.step()
        device.step()

    assert split > 0
    assert hostParticles.nParticles == deviceParticles.nParticles
    assert device.time == pytest.approx(host.time, rel = 1e-6)
    assert deviceParticles.positions == pytest.approx(hostParticles.positions, abs = 1e-7)
    assert deviceParticles.velocities == pytest.approx(hostParticles.velocities, abs = 1e-4)

def testTheTimeStepFollowsTheShortestLength():

    solver, particles, _, _ = tankSolver('numpy', tilt = 0.0)
    state = solver.step()

    assert solver.shortestLength == pytest.approx(0.5 * solver.config.smoothingLength)
    assert state.dt <= solver.config.cflNumber * solver.shortestLength / solver.speedOfSound * (1.0 + 1e-12)

@pytest.mark.parametrize('resident', [False, True])
def testAStillTankSplitOverHalfItsWidthStaysAtRest(resident):

    # The uniform still tank's speed criterion (tests/testSphGpu.py), the split particles keeping h; the daughters
    # settle out of their squares, so the displacement allowed is half a coarse spacing rather than a quarter
    pytest.importorskip('warp')
    solver, particles, tank, _ = tankSolver('gpu', tilt = 0.0, lengthRatio = 1.0)
    start = particles.positions[particles.isFluid].copy()
    if resident:
        state = solver.advance(0.2)
    else:
        while solver.time < 0.2:
            state = solver.step()
    fillHeight = tank.fillRatio * tank.tankHeight

    assert state.maxVelocity < 0.01 * math.sqrt(constants.gravity * fillHeight)
    assert np.max(np.abs(particles.positions[particles.isFluid] - start)) < 0.5 * tank.particleSpacing

def testRefiningGrowsThePistonsMask():

    spacing = 0.05
    bed = Bathymetry([0.0, 3.0], [0.0, 0.0], stillWaterLevel = 1.0)
    boundary = BedBoundary(bed, spacing, nLayers = 3, wallTop = 1.5, endWallTop = 1.5)
    fluid = latticePoints(np.array([0.0, 0.0]), np.array([3.0, 1.0]), spacing)
    particles = ParticleSystem.assemble(fluid, boundary.positions(), spacing, rho0)
    config = SimulationConfig(domainMin = np.array([-0.3, -0.3]), domainMax = np.array([3.3, 1.5]), particleSpacing = spacing,
                              referenceDensity = rho0, speedOfSound = 20.0, maxTimeStep = 1e-3)
    solver = WcsphSolver(config, boundary = boundary)
    solver.initialize(particles)
    boundary.attachPiston(particles)
    pistons = int(boundary._pistonMask.sum())
    split = solver.refine(particles.positions[:, 0] > 2.0)
    boundary.movePiston(particles, boundary.wallX + 0.01, 0.1)

    assert split > 0
    assert len(boundary._pistonMask) == particles.nParticles
    assert int(boundary._pistonMask.sum()) == pistons
