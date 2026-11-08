# -- pySurf: GPU SPH Kernel Tests -- #

'''

Checks of pysurf.sphGpu against the array formulation in pysurf.sphSolver, on random particle
clouds in 2D and 3D, with and without a periodic axis, on Warp's CPU device and on CUDA where
present: every per-particle sum agrees to float32 rounding, and a solver run on the GPU backend
follows the compiled backend.

Sean Bowman [11/05/2026]

'''

import numpy as np
import pytest

wp = pytest.importorskip('warp')

from pysurf import sphGpu
from pysurf.sphKernels import CubicSplineKernel, createKernel
from pysurf.sphNeighbors import neighborPairs
from pysurf.sphParticles import latticePoints
from pysurf.sphSolver import (artificialViscosity, continuityRate, correctionMatrices, densityDiffusionRate, extrapolatedWallPressure, pairAccelerations,
                              freeSurface, limitedVelocityDifference, shepardDensity, shiftedVelocityChange, shiftingDisplacement, shiftingFields,
                              taitEquationOfState, velocityGradients, xsphVelocity)

rho0, c0, alpha, epsilon = 1000.0, 20.0, 0.1, 0.5
devices = ['cpu'] + (['cuda:0'] if wp.get_cuda_device_count() > 0 else [])

def cloud(dimensions: int, periodic: bool, seed: int = 3, kernelType: str = 'cubicSpline') -> dict:

    '''A random particle cloud with fluid and boundary members, its pairs and per-pair terms.'''

    rng = np.random.default_rng(seed)
    count, spacing = (600, 0.04) if dimensions == 2 else (900, 0.045)
    extent = np.full(dimensions, spacing * count**(1.0 / dimensions))
    # Offset from the origin, as a real domain is, so float32 positions lose digits a separation needs
    positions = 25.0 + rng.uniform(0.0, 1.0, size = (count, dimensions)) * extent
    period = None
    if periodic:
        positions[:, 1] -= 25.0
        period = float(extent[1])
    h = (2.0 if kernelType == 'wendlandC2' else 1.3) * spacing
    velocities = rng.normal(0.0, 1.0, size = (count, dimensions))
    densities = rho0 * (1.0 + rng.uniform(-0.01, 0.01, size = count))
    masses = rho0 * spacing**dimensions * (1.0 + rng.uniform(-0.05, 0.05, size = count))
    isFluid = rng.uniform(size = count) < 0.8
    pressures = taitEquationOfState(densities, rho0, c0)

    i, j = neighborPairs(positions, 2.0 * h, period)
    separation = positions[i] - positions[j]
    if period is not None:
        separation[:, 1] -= period * np.round(separation[:, 1] / period)
    distance = np.linalg.norm(separation, axis = 1)
    kernel = createKernel(kernelType, dimensions)

    return dict(kernelType = kernelType, positions = positions, period = period, h = h, velocities = velocities, densities = densities,
                masses = masses, isFluid = isFluid, pressures = pressures, i = i, j = j, separation = separation,
                distance = distance, values = kernel.evaluateBatch(distance, h),
                gradients = kernel.gradientBatch(separation, distance, h), kernel = kernel)

def sums(case: dict, device: str, precision: str = 'float64') -> sphGpu.PairSums:

    '''Pair sums over the case's cloud on a device, updated to its positions.'''

    positions = case['positions']
    result = sphGpu.PairSums(positions.shape[1], case['h'], positions.min(axis = 0), positions.max(axis = 0),
                             case['period'] or 0.0, precision, device, case['kernelType'])
    result.update(positions, case['masses'], case['isFluid'])

    return result

cases = [(2, False), (2, True), (3, False), (3, True)]
cases = [case for case in cases if not (case[0] == 2 and case[1])]      # the periodic axis is y, in 3D only
kernelTypes = ['cubicSpline', 'wendlandC2']

@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('dimensions, periodic', cases)
@pytest.mark.parametrize('kernelType', kernelTypes)
def testAccelerationsMatchTheArrayFormulation(device, dimensions, periodic, kernelType):

    case = cloud(dimensions, periodic, kernelType = kernelType)
    i, j, fluid = case['i'], case['j'], case['isFluid']
    gravity = np.zeros(dimensions)
    gravity[-1] = -9.81
    factor = 0.3
    viscosity = artificialViscosity(case['velocities'][i] - case['velocities'][j], case['separation'],
                                    case['distance'], 0.5 * (case['densities'][i] + case['densities'][j]),
                                    case['h'], c0, alpha)
    viscosity[~(fluid[i] & fluid[j])] *= factor
    expected = pairAccelerations(case['masses'], case['densities'], case['pressures'], i, j, case['gradients'],
                                 viscosity, len(fluid))
    expected[fluid] += gravity
    expected[~fluid] = 0.0
    result = sums(case, device).pairAccelerations(case['velocities'], case['densities'], case['pressures'], c0,
                                                  alpha, factor, gravity)

    assert result == pytest.approx(expected, abs = 1e-5 * np.abs(expected).max())

@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('dimensions, periodic', cases)
@pytest.mark.parametrize('kernelType', kernelTypes)
def testLimitedAccelerationsMatchTheArrayFormulation(device, dimensions, periodic, kernelType):

    # A linear flow under noise, so the limiter meets slopes that agree and slopes that part
    case = cloud(dimensions, periodic, kernelType = kernelType)
    i, j, fluid = case['i'], case['j'], case['isFluid']
    positions = case['positions']
    velocities = (positions - positions.mean(axis = 0)) @ np.random.default_rng(11).normal(size = (dimensions, dimensions)).T \
        + 0.3 * case['velocities']
    gravity = np.zeros(dimensions)
    gravity[-1] = -9.81
    factor = 0.3
    gradients = velocityGradients(case['masses'], case['densities'], velocities, i, j, case['separation'], case['gradients'], fluid)
    relative = velocities[i] - velocities[j]
    reconstructed = limitedVelocityDifference(relative, case['separation'], case['distance'], case['h'], gradients[i], gradients[j])
    boundaryPairs = ~(fluid[i] & fluid[j])
    reconstructed[boundaryPairs] = relative[boundaryPairs]
    viscosity = artificialViscosity(relative, case['separation'], case['distance'], 0.5 * (case['densities'][i] + case['densities'][j]),
                                    case['h'], c0, alpha, reconstructed)
    viscosity[boundaryPairs] *= factor
    expected = pairAccelerations(case['masses'], case['densities'], case['pressures'], i, j, case['gradients'], viscosity, len(fluid))
    expected[fluid] += gravity
    expected[~fluid] = 0.0
    pairSums = sums(case, device)
    pairSums.setViscosityLimiter(True)
    matrices = pairSums.velocityGradients(velocities, case['densities'])
    result = pairSums.pairAccelerations(velocities, case['densities'], case['pressures'], c0, alpha, factor, gravity)

    assert matrices == pytest.approx(gradients, abs = 2e-5 * np.abs(gradients).max())
    assert result == pytest.approx(expected, abs = 1e-5 * np.abs(expected).max())

@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('dimensions, periodic', cases)
@pytest.mark.parametrize('kernelType', kernelTypes)
def testContinuityMatchesTheArrayFormulation(device, dimensions, periodic, kernelType):

    case = cloud(dimensions, periodic, kernelType = kernelType)
    i, j, fluid = case['i'], case['j'], case['isFluid']
    expected = continuityRate(case['masses'], case['velocities'], i, j, case['gradients'], len(fluid), fluid[i] | fluid[j])
    result = sums(case, device).continuityRate(case['velocities'])

    assert result == pytest.approx(expected, abs = 1e-5 * np.abs(expected).max())

@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('dimensions, periodic', cases)
@pytest.mark.parametrize('kernelType', kernelTypes)
def testCorrectedContinuityMatchesTheArrayFormulation(device, dimensions, periodic, kernelType):

    case = cloud(dimensions, periodic, kernelType = kernelType)
    i, j, fluid = case['i'], case['j'], case['isFluid']
    corrections = correctionMatrices(case['masses'], case['densities'], i, j, case['separation'], case['gradients'], fluid)
    expected = continuityRate(case['masses'], case['velocities'], i, j, case['gradients'], len(fluid), fluid[i] | fluid[j],
                              corrections)
    pairSums = sums(case, device)
    pairSums.setContinuityCorrection(True)
    matrices = pairSums.correctionMatrices(case['densities'])
    result = pairSums.continuityRate(case['velocities'], case['densities'])

    # The cloud's edges hold a few particles the correction leaves out; float32 must agree on which
    assert np.sum(np.all(np.isclose(corrections, np.eye(dimensions)), axis = (1, 2)) & fluid) > 0
    assert matrices == pytest.approx(corrections, abs = 2e-5 * np.abs(corrections).max())
    assert result == pytest.approx(expected, abs = 1e-5 * np.abs(expected).max())

@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('dimensions, periodic', cases)
@pytest.mark.parametrize('mode', ['molteni', 'hydrostatic'])
@pytest.mark.parametrize('kernelType', kernelTypes)
def testDiffusiveContinuityMatchesTheArrayFormulation(device, dimensions, periodic, mode, kernelType):

    case = cloud(dimensions, periodic, kernelType = kernelType)
    i, j, fluid = case['i'], case['j'], case['isFluid']
    gravity = 9.81 if mode == 'hydrostatic' else None
    expected = continuityRate(case['masses'], case['velocities'], i, j, case['gradients'], len(fluid), fluid[i] | fluid[j])
    expected += densityDiffusionRate(case['masses'], case['densities'], i, j, case['separation'], case['distance'],
                                     case['gradients'], len(fluid), case['h'], c0, 0.1, rho0, gravity, fluid[i] & fluid[j])
    pairSums = sums(case, device)
    pairSums.setDiffusion(mode, 0.1, c0, rho0, 9.81)
    result = pairSums.continuityRate(case['velocities'], case['densities'])

    assert result == pytest.approx(expected, abs = 1e-5 * np.abs(expected).max())

@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('dimensions, periodic', cases)
@pytest.mark.parametrize('kernelType', kernelTypes)
def testWallPressureMatchesTheArrayFormulation(device, dimensions, periodic, kernelType):

    case = cloud(dimensions, periodic, kernelType = kernelType)
    gravity = np.zeros(dimensions)
    gravity[-1] = -9.81
    expected = extrapolatedWallPressure(case['densities'], case['pressures'], case['i'], case['j'], case['separation'],
                                        case['values'], case['isFluid'], gravity)
    result = sums(case, device).wallPressure(case['densities'], case['pressures'], gravity, rho0, c0)

    assert result == pytest.approx(expected, abs = 1e-5 * np.abs(expected).max())

@pytest.mark.parametrize('device', devices)
def testDiffusionDifferencesKeepTheirDigits(device):

    # Densities of rho0 plus a gradient of 1e-4 kg/m^3 across a spacing: float32 densities round
    # at 6e-5 and would lose it, the float32 deviations from rho0 do not
    case = cloud(2, False)
    i, j, fluid = case['i'], case['j'], case['isFluid']
    densities = rho0 + 1e-4 * case['positions'][:, 0] / 0.04
    expected = densityDiffusionRate(case['masses'], densities, i, j, case['separation'], case['distance'],
                                    case['gradients'], len(fluid), case['h'], c0, 0.1, rho0, None, fluid[i] & fluid[j])
    pairSums = sums(case, device)
    pairSums.setDiffusion('molteni', 0.1, c0, rho0, 9.81)
    result = pairSums.continuityRate(np.zeros_like(case['velocities']), densities)

    assert result == pytest.approx(expected, abs = 1e-3 * np.abs(expected).max())

@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('dimensions', [2, 3])
@pytest.mark.parametrize('kernelType', kernelTypes)
def testShiftsMatchTheArrayFormulation(device, dimensions, kernelType):

    # A jittered block of water under a free surface, far from the origin, beside a wall of
    # boundary particles: interior, surface and next-to-surface particles all shift
    rng = np.random.default_rng(4)
    spacing = 0.04
    h = (2.0 if kernelType == 'wendlandC2' else 1.3) * spacing
    # The Wendland's support reaches two spacings deeper from the surface, so its 2D block is larger
    upper = ([1.2, 0.64] if kernelType == 'wendlandC2' else [0.8, 0.4]) if dimensions == 2 else [0.56, 0.56, 0.4]
    positions = 25.0 + latticePoints(np.zeros(dimensions), upper, spacing)
    positions += rng.uniform(-0.08, 0.08, size = positions.shape) * spacing
    isFluid = positions[:, 0] > 25.0 + 2.0 * spacing
    densities = rho0 * (1.0 + rng.uniform(-0.002, 0.002, size = len(positions)))
    masses = np.full(len(positions), rho0 * spacing**dimensions)
    kernel = createKernel(kernelType, dimensions)
    i, j = neighborPairs(positions, 2.0 * h)
    separation = positions[i] - positions[j]
    distance = np.linalg.norm(separation, axis = 1)
    values = kernel.evaluateBatch(distance, h)
    fields = shiftingFields(masses, densities, i, j, separation, values, kernel.gradientBatch(separation, distance, h),
                            len(positions), kernel.evaluate(spacing, h))
    timeStep, mach = 2e-4, 0.1
    size = (c0 * timeStep / h) * mach * (2.0 * h)**2
    expected, expectedChange = shiftingDisplacement(*fields, i, j, values, isFluid, size, h, kernelType)
    velocities = rng.normal(0.0, 0.3, size = positions.shape)
    expectedVelocity = shiftedVelocityChange(masses, densities, velocities, i, j,
                                             kernel.gradientBatch(separation, distance, h), expected,
                                             isFluid[i] & isFluid[j])
    pairSums = sphGpu.PairSums(dimensions, h, positions.min(axis = 0), positions.max(axis = 0), device = device,
                               kernelType = kernelType)
    pairSums.setDiffusion('none', 0.1, c0, rho0, 9.81)
    pairSums.setShifting(True, mach, spacing, c0)
    pairSums.update(positions, masses, isFluid)
    result, change, velocityChange = pairSums.shiftingDisplacement(densities, timeStep, velocities)

    surface = freeSurface(fields[1], fields[2], isFluid, h, kernelType)
    assert surface.sum() > 10 and (isFluid & ~surface & (np.abs(expected).sum(axis = 1) > 0)).sum() > 50
    assert result == pytest.approx(expected, abs = 1e-4 * np.abs(expected).max())
    assert change == pytest.approx(expectedChange, abs = 1e-3 * np.abs(expectedChange).max())
    assert velocityChange == pytest.approx(expectedVelocity, abs = 1e-4 * np.abs(expectedVelocity).max())

@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('dimensions, periodic', cases)
@pytest.mark.parametrize('kernelType', kernelTypes)
def testXsphMatchesTheArrayFormulation(device, dimensions, periodic, kernelType):

    case = cloud(dimensions, periodic, kernelType = kernelType)
    i, j, fluid = case['i'], case['j'], case['isFluid']
    pairs = fluid[i] & fluid[j]
    expected = xsphVelocity(case['velocities'], case['masses'], case['densities'], i[pairs], j[pairs],
                            case['values'][pairs], epsilon)
    result = sums(case, device).xsphVelocity(case['velocities'], case['densities'], epsilon)

    assert result[fluid] == pytest.approx(expected[fluid], abs = 1e-5 * np.abs(expected).max())
    assert result[~fluid] == pytest.approx(case['velocities'][~fluid], abs = 1e-6)

@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('dimensions, periodic', cases)
@pytest.mark.parametrize('kernelType', kernelTypes)
def testShepardMatchesTheArrayFormulation(device, dimensions, periodic, kernelType):

    case = cloud(dimensions, periodic, kernelType = kernelType)
    expected = shepardDensity(case['masses'], case['densities'], case['i'], case['j'], case['values'],
                              case['kernel'].evaluate(0.0, case['h']))
    result = sums(case, device).shepardDensity(case['densities'])

    # Two float32 sums over up to 270 neighbors, the Wendland's count in 3D
    assert result == pytest.approx(expected, rel = 2e-6)

@pytest.mark.parametrize('device', devices)
def testFloat32SeparationsLoseDigitsFarFromTheOrigin(device):

    # 25 m from the origin a float32 coordinate rounds to 2e-6 m; the float64 path keeps the sum to
    # float32 rounding of the terms, the float32 path does not
    case = cloud(3, False)
    i, j, fluid = case['i'], case['j'], case['isFluid']
    expected = continuityRate(case['masses'], case['velocities'], i, j, case['gradients'], len(fluid), fluid[i] | fluid[j])
    errors = {}
    for precision in ('float64', 'float32'):
        result = sums(case, device, precision).continuityRate(case['velocities'])
        errors[precision] = float(np.abs(result - expected).max() / np.abs(expected).max())

    assert errors['float64'] < 1e-5
    assert errors['float32'] > 3.0 * errors['float64']

def testPairSumsFollowAGrowingParticleCount():

    # Arrays grow when particles are added, as a peeling wave adds its planes
    small, large = cloud(3, False, seed = 4), cloud(3, False, seed = 5)
    pairSums = sums(small, devices[-1])
    pairSums.update(large['positions'], large['masses'], large['isFluid'])
    expected = shepardDensity(large['masses'], large['densities'], large['i'], large['j'], large['values'],
                              large['kernel'].evaluate(0.0, large['h']))

    assert pairSums.shepardDensity(large['densities']) == pytest.approx(expected, rel = 1e-6)

def testTheGpuSolverFollowsTheCompiledSolver():

    # A sloshing tank stepped by both backends stays together to float32 rounding
    pytest.importorskip('numba')
    from pysurf.sphSolver import WcsphSolver
    from pysurf.WaterSim.scenarios.sloshingTank import SloshingTankConfig, createSloshingTank

    states = []
    for backend in ('compiled', 'gpu'):
        tank = SloshingTankConfig(tankWidth = 0.2, tankHeight = 0.15, fillRatio = 0.5, initialTiltDeg = 10.0,
                                  particleSpacing = 0.005)
        config, particles, boundary = createSloshingTank(tank)
        config.backend = backend
        solver = WcsphSolver(config, boundary = boundary)
        solver.initialize(particles)
        for _ in range(60):
            solver.step()
        states.append((particles.positions.copy(), particles.velocities.copy(), solver.time))

    assert states[1][2] == pytest.approx(states[0][2], rel = 1e-5)
    assert states[1][0] == pytest.approx(states[0][0], abs = 1e-7)
    assert states[1][1] == pytest.approx(states[0][1], abs = 1e-4)

def sloshingSolver(backend: str, tilt: float = 10.0, **options):

    '''A small sloshing tank on a backend, initialized, with any further configuration fields.'''

    from pysurf.sphSolver import WcsphSolver
    from pysurf.WaterSim.scenarios.sloshingTank import SloshingTankConfig, createSloshingTank

    tank = SloshingTankConfig(tankWidth = 0.2, tankHeight = 0.15, fillRatio = 0.5, initialTiltDeg = tilt,
                              particleSpacing = 0.005)
    config, particles, boundary = createSloshingTank(tank)
    config.backend = backend
    config.maxTimeStep = 1e-3
    for name, value in options.items():
        setattr(config, name, value)
    solver = WcsphSolver(config, boundary = boundary)
    solver.initialize(particles)

    return solver, particles, tank

def testDeviceStepsFollowHostSteps():

    # The same 60 steps on the device and from the host stay together to float32 rounding
    host, hostParticles, _ = sloshingSolver('gpu')
    for _ in range(60):
        host.step()
    device, deviceParticles, _ = sloshingSolver('gpu')
    state = device.advance(1.0, maxSteps = 60)

    assert state.step == 60
    assert device.time == pytest.approx(host.time, rel = 1e-5)
    assert deviceParticles.positions == pytest.approx(hostParticles.positions, abs = 1e-7)
    assert deviceParticles.velocities == pytest.approx(hostParticles.velocities, abs = 1e-4)
    assert deviceParticles.densities == pytest.approx(hostParticles.densities, rel = 1e-6)

def testDeviceStepsFollowHostStepsWithTheViscosityLimiter():

    host, hostParticles, _ = sloshingSolver('gpu', viscosityLimiter = True)
    for _ in range(60):
        host.step()
    device, deviceParticles, _ = sloshingSolver('gpu', viscosityLimiter = True)
    device.advance(1.0, maxSteps = 60)

    assert device.time == pytest.approx(host.time, rel = 1e-5)
    assert deviceParticles.positions == pytest.approx(hostParticles.positions, abs = 1e-7)
    assert deviceParticles.velocities == pytest.approx(hostParticles.velocities, abs = 1e-4)

def testTheArraySolverFollowsTheGpuSolverWithTheViscosityLimiter():

    # The numpy solver's limited term, boundary pairs left whole, against the device's over 30 steps
    reference, referenceParticles, _ = sloshingSolver('numpy', viscosityLimiter = True)
    for _ in range(30):
        reference.step()
    device, deviceParticles, _ = sloshingSolver('gpu', viscosityLimiter = True)
    device.advance(1.0, maxSteps = 30)

    assert device.time == pytest.approx(reference.time, rel = 1e-4)
    assert deviceParticles.positions == pytest.approx(referenceParticles.positions, abs = 1e-6)
    assert deviceParticles.velocities == pytest.approx(referenceParticles.velocities, abs = 1e-3)

def testDeviceStepsCarryNegativePressureAsTheHostDoes():

    # With the fluid allowed tension, the device's pressure follows the host's, negative values included;
    # the water starts 0.2 percent light, so its upper part is in tension
    host, hostParticles, _ = sloshingSolver('gpu', negativePressure = True)
    hostParticles.densities[hostParticles.isFluid] *= 0.998
    for _ in range(60):
        host.step()
    device, deviceParticles, _ = sloshingSolver('gpu', negativePressure = True)
    deviceParticles.densities[deviceParticles.isFluid] *= 0.998
    device.advance(1.0, maxSteps = 60)
    fluid = hostParticles.isFluid

    assert np.min(hostParticles.pressures[fluid]) < -1.0
    assert np.min(hostParticles.pressures[~fluid]) >= 0.0
    assert deviceParticles.positions == pytest.approx(hostParticles.positions, abs = 1e-7)
    assert deviceParticles.pressures == pytest.approx(hostParticles.pressures, abs = 1e-3 * np.abs(hostParticles.pressures).max())

@pytest.mark.parametrize('resident', [False, True])
def testAHydrostaticStillTankStaysAtRest(resident):

    # The still tank of tests/testSphSolver.py, with its criteria, from the host and on the device
    import math
    from pysurf import constants

    solver, particles, tank = sloshingSolver('gpu', tilt = 0.0)
    start = particles.positions[particles.isFluid].copy()
    if resident:
        state = solver.advance(0.2)
    else:
        while solver.time < 0.2:
            state = solver.step()
    fillHeight = tank.fillRatio * tank.tankHeight

    assert state.maxVelocity < 0.01 * math.sqrt(constants.gravity * fillHeight)
    assert np.max(np.abs(particles.positions[particles.isFluid] - start)) < 0.25 * tank.particleSpacing

@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('dimensions', [2, 3])
def testBedEnforcementMatchesTheHost(device, dimensions):

    # Fluid scattered around a reef profile, a moving piston and both end walls, enforced by
    # BedBoundary on the host and by the device kernel
    from pysurf.bathymetry import Bathymetry
    from pysurf.sphBoundaries import BedBoundary
    from pysurf.sphParticles import ParticleSystem

    rng = np.random.default_rng(7)
    bed = Bathymetry.reefBreak(depth = 1.0, toeX = 2.0, faceSlope = 0.2, reefDepth = 0.3, reefLength = 2.0)
    s, width = 0.05, (0.6 if dimensions == 3 else None)
    boundary = BedBoundary(bed, s, nLayers = 3, wallTop = 1.5, width = width, endWallTop = 1.5 if dimensions == 2 else None)
    count = 4000
    fluid = np.column_stack([rng.uniform(bed.startX - 0.2, bed.endX + 0.2, count)] +
                            ([rng.uniform(-0.1, 0.7, count)] if dimensions == 3 else []) +
                            [rng.uniform(-1.2, 1.2, count)])
    host = ParticleSystem.assemble(fluid, boundary.positions(), s, 1000.0, rng.normal(0.0, 1.0, (count, dimensions)))
    boundary.attachPiston(host)
    boundary.movePiston(host, bed.startX + 0.03, 0.4)
    resident = ParticleSystem(**{name: getattr(host, name).copy() for name in
                                 ('positions', 'velocities', 'accelerations', 'densities', 'pressures', 'masses',
                                  'isFluid')})
    moved = host.positions.copy()
    pairSums = sphGpu.PairSums(dimensions, 1.3 * s, resident.positions.min(axis = 0),
                               resident.positions.max(axis = 0), device = device)
    pairSums.push(resident)
    pairSums.setBoundary(boundary, resident)
    pairSums.enforceBoundary()
    pairSums.pull(resident)

    boundary.enforce(host)
    assert np.count_nonzero(np.any(host.positions != moved, axis = 1)) > 0.2 * count
    assert resident.positions == pytest.approx(host.positions, abs = 1e-12)
    assert resident.velocities == pytest.approx(host.velocities, abs = 1e-12)

def testDeviceStepsFollowHostStepsOverABed():

    # A solitary wave in a short channel: 100 steps from the host and on the device
    from pysurf import constants, waveTheory
    from pysurf.bathymetry import Bathymetry
    from pysurf.sphBoundaries import BedBoundary
    from pysurf.sphParticles import ParticleSystem
    from pysurf.sphSolver import SimulationConfig, WcsphSolver
    from pysurf.waveMakers import solitaryWaveFluid

    def build():
        depth, height, spacing = 1.0, 0.3, 1.0 / 8.0
        crestX = waveTheory.solitaryWaveHalfLength(height, depth, 0.01)
        channel = Bathymetry([0.0, 12.0], [0.0, 0.0], stillWaterLevel = depth)
        boundary = BedBoundary(channel, spacing, nLayers = 3, wallTop = 1.9, endWallTop = 1.9)
        fluid, velocities = solitaryWaveFluid(channel, height, crestX, spacing)
        particles = ParticleSystem.assemble(fluid, boundary.positions(), spacing, constants.freshwaterDensity,
                                            velocities)
        config = SimulationConfig(domainMin = np.array([-0.5, -0.5]), domainMax = np.array([12.5, 1.9]),
                                  particleSpacing = spacing, maxTimeStep = 1.0,
                                  speedOfSound = 10.0 * waveTheory.solitaryWaveSpeed(height, depth),
                                  alphaViscosity = 0.02, backend = 'gpu',
                                  initialSurface = lambda x: depth + waveTheory.solitaryWaveProfile(x, height, depth,
                                                                                                   crestX))
        solver = WcsphSolver(config, boundary = boundary)
        solver.initialize(particles)
        return solver, particles

    host, hostParticles = build()
    for _ in range(100):
        host.step()
    device, deviceParticles = build()
    device.advance(10.0, maxSteps = 100)

    assert device.time == pytest.approx(host.time, rel = 1e-5)
    assert deviceParticles.positions == pytest.approx(hostParticles.positions, abs = 1e-6)
    assert deviceParticles.velocities == pytest.approx(hostParticles.velocities, abs = 1e-4)
