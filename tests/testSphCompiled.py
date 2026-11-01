
# -- pySurf: Compiled SPH Kernel Tests -- #

'''

Checks of pysurf.sphCompiled against the array formulation in pysurf.sphSolver and
pysurf.sphNeighbors, on random particle clouds in 2D and 3D, with and without a periodic axis: the
neighbor rows hold exactly the brute-force pairs, and every per-particle sum agrees to rounding.

Sean Bowman [10/31/2026]

'''

import numpy as np
import pytest

pytest.importorskip('numba')

from pysurf import sphCompiled
from pysurf.sphKernels import CubicSplineKernel
from pysurf.sphNeighbors import neighborPairs
from pysurf.sphSolver import (artificialViscosity, continuityRate, pairAccelerations, shepardDensity,
                              taitEquationOfState, xsphVelocity)

rho0, c0, alpha, epsilon = 1000.0, 20.0, 0.1, 0.5

def cloud(dimensions: int, periodic: bool, seed: int = 3) -> dict:

    '''A random particle cloud with fluid and boundary members, its pairs and per-pair terms.'''

    rng = np.random.default_rng(seed)
    count, spacing = (600, 0.04) if dimensions == 2 else (900, 0.045)
    extent = np.full(dimensions, spacing * count**(1.0 / dimensions))
    positions = rng.uniform(0.0, 1.0, size = (count, dimensions)) * extent
    period = float(extent[1]) if periodic else None
    h = 1.3 * spacing
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
    kernel = CubicSplineKernel(dimensions)

    return dict(positions = positions, period = period, h = h, velocities = velocities, densities = densities,
                masses = masses, isFluid = isFluid, pressures = pressures, i = i, j = j, separation = separation,
                distance = distance, values = kernel.evaluateBatch(distance, h),
                gradients = kernel.gradientBatch(separation, distance, h),
                sigma = sphCompiled.cubicSplineNormalization(h, dimensions), kernel = kernel)

def rows(case: dict) -> tuple[np.ndarray, np.ndarray]:

    return sphCompiled.neighborList(case['positions'], 2.0 * case['h'], case['period'] or 0.0)

cases = [(2, False), (2, True), (3, False), (3, True)]

@pytest.mark.parametrize('dimensions, periodic', cases)
def testNeighborRowsHoldExactlyThePairs(dimensions, periodic):

    case = cloud(dimensions, periodic)
    offsets, neighbors = rows(case)
    owner = np.repeat(np.arange(len(case['positions'])), np.diff(offsets))
    found = {(min(a, b), max(a, b)) for a, b in zip(owner.tolist(), neighbors.tolist())}

    assert len(neighbors) == 2 * len(case['i'])
    assert found == set(zip(case['i'].tolist(), case['j'].tolist()))

@pytest.mark.parametrize('dimensions, periodic', cases)
def testAccelerationsMatchTheArrayFormulation(dimensions, periodic):

    case = cloud(dimensions, periodic)
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
    offsets, neighbors = rows(case)
    result = sphCompiled.pairAccelerations(case['positions'], case['velocities'], case['densities'],
                                           case['pressures'], case['masses'], fluid, offsets, neighbors, case['h'],
                                           case['sigma'], c0, alpha, factor, gravity, case['period'] or 0.0)

    assert result == pytest.approx(expected, abs = 1e-9 * np.abs(expected).max())

@pytest.mark.parametrize('dimensions, periodic', cases)
def testContinuityMatchesTheArrayFormulation(dimensions, periodic):

    case = cloud(dimensions, periodic)
    i, j, fluid = case['i'], case['j'], case['isFluid']
    expected = continuityRate(case['masses'], case['velocities'], i, j, case['gradients'], len(fluid), fluid[i] | fluid[j])
    offsets, neighbors = rows(case)
    result = sphCompiled.continuityRate(case['positions'], case['velocities'], case['masses'], fluid, offsets,
                                        neighbors, case['h'], case['sigma'], case['period'] or 0.0)

    assert result == pytest.approx(expected, abs = 1e-9 * np.abs(expected).max())

@pytest.mark.parametrize('dimensions, periodic', cases)
def testXsphMatchesTheArrayFormulation(dimensions, periodic):

    case = cloud(dimensions, periodic)
    i, j, fluid = case['i'], case['j'], case['isFluid']
    pairs = fluid[i] & fluid[j]
    expected = xsphVelocity(case['velocities'], case['masses'], case['densities'], i[pairs], j[pairs],
                            case['values'][pairs], epsilon)
    offsets, neighbors = rows(case)
    result = sphCompiled.xsphVelocity(case['positions'], case['velocities'], case['densities'], case['masses'],
                                      fluid, offsets, neighbors, case['h'], case['sigma'], epsilon,
                                      case['period'] or 0.0)

    assert result[fluid] == pytest.approx(expected[fluid], abs = 1e-12)
    assert result[~fluid] == pytest.approx(case['velocities'][~fluid], abs = 0.0)

@pytest.mark.parametrize('dimensions, periodic', cases)
def testShepardMatchesTheArrayFormulation(dimensions, periodic):

    case = cloud(dimensions, periodic)
    expected = shepardDensity(case['masses'], case['densities'], case['i'], case['j'], case['values'],
                              case['kernel'].evaluate(0.0, case['h']))
    offsets, neighbors = rows(case)
    result = sphCompiled.shepardDensity(case['positions'], case['densities'], case['masses'], offsets, neighbors,
                                        case['h'], case['sigma'], case['period'] or 0.0)

    assert result == pytest.approx(expected, rel = 1e-12)

def testTheCompiledSolverFollowsTheArraySolver():

    # A sloshing tank stepped by both backends stays the same to rounding
    from pysurf.sphSolver import WcsphSolver
    from pysurf.WaterSim.scenarios.sloshingTank import SloshingTankConfig, createSloshingTank

    states = []
    for backend in ('numpy', 'compiled'):
        tank = SloshingTankConfig(tankWidth = 0.2, tankHeight = 0.15, fillRatio = 0.5, initialTiltDeg = 10.0,
                                  particleSpacing = 0.005)
        config, particles, boundary = createSloshingTank(tank)
        config.backend = backend
        solver = WcsphSolver(config, boundary = boundary)
        solver.initialize(particles)
        for _ in range(60):
            solver.step()
        states.append((particles.positions.copy(), particles.velocities.copy(), solver.time))

    assert states[1][2] == pytest.approx(states[0][2], rel = 1e-12)
    assert states[1][0] == pytest.approx(states[0][0], abs = 1e-12)
    assert states[1][1] == pytest.approx(states[0][1], abs = 1e-9)
