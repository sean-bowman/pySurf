
# -- Tests for the a priori delta-LES-SPH closure estimate (item 8a) -- #

'''

The estimator in validation/gpuSph/closureAPriori.py against known answers. The closure's viscous
power for a simple shear against the continuum dissipation mu gamma^2 per unit area, mu =
alpha h c0 rho0 / K, in the interior of a lattice; the strain rate norm exact for linear velocity
fields on jittered particles, a free surface included; the coefficients against Meringolo et al.'s
formulas and caps; the closure's diffusion zero for a linear density field. The current scheme's
viscosity and diffusion, rebuilt from a state, against the array solver's own energy budget over
one step, to rounding.

Sean Bowman [11/27/2026]

'''

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'validation' / 'gpuSph'))

from closureAPriori import (Scheme, closureCoefficients, closureDiffusivePower, closureViscousPower, elasticPower,
                            harmonicMean, pairsOf, schemeDiffusionRate, schemeViscousAccelerations, strainRateNorms)

from pysurf import constants, sphSolver, waveTheory
from pysurf.bathymetry import Bathymetry
from pysurf.sphBoundaries import BedBoundary
from pysurf.sphEnergy import elasticSlope
from pysurf.sphParticles import ParticleSystem
from pysurf.sphSolver import SimulationConfig, WcsphSolver, correctionMatrices
from pysurf.waveMakers import solitaryWaveFluid

rho0, c0, s = 1000.0, 20.0, 0.01
scheme = Scheme(smoothingLength = 3.0 * s, soundSpeed = c0, referenceDensity = rho0, alpha = 0.01, delta = 0.1,
                gravity = 9.81, kinematicViscosity = 0.0)

def lattice(columns: int, rows: int, jitter: float = 0.0, seed: int = 3) -> np.ndarray:

    '''A square lattice of spacing s from the origin, each point moved by up to `jitter` spacings.'''

    x, y = np.meshgrid((np.arange(columns) + 0.5) * s, (np.arange(rows) + 0.5) * s)
    positions = np.column_stack([x.ravel(), y.ravel()])
    if jitter:
        positions += np.random.default_rng(seed).uniform(-jitter, jitter, size = positions.shape) * s

    return positions

def testTheClosuresViscousPowerIsTheContinuumDissipationOfASimpleShear():

    # u = (gamma y, 0) dissipates mu gamma^2 per unit area, mu = alpha h c0 rho0 / K for the delta-SPH term
    gamma, alpha = 3.0, 0.02
    positions = lattice(40, 40)
    count = len(positions)
    fluid = np.ones(count, dtype = bool)
    velocities = np.column_stack([gamma * positions[:, 1], np.zeros(count)])
    densities, masses = np.full(count, rho0), np.full(count, rho0 * s * s)
    power = closureViscousPower(masses, densities, velocities, pairsOf(positions, fluid, scheme), np.full(count, alpha),
                                scheme)
    reach = 2.0 * scheme.smoothingLength
    interior = np.all((positions > reach) & (positions < 40 * s - reach), axis = 1)
    mu = alpha * scheme.smoothingLength * c0 * rho0 / scheme.factorK

    assert np.mean(power[interior]) / (s * s) == pytest.approx(-mu * gamma**2, rel = 0.02)
    assert np.all(power <= 1e-12)

@pytest.mark.parametrize('gradient', [[[0.0, 3.0], [0.0, 0.0]], [[1.5, -2.0], [4.0, -1.5]]])
def testTheStrainRateNormIsExactForALinearField(gradient):

    # Jittered particles to a free surface: the renormalized gradient is exact for a linear field
    gradient = np.array(gradient)
    positions = lattice(24, 12, jitter = 0.15)
    count = len(positions)
    fluid = np.ones(count, dtype = bool)
    velocities = positions @ gradient.T
    norms, _ = strainRateNorms(np.full(count, rho0 * s * s), np.full(count, rho0), velocities, fluid,
                               pairsOf(positions, fluid, scheme))
    strain = 0.5 * (gradient + gradient.T)

    assert norms == pytest.approx(np.full(count, np.sqrt(2.0 * np.sum(strain * strain))), rel = 1e-9)

def testTheCoefficientsFollowTheClosureAndItsCaps():

    h = scheme.smoothingLength
    norms = np.array([0.0, 2.0, 1e5])
    alphas, deltas = closureCoefficients(norms, scheme)
    width = 2.0 * h

    assert alphas[1] == pytest.approx(8.0 * (0.12 * width)**2 * 2.0 / (c0 * h), rel = 1e-12)
    assert deltas[1] == pytest.approx((1.5 * width)**2 * 2.0 / (c0 * h), rel = 1e-12)
    assert alphas[0] == deltas[0] == 0.0
    assert alphas[2] == deltas[2] == 0.2
    assert harmonicMean(np.array([0.0, 0.1]), np.array([0.0, 0.3])) == pytest.approx([0.0, 0.15])

def testTheClosuresDiffusionLeavesALinearDensityAlone():

    positions = lattice(24, 12, jitter = 0.15)
    count = len(positions)
    fluid = np.ones(count, dtype = bool)
    masses = np.full(count, rho0 * s * s)
    pairs = pairsOf(positions, fluid, scheme)
    deltas = np.random.default_rng(5).uniform(0.01, 0.2, count)
    linear = rho0 * (1.002 + positions @ np.array([0.03, -0.05]))
    curved = linear + rho0 * 2.0 * (positions[:, 0] - 0.12)**2
    flat = closureDiffusivePower(masses, linear, fluid, pairs, deltas, scheme)
    bent = closureDiffusivePower(masses, curved, fluid, pairs, deltas, scheme)
    # A corner particle with too few neighbors keeps L = I (`sphSolver.correctionFloor`), so its
    # gradient, and its pairs' differences, are not exact; everywhere else the term is zero
    corrections = correctionMatrices(masses, linear, pairs.i, pairs.j, pairs.separation, pairs.gradients, fluid)
    fallback = np.all(corrections == np.eye(2), axis = (1, 2))
    touched = fallback.copy()
    touched[pairs.i[fallback[pairs.j]]] = True
    touched[pairs.j[fallback[pairs.i]]] = True

    assert 0 < np.count_nonzero(fallback) <= 4
    assert np.max(np.abs(flat[~touched])) < 1e-8 * np.max(np.abs(bent))

def solitaryWave() -> WcsphSolver:

    '''A solitary wave of H/d = 0.3 in a short channel at 8 particles per depth, on the deltaSph scheme.'''

    depth, height, spacing = 1.0, 0.3, 1.0 / 8.0
    crestX = waveTheory.solitaryWaveHalfLength(height, depth, 0.01)
    channel = Bathymetry([0.0, 12.0], [0.0, 0.0], stillWaterLevel = depth)
    boundary = BedBoundary(channel, spacing, nLayers = 6, wallTop = 1.9, endWallTop = 1.9)
    fluidPositions, velocities = solitaryWaveFluid(channel, height, crestX, spacing)
    particles = ParticleSystem.assemble(fluidPositions, boundary.positions(), spacing, constants.freshwaterDensity,
                                        velocities)
    config = SimulationConfig(domainMin = np.array([-1.0, -1.0]), domainMax = np.array([13.0, 1.9]),
                              particleSpacing = spacing, maxTimeStep = 1.0, backend = 'numpy',
                              speedOfSound = 10.0 * waveTheory.solitaryWaveSpeed(height, depth),
                              initialSurface = lambda x: depth + waveTheory.solitaryWaveProfile(x, height, depth, crestX),
                              kernelType = 'wendlandC2', smoothingLengthRatio = 3.0, densityFilterInterval = 0,
                              densityDiffusion = 'hydrostatic', xsphEpsilon = 0.0, wallPressure = 'extrapolated',
                              continuityCorrection = True, densityFloor = 0.9, viscosityLimiter = True,
                              alphaViscosity = 0.01)
    solver = WcsphSolver(config, boundary = boundary)
    solver.initialize(particles)

    return solver

def testTheSchemesTermsRebuiltFromAStateMatchTheSolversBudget():

    solver = solitaryWave()
    solver.trackEnergy()
    for _ in range(30):
        solver.step()
    p = solver.particles
    positions, velocities, densities = p.positions.copy(), p.velocities.copy(), p.densities.copy()
    fluid, masses = p.isFluid.copy(), np.asarray(p.masses, dtype = float).copy()
    before = dict(solver.energyBudget)
    state = solver.step()
    after = solver.energyBudget
    dt = state.dt

    current = Scheme.fromSolver(solver)
    pairs = pairsOf(positions, fluid, current)
    _, gradient = strainRateNorms(masses, densities, velocities, fluid, pairs)
    viscous = schemeViscousAccelerations(masses, densities, velocities, fluid, pairs, gradient, current)
    # The budget's work over the step: sum m a . (v + v') / 2 dt, v' the kicked velocity
    kicked = velocities + p.accelerations * dt
    work = dt * np.sum((masses * np.sum(viscous * 0.5 * (velocities + kicked), axis = 1))[fluid])
    rate = schemeDiffusionRate(masses, densities, fluid, pairs, current)
    slope = elasticSlope(densities[fluid], p.densities[fluid], current.referenceDensity, current.soundSpeed)
    elastic = dt * np.sum(masses[fluid] * slope * rate[fluid])

    assert work < 0.0 and elastic != 0.0
    assert work == pytest.approx(after['viscosityFluid'] - before['viscosityFluid'], rel = 1e-9)
    assert elastic == pytest.approx(after['diffusion'] - before['diffusion'], rel = 1e-9)

def testTheSolversClosureIsTheOneTheEstimateIntegrates():

    # sphSolver's closure (turbulenceClosure = 'deltaLes') on a state of the solitary wave: its
    # coefficients, the power of its viscous term and of its diffusion at each pair's delta, as the
    # solver composes them, against the estimator's, to rounding
    solver = solitaryWave()
    for _ in range(30):
        solver.step()
    p = solver.particles
    positions, velocities, densities = p.positions.copy(), p.velocities.copy(), p.densities.copy()
    fluid, masses = p.isFluid.copy(), np.asarray(p.masses, dtype = float).copy()
    current = Scheme.fromSolver(solver)
    h, c0, rho0 = current.smoothingLength, current.soundSpeed, current.referenceDensity
    pairs = pairsOf(positions, fluid, current)
    norms, gradient = strainRateNorms(masses, densities, velocities, fluid, pairs)
    alphas, deltas = closureCoefficients(norms, current)
    solverAlphas, solverDeltas = sphSolver.closureCoefficients(gradient, fluid, h, c0)

    f = pairs.fluid
    i, j = pairs.i[f], pairs.j[f]
    relative = velocities[i] - velocities[j]
    viscosity = sphSolver.closureViscosity(relative, pairs.separation[f], pairs.distance[f], densities[i], densities[j],
                                           h, c0, rho0, solverAlphas[i], solverAlphas[j], current.kinematicViscosity)
    solverViscous = -np.sum(masses[i] * masses[j] * viscosity * np.sum(relative * pairs.gradients[f], axis = 1))
    estimateViscous = np.sum(closureViscousPower(masses, densities, velocities, pairs, alphas, current))

    corrections = sphSolver.correctionMatrices(masses, densities, pairs.i, pairs.j, pairs.separation, pairs.gradients, fluid)
    densityGradients = sphSolver.renormalizedDensityGradient(masses, densities, pairs.i, pairs.j, pairs.gradients,
                                                             corrections, fluid)
    rate = sphSolver.densityDiffusionRate(masses, densities, pairs.i, pairs.j, pairs.separation, pairs.distance,
                                          pairs.gradients, len(densities), h, c0,
                                          sphSolver.harmonicMean(solverDeltas[pairs.i], solverDeltas[pairs.j]), rho0,
                                          None, f, densityGradients)
    solverDiffusive = np.sum(elasticPower(masses, densities, rate, current)[fluid])
    estimateDiffusive = np.sum(closureDiffusivePower(masses, densities, fluid, pairs, deltas, current))

    assert solverAlphas == pytest.approx(alphas, rel = 1e-12, abs = 1e-15)
    assert solverDeltas == pytest.approx(deltas, rel = 1e-12, abs = 1e-15)
    assert estimateViscous < 0.0
    assert solverViscous == pytest.approx(estimateViscous, rel = 1e-10)
    assert solverDiffusive == pytest.approx(estimateDiffusive, rel = 1e-10)
