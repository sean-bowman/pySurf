
# -- Tests for the SPH energy budget -- #

'''

The elastic energy of the Tait equation of state against a numerical integral of p / rho^2, and its
secant slope against p / rho^2. The budget against the change of K + P + E_C over steps of the
array solver and of the device-resident steps, a solitary wave over a bed on the deltaSph scheme
(continuity correction, hydrostatic diffusion, extrapolated wall pressure, limited viscosity, density
floor): its terms close on that change to rounding. The viscous term against ever adding energy,
the correction's term against appearing without the correction, the device's terms against the
array solver's, and the device step against changing when the budget is on.

Sean Bowman [11/22/2026]

'''

import numpy as np
import pytest
from scipy.integrate import quad

from pysurf import constants, waveTheory
from pysurf.bathymetry import Bathymetry
from pysurf.sphBoundaries import BedBoundary
from pysurf.sphEnergy import budgetGroups, elasticEnergy, elasticSlope, energyTerms
from pysurf.sphParticles import ParticleSystem
from pysurf.sphSolver import SimulationConfig, WcsphSolver, taitEquationOfState
from pysurf.waveMakers import solitaryWaveFluid

rho0, c0 = 1000.0, 20.0

def solitaryWave(backend: str, **options) -> WcsphSolver:

    '''A solitary wave of H/d = 0.3 in a short channel at 8 particles per depth, on the deltaSph scheme.'''

    depth, height, spacing = 1.0, 0.3, 1.0 / 8.0
    crestX = waveTheory.solitaryWaveHalfLength(height, depth, 0.01)
    channel = Bathymetry([0.0, 12.0], [0.0, 0.0], stillWaterLevel = depth)
    boundary = BedBoundary(channel, spacing, nLayers = 6, wallTop = 1.9, endWallTop = 1.9)
    fluid, velocities = solitaryWaveFluid(channel, height, crestX, spacing)
    particles = ParticleSystem.assemble(fluid, boundary.positions(), spacing, constants.freshwaterDensity, velocities)
    scheme = dict(kernelType = 'wendlandC2', smoothingLengthRatio = 3.0, densityFilterInterval = 0,
                  densityDiffusion = 'hydrostatic', xsphEpsilon = 0.0, wallPressure = 'extrapolated',
                  continuityCorrection = True, densityFloor = 0.9, viscosityLimiter = True, alphaViscosity = 0.01)
    scheme.update(options)
    config = SimulationConfig(domainMin = np.array([-1.0, -1.0]), domainMax = np.array([13.0, 1.9]),
                              particleSpacing = spacing, maxTimeStep = 1.0, backend = backend,
                              speedOfSound = 10.0 * waveTheory.solitaryWaveSpeed(height, depth),
                              initialSurface = lambda x: depth + waveTheory.solitaryWaveProfile(x, height, depth, crestX),
                              **scheme)
    solver = WcsphSolver(config, boundary = boundary)
    solver.initialize(particles)

    return solver

def fluidEnergy(solver: WcsphSolver) -> float:

    '''K + P + E_C of the fluid [J/m].'''

    state = solver.currentState

    return state.kineticEnergy + state.potentialEnergy + solver.elasticEnergy

def throughput(budget: dict) -> float:

    '''The energy the terms move, summed without sign [J/m]: the scale closure is measured against.'''

    return sum(abs(value) for value in budget.values())

@pytest.mark.parametrize('tension', [False, True])
def testElasticEnergyIsTheIntegralOfPressureOverDensitySquared(tension):

    for density in (985.0, 999.0, 1000.0, 1000.2, 1003.0, 1030.0):
        integral, _ = quad(lambda s: taitEquationOfState(s, rho0, c0, clampNegative = not tension) / s**2, rho0, density,
                           epsabs = 1e-14, epsrel = 1e-12)
        assert float(elasticEnergy(density, rho0, c0, tension)) == pytest.approx(integral, rel = 1e-9, abs = 1e-15)
    # Near rho0 it is linear acoustics' c0^2 x^2 / 2
    assert float(elasticEnergy(1000.01, rho0, c0)) == pytest.approx(0.5 * c0**2 * 1e-5**2, rel = 1e-4)

def testTheElasticSlopeIsTheSecantAndTendsToPressureOverDensitySquared():

    before, after = np.array([1001.0, 1002.0, 1002.0]), np.array([1003.0, 1002.0 + 1e-9, 1002.0])
    slope = elasticSlope(before, after, rho0, c0)
    secant = (elasticEnergy(after[0], rho0, c0) - elasticEnergy(before[0], rho0, c0)) / 2.0
    pointSlope = taitEquationOfState(1002.0, rho0, c0) / 1002.0**2

    assert slope[0] == pytest.approx(secant, rel = 1e-12)
    assert slope[1] == pytest.approx(pointSlope, rel = 1e-6)
    assert slope[2] == pytest.approx(pointSlope, rel = 1e-12)

def testTheBudgetClosesOnTheArraySolver():

    solver = solitaryWave('numpy')
    solver.trackEnergy()
    start = fluidEnergy(solver)
    for _ in range(40):
        solver.step()
    budget = solver.energyBudget
    groups = budgetGroups(budget)

    assert set(budget) == set(energyTerms)
    assert groups['total'] == pytest.approx(fluidEnergy(solver) - start, abs = 1e-9 * throughput(budget))
    # The limited viscosity never feeds energy into the flow, and the correction is at work
    assert budget['viscosityFluid'] <= 0.0
    assert budget['correction'] != 0.0
    # Free-slip walls with no viscosity against them, no shifting, filter or refinement
    for name in ('viscosityWall', 'shifting', 'filter', 'refinement'):
        assert budget[name] == 0.0

def testWithoutTheCorrectionItsTermIsZero():

    solver = solitaryWave('numpy', continuityCorrection = False)
    solver.trackEnergy()
    start = fluidEnergy(solver)
    for _ in range(20):
        solver.step()
    budget = solver.energyBudget

    assert budget['correction'] == pytest.approx(0.0, abs = 1e-12 * throughput(budget))
    assert budgetGroups(budget)['total'] == pytest.approx(fluidEnergy(solver) - start, abs = 1e-9 * throughput(budget))

def testRefinementEntersTheBudget():

    solver = solitaryWave('numpy')
    solver.trackEnergy()
    start = fluidEnergy(solver)
    for _ in range(5):
        solver.step()
    p = solver.particles
    crest = p.isFluid & (p.positions[:, 1] > 0.8)
    solver.refine(crest)
    for _ in range(5):
        solver.step()
    budget = solver.energyBudget

    assert budgetGroups(budget)['total'] == pytest.approx(fluidEnergy(solver) - start, abs = 1e-9 * throughput(budget))

def testTheBreakingBudgetClosesAndItsSharesSumToOne(monkeypatch):

    # Frames whose budget closes on their energies by construction, the still reference's too; over
    # active breaking the closure is then zero and the shares, elastic energy included, sum to one
    from pysurf.BreakingWave import BreakingWave, WaveFrame

    def frames(scale: float) -> list:
        made = []
        for k, t in enumerate(np.linspace(0.0, 1.0, 11)):
            terms = {name: 0.0 for name in energyTerms}
            terms['viscosityFluid'] = -2.0 * scale * t
            terms['pressureWall'] = 0.5 * scale * t
            terms['diffusion'] = -0.25 * scale * t**2
            elastic = 0.1 * scale * t
            mechanical = sum(terms.values()) - elastic
            made.append(WaveFrame(time = t, positions = np.zeros((1, 2)), speeds = np.zeros(1), kineticEnergy = mechanical,
                                  potentialEnergy = 0.0, elasticEnergy = elastic, budget = terms))
        return made

    wave, still = BreakingWave(), BreakingWave()
    wave.frames, still.frames = frames(1.0), frames(0.1)
    times = np.array([frame.time for frame in wave.frames])
    energy = np.array([frame.kineticEnergy for frame in wave.frames]) - np.array([frame.kineticEnergy for frame in still.frames])
    start, end = 0.25, 0.85
    monkeypatch.setattr(wave, 'breakingDissipation', lambda reference = None: {
        'breakingTime': start, 'activeEndTime': end, 'breakingEnergy': float(np.interp(start, times, energy)),
        'activeEndEnergy': float(np.interp(end, times, energy))})
    budget = wave.breakingBudget(reference = still)

    assert budget['closure'] == pytest.approx(0.0, abs = 1e-12)
    assert sum(budget['shareOfLoss'].values()) == pytest.approx(1.0, rel = 1e-12)
    assert budget['terms']['viscosityFluid'] == pytest.approx(-2.0 * 0.9 * (end - start), rel = 1e-12)

def testTheCompiledBackendHasNoBudget():

    config = SimulationConfig(domainMin = np.zeros(2), domainMax = np.ones(2), backend = 'compiled')
    with pytest.raises(ValueError):
        WcsphSolver(config).trackEnergy()

wp = pytest.importorskip('warp')
cuda = pytest.mark.skipif(wp.get_cuda_device_count() == 0, reason = 'needs a CUDA device')

@cuda
def testTheDeviceBudgetClosesAcrossAdvanceCalls():

    solver = solitaryWave('gpu')
    solver.trackEnergy()
    start = fluidEnergy(solver)
    for until in (0.05, 0.1, 0.15):
        solver.advance(until)
    budget = solver.energyBudget

    # The parts are summed in float32 and the state integrated in float64
    assert budgetGroups(budget)['total'] == pytest.approx(fluidEnergy(solver) - start, abs = 1e-5 * throughput(budget))
    assert budget['viscosityFluid'] <= 0.0

@cuda
def testTheDeviceBudgetFollowsTheArrayBudget():

    reference = solitaryWave('numpy')
    reference.trackEnergy()
    for _ in range(40):
        reference.step()
    device = solitaryWave('gpu')
    device.trackEnergy()
    device.advance(10.0, maxSteps = 40)
    expected, budget = reference.energyBudget, device.energyBudget
    scale = throughput(expected)

    for name in energyTerms:
        assert budget[name] == pytest.approx(expected[name], abs = 1e-3 * scale), name

@cuda
def testTheBudgetLeavesTheDeviceStepAlone():

    plain = solitaryWave('gpu')
    plain.advance(10.0, maxSteps = 40)
    tracked = solitaryWave('gpu')
    tracked.trackEnergy()
    tracked.advance(10.0, maxSteps = 40)

    for name in ('positions', 'velocities', 'densities'):
        assert np.array_equal(getattr(tracked.particles, name), getattr(plain.particles, name)), name
