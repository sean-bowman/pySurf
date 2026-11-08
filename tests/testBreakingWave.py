
# -- pySurf: Breaking Wave Tests -- #

'''

Smoke tests of pysurf.BreakingWave: inputs are checked, the geometry follows its definitions, a
coarse half-second run records frames and tracker history, and the transient writes as a GIF with
one image per frame. Breaking itself is validated in tests/testValidationBreaking.py.

Sean Bowman [10/11/2026]

'''

import math

import numpy as np
import pytest
from PIL import Image

from pysurf import constants, waveTheory
from pysurf.bathymetry import Bathymetry
from pysurf.BreakingWave import BreakingWave
from pysurf.errors import InvalidInputError
from pysurf.sphRefinement import refinementLevels

@pytest.fixture(scope = 'module')
def shortRun() -> BreakingWave:

    wave = BreakingWave(particlesPerDepth = 6, frameInterval = 0.25, endTime = 0.5)
    wave.run(progress = False)

    return wave

@pytest.mark.parametrize('inputs', [{'depth': -1.0}, {'beachSlope': 0.0}, {'particlesPerDepth': 2},
                                    {'waveHeight': 0.9}, {'scheme': 'leapfrog'}, {'smoothingLengthRatio': 0.5},
                                    {'backend': 'fortran'}])
def testInvalidInputsAreRejected(inputs):

    with pytest.raises(InvalidInputError):
        BreakingWave(**inputs)

def testTheGeometryFollowsItsDefinitions():

    wave = BreakingWave(depth = 2.0, waveHeight = 0.6, beachSlope = 1.0 / 20.0, particlesPerDepth = 10)

    assert wave.spacing == pytest.approx(0.2)
    assert waveTheory.solitaryWaveProfile(0.0, 0.6, 2.0, wave.crestX) == pytest.approx(0.006)
    assert wave.toeX == pytest.approx(2.0 * wave.crestX)
    assert wave.shorelineX == pytest.approx(wave.toeX + 40.0)
    assert wave.slopeParameter == pytest.approx(1.521 / 20.0 / math.sqrt(0.3))
    assert wave.predictedBreakerType == 'plunging'

def testTheArrivalEstimateLiesBetweenItsBounds():

    '''Slower than the offshore celerity over the whole path, faster than the shallowest point.'''

    wave = BreakingWave()
    distance = wave.shorelineX - wave.crestX
    fastest = distance / waveTheory.solitaryWaveSpeed(wave.waveHeight, wave.depth)
    slowest = distance / math.sqrt(9.80665 * wave.waveHeight)

    assert fastest < wave.estimatedArrivalTime() < slowest

def testAShortRunRecordsFramesAndHistory(shortRun):

    assert [frame.time for frame in shortRun.frames][:1] == [0.0]
    assert len(shortRun.frames) == 3
    assert len(shortRun.tracker.times) > 10
    assert shortRun.frames[-1].positions.shape == (shortRun.particles.nFluid, 2)
    assert np.all(np.isfinite(shortRun.frames[-1].positions))
    # The crest has moved shoreward
    assert shortRun.tracker.crestX[-1] > shortRun.crestX

def testTheSummaryCarriesThePrediction(shortRun):

    summary = shortRun.summary()

    assert summary['predictedBreakerType'] == 'plunging'
    assert summary['particlesPerDepth'] == 6
    assert summary['broke'] is False

def testTheTransientWritesAsAGif(shortRun, tmp_path):

    path = shortRun.writeAnimation(str(tmp_path / 'wave.gif'), fps = 4)

    with Image.open(path) as gif:
        assert gif.n_frames == len(shortRun.frames)

def testAFrameDraws(shortRun):

    figure = shortRun.drawFrame(-1)

    assert len(figure.axes) == 3

def testTheDeltaSchemeStartsFromTheExactWave():

    '''The delta-SPH scheme takes the Wendland kernel at h = 3 s, six wall layers, and the exact wave's pressure.'''

    wave = BreakingWave(particlesPerDepth = 6, frameInterval = 0.1, endTime = 0.2, scheme = 'deltaSph')
    wave.build()
    config, p = wave.solver.config, wave.particles

    assert wave.smoothingLengthRatio == 3.0 and wave.boundaryLayers == 6 and config.alphaViscosity == 0.01
    assert (config.kernelType, config.densityDiffusion, config.wallPressure) == ('wendlandC2', 'hydrostatic', 'extrapolated')
    assert config.densityFilterInterval == 0 and config.xsphEpsilon == 0.0 and config.continuityCorrection
    assert config.densityFloor == 0.9 and config.viscosityLimiter
    # Under the crest the bottom row's pressure head lies between the still water's and the
    # hydrostatic head under the crest: the water there decelerates downward, so p < rho g (d + H - z)
    bottom = p.isFluid & (np.abs(p.positions[:, 1] - 0.5 * wave.spacing) < 1e-9)
    crestRow = bottom & (np.abs(p.positions[:, 0] - wave.crestX) < wave.spacing)
    head = float(p.pressures[crestRow][0]) / (constants.freshwaterDensity * constants.gravity)
    assert 1.0 - 0.5 * wave.spacing < head < 1.3 - 0.5 * wave.spacing

    wave.run(progress = False)
    assert len(wave.frames) == 3 and np.all(np.isfinite(wave.frames[-1].positions))
    assert wave.tracker.crestX[-1] > wave.crestX

def testSolverOptionsOverrideTheDeltaScheme():

    '''A solver option replaces the scheme's own setting, so the plain artificial viscosity stays available.'''

    wave = BreakingWave(particlesPerDepth = 6, frameInterval = 0.1, endTime = 0.2, scheme = 'deltaSph', backend = 'numpy',
                        solverOptions = {'viscosityLimiter': False})
    wave.build()

    assert not wave.solver.config.viscosityLimiter
    assert wave.solver.config.continuityCorrection

def testAGivenBedSetsTheToeTheEndWallAndTheArrival():

    '''A reef shelf at 0.371 d behind a 0.0693 face from x = 30 d, ending at a wall at 60 d: the Surf Ranch bed.'''

    slope, shelf = 0.0693, 0.371
    bed = Bathymetry.reefBreak(depth = 1.0, toeX = 30.0, faceSlope = slope, reefDepth = shelf,
                               reefLength = 60.0 - (30.0 + (1.0 - shelf) / slope))
    wave = BreakingWave(depth = 1.0, waveHeight = 0.6, beachSlope = slope, bathymetry = bed, startX = 15.0,
                        particlesPerDepth = 4)

    assert wave.crestX == 15.0 and wave.toeX == pytest.approx(30.0)
    assert wave.shorelineX == pytest.approx(60.0)
    # Slower than the offshore celerity over the whole path, faster than over the shelf
    distance = 60.0 - 15.0
    assert distance / waveTheory.solitaryWaveSpeed(0.6, 1.0) < wave.estimatedArrivalTime() < distance / math.sqrt(9.80665 * (shelf + 0.6))
    wave.build()
    walls = wave.particles.positions[~wave.particles.isFluid]
    assert np.max(walls[:, 0]) > 60.0 and np.max(wave.particles.positions[wave.particles.isFluid, 0]) < 60.0

def testAStartInsideTheHalfLengthAndAnUnknownStopAreRejected():

    with pytest.raises(InvalidInputError):
        BreakingWave(startX = 1.0)
    wave = BreakingWave(particlesPerDepth = 4, endTime = 0.01)
    with pytest.raises(ValueError):
        wave.run(progress = False, stopAt = 'impact')

def testTheSerreStartTakesTheBoundaryElementFlow():

    '''

    The delta-SPH scheme started from `OverturningWave`'s state: the crest stands where the boundary
    element start puts it, the water moves shoreward under it at about the wave's particle speed,
    still water fills the beach past the boundary element tank's wall, and a short run stays finite.

    '''

    wave = BreakingWave(particlesPerDepth = 8, frameInterval = 0.1, endTime = 0.2, scheme = 'deltaSph',
                        startingWave = 'serre')
    wave.build()
    p, s = wave.particles, wave.spacing
    fluid = p.positions[p.isFluid]
    velocities = p.velocities[p.isFluid]
    top = fluid[:, 1].max()
    crest = fluid[fluid[:, 1] > top - 0.5 * s, 0]

    assert abs(float(np.mean(crest)) - wave.crestX) < 0.5
    assert 1.0 + 0.3 - 1.5 * s < top < 1.0 + 0.3 + 0.5 * s
    under = np.abs(fluid[:, 0] - wave.crestX) < s
    # Under the crest the depth-averaged speed is c H / (d + H), 0.30 sqrt(g d) at H/d = 0.3
    meanSpeed = float(np.mean(velocities[under, 0])) / math.sqrt(constants.gravity)
    assert 0.25 < meanSpeed < 0.35
    # The boundary element tank ends 3.5 d short of the shoreline, at a depth of 0.1 d; past it the
    # water lies still, up to where the beach leaves less than a spacing of it
    tankEnd = wave.shorelineX - 0.1 / wave.beachSlope
    assert np.all(velocities[fluid[:, 0] > tankEnd] == 0.0)
    assert np.max(fluid[:, 0]) > wave.shorelineX - 1.0 / (8 * wave.beachSlope) - s

    wave.run(progress = False)
    assert np.all(np.isfinite(wave.frames[-1].positions))

def testOnlyTheDeltaSchemeTakesTheSerreStart():

    for start in ('serre', 'still'):
        with pytest.raises(InvalidInputError):
            BreakingWave(startingWave = start)

def testTheStillStartLaysTheWavesLatticeAtRest():

    '''

    The still water start fills the same tank on the lattice the exact wave lays: every still
    water particle stands where one of the wave's does, at rest, under hydrostatic pressure, and a
    wall particle deep under the flat bottom carries the hydrostatic head too.

    '''

    wave = BreakingWave(particlesPerDepth = 6, scheme = 'deltaSph')
    still = BreakingWave(particlesPerDepth = 6, scheme = 'deltaSph', startingWave = 'still')
    wave.build()
    still.build()
    p, rho, g = still.particles, constants.freshwaterDensity, constants.gravity
    stillFluid = p.positions[p.isFluid]
    points = lambda positions: set(map(tuple, np.round(positions, 9)))

    assert points(stillFluid) <= points(wave.particles.positions[wave.particles.isFluid])
    assert np.all(p.velocities == 0.0) and still.startingWavePotentialEnergy() == 0.0
    assert np.allclose(p.pressures[p.isFluid] / (rho * g), 1.0 - stillFluid[:, 1], rtol = 1e-9, atol = 1e-9)
    walls = p.positions[~p.isFluid]
    deep = np.flatnonzero(~p.isFluid)[(walls[:, 0] > 1.0) & (walls[:, 0] < still.toeX - 1.0) & (walls[:, 1] < -0.5 * still.spacing)]
    assert deep.size and np.allclose(p.pressures[deep] / (rho * g), 1.0 - p.positions[deep, 1], rtol = 1e-9)

def testTheStillRunsDriftComesOutOfTheWavesEnergies():

    '''

    A still water run whose particles' potential energy falls 300 J/m over its first second and
    then holds, with 5 J/m of kinetic energy throughout: taken out of a wave run that drifts the
    same way, the wave keeps 1500 J/m of kinetic energy and its starting potential energy. A
    reference with a wave in it, or one shorter than the run, is refused.

    '''

    from pysurf.BreakingWave import WaveFrame

    wave = BreakingWave(particlesPerDepth = 4, scheme = 'deltaSph')
    still = BreakingWave(particlesPerDepth = 4, scheme = 'deltaSph', startingWave = 'still')
    wave.build()
    times = np.arange(0.0, 4.0, 0.5)
    settling = lambda t: -300.0 * min(t, 1.0)
    frame = lambda t, kinetic, potential: WaveFrame(time = t, positions = np.zeros((0, 2)), speeds = np.zeros(0),
                                                    kineticEnergy = kinetic, potentialEnergy = potential)
    still.frames = [frame(t, 5.0, 2.0e5 + settling(t)) for t in times]
    wave.frames = [frame(t, 1505.0, 3.0e5 + settling(t)) for t in times]
    _, kinetic, potential = wave.energyHistory(reference = still)

    assert np.allclose(kinetic, 1500.0)
    assert np.allclose(potential, wave.startingWavePotentialEnergy())
    with pytest.raises(ValueError):
        wave.energyHistory(reference = wave)
    still.frames = still.frames[:-1]
    with pytest.raises(ValueError):
        wave.energyHistory(reference = still)

def testTheDissipationFollowsTheEnergiesAndTheBreakingPoint():

    '''

    Kinetic energy falling at 50 J/m/s from 2000 J/m and the wave's potential energy rising at
    30 J/m/s from the start's: with breaking at t1 = 2 s, kinetic meets potential at
    t2 = (2000 - P0) / 80, and the rate is the fall in their sum over t2 - t1.

    '''

    from pysurf.BreakingWave import WaveFrame
    from pysurf.waveBreaking import BreakingPoint, BreakingTracker

    wave = BreakingWave(particlesPerDepth = 4, scheme = 'deltaSph')
    wave.build()
    start = wave.startingWavePotentialEnergy()
    times = np.arange(0.0, 20.0, 0.5)
    # The particles' potential energy carries an arbitrary constant that the history removes
    wave.frames = [WaveFrame(time = t, positions = np.zeros((0, 2)), speeds = np.zeros(0), kineticEnergy = 2000.0 - 50.0 * t,
                             potentialEnergy = 1.0e6 + 30.0 * t) for t in times]
    wave.tracker.breakingPoint = BreakingPoint(time = 2.0, lean = float('nan'), crestX = 30.0, crestHeight = 0.48,
                                               depth = 0.35, faceX = 30.2, faceDepth = 0.34)
    result = wave.breakingDissipation()
    endTime = (2000.0 - start) / 80.0

    assert result['activeEndReached']
    assert result['activeEndTime'] == pytest.approx(endTime)
    assert result['dissipationRate'] == pytest.approx(20.0)
    assert result['energyShareLost'] == pytest.approx(20.0 * (endTime - 2.0) / (2000.0 + start - 40.0))
    assert result['scaledDissipation'] == pytest.approx(20.0 / (constants.freshwaterDensity * constants.gravity**1.5) * 0.34)
    assert result['scaledDissipationFit'] == pytest.approx(0.0281 * 0.48**3.152)

def testTheDissipationNeedsAFaceFromTheSmoothedSurface():

    wave = BreakingWave(particlesPerDepth = 4)
    with pytest.raises(ValueError):
        wave.breakingDissipation()

def testTheRefinementZoneSplitsTheCrestAndNothingElse():

    # Two levels in the zone 1 depth behind to 1.5 ahead of the crest and 0.15 depths over still water
    wave = BreakingWave(depth = 1.0, waveHeight = 0.3, particlesPerDepth = 8, scheme = 'deltaSph', backend = 'numpy',
                        refinement = {'levels': 2, 'fromX': 0.0})
    wave.build()
    p = wave.particles
    mass, momentum = p.masses[p.isFluid].sum(), (p.masses[p.isFluid, None] * p.velocities[p.isFluid]).sum(axis = 0)
    crestX = wave.crestX
    wave.tracker.times.append(0.0)
    wave.tracker.crestX.append(crestX)
    split = wave._refineCrest()
    p = wave.particles
    levels = refinementLevels(p, wave._fluidMass)
    x, z, s = p.positions[:, 0], p.positions[:, 1], wave.spacing
    near = p.isFluid & (x > crestX - 1.0 - s) & (x < crestX + 1.5 + s) & (z > 1.15 - s)
    deep = p.isFluid & (x > crestX - 1.0 + s) & (x < crestX + 1.5 - s) & (z > 1.15 + s)

    assert split > 0
    assert np.all(levels[deep] == 2)
    assert np.all(levels[p.isFluid & ~near] == 0)
    assert np.all(p.lengths == pytest.approx(wave.solver.config.smoothingLength))
    assert p.masses[p.isFluid].sum() == pytest.approx(mass, rel = 1e-13)
    assert (p.masses[p.isFluid, None] * p.velocities[p.isFluid]).sum(axis = 0) == pytest.approx(momentum, rel = 1e-12)

def testTheRefinementZoneWaitsForTheCrest():

    wave = BreakingWave(depth = 1.0, waveHeight = 0.3, particlesPerDepth = 8, scheme = 'deltaSph', backend = 'numpy',
                        refinement = {'levels': 1})
    wave.build()
    wave.tracker.times.append(0.0)
    wave.tracker.crestX.append(wave.crestX)

    assert wave.crestX < wave.toeX
    assert wave._refineCrest() == 0

@pytest.mark.parametrize('settings, backend', [({'level': 1}, 'numpy'), ({'levels': 0}, 'numpy'), ({'levels': 4}, 'numpy'),
                                               ({'levels': 1}, 'compiled')])
def testBadRefinementSettingsAreRefused(settings, backend):

    with pytest.raises(InvalidInputError):
        BreakingWave(particlesPerDepth = 8, scheme = 'deltaSph' if backend != 'compiled' else 'shepard', backend = backend,
                     refinement = settings)

