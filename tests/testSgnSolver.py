
# -- pySurf: Serre-Green-Naghdi Solver Tests -- #

'''

Checks of pysurf.sgnSolver against closed forms.

The shallow-water part: a lake at rest over a beach that rises out of the water stays at rest to
rounding (the hydrostatic reconstruction's well-balanced property), volume is conserved to rounding
between walls, and a dam break onto a dry bed converges to Ritter's (1892) solution. The dispersive
part: a small standing wave in a closed basin oscillates at the SGN linear frequency, which differs
from the shallow-water frequency by 15 percent at k d = 1, and the exact SGN solitary wave keeps
its height and travels at sqrt(g (d + H)). The breaking switch: a solitary wave on a flat bed never
breaks, and a steep front does.

Sean Bowman [10/11/2026]

'''

import math

import numpy as np
import pytest

from pysurf import constants, waveTheory
from pysurf.bathymetry import Bathymetry
from pysurf.sgnSolver import (BreakingFront, SgnConfig, SgnSolver, WaveSource, boreFroudeNumber, dispersiveAcceleration,
                              mirrored)
from pysurf.waveBreaking import profileCrest

g = constants.gravity

#----------------------------------------------------------------------#
# -- Shallow-Water Part -- #
#----------------------------------------------------------------------#

def testMirroringReflectsAboutTheEndFaces():

    values = np.array([1.0, 2.0, 3.0])

    assert mirrored(values, 2) == pytest.approx([2.0, 1.0, 1.0, 2.0, 3.0, 3.0, 2.0])
    assert mirrored(values, 1, -1.0) == pytest.approx([-1.0, 1.0, 2.0, 3.0, -3.0])

@pytest.mark.parametrize('dispersive', [False, True])
def testALakeAtRestOverADryBeachStaysAtRest(dispersive):

    beach = Bathymetry.planeBeach(depth = 1.0, toeX = 3.0, slope = 0.1, endX = 20.0)
    solver = SgnSolver(beach, SgnConfig(cellSize = 0.05, dispersive = dispersive, dispersiveDepth = 0.02))
    start = solver.surface.copy()

    assert np.any(~solver.wet)
    for _ in range(200):
        solver.step()

    assert np.max(np.abs(solver.discharge)) < 1e-12
    assert solver.surface[solver.wet] == pytest.approx(start[solver.wet], abs = 1e-12)

def testVolumeIsConservedBetweenWalls():

    beach = Bathymetry.planeBeach(depth = 1.0, toeX = 10.0, slope = 0.05, endX = 40.0)
    solver = SgnSolver(beach, SgnConfig(cellSize = 0.05, dispersiveDepth = 0.02))
    eta = waveTheory.serreSolitaryWaveProfile(solver.x, 0.2, 1.0, 5.0)
    solver.setState(np.maximum(1.0 + eta - solver.bed, 0.0),
                    waveTheory.serreSolitaryWaveVelocity(solver.x, 0.2, 1.0, 5.0) * (solver.bed < 1.0))
    volume = solver.volume

    while solver.time < 3.0:
        solver.step()

    assert solver.volume == pytest.approx(volume, rel = 1e-12)

def ritterDepth(x: np.ndarray, time: float, depth: float) -> np.ndarray:

    '''Ritter's dam break onto a dry bed, dam at x = 0, still water of `depth` behind it.'''

    c0 = math.sqrt(g * depth)
    xi = x / time

    return np.where(xi <= -c0, depth, np.where(xi >= 2.0 * c0, 0.0, (2.0 * c0 - xi)**2 / (9.0 * g)))

def damBreakError(cellSize: float) -> float:

    '''Relative L1 depth error of a dam break with the dispersive part off, at t = 1 s.'''

    flat = Bathymetry([-10.0, 10.0], [0.0, 0.0], stillWaterLevel = 1.0)
    solver = SgnSolver(flat, SgnConfig(cellSize = cellSize, dispersive = False))
    solver.setState(np.where(solver.x < 0.0, 1.0, 0.0), np.zeros_like(solver.x))
    while solver.time < 1.0 - 1e-12:
        solver.step(min(solver.stableTimeStep(), 1.0 - solver.time))
    exact = ritterDepth(solver.x, solver.time, 1.0)

    return float(np.sum(np.abs(solver.depth - exact)) / np.sum(exact))

def testADamBreakOntoADryBedConvergesToRitter():

    coarse, fine = damBreakError(0.1), damBreakError(0.05)

    assert fine < 0.002
    assert fine < 0.6 * coarse

#----------------------------------------------------------------------#
# -- Dispersive Part -- #
#----------------------------------------------------------------------#

def testTheDispersiveTermVanishesForAStillLake():

    beach = Bathymetry.planeBeach(depth = 1.0, toeX = 3.0, slope = 0.1, endX = 9.0)
    solver = SgnSolver(beach, SgnConfig(cellSize = 0.05))
    phi = dispersiveAcceleration(solver.depth, np.zeros_like(solver.x), solver.bed, solver.cellSize, g, solver.wet)

    assert np.max(np.abs(phi)) < 1e-12

def standingWavePeriod(dispersive: bool) -> float:

    '''

    Period of the first standing mode in a flat basin of length pi d, so k d = 1, from upward
    crossings of the surface at the left wall.

    '''

    depth, amplitude = 1.0, 0.002
    length = math.pi * depth
    basin = Bathymetry([0.0, length], [0.0, 0.0], stillWaterLevel = depth)
    solver = SgnSolver(basin, SgnConfig(cellSize = length / 80, dispersive = dispersive))
    solver.setState(depth + amplitude * np.cos(math.pi * solver.x / length), np.zeros_like(solver.x))

    times, levels = [], []
    while solver.time < 2.3 * 2.0 * math.pi / float(waveTheory.serreAngularFrequency(1.0, depth)):
        solver.step()
        times.append(solver.time)
        levels.append(solver.surface[0] - depth)
    times, levels = np.asarray(times), np.asarray(levels)
    up = np.flatnonzero((levels[:-1] < 0.0) & (levels[1:] >= 0.0))
    crossings = times[up] - levels[up] * (times[up + 1] - times[up]) / (levels[up + 1] - levels[up])

    return float(np.mean(np.diff(crossings)))

def testAStandingWaveOscillatesAtTheSgnFrequency():

    '''

    SGN: omega^2 = g k^2 d / (1 + (k d)^2 / 3); shallow water: omega^2 = g k^2 d. At k d = 1 the two
    periods differ by 15 percent, so the check separates the dispersive term from the hyperbolic.
    Both come out within 0.01 percent at 80 cells per wavelength.

    '''

    sgnPeriod = 2.0 * math.pi / float(waveTheory.serreAngularFrequency(1.0, 1.0))
    shallowPeriod = 2.0 * math.pi / math.sqrt(g)

    assert standingWavePeriod(True) == pytest.approx(sgnPeriod, rel = 0.001)
    assert standingWavePeriod(False) == pytest.approx(shallowPeriod, rel = 0.001)

def solitaryRun(cellsPerDepth: int, travelTime: float = 2.0) -> dict:

    '''The exact SGN solitary wave, H/d = 0.3, on a flat bed for `travelTime`.'''

    depth, height = 1.0, 0.3
    crestX = waveTheory.serreSolitaryWaveHalfLength(height, depth, 0.001)
    channel = Bathymetry([0.0, 2.0 * crestX + 10.0], [0.0, 0.0], stillWaterLevel = depth)
    solver = SgnSolver(channel, SgnConfig(cellSize = depth / cellsPerDepth))
    solver.setState(depth + waveTheory.serreSolitaryWaveProfile(solver.x, height, depth, crestX),
                    waveTheory.serreSolitaryWaveVelocity(solver.x, height, depth, crestX))

    times, crests, heights = [], [], []
    while solver.time < travelTime:
        solver.step()
        eta = solver.surface - depth
        times.append(solver.time)
        crests.append(profileCrest(solver.x, eta))
        heights.append(float(np.max(eta)))
        breaking = solver.breaking
    celerity = float(np.polyfit(times[len(times) // 4:], crests[len(times) // 4:], 1)[0])

    return {'celerity': celerity, 'reference': waveTheory.solitaryWaveSpeed(height, depth),
            'height': heights[-1], 'everBroke': breaking}

def testTheExactSolitaryWaveKeepsItsHeightAndSpeed():

    '''

    Over 2 s (seven depths of travel) at 40 cells per depth, speed and height stay within 0.01
    percent of the exact wave, and the errors fall by four when the cells halve: second order.

    '''

    coarse, fine = solitaryRun(20), solitaryRun(40)
    celerityError = [abs(run['celerity'] / run['reference'] - 1.0) for run in (coarse, fine)]
    heightError = [abs(run['height'] / 0.3 - 1.0) for run in (coarse, fine)]

    assert celerityError[1] < 1e-4 and heightError[1] < 1e-4
    assert celerityError[0] / celerityError[1] > 3.0
    assert heightError[0] / heightError[1] > 3.0
    assert not coarse['everBroke'] and not fine['everBroke']

#----------------------------------------------------------------------#
# -- Breaking Switch -- #
#----------------------------------------------------------------------#

def testTheBoreFroudeNumber():

    assert boreFroudeNumber(1.0, 1.0) == pytest.approx(1.0)
    assert boreFroudeNumber(2.0, 1.0) == pytest.approx(math.sqrt(3.0))
    assert boreFroudeNumber(1.0, 0.0) == math.inf

def testASteepFrontStartsBreakingAndSwitchesOffDispersionAroundIt():

    flat = Bathymetry([0.0, 20.0], [0.0, 0.0], stillWaterLevel = 1.0)
    solver = SgnSolver(flat, SgnConfig(cellSize = 0.02))
    # A bore-like step: 0.5 m of extra water falling to still water over 0.2 m, a 68 degree front
    x = solver.x
    surface = 1.0 + 0.5 * np.clip((10.0 - x) / 0.2, 0.0, 1.0)
    solver.setState(surface, np.zeros_like(x))
    solver.step()

    assert solver.breaking
    assert not np.all(solver.dispersiveCells[solver.wet])
    zone = x[solver.breakingZone]
    assert zone.min() < 10.0 < zone.max()

#----------------------------------------------------------------------#
# -- Improved Dispersion, Wave Source and Absorbing Layers -- #
#----------------------------------------------------------------------#

def testTheImprovedDispersionStandingWave():

    '''With alpha = 1.159 a standing wave at k d = 2 oscillates at the alpha-SGN frequency, within 0.6 percent of Airy.'''

    depth, amplitude, kd, alpha = 1.0, 0.002, 2.0, 1.159
    length = math.pi * depth / kd
    basin = Bathymetry([0.0, length], [0.0, 0.0], stillWaterLevel = depth)
    solver = SgnSolver(basin, SgnConfig(cellSize = length / 80, dispersionAlpha = alpha))
    solver.setState(depth + amplitude * np.cos(math.pi * solver.x / length), np.zeros_like(solver.x))
    expected = 2.0 * math.pi / float(waveTheory.serreAngularFrequency(kd, depth, alpha = alpha))

    times, levels = [], []
    while solver.time < 2.3 * expected:
        solver.step()
        times.append(solver.time)
        levels.append(solver.surface[0] - depth)
    times, levels = np.asarray(times), np.asarray(levels)
    up = np.flatnonzero((levels[:-1] < 0.0) & (levels[1:] >= 0.0))
    crossings = times[up] - levels[up] * (times[up + 1] - times[up]) / (levels[up + 1] - levels[up])
    airy = 2.0 * math.pi / math.sqrt(g * kd * math.tanh(kd))

    assert np.mean(np.diff(crossings)) == pytest.approx(expected, rel = 0.002)
    assert expected == pytest.approx(airy, rel = 0.007)

def generatedWave(kd: float, alpha: float) -> dict:

    '''

    A small regular wave from the source in a flat channel with absorbing layers at both ends: the
    amplitude between the source and the far layer, and the reflection read from its envelope.

    '''

    depth, height = 1.0, 0.01
    omega = float(waveTheory.serreAngularFrequency(kd, depth, alpha = alpha))
    period, wavelength = 2.0 * math.pi / omega, 2.0 * math.pi / kd
    channel = Bathymetry([0.0, 5.5 * wavelength], [0.0, 0.0], stillWaterLevel = depth)
    config = SgnConfig(cellSize = wavelength / 80, dispersionAlpha = alpha, spongeLeft = 1.5 * wavelength,
                       spongeRight = 1.5 * wavelength, spongeRate = 4.0 * omega)
    source = WaveSource(height, period, depth, 2.0 * wavelength, alpha = alpha)
    solver = SgnSolver(channel, config, source = source)
    gauges = (solver.x > 2.5 * wavelength) & (solver.x < 3.7 * wavelength)
    groupSpeed = float(waveTheory.linearGroupSpeed(period, depth))
    settle = source.rampPeriods * period + 3.5 * wavelength / groupSpeed

    high = np.full(np.sum(gauges), -np.inf)
    low = np.full(np.sum(gauges), np.inf)
    while solver.time < settle + 4.0 * period:
        solver.step()
        if solver.time > settle:
            elevation = solver.surface[gauges] - depth
            high, low = np.maximum(high, elevation), np.minimum(low, elevation)
    envelope = high - low

    return {'amplitude': float(np.mean(envelope)) / 2.0, 'target': height / 2.0,
            'reflection': float((envelope.max() - envelope.min()) / (envelope.max() + envelope.min()))}

@pytest.mark.parametrize('kd, alpha', [(0.5, 1.0), (1.5, 1.159)])
def testTheSourceRadiatesTheRequestedHeight(kd, alpha):

    '''Linear theory sizes the source; at 80 cells per wavelength the radiated amplitude is within 1 percent.'''

    result = generatedWave(kd, alpha)

    assert result['amplitude'] == pytest.approx(result['target'], rel = 0.015)

def testTheAbsorbingLayerReflectsLittle():

    '''The envelope between source and far layer stays within 1 percent of uniform: little comes back.'''

    assert generatedWave(1.0, 1.159)['reflection'] < 0.01

#----------------------------------------------------------------------#
# -- Front-by-Front Breaking -- #
#----------------------------------------------------------------------#

def stepFront(x: np.ndarray, position: float, height: float, width: float) -> np.ndarray:

    '''A bore-like step of `height` falling to still water over `width` at `position`.'''

    return height * np.clip((position - x) / width, 0.0, 1.0)

def testTwoSeparateBoresBothBreak():

    flat = Bathymetry([0.0, 40.0], [0.0, 0.0], stillWaterLevel = 1.0)
    solver = SgnSolver(flat, SgnConfig(cellSize = 0.02))
    x = solver.x
    # Two plateaus with steep fronts, each 0.5 m over still water ahead: bore Froude number 1.37
    surface = 1.0 + np.where(x < 15.0, stepFront(x, 10.0, 0.5, 0.2), 0.0) + np.where(x > 25.0, stepFront(x, 30.0, 0.5, 0.2), 0.0)
    solver.setState(surface, np.zeros_like(x))
    solver.step()

    assert len(solver.breakingFronts) == 2
    assert len(solver.startedFronts) == 2

def testAWeakBoreStopsBreakingAfterTheTransitionTimeAndCannotRestart():

    '''

    A 5 cm bore on 1 m of water has a Froude number of 1.04, under the 1.3 that ends breaking. It
    keeps breaking for 5 sqrt(h / g) = 1.6 s, then stops, and is too weak to start again.

    '''

    flat = Bathymetry([0.0, 40.0], [0.0, 0.0], stillWaterLevel = 1.0)
    solver = SgnSolver(flat, SgnConfig(cellSize = 0.05, dispersive = False))
    x = solver.x
    solver.setState(1.0 + stepFront(x, 20.0, 0.05, 0.2), np.where(x < 20.0, 0.15, 0.0))
    front = solver._detectFronts(solver.wet, solver.surface)[0]
    solver.breakingFronts = [BreakingFront(x = float(x[front.steepestIndex]), halfWidth = 0.5, started = 0.0)]
    delay = 5.0 * math.sqrt(1.0 / g)

    stoppedAt = None
    while solver.time < 1.5 * delay and stoppedAt is None:
        solver.step()
        if solver.stoppedFronts:
            stoppedAt = solver.time
            froude = solver.stoppedFronts[0][2]

    assert stoppedAt == pytest.approx(delay, abs = 0.05)
    assert froude < 1.3
    for _ in range(5):
        solver.step()
        assert solver.startedFronts == []
