
# -- Validation: Strip Theory on a Sloping Ridge -- #

'''

Two-dimensional boundary element slices (`pysurf.OverturningWave`, stacked by
`pysurf.sliceStack.SliceStack`) against the sloping ridge of Guyenne and Grilli (2006), "Numerical
study of three-dimensional overturning waves in shallow water", J. Fluid Mech. 547, 361-388. They
ran a solitary wave of H0/h0 = 0.6, crest starting at x/h0 = 5.785, over a ridge that rises from
x/h0 = 5.225,

    z_b / h0 = -1 + s0 (x - 5.225) sech^2(k y),     s0 = 1/15,

in a tank 8 h0 wide (|y| <= 4 h0) ending at a wall at x/h0 = 19, with a fully three-dimensional
fully nonlinear potential flow model, and the cross-sections at y = 0, 2 and 4 h0 (slopes 0.067,
0.052 and 0.028) with the two-dimensional model of Grilli et al. (1997). Their breaking point is the
front face first vertical at y = 0, the definition `OverturningWave` applies. All quantities below
are in h0 and sqrt(h0 / g).

**Two-dimensional breaking points** (their table 1, against single slices). Like theirs, the slices
start from the exact potential flow solitary wave (`waveTheory.exactSolitaryWave`; they used
Tanaka's 1986 method, which gives the same wave). The wave starts with its front already on the
slope, so an approximate starting wave would have no flat run in which to settle: the SGN wave of
the same height carries 17 percent more water and breaks 8 percent early here. The 0.067 slice ends
at the tank's wall at x = 19, the others at a wall in 0.1 h0 of water, past their breaking points. Criteria, set before the comparison: breaking time within
3 percent, crest position within 0.3 h0, crest height within 5 percent.

**Strip theory against the three-dimensional ridge.** The stack's centre slice is the 0.067
cross-section run on its own, so on the centre line strip theory predicts one breaking point for
every k. Guyenne and Grilli's three-dimensional breaking point moves later and higher as k grows,
because the ridge focuses the wave on its centre line; their own two-dimensional run differs from
the three-dimensional one by 0.4 percent in breaking time at k = 0.1 and by 11.7 percent at k = 0.5.
Criteria, set before the comparison: breaking time and crest height within 5 percent and crest
position within 0.5 h0 for k = 0.10, 0.17 and 0.25. At k = 0.34 and 0.50 their own
two-dimensional result is already outside 5 percent in breaking time, and strip theory is expected
to fail there; those cases are marked as expected failures.

**Spanwise velocity.** For k = 0.25, their figure 16 gives the horizontal velocity at z = -0.2 h0 at
t = 7.973, on a grid with spacings 0.35 h0 in x and 0.38 h0 in y over 14 <= x <= 19 and
|y| <= 2: the largest horizontal speed is 0.651 sqrt(g h0), and the largest spanwise velocity is
0.085 of the largest streamwise one. The stack takes v = dphi/dy from the slices' potentials.
Criteria, set before the comparison: the largest speed within 10 percent, the ratio within 30
percent.

**Result.** The slices meet every two-dimensional criterion; the centre slice breaks at t = 7.572,
x = 15.769 with its crest 0.680 above still water, against 7.629, 15.752 and 0.680. Strip theory
meets the three-dimensional criteria for k = 0.10 to 0.25 and fails at 0.34 and 0.50, as expected,
and the largest speed at z = -0.2 meets its criterion. The spanwise velocity does not: the slices
give 0.144 of the streamwise velocity against the three-dimensional 0.085, 70 percent high. The
slices on the ridge's flanks lag the centre slice, so their potentials at the same (x, z) drift
apart; strip theory has no spanwise momentum balance, so the spanwise pressure gradient that
difference would build up in the three-dimensional flow never acts on it.

Run with `pytest -m slow`.

Sean Bowman [10/30/2026]

'''

import functools
import math
import os
from concurrent.futures import ProcessPoolExecutor

import pytest

import numpy as np

from pysurf import constants
from pysurf.bathymetry import Bathymetry
from pysurf.sliceStack import SliceStack, runSlice, threadVariables

timeScale = math.sqrt(1.0 / constants.gravity)  # [s] sqrt(h0 / g) with h0 = 1 m
toeX = 5.225                                    # [h0] where the ridge starts to rise
startX = 5.785                                  # [h0] starting crest position
tankEndX = 19.0                                 # [h0] the wall that ends the tank
waveHeight = 0.6                                # [h0]
ridgeSlope = 1.0 / 15.0                         # [-] slope of the ridge's centre line
ridgeK = 0.25                                   # [1/h0] the transverse modulation studied in detail

# Guyenne and Grilli (2006), table 1: breaking time, crest position and crest height
twoDimensionalBreaking = {
    'centre': (ridgeSlope, 7.629, 15.752, 0.680),
    'y2': (ridgeSlope / math.cosh(2.0 * ridgeK)**2, 8.570, 16.885, 0.709),
    'y4': (ridgeSlope / math.cosh(4.0 * ridgeK)**2, 11.501, 20.457, 0.754),
}
threeDimensionalBreaking = {
    0.10: (7.661, 15.802, 0.685),
    0.17: (7.757, 15.904, 0.693),
    0.25: (7.890, 16.093, 0.706),
    0.34: (8.147, 16.470, 0.719),
    0.50: (8.639, 17.100, 0.740),
}

# Guyenne and Grilli (2006), figure 16(a), k = 0.25
velocityTime = 7.973                            # [sqrt(h0 / g)]
velocityDepth = -0.2                            # [h0] below still water
ridgeMaxSpeed = 0.651                           # [sqrt(g h0)] largest horizontal speed on the grid
ridgeSpanwiseRatio = 0.085                      # [-] largest |v| over largest |u|

# Slice stations across half the ridge: the figure 16 grid rows, then out to the side wall
ridgeStations = (0.0, 0.38, 0.76, 1.14, 1.52, 1.90, 2.28, 2.66, 3.04, 3.42, 3.80, 4.0)

def ridgeSection(y: float) -> Bathymetry:

    '''The ridge's cross-section at y [h0], ending at the tank's wall.'''

    return Bathymetry.planeBeach(depth = 1.0, toeX = toeX, slope = ridgeSlope / math.cosh(ridgeK * y)**2,
                                 endX = tankEndX)

def waveOptions(**extra) -> dict:

    '''The incident wave and run settings every slice shares.'''

    return dict(depth = 1.0, waveHeight = waveHeight, startX = startX, startingWave = 'exact', **extra)

@functools.lru_cache(maxsize = None)
def ridgeStack() -> SliceStack:

    '''The k = 0.25 ridge as slices, recorded at the time of figure 16.'''

    stack = SliceStack(ridgeSection, ridgeStations, symmetric = True,
                       **waveOptions(recordTimes = [velocityTime * timeScale], endTime = 9.2 * timeScale))
    stack.run(progress = False)

    return stack

@functools.lru_cache(maxsize = None)
def sideSlopeRuns() -> dict:

    '''The y = 2 and 4 cross-sections as plain two-dimensional runs, each to a wall in 0.1 h0 of water.'''

    saved = {name: os.environ.get(name) for name in threadVariables}
    os.environ.update({name: '1' for name in threadVariables})
    try:
        with ProcessPoolExecutor(max_workers = 2) as pool:
            futures = {}
            for key in ('y2', 'y4'):
                slope = twoDimensionalBreaking[key][0]
                bed = Bathymetry.planeBeach(depth = 1.0, toeX = toeX, slope = slope, endX = toeX + 0.9 / slope)
                futures[key] = pool.submit(runSlice, bed, waveOptions())
            return {key: future.result() for key, future in futures.items()}
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

def breakingPoint(wave) -> tuple[float, float, float]:

    '''A run's breaking time, crest position and crest height, in h0 and sqrt(h0 / g).'''

    point = wave.breakingPoint
    assert point is not None, wave.stopReason

    return point.time / timeScale, point.crestX, point.crestHeight

@pytest.mark.slow
@pytest.mark.parametrize('key', ['centre', 'y2', 'y4'])
def testSliceBreakingMatchesTheTwoDimensionalModel(key):

    wave = ridgeStack().slices[0] if key == 'centre' else sideSlopeRuns()[key]
    time, crestX, height = breakingPoint(wave)
    _, referenceTime, referenceX, referenceHeight = twoDimensionalBreaking[key]

    assert abs(time / referenceTime - 1.0) < 0.03, (time, referenceTime)
    assert abs(crestX - referenceX) < 0.3, (crestX, referenceX)
    assert abs(height / referenceHeight - 1.0) < 0.05, (height, referenceHeight)

strongFocusing = pytest.mark.xfail(strict = True, reason = 'the ridge focuses the wave on its centre line, which '
                                   'slices cannot do; the published 2D run is already outside 5 percent')

@pytest.mark.slow
@pytest.mark.parametrize('k', [0.10, 0.17, 0.25, pytest.param(0.34, marks = strongFocusing),
                               pytest.param(0.50, marks = strongFocusing)])
def testStripCentreMatchesTheThreeDimensionalRidge(k):

    time, crestX, height = breakingPoint(ridgeStack().slices[0])
    referenceTime, referenceX, referenceHeight = threeDimensionalBreaking[k]

    assert abs(time / referenceTime - 1.0) < 0.05, (time, referenceTime)
    assert abs(crestX - referenceX) < 0.5, (crestX, referenceX)
    assert abs(height / referenceHeight - 1.0) < 0.05, (height, referenceHeight)

def figureSixteenVelocity() -> np.ndarray:

    '''The stack's velocity on the figure 16 grid, inside the three-dimensional water, in sqrt(g h0).'''

    x = np.arange(14.0, 19.0 + 1e-9, 0.35)
    y = np.arange(-1.90, 1.90 + 1e-9, 0.38)
    gridX, gridY = np.meshgrid(x, y)
    z = 1.0 + velocityDepth
    bedZ = (gridX - toeX).clip(min = 0.0) * ridgeSlope / np.cosh(ridgeK * gridY)**2
    wet = bedZ < z
    points = np.column_stack([gridX[wet], gridY[wet], np.full(int(wet.sum()), z)])
    velocity = ridgeStack().velocity(points, velocityTime * timeScale)

    return velocity / math.sqrt(constants.gravity)

@pytest.mark.slow
def testStripSpeedMatchesTheThreeDimensionalRidge():

    velocity = figureSixteenVelocity()
    finite = np.all(np.isfinite(velocity), axis = 1)
    speed = np.hypot(velocity[finite, 0], velocity[finite, 1])

    assert finite.mean() > 0.9, finite.mean()
    assert abs(speed.max() / ridgeMaxSpeed - 1.0) < 0.10, speed.max()

@pytest.mark.slow
@pytest.mark.xfail(strict = True, reason = 'strip theory gives |v| / |u| = 0.144 against the 3D 0.085')
def testStripSpanwiseVelocityMatchesTheThreeDimensionalRidge():

    velocity = figureSixteenVelocity()
    finite = np.all(np.isfinite(velocity), axis = 1)
    ratio = np.abs(velocity[finite, 1]).max() / np.abs(velocity[finite, 0]).max()

    assert abs(ratio / ridgeSpanwiseRatio - 1.0) < 0.30, ratio
