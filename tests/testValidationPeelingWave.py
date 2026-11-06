
# -- Consistency: Peeling Wave on an Oblique Reef -- #

'''

Checks of `pysurf.PeelingWave` against what its construction requires, not against a reference:
no measurement or three-dimensional simulation of a solitary wave peeling along an oblique reef was
found. The case is the Surf Ranch reef of tests/testValidationOverturning.py (a 0.0693 face up to a
flat 0.371 h0 deep) with its contours at 30 degrees to the incident crest, started from the exact
solitary wave of H0/h0 = 0.6, handed to SPH at 16 particles per depth for 1.6 sqrt(h0 / g).

**The incident wave travels unchanged.** The construction delays every cross-section by
y tan(alpha) / c, which is exact only for a wave that travels unchanged at c over the flat bed.
Criterion: the speed measured over the flat bed within 0.2 percent of the exact solitary wave's.

**The peel is steady.** Away from the wall at y = 0 the flow is steady in a frame moving with the
peel, so every plane's energy, as a function of the time since its own impact, must be the same.
Criterion: at 0.25, 0.5 and 0.8 sqrt(h0 / g) after impact, the energies of the newest half of the
planes that have reached that age lie within 0.05 of each other, as a share of the plane's energy at
impact; with strip theory's spanwise velocity and without it.

**Without spanwise velocity each plane repeats the 2D splash-up.** With the spanwise velocity set to
zero, the planes start as the two-dimensional splash-up does, and the extruded check
(tests/testValidationExtrudedSplashUp.py) bounds the difference the 3D kernel makes at 2 percent.
Criterion: the median of those planes within 0.05 of the two-dimensional `SplashUp` at the same
age and resolution, with the same window and piston.

The criteria were set before the comparison was run. Run with `pytest -m slow`.

Sean Bowman [10/31/2026]

'''

import functools
import math

import pytest

import numpy as np

from pysurf import constants, waveTheory
from pysurf.bathymetry import Bathymetry
from pysurf.OverturningWave import OverturningWave
from pysurf.PeelingWave import PeelingWave
from pysurf.SplashUp import SplashUp

particlesPerDepth = 16
angle = 30.0
ages = (0.25, 0.5, 0.8)             # [sqrt(h0 / g)]

@functools.lru_cache(maxsize = None)
def surfRanchExact() -> OverturningWave:

    '''The Surf Ranch overturn from the exact solitary wave, run to the jet's impact.'''

    slope, shallowDepth = 0.0693, 0.371
    reefLength = 60.0 - (30.0 + (1.0 - shallowDepth) / slope)
    bed = Bathymetry.reefBreak(depth = 1.0, toeX = 30.0, faceSlope = slope, reefDepth = shallowDepth,
                               reefLength = reefLength)
    wave = OverturningWave(depth = 1.0, waveHeight = 0.6, bathymetry = bed, beachSlope = slope, startX = 15.0,
                           crestSpacingRatio = 0.02, startingWave = 'exact')
    wave.run(progress = False)

    return wave

@functools.lru_cache(maxsize = None)
def peelingRun(spanwiseScale: float, backend: str = 'compiled') -> PeelingWave:

    peel = PeelingWave(surfRanchExact(), obliqueAngle = angle, particlesPerDepth = particlesPerDepth,
                       spanwiseScale = spanwiseScale, backend = backend)
    peel.run(progress = False)

    return peel

@functools.lru_cache(maxsize = None)
def planarRun() -> tuple[np.ndarray, np.ndarray]:

    '''The two-dimensional splash-up's window energy against the time since the hand-off.'''

    wave = surfRanchExact()
    scale = math.sqrt(wave.depth / constants.gravity)
    splash = SplashUp(wave, particlesPerDepth = particlesPerDepth, duration = 1.6 * scale,
                      frameInterval = 1.6 * scale / 40.0, behindDepths = 3.0, aheadDepths = 4.0, backend = 'compiled')
    splash.run(progress = False)

    return (np.array([frame.time for frame in splash.frames]) - splash.handoffTime,
            np.array([frame.kineticEnergy + frame.potentialEnergy for frame in splash.frames]))

def newestPlanes(peel: PeelingWave, age: float) -> np.ndarray:

    '''Energies of the newest half of the planes that have reached an age, over each one's energy at impact.'''

    times, energy = peel.planeHistory()
    target = age * math.sqrt(peel.depth / constants.gravity)
    values = []
    for plane in range(energy.shape[1]):
        valid = np.isfinite(energy[:, plane])
        if valid.sum() >= 2 and times[valid, plane].min() <= target <= times[valid, plane].max():
            values.append(np.interp(target, times[valid, plane], energy[valid, plane]) / energy[valid, plane][0])

    return np.array(values[len(values) // 2:])

@pytest.mark.slow
def testTheIncidentWaveTravelsAtTheExactSolitarySpeed():

    peel = PeelingWave(surfRanchExact(), obliqueAngle = angle, particlesPerDepth = particlesPerDepth)
    exact = waveTheory.exactSolitaryWaveSpeed(0.6, 1.0)

    assert abs(peel._measureIncidentSpeed() / exact - 1.0) < 0.002

@pytest.mark.slow
@pytest.mark.parametrize('spanwiseScale', [1.0, 0.0])
def testThePeelIsSteady(spanwiseScale):

    peel = peelingRun(spanwiseScale)
    for age in ages:
        values = newestPlanes(peel, age)
        assert len(values) >= 5, age
        assert np.ptp(values) < 0.05, (age, values)

@pytest.mark.slow
def testWithoutSpanwiseVelocityEachPlaneRepeatsThePlanarSplashUp():

    peel = peelingRun(0.0)
    times, energy = planarRun()
    scale = math.sqrt(peel.depth / constants.gravity)
    for age in ages:
        planar = np.interp(age * scale, times, energy) / energy[0]
        assert abs(np.median(newestPlanes(peel, age)) - planar) < 0.05, age
