
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

from pysurf import waveTheory
from pysurf.BreakingWave import BreakingWave
from pysurf.errors import InvalidInputError

@pytest.fixture(scope = 'module')
def shortRun() -> BreakingWave:

    wave = BreakingWave(particlesPerDepth = 6, frameInterval = 0.25, endTime = 0.5)
    wave.run(progress = False)

    return wave

@pytest.mark.parametrize('inputs', [{'depth': -1.0}, {'beachSlope': 0.0}, {'particlesPerDepth': 2},
                                    {'waveHeight': 0.9}])
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
