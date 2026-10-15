
# -- pySurf: Surface Breaking Wave Tests -- #

'''

Smoke tests of pysurf.SurfaceBreakingWave: inputs are checked, the geometry follows its
definitions, the starting state is the exact SGN solitary wave over still water, a short coarse
run records frames and tracker history, and the transient writes as a GIF with one image per
frame. Breaking and runup are validated in tests/testValidationSurfaceWave.py.

Sean Bowman [10/11/2026]

'''

import math

import numpy as np
import pytest
from PIL import Image

from pysurf import waveTheory
from pysurf.errors import InvalidInputError
from pysurf.SurfaceBreakingWave import SurfaceBreakingWave

@pytest.fixture(scope = 'module')
def shortRun() -> SurfaceBreakingWave:

    wave = SurfaceBreakingWave(cellsPerDepth = 10, frameInterval = 0.25, endTime = 0.5)
    wave.run(progress = False)

    return wave

@pytest.mark.parametrize('inputs', [{'depth': -1.0}, {'beachSlope': 0.0}, {'cellsPerDepth': 4},
                                    {'waveHeight': 0.9}, {'breakingAngle': 95.0}])
def testInvalidInputsAreRejected(inputs):

    with pytest.raises(InvalidInputError):
        SurfaceBreakingWave(**inputs)

def testTheGeometryFollowsItsDefinitions():

    wave = SurfaceBreakingWave(depth = 2.0, waveHeight = 0.6, beachSlope = 1.0 / 20.0, cellsPerDepth = 40)

    assert wave.cellSize == pytest.approx(0.05)
    assert waveTheory.serreSolitaryWaveProfile(0.0, 0.6, 2.0, wave.crestX) == pytest.approx(0.006)
    assert wave.toeX == pytest.approx(2.0 * wave.crestX)
    assert wave.shorelineX == pytest.approx(wave.toeX + 40.0)
    assert wave.slopeParameter == pytest.approx(1.521 / 20.0 / math.sqrt(0.3))

def testTheWaveStartsAsTheExactSolitaryWaveOverStillWater():

    wave = SurfaceBreakingWave(cellsPerDepth = 20)
    wave.build()
    solver, x = wave.solver, wave.solver.x
    eta = waveTheory.serreSolitaryWaveProfile(x, 0.3, 1.0, wave.crestX)
    offshore = x < wave.toeX

    assert solver.surface[offshore] == pytest.approx(1.0 + eta[offshore])
    assert solver.velocity[offshore] == pytest.approx(waveTheory.serreSolitaryWaveVelocity(x, 0.3, 1.0, wave.crestX)[offshore])
    assert np.all(solver.velocity[~offshore] == 0.0)
    assert np.all(solver.depth[x > wave.shorelineX + 0.1] == 0.0)

def testAShortRunRecordsFramesAndHistory(shortRun):

    assert [frame.time for frame in shortRun.frames][:1] == [0.0]
    assert len(shortRun.frames) == 3
    assert len(shortRun.tracker.times) > 10
    assert shortRun.tracker.crestX[-1] > shortRun.crestX
    summary = shortRun.summary()
    assert summary['predictedBreakerType'] == 'plunging' and summary['broke'] is False

def testTheTransientWritesAsAGif(shortRun, tmp_path):

    path = shortRun.writeAnimation(str(tmp_path / 'surface.gif'), fps = 4)

    with Image.open(path) as gif:
        assert gif.n_frames == len(shortRun.frames)

def testAFrameDraws(shortRun):

    figure = shortRun.drawFrame(-1)

    assert len(figure.axes) == 3
