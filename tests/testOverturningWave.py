
# -- pySurf: Overturning Wave Tests -- #

'''

Smoke tests of pysurf.OverturningWave: inputs are checked, the default and reef beds are built, the
starting surface is the SGN solitary wave and can start anywhere the wave fits, a short run records
frames, the frames draw and write as a GIF, and an overturn draws with its fitted curve. The breaking
point and the overturn are validated in tests/testValidationOverturning.py.

Sean Bowman [10/11/2026]

'''

import copy

import numpy as np
import pytest
from PIL import Image

from pysurf import waveTheory
from pysurf.bathymetry import Bathymetry
from pysurf.errors import InvalidInputError
from pysurf.OverturningWave import OverturningWave

@pytest.fixture(scope = 'module')
def shortRun() -> OverturningWave:

    wave = OverturningWave(crestSpacingRatio = 0.05, surfaceSpacingRatio = 0.15, frameInterval = 0.1)
    wave.run(endTime = 0.3, progress = False)

    return wave

@pytest.mark.parametrize('inputs', [{'depth': -1.0}, {'beachSlope': 0.0}, {'waveHeight': 0.9}, {'wallDepthRatio': 1.5}])
def testInvalidInputsAreRejected(inputs):

    with pytest.raises(InvalidInputError):
        OverturningWave(**inputs)

def testThePlaneBeachEndsInAWallAtATenthOfTheDepth():

    wave = OverturningWave()
    wave.build()

    assert wave.bathymetry.depth(wave.bathymetry.endX) == pytest.approx(0.1)
    assert wave.solver.surface[0, 0] == 0.0 and wave.solver.surface[-1, 0] == pytest.approx(wave.bathymetry.endX)

def testTheStartingSurfaceIsTheSolitaryWave():

    wave = OverturningWave()
    wave.build()
    surface = wave.solver.surface
    profile = waveTheory.serreSolitaryWaveProfile(surface[:, 0], 0.3, 1.0, wave.crestX)

    assert surface[:, 1] == pytest.approx(1.0 + profile, abs = 1e-6)

def testAReefBedIsAccepted():

    reef = Bathymetry.reefBreak(depth = 1.0, toeX = 14.0, faceSlope = 0.1, reefDepth = 0.3, reefLength = 8.0)
    wave = OverturningWave(bathymetry = reef)
    wave.build()

    assert wave.solver.bed[:, 1].max() == pytest.approx(0.7)

def testTheWaveStartsWhereItIsPlacedAndNoCloserToTheWall():

    wave = OverturningWave(startX = 12.0)

    assert wave.crestX == pytest.approx(12.0)
    assert wave.toeX == pytest.approx(12.0 + wave.halfLength)
    with pytest.raises(InvalidInputError):
        OverturningWave(startX = 0.5 * wave.halfLength)

def testTheToeOfAGivenBedIsWhereItFirstRises():

    reef = Bathymetry.reefBreak(depth = 1.0, toeX = 20.0, faceSlope = 0.1, reefDepth = 0.3, reefLength = 5.0)

    assert OverturningWave(bathymetry = reef).toeX == pytest.approx(20.0)

def testAShortRunRecordsFrames(shortRun):

    assert len(shortRun.frames) >= 3
    assert shortRun.summary()['broke'] is False
    assert shortRun.tracker.crestX[-1] > shortRun.crestX

def testTheFramesDrawAndWriteAsAGif(shortRun, tmp_path):

    assert len(shortRun.drawFrame(-1).axes) == 3
    assert len(shortRun.drawProfiles(4).axes) == 1
    path = shortRun.writeAnimation(str(tmp_path / 'overturn.gif'), fps = 4)

    with Image.open(path) as gif:
        assert gif.n_frames >= len(shortRun.frames)

def testAnOverturnDrawsWithItsFittedCurve(shortRun):

    # A jet curled over a tube in place of the short run's surface, which does not overhang
    assert shortRun.drawOverturn() is None
    wave = copy.deepcopy(shortRun)
    level = wave.bathymetry.stillWaterLevel
    outer = np.radians(np.linspace(160.0, -20.0, 73))
    inner = np.radians(np.linspace(-20.0, 270.0, 117))
    back = np.column_stack([np.linspace(wave.bathymetry.startX, 9.765, 50, endpoint = False),
                            np.linspace(level, level + 0.335, 50, endpoint = False)])
    jet = np.column_stack([10.0 + 0.25 * np.cos(outer), level + 0.25 + 0.25 * np.sin(outer)])
    tube = np.column_stack([10.0 + 0.15 * np.cos(inner), level + 0.25 + 0.15 * np.sin(inner)])
    face = np.column_stack([np.linspace(10.0, wave.bathymetry.endX, 61)[1:], np.linspace(level + 0.1, level, 61)[1:]])
    wave.solver.surface = np.vstack([back, jet, tube, face])
    figure = wave.drawOverturn()

    assert len(figure.axes) == 1
    assert wave.jet.fitError < 0.1 * wave.jet.overturnLength
