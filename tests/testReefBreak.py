
# -- pySurf: Reef Break Tests -- #

'''

Smoke tests of pysurf.ReefBreak and Bathymetry.reefBreak: the reef profile follows its definition,
inputs are checked, a short coarse run records frames and its summary groups breaking by zone, and
the transient writes as a GIF. Breaking over a reef has no reference in this repository.

Sean Bowman [10/11/2026]

'''

import pytest
from PIL import Image

from pysurf.bathymetry import Bathymetry
from pysurf.errors import InvalidInputError
from pysurf.ReefBreak import ReefBreak

def testTheReefProfileFollowsItsDefinition():

    reef = Bathymetry.reefBreak(depth = 10.0, toeX = 100.0, faceSlope = 0.2, reefDepth = 2.5, reefLength = 50.0,
                                lagoonDepth = 6.0, lagoonLength = 80.0, beachSlope = 0.05, beachHeight = 3.0)

    assert reef.x.tolist() == pytest.approx([0.0, 100.0, 137.5, 187.5, 205.0, 285.0, 465.0])
    assert reef.depth([50.0, 160.0, 250.0]) == pytest.approx([10.0, 2.5, 6.0])
    assert reef.shorelineX() == pytest.approx(285.0 + 6.0 / 0.05)

def testAReefWithoutLagoonEndsAtTheBackOfTheFlat():

    reef = Bathymetry.reefBreak(depth = 1.0, toeX = 10.0, faceSlope = 0.1, reefDepth = 0.3, reefLength = 5.0)

    assert reef.endX == pytest.approx(22.0)
    assert reef.depth(reef.endX) == pytest.approx(0.3)

def testAReefNeedsAShallowerFlat():

    with pytest.raises(InvalidInputError):
        Bathymetry.reefBreak(depth = 1.0, toeX = 10.0, faceSlope = 0.1, reefDepth = 1.5, reefLength = 5.0)

def testInvalidInputsAreRejected():

    with pytest.raises(InvalidInputError):
        ReefBreak(period = -1.0)

@pytest.fixture(scope = 'module')
def shortRun() -> ReefBreak:

    reef = ReefBreak(cellSize = 1.0, periods = 1.0, frameInterval = 2.0)
    reef.run(endTime = 30.0, progress = False)

    return reef

def testAShortRunRecordsFramesAndASummary(shortRun):

    summary = shortRun.summary(settledPeriods = 3.0)

    assert len(shortRun.frames) >= 10
    assert summary['wavelength'] == pytest.approx(shortRun.wavelength)
    assert set(summary) >= {'zones', 'stops', 'cells'}

def testTheTransientWritesAsAGif(shortRun, tmp_path):

    assert len(shortRun.drawFrame(-1).axes) == 3
    path = shortRun.writeAnimation(str(tmp_path / 'reef.gif'), fps = 4)

    with Image.open(path) as gif:
        assert gif.n_frames == len(shortRun.frames)
