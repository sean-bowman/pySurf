
# -- pySurf: Reef Overturn Tests -- #

'''

Smoke tests of pysurf.ReefOverturn: the flume follows Blenkinsopp and Chaplin's reef, the source
makes the height a deep-water wave shoals to at the flume's depth, inputs are checked, and a coarse
SGN train breaks at the reef and hands off a surface whose tank walls carry no discharge.

Sean Bowman [10/14/2026]

'''

import numpy as np
import pytest

from pysurf.errors import InvalidInputError
from pysurf.ReefOverturn import ReefOverturn

def testTheFlumeFollowsTheLaboratoryReef():

    reef = ReefOverturn(crestDepth = 0.082)
    bed = reef.reefBathymetry()

    assert reef.depth == pytest.approx(0.7)
    assert bed.z.tolist() == pytest.approx([0.0, 0.0, 0.618, 0.0, 0.0])
    assert bed.x[2] - bed.x[1] == pytest.approx(6.18)
    assert bed.x[1] - reef.sourceX == pytest.approx(3.8)
    assert bed.depth(reef.crestX) == pytest.approx(0.082)

def testTheSourceMakesTheShoaledDeepWaterHeight():

    # f = 0.6 Hz in 0.7 m of water has k d = 1.21 and a linear shoaling coefficient of 0.914
    reef = ReefOverturn(crestDepth = 0.082, deepWaterHeight = 0.0755, frequency = 0.6)

    assert reef.deepWaterHeight / reef.deepWaterWavelength == pytest.approx(0.0174, abs = 1e-4)
    assert reef.incidentHeight == pytest.approx(0.914 * 0.0755, rel = 0.01)

def testInvalidInputsAreRejected():

    with pytest.raises(InvalidInputError):
        ReefOverturn(crestDepth = -0.01)
    with pytest.raises(InvalidInputError):
        ReefOverturn(lead = 1.5)

def testACoarseTrainBreaksAtTheReefAndHandsOff():

    reef = ReefOverturn(crestDepth = 0.082, cellsPerDepth = 20, periods = 4.0)
    reef.runWaveTrain(progress = False)
    reef.handOff()
    tank = reef.wave.customBathymetry
    state = reef.tankState
    surface, potential = reef.wave.initialState

    assert len(reef.breakingStarts) > 0
    assert np.interp(tank.startX, reef.sgn.x, state.discharge) == pytest.approx(0.0, abs = 1e-6)
    assert np.interp(tank.endX, reef.sgn.x, state.discharge) == pytest.approx(0.0, abs = 1e-6)
    assert tank.startX < reef.crestX < tank.endX
    assert np.all(np.isfinite(potential)) and surface[0, 0] == pytest.approx(tank.startX)
