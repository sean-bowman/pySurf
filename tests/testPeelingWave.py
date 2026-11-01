
# -- pySurf: Peeling Wave Tests -- #

'''

Smoke tests of pysurf.PeelingWave that need no run: inputs are checked, a wave whose jet has not
landed is refused, and the plane lag and peel speed follow from the angle. The three-dimensional
run itself is exercised in tests/testValidationPeelingWave.py.

Sean Bowman [10/31/2026]

'''

import math
from types import SimpleNamespace

import pytest

from pysurf.errors import InvalidInputError
from pysurf.PeelingWave import PeelingWave

def landedWave() -> SimpleNamespace:

    '''Stands in for an OverturningWave run to the jet's impact.'''

    return SimpleNamespace(depth = 1.0, waveHeight = 0.6, solver = object(), touchedDown = True)

@pytest.mark.parametrize('inputs', [{'obliqueAngle': 0.0}, {'obliqueAngle': 70.0}, {'bufferPlanes': 2},
                                    {'spanwiseScale': -1.0}, {'particlesPerDepth': 2}])
def testInvalidInputsAreRejected(inputs):

    with pytest.raises(InvalidInputError):
        PeelingWave(landedWave(), **inputs)

def testAWaveWhoseJetHasNotLandedIsRejected():

    wave = landedWave()
    wave.touchedDown = False

    with pytest.raises(InvalidInputError):
        PeelingWave(wave)

def testThePlaneLagAndPeelSpeedFollowTheAngle():

    peel = PeelingWave(landedWave(), obliqueAngle = 30.0, particlesPerDepth = 32)
    peel.incidentSpeed = 4.0

    assert peel.planeLag == pytest.approx((1.0 / 32.0) * math.tan(math.radians(30.0)) / 4.0)
    assert peel.peelSpeed == pytest.approx(4.0 / math.tan(math.radians(30.0)))
