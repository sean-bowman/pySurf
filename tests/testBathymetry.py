
# -- pySurf: Bathymetry Tests -- #

'''

Checks of pysurf.bathymetry against the closed forms of a plane beach: flat at z = 0 to the toe,
then z = (x - toe) tan(beta), with the still water level at z = d.

Sean Bowman [10/11/2026]

'''

import math

import numpy as np
import pytest

from pysurf.bathymetry import Bathymetry
from pysurf.errors import InvalidInputError

depth, toeX, slope, endX = 1.0, 10.0, 1.0 / 35.0, 60.0

@pytest.fixture
def beach() -> Bathymetry:

    return Bathymetry.planeBeach(depth, toeX, slope, endX)

def testElevationAndDepthFollowThePlaneBeach(beach):

    x = np.array([0.0, 5.0, 10.0, 20.0, 45.0, 55.0])
    expected = np.where(x <= toeX, 0.0, (x - toeX) * slope)

    assert beach.elevation(x) == pytest.approx(expected)
    assert beach.depth(x) == pytest.approx(np.maximum(0.0, depth - expected))
    assert beach.stillWaterLevel == depth

def testTheElevationHoldsBeyondTheEnds(beach):

    assert beach.elevation(-5.0) == 0.0
    assert beach.elevation(endX + 5.0) == pytest.approx((endX - toeX) * slope)

def testTheShorelineIsWhereTheSlopeMeetsStillWater(beach):

    assert beach.shorelineX() == pytest.approx(toeX + depth / slope)

def testABedBelowStillWaterHasNoShoreline():

    with pytest.raises(InvalidInputError):
        Bathymetry([0.0, 10.0], [0.0, 0.5], stillWaterLevel = 1.0).shorelineX()

def testSlopeIsPerSegment(beach):

    assert beach.slope(np.array([5.0, 30.0])) == pytest.approx([0.0, slope])

def testTheNormalIsAUnitVectorPerpendicularToTheBed(beach):

    normal = beach.normal(np.array([5.0, 30.0]))

    assert np.linalg.norm(normal, axis = 1) == pytest.approx([1.0, 1.0])
    assert normal[0] == pytest.approx([0.0, 1.0])
    assert np.dot(normal[1], [1.0, slope]) == pytest.approx(0.0, abs = 1e-15)
    assert normal[1, 1] > 0.0

def testSamplesAreEquallySpacedAlongTheBed(beach):

    '''Measured along the bed; a chord across the toe cuts the corner and is slightly shorter.'''

    x, z = beach.sample(0.1)
    stretch = math.sqrt(1.0 + slope**2)
    alongBed = np.where(x <= toeX, x, toeX + (x - toeX) * stretch)
    steps = np.diff(alongBed)

    assert (x[0], x[-1]) == (0.0, endX)
    assert steps == pytest.approx(np.full(steps.size, alongBed[-1] / steps.size), rel = 1e-9)
    assert steps[0] == pytest.approx(0.1, rel = 0.01)
    assert z == pytest.approx(beach.elevation(x), abs = 1e-12)

def testTheSandPolygonClosesOnTheFloor(beach):

    x, z = beach.sandPolygon(floor = -0.5)

    assert (x[-2], z[-2]) == (endX, -0.5)
    assert (x[-1], z[-1]) == (0.0, -0.5)

@pytest.mark.parametrize('x, z', [([0.0], [0.0]), ([0.0, 1.0], [0.0]), ([0.0, 0.0], [0.0, 1.0]), ([0.0, np.nan], [0.0, 0.0])])
def testInvalidProfilesAreRejected(x, z):

    with pytest.raises(InvalidInputError):
        Bathymetry(x, z, stillWaterLevel = 1.0)

def testAPlaneBeachNeedsItsToeInsideTheProfile():

    with pytest.raises(InvalidInputError):
        Bathymetry.planeBeach(depth, toeX = 70.0, slope = slope, endX = endX)
    with pytest.raises(InvalidInputError):
        Bathymetry.planeBeach(depth, toeX, slope = -0.1, endX = endX)
