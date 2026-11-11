
# -- pySurf: Bathymetry Tests -- #

'''

Checks of pysurf.bathymetry against the closed forms of a plane beach: flat at z = 0 to the toe,
then z = (x - toe) tan(beta), with the still water level at z = d; and of the height field against a
quadratic surface, which cubic convolution reproduces exactly, and the ridge's formula.

Sean Bowman [10/11/2026]

'''

import math

import numpy as np
import pytest

from pysurf.bathymetry import Bathymetry, HeightField
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

def quadratic(x: np.ndarray, y: np.ndarray) -> np.ndarray:

    return 0.3 + 0.05 * x - 0.02 * y + 0.01 * x * x - 0.03 * x * y + 0.02 * y * y + 0.004 * x * x * y * y

def testTheHeightFieldReproducesAQuadraticAndItsGradient():

    field = HeightField.fromFunction(quadratic, (0.0, 3.0, -1.0, 2.0), 0.13, stillWaterLevel = 1.0)
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0.0, 3.0, 500), rng.uniform(-1.0, 2.0, 500)
    gradient = field.gradient(x, y)

    assert field.elevation(x, y) == pytest.approx(quadratic(x, y), abs = 1e-12)
    assert gradient[:, 0] == pytest.approx(0.05 + 0.02 * x - 0.03 * y + 0.008 * x * y * y, abs = 1e-12)
    assert gradient[:, 1] == pytest.approx(-0.02 - 0.03 * x + 0.04 * y + 0.008 * x * x * y, abs = 1e-12)

def testTheHeightFieldPassesThroughItsGridAndHoldsBeyondIt():

    field = HeightField.fromFunction(lambda x, y: np.sin(x) * np.cos(y), (0.0, 2.0, 0.0, 1.0), 0.1, stillWaterLevel = 1.0)
    gridX, gridY = np.meshgrid(field.x, field.y)

    assert field.elevation(gridX, gridY) == pytest.approx(field.z, abs = 1e-14)
    assert field.elevation(100.0, 0.5) == pytest.approx(field.elevation(field.x[-1], 0.5))
    assert field.gradient(100.0, 0.5)[0] == 0.0

def testTheNormalIsAUnitVectorAgainstTheGradient():

    field = HeightField.fromFunction(quadratic, (0.0, 3.0, -1.0, 2.0), 0.13, stillWaterLevel = 1.0)
    x, y = np.array([0.5, 1.7, 2.9]), np.array([-0.5, 0.3, 1.9])
    normal, gradient = field.normal(x, y), field.gradient(x, y)

    assert np.linalg.norm(normal, axis = 1) == pytest.approx(1.0)
    assert np.all(normal[:, 2] > 0.0)
    assert normal[:, 0] / normal[:, 2] == pytest.approx(-gradient[:, 0])
    assert normal[:, 1] / normal[:, 2] == pytest.approx(-gradient[:, 1])

def testTheRidgeFollowsGuyenneAndGrilli():

    field = HeightField.ridge(0.25)
    gridX, gridY = np.meshgrid(field.x, field.y)
    exact = np.maximum(gridX - 5.225, 0.0) / 15.0 / np.cosh(0.25 * gridY)**2
    section = field.section(0.0)

    assert field.bounds == (0.0, 19.0, -4.0, 4.0)
    assert HeightField.ridge(0.25, symmetric = True).bounds == (0.0, 19.0, 0.0, 4.0)
    assert field.z == pytest.approx(exact, abs = 1e-15)
    # The centre line is the 1:15 plane beach past the toe
    assert section.elevation(np.array([8.0, 16.0, 19.0])) == pytest.approx(np.array([2.775, 10.775, 13.775]) / 15.0)
    assert field.lowestElevation() == 0.0

@pytest.mark.parametrize('arguments', [
    dict(x = [0.0, 1.0, 2.0], y = [0.0, 1.0, 2.0, 3.0], z = np.zeros((4, 3))),
    dict(x = [0.0, 1.0, 2.5, 3.0], y = [0.0, 1.0, 2.0, 3.0], z = np.zeros((4, 4))),
    dict(x = [0.0, 1.0, 2.0, 3.0], y = [0.0, 1.0, 2.0, 3.0], z = np.zeros((3, 4))),
])
def testIrregularHeightFieldsAreRejected(arguments):

    with pytest.raises(InvalidInputError):
        HeightField(stillWaterLevel = 1.0, **arguments)
