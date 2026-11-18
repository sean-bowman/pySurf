
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

def plane(x: np.ndarray, y: np.ndarray) -> np.ndarray:

    return 0.3 + 0.05 * x - 0.02 * y

def testSmoothingKeepsTheGaussianTransformOfASinusoid():

    # The grid ends on nodes of the sine, where odd reflection continues it exactly
    step, sigma, wavelength = 0.05, 0.1, 1.0
    k = 2.0 * math.pi / wavelength
    field = HeightField.fromFunction(lambda x, y: 0.2 * np.sin(k * x) + 0.0 * y, (0.0, 3.0, 0.0, 1.0), step,
                                     stillWaterLevel = 1.0, padding = 0)
    smooth = field.smoothed(sigma)
    away = np.abs(field.z) > 0.05
    kept = smooth.z[away] / field.z[away]

    # The same share everywhere, edges included; the weights stop at four sigma, so the share
    # departs from the transform by 4e-5
    assert kept.max() - kept.min() < 1e-12
    assert kept == pytest.approx(math.exp(-0.5 * (k * sigma)**2), rel = 1e-4)

def testSmoothingPassesAPlaneUnchangedToItsEdges():

    field = HeightField.fromFunction(plane, (0.0, 3.0, -1.0, 2.0), 0.1, stillWaterLevel = 1.0)

    assert field.smoothed(0.3).z == pytest.approx(field.z, abs = 1e-12)

@pytest.mark.parametrize('sigma', [0.0, -0.1, 2.0])
def testSmoothingNeedsAPositiveLengthInsideTheGrid(sigma):

    field = HeightField.fromFunction(plane, (0.0, 3.0, -1.0, 2.0), 0.1, stillWaterLevel = 1.0)
    with pytest.raises(InvalidInputError):
        field.smoothed(sigma)

@pytest.mark.parametrize('bounds', [(6.13, 9.31, -1.71, 2.2), (0.0, 3.0, -4.0, -2.5), (17.0, 19.0, 3.0, 4.0)])
def testACropReadsItsParentsBed(bounds):

    field = HeightField.ridge(0.25, gridSpacing = 1.0 / 16.0)
    crop = field.cropped(bounds)
    rng = np.random.default_rng(3)
    x, y = rng.uniform(bounds[0], bounds[1], 400), rng.uniform(bounds[2], bounds[3], 400)

    assert crop.bounds == bounds
    assert crop.x.size < field.x.size
    assert crop.elevation(x, y) == pytest.approx(field.elevation(x, y), abs = 1e-13)
    assert crop.gradient(x, y) == pytest.approx(field.gradient(x, y), abs = 1e-12)

@pytest.mark.parametrize('bounds', [(-1.0, 3.0, -1.0, 1.0), (2.0, 21.0, -1.0, 1.0), (3.0, 3.0, -1.0, 1.0)])
def testACropMustLieInsideItsParent(bounds):

    with pytest.raises(InvalidInputError):
        HeightField.ridge(0.25, gridSpacing = 1.0 / 16.0).cropped(bounds)

def testATextGridReadsBackItsField(tmp_path):

    field = HeightField.fromFunction(quadratic, (0.0, 3.0, -1.0, 2.0), 0.13, stillWaterLevel = 1.0)
    gridX, gridY = np.meshgrid(field.x, field.y)
    points = np.column_stack([gridX.ravel(), gridY.ravel(), field.z.ravel()])
    points = points[np.random.default_rng(5).permutation(len(points))]
    elevations, depths = tmp_path / 'elevations.csv', tmp_path / 'depths.txt'
    np.savetxt(elevations, points, fmt = '%.17g', delimiter = ',', header = 'x, y, z')
    np.savetxt(depths, np.column_stack([points[:, :2], 1.0 - points[:, 2]]), fmt = '%.17g')

    for path, isDepth in ((elevations, False), (depths, True)):
        read = HeightField.fromGridFile(path, stillWaterLevel = 1.0, bounds = field.bounds, depths = isDepth)
        assert read.x == pytest.approx(field.x, abs = 1e-12)
        assert read.y == pytest.approx(field.y, abs = 1e-12)
        assert read.z == pytest.approx(field.z, abs = 1e-12)

@pytest.mark.parametrize('rows', [
    ['0 0 0', '1 0 0', '2 0 0', '3 0 0'] * 4,
    [f'{x} {y} 0' for x in range(4) for y in range(4)][:-1],
    [f'{x} {y} 0' for x in range(4) for y in range(4)][:-1] + ['0 0 0'],
    [f'{x} {y} 0' for x in range(4) for y in range(4)] + ['1 2'],
    [f'{x} {y} 0' for x in range(4) for y in range(4)] + ['1 2 deep'],
    [f'{x} {y} 0' for x in (0, 1, 2, 4) for y in range(4)],
])
def testIrregularTextGridsAreRejected(tmp_path, rows):

    path = tmp_path / 'grid.txt'
    path.write_text('\n'.join(rows) + '\n', encoding = 'utf-8')
    with pytest.raises(InvalidInputError):
        HeightField.fromGridFile(path, stillWaterLevel = 1.0)

def secondDegree(x: np.ndarray, y: np.ndarray) -> np.ndarray:

    # Total degree two stays so under a turn; quadratic()'s x^2 y^2 term would not
    return 0.3 + 0.05 * x - 0.02 * y + 0.01 * x * x - 0.03 * x * y + 0.02 * y * y

@pytest.mark.parametrize('direction', [0.0, 30.0, 90.0, -135.0])
def testATurnedQuadraticBedIsExact(direction):

    field = HeightField.fromFunction(secondDegree, (-3.0, 3.0, -3.0, 3.0), 0.1, stillWaterLevel = 1.0)
    pivot = (0.4, -0.3)
    turned = field.rotated(direction, (-1.5, 1.6, -1.2, 1.4), pivot = pivot)
    rng = np.random.default_rng(7)
    x, y = rng.uniform(-1.5, 1.6, 400), rng.uniform(-1.2, 1.4, 400)
    angle = math.radians(direction)
    sourceX = pivot[0] + math.cos(angle) * (x - pivot[0]) - math.sin(angle) * (y - pivot[1])
    sourceY = pivot[1] + math.sin(angle) * (x - pivot[0]) + math.cos(angle) * (y - pivot[1])

    assert turned.bounds == (-1.5, 1.6, -1.2, 1.4)
    assert turned.elevation(x, y) == pytest.approx(secondDegree(sourceX, sourceY), abs = 1e-12)

def testAWaveTravellingAlongYSeesTheBedAlongX():

    # A bed rising along y; a wave travelling along +y meets it along +x once the bed turns by 90 degrees
    field = HeightField.fromFunction(lambda x, y: 0.05 * y * y + 0.0 * x, (-4.0, 4.0, -4.0, 4.0), 0.1,
                                     stillWaterLevel = 1.0)
    turned = field.rotated(90.0, (-3.0, 3.0, -2.0, 2.0), pivot = (0.0, 0.0))
    x = np.linspace(-3.0, 3.0, 13)

    for y in (-2.0, 0.0, 1.5):
        assert turned.elevation(x, np.full_like(x, y)) == pytest.approx(0.05 * x * x, abs = 1e-12)

def testATurnedRegionMustLieInsideTheField():

    field = HeightField.fromFunction(quadratic, (-3.0, 3.0, -3.0, 3.0), 0.1, stillWaterLevel = 1.0)
    # The square's corners reach 2.5 sqrt(2) from the centre once turned by 45 degrees
    with pytest.raises(InvalidInputError):
        field.rotated(45.0, (-2.5, 2.5, -2.5, 2.5))

    assert field.rotated(45.0, (-2.1, 2.1, -2.1, 2.1)).bounds == (-2.1, 2.1, -2.1, 2.1)

def testShoalingStartsWhereTheBedFirstRisesBySomeShare():

    # A plane beach from x = 10 on one side of the tank, flat on the other: the first side decides
    field = HeightField.fromFunction(lambda x, y: np.where(y > 0.0, np.maximum(x - 10.0, 0.0) / 20.0, 0.0),
                                     (0.0, 30.0, -1.0, 1.0), 0.05, stillWaterLevel = 1.0)

    assert field.offshoreDepth() == pytest.approx(1.0)
    # The depth falls below 0.95 just past x = 11
    assert field.shoalingStart() == pytest.approx(11.05)
    assert field.shoalingStart(share = 0.2) == pytest.approx(14.05)
    flat = HeightField.fromFunction(lambda x, y: 0.0 * x, (0.0, 5.0, 0.0, 1.0), 0.1, stillWaterLevel = 1.0)
    assert flat.shoalingStart() is None
    with pytest.raises(InvalidInputError):
        field.shoalingStart(share = 1.0)
