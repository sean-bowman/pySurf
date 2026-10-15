
# -- pySurf: Geometry Tools Tests -- #

'''

Checks on the curve and surface primitives in pysurf.geometryTools.

Each primitive is checked against a case with a known answer: a Bezier against its own end
points and tangents, a subdivision against the curve it splits, an offset of a circle against a
concentric circle, a fillet between two perpendicular lines against a quarter circle tangent to
both, a rotation against its inverse. The intersection tests carry over from NOVA, where the
first and last segment of a curve were once missed.

Sean Bowman [10/10/2026]

'''

import numpy as np
import pytest

from pysurf.geometryTools import (DCM, bezierCurve, bezierEval, bezierSubdivide, fillet, intersection,
                                  lineIntersection, nonLinspace, parallelOffset, revolveContour)

#----------------------------------------------------------------------#
# -- Bezier Curves -- #
#----------------------------------------------------------------------#

class TestBezierCurve:

    '''The curve starts and ends where asked, leaving and arriving at the angles asked.'''

    def testEndPointsAreHeld(self):

        (x, y) = bezierCurve([0.0, 1.0], [3.0, 2.0], 30.0, -10.0, [0.35, 0.35], 200)

        assert (x[0], y[0]) == pytest.approx((0.0, 1.0), abs = 1e-12)
        assert (x[-1], y[-1]) == pytest.approx((3.0, 2.0), abs = 1e-12)

    @pytest.mark.parametrize('departure, arrival', [(30.0, -10.0), (95.0, 0.0), (-42.0, 82.0)])
    def testTangentsMatchTheAngles(self, departure, arrival):

        '''The end tangent of a cubic points from p1 to p2 and from p3 to p4.'''

        (x, y) = bezierCurve([0.0, 0.0], [3.0, 1.0], departure, arrival, [0.35, 0.35], 20001)

        departureAngle = np.degrees(np.arctan2(y[1] - y[0], x[1] - x[0]))
        arrivalAngle = np.degrees(np.arctan2(y[-1] - y[-2], x[-1] - x[-2]))

        assert departureAngle == pytest.approx(departure, abs = 0.05)
        assert arrivalAngle == pytest.approx(arrival, abs = 0.05)

    def testHandlesScaleWithTheChord(self):

        '''Magnitudes are fractions of the chord, so the handle sits that fraction away.'''

        _, controlPoints = bezierCurve([0.0, 0.0], [4.0, 0.0], 0.0, 0.0, [0.25, 0.5], 10, returnControlPoints = True)

        assert controlPoints['p2'] == pytest.approx([1.0, 0.0])
        assert controlPoints['p3'] == pytest.approx([2.0, 0.0])

class TestBezierSubdivide:

    '''de Casteljau subdivision reproduces the original curve exactly.'''

    controlPolygon = (np.array([0.0, 0.0]), np.array([1.0, 2.0]), np.array([3.0, 3.0]), np.array([4.0, 0.0]))

    @pytest.mark.parametrize('t', [0.5, 0.2, 0.83])
    def testTheHalvesLieOnTheOriginalCurve(self, t):

        left, right = bezierSubdivide(*self.controlPolygon, t = t)
        (xFull, yFull) = bezierEval(*self.controlPolygon, 4001)

        for half, (start, stop) in ((left, (0.0, t)), (right, (t, 1.0))):
            (xHalf, yHalf) = bezierEval(*half, 101)
            # The half evaluated at its own parameter s equals the full curve at start + s * (stop - start)
            parameter = start + np.linspace(0.0, 1.0, 101) * (stop - start)
            u = 1.0 - parameter
            p1, p2, p3, p4 = self.controlPolygon
            xExpected = u**3 * p1[0] + 3 * parameter * u**2 * p2[0] + 3 * parameter**2 * u * p3[0] + parameter**3 * p4[0]
            yExpected = u**3 * p1[1] + 3 * parameter * u**2 * p2[1] + 3 * parameter**2 * u * p3[1] + parameter**3 * p4[1]
            assert xHalf == pytest.approx(xExpected, abs = 1e-12)
            assert yHalf == pytest.approx(yExpected, abs = 1e-12)

    def testTheJoinIsContinuousInPositionAndTangent(self):

        left, right = bezierSubdivide(*self.controlPolygon)

        assert left[3] == pytest.approx(right[0])
        leftTangent = left[3] - left[2]
        rightTangent = right[1] - right[0]
        crossProduct = leftTangent[0] * rightTangent[1] - leftTangent[1] * rightTangent[0]
        assert crossProduct == pytest.approx(0.0, abs = 1e-12)
        assert np.dot(leftTangent, rightTangent) > 0.0

#----------------------------------------------------------------------#
# -- Intersections -- #
#----------------------------------------------------------------------#

def crossingsOfAVerticalLine(at, xCurve, yCurve):

    '''Where a vertical line at x = at crosses the curve.'''

    return intersection(np.asarray(xCurve, dtype = float), np.asarray(yCurve, dtype = float),
                        np.array([at, at]), np.array([-1.0, 1.0]))

class TestIntersection:

    '''Every segment is tested, and one crossing in the first segments is found.'''

    xCurve, yCurve = [0.0, 0.5, 1.0], [0.0, 0.0, 0.0]

    @pytest.mark.parametrize('at', [0.2, 0.9])
    def testACrossingInTheFirstOrLastSegmentIsFound(self, at):

        x, y = crossingsOfAVerticalLine(at, self.xCurve, self.yCurve)

        assert np.ravel(x).tolist() == [pytest.approx(at)]
        assert np.ravel(y).tolist() == [pytest.approx(0.0)]

    def testAMissIsEmpty(self):

        x, y = crossingsOfAVerticalLine(1.2, self.xCurve, self.yCurve)

        assert len(x) == 0 and len(y) == 0

    def testTwoSegmentsCrossingAtTheirFirstSegmentsAreFound(self):

        x, y = intersection(np.array([0.0, 1.0]), np.array([0.0, 1.0]),
                            np.array([0.0, 1.0]), np.array([1.0, 0.0]))

        assert np.ravel(x).tolist() == [pytest.approx(0.5)]
        assert np.ravel(y).tolist() == [pytest.approx(0.5)]

    def testEveryCrossingOfALongCurveIsFound(self):

        # A sine sampled coarsely crosses its axis once per half period, the last one included
        x = np.linspace(0.0, 3.0, 31)
        y = np.sin(np.pi * x + 0.3)
        xCross, _ = intersection(x, y, np.array([-1.0, 4.0]), np.array([0.0, 0.0]))

        expected = (np.arange(1, 4) - 0.3 / np.pi)
        assert np.allclose(np.sort(np.ravel(xCross)), expected, atol = 5e-3)

def testLineIntersectionOfTwoKnownLines():

    '''y = x through the origin and y = 2 - x meet at (1, 1).'''

    x, y = lineIntersection([0.0, 0.0], np.pi / 4, [2.0, 0.0], 3 * np.pi / 4)

    assert (x, y) == pytest.approx((1.0, 1.0))

#----------------------------------------------------------------------#
# -- Offsets and Fillets -- #
#----------------------------------------------------------------------#

def testOffsetOfACircleIsAConcentricCircle():

    '''A circle of radius 2 offset by 0.5 along its normal is a circle of radius 1.5 or 2.5.'''

    theta = np.linspace(0.0, np.pi, 2001)
    x, y = 2.0 * np.cos(theta), 2.0 * np.sin(theta)

    xOffset, yOffset = parallelOffset(x, y, 0.5)
    radius = np.hypot(xOffset, yOffset)[5:-5]

    # Counterclockwise travel puts the positive offset on the inside
    assert radius == pytest.approx(1.5, abs = 1e-5)

def testFilletBetweenPerpendicularLinesIsATangentQuarterCircle():

    '''

    A fillet of radius r in the corner of y = 0 (x from 2 to 0) and x = 0 (y from 0 to 2) has its
    center at (r, r) and touches each line at distance r from the corner.

    '''

    radius = 0.5
    xFillet, yFillet = fillet(np.array([2.0, 0.0]), np.array([0.0, 0.0]),
                              np.array([0.0, 0.0]), np.array([0.0, 2.0]), radius, numPoints = 200)

    distanceFromCenter = np.hypot(xFillet - radius, yFillet - radius)
    assert distanceFromCenter == pytest.approx(radius, abs = 1e-9)

    endPoints = sorted([(round(xFillet[0], 9), round(yFillet[0], 9)), (round(xFillet[-1], 9), round(yFillet[-1], 9))])
    assert endPoints[0] == pytest.approx((0.0, radius), abs = 1e-9)
    assert endPoints[1] == pytest.approx((radius, 0.0), abs = 1e-9)

#----------------------------------------------------------------------#
# -- Spacing, Revolution and Rotation -- #
#----------------------------------------------------------------------#

@pytest.mark.parametrize('method', ['exp', 'cos', 'log'])
def testNonLinspaceHoldsItsEndsAndIncreases(method):

    values = nonLinspace(2.0, 5.0, 50, method = method)

    assert values[0] == pytest.approx(2.0)
    assert values[-1] == pytest.approx(5.0)
    assert np.all(np.diff(values) > 0.0)

def testNonLinspaceCosineClustersAtBothEnds():

    spacing = np.diff(nonLinspace(0.0, 1.0, 41, method = 'cos'))

    assert spacing[0] < spacing[20] and spacing[-1] < spacing[20]

def testNonLinspaceRefusesAnUnknownMethodByName():

    with pytest.raises(ValueError, match = 'method'):
        nonLinspace(0.0, 1.0, 10, method = 'quadratic')

def testRevolvedContourKeepsItsRadius():

    xContour = np.linspace(0.0, 1.0, 11)
    rContour = 0.2 + 0.1 * xContour

    x, y, z = revolveContour(xContour, rContour, numSlices = 37)

    assert np.hypot(x, y) == pytest.approx(np.tile(rContour, (37, 1)), abs = 1e-12)
    assert z == pytest.approx(np.tile(xContour, (37, 1)), abs = 1e-12)

@pytest.mark.parametrize('order', ['zyx', 'yzx', 'xyz'])
def testTheTransposedRotationUndoesTheRotation(order):

    '''

    DCM with transpose = True is the inverse rotation: the transpose of the whole product. A
    transpose applied to one factor alone fails this for any angle set with more than one axis.

    '''

    rng = np.random.default_rng(7)
    points = rng.normal(size = (3, 25))
    angles = [0.3, -0.7, 1.1]

    rotated = np.array(DCM(angles, points, rotationOrder = order))
    restored = np.array(DCM(angles, rotated, transpose = True, rotationOrder = order))

    assert restored == pytest.approx(points, abs = 1e-12)

def testRotationPreservesLength():

    points = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, -1.0], [0.5, 0.5, 0.5]])
    rotated = np.array(DCM([0.4, 0.2, -1.3], points))

    assert np.linalg.norm(rotated, axis = 0) == pytest.approx(np.linalg.norm(points, axis = 0))

def testDcmRefusesAnUnknownOrder():

    with pytest.raises(ValueError, match = 'rotationOrder'):
        DCM([0.0, 0.0, 0.0], np.eye(3), rotationOrder = 'zxy')
