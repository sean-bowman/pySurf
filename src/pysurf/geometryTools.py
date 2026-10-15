
# -- pySurf: Curve and Surface Geometry Primitives -- #

'''

Pure geometry math with no board state and no physics: cubic Bezier construction and
subdivision, curve offsets, intersections, fillets, arc-length resampling, revolutions, and the
direction-cosine rotation.

Every module that builds a board profile, a rail, or a swept surface calls in here rather than
differentiating a curve or rotating a point cloud itself, so one numerical technique backs the
planform, the rocker, and the cross sections alike.

The curve tools are shared with NOVA's geometryTools. Where the two copies differ, this file
carries the correct one of each and tests/testGeometryTools.py pins the behavior:

    intersection    every segment of both curves is a candidate, the last one included, and a
                    single crossing in the first segment of both curves is reported
    fillet          the fourth offset intersection keeps its own y array
    DCM             the transposed rotation is the transpose of the full product
    arcSpline       vectorized arc-length inversion with a shape-preserving default fit

Sean Bowman [10/10/2026]

'''

# Third-party imports
import numpy as np

# Permissive numeric-input alias: these helpers accept arrays, lists, or scalars interchangeably.
ArrayLike = np.ndarray | list | float | int

#----------------------------------------------------------------------#
# -- Bezier Curves -- #
#----------------------------------------------------------------------#

def bezierCurve(p1       : list[float],
                p4       : list[float],
                theta_1  : float,
                theta_2  : float,
                magnitude: list[float],
                res      : int,
                returnControlPoints: bool = False) -> list[list[float]] | tuple:

    '''

    Create a cubic Bezier curve from start/end points, departure/arrival angles, and magnitudes.

    Constructs a cubic Bezier curve by computing interior control points (p2, p3)
    from the specified start/end angles and magnitude scaling factors. The magnitudes
    are normalized relative to the chord length between p1 and p4.

    Parameters:
    -----------
    p1 : list[float]
        Start point as [x, y].
    p4 : list[float]
        End point as [x, y].
    theta_1 : float
        Departure angle at p1, in degrees, measured from the positive x-axis.
        Controls the direction the curve leaves the start point.
    theta_2 : float
        Arrival angle at p4, in degrees, measured from the positive x-axis.
        Controls the direction the curve arrives at the end point.
    magnitude : list[float]
        Two-element list [mag1, mag2] controlling the distance of the interior
        control points from p1 and p4 respectively, as fractions of the chord length.
        Larger values produce a fuller curve; typical range 0.2 to 0.6.
    res : int
        Number of points to sample along the Bezier curve.
    returnControlPoints : bool, optional
        If True, also returns a dict with the four control points and the two tangent
        line segments. Default False.

    Returns:
    --------
    list[list[float]] : [x, y] coordinate lists along the curve.
    dict (only when returnControlPoints=True) : {
        'p1': start point,
        'p2': first interior control point,
        'p3': second interior control point,
        'p4': end point,
        'tangent1': [p1, p2]: departure tangent line segment,
        'tangent2': [p3, p4]: arrival tangent line segment,
    }

    '''

    # Scale magnitudes by chord length so values are shape-independent fractions.
    # Use local copies to avoid mutating the caller's list.
    chordLength = np.sqrt(((p4[1] - p1[1])**2) + ((p4[0] - p1[0])**2))
    mag1 = magnitude[0] * chordLength
    mag2 = magnitude[1] * chordLength

    # Build interior control points from angles and scaled magnitudes.
    # p2 is offset from p1 in the direction of theta_1.
    # p3 is offset from p4 opposite to the direction of theta_2
    # (i.e., the curve arrives at p4 traveling in the theta_2 direction).
    # cos/sin carry the sign of both components, so angles beyond +/-90 degrees place the
    # handle behind the end point rather than mirroring it.
    p2 = [p1[0] + mag1 * np.cos(np.radians(theta_1)),
          p1[1] + mag1 * np.sin(np.radians(theta_1))]

    p3 = [p4[0] - mag2 * np.cos(np.radians(theta_2)),
          p4[1] - mag2 * np.sin(np.radians(theta_2))]

    # Cubic Bezier parametric evaluation in the Bernstein basis
    def bernstein(p1: float, p2: float, p3: float, p4: float, t: np.ndarray) -> np.ndarray:
        return (1-t)**3 * p1 + 3*t*(1-t)**2 * p2 + 3*t**2*(1-t) * p3 + t**3 * p4

    t = np.linspace(0, 1, res)
    x = bernstein(p1[0], p2[0], p3[0], p4[0], t)
    y = bernstein(p1[1], p2[1], p3[1], p4[1], t)

    if returnControlPoints:
        controlPoints = {
            'p1'      : p1,
            'p2'      : p2,
            'p3'      : p3,
            'p4'      : p4,
            'tangent1': [p1, p2],  # departure tangent at start
            'tangent2': [p3, p4],  # arrival tangent at end
        }
        return [x, y], controlPoints

    return [x, y]

def bezierSubdivide(p1       : list[float],
                    p2       : list[float],
                    p3       : list[float],
                    p4       : list[float],
                    t        : float = 0.5) -> tuple:

    '''

    Exactly subdivide a cubic Bezier at parameter t using de Casteljau's algorithm.

    Returns two sets of 4 control points that together reproduce the original curve
    exactly: no approximation. Useful for splitting a segment at any interior point
    while preserving the full shape.

    Parameters:
    -----------
    p1, p2, p3, p4 : list[float]
        The four cubic Bezier control points as [x, y].
    t : float
        Parameter value at which to subdivide (0 < t < 1). Default 0.5 = midpoint.

    Returns:
    --------
    tuple : (leftCtrl, rightCtrl) where each is a tuple of four [x, y] arrays
        representing the control polygon of the left and right sub-curves.

    '''

    pA = np.array(p1, dtype=float)
    pB = np.array(p2, dtype=float)
    pC = np.array(p3, dtype=float)
    pD = np.array(p4, dtype=float)

    # Level 1: linear interpolation between adjacent pairs
    pAB  = (1 - t) * pA  + t * pB
    pBC  = (1 - t) * pB  + t * pC
    pCD  = (1 - t) * pC  + t * pD

    # Level 2
    pABC = (1 - t) * pAB + t * pBC
    pBCD = (1 - t) * pBC + t * pCD

    # Level 3: point on the curve at parameter t
    pM   = (1 - t) * pABC + t * pBCD

    leftCtrl  = (pA, pAB, pABC, pM)
    rightCtrl = (pM, pBCD, pCD, pD)

    return leftCtrl, rightCtrl

def bezierEval(p1               : np.ndarray,
               p2               : np.ndarray,
               p3               : np.ndarray,
               p4               : np.ndarray,
               res              : int,
               returnControlPoints: bool = False) -> list | tuple:

    '''

    Evaluate a cubic Bezier curve directly from its four control points using the
    Bernstein basis. Companion to bezierSubdivide: accepts the control point arrays
    that bezierSubdivide returns without needing to convert back to angle/magnitude form.

    Parameters:
    -----------
    p1, p2, p3, p4 : np.ndarray
        Control points as arrays of shape (2,): [x, y].
    res : int
        Number of sample points along the curve.
    returnControlPoints : bool, optional
        If True, also return a control-point dict matching the bezierCurve format.

    Returns:
    --------
    list[np.ndarray] : [x, y] coordinate arrays.
    dict (only when returnControlPoints=True) : {p1, p2, p3, p4, tangent1, tangent2}

    '''

    t  = np.linspace(0, 1, res)
    x  = (1-t)**3 * p1[0] + 3*t*(1-t)**2 * p2[0] + 3*t**2*(1-t) * p3[0] + t**3 * p4[0]
    y  = (1-t)**3 * p1[1] + 3*t*(1-t)**2 * p2[1] + 3*t**2*(1-t) * p3[1] + t**3 * p4[1]

    if returnControlPoints:
        cp = {
            'p1'      : list(p1),
            'p2'      : list(p2),
            'p3'      : list(p3),
            'p4'      : list(p4),
            'tangent1': [list(p1), list(p2)],
            'tangent2': [list(p3), list(p4)],
        }
        return [x, y], cp

    return [x, y]

#----------------------------------------------------------------------#
# -- Curve Tools -- #
#----------------------------------------------------------------------#

def parallelOffset(xCurve: ArrayLike, yCurve: ArrayLike, offsetDistance: ArrayLike, centralDifference: bool = True) -> tuple:

    '''

    This function takes in the X and Y coordinates of a curve and creates a curve that is
    an equal distance to that curve in the curve normal direction. This acts like a curve offset
    in a traditional CAD software. The equations that drive this process are given as:

    x_parallel = x + (-offset_distance)*dy / sqrt(dx^2 + dy^2)
    y_parallel = y - (-offset_distance)*dx / sqrt(dx^2 + dy^2)

    '''

    # If the user has a constant offset distance, make it into an array for the loop
    if isinstance(offsetDistance, float):
        offsetDistance = offsetDistance * np.ones(len(xCurve))

    # Declare empty arrays
    deltaX = np.zeros(len(xCurve))
    deltaY = np.zeros(len(yCurve))
    xCurveParalleloffset = np.zeros(len(xCurve))
    yCurveParalleloffset = np.zeros(len(xCurve))

    # Do numerical derivative for passed in curve
    if centralDifference:
        deltaX = np.gradient(xCurve)
        deltaY = np.gradient(yCurve)
    else:
        for i in range(len(xCurve) - 1):
            deltaX[i] = xCurve[i+1] - xCurve[i]
            deltaY[i] = yCurve[i+1] - yCurve[i]
        deltaX[-1] = deltaX[-2]
        deltaY[-1] = deltaY[-2]

    norm = np.sqrt(deltaX**2 + deltaY**2)

    # Calculate points of parallel offset curve
    xCurveParalleloffset = xCurve + -offsetDistance * deltaY / norm
    yCurveParalleloffset = yCurve - -offsetDistance * deltaX / norm

    return xCurveParalleloffset, yCurveParalleloffset

def intersection(xCurve1: ArrayLike, yCurve1: ArrayLike, xCurve2: ArrayLike, yCurve2: ArrayLike) -> tuple:

    r'''

    Given two line segments segment1 and segment2 of curve1 and curve2 with endpoints:

    segment1 endpoints:  (x1[0], y1[0]) and (x1[-1], y1[-1])
    segment2 endpoints:  (x2[0], y2[0]) and (x2[-1], y2[-1])

    we can write four equations with four unknowns and solve them. The
    four unknowns are a, b, x, and y, where (x, y) is the intersection of
    segment1 and segment2. a is the distance from the starting point of segment1 to the
    intersection relative to the length of segment1, and b is the distance from the
    starting point of segment2 to the intersection relative to the length of segment2.
    For this function, we only care about the intersection point: (x, y).

    The four equations are:

    (x1[-1] - x1[0]) * a = x - x1[0]
    (x2[-1] - x2[0]) * b = x - x2[0]
    (y1[-1] - y1[0]) * a = y - y1[0]
    (y2[-1] - y2[0]) * b = y - y2[0]

    Rearranging and writing in matrix form:

    [dx1    0   -1   0      [a      [-x1[0]
      0    dx2  -1   0   *   b   =   -x2[0]
     dy1    0    0  -1       x       -y1[0]
      0    dy2   0  -1]      y]      -y2[0]]

    This is A*z = B. We can solve for z with z = A\B, where A\B is solvable with numpy.linalg.solve(A, B)

    Once we have our solution, z, we just have to look at a and b to determine
    whether segment1 and segment2 intersect.  If 0 <= a <= 1 and 0 <= b <= 1, then the two
    line segments cross and the intersection point is z[3, 4] = (x, y)

    '''

    # Helper functions to calculate moving minimum and maximum for passed in curves: the bounding
    # box of each of the n - 1 segments of an n point curve, the last one included
    def movingMin(array):
        if len(array) == 2:
            movingMinimum = np.minimum(array[0],array[1])
        else:
            movingMinimum = np.minimum(array[:-1],array[1:])
        return movingMinimum
    def movingMax(array):
        if len(array) == 2:
            movingMaximum = np.maximum(array[0],array[1])
        else:
            movingMaximum = np.maximum(array[:-1],array[1:])
        return movingMaximum

    # If passed in arrays are lists, make them numpy arrays
    if type(xCurve1) is list:
        xCurve1 = np.array(xCurve1)
    if type(xCurve2) is list:
        xCurve2 = np.array(xCurve2)
    if type(yCurve1) is list:
        yCurve1 = np.array(yCurve1)
    if type(yCurve2) is list:
        yCurve2 = np.array(yCurve2)

    # If passed in arrays have no second dimension, give them one for convenience
    if len(xCurve1.shape) < 2:
        xCurve1 = xCurve1.reshape(len(xCurve1),1)
        yCurve1 = yCurve1.reshape(len(yCurve1),1)
    if len(xCurve2.shape) < 2:
        xCurve2 = xCurve2.reshape(len(xCurve2),1)
        yCurve2 = yCurve2.reshape(len(yCurve2),1)

    # Concatenate x and y parts of each curve into a single array and take their derivative
    curve1 = np.concatenate((xCurve1, yCurve1), axis = 1)
    curve2 = np.concatenate((xCurve2, yCurve2), axis = 1)
    dydx1 = np.diff(curve1, axis = 0)
    dydx2 = np.diff(curve2, axis = 0)

    # Draw bounding box for intersection
    foundIndex = np.argwhere((movingMin(xCurve1) <= movingMax(xCurve2).T) &
                             (movingMax(xCurve1) >= movingMin(xCurve2).T) &
                             (movingMin(yCurve1) <= movingMax(yCurve2).T) &
                             (movingMax(yCurve1) >= movingMin(yCurve2).T))

    # Special case for when each array is 2 elements long
    if foundIndex.shape == (1, 1):
        foundIndex_int = np.zeros((1,2), dtype = int)
        foundIndex_int[0,0] = foundIndex_int[0,1] = foundIndex[0][0]
        foundIndex = foundIndex_int
    # Special case for when there is no intersection (return an empty element for x and y). The
    # test is on the count of candidates, not their values: a crossing in the first segment of
    # both curves is the candidate (0, 0).
    elif foundIndex.size == 0:
        xIntersect, yIntersect = [], []
        return xIntersect, yIntersect

    # Count number of intersections and build A, B, and z (output) arrays
    numIntersections = foundIndex.shape[0]
    outputArray = np.zeros((4,numIntersections))
    A = np.zeros((numIntersections,4,4))
    B = np.zeros((4,numIntersections))

    # Build A and B arrays
    A[:, [0, 1], 2] = -1
    A[:, [2, 3], 3] = -1
    A[:, [0, 2], 0] = dydx1[foundIndex[:,0],:]
    A[:, [1, 3], 1] = dydx2[foundIndex[:,1],:]

    B = -np.array([xCurve1[foundIndex[:,0]], xCurve2[foundIndex[:,1]], yCurve1[foundIndex[:,0]], yCurve2[foundIndex[:,1]]]).reshape((4,numIntersections))

    # Perform matrix solve operation to solve for z (outputArray)
    for i in range(numIntersections):
        outputArray[:,i] = np.linalg.solve(A[i,:,:], B[:,i])

    # Find where first two elements of outputArray are between 0 and 1
    inRange = np.argwhere((outputArray[0,:] >= 0) &
                          (outputArray[0,:] <= 1) &
                          (outputArray[1,:] >= 0) &
                          (outputArray[1,:] <= 1))

    # Return x and y coordinates of intersection point by sampling out of the 3rd and 4th elements of outputArray
    xIntersect = outputArray[2,inRange][:]
    yIntersect = outputArray[3,inRange][:]

    return xIntersect, yIntersect

def fillet(xCurve1: ArrayLike, yCurve1: ArrayLike, xCurve2: ArrayLike, yCurve2: ArrayLike, radiusOfCurvature: float, numPoints: int = 50) -> tuple:

    '''

    This function is intended to be used for filleting two linear segments that are
    joined at a single point (or intersect at a single point that can be used as the join point).
    The two segments can be any length and any separation angle, and may be comprised of as few as 2 points each.

    This function works by first creating parallel offset curves of each of the passed in line segments. These offset
    curves are used to determine the unique solution where a fillet is applicable, which exists in the only intersection
    between the offset curves so long as the offset distance is the radius of curvature specified. If there are no intersections,
    then the fillet radius of curvature is too large to create a fillet between the given line segments.

    The fillet start and end angles are determined by the slope of each line segment at its respective endpoint,
    and a conditional statement built with trial and error catches all possible orientations of the fillet
    such that the function will work regardless of the orientation of the passed-in lines.

    TODO: re-orient the lines inside the function so that the fillet always occurs under the same spatial
    coordinates and then transform back to the original reference frame, which removes the orientation conditional.

    '''

    # Find x and y differences for each curve
    dx1 = np.diff(xCurve1)
    dy1 = np.diff(yCurve1)

    dx2 = np.diff(xCurve2)
    dy2 = np.diff(yCurve2)

    # Create parallel offset curves
    xCurve1Offset1, yCurve1Offset1 = parallelOffset(xCurve1, yCurve1, radiusOfCurvature)
    xCurve1Offset2, yCurve1Offset2 = parallelOffset(xCurve1, yCurve1, -radiusOfCurvature)
    xCurve2Offset1, yCurve2Offset1 = parallelOffset(xCurve2, yCurve2, radiusOfCurvature)
    xCurve2Offset2, yCurve2Offset2 = parallelOffset(xCurve2, yCurve2, -radiusOfCurvature)

    # Find intersections between offset curves
    x1CircleIntersect, y1CircleIntersect = intersection(xCurve1Offset1, yCurve1Offset1, xCurve2Offset1, yCurve2Offset1)
    x2CircleIntersect, y2CircleIntersect = intersection(xCurve1Offset1, yCurve1Offset1, xCurve2Offset2, yCurve2Offset2)
    x3CircleIntersect, y3CircleIntersect = intersection(xCurve1Offset2, yCurve1Offset2, xCurve2Offset1, yCurve2Offset1)
    x4CircleIntersect, y4CircleIntersect = intersection(xCurve1Offset2, yCurve1Offset2, xCurve2Offset2, yCurve2Offset2)

    # Store intersection (x, y) pairs
    intersectionPoints    = [0 for x in range(4)]
    intersectionPoints[0] = [x1CircleIntersect, y1CircleIntersect]
    intersectionPoints[1] = [x2CircleIntersect, y2CircleIntersect]
    intersectionPoints[2] = [x3CircleIntersect, y3CircleIntersect]
    intersectionPoints[3] = [x4CircleIntersect, y4CircleIntersect]

    # Check for which of these are empty. intersection() returns an empty list for a miss and an
    # array otherwise, so both forms of empty are tested by size rather than by comparison.
    intersectionFlag = [1 for i in range(4)]
    for i in range(4):
        if np.size(intersectionPoints[i][0]) == 0:
            intersectionFlag[i] = 0
    # Find the index of the non-zero element
    intersectionIndex = [i for i, value in enumerate(intersectionFlag) if value != 0]

    # Calculate start and end angles of the fillet
    thetaFillet1 = np.arctan2(dy1[-1], dx1[-1])
    thetaFillet2 = np.arctan2(dy2[-1], dx2[-1])

    # Conditional statement that catches all possible orientations of the fillet
    if ((thetaFillet1 - thetaFillet2) >= np.pi) & (thetaFillet2 < 0):
        thetaFillet2 = abs(thetaFillet2)
        thetaFillet = np.linspace(thetaFillet1, thetaFillet2, numPoints)
    elif (abs(thetaFillet1 - thetaFillet2) >= np.pi) & (thetaFillet1 < 0):
        thetaFillet1 = abs(thetaFillet1)
        thetaFillet = np.linspace(thetaFillet1, thetaFillet2, numPoints) + np.pi
    elif ((thetaFillet1 < 0) & (thetaFillet2 >= 0)) | ((thetaFillet1 <= 0) & (thetaFillet2 > 0)):
        thetaFillet = np.linspace(thetaFillet1, thetaFillet2, numPoints) - np.pi/2
    else:
        thetaFillet = np.linspace(thetaFillet1, thetaFillet2, numPoints) + np.pi/2

    # Draw a circle centered at the intersection location comprised of only the angles required to draw the fillet
    xFillet = np.ravel(intersectionPoints[intersectionIndex[0]][0])[0] + radiusOfCurvature * np.cos(thetaFillet)
    yFillet = np.ravel(intersectionPoints[intersectionIndex[0]][1])[0] + radiusOfCurvature * np.sin(thetaFillet)

    return xFillet, yFillet

def arcSpline(xPoints: ArrayLike, yPoints: ArrayLike, zPoints: ArrayLike | None = None,
              newNumPoints: int = 100, method: str = 'shapePreserving') -> tuple:

    '''

    Resample a polyline onto points spaced evenly along its own arc length.

    The input is a curve given as unevenly spaced points, in 2D or 3D. The output is
    `newNumPoints` points on a smooth curve through them, spaced at equal arc length rather than
    at equal parameter.

    Parameters:
    -----------
    xPoints, yPoints : ArrayLike
        The curve. Consecutive duplicates are dropped before fitting.
    zPoints : ArrayLike | None
        Third coordinate. None makes the call 2D and returns two arrays.
    newNumPoints : int
        Points in the result.
    method : str
        'shapePreserving' fits a PCHIP interpolant, which cannot leave the range of the points it
        passes through. 'curvatureContinuous' fits a natural cubic, which is C2 but overshoots at
        a corner.

    Returns:
    --------
    tuple
        (x, y) for a 2D call, (x, y, z) for a 3D one.

    Raises:
    -------
    ValueError
        If fewer than two distinct points remain after duplicates are dropped, or if `method` is
        not one of the two above.

    ----------------------------------------------------------------------
                    Why the default fit is shape preserving
    ----------------------------------------------------------------------

    A C2 cubic through a curve with a corner in it must overshoot: continuity of curvature across
    a point where the curvature is genuinely discontinuous can only be bought by bending the
    curve out past the data on both sides. Stitched curves have corners at their joins (a
    planform section meeting its neighbor, a rail meeting the deck), and an overshoot there is
    geometry nobody asked for. PCHIP overshoots by zero, which is a property of the interpolant.

    The cost is curvature: PCHIP is C1, so the second derivative jumps at each input point.
    'curvatureContinuous' is there for a caller that knows its input is smooth and wants C2.

    ----------------------------------------------------------------------
                    Arc length
    ----------------------------------------------------------------------

    Spacing points evenly along the curve means inverting s(t), the arc length as a function of
    the spline parameter. That is done by sampling the fitted curve densely, integrating the
    speed |dP/dt| with the trapezoid rule, and interpolating the inverse.

    '''

    from scipy.interpolate import CubicSpline, PchipInterpolator

    fitters = {'shapePreserving': PchipInterpolator, 'curvatureContinuous': CubicSpline}
    if method not in fitters:
        raise ValueError(f'arcSpline: method must be one of {sorted(fitters)}, got {method!r}.')

    is3D = zPoints is not None
    if not is3D:
        zPoints = np.zeros(len(xPoints))

    points = np.column_stack([np.asarray(array, dtype = float).ravel()
                              for array in (xPoints, yPoints, zPoints)])

    # A repeated point gives a zero-length segment, which is a zero-width parameter interval and
    # an unbounded derivative
    segment = np.linalg.norm(np.diff(points, axis = 0), axis = 1)
    scale = segment.sum()
    if scale <= 0:
        raise ValueError('arcSpline: every point is coincident, so there is no curve to resample.')

    keep = np.insert(segment > scale * 1e-12, 0, True)
    points = points[keep]

    if len(points) < 2:
        raise ValueError('arcSpline: fewer than two distinct points, so there is no curve.')

    # Chordal parameterization, normalized so the fit is scale independent
    segment = np.linalg.norm(np.diff(points, axis = 0), axis = 1)
    parameter = np.insert(np.cumsum(segment / segment.sum()), 0, 0.0)

    fitter = fitters[method]
    curves = [fitter(parameter, points[:, i]) for i in range(3)]

    # Arc length along the fitted curve. The sample count scales with the input so a long,
    # detailed curve is not integrated more coarsely than a short one.
    sampleCount = max(4001, 40 * len(points))
    dense = np.linspace(0.0, 1.0, sampleCount)

    speed = np.linalg.norm(np.column_stack([curve.derivative()(dense) for curve in curves]),
                           axis = 1)
    arclength = np.concatenate([[0.0],
                                np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(dense))])

    # Invert s(t) at evenly spaced arc lengths. np.interp needs an increasing first argument,
    # which arclength is by construction: speed is a norm, so it cannot be negative.
    targets = np.linspace(0.0, arclength[-1], newNumPoints)
    atParameter = np.interp(targets, arclength, dense)

    resampled = np.column_stack([curve(atParameter) for curve in curves])

    if is3D:
        return resampled[:, 0], resampled[:, 1], resampled[:, 2]

    return resampled[:, 0], resampled[:, 1]

def lineIntersection(point1: ArrayLike, angle1: float, point2: ArrayLike, angle2: float) -> tuple:

    '''

    This function takes in, for two different lines, a point on each line and the angle that each
    line makes with the X axis (radians) and calculates the intersection point between the two lines.

    '''

    slope1, slope2 = np.tan(angle1), np.tan(angle2)
    yIntercept1, yIntercept2 = point1[1] - slope1 * point1[0], point2[1] - slope2 * point2[0]

    xIntersection = (yIntercept2 - yIntercept1) / (slope1 - slope2)
    yIntersection = (slope1 * xIntersection + yIntercept1 + slope2 * xIntersection + yIntercept2) / 2

    return xIntersection, yIntersection

def nonLinspace(start: float, end: float, numPoints: int = 100, method: str = 'exp') -> np.ndarray:

    '''

    Create a vector of nonlinearly spaced points from a start, end, and method.

    'cos' clusters points at both ends, which is the distribution a lofted board wants: the
    curvature of the planform and rocker is highest at the nose and tail tips.

    Available methods:
    - Exponential ['exp']: clustered toward start
    - Cosine      ['cos']: clustered toward both ends
    - Logarithmic ['log']: clustered toward end

    Raises:
    -------
    ValueError : If method is not one of the three above

    '''

    match method:
        case 'exp':
            # 10**linspace runs from 1 to expFactor; mapping that onto [0, 1] holds both ends
            expFactor = 20
            nonLinearVector = (end - start) * (10**np.linspace(0, np.log10(expFactor), numPoints) - 1) / (expFactor - 1) + start
        case 'cos':
            nonLinearVector = (end - start) * (0.5 * (1 - np.cos(np.linspace(0, np.pi, numPoints)))) + start
        case 'log':
            logFactor = 1.5
            nonLinearVector = (end - start) / logFactor * np.log10(np.linspace(0, 10**logFactor - 1, numPoints) + 1) + start
        case _:
            raise ValueError(f'nonLinspace: method must be one of \'exp\', \'cos\', \'log\', got {method!r}.')

    return nonLinearVector

#----------------------------------------------------------------------#
# -- Surfaces and Rotation -- #
#----------------------------------------------------------------------#

def revolveContour(xContour: list | np.ndarray, rContour: list | np.ndarray, numSlices: int = 50) -> tuple:

    '''

    Given a 2D contour, revolve it about the central axis to create a surface.

    '''

    if isinstance(xContour, list):
        xContour  = np.array(xContour)
        rContour  = np.array(rContour)

    zContour       = np.zeros(len(xContour))
    afterRevolve   = np.zeros((numSlices, 3, len(xContour)))
    rotationMatrix = np.zeros((numSlices, 3, 3))
    xContourRevolved, yContourRevolved, zContourRevolved = [np.zeros((numSlices, len(xContour))) for _ in range(3)]
    rollAngle      = np.linspace(0, 2 * np.pi, numSlices)

    beforeRevolve = np.array([zContour, rContour, xContour])

    for i in range(numSlices):

        # Create an array to hold the Euler angles for the rotation calculation
        rotationValue = [0, 0, rollAngle[i]]

        # Create the rotation cosine matrices in each cartesian direction
        Rx = np.array([[1, 0, 0],
                [0, np.cos(rotationValue[0]), -np.sin(rotationValue[0])],
                [0, np.sin(rotationValue[0]), np.cos(rotationValue[0])]])

        Ry = np.array([[np.cos(rotationValue[1]), 0, np.sin(rotationValue[1])],
                [0, 1, 0],
                [-np.sin(rotationValue[1]), 0, np.cos(rotationValue[1])]])

        Rz = np.array([[np.cos(rotationValue[2]), -np.sin(rotationValue[2]), 0],
                [np.sin(rotationValue[2]), np.cos(rotationValue[2]), 0],
                [0, 0, 1]])

        # Matrix multiply the rotation matrices to get a single rotation matrix that describes this rotation
        rotationMatrix[i,:,:] = Rz @ Ry @ Rx

        # Apply that rotation matrix to the pre-rotated contour
        afterRevolve[i,:,:] = rotationMatrix[i,:,:] @ beforeRevolve

        # Extract the X, Y, and Z data from the rotated contour
        xContourRevolved[i,:] = afterRevolve[i,0,:]
        yContourRevolved[i,:] = afterRevolve[i,1,:]
        zContourRevolved[i,:] = afterRevolve[i,2,:]

    return xContourRevolved, yContourRevolved, zContourRevolved

def DCM(eulerAngles: ArrayLike, valueMatrix: ArrayLike, transpose: bool = False, rotationOrder: str = 'zyx') -> tuple:

    '''

    Apply direction cosine matrix transformations to 'valueMatrix' given the angles in 'eulerAngles' (radians)
    and the order specified by 'rotationOrder'. With transpose = True the inverse rotation is applied: the
    transpose of the whole product, so DCM(angles, DCM(angles, v), transpose = True) returns v.

    '''

    # Create the rotation cosine matrices in each cartesian direction
    Rx = np.array([                                                                 \
                [1,             0,                        0           ],            \
                [0,             np.cos(eulerAngles[0]),  -np.sin(eulerAngles[0])],  \
                [0,             np.sin(eulerAngles[0]),   np.cos(eulerAngles[0])]]) # X-Axis rotation

    Ry = np.array([                                                                 \
                [np.cos(eulerAngles[1]),  0,              np.sin(eulerAngles[1])],  \
                [0,                       1,              0                     ],  \
                [-np.sin(eulerAngles[1]), 0,              np.cos(eulerAngles[1])]]) # Y-axis rotation

    Rz = np.array([                                                                 \
                [np.cos(eulerAngles[2]), -np.sin(eulerAngles[2]),   0           ],  \
                [np.sin(eulerAngles[2]),  np.cos(eulerAngles[2]),   0           ],  \
                [0,                       0,                        1           ]]) # Z-axis rotation

    # Matrix multiply the rotation matrices to get a single rotation matrix that describes this rotation
    match rotationOrder:
        case 'zyx':
            # Default operation order
            rotationMatrix = Rz @ Ry @ Rx
        case 'yzx':
            rotationMatrix = Ry @ Rz @ Rx
        case 'xyz':
            rotationMatrix = Rx @ Ry @ Rz
        case _:
            raise ValueError(f'DCM: rotationOrder must be one of \'zyx\', \'yzx\', \'xyz\', got {rotationOrder!r}.')

    # The inverse of a rotation is its transpose, taken over the whole product
    if transpose:
        rotationMatrix = rotationMatrix.T

    # Apply the rotation matrix to the point cloud
    rotatedValueMatrix = rotationMatrix @ valueMatrix

    # Extract the X, Y, and Z data from the rotated point cloud
    rotatedX = rotatedValueMatrix[0]
    rotatedY = rotatedValueMatrix[1]
    rotatedZ = rotatedValueMatrix[2]

    return rotatedX, rotatedY, rotatedZ
