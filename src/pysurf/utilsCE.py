
# -- General Utilities for Computational Engineering -- #

'''

General-purpose geometry generation, curve manipulation, and mathematical
utility functions for the computational engineering repository.

Sean Bowman [02/09/2026]

'''

# Global imports
import numpy as np

#--------------------------------------------------------------------#
# -- Geometry Generation Tools -- #
#--------------------------------------------------------------------#

def py2cad(filename: str, xData: np.ndarray | list, yData: np.ndarray | list, zData: np.ndarray | list) -> None:

    '''

    Export a surface mesh to a binary .stl file from raw coordinate arrays.

    Takes in arrays containing spatial coordinates of a surface mesh and exports
    the surface as a .stl file. Useful for exporting geometry out of Python CAD
    design tools and into CAD softwares such as NX.

    Parameters:
    -----------
    filename : str
        Output file path. The .stl extension is appended if not provided.
    xData : np.ndarray | list
        X coordinates of the surface mesh, shape (m, n).
    yData : np.ndarray | list
        Y coordinates of the surface mesh, shape (m, n).
    zData : np.ndarray | list
        Z coordinates of the surface mesh, shape (m, n).
        Numpy arrays are anticipated by default, however if a list is passed
        the function will convert it into a numpy array.

    Returns:
    --------
    None : A .stl file is created and saved to the specified filename location.

    '''

    from tqdm import tqdm

    # Helper function to find the facet normal and write the current facet data to the file
    def writeFacet(fileID, point1, point2, point3):

        # Find face normal
        vector1 = point2 - point1
        vector2 = point3 - point1
        vector3 = np.cross(vector1, vector2)
        normal = vector3 / np.sqrt(np.sum(vector3**2))

        # Write data to file, ensure data types are what .stl expects
        fileID.write(np.float32(normal))
        fileID.write(np.float32(point1))
        fileID.write(np.float32(point2))
        fileID.write(np.float32(point3))
        fileID.write(np.int16(0))

        # Declare success flag to count up generated facets
        successFlag = 1

        return successFlag

    # Append file extension if the given name does not contain it
    if '.' not in filename:
        filename += '.stl'
    
    # Locally re-scope mesh data
    # If passed in arrays are lists, make them numpy arrays
    if type(xData) is list:
        x = np.array(xData)
    else:
        x = xData
    if type(yData) is list:
        y = np.array(yData)
    else:
        y = yData
    if type(zData) is list:
        z = np.array(zData)
    else:
        z = zData

    # Initialize facet counter to 0
    nFacets = 0

    # Open a file for writing in binary mode
    fileID = open(filename, 'wb+')
    # .stl files start with 80 characters of metadata, pre-append a message for our .stl files and fill the rest
    # with spaces to eat up the remaining 80 characters
    metadataString = 'Created by py2cad.py [Sean Bowman]'
    # bytearray() casts the strings as unsigned character bytes so that they can be written to tbe binary file
    metadataTitle = bytearray(b'Created by py2cad.py [Sean Bowman]' + b' '*(80 - len(metadataString)))
    fileID.write(metadataTitle)
    # Placeholder for the number of facets that the model has, cast as a 32-bit integer (0 at the start)
    fileID.write(np.int32(nFacets))

    # Loop over all vertices of the mesh and call write_facet() to write mesh data to the .stl file
    for i in tqdm(range(len(z[:,0]) - 1)):
        for j in range(len(z[0,:]) - 1):

            # Draw a triangle to make a facet
            point1 = np.array([[x[i,j],     y[i,j],     z[i,j]]])
            point2 = np.array([[x[i,j+1],   y[i,j+1],   z[i,j+1]]])
            point3 = np.array([[x[i+1,j+1], y[i+1,j+1], z[i+1,j+1]]])
            # Write that facet to the file
            successFlag = writeFacet(fileID, point1, point2, point3)
            # Count 'em up
            nFacets += successFlag

            # Draw the corresponding triangle to the previous one
            point1 = np.array([[x[i+1,j+1], y[i+1,j+1], z[i+1,j+1]]])
            point2 = np.array([[x[i+1,j],   y[i+1,j],   z[i+1,j]]])
            point3 = np.array([[x[i,j],     y[i,j],     z[i,j]]])
            # Write that facet to the file and count it up
            successFlag = writeFacet(fileID, point1, point2, point3)
            nFacets += successFlag

    # After we've written all the facets, move the pointer in the file back to the beginning
    fileID.seek(0,0)
    # Then move it to the end of the metadata string, now we're at the location we put a placeholder for the number
    # of facets
    fileID.seek(len(metadataTitle),0)
    # Write the actual number of facets to the file
    fileID.write(np.int32(nFacets))
    # Don't forget to close the file
    fileID.close()
    
def parallelOffset(xCurve: np.ndarray | list, yCurve: np.ndarray | list, offsetDistance: float | np.ndarray, centralDifference: bool = True) -> list[float, float]:

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

def intersection(xCurve1: np.ndarray | list, yCurve1: np.ndarray | list, xCurve2: np.ndarray | list, yCurve2: np.ndarray | list) -> list[float, float]:

    '''
    
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

    # Helper functions to calculate moving minimum and maximum for passed in curves
    def movingMin(array):
        if len(array) == 2:
            movingMinimum = np.minimum(array[0],array[1])
        else:
            movingMinimum = np.minimum(array[0:-2],array[1:-1])
        return movingMinimum
    def movingMax(array):
        if len(array) == 2:
            movingMaximum = np.maximum(array[0],array[1])
        else:
            movingMaximum = np.maximum(array[0:-2],array[1:-1])
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
    # Special case for when there is no intersection (return an empty element for x and y)
    elif not foundIndex.any():
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

def discreteIntersection(xCurve1: np.ndarray | list, yCurve1: np.ndarray | list, xCurve2: np.ndarray | list, yCurve2: np.ndarray | list, resolution: int = 1000) -> list[float, float]:
    
    '''
    
    Replacement function for testing for curve intersections by checking each curve in discrete steps.

    This function only works in 2D.
    
    '''

    from scipy.interpolate import interp1d
    
    # Create interpolation functions for both curves
    curve1Interpolator = interp1d(xCurve1, yCurve1, kind = 'linear', fill_value = 'extrapolate')
    curve2Interpolator = interp1d(xCurve2, yCurve2, kind = 'linear', fill_value = 'extrapolate')
    
    # Find the overlapping x-range
    xRangeMin = max(min(xCurve1), min(xCurve2))
    xRangeMax = min(max(xCurve1), max(xCurve2))
    
    xRange = np.linspace(xRangeMin, xRangeMax, resolution)

    y1Range = curve1Interpolator(xRange)
    y2Range = curve2Interpolator(xRange)
    
    # Find where the curves cross (sign changes in their difference)
    differences = y1Range - y2Range
    signChanges = np.where(np.diff(np.signbit(differences)))[0]

    if not any(signChanges):
        print('No intersections detected')
        return [], []
    
    # Initialize lists to store caught intersections
    xIntersection, yIntersection = [], []

    for signChange in signChanges:

        # Use linear interpolation between points where sign changes
        xLeft, xRight   = xRange[signChange],  xRange[signChange + 1]
        y1Left, y1Right = y1Range[signChange], y1Range[signChange + 1]
        y2Left, y2Right = y2Range[signChange], y2Range[signChange + 1]
        
        # Find intersection via linear interpolation
        xIntersectionValue = xLeft + (xRight - xLeft) * \
                             (y2Left - y1Left) / ((y1Right - y1Left) - (y2Right - y2Left))
        yIntersectionValue = curve1Interpolator(xIntersectionValue)

        xIntersection.append(xIntersectionValue)
        yIntersection.append(yIntersectionValue)
            
    return xIntersection, yIntersection

def fillet(xCurve1: np.ndarray | list, yCurve1: np.ndarray | list, xCurve2: np.ndarray | list, yCurve2: np.ndarray | list, radiusOfCurvature: float, numPoints: int = 50) -> list[float]:

    '''
    
    This function is intended to be used for filleting two linear segments that are
    joined at a single point (or intersect at a single point that can be used as the join point).
    The two segments can be any length and any separation angle, and may be comprised of as few as 2 points each.

    This function works by first creating parallel offset curves of each of the passed in line segments. These offset
    curves are used to determine the unique solution where a fillet is applicable, which exists in the only intersection
    between the offset curves so long as the offset distance is the radius of curvature specified. If there are no intersections,
    then the fillet radius of curvature is too large to create a fillet between the given line segments.

    The fillet start and end angles are determined by the slope of each line segment at its respective endpoint,
    and a super ugly conditional statement built with trial and error catches all possible orientations of the fillet
    such that the function will work regardless of the orientation of the passed-in lines.

    Future update: Would love to re-orient the lines inside the function so that the fillet always occurs under the same
    spatial coordinates and then transform back to the original reference frame. That will get rid of the ugly
    conditional statement.
    
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

    # Check for which of these are empty
    intersectionFlag = [1 for i in range(4)]
    for i in range(4):
        # If the given index is empty, change the 1 in the array to a 0
        if intersectionPoints[i][0] == []:
            intersectionFlag[i] = 0
    # Find the index of the non-zero element
    intersectionIndex = [i for i, value in enumerate(intersectionFlag) if value != 0]

    # Calculate start and end angles of the fillet
    thetaFillet1 = np.arctan2(dy1[-1], dx1[-1])
    thetaFillet2 = np.arctan2(dy2[-1], dx2[-1])

    # Ugly conditional statement that catches all possible orientations of the fillet
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
    xFillet = intersectionPoints[intersectionIndex[0]][0][0] + radiusOfCurvature * np.cos(thetaFillet)
    yFillet = intersectionPoints[intersectionIndex[0]][1][0] + radiusOfCurvature * np.sin(thetaFillet)

    return xFillet, yFillet

def nonLinspace(start: float, end: float, numPoints: int = 100, method: str = 'exp') -> list[float]:

    '''
    
    Create a vector of nonlinearly spaced points from a start, end, and method.

    Available methods:
    - Exponential ['exp']
    - Cosine      ['cos']
    - Logarithmic ['log']
    
    '''

    match method:
        case 'exp':
            expFactor = 20
            nonLinearVector = (end - start) / expFactor * (10**np.linspace(0, np.log10(expFactor), numPoints)) + start
        case 'cos':
            nonLinearVector = (end - start) * (0.5 * (1 - np.cos(np.linspace(0, np.pi, numPoints)))) + start
        case 'log':
            logFactor = 1.5
            nonLinearVector = (end - start) / logFactor * np.log10(np.linspace(0, 10**logFactor - 1, numPoints) + 1) + start

    return nonLinearVector

def arcSpline(xPoints: np.ndarray, yPoints: np.ndarray, zPoints: np.ndarray = None, newNumPoints: int = 100) -> list[float]:

    '''
    
    This function takes in a collection of 3D points, broken into X, Y, and Z arrays, that are unevenly distributed
    with respect to their arclength. The final input, newNumPoints, specifies the total number of points in the
    final spline that will be evenly distributed with respect to the spline arclength.

    This function has been stolen to be used as an arclength-based spline interpolation tool, the death of the artist is real.
    So now it works with 2D points as well.
    
    '''

    from copy import copy
    import scipy.interpolate as spi
    import scipy.integrate as sps

    if zPoints is None:
        zPoints = np.zeros(len(xPoints))
        is2D = True
        is3D = False
    else:
        is2D = False
        is3D = True

    # Helper functions to handle spline segment integration and event handling for line integration
    def segmentIntegrator(t, y, polyCoefs):
        output = np.zeros(np.size(t))
        for k in range(3):
            output += np.polyval(polyCoefs[k,:], t)**2
        return np.sqrt(output)
    
    def integrationEvents(t, y):
        value = y[0]
        return value
    # scipy solve_ivp requires event functions to be monkey patched with terminal and direction handles
    integrationEvents.terminal  = True
    integrationEvents.direction = 1
    
    # Initialize arrays to hold the newly generated points, as well as reorienting the passed-in points
    newNumPointsArray = np.linspace(0, 1, newNumPoints)
    newPoints = np.zeros((newNumPoints,3))
    oldPoints = np.array([xPoints, yPoints, zPoints]).T

    # Compute linear (chordal) arclength of each curve segment, as well as cumulative arclength
    linearArcLength = np.sqrt(np.sum(np.diff(oldPoints, axis = 0)**2, axis = 1))
    linearArcLengthNormalized = linearArcLength / np.sum(linearArcLength)
    cumulativeLinearArclength = np.cumsum(linearArcLengthNormalized)
    cumulativeLinearArclength = np.insert(cumulativeLinearArclength, 0, 0)

    # Initialize empty arrays to store piecewise cubic splines for each dimension of the old points, as
    # well as the derivative for each spline
    spline, splineDerivative = [[[], [], []] for _ in range(2)]
    # Make a (4, 3) array to be used to take the derivative of each spline (works because each
    # spline segment is by definition a cubic polynomial)
    derivativeArray = np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1], [0, 0, 0]])
    for i in range(3):
        spline[i] = spi.CubicSpline(cumulativeLinearArclength, oldPoints[:,i])
        # Make a copy of the spline object to take the derivative so we don't overwrite the original spline
        # object properties
        splineDerivativeIntermediate = copy(spline[i])
        # Calculate the coefficients of the derivative by differentiating the spline coefficients
        splineDerivativeIntermediate.c = (splineDerivativeIntermediate.c.T @ derivativeArray).T
        splineDerivative[i] = splineDerivativeIntermediate
    
    # Create dummy array to store the coefficients of each spline derivative segment (to be used while integrating)
    polynomialCoefsDerivatives = np.zeros((3, 3))
    # Initialize array to hold integrated spline segment length
    segmentLength = np.zeros(len(xPoints)-1)
    # Set integration option relative tolerance to 1e-9
    integrationOptions = {'rtol': 1e-9}

    # First integration step: Determine the segment lengths of each cubic spline for the original points
    for i in range(len(splineDerivative[i].c[0,:])):

        # Store the coefficients of the polynomials describing the spline derivative for this segment
        for j in range(3):
            polynomialCoefsDerivatives[j,:] = splineDerivative[j].c[:,i]

        # Perform integration along the polynomial over the interval of [0, length of linear normalized spline segments up to this point]
        solution = sps.solve_ivp(lambda t, y: segmentIntegrator(t, y, polynomialCoefsDerivatives), \
                                 [0, linearArcLengthNormalized[i]], [0], **integrationOptions)
        # Store the output solution for cubic spline segment length
        segmentLength[i] = solution.y[0][-1]

    # Calculate total spline length and cumulative spline length for the old spline
    totalSplineLength = np.sum(segmentLength)
    cumulativeSplineLength = np.cumsum(segmentLength)
    cumulativeSplineLength = np.insert(cumulativeSplineLength, 0, 0)

    # Break spline length up over a uniformly spaced array of length newNumPoints
    # and compare the regions where each old point falls relative to the new spacing
    arclengthAlongSpline = newNumPointsArray * totalSplineLength
    bins = np.digitize(arclengthAlongSpline, cumulativeSplineLength) - 1
    bins[-1] = bins[-2]
    newInterpolationPoints = newNumPointsArray

    # Second integration step: Integrate over the newly spaced points and catch "zero-crossing" events,
    # or function events where the sign changes from positive to negative. These are important to track as
    # we re-space the points so that we can specify where the new spline points should land along the integration
    # path. We will be integrating in 't' until the integral crosses the specified value of 'splineSegment', and
    # because we normalized the total length each segment length is also normalized. Importantly, we start
    # the integration at -splineSegment so the event handler for zero-crossings can catch when the function output
    # 'y' reaches 0.
    for i in range(newNumPoints):

        # Define the length of the current spline segment
        splineSegment = arclengthAlongSpline[i] - cumulativeSplineLength[bins[i]]

        # Store the coefficients of the polynomials describing the spline derivative for this segment
        for j in range(3):
            polynomialCoefsDerivatives[j,:] = splineDerivative[j].c[:,bins[i]]

        # Perform aforementioned integration
        try:
            solution = sps.solve_ivp(lambda t, y: segmentIntegrator(t, y, polynomialCoefsDerivatives), \
                                    [0, linearArcLengthNormalized[bins[i]]], [-splineSegment], events = integrationEvents, **integrationOptions)
        except Exception:
            class Solution():
                def __init__(self):
                    self.t_events = [[0]]
            solution = Solution()

        # Scale the new spline sample points by the result of the integration terminated at the zero crossing event
        if any(solution.t_events[0]):
            newInterpolationPoints[i] = solution.t_events[0] + cumulativeLinearArclength[bins[i]]

    # Perform final sampling of the spline at the new interpolation points
    for i in range(3):
        newPoints[:,i] = spline[i](newInterpolationPoints)

    if is3D:
        return newPoints[:,0], newPoints[:,1], newPoints[:,2]
    elif is2D:
        return newPoints[:,0], newPoints[:,1]

def filletCurves(radius: float, x1Points: np.ndarray, y1Points: np.ndarray, x2Points: np.ndarray, y2Points: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:

    '''

    Fillet two arbitrary curves using polynomial fits and a tangent-matching circle.

    Finds the intersection of two cubic-spline-interpolated curves, orients the problem
    along a bisector angle, fits cubic polynomials near the intersection, and solves for
    a circle of the specified radius that is tangent to both polynomial fits. The fillet
    arc replaces the intersection region and the result is transformed back to the original
    coordinate frame.

    Parameters:
    -----------
    radius : float
        Radius of curvature for the fillet arc.
    x1Points : np.ndarray
        X coordinates of the first curve.
    y1Points : np.ndarray
        Y coordinates of the first curve.
    x2Points : np.ndarray
        X coordinates of the second curve.
    y2Points : np.ndarray
        Y coordinates of the second curve.
    n : int
        Number of points to generate along the fillet arc.

    Returns:
    --------
    tuple[np.ndarray, np.ndarray] : (xOutput, yOutput) coordinates of the
        filleted curve combining both input curves with the fillet arc.

    '''

    # Local imports
    from scipy.interpolate import CubicSpline
    from scipy.optimize import newton, root

    # Find intersect location using splines
    splineCurve1 = CubicSpline(x1Points, y1Points)
    splineCurve2 = CubicSpline(x2Points, y2Points)

    def intersectFunction(x):
        return splineCurve1(x) - splineCurve2(x)

    xIntersection = newton(intersectFunction, 0.5)
    yIntersection = splineCurve1(xIntersection)

    # Select data sets such that x1 and x2 are in order and the intersect. Location is at the origin
    if x1Points[1] < xIntersection and x2Points[-1] > xIntersection:
        x1Points, x2Points, y1Points,  y2Points = x1Points - xIntersection, x2Points - xIntersection, \
                                                  y1Points - yIntersection, y2Points - yIntersection
        
    elif x2Points[1] < xIntersection and x1Points[-1] > xIntersection:
        xPointA, x1Points =                       x1Points - xIntersection, x2Points - xIntersection
        x2Points =                                xPointA
        yPointA, y1Points =                       y1Points - yIntersection, y2Points - yIntersection
        y2Points =                                yPointA
    
    # Find orientation of bisector and orient problem with the vertical bisector in order to simplify problem to one half of circle formula

    # Slope between endpoints
    slopeEndpoint1, slopeEndpoint2 =   (y1Points[-1] - y1Points[1]) / (x1Points[-1] - x1Points[1]), \
                                       (y2Points[-1] - y2Points[1]) / (x2Points[-1] - x2Points[1])
    # Slope at intersect
    slopeIntersect1, slopeIntersect2 = (y1Points[-1] - y1Points[-2]) / (x1Points[-1] - x1Points[-2]), \
                                       (y2Points[1] - y2Points[0]) / (x2Points[1] - x2Points[0])
    
    # Characteristic slopes (average of slopes)
    characteristicSlope1, characteristicSlope2 = (slopeIntersect1 + slopeEndpoint1) / 2,  \
                                                 (slopeIntersect2 + slopeEndpoint2) / 2
    
    # Find bisector angle from the characteristic angles of both curves 
    characteristicAngle1, characteristicAngle2 = np.arctan2(characteristicSlope1 * x1Points[-1], x1Points[-1]), np.arctan2(characteristicSlope2 * x2Points[-1], x2Points[-1])
    characteristicTotalAngle                   = (characteristicAngle1 + characteristicAngle2) / 2
    
    # Cosine transformation matrix
    transformationMatrix = [[np.cos(np.pi/2 - characteristicTotalAngle), -np.sin(np.pi/2 - characteristicTotalAngle)], \
                            [np.sin(np.pi / 2 - characteristicTotalAngle), np.cos(np.pi / 2 - characteristicTotalAngle)]]
    
    #Orient by cosine transformation matrix 
    for i in range(len(x1Points)):
        x1NewPoints = np.array(transformationMatrix) @ np.array([x1Points[i], y1Points[i]]).T
        x1Points[i] = x1NewPoints[0]
        y1Points[i] = x1NewPoints[1]
    
    for i in range (len(x2Points)):
        x2NewPoints = np.array(transformationMatrix) @ np.array([x2Points[i], y2Points[i]]).T
        x2Points[i] = x2NewPoints[0]
        y2Points[i] = x2NewPoints[1]

    # Trim curves to only segments near fillet
    fitFactor = 2 # number of radii considered in curve fit 
    x1Fit = x1Points[x1Points > -fitFactor * radius]
    y1Fit = y1Points[x1Points > -fitFactor * radius]
    x2Fit = x2Points[x2Points < fitFactor * radius]
    y2Fit = y2Points[x2Points < fitFactor * radius]

    # Generate 4th Order polynomial fit for either dataset
    polynomialFit1 = np.polyfit(x1Fit, y1Fit, 3)
    polynomialFit2 = np.polyfit(x2Fit, y2Fit, 3)

    # Build system of equations to solve
    # f1 - set curve 1 equal to fillet circle at intersect
    # f2 - set curve 2 equal to fillet circle at intersect
    # f3 - set curve 1 slope equal to slope of fillet circle at intersect
    # f4 - set curve 2 slope equal to slope of fillet circle at intersect

    def equations(x): 
        return [(polynomialFit1[0] * x[0]**3 + polynomialFit1[1]   * x[0]**2 + polynomialFit1[2] * x[0] + polynomialFit1[3] - (-np.sqrt(radius**2 - (x[0] - x[2])**2) + x[3])), 
                 polynomialFit2[0] * x[1]**3 + polynomialFit2[1]   * x[1]**2 + polynomialFit2[2] * x[1] + polynomialFit2[3] - (-np.sqrt(radius**2 - (x[1] - x[2])**2) + x[3]),
               3*polynomialFit1[0] * x[0]**1 + 2*polynomialFit1[1] * x[0]    + polynomialFit1[2] - (x[0] - x[2]) / np.sqrt(radius**2 - (x[0] - x[2])**2),
               3*polynomialFit2[0] * x[1]**1 + 2*polynomialFit2[1] * x[1]    + polynomialFit2[2] - (x[1] - x[2]) / np.sqrt(radius**2 - (x[1] - x[2])**2)]

    # Initial guess
    initialGuess = [np.mean(x1Fit), np.mean(x2Fit), 0, radius]

    # Solve system of equations
    systemOfEqSolution = root(equations, initialGuess)

    # Process outputs 
    xOutput1, xOutput2, x0, y0 = np.real(systemOfEqSolution.x[0]), np.real(systemOfEqSolution.x[1]), np.real(systemOfEqSolution.x[2]), \
                                 np.real(systemOfEqSolution.x[3])
    
    # Find angle from fillet origin
    angleFillet1 = np.arctan2((np.polyval(polynomialFit1, xOutput1) - y0), (xOutput1 - x0))
    angleFillet2 = np.arctan2((np.polyval(polynomialFit2, xOutput2) - y0), (xOutput2 - x0))

    # Generate x and y points by linear angular spacing
    angleFillet = np.linspace(angleFillet1, angleFillet2, n)
    xFillet = radius * np.cos(angleFillet) + x0
    yFillet = radius * np.sin(angleFillet) + y0

    # Concatenate Results 
    xOutput = np.concatenate([x1Points[x1Points < xOutput1], xFillet, x2Points[x2Points > xOutput2]])
    yOutput = np.concatenate([y1Points[x1Points < xOutput1], yFillet, y2Points[x2Points > xOutput2]])

    # Post-process Results
    # Cosine Transformation Matrix to reorient results
    cosineTransformationMatrix = [[np.cos(characteristicTotalAngle - np.pi/2), -np.sin(characteristicTotalAngle - np.pi/2)],\
                                  [np.sin(characteristicTotalAngle - np.pi/2), np.cos(characteristicTotalAngle - np.pi/2)]]
    
    # Reorient results
    for i in range(len(xOutput)):
       systemOfEqSolution = np.array(cosineTransformationMatrix) @ np.array([xOutput[i], yOutput[i]]).T
       xOutput[i] = systemOfEqSolution[0] # no se si es 1 o 0 me vuelvo lOCA - c
       yOutput[i] = systemOfEqSolution[1] 

    # Translate results to intial coordinates
    xOutput = xOutput + xIntersection
    yOutput = yOutput + yIntersection

    return xOutput, yOutput

def lineIntersection(point1: list[float, float], angle1: float, point2: list[float, float], angle2: float) -> list[float, float]:

    '''
    
    This function takes in, for two different lines, a point on each line and the angle that each 
    line makes with the X axis and calculates the intersection point between the two lines.
    
    '''

    slope1, slope2 = np.tan(angle1), np.tan(angle2)
    yIntercept1, yIntercept2 = point1[1] - slope1 * point1[0], point2[1] - slope2 * point2[0]

    xIntersection = (yIntercept2 - yIntercept1) / (slope1 - slope2)
    yIntersection = (slope1 * xIntersection + yIntercept1 + slope2 * xIntersection + yIntercept2) / 2

    return xIntersection, yIntersection

def revolveContour(xContour: list | np.ndarray, rContour: list | np.ndarray, numSlices: int = 50) -> np.ndarray:

    '''
    
    Given a 2D contour, revolve it about the central axis to create a surface.
    
    '''

    if isinstance(xContour, list):
        xContour  = np.array(xContour)
        rContour  = np.array(rContour)

    zContour       = np.zeros(len(xContour))
    beforeRevolve  = np.zeros((numSlices, 3, len(xContour)))
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

        # Apply that rotation matrix to the pre-rotated volute matrix
        afterRevolve[i,:,:] = rotationMatrix[i,:,:] @ beforeRevolve

        # Extract the X, Y, and Z data from the 3D rotated volute matrix
        xContourRevolved[i,:] = afterRevolve[i,0,:]
        yContourRevolved[i,:] = afterRevolve[i,1,:]
        zContourRevolved[i,:] = afterRevolve[i,2,:]

    return xContourRevolved, yContourRevolved, zContourRevolved

def jtan2(theta: np.ndarray | list, exact: bool = False) -> list[float]:

    '''

    Ported from jtan2.m (Sean Bowman [10/11/2022])

    This function fixes the discontinuities in atan2.m caused by the traditional range of arctangent.
        in :    janky angles        [1xn] radians
        out:    corrected angles    [1xn] radians
    Angles are corrected by counting the discontiuties with respect to sign and applying a correction
    factor of pi*(number of discontinuities) between discontinuity n and n+1 (or n and end in the
    case of the final discontinuity). If no discontinuities are found, theta is returned unchanged

    Isabella Duprey-Churn [7/2/2024]

    '''

    n = len(theta)

    # better method for finding discontinuities? {
    c = 1                 # correction accounts for the discrete nature of theta i.e. the magnitude of discntinuity may be < 2*pi
    d = np.pi-c             # discontinuity criteria

    disc = []               # store indicies of discontinuity
    gap = []
    for i in range(n-1):
        if abs(theta[i+1]-theta[i]) >= d:
            gap.append(theta[i+1]-theta[i])
            disc.append(i)
    m = len(disc)
    # }

    if not disc:
        # if no discontinuities found, return theta unchanged:
        thetaCorrected = theta.copy()
    else:
        # preallocate:
        thetaCorrected = theta.copy()
        signDiff = theta[0:m].copy()
        # initalize discontinuity counter:
        count = 0              
        # store sign of discontinuties:
        for i in range(m):
            signDiff[i] = np.sign(theta[disc[i]+1] - theta[disc[i]])
        for p in range(m):          # for all discontinuities
            count += signDiff[p]    # update counter for appropriate correction factor
            if p < m-1:             # not final discountinuity
                for q in range(disc[p]+1,disc[p+1]+1):
                    if not exact:
                        thetaCorrected[q] = theta[q] - (2*np.pi*count)    
                    else:
                        thetaCorrected[q] = theta[q] - (gap[p]*count)
            else:                   # final discontinuity
                for q in range(disc[p]+1,n):
                    if not exact:
                        thetaCorrected[q] = theta[q] - (2*np.pi*count)
                    else:
                        thetaCorrected[q] = theta[q] - (gap[p]*count)

    return thetaCorrected

def DCM(eulerAngles: list[float, float, float], valueMatrix: list[list[list]], transpose: bool = False, rotationOrder: str = 'zyx') -> np.ndarray:

    '''
    
    Apply direction cosine matrix transformations to 'valueMatrix' given the angles in 'eulerAngles' and the order specified by 'rotationOrder'.
    
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
    if not transpose:
        match rotationOrder:
            case 'zyx':
                # Default operation order
                rotationMatrix = Rz @ Ry @ Rx
            case 'yzx':
                rotationMatrix = Ry @ Rz @ Rx
            case 'xyz':
                rotationMatrix = Rx @ Ry @ Rz
    elif transpose:
        match rotationOrder:
            case 'zyx':
                rotationMatrix = (Rz @ Ry @ Rx).T
            case 'yzx':
                rotationMatrix = (Ry @ Rz @ Rx).T
            case 'xyz':
                rotationMatrix = (Rx @ Ry @ Rz).T

    # Apply that rotation matrix to the pre-rotated volute matrix
    unCenteredRotatedValueMatrix = rotationMatrix @ valueMatrix

    # Extract the X, Y, and Z data from the 3D rotated volute matrix
    rotatedX = unCenteredRotatedValueMatrix[0]
    rotatedY = unCenteredRotatedValueMatrix[1]
    rotatedZ = unCenteredRotatedValueMatrix[2]

    return rotatedX, rotatedY, rotatedZ

def rescaleData(rawData: np.ndarray | list, newMax: float = None, newMin: float = None) -> np.ndarray | list:

    '''
    
    Rescale data between a new specified max and min while retaining data shape.
    
    '''

    # Handle cases where the user wants only to rescale in one direction
    if not newMax:

        newMax = np.max(rawData)

    if not newMin:

        newMin = np.min(rawData)

    # First, normalize to [0, 1]
    normalizedData = (rawData - np.min(rawData)) / (np.max(rawData) - np.min(rawData))
    
    # Then scale to [new_min, new_max]
    rescaledData = normalizedData * (newMax - newMin) + newMin
    
    return rescaledData

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
        Larger values produce a fuller curve; typical range 0.2–0.6.
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
    # Bug fix: use cos/sin directly: the previous sqrt approach forced the x-component
    # to always be non-negative, mirroring control points for angles > 90° or < -90°.
    p2 = [p1[0] + mag1 * np.cos(np.radians(theta_1)),
          p1[1] + mag1 * np.sin(np.radians(theta_1))]

    p3 = [p4[0] - mag2 * np.cos(np.radians(theta_2)),
          p4[1] - mag2 * np.sin(np.radians(theta_2))]

    # Cubic Bezier parametric evaluation using the correct Bernstein basis.
    # Bug fix: original code used (1 - t**3) for the first coefficient instead of
    # the correct (1-t)**3, which over-weighted p1 through most of the parameter range.
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
# -- File I/O Tools -- #
#----------------------------------------------------------------------#

def writeFile(filename: str, data: np.ndarray | list, headers: bool = False) -> None:

    '''

    Write array data to a .csv or .txt file.

    Parameters:
    -----------
    filename : str
        Output file path including extension (.csv or .txt).
    data : np.ndarray | list
        Data matrix of shape (n, m) to write.
    headers : bool, optional
        If True, write default X/Y/Z headers for CSV files (default False).

    Returns:
    --------
    None : A file is created at the specified path.

    '''

    import csv

    if '.' not in filename:
        raise ValueError('Filename must include a file extension (.csv or .txt).')

    if isinstance(data, list):
        data = np.array(data)

    whichType = filename[-4:]

    match whichType:

        case '.csv':

            with open(filename, 'w', newline='') as csvFile:
                writer = csv.writer(csvFile)
                if headers:
                    writer.writerow(['X', 'Y', 'Z'])
                writer.writerows(data)

        case '.txt':

            with open(filename, 'w') as txtFile:
                for i, _ in enumerate(data[:, 0]):
                    txtFile.write('\t'.join([str(val) for val in list(data[i, :])]) + '\n')

#----------------------------------------------------------------------#
# -- Plotly Visualization Tools -- #
#----------------------------------------------------------------------#

def plotly3DGeometry(xData: np.ndarray | list, yData: np.ndarray | list, zData: np.ndarray | list,
                     title: str = 'Geometry Plot', xLabel: str = 'X', yLabel: str = 'Y', zLabel: str = 'Z',
                     alpha: float = 1, color: str = 'lightslategray') -> None:

    '''

    Plot a 3D surface geometry with uniform color using Plotly.

    Parameters:
    -----------
    xData, yData, zData : np.ndarray | list
        Surface mesh coordinates.
    title : str
        Plot title (default 'Geometry Plot').
    xLabel, yLabel, zLabel : str
        Axis labels.
    alpha : float
        Surface opacity, 0 to 1 (default 1).
    color : str
        Surface color (default 'lightslategray').

    Returns:
    --------
    None : Renders an interactive Plotly figure.

    '''

    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Surface(x=xData, y=yData, z=zData,
                             colorscale=[[0, color], [1, color]],
                             opacity=alpha,
                             showscale=False))

    fig.update_layout(scene=dict(xaxis_title=xLabel,
                                 yaxis_title=yLabel,
                                 zaxis_title=zLabel),
                      title={'text': title, 'x': 0.5, 'xanchor': 'center',
                             'y': 0.9, 'yanchor': 'top'},
                      scene_aspectmode='data',
                      template='plotly_dark')

    fig.show()

def plotlySurface(xData: np.ndarray | list, yData: np.ndarray | list, zData: np.ndarray | list,
                  colorMap: str = 'Plasma', alpha: float = 1, title: str = 'Surface Plot',
                  xLabel: str = 'X', yLabel: str = 'Y', zLabel: str = 'Z') -> None:

    '''

    Plot a 3D surface with a colormap using Plotly.

    Parameters:
    -----------
    xData, yData, zData : np.ndarray | list
        Surface mesh coordinates.
    colorMap : str
        Plotly colorscale name (default 'Plasma').
    alpha : float
        Surface opacity, 0 to 1 (default 1).
    title : str
        Plot title (default 'Surface Plot').
    xLabel, yLabel, zLabel : str
        Axis labels.

    Returns:
    --------
    None : Renders an interactive Plotly figure.

    '''

    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Surface(x=xData, y=yData, z=zData,
                             colorscale=colorMap,
                             opacity=alpha))

    fig.update_layout(scene=dict(xaxis_title=xLabel,
                                 yaxis_title=yLabel,
                                 zaxis_title=zLabel),
                      title={'text': title, 'x': 0.5, 'xanchor': 'center',
                             'y': 0.9, 'yanchor': 'top'},
                      scene_aspectmode='data',
                      template='plotly_dark')

    fig.show()

def plotly3DLine(xData: np.ndarray | list, yData: np.ndarray | list, zData: np.ndarray | list,
                 title: str = '3D Line Plot', xLabel: str = 'X', yLabel: str = 'Y', zLabel: str = 'Z',
                 color: str | list = 'cyan', lineWidth: float = 1, lineStyle: str = None,
                 markerStyle: str = None, markerSize: float = 4, label: str = '3D Line') -> None:

    '''

    Plot a 3D line/scatter using Plotly.

    Parameters:
    -----------
    xData, yData, zData : np.ndarray | list
        Line coordinates.
    title : str
        Plot title.
    xLabel, yLabel, zLabel : str
        Axis labels.
    color : str | list
        Line/marker color.
    lineWidth : float
        Line width (default 1).
    lineStyle : str
        Dash style ('dash', 'dot', 'dashdot', or None for solid).
    markerStyle : str
        Plotly marker symbol (default None).
    markerSize : float
        Marker size (default 4).
    label : str
        Trace name for the legend.

    Returns:
    --------
    None : Renders an interactive Plotly figure.

    '''

    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Scatter3d(x=xData, y=yData, z=zData,
                                       mode='lines+markers',
                                       marker=dict(symbol=markerStyle, size=markerSize, color=color),
                                       line=dict(dash=lineStyle, width=lineWidth, color=color),
                                       name=label)])

    fig.update_layout(scene=dict(xaxis_title=xLabel,
                                 yaxis_title=yLabel,
                                 zaxis_title=zLabel),
                      title={'text': title, 'x': 0.5, 'xanchor': 'center',
                             'y': 0.9, 'yanchor': 'top'},
                      scene_aspectmode='data',
                      template='plotly_dark')

    fig.show()

def plotly2DLine(xData: np.ndarray | list, yData: np.ndarray | list,
                 title: str = '2D Line Plot', xLabel: str = 'X', yLabel: str = 'Y',
                 color: str = 'cyan', lineWidth: float = 1, lineStyle: str = None,
                 markerStyle: str = None, markerSize: float = 4, fontSize: int = 12,
                 label: str = '2D Line') -> None:

    '''

    Plot a 2D line/scatter using Plotly.

    Parameters:
    -----------
    xData, yData : np.ndarray | list
        Line coordinates.
    title : str
        Plot title.
    xLabel, yLabel : str
        Axis labels.
    color : str
        Line/marker color (default 'cyan').
    lineWidth : float
        Line width (default 1).
    lineStyle : str
        Dash style ('dash', 'dot', 'dashdot', or None for solid).
    markerStyle : str
        Plotly marker symbol (default None).
    markerSize : float
        Marker size (default 4).
    fontSize : int
        Font size (default 12).
    label : str
        Trace name for the legend.

    Returns:
    --------
    None : Renders an interactive Plotly figure.

    '''

    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Scatter(x=xData, y=yData,
                                     mode='lines+markers',
                                     marker=dict(symbol=markerStyle, size=markerSize, color=color),
                                     line=dict(dash=lineStyle, width=lineWidth, color=color),
                                     name=label)])

    fig.update_layout(xaxis_title=xLabel,
                      yaxis_title=yLabel,
                      title={'text': title, 'x': 0.5, 'xanchor': 'center',
                             'y': 0.9, 'yanchor': 'top'},
                      template='plotly_dark',
                      font=dict(size=fontSize))

    fig.show()
