
# -- pySurf: Bathymetry -- #

'''

The sea bed: a bottom contour z_b(x) in 2D, or a surface z_b(x, y) in 3D, that water flows over
and a wave breaks on.

A `Bathymetry` is a piecewise-linear profile through (x, z) vertices, with x along the direction
of wave travel (offshore to onshore) and z up. Everything a wave model asks of the bed comes from
that one polyline: the bed elevation and still water depth at any x, the slope, where the still
water line meets the beach, the outward unit normal a boundary is built along, and the closed
polygon a figure fills as sand.

`Bathymetry.planeBeach` builds the profile most breaking studies use: a flat bottom offshore, then
a constant slope rising through the still water line.

A `HeightField` is a bed in 3D: z_b on a regular (x, y) grid, interpolated by cubic convolution
(Keys 1981), with its gradient and normal. `HeightField.fromFunction` samples an analytic bed onto
a grid, and `HeightField.ridge` builds the sloping ridge of Guyenne and Grilli (2006). A section at
fixed y is a `Bathymetry`, for the 2D measures.

References:
-----------
Guyenne, P. & Grilli, S.T. (2006) -- Numerical study of three-dimensional overturning waves in
    shallow water, J. Fluid Mech. 547, 361-388.
Keys, R.G. (1981) -- Cubic convolution interpolation for digital image processing, IEEE Trans.
    Acoustics, Speech, and Signal Processing 29(6), 1153-1160.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

Nothing here is a model: the bed is geometry given as input, and the queries are interpolation
along it. tests/testBathymetry.py checks each profile query against the closed form for a plane
beach (elevation, depth, shoreline position, slope, normal direction), and the height field's
interpolation against a quadratic surface, which cubic convolution reproduces exactly, its gradient
against the quadratic's, and its values at the grid points against the sampled function.

Sean Bowman [10/11/2026]

'''

# Standard library imports
from __future__ import annotations

import math
from typing import Callable

# Third-party imports
import numpy as np

# Local imports
from pysurf.errors import InvalidInputError

class Bathymetry:

    '''

    A 2D bed profile, piecewise linear through its vertices.

    Parameters:
    -----------
    x : array-like
        Vertex positions along the direction of wave travel, strictly increasing [m]
    z : array-like
        Bed elevation at each vertex [m]
    stillWaterLevel : float
        Elevation of the still water surface [m]

    '''

    def __init__(self, x, z, stillWaterLevel: float) -> None:

        self.x = np.asarray(x, dtype = float)
        self.z = np.asarray(z, dtype = float)
        self.stillWaterLevel = float(stillWaterLevel)    # [m]
        self._validateInputs()

    # -- Private Helper Methods -- #
    def _validateInputs(self) -> None:

        '''

        The vertices must form a function of x: at least two, matched in length, x strictly
        increasing, all finite.

        Raises:
        -------
        InvalidInputError : If any of those fails.

        '''

        if self.x.ndim != 1 or self.x.size < 2 or self.z.shape != self.x.shape:
            raise InvalidInputError('A bed profile needs at least two (x, z) vertices of matching length.',
                                    parameterName = 'x', value = (self.x.size, self.z.size))
        if not (np.all(np.isfinite(self.x)) and np.all(np.isfinite(self.z))):
            raise InvalidInputError('Bed vertices must be finite.', parameterName = 'z')
        if np.any(np.diff(self.x) <= 0.0):
            raise InvalidInputError('Bed vertex x must increase strictly, so the bed is a function of x.',
                                    parameterName = 'x')

    # -- Constructors -- #
    @classmethod
    def planeBeach(cls, depth: float, toeX: float, slope: float, endX: float, startX: float = 0.0) -> Bathymetry:

        '''

        A flat bottom at z = 0 from startX to the toe of the beach, then a constant slope to endX.

        The still water level sits at z = depth, so the water is `depth` deep offshore and the
        still water line meets the beach at toeX + depth / slope.

        Parameters:
        -----------
        depth : float
            Still water depth over the flat bottom [m]
        toeX : float
            Position of the toe of the slope [m]
        slope : float
            Beach slope tan(beta) [-], e.g. 1/35
        endX : float
            Landward end of the profile [m]
        startX : float
            Offshore end of the profile [m]

        '''

        if not (startX < toeX < endX):
            raise InvalidInputError('A plane beach needs startX < toeX < endX.', parameterName = 'toeX', value = toeX)
        if slope <= 0.0 or depth <= 0.0:
            raise InvalidInputError('Depth and slope must be positive.', parameterName = 'slope', value = slope)

        return cls([startX, toeX, endX], [0.0, 0.0, (endX - toeX) * slope], stillWaterLevel = depth)

    @classmethod
    def reefBreak(cls, depth: float, toeX: float, faceSlope: float, reefDepth: float, reefLength: float,
                  lagoonDepth: float | None = None, lagoonLength: float = 0.0, beachSlope: float | None = None,
                  beachHeight: float = 0.0) -> Bathymetry:

        '''

        A reef break in profile: a flat bottom offshore, a steep reef face rising to a shallow reef
        flat, and behind it optionally a lagoon and a beach.

        The still water level sits at z = depth. Without a lagoon or beach the profile ends at the
        back of the reef flat, where a model closes it with a wall. With a lagoon deeper than the
        reef flat, the back of the reef drops at the face slope to the lagoon floor; a beach then
        rises from the lagoon floor at `beachSlope` to `beachHeight` above still water.

        Parameters:
        -----------
        depth : float
            Still water depth offshore [m]
        toeX : float
            Position of the toe of the reef face [m]
        faceSlope : float
            Slope of the reef face, tan(beta) [-]; steep for a reef, e.g. 1/5
        reefDepth : float
            Still water depth over the reef flat [m]
        reefLength : float
            Length of the reef flat [m]
        lagoonDepth : float | None
            Still water depth in the lagoon behind the reef [m]; None for no lagoon
        lagoonLength : float
            Length of the lagoon floor [m]
        beachSlope : float | None
            Slope of the beach behind the lagoon [-]; None for no beach
        beachHeight : float
            Height of the beach top above still water [m]

        '''

        if not (0.0 < reefDepth < depth) or faceSlope <= 0.0 or reefLength <= 0.0 or toeX <= 0.0:
            raise InvalidInputError('A reef needs 0 < reefDepth < depth, a positive face slope, reef length and toe.',
                                    parameterName = 'reefDepth', value = reefDepth)

        top = depth - reefDepth
        x = [0.0, toeX, toeX + top / faceSlope]
        z = [0.0, 0.0, top]
        x.append(x[-1] + reefLength)
        z.append(top)
        floor = top
        if lagoonDepth is not None and lagoonDepth > reefDepth:
            floor = depth - lagoonDepth
            x.append(x[-1] + (lagoonDepth - reefDepth) / faceSlope)
            z.append(floor)
            if lagoonLength > 0.0:
                x.append(x[-1] + lagoonLength)
                z.append(floor)
        if beachSlope is not None:
            if beachSlope <= 0.0:
                raise InvalidInputError('The beach slope must be positive.', parameterName = 'beachSlope', value = beachSlope)
            rise = depth + beachHeight - floor
            x.append(x[-1] + rise / beachSlope)
            z.append(depth + beachHeight)

        return cls(x, z, stillWaterLevel = depth)

    # -- Public Methods -- #
    @property
    def startX(self) -> float:
        '''Offshore end of the profile [m].'''
        return float(self.x[0])

    @property
    def endX(self) -> float:
        '''Landward end of the profile [m].'''
        return float(self.x[-1])

    def elevation(self, x) -> np.ndarray:

        '''Bed elevation z_b(x) [m], held constant beyond the ends of the profile.'''

        return np.interp(x, self.x, self.z)

    def depth(self, x) -> np.ndarray:

        '''Still water depth max(0, SWL - z_b(x)) [m]: zero on the dry beach.'''

        return np.maximum(0.0, self.stillWaterLevel - self.elevation(x))

    def slope(self, x) -> np.ndarray:

        '''Bed slope dz_b/dx of the segment containing x [-].'''

        segmentSlopes = np.diff(self.z) / np.diff(self.x)
        index = np.clip(np.searchsorted(self.x, x, side = 'right') - 1, 0, segmentSlopes.size - 1)

        return segmentSlopes[index]

    def normal(self, x) -> np.ndarray:

        '''Unit normal pointing up out of the bed into the water, shape (..., 2).'''

        slope = np.asarray(self.slope(x), dtype = float)
        length = np.sqrt(1.0 + slope**2)

        return np.stack([-slope / length, 1.0 / length], axis = -1)

    def shorelineX(self) -> float:

        '''

        Where the still water line meets the bed, scanning landward [m].

        Returns the first crossing of SWL by the bed from below. Raises when the bed never
        rises through the still water line.

        '''

        relative = self.z - self.stillWaterLevel
        crossings = np.nonzero((relative[:-1] < 0.0) & (relative[1:] >= 0.0))[0]
        if crossings.size == 0:
            raise InvalidInputError('The bed never rises through the still water line.',
                                    parameterName = 'stillWaterLevel', value = self.stillWaterLevel)
        i = int(crossings[0])

        return float(self.x[i] + (self.stillWaterLevel - self.z[i]) * (self.x[i + 1] - self.x[i])
                     / (self.z[i + 1] - self.z[i]))

    def sample(self, spacing: float) -> tuple[np.ndarray, np.ndarray]:

        '''

        Points along the bed at equal arc-length spacing, including both ends.

        Parameters:
        -----------
        spacing : float
            Target distance between consecutive points along the bed [m]

        Returns:
        --------
        tuple : (x, z) arrays

        '''

        segmentLengths = np.hypot(np.diff(self.x), np.diff(self.z))
        arcLength = np.concatenate([[0.0], np.cumsum(segmentLengths)])
        count = max(2, int(round(arcLength[-1] / spacing)) + 1)
        targets = np.linspace(0.0, arcLength[-1], count)

        return np.interp(targets, arcLength, self.x), np.interp(targets, arcLength, self.z)

    def sandPolygon(self, floor: float) -> tuple[np.ndarray, np.ndarray]:

        '''

        The closed outline of the sand between the bed and a floor elevation below it, for a
        figure to fill.

        '''

        x = np.concatenate([self.x, [self.endX, self.startX]])
        z = np.concatenate([self.z, [floor, floor]])

        return x, z

def convolutionWeights(u: np.ndarray) -> np.ndarray:

    '''

    Cubic convolution's weights (Keys 1981, a = -1/2) on the four grid points around a query a
    fraction u past the second of them, shape (4, ...). They sum to one, and reproduce any
    quadratic exactly.

    '''

    u = np.asarray(u, dtype = float)
    u2, u3 = u * u, u * u * u

    return np.stack([-0.5 * u3 + u2 - 0.5 * u, 1.5 * u3 - 2.5 * u2 + 1.0, -1.5 * u3 + 2.0 * u2 + 0.5 * u,
                     0.5 * u3 - 0.5 * u2])

def convolutionSlopeWeights(u: np.ndarray) -> np.ndarray:

    '''The derivatives of `convolutionWeights` with respect to u, shape (4, ...).'''

    u = np.asarray(u, dtype = float)
    u2 = u * u

    return np.stack([-1.5 * u2 + 2.0 * u - 0.5, 4.5 * u2 - 5.0 * u, -4.5 * u2 + 4.0 * u + 0.5, 1.5 * u2 - u])

class HeightField:

    '''

    A bed in 3D: the elevation z_b(x, y) on a regular grid, interpolated by cubic convolution.

    Cubic convolution (Keys 1981, a = -1/2) builds each value from the 4 by 4 grid points around
    it, so the surface and its gradient are continuous, any quadratic is reproduced exactly, and a
    query needs only its neighborhood: the GPU evaluates the same formula (`pysurf.sphGpu`). A query
    beyond the grid takes the value at the nearest point of its edge, and a zero gradient across
    that edge. The grid usually reaches two cells past the region the bed covers (`bounds`), so
    the interpolation inside that region has a full stencil everywhere.

    x runs along the direction of wave travel and y across it; z is up.

    Parameters:
    -----------
    x : array-like
        Grid positions along x, equally spaced and increasing [m]
    y : array-like
        Grid positions along y, equally spaced and increasing [m]
    z : array-like
        Bed elevation at each grid point, shape (len(y), len(x)) [m]
    stillWaterLevel : float
        Elevation of the still water surface [m]
    bounds : tuple | None
        (startX, endX, startY, endY), the region the bed covers, inside the grid [m]; None for the
        whole grid

    '''

    def __init__(self, x, y, z, stillWaterLevel: float, bounds: tuple | None = None) -> None:

        self.x = np.asarray(x, dtype = float)
        self.y = np.asarray(y, dtype = float)
        self.z = np.asarray(z, dtype = float)
        self.stillWaterLevel = float(stillWaterLevel)       # [m]
        if bounds is None:
            bounds = (self.x[0], self.x[-1], self.y[0], self.y[-1]) if self.x.size and self.y.size else (0.0,) * 4
        self.bounds = tuple(float(value) for value in bounds)  # [m]
        self._validateInputs()

    # -- Private Helper Methods -- #
    def _validateInputs(self) -> None:

        '''

        The grid must be regular: at least four points on each axis, equally spaced and
        increasing, z shaped (len(y), len(x)), every value finite, and the bounds inside it.

        Raises:
        -------
        InvalidInputError : If any of those fails.

        '''

        if self.x.ndim != 1 or self.y.ndim != 1 or self.x.size < 4 or self.y.size < 4:
            raise InvalidInputError('A height field needs at least four grid points along x and y.',
                                    parameterName = 'x', value = (self.x.size, self.y.size))
        if self.z.shape != (self.y.size, self.x.size):
            raise InvalidInputError('The elevations must be shaped (len(y), len(x)).', parameterName = 'z',
                                    value = self.z.shape)
        if not (np.all(np.isfinite(self.x)) and np.all(np.isfinite(self.y)) and np.all(np.isfinite(self.z))):
            raise InvalidInputError('Height field values must be finite.', parameterName = 'z')
        for name, axis in (('x', self.x), ('y', self.y)):
            steps = np.diff(axis)
            if np.any(steps <= 0.0) or not np.allclose(steps, steps[0], rtol = 1e-9, atol = 0.0):
                raise InvalidInputError(f'Grid {name} must increase in equal steps.', parameterName = name)
        startX, endX, startY, endY = self.bounds
        inside = (self.x[0] - 1e-9 <= startX < endX <= self.x[-1] + 1e-9
                  and self.y[0] - 1e-9 <= startY < endY <= self.y[-1] + 1e-9)
        if not inside:
            raise InvalidInputError('The bounds must lie inside the grid.', parameterName = 'bounds', value = self.bounds)

    def _interpolate(self, x, y, derivative: str = '') -> np.ndarray:

        '''

        Cubic convolution of the grid at (x, y): the value, or with `derivative` 'x' or 'y' that
        partial derivative.

        '''

        x, y = np.broadcast_arrays(np.asarray(x, dtype = float), np.asarray(y, dtype = float))
        nx, ny = self.x.size, self.y.size
        rawX = (x - self.x[0]) / self.spacingX
        rawY = (y - self.y[0]) / self.spacingY
        fx, fy = np.clip(rawX, 0.0, nx - 1.0), np.clip(rawY, 0.0, ny - 1.0)
        i, j = np.floor(fx).astype(np.intp), np.floor(fy).astype(np.intp)
        u, v = fx - i, fy - j
        wx = convolutionSlopeWeights(u) / self.spacingX if derivative == 'x' else convolutionWeights(u)
        wy = convolutionSlopeWeights(v) / self.spacingY if derivative == 'y' else convolutionWeights(v)

        result = np.zeros(x.shape)
        for a in range(4):
            row = np.clip(j - 1 + a, 0, ny - 1)
            along = np.zeros(x.shape)
            for b in range(4):
                along += wx[b] * self.z[row, np.clip(i - 1 + b, 0, nx - 1)]
            result += wy[a] * along

        # Past an edge the surface is constant across it
        if derivative == 'x':
            result[(rawX < 0.0) | (rawX > nx - 1.0)] = 0.0
        elif derivative == 'y':
            result[(rawY < 0.0) | (rawY > ny - 1.0)] = 0.0

        return result

    # -- Constructors -- #
    @classmethod
    def fromFunction(cls, function: Callable[[np.ndarray, np.ndarray], np.ndarray], bounds: tuple,
                     gridSpacing: float, stillWaterLevel: float, padding: int = 2) -> HeightField:

        '''

        An analytic bed sampled onto a grid over its bounds and `padding` cells beyond them.

        Parameters:
        -----------
        function : Callable
            z_b(x, y) [m], taking arrays
        bounds : tuple
            (startX, endX, startY, endY), the region the bed covers [m]
        gridSpacing : float
            Target grid step [m]; each axis takes the step that divides its bounds evenly
        stillWaterLevel : float
            Elevation of still water [m]
        padding : int
            Grid cells beyond the bounds on every side

        '''

        startX, endX, startY, endY = (float(value) for value in bounds)
        if not (endX > startX and endY > startY and gridSpacing > 0.0):
            raise InvalidInputError('A height field needs positive extents and grid spacing.', parameterName = 'bounds',
                                    value = bounds)
        axes = []
        for low, high in ((startX, endX), (startY, endY)):
            cells = max(1, int(math.ceil((high - low) / gridSpacing - 1e-9)))
            step = (high - low) / cells
            axes.append(low + step * np.arange(-padding, cells + padding + 1))
        gridX, gridY = np.meshgrid(*axes, indexing = 'xy')

        return cls(axes[0], axes[1], np.asarray(function(gridX, gridY), dtype = float), stillWaterLevel,
                   bounds = (startX, endX, startY, endY))

    @classmethod
    def ridge(cls, k: float, depth: float = 1.0, toeX: float = 5.225, slope: float = 1.0 / 15.0, endX: float = 19.0,
              halfWidth: float = 4.0, startX: float = 0.0, symmetric: bool = False,
              gridSpacing: float | None = None) -> HeightField:

        '''

        The sloping ridge of Guyenne and Grilli (2006), in units of the depth h0:

            z_b / h0 = slope (x - toeX) sech^2(k y)   for x > toeX, flat offshore,

        with still water at z = depth and the tank's wall at endX, across -halfWidth <= y <= halfWidth;
        with `symmetric`, only the half 0 <= y <= halfWidth, whose mirror image is the other.

        Parameters:
        -----------
        k : float
            Spanwise wavenumber of the ridge [1/h0]
        depth : float
            Still water depth offshore, h0 [m]
        toeX, slope, endX, halfWidth, startX : float
            The ridge's toe [h0], its centre-line slope [-], the tank's end, half width and start [h0]
        symmetric : bool
            Cover half the tank, from the plane of symmetry y = 0
        gridSpacing : float | None
            Grid step [m]; None for depth / 64

        '''

        def elevation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return depth * slope * np.maximum(x / depth - toeX, 0.0) / np.cosh(k * y / depth)**2

        bounds = (startX * depth, endX * depth, 0.0 if symmetric else -halfWidth * depth, halfWidth * depth)

        return cls.fromFunction(elevation, bounds, gridSpacing if gridSpacing is not None else depth / 64.0,
                                stillWaterLevel = depth)

    # -- Public Methods -- #
    @property
    def spacingX(self) -> float:
        '''Grid step along x [m].'''
        return float(self.x[1] - self.x[0])

    @property
    def spacingY(self) -> float:
        '''Grid step along y [m].'''
        return float(self.y[1] - self.y[0])

    @property
    def startX(self) -> float:
        '''Offshore end of the region the bed covers [m].'''
        return self.bounds[0]

    @property
    def endX(self) -> float:
        '''Shoreward end of the region [m].'''
        return self.bounds[1]

    @property
    def startY(self) -> float:
        '''Low side of the region across the direction of travel [m].'''
        return self.bounds[2]

    @property
    def endY(self) -> float:
        '''High side of the region [m].'''
        return self.bounds[3]

    def elevation(self, x, y) -> np.ndarray:

        '''Bed elevation z_b(x, y) [m].'''

        return self._interpolate(x, y)

    def gradient(self, x, y) -> np.ndarray:

        '''The bed's gradient (dz_b/dx, dz_b/dy) [-], shape (..., 2).'''

        return np.stack([self._interpolate(x, y, 'x'), self._interpolate(x, y, 'y')], axis = -1)

    def normal(self, x, y) -> np.ndarray:

        '''Unit normal pointing up out of the bed into the water, shape (..., 3).'''

        gradient = self.gradient(x, y)
        normal = np.concatenate([-gradient, np.ones(gradient.shape[:-1] + (1,))], axis = -1)

        return normal / np.linalg.norm(normal, axis = -1, keepdims = True)

    def depth(self, x, y) -> np.ndarray:

        '''Still water depth max(0, SWL - z_b(x, y)) [m].'''

        return np.maximum(0.0, self.stillWaterLevel - self.elevation(x, y))

    def lowestElevation(self) -> float:

        '''The lowest bed elevation at the grid points inside the bounds [m].'''

        startX, endX, startY, endY = self.bounds
        inX = (self.x >= startX - 1e-9) & (self.x <= endX + 1e-9)
        inY = (self.y >= startY - 1e-9) & (self.y <= endY + 1e-9)

        return float(np.min(self.z[np.ix_(inY, inX)]))

    def section(self, y: float) -> Bathymetry:

        '''

        The bed's profile along x at fixed y, over the bounds, as a `Bathymetry` through the grid's
        x positions and the bounds' two ends.

        '''

        inside = self.x[(self.x > self.startX + 1e-9) & (self.x < self.endX - 1e-9)]
        x = np.concatenate([[self.startX], inside, [self.endX]])

        return Bathymetry(x, self.elevation(x, np.full_like(x, float(y))), self.stillWaterLevel)
