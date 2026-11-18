
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
a grid, `HeightField.ridge` builds the sloping ridge of Guyenne and Grilli (2006), and
`HeightField.fromGridFile` reads a text grid of x, y and z (or depth) rows. `smoothed` passes a
bed through a Gaussian, so features shorter than the particles can hold drop out, `cropped`
keeps the region a wave crosses, `rotated` turns a bed so a wave arriving at an angle travels
along +x, and `shoalingStart` finds where the bed first rises under the wave. A section at fixed y
is a `Bathymetry`, for the 2D measures.

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
against the quadratic's, and its values at the grid points against the sampled function. The
smoothing is checked against the Gaussian's transform: a sinusoid keeps exp(-k^2 sigma^2 / 2) of
its amplitude to 4×10⁻⁵, the weight dropped beyond four sigma, uniformly to the edges; and a plane
passes unchanged. A crop reads its parent's elevation and gradient to rounding, and a text grid
written from a field, its rows shuffled, reads back to it. A bed of second degree turned through
any angle matches the closed form to rounding, and the shoaling start falls on the grid column
where a plane beach first rises past the share.

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

def gaussianAlong(values: np.ndarray, sigma: float, step: float, axis: int) -> np.ndarray:

    '''

    A Gaussian of standard deviation `sigma` applied along one axis of a grid of spacing `step`:
    the weights sampled at the grid points to four sigma and summed to one. The ends are padded by
    odd reflection, 2 v_end - v_mirror, which continues a straight line exactly.

    Raises:
    -------
    InvalidInputError : If four sigma reaches past the far end of the axis.

    '''

    reach = int(math.ceil(4.0 * sigma / step))
    values = np.moveaxis(np.asarray(values, dtype = float), axis, -1)
    count = values.shape[-1]
    if reach > count - 1:
        raise InvalidInputError(f'A smoothing length of {sigma:g} reaches {reach} grid steps, past the grid of '
                                f'{count}.', parameterName = 'sigma', value = sigma)
    offsets = np.arange(-reach, reach + 1)
    weights = np.exp(-0.5 * (offsets * step / sigma)**2)
    weights /= weights.sum()

    index = np.arange(-reach, count + reach)
    mirror = np.where(index < 0, -index, np.where(index > count - 1, 2 * (count - 1) - index, index))
    padded = values[..., mirror]
    padded = np.where(index < 0, 2.0 * values[..., :1] - padded, padded)
    padded = np.where(index > count - 1, 2.0 * values[..., -1:] - padded, padded)
    result = np.zeros_like(values)
    for offset, weight in enumerate(weights):
        result += weight * padded[..., offset:offset + count]

    return np.moveaxis(result, -1, axis)

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

    @classmethod
    def fromGridFile(cls, path, stillWaterLevel: float, bounds: tuple | None = None,
                     depths: bool = False) -> HeightField:

        '''

        A bed read from a text grid: one point a row, its x, y and z separated by spaces, tabs or
        commas, the rows in any order, '#' starting a comment. The points must fill a regular grid,
        each grid point once.

        Parameters:
        -----------
        path : str | Path
            The text grid
        stillWaterLevel : float
            Elevation of still water [m]
        bounds : tuple | None
            (startX, endX, startY, endY), the region the bed covers [m]; None for the whole grid
        depths : bool
            The third column holds depths below still water, positive down, rather than
            elevations; each becomes stillWaterLevel - depth

        Raises:
        -------
        InvalidInputError : If a row does not hold three numbers, a value is not finite, a grid point
            repeats or is missing, or the grid is not regular.

        '''

        with open(path, encoding = 'utf-8') as handle:
            rows = [line.split('#', 1)[0].replace(',', ' ').split() for line in handle]
        rows = [row for row in rows if row]
        try:
            points = np.array(rows, dtype = float)
        except ValueError as error:
            raise InvalidInputError('Each row of a text grid must hold three numbers: x, y and z.',
                                    parameterName = 'path', value = str(path)) from error
        if points.ndim != 2 or points.shape[1] != 3:
            raise InvalidInputError('Each row of a text grid must hold three numbers: x, y and z.',
                                    parameterName = 'path', value = str(path))
        if not np.all(np.isfinite(points)):
            raise InvalidInputError('Text grid values must be finite.', parameterName = 'path', value = str(path))

        # Coordinates written with rounding noise still name one grid line each
        x, y = np.round(points[:, 0], 9), np.round(points[:, 1], 9)
        gridX, gridY = np.unique(x), np.unique(y)
        if gridX.size * gridY.size != len(points):
            raise InvalidInputError(f'The text grid holds {len(points)} points, not the {gridX.size} by {gridY.size} '
                                    'its x and y values span.', parameterName = 'path', value = str(path))
        z = np.full((gridY.size, gridX.size), np.nan)
        z[np.searchsorted(gridY, y), np.searchsorted(gridX, x)] = points[:, 2]
        # With the count right, a repeated point leaves another empty
        if np.isnan(z).any():
            raise InvalidInputError('A grid point repeats in the text grid, and another is missing.',
                                    parameterName = 'path', value = str(path))
        if depths:
            z = float(stillWaterLevel) - z

        return cls(gridX, gridY, z, stillWaterLevel, bounds = bounds)

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

    def shoalingStart(self, depth: float | None = None, share: float = 0.05) -> float | None:

        '''

        Where the bed starts to shoal: the first grid column inside the bounds, going shoreward, at
        which the still water depth somewhere across y falls below (1 - share) of the offshore
        depth [m]; None if it never does.

        Parameters:
        -----------
        depth : float | None
            The offshore depth [m]; None for the deepest still water along the offshore edge
        share : float
            The share of the depth the bed must rise by, from 0 to 1

        '''

        if not 0.0 < share < 1.0:
            raise InvalidInputError('The shoaling share must lie between 0 and 1.', parameterName = 'share', value = share)
        startX, endX, startY, endY = self.bounds
        inX = (self.x >= startX - 1e-9) & (self.x <= endX + 1e-9)
        inY = (self.y >= startY - 1e-9) & (self.y <= endY + 1e-9)
        depths = self.stillWaterLevel - self.z[np.ix_(inY, inX)]
        depth = self.offshoreDepth() if depth is None else float(depth)
        shoaled = np.nonzero(depths.min(axis = 0) < (1.0 - share) * depth)[0]

        return float(self.x[inX][shoaled[0]]) if shoaled.size else None

    def offshoreDepth(self) -> float:

        '''The deepest still water along the offshore edge, x = startX, at the grid's y inside the bounds [m].'''

        inY = (self.y >= self.startY - 1e-9) & (self.y <= self.endY + 1e-9)
        y = self.y[inY]

        return float(np.max(self.stillWaterLevel - self.elevation(np.full(y.size, self.startX), y)))

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

    def smoothed(self, sigma: float) -> HeightField:

        '''

        The bed under a Gaussian of standard deviation `sigma`, applied on the grid along x, then y.
        A sinusoid of wavenumber k keeps exp(-k^2 sigma^2 / 2) of its amplitude, to 10^-4 with the
        weights stopping at four sigma, so features much shorter than sigma vanish and features
        much longer pass. Each edge is padded by odd reflection, 2 z_edge - z_mirror, which
        continues a plane exactly, so a plane passes unchanged up to the edges.

        Parameters:
        -----------
        sigma : float
            Standard deviation of the Gaussian [m]; its reach, four sigma, must fit inside the grid
            on each axis

        Raises:
        -------
        InvalidInputError : If sigma is not positive, or reaches past the grid.

        '''

        sigma = float(sigma)
        if not sigma > 0.0:
            raise InvalidInputError('The smoothing length must be positive.', parameterName = 'sigma', value = sigma)
        z = gaussianAlong(self.z, sigma, self.spacingX, axis = 1)
        z = gaussianAlong(z, sigma, self.spacingY, axis = 0)

        return HeightField(self.x, self.y, z, self.stillWaterLevel, bounds = self.bounds)

    def cropped(self, bounds: tuple) -> HeightField:

        '''

        The bed over a smaller region: the grid points within `bounds` and the cells beyond them
        that the interpolation reaches, where this grid has them, so every elevation and gradient
        inside `bounds` matches this field's to rounding.

        Parameters:
        -----------
        bounds : tuple
            (startX, endX, startY, endY), inside this field's bounds [m]

        Raises:
        -------
        InvalidInputError : If the region is empty or reaches past this field's bounds.

        '''

        startX, endX, startY, endY = (float(value) for value in bounds)
        ownStartX, ownEndX, ownStartY, ownEndY = self.bounds
        inside = (ownStartX - 1e-9 <= startX < endX <= ownEndX + 1e-9
                  and ownStartY - 1e-9 <= startY < endY <= ownEndY + 1e-9)
        if not inside:
            raise InvalidInputError("A crop must lie inside the field's bounds.", parameterName = 'bounds',
                                    value = bounds)

        # A query in cell i reads grid points i - 1 to i + 2; one more on each side covers rounding
        kept = []
        for axis, step, low, high in ((self.x, self.spacingX, startX, endX), (self.y, self.spacingY, startY, endY)):
            first = max(0, int(math.floor((low - axis[0]) / step + 1e-9)) - 2)
            last = min(axis.size - 1, int(math.floor((high - axis[0]) / step + 1e-9)) + 3)
            kept.append(slice(first, last + 1))

        return HeightField(self.x[kept[0]], self.y[kept[1]], self.z[kept[1], kept[0]], self.stillWaterLevel,
                           bounds = (startX, endX, startY, endY))

    def rotated(self, direction: float, bounds: tuple, pivot: tuple | None = None,
                gridSpacing: float | None = None) -> HeightField:

        '''

        The bed in a frame turned so that a wave travelling in `direction` over this field travels
        along +x. A point p' of the turned frame lies at pivot + R (p' - pivot) here, R the
        counterclockwise rotation through `direction`, so the pivot keeps its coordinates. The
        turned bed is this one's interpolation sampled on a new grid over `bounds`; a bed of second
        degree turns exactly, as cubic convolution reproduces it in both frames.

        Parameters:
        -----------
        direction : float
            The wave's direction of travel over this field, counterclockwise from +x [deg]
        bounds : tuple
            (startX, endX, startY, endY), the region in the turned frame [m]; turned back, it must
            lie inside this field's bounds
        pivot : tuple | None
            (x, y), the point the frame turns about [m]; None for the centre of this field's bounds
        gridSpacing : float | None
            The new grid's step [m]; None for the smaller of this grid's two

        Raises:
        -------
        InvalidInputError : If the region, turned back, reaches past this field's bounds.

        '''

        angle = math.radians(float(direction))
        cosine, sine = math.cos(angle), math.sin(angle)
        if pivot is None:
            pivot = (0.5 * (self.startX + self.endX), 0.5 * (self.startY + self.endY))
        pivotX, pivotY = (float(value) for value in pivot)

        def toSource(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            dx, dy = x - pivotX, y - pivotY
            return pivotX + cosine * dx - sine * dy, pivotY + sine * dx + cosine * dy

        # The region and this field's bounds are both rectangles, so the corners decide
        startX, endX, startY, endY = (float(value) for value in bounds)
        cornersX, cornersY = toSource(np.array([startX, endX, endX, startX]), np.array([startY, startY, endY, endY]))
        inside = (cornersX.min() >= self.startX - 1e-9 and cornersX.max() <= self.endX + 1e-9
                  and cornersY.min() >= self.startY - 1e-9 and cornersY.max() <= self.endY + 1e-9)
        if not inside:
            raise InvalidInputError("The turned region must lie inside the field's bounds.", parameterName = 'bounds',
                                    value = bounds)
        step = float(gridSpacing) if gridSpacing is not None else min(self.spacingX, self.spacingY)

        return HeightField.fromFunction(lambda x, y: self.elevation(*toSource(x, y)), (startX, endX, startY, endY),
                                        step, self.stillWaterLevel)
