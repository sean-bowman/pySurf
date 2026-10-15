
# -- pySurf: Bathymetry -- #

'''

The sea bed in 2D: a bottom contour z_b(x) that water flows over and a wave breaks on.

A `Bathymetry` is a piecewise-linear profile through (x, z) vertices, with x along the direction
of wave travel (offshore to onshore) and z up. Everything a wave model asks of the bed comes from
that one polyline: the bed elevation and still water depth at any x, the slope, where the still
water line meets the beach, the outward unit normal a boundary is built along, and the closed
polygon a figure fills as sand.

`Bathymetry.planeBeach` builds the profile most breaking studies use: a flat bottom offshore, then
a constant slope rising through the still water line.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

Nothing here is a model: the bed is geometry given as input, and the queries are linear
interpolation along it. tests/testBathymetry.py checks each query against the closed form for a
plane beach (elevation, depth, shoreline position, slope, normal direction).

Sean Bowman [10/11/2026]

'''

# Standard library imports
from __future__ import annotations

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
