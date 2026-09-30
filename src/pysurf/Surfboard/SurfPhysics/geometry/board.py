# -- Board Geometry Facade -- #

'''

Unified board geometry interface for physics calculations.

Composes Outline, RockerProfile, and CrossSection into a single facade
that provides higher-level geometry queries (volume, area, submerged volume)
needed by the hydrodynamics modules.

All output in SI units (meters): converts from mm at the boundary.

Sean Bowman [02/03/2026]

'''

from __future__ import annotations

import math

import numpy as np

from pysurf.Surfboard.SurfPhysics.geometry.parameters import SurfboardParameters
from pysurf.Surfboard.SurfPhysics.geometry.outline import Outline
from pysurf.Surfboard.SurfPhysics.geometry.rocker import RockerProfile
from pysurf.Surfboard.SurfPhysics.geometry.crossSection import CrossSection
from pysurf.Surfboard.SurfPhysics.units import mmToM


class BoardGeometry:

    '''

    Unified board geometry interface for physics calculations.

    Composes Outline, RockerProfile, and CrossSection, providing
    geometry queries in SI meters and numerical integration methods
    for volume, area, and submerged volume.

    '''

    def __init__(self, params: SurfboardParameters) -> None:

        '''

        Initialize board geometry from surfboard parameters.

        Parameters:
        -----------
        params : SurfboardParameters
            Board dimensions (all in mm)

        '''

        self._params = params
        self._outline = Outline(params)
        self._rocker = RockerProfile(params)
        self._crossSection = CrossSection(params)

    @property
    def params(self) -> SurfboardParameters:
        '''Access the underlying parameters.'''
        return self._params

    #--------------------------------------------------------------------#
    # -- Point Queries (SI meters) -- #
    #--------------------------------------------------------------------#

    def getHalfWidthM(self, t: float) -> float:

        '''

        Half-width in meters at normalized position t.

        Parameters:
        -----------
        t : float
            Normalized position (0 = nose, 1 = tail)

        Returns:
        --------
        float : Half-width in meters

        '''

        return mmToM(self._outline.getHalfWidth(t))

    def getRockerHeightM(self, t: float) -> float:

        '''

        Rocker height in meters at normalized position t.

        Parameters:
        -----------
        t : float
            Normalized position (0 = nose, 1 = tail)

        Returns:
        --------
        float : Rocker Z-offset in meters (positive = up from flat spot)

        '''

        return mmToM(self._rocker.getRockerHeight(t))

    def getThicknessM(self, t: float) -> float:

        '''

        Local thickness in meters at normalized position t.

        Parameters:
        -----------
        t : float
            Normalized position (0 = nose, 1 = tail)

        Returns:
        --------
        float : Thickness in meters

        '''

        return mmToM(self._crossSection.getLocalThickness(t))

    def getDeckHeightM(self, t: float, lateralFraction: float) -> float:

        '''

        Deck height in meters at a given position.

        Parameters:
        -----------
        t : float
            Normalized longitudinal position (0 = nose, 1 = tail)
        lateralFraction : float
            Normalized lateral position (0 = centerline, 1 = rail)

        Returns:
        --------
        float : Deck height in meters above center plane

        '''

        return mmToM(self._crossSection.getDeckHeight(t, lateralFraction))

    def getBottomHeightM(self, t: float, lateralFraction: float) -> float:

        '''

        Bottom height in meters at a given position.

        Parameters:
        -----------
        t : float
            Normalized longitudinal position (0 = nose, 1 = tail)
        lateralFraction : float
            Normalized lateral position (0 = centerline, 1 = rail)

        Returns:
        --------
        float : Bottom height in meters below center plane (negative)

        '''

        return mmToM(self._crossSection.getBottomHeight(t, lateralFraction))

    #--------------------------------------------------------------------#
    # -- Integrated Geometry Quantities -- #
    #--------------------------------------------------------------------#

    def computeVolume(self, nStationsLength: int = 200, nStationsWidth: int = 50) -> float:

        '''

        Numerical volume integration using trapezoidal rule.

        Integrates the cross-sectional area along the board length.
        At each station, the cross-section is integrated across the width
        to get the local area, accounting for deck crown and bottom concave.

        Parameters:
        -----------
        nStationsLength : int
            Number of longitudinal integration stations
        nStationsWidth : int
            Number of lateral integration points per station

        Returns:
        --------
        float : Board volume in m^3

        '''

        lengthM = mmToM(self._params.length)
        dt = 1.0 / nStationsLength

        totalVolume = 0.0

        for i in range(nStationsLength + 1):
            t = i * dt
            halfWidthM = self.getHalfWidthM(t)

            if halfWidthM <= 0.0:
                continue

            # Integrate cross-sectional area at this station
            # using trapezoidal rule across half-width, then double for symmetry
            dLateral = 1.0 / nStationsWidth
            localArea = 0.0

            for j in range(nStationsWidth + 1):
                lateralFraction = j * dLateral
                deckZ = self.getDeckHeightM(t, lateralFraction)
                bottomZ = self.getBottomHeightM(t, lateralFraction)
                localThickness = deckZ - bottomZ  # always positive

                # Trapezoidal weight
                weight = 0.5 if (j == 0 or j == nStationsWidth) else 1.0
                localArea += weight * localThickness * dLateral

            # localArea is for one half of the width (lateralFraction 0..1)
            # Scale by actual half-width and double for symmetry
            crossSectionArea = 2.0 * localArea * halfWidthM

            # Trapezoidal weight for longitudinal integration
            lengthWeight = 0.5 if (i == 0 or i == nStationsLength) else 1.0
            totalVolume += lengthWeight * crossSectionArea * dt

        # Scale by total length
        totalVolume *= lengthM

        return totalVolume

    def computePlanformArea(self, nStations: int = 200) -> float:

        '''

        Planform area in m^2 from integrating 2 * halfWidth over length.

        Parameters:
        -----------
        nStations : int
            Number of integration stations

        Returns:
        --------
        float : Planform area in m^2

        '''

        lengthM = mmToM(self._params.length)
        dt = 1.0 / nStations
        totalArea = 0.0

        for i in range(nStations + 1):
            t = i * dt
            width = 2.0 * self.getHalfWidthM(t)
            weight = 0.5 if (i == 0 or i == nStations) else 1.0
            totalArea += weight * width * dt

        return totalArea * lengthM

    def computeWettedSurfaceArea(
        self, nStationsLength: int = 200, nStationsWidth: int = 50
    ) -> float:

        '''

        Bottom wetted surface area in m^2.

        Integrates the bottom surface arc length across width at each station,
        then integrates along the length. Accounts for bottom concave curvature.

        Parameters:
        -----------
        nStationsLength : int
            Number of longitudinal stations
        nStationsWidth : int
            Number of lateral points per station

        Returns:
        --------
        float : Bottom surface area in m^2

        '''

        lengthM = mmToM(self._params.length)
        dt = 1.0 / nStationsLength

        totalArea = 0.0

        for i in range(nStationsLength + 1):
            t = i * dt
            halfWidthM = self.getHalfWidthM(t)

            if halfWidthM <= 0.0:
                continue

            # Approximate bottom surface arc length across the width
            dLateral = 1.0 / nStationsWidth
            arcLength = 0.0

            for j in range(nStationsWidth):
                lat1 = j * dLateral
                lat2 = (j + 1) * dLateral

                y1 = lat1 * halfWidthM
                y2 = lat2 * halfWidthM
                z1 = self.getBottomHeightM(t, lat1)
                z2 = self.getBottomHeightM(t, lat2)

                dy = y2 - y1
                dz = z2 - z1
                arcLength += math.sqrt(dy * dy + dz * dz)

            # Double for symmetry
            localWidth = 2.0 * arcLength

            lengthWeight = 0.5 if (i == 0 or i == nStationsLength) else 1.0
            totalArea += lengthWeight * localWidth * dt

        return totalArea * lengthM

    def estimateDeadriseAngle(self, t: float) -> float:

        '''

        Estimate the deadrise angle in degrees at a given longitudinal station.

        Deadrise is the angle between the bottom surface and the horizontal plane,
        measured from the centerline outward. Driven by concave depth relative
        to the half-width.

        Parameters:
        -----------
        t : float
            Normalized position (0 = nose, 1 = tail)

        Returns:
        --------
        float : Deadrise angle in degrees

        '''

        halfWidthM = self.getHalfWidthM(t)
        if halfWidthM <= 0.0:
            return 0.0

        # Bottom height at center vs rail
        bottomCenter = self.getBottomHeightM(t, 0.0)
        bottomRail = self.getBottomHeightM(t, 1.0)

        # The rise from center to rail over the half-width
        rise = bottomRail - bottomCenter  # positive if rail higher than center (concave)
        angleRad = math.atan2(abs(rise), halfWidthM)

        return math.degrees(angleRad)

    def computeSubmergedVolume(
        self,
        draft: float,
        trimAngleDeg: float = 0.0,
        nStationsLength: int = 200,
        nStationsWidth: int = 50,
    ) -> float:

        '''

        Volume of the board below a given waterplane height.

        The waterplane is defined at height 'draft' above the lowest point
        of the board (the rocker flat spot). With trim, the waterplane tilts
        so the nose lifts and tail sinks (or vice versa).

        Parameters:
        -----------
        draft : float
            Waterplane height in meters above the lowest point of the board
        trimAngleDeg : float
            Pitch angle in degrees (positive = nose up)
        nStationsLength : int
            Number of longitudinal integration stations
        nStationsWidth : int
            Number of lateral integration points per station

        Returns:
        --------
        float : Submerged volume in m^3

        '''

        lengthM = mmToM(self._params.length)
        trimRad = math.radians(trimAngleDeg)
        dt = 1.0 / nStationsLength

        # The waterplane height varies along the board with trim
        # At the flat spot (t=0.6), the draft is as specified
        # Forward or aft, it changes by length * sin(trim)
        flatSpotT = 0.6

        totalVolume = 0.0

        for i in range(nStationsLength + 1):
            t = i * dt
            halfWidthM = self.getHalfWidthM(t)

            if halfWidthM <= 0.0:
                continue

            # Local waterplane height adjusted for trim
            distFromFlatSpot = (t - flatSpotT) * lengthM
            localWaterlineZ = draft - distFromFlatSpot * math.sin(trimRad)

            # Rocker height at this station (board bottom Z reference)
            rockerZ = self.getRockerHeightM(t)

            # Integrate submerged thickness across the width
            dLateral = 1.0 / nStationsWidth
            localSubmergedArea = 0.0

            for j in range(nStationsWidth + 1):
                lateralFraction = j * dLateral
                bottomZ = self.getBottomHeightM(t, lateralFraction) + rockerZ
                deckZ = self.getDeckHeightM(t, lateralFraction) + rockerZ

                # Submerged thickness at this point
                if localWaterlineZ <= bottomZ:
                    subThick = 0.0
                elif localWaterlineZ >= deckZ:
                    subThick = deckZ - bottomZ
                else:
                    subThick = localWaterlineZ - bottomZ

                weight = 0.5 if (j == 0 or j == nStationsWidth) else 1.0
                localSubmergedArea += weight * subThick * dLateral

            # Scale by half-width, double for symmetry
            crossArea = 2.0 * localSubmergedArea * halfWidthM

            lengthWeight = 0.5 if (i == 0 or i == nStationsLength) else 1.0
            totalVolume += lengthWeight * crossArea * dt

        return totalVolume * lengthM

    def computeWettedLengthAndArea(
        self,
        draft: float,
        trimAngleDeg: float = 0.0,
        nStations: int = 200,
    ) -> tuple[float, float]:

        '''

        Compute wetted length and wetted bottom area for a given draft and trim.

        Wetted length is the longitudinal extent where the board is below
        the waterline. Wetted area is the bottom surface area in contact
        with water.

        Parameters:
        -----------
        draft : float
            Waterplane height in meters above the lowest point
        trimAngleDeg : float
            Pitch angle in degrees (positive = nose up)
        nStations : int
            Number of integration stations

        Returns:
        --------
        tuple[float, float] : (wettedLengthM, wettedAreaM2)

        '''

        lengthM = mmToM(self._params.length)
        trimRad = math.radians(trimAngleDeg)
        dt = 1.0 / nStations
        flatSpotT = 0.6

        firstWet = None
        lastWet = None
        wettedArea = 0.0

        for i in range(nStations + 1):
            t = i * dt
            halfWidthM = self.getHalfWidthM(t)

            if halfWidthM <= 0.0:
                continue

            distFromFlatSpot = (t - flatSpotT) * lengthM
            localWaterlineZ = draft - distFromFlatSpot * math.sin(trimRad)
            rockerZ = self.getRockerHeightM(t)

            # Check if any part of the bottom is below waterline
            bottomCenterZ = self.getBottomHeightM(t, 0.0) + rockerZ

            if localWaterlineZ > bottomCenterZ:
                if firstWet is None:
                    firstWet = t
                lastWet = t

                # Approximate wetted width at this station
                localWettedWidth = 2.0 * halfWidthM
                lengthWeight = 0.5 if (i == 0 or i == nStations) else 1.0
                wettedArea += lengthWeight * localWettedWidth * dt

        if firstWet is None or lastWet is None:
            return (0.0, 0.0)

        wettedLengthM = (lastWet - firstWet) * lengthM
        wettedAreaM2 = wettedArea * lengthM

        return (wettedLengthM, wettedAreaM2)

    def getLengthM(self) -> float:
        '''Board length in meters.'''
        return mmToM(self._params.length)

    def getMaxWidthM(self) -> float:
        '''Maximum width in meters.'''
        return mmToM(self._params.maxWidth)

    def getMaxThicknessM(self) -> float:
        '''Maximum thickness in meters.'''
        return mmToM(self._params.maxThickness)
