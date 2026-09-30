# -- Surfboard Cross-Section Profile -- #

'''

Computes the cross-sectional profile (deck and bottom heights) at each station.

Direct port of SurfboardGeometry/Surfboard/CrossSection.cs.
Defines the thickness distribution, deck crown dome, and bottom concave
channel at any point on the board surface.

Sean Bowman [02/03/2026]

'''

from __future__ import annotations

import math

from pysurf.Surfboard.SurfPhysics.geometry.parameters import SurfboardParameters


class CrossSection:

    '''

    Surfboard cross-section: deck and bottom heights at any (t, lateralFraction).

    For a given longitudinal position t and lateral position (0 = center, 1 = rail),
    computes the deck (top surface) and bottom surface heights relative to the
    board center plane.

    The thickness envelope varies along the board:
    - Thickest at 40% from nose (100% of maxThickness)
    - Thin at nose (35% of max) and tail (45% of max)
    - Cosine interpolation for smooth transitions

    '''

    def __init__(self, params: SurfboardParameters) -> None:

        '''

        Initialize cross-section from surfboard parameters.

        Parameters:
        -----------
        params : SurfboardParameters
            Board dimensions including crown and concave

        '''

        self._params = params

    def getDeckHeight(self, t: float, lateralFraction: float) -> float:

        '''

        Get the deck height (top surface Z) at a given position.

        The deck profile is a cosine dome that peaks at the centerline
        and drops to zero at the rail edge.

        Parameters:
        -----------
        t : float
            Normalized longitudinal position (0 = nose, 1 = tail)
        lateralFraction : float
            Normalized lateral position (0 = centerline, 1 = rail edge)

        Returns:
        --------
        float : Deck height in mm above the center plane (positive = up)

        '''

        t = max(0.0, min(1.0, t))
        lateralFraction = max(0.0, min(1.0, lateralFraction))

        thickness = self.getLocalThickness(t)
        halfThick = thickness / 2.0

        # Crown: cosine dome peaking at centerline, zero at rail
        crownAtStation = self._getLocalCrown(t)
        crownContribution = crownAtStation * math.cos(lateralFraction * math.pi / 2.0)

        return halfThick + crownContribution

    def getBottomHeight(self, t: float, lateralFraction: float) -> float:

        '''

        Get the bottom height (bottom surface Z) at a given position.

        The bottom surface includes optional single concave: a cosine valley
        at the centerline that fades toward the rails.

        Parameters:
        -----------
        t : float
            Normalized longitudinal position (0 = nose, 1 = tail)
        lateralFraction : float
            Normalized lateral position (0 = centerline, 1 = rail edge)

        Returns:
        --------
        float : Bottom height in mm below center plane (negative = down)

        '''

        t = max(0.0, min(1.0, t))
        lateralFraction = max(0.0, min(1.0, lateralFraction))

        thickness = self.getLocalThickness(t)
        halfThick = thickness / 2.0

        # Concave: cosine^2 valley at centerline
        concaveAtStation = self._getLocalConcave(t)
        concaveContribution = concaveAtStation * math.pow(
            math.cos(lateralFraction * math.pi / 2.0), 2.0
        )

        return -(halfThick + concaveContribution)

    def getLocalThickness(self, t: float) -> float:

        '''

        Get the local board thickness at a normalized longitudinal position.

        Thickness distribution:
        - Thickest at 40% from nose (100% of max)
        - Nose tip: 35% of max
        - Tail tip: 45% of max
        - Cosine interpolation between regions

        Parameters:
        -----------
        t : float
            Normalized position (0 = nose, 1 = tail)

        Returns:
        --------
        float : Thickness in mm

        '''

        # Thickest point at 40% from nose
        thickestPointT = 0.40

        # Min thickness fractions at nose and tail
        noseThicknessFraction = 0.35
        tailThicknessFraction = 0.45

        if t <= thickestPointT:
            # Nose side: interpolate from nose fraction up to 1.0
            localT = t / thickestPointT
            fraction = self._cosineInterpolate(noseThicknessFraction, 1.0, localT)
        else:
            # Tail side: interpolate from 1.0 down to tail fraction
            localT = (t - thickestPointT) / (1.0 - thickestPointT)
            fraction = self._cosineInterpolate(1.0, tailThicknessFraction, localT)

        return self._params.maxThickness * fraction

    def _getLocalCrown(self, t: float) -> float:

        '''

        Get the deck crown height at a normalized longitudinal position.

        Crown is greatest at ~45% from nose and fades toward nose and tail
        using a cosine^1.5 falloff from the peak.

        Parameters:
        -----------
        t : float
            Normalized position (0 = nose, 1 = tail)

        Returns:
        --------
        float : Crown height in mm

        '''

        peakT = 0.45
        distance = abs(t - peakT)
        maxDistance = max(peakT, 1.0 - peakT)
        normalized = distance / maxDistance

        # Cosine falloff from peak
        return self._params.deckCrown * math.pow(
            math.cos(normalized * math.pi / 2.0), 1.5
        )

    def _getLocalConcave(self, t: float) -> float:

        '''

        Get the bottom concave depth at a normalized longitudinal position.

        Concave is at full strength between 25% and 75% of board length,
        with cosine fade-in/fade-out at the ends.

        Parameters:
        -----------
        t : float
            Normalized position (0 = nose, 1 = tail)

        Returns:
        --------
        float : Concave depth in mm

        '''

        startFade = 0.25
        endFade = 0.75

        if t < startFade:
            localT = t / startFade
            return self._params.bottomConcave * self._cosineInterpolate(0.0, 1.0, localT)
        elif t > endFade:
            localT = (t - endFade) / (1.0 - endFade)
            return self._params.bottomConcave * self._cosineInterpolate(1.0, 0.0, localT)
        else:
            return self._params.bottomConcave

    @staticmethod
    def _cosineInterpolate(a: float, b: float, t: float) -> float:

        '''

        Cosine interpolation between two values for smooth transitions.

        Parameters:
        -----------
        a : float
            Start value
        b : float
            End value
        t : float
            Interpolation parameter [0, 1]

        Returns:
        --------
        float : Interpolated value

        '''

        blend = (1.0 - math.cos(t * math.pi)) / 2.0
        return a + (b - a) * blend
