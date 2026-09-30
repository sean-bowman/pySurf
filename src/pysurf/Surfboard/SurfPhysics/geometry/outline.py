# -- Surfboard Planform Outline -- #

'''

Computes the surfboard planform outline (half-width at each longitudinal station).

Direct port of SurfboardGeometry/Surfboard/Outline.cs.
Uses a two-piece power curve split at the wide point, with exponents
calibrated from NoseWidth and TailWidth measurements.

Sean Bowman [02/03/2026]

'''

from __future__ import annotations

import math

from pysurf.Surfboard.SurfPhysics.geometry.parameters import SurfboardParameters


class Outline:

    '''

    Surfboard planform outline: half-width at each longitudinal station.

    The board is symmetric about its centerline, so only the half-width is
    computed. Given a normalized position t (0 = nose tip, 1 = tail tip),
    returns the half-width in mm.

    The outline is a two-piece power curve split at the wide point:
    - Nose side: halfWidth = maxHalfWidth * pow(localT, noseExponent)
    - Tail side: halfWidth = tailTipHalf + (maxHalfWidth - tailTipHalf) * pow(1 - localT, tailExponent)

    Exponents are calibrated so the curve passes through the NoseWidth
    and TailWidth measurements at their respective 305mm stations.

    '''

    def __init__(self, params: SurfboardParameters) -> None:

        '''

        Initialize outline from surfboard parameters.

        Parameters:
        -----------
        params : SurfboardParameters
            Board dimensions including outline control points

        '''

        self._params = params

        # Normalized wide point position (0 = nose, 1 = tail)
        self._widePointT: float = params.widePointX / params.length
        self._maxHalfWidth: float = params.halfWidth
        self._tailTipHalf: float = params.tailTipHalfWidth

        # Calibrate nose exponent from NoseWidth at 305mm station
        # The nose power curve: halfWidth = maxHalfWidth * pow(localT, n)
        # We want it to pass through noseWidth/2 at noseStationT
        noseStationT = 305.0 / params.length
        noseRatio = (params.noseWidth / 2.0) / self._maxHalfWidth
        nosePositionRatio = noseStationT / self._widePointT

        if 0.0 < noseRatio < 1.0 and 0.0 < nosePositionRatio < 1.0:
            self._noseExponent = math.log(noseRatio) / math.log(nosePositionRatio)
        else:
            self._noseExponent = 0.5  # fallback: square root curve

        # Clamp to reasonable range
        self._noseExponent = max(0.3, min(1.5, self._noseExponent))

        # Calibrate tail exponent from TailWidth at 305mm from tail tip
        # The tail power curve:
        #   halfWidth = tailTipHalf + (maxHalfWidth - tailTipHalf) * pow(1 - localT, n)
        tailStationFromTail = 305.0 / params.length
        tailStationT = 1.0 - tailStationFromTail  # position from nose
        tailLocalT = (tailStationT - self._widePointT) / (1.0 - self._widePointT)

        tailTargetHalf = params.tailWidth / 2.0
        tailFraction = (tailTargetHalf - self._tailTipHalf) / (self._maxHalfWidth - self._tailTipHalf)
        tailBlendArg = 1.0 - tailLocalT

        if 0.0 < tailFraction < 1.0 and 0.0 < tailBlendArg < 1.0:
            self._tailExponent = math.log(tailFraction) / math.log(tailBlendArg)
        else:
            self._tailExponent = 0.6  # fallback

        # Clamp to reasonable range
        self._tailExponent = max(0.3, min(2.0, self._tailExponent))

        # -- Wide-point smoothing via cubic Hermite blending --
        # The two power curves have non-zero slopes at the wide point,
        # creating a cusp. Add a blend zone on each side where a cubic
        # Hermite spline smoothly transitions to zero slope at the max.
        blendFraction = 0.30  # 30% of segment length on each side

        self._noseBlendStart = self._widePointT * (1.0 - blendFraction)
        self._tailBlendEnd = (
            self._widePointT + (1.0 - self._widePointT) * blendFraction
        )

        # Pre-compute nose-side power curve value and slope at blend boundary
        noseBlendLocalT = self._noseBlendStart / self._widePointT
        self._noseBlendValue = self._maxHalfWidth * math.pow(
            noseBlendLocalT, self._noseExponent
        )
        # d(halfWidth)/dt = maxHalfWidth * n * localT^(n-1) / widePointT
        self._noseBlendSlope = (
            self._maxHalfWidth * self._noseExponent
            * math.pow(noseBlendLocalT, self._noseExponent - 1.0)
            / self._widePointT
        )

        # Pre-compute tail-side power curve value and slope at blend boundary
        tailBlendLocalT = (
            (self._tailBlendEnd - self._widePointT)
            / (1.0 - self._widePointT)
        )
        tailBlendComplement = 1.0 - tailBlendLocalT
        self._tailBlendValue = (
            self._tailTipHalf
            + (self._maxHalfWidth - self._tailTipHalf)
            * math.pow(tailBlendComplement, self._tailExponent)
        )
        # d(halfWidth)/dt = -(maxHW - tipHW) * n * (1-localT)^(n-1) / (1-widePointT)
        self._tailBlendSlope = (
            -(self._maxHalfWidth - self._tailTipHalf) * self._tailExponent
            * math.pow(tailBlendComplement, self._tailExponent - 1.0)
            / (1.0 - self._widePointT)
        )

    def getHalfWidth(self, t: float) -> float:

        '''

        Get the half-width of the board at a normalized longitudinal position.

        Uses two power curves (nose and tail) with cubic Hermite blending
        near the wide point to ensure C1 smoothness at the outline maximum.

        Parameters:
        -----------
        t : float
            Normalized position: 0.0 = nose tip, 1.0 = tail tip

        Returns:
        --------
        float : Half-width in mm at the given position

        '''

        t = max(0.0, min(1.0, t))

        if t <= 0.0:
            return 0.0

        elif t < self._noseBlendStart:
            # Nose side: unmodified power curve
            localT = t / self._widePointT
            return self._maxHalfWidth * math.pow(localT, self._noseExponent)

        elif t <= self._widePointT:
            # Nose-side blend zone: Hermite spline from power curve to wide point
            return self._hermiteBlend(
                t, self._noseBlendStart, self._widePointT,
                self._noseBlendValue, self._maxHalfWidth,
                self._noseBlendSlope, 0.0,
            )

        elif t < self._tailBlendEnd:
            # Tail-side blend zone: Hermite spline from wide point to power curve
            return self._hermiteBlend(
                t, self._widePointT, self._tailBlendEnd,
                self._maxHalfWidth, self._tailBlendValue,
                0.0, self._tailBlendSlope,
            )

        elif t < 1.0:
            # Tail side: unmodified power curve
            localT = (t - self._widePointT) / (1.0 - self._widePointT)
            blend = math.pow(1.0 - localT, self._tailExponent)
            return self._tailTipHalf + (self._maxHalfWidth - self._tailTipHalf) * blend

        else:
            return self._tailTipHalf

    @staticmethod
    def _hermiteBlend(
        t: float,
        t0: float,
        t1: float,
        p0: float,
        p1: float,
        m0: float,
        m1: float,
    ) -> float:

        '''

        Cubic Hermite interpolation between two endpoints with specified slopes.

        Evaluates the Hermite spline at position t within [t0, t1].
        The spline passes through (t0, p0) with slope m0 and (t1, p1) with
        slope m1, providing C1 continuity at both boundaries.

        Parameters:
        -----------
        t : float
            Evaluation position within [t0, t1]
        t0 : float
            Start of the Hermite interval
        t1 : float
            End of the Hermite interval
        p0 : float
            Function value at t0
        p1 : float
            Function value at t1
        m0 : float
            Derivative (slope) at t0 in units per t
        m1 : float
            Derivative (slope) at t1 in units per t

        Returns:
        --------
        float : Interpolated value at t

        '''

        dt = t1 - t0
        s = (t - t0) / dt  # normalized parameter [0, 1]

        s2 = s * s
        s3 = s2 * s

        # Standard Hermite basis functions
        h00 = 2.0 * s3 - 3.0 * s2 + 1.0
        h10 = s3 - 2.0 * s2 + s
        h01 = -2.0 * s3 + 3.0 * s2
        h11 = s3 - s2

        # Tangent vectors are scaled by the interval width (dt)
        return h00 * p0 + h10 * (m0 * dt) + h01 * p1 + h11 * (m1 * dt)
