# -- Fin Force Model -- #

'''

Fin lift and drag using thin airfoil theory with finite-wing corrections.

At surfing speeds (2-8 m/s) and typical fin chord Reynolds numbers
(~1e5 to 5e5), fins operate in a transitional regime. Thin airfoil
theory provides the lift curve slope, with viscous corrections for drag.

References:
-----------
Gudimetla et al. (2019) -- CFD 3-fin NACA optimization
Anderson, J. -- Fundamentals of Aerodynamics (thin airfoil theory)
Thin airfoil theory: dCl/d_alpha = 2*pi (2D)
Finite wing correction: dCL/d_alpha = 2*pi*AR / (AR + 2)

Sean Bowman [02/03/2026]

'''

from __future__ import annotations

import math
from dataclasses import dataclass

from pysurf.Surfboard.SurfPhysics import constants as const
from pysurf.Surfboard.SurfPhysics.geometry.parameters import SurfboardParameters
from pysurf.Surfboard.SurfPhysics.hydrodynamics.protocols import ForceResult


@dataclass
class FinParameters:

    '''

    Physical parameters for a single fin.

    Parameters:
    -----------
    heightM : float
        Fin span (base to tip) in meters
    baseChordM : float
        Root chord length in meters
    tipChordM : float
        Tip chord length in meters (approximate)
    cantAngleDeg : float
        Outward tilt angle in degrees
    lateralOffsetM : float
        Lateral distance from centerline in meters

    '''

    heightM: float
    baseChordM: float
    tipChordM: float
    cantAngleDeg: float
    lateralOffsetM: float

    @property
    def meanChord(self) -> float:
        '''Mean aerodynamic chord [m].'''
        return (self.baseChordM + self.tipChordM) / 2.0

    @property
    def planformArea(self) -> float:
        '''Approximate trapezoidal planform area [m^2].'''
        return self.heightM * self.meanChord

    @property
    def aspectRatio(self) -> float:
        '''Aspect ratio AR = span^2 / area.'''
        area = self.planformArea
        if area <= 0.0:
            return 0.0
        return self.heightM * self.heightM / area


class FinForceModel:

    '''

    Fin lift and drag model using thin airfoil theory.

    Computes forces on individual fins and sums them for the
    full fin configuration (thruster, twin, quad, single).

    '''

    def __init__(self, finConfig: str, params: SurfboardParameters) -> None:

        '''

        Initialize fin force model for a given configuration.

        Fin dimensions are extracted from the C# FinSystem.cs scaling rules:
        all dimensions scale with boardLength / 1828 (normalized to 6'0" shortboard).

        Parameters:
        -----------
        finConfig : str
            Fin configuration: 'thruster', 'twin', 'quad', or 'single'
        params : SurfboardParameters
            Board parameters for dimensional scaling

        '''

        self._config = finConfig
        self._params = params
        self._fins: list[FinParameters] = []

        scaleFactor = params.length / 1828.0
        self._buildFinConfiguration(scaleFactor)

    def _buildFinConfiguration(self, sf: float) -> None:
        '''Build the fin parameter list based on configuration.'''
        # Convert mm to m
        def mm(val: float) -> float:
            return val * sf * 0.001

        if self._config == 'thruster':
            # Center fin
            self._fins.append(FinParameters(
                heightM=mm(115.0), baseChordM=mm(110.0), tipChordM=mm(15.0),
                cantAngleDeg=0.0, lateralOffsetM=0.0,
            ))
            # Side fins (x2, symmetric)
            for sign in [1.0, -1.0]:
                self._fins.append(FinParameters(
                    heightM=mm(105.0), baseChordM=mm(100.0), tipChordM=mm(15.0),
                    cantAngleDeg=5.0 * sign, lateralOffsetM=0.100 * sf,
                ))

        elif self._config == 'twin':
            # Keel-style twin fins
            for sign in [1.0, -1.0]:
                self._fins.append(FinParameters(
                    heightM=mm(140.0), baseChordM=mm(145.0), tipChordM=mm(15.0),
                    cantAngleDeg=7.0 * sign, lateralOffsetM=0.120 * sf,
                ))

        elif self._config == 'quad':
            # Front pair
            for sign in [1.0, -1.0]:
                self._fins.append(FinParameters(
                    heightM=mm(105.0), baseChordM=mm(100.0), tipChordM=mm(15.0),
                    cantAngleDeg=5.0 * sign, lateralOffsetM=0.115 * sf,
                ))
            # Rear trailer pair
            for sign in [1.0, -1.0]:
                self._fins.append(FinParameters(
                    heightM=mm(75.0), baseChordM=mm(70.0), tipChordM=mm(10.0),
                    cantAngleDeg=3.0 * sign, lateralOffsetM=0.085 * sf,
                ))

        elif self._config == 'single':
            self._fins.append(FinParameters(
                heightM=mm(175.0), baseChordM=mm(165.0), tipChordM=mm(15.0),
                cantAngleDeg=0.0, lateralOffsetM=0.0,
            ))

    def liftCurveSlope(self, fin: FinParameters) -> float:

        '''

        Finite-wing lift curve slope.

        2D thin airfoil: dCl/d_alpha = 2*pi
        3D finite wing: dCL/d_alpha = 2*pi*AR / (AR + 2)

        Parameters:
        -----------
        fin : FinParameters
            Fin geometry

        Returns:
        --------
        float : Lift curve slope in 1/rad

        '''

        ar = fin.aspectRatio
        if ar <= 0.0:
            return 0.0
        return 2.0 * math.pi * ar / (ar + 2.0)

    def computeFinLift(
        self, speed: float, angleOfAttackDeg: float, fin: FinParameters
    ) -> ForceResult:

        '''

        Lift force on a single fin.

        L = 0.5 * rho * V^2 * S * CL
        CL = (dCL/d_alpha) * alpha

        Valid for alpha < stall angle (~12-15 degrees for typical fins).

        Parameters:
        -----------
        speed : float
            Flow speed in m/s
        angleOfAttackDeg : float
            Angle of attack in degrees
        fin : FinParameters
            Fin geometry

        Returns:
        --------
        ForceResult : Fin lift force

        '''

        if speed <= 0.0:
            return ForceResult(force=0.0, description='Fin lift: 0 N')

        alphaRad = math.radians(angleOfAttackDeg)

        # Stall limit at ~15 degrees
        if abs(angleOfAttackDeg) > 15.0:
            alphaRad = math.radians(15.0) * (1.0 if angleOfAttackDeg > 0 else -1.0)

        cl = self.liftCurveSlope(fin) * alphaRad
        area = fin.planformArea
        lift = 0.5 * const.seawaterDensity * speed * speed * area * cl

        return ForceResult(force=lift, description=f'Fin lift: {lift:.2f} N')

    def computeFinDrag(
        self, speed: float, angleOfAttackDeg: float, fin: FinParameters,
        liftCoeff: float = 0.0
    ) -> ForceResult:

        '''

        Drag force on a single fin.

        CD = CD0 + CL^2 / (pi * e * AR)
        CD0 ~ 0.01 (skin friction on thin foil at moderate Re)
        e ~ 0.85 (Oswald span efficiency for swept fin)

        Parameters:
        -----------
        speed : float
            Flow speed in m/s
        angleOfAttackDeg : float
            Angle of attack in degrees
        fin : FinParameters
            Fin geometry
        liftCoeff : float
            Lift coefficient (if already computed)

        Returns:
        --------
        ForceResult : Fin drag force

        '''

        if speed <= 0.0:
            return ForceResult(force=0.0, description='Fin drag: 0 N')

        cd0 = 0.01  # parasitic drag coefficient
        oswaldEfficiency = 0.85
        ar = fin.aspectRatio

        # Induced drag
        if ar > 0.0 and liftCoeff != 0.0:
            cdInduced = liftCoeff * liftCoeff / (math.pi * oswaldEfficiency * ar)
        else:
            # Compute CL from angle of attack
            alphaRad = math.radians(min(abs(angleOfAttackDeg), 15.0))
            cl = self.liftCurveSlope(fin) * alphaRad
            cdInduced = cl * cl / (math.pi * oswaldEfficiency * max(0.1, ar))

        cd = cd0 + cdInduced
        area = fin.planformArea
        drag = 0.5 * const.seawaterDensity * speed * speed * area * cd

        return ForceResult(force=drag, description=f'Fin drag: {drag:.2f} N (Cd = {cd:.4f})')

    def totalFinSystemForce(
        self, speed: float, yawAngleDeg: float, rollAngleDeg: float = 0.0
    ) -> tuple[ForceResult, ForceResult]:

        '''

        Total side force and drag from all fins in the configuration.

        Accounts for cant angle projecting fin forces into body axes.
        Yaw angle creates differential angle of attack on left/right fins.

        Parameters:
        -----------
        speed : float
            Forward speed in m/s
        yawAngleDeg : float
            Yaw (sideslip) angle in degrees
        rollAngleDeg : float
            Roll angle in degrees (default 0)

        Returns:
        --------
        tuple[ForceResult, ForceResult] : (totalSideForce, totalDragForce)

        '''

        totalSide = 0.0
        totalDrag = 0.0

        for fin in self._fins:
            # Effective angle of attack depends on yaw and fin lateral position
            # Side fins see the yaw angle directly
            if fin.lateralOffsetM > 0:
                # Right fin: positive yaw increases AoA
                effectiveAoa = yawAngleDeg
            elif fin.lateralOffsetM < 0:
                # Left fin: positive yaw decreases AoA
                effectiveAoa = -yawAngleDeg
            else:
                # Center fin: sees the yaw angle directly
                effectiveAoa = yawAngleDeg

            liftResult = self.computeFinLift(speed, effectiveAoa, fin)
            dragResult = self.computeFinDrag(speed, effectiveAoa, fin)

            # Project lift into side force accounting for cant angle
            cantRad = math.radians(abs(fin.cantAngleDeg))
            sideComponent = liftResult.force * math.cos(cantRad)
            dragComponent = dragResult.force

            totalSide += sideComponent
            totalDrag += dragComponent

        return (
            ForceResult(force=totalSide, description=f'Fin side force: {totalSide:.2f} N'),
            ForceResult(force=totalDrag, description=f'Fin drag: {totalDrag:.2f} N'),
        )
