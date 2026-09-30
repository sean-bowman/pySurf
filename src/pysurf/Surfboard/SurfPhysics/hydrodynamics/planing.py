# -- Planing Lift and Drag Model -- #

'''

Planing hull hydrodynamic model adapted from Savitsky (1964).

The Savitsky method predicts lift, drag, wetted area, and center of
pressure for prismatic planing surfaces. For surfboards (non-prismatic
hulls with variable beam, deadrise, and rocker), the method is adapted
by averaging geometry properties over the wetted region.

References:
-----------
Savitsky, D. (1964) -- Hydrodynamic Design of Planing Hulls,
    Marine Technology and SNAME News, 1(04):71-95
    https://dlba-inc.com/library/hydrodynamic-design-of-planing-hulls/
Paine, F. -- The Science of Surfing (Cambridge University Press)
Matveev (2024) -- Bottom modifications for planing boards

Sean Bowman [02/03/2026]

'''

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.optimize import fsolve

from pysurf.Surfboard.SurfPhysics import constants as const
from pysurf.Surfboard.SurfPhysics.geometry.board import BoardGeometry
from pysurf.Surfboard.SurfPhysics.geometry.parameters import SurfboardParameters
from pysurf.Surfboard.SurfPhysics.hydrodynamics.protocols import ForceResult, PlaningState
from pysurf.Surfboard.SurfPhysics.hydrodynamics.friction import FrictionDragModel


class PlaningModel:

    '''

    Planing lift and drag model adapted from Savitsky (1964).

    Predicts hydrodynamic lift, drag, and equilibrium conditions for
    a surfboard at planing speeds. The Savitsky formulas are adapted
    for non-prismatic surfboard geometry by averaging deadrise and beam
    over the wetted region.

    '''

    def __init__(self, board: BoardGeometry, params: SurfboardParameters) -> None:

        '''

        Initialize planing model.

        Parameters:
        -----------
        board : BoardGeometry
            Board geometry for shape queries
        params : SurfboardParameters
            Board parameters

        '''

        self._board = board
        self._params = params
        self._friction = FrictionDragModel()

    def _effectiveBeam(self, wettedLengthFraction: float = 0.5) -> float:

        '''

        Effective beam (width) in meters, averaged over the wetted region.

        For planing, the wetted region is typically the aft portion of the board.
        The effective beam is taken at the centroid of the wetted area.

        Parameters:
        -----------
        wettedLengthFraction : float
            Fraction of board length that is wetted (from tail forward)

        Returns:
        --------
        float : Effective beam in meters

        '''

        # Sample the width over the wetted region (from tail backward)
        nSamples = 20
        totalWidth = 0.0

        for i in range(nSamples):
            t = 1.0 - wettedLengthFraction + wettedLengthFraction * (i + 0.5) / nSamples
            totalWidth += 2.0 * self._board.getHalfWidthM(t)

        return totalWidth / nSamples

    def _effectiveDeadrise(self, wettedLengthFraction: float = 0.5) -> float:

        '''

        Effective deadrise angle in degrees, averaged over the wetted region.

        Parameters:
        -----------
        wettedLengthFraction : float
            Fraction of board length that is wetted

        Returns:
        --------
        float : Average deadrise angle in degrees

        '''

        nSamples = 10
        totalAngle = 0.0

        for i in range(nSamples):
            t = 1.0 - wettedLengthFraction + wettedLengthFraction * (i + 0.5) / nSamples
            totalAngle += self._board.estimateDeadriseAngle(t)

        return totalAngle / nSamples

    def liftCoefficient(
        self,
        speed: float,
        trimAngleDeg: float,
        wettedLengthM: float,
        beamM: float,
        deadriseDeg: float = 0.0,
    ) -> float:

        '''

        Savitsky lift coefficient for a planing surface.

        For zero deadrise:
            CL0 = tau^1.1 * (0.0120 * lambda^0.5 / Cv^2 + 0.0055 * lambda^2.5 / Cv^2)

        Deadrise correction:
            CL_beta = CL0 - 0.0065 * beta * CL0^0.60

        where:
            tau = trim angle in degrees
            lambda = wettedLength / beam
            Cv = V / sqrt(g * beam) (speed coefficient)
            beta = deadrise angle in degrees

        Parameters:
        -----------
        speed : float
            Forward speed in m/s
        trimAngleDeg : float
            Trim angle in degrees (must be > 0)
        wettedLengthM : float
            Wetted length in meters
        beamM : float
            Beam (width) in meters
        deadriseDeg : float
            Deadrise angle in degrees

        Returns:
        --------
        float : Lift coefficient (dimensionless)

        '''

        if speed <= 0.0 or beamM <= 0.0 or trimAngleDeg <= 0.0 or wettedLengthM <= 0.0:
            return 0.0

        tau = trimAngleDeg
        lam = wettedLengthM / beamM  # wetted length-to-beam ratio
        cv = speed / math.sqrt(const.gravity * beamM)  # speed coefficient

        if cv <= 0.0:
            return 0.0

        cv2 = cv * cv

        # Zero-deadrise lift coefficient (Savitsky Eq. 1)
        cl0 = math.pow(tau, 1.1) * (
            0.0120 * math.pow(lam, 0.5) / cv2
            + 0.0055 * math.pow(lam, 2.5) / cv2
        )

        # Deadrise correction (Savitsky Eq. 2)
        if deadriseDeg > 0.0 and cl0 > 0.0:
            cl = cl0 - 0.0065 * deadriseDeg * math.pow(cl0, 0.60)
        else:
            cl = cl0

        return max(0.0, cl)

    def computePlaningLift(self, speed: float, trimAngleDeg: float) -> ForceResult:

        '''

        Hydrodynamic planing lift force.

        L = 0.5 * rho * V^2 * beam^2 * CL

        Parameters:
        -----------
        speed : float
            Forward speed in m/s
        trimAngleDeg : float
            Trim angle in degrees

        Returns:
        --------
        ForceResult : Planing lift force

        '''

        if speed <= 0.0 or trimAngleDeg <= 0.0:
            return ForceResult(force=0.0, description='Planing lift: 0 N (no speed/trim)')

        # Estimate wetted length fraction from trim and speed
        # At higher speeds and lower trim, less length is wetted
        wettedFraction = max(0.1, min(0.8, 0.5 / max(0.1, trimAngleDeg / 5.0)))
        wettedLengthM = wettedFraction * self._board.getLengthM()

        beamM = self._effectiveBeam(wettedFraction)
        deadriseDeg = self._effectiveDeadrise(wettedFraction)

        cl = self.liftCoefficient(speed, trimAngleDeg, wettedLengthM, beamM, deadriseDeg)
        lift = 0.5 * const.seawaterDensity * speed * speed * beamM * beamM * cl

        return ForceResult(
            force=lift,
            description=f'Planing lift: {lift:.1f} N (CL = {cl:.4f})',
        )

    def computePlaningDrag(
        self, speed: float, trimAngleDeg: float, liftForce: float
    ) -> ForceResult:

        '''

        Total planing drag = friction + pressure + spray.

        Pressure drag: D_pressure = lift * tan(trimAngle)
        Spray drag: ~10% of friction drag (empirical)

        Parameters:
        -----------
        speed : float
            Forward speed in m/s
        trimAngleDeg : float
            Trim angle in degrees
        liftForce : float
            Lift force in Newtons (needed for pressure drag)

        Returns:
        --------
        ForceResult : Total planing drag

        '''

        if speed <= 0.0 or trimAngleDeg <= 0.0:
            return ForceResult(force=0.0, description='Planing drag: 0 N')

        # Pressure drag component (Savitsky): D_p = L * tan(tau)
        trimRad = math.radians(trimAngleDeg)
        pressureDrag = liftForce * math.tan(trimRad)

        # Friction drag on the wetted bottom
        wettedFraction = max(0.1, min(0.8, 0.5 / max(0.1, trimAngleDeg / 5.0)))
        _, wettedArea = self._board.computeWettedLengthAndArea(
            draft=0.01,  # small draft in planing
            trimAngleDeg=trimAngleDeg,
        )
        # Fallback if wetted area computation returns 0
        if wettedArea <= 0.0:
            wettedArea = self._board.computePlanformArea() * wettedFraction

        wettedLength = wettedFraction * self._board.getLengthM()
        frictionResult = self._friction.computeFrictionDrag(speed, wettedArea, wettedLength)

        # Spray drag: empirical ~10% of friction
        sprayDrag = 0.1 * frictionResult.force

        totalDrag = pressureDrag + frictionResult.force + sprayDrag

        return ForceResult(
            force=totalDrag,
            description=(
                f'Planing drag: {totalDrag:.2f} N '
                f'(pressure={pressureDrag:.2f}, friction={frictionResult.force:.2f}, '
                f'spray={sprayDrag:.2f})'
            ),
        )

    def findPlaningEquilibrium(
        self, speed: float, totalWeightN: float
    ) -> PlaningState:

        '''

        Find the trim angle where lift equals weight.

        Solves for the equilibrium trim angle at a given speed using
        root-finding. At equilibrium, planing lift + residual buoyancy = weight.

        Parameters:
        -----------
        speed : float
            Forward speed in m/s
        totalWeightN : float
            Total weight (board + rider) in Newtons

        Returns:
        --------
        PlaningState : Equilibrium state

        '''

        if speed <= 0.5:
            # Below planing speed: displacement mode
            return PlaningState(
                speed=speed,
                trimAngleDeg=0.0,
                wettedLengthM=self._board.getLengthM(),
                draftM=0.0,
                liftForceN=0.0,
                dragForceN=0.0,
                liftToDrag=0.0,
                isPlaning=False,
            )

        # Sweep trim angles to find where lift is maximized, then find
        # the equilibrium where lift = weight (if achievable)
        bestTau = 5.0
        bestLift = 0.0

        for testTau in np.linspace(1.0, 15.0, 30):
            testLift = self.computePlaningLift(speed, testTau).force
            if testLift > bestLift:
                bestLift = testLift
                bestTau = testTau

        if bestLift < totalWeightN:
            # Planing lift alone cannot support the weight at this speed.
            # Use the trim angle that maximizes lift (best partial planing).
            tauSolution = bestTau
        else:
            # Lift can support weight: find the exact equilibrium trim
            def residual(tau: float) -> float:
                tau = max(0.5, tau)
                return self.computePlaningLift(speed, tau).force - totalWeightN

            try:
                tauSolution = fsolve(residual, x0=bestTau, full_output=False)[0]
                tauSolution = max(0.5, min(15.0, float(tauSolution)))
            except Exception:
                tauSolution = bestTau

        # Compute forces at equilibrium
        liftResult = self.computePlaningLift(speed, tauSolution)
        dragResult = self.computePlaningDrag(speed, tauSolution, liftResult.force)

        wettedFraction = max(0.1, min(0.8, 0.5 / max(0.1, tauSolution / 5.0)))
        wettedLength = wettedFraction * self._board.getLengthM()

        lToDrag = liftResult.force / max(0.01, dragResult.force)

        isPlaning = speed > self.planingThresholdSpeed(totalWeightN)

        return PlaningState(
            speed=speed,
            trimAngleDeg=tauSolution,
            wettedLengthM=wettedLength,
            draftM=0.01,  # approximate small draft in planing
            liftForceN=liftResult.force,
            dragForceN=dragResult.force,
            liftToDrag=lToDrag,
            isPlaning=isPlaning,
        )

    def sweepSpeed(
        self, speedRange: np.ndarray, totalWeightN: float
    ) -> list[PlaningState]:

        '''

        Compute planing equilibrium at each speed in the range.

        Parameters:
        -----------
        speedRange : np.ndarray
            Array of speeds in m/s
        totalWeightN : float
            Total weight in Newtons

        Returns:
        --------
        list[PlaningState] : Equilibrium state at each speed

        '''

        return [self.findPlaningEquilibrium(v, totalWeightN) for v in speedRange]

    def planingThresholdSpeed(self, totalWeightN: float) -> float:

        '''

        Estimate the minimum speed for planing onset.

        Based on Froude number criterion: Fn = V / sqrt(g * L) ~ 1.0

        Parameters:
        -----------
        totalWeightN : float
            Total weight in Newtons (not used directly, but included
            for interface consistency)

        Returns:
        --------
        float : Planing threshold speed in m/s

        '''

        lengthM = self._board.getLengthM()
        return math.sqrt(const.gravity * lengthM)
