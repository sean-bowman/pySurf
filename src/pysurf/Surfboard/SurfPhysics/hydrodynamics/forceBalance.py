# -- Combined Force Balance Solver -- #

'''

Brings together buoyancy, friction, planing, and fin forces to compute
the total force state and find equilibrium conditions across speed ranges.

Sean Bowman [02/03/2026]

'''

from __future__ import annotations

import numpy as np

from pysurf.Surfboard.SurfPhysics import constants as const
from pysurf.Surfboard.SurfPhysics.geometry.board import BoardGeometry
from pysurf.Surfboard.SurfPhysics.geometry.parameters import SurfboardParameters
from pysurf.Surfboard.SurfPhysics.hydrodynamics.protocols import ForceResult, PlaningState
from pysurf.Surfboard.SurfPhysics.hydrodynamics.buoyancy import BuoyancyModel
from pysurf.Surfboard.SurfPhysics.hydrodynamics.friction import FrictionDragModel
from pysurf.Surfboard.SurfPhysics.hydrodynamics.planing import PlaningModel
from pysurf.Surfboard.SurfPhysics.hydrodynamics.fins import FinForceModel


class ForceBalance:

    '''

    Combined force and moment balance for a surfboard at speed.

    Composes buoyancy, friction, planing, and fin models to compute
    the total force state and generate performance curves.

    '''

    def __init__(
        self,
        board: BoardGeometry,
        params: SurfboardParameters,
        riderMassKg: float = 75.0,
    ) -> None:

        '''

        Initialize force balance model.

        Parameters:
        -----------
        board : BoardGeometry
            Board geometry
        params : SurfboardParameters
            Board parameters
        riderMassKg : float
            Rider mass in kg

        '''

        self._board = board
        self._params = params
        self._buoyancy = BuoyancyModel(board, params)
        self._friction = FrictionDragModel()
        self._planing = PlaningModel(board, params)
        self._fins = FinForceModel(params.defaultFinConfiguration, params)

        self._boardMassKg = self._buoyancy.estimateBoardMass()
        self._riderMassKg = riderMassKg
        self._totalMassKg = self._boardMassKg + riderMassKg
        self._totalWeightN = self._totalMassKg * const.gravity

    @property
    def boardMassKg(self) -> float:
        '''Estimated board mass in kg.'''
        return self._boardMassKg

    @property
    def totalMassKg(self) -> float:
        '''Total system mass (board + rider) in kg.'''
        return self._totalMassKg

    @property
    def totalWeightN(self) -> float:
        '''Total weight in Newtons.'''
        return self._totalWeightN

    def computeForcesAtState(
        self,
        speed: float,
        trimAngleDeg: float,
        yawAngleDeg: float = 0.0,
    ) -> dict[str, ForceResult]:

        '''

        Compute all forces at a given speed, trim, and yaw.

        Parameters:
        -----------
        speed : float
            Forward speed in m/s
        trimAngleDeg : float
            Pitch angle in degrees
        yawAngleDeg : float
            Yaw angle in degrees

        Returns:
        --------
        dict[str, ForceResult] : Named force components

        '''

        results: dict[str, ForceResult] = {}

        # Weight (downward)
        results['weight'] = ForceResult(
            force=-self._totalWeightN,
            description=f'Weight: {self._totalWeightN:.1f} N',
        )

        # Buoyancy (if at low speed / displacement mode)
        draft = self._buoyancy.findEquilibriumDraft(self._totalMassKg)
        results['buoyancy'] = self._buoyancy.computeBuoyancyForce(draft, trimAngleDeg)

        # Planing lift
        results['planing_lift'] = self._planing.computePlaningLift(speed, trimAngleDeg)

        # Planing drag (includes friction, pressure, spray components)
        results['planing_drag'] = self._planing.computePlaningDrag(
            speed, trimAngleDeg, results['planing_lift'].force
        )

        # Form drag (frontal area)
        frontalArea = self._board.getMaxWidthM() * self._board.getMaxThicknessM() * 0.5
        results['form_drag'] = self._friction.computeFormDrag(speed, frontalArea)

        # Fin forces
        finSide, finDrag = self._fins.totalFinSystemForce(speed, yawAngleDeg)
        results['fin_side'] = finSide
        results['fin_drag'] = finDrag

        return results

    def totalDrag(self, speed: float, trimAngleDeg: float) -> float:

        '''

        Sum of all drag components in Newtons.

        Parameters:
        -----------
        speed : float
            Forward speed in m/s
        trimAngleDeg : float
            Trim angle in degrees

        Returns:
        --------
        float : Total drag in Newtons

        '''

        forces = self.computeForcesAtState(speed, trimAngleDeg)
        return (
            forces['planing_drag'].force
            + forces['form_drag'].force
            + forces['fin_drag'].force
        )

    def totalLift(self, speed: float, trimAngleDeg: float) -> float:

        '''

        Sum of buoyancy + planing lift in Newtons.

        Parameters:
        -----------
        speed : float
            Forward speed in m/s
        trimAngleDeg : float
            Trim angle in degrees

        Returns:
        --------
        float : Total upward force in Newtons

        '''

        forces = self.computeForcesAtState(speed, trimAngleDeg)
        return forces['buoyancy'].force + forces['planing_lift'].force

    def dragBreakdown(
        self, speedRange: np.ndarray, trimAngleDeg: float = 5.0
    ) -> dict[str, np.ndarray]:

        '''

        Drag breakdown across a speed range at a fixed trim.

        Parameters:
        -----------
        speedRange : np.ndarray
            Array of speeds in m/s
        trimAngleDeg : float
            Fixed trim angle for the sweep

        Returns:
        --------
        dict[str, np.ndarray] : Component name -> array of drag values

        '''

        breakdown: dict[str, list[float]] = {
            'planing_drag': [],
            'form_drag': [],
            'fin_drag': [],
            'total': [],
        }

        for v in speedRange:
            forces = self.computeForcesAtState(v, trimAngleDeg)
            breakdown['planing_drag'].append(forces['planing_drag'].force)
            breakdown['form_drag'].append(forces['form_drag'].force)
            breakdown['fin_drag'].append(forces['fin_drag'].force)
            breakdown['total'].append(
                forces['planing_drag'].force
                + forces['form_drag'].force
                + forces['fin_drag'].force
            )

        return {k: np.array(v) for k, v in breakdown.items()}

    def findEquilibrium(self, speed: float) -> PlaningState:

        '''

        Find equilibrium planing state at a given speed.

        Parameters:
        -----------
        speed : float
            Forward speed in m/s

        Returns:
        --------
        PlaningState : Equilibrium state

        '''

        return self._planing.findPlaningEquilibrium(speed, self._totalWeightN)

    def performanceCurves(
        self,
        minSpeed: float = 0.5,
        maxSpeed: float = 10.0,
        nPoints: int = 50,
    ) -> dict[str, np.ndarray]:

        '''

        Compute equilibrium states across a speed range.

        Parameters:
        -----------
        minSpeed : float
            Minimum speed in m/s
        maxSpeed : float
            Maximum speed in m/s
        nPoints : int
            Number of speed points

        Returns:
        --------
        dict[str, np.ndarray] : Arrays keyed by 'speed', 'trimAngleDeg',
            'wettedLengthM', 'liftToDrag', 'dragN', 'liftN'

        '''

        speeds = np.linspace(minSpeed, maxSpeed, nPoints)
        states = self._planing.sweepSpeed(speeds, self._totalWeightN)

        return {
            'speed': speeds,
            'trimAngleDeg': np.array([s.trimAngleDeg for s in states]),
            'wettedLengthM': np.array([s.wettedLengthM for s in states]),
            'liftToDrag': np.array([s.liftToDrag for s in states]),
            'dragN': np.array([s.dragForceN for s in states]),
            'liftN': np.array([s.liftForceN for s in states]),
        }
