# -- Hydrodynamic Force Protocols -- #

'''

Abstract protocols and result dataclasses for hydrodynamic force models.

Sean Bowman [02/03/2026]

'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pysurf.Surfboard.SurfPhysics.geometry.board import BoardGeometry


@dataclass
class ForceResult:

    '''

    Result of a single force computation.

    Parameters:
    -----------
    force : float
        Force magnitude in Newtons
    moment : float
        Moment about center of gravity in N*m
    description : str
        Human-readable description of the force component

    '''

    force: float
    moment: float = 0.0
    description: str = ''


@dataclass
class PlaningState:

    '''

    Planing equilibrium state at a given speed.

    Parameters:
    -----------
    speed : float
        Forward speed in m/s
    trimAngleDeg : float
        Pitch angle in degrees
    wettedLengthM : float
        Wetted length in meters
    draftM : float
        Draft in meters
    liftForceN : float
        Total lift (buoyancy + hydrodynamic) in Newtons
    dragForceN : float
        Total drag in Newtons
    liftToDrag : float
        Lift-to-drag ratio (dimensionless)
    isPlaning : bool
        True if the board is in the planing regime

    '''

    speed: float
    trimAngleDeg: float
    wettedLengthM: float
    draftM: float
    liftForceN: float
    dragForceN: float
    liftToDrag: float
    isPlaning: bool


class ForceModel(Protocol):
    '''Protocol for hydrodynamic force models.'''

    def computeLift(
        self, speed: float, trimAngleDeg: float, board: BoardGeometry
    ) -> ForceResult:
        '''Compute lift force at given speed and trim.'''
        ...

    def computeDrag(
        self, speed: float, trimAngleDeg: float, board: BoardGeometry
    ) -> ForceResult:
        '''Compute drag force at given speed and trim.'''
        ...
