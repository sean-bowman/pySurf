# -- Static Buoyancy Model -- #

'''

Static buoyancy and mass estimation for surfboards.

Computes board mass from material properties, Archimedes buoyancy force,
and equilibrium waterline using root-finding.

Sean Bowman [02/03/2026]

'''

from __future__ import annotations

from scipy.optimize import brentq

from pysurf.Surfboard.SurfPhysics import constants as const
from pysurf.Surfboard.SurfPhysics.geometry.board import BoardGeometry
from pysurf.Surfboard.SurfPhysics.geometry.parameters import SurfboardParameters
from pysurf.Surfboard.SurfPhysics.hydrodynamics.protocols import ForceResult


class BuoyancyModel:

    '''

    Static buoyancy and mass estimation.

    Computes Archimedes buoyancy force and board mass from
    material properties and geometry.

    '''

    def __init__(self, board: BoardGeometry, params: SurfboardParameters) -> None:

        '''

        Initialize buoyancy model.

        Parameters:
        -----------
        board : BoardGeometry
            Board geometry for volume/area calculations
        params : SurfboardParameters
            Board parameters including foam type

        '''

        self._board = board
        self._params = params

        # Cache computed values
        self._totalVolume: float | None = None
        self._surfaceArea: float | None = None
        self._boardMass: float | None = None

    def _getTotalVolume(self) -> float:
        '''Get or compute total board volume [m^3].'''
        if self._totalVolume is None:
            self._totalVolume = self._board.computeVolume()
        return self._totalVolume

    def _getSurfaceArea(self) -> float:
        '''Get or compute total surface area [m^2].'''
        if self._surfaceArea is None:
            # Approximate total surface area as ~2x the bottom wetted area
            # (top + bottom surfaces are similar)
            self._surfaceArea = 2.0 * self._board.computeWettedSurfaceArea()
        return self._surfaceArea

    def estimateBoardMass(self) -> float:

        '''

        Estimate board mass from material properties.

        mass = foamDensity * coreVolume + fiberglassDensity * shellVolume
        where shellVolume = surfaceArea * shellThickness
        and coreVolume = totalVolume - shellVolume

        Returns:
        --------
        float : Board mass in kg

        '''

        if self._boardMass is not None:
            return self._boardMass

        totalVolume = self._getTotalVolume()
        surfaceArea = self._getSurfaceArea()

        # Shell volume from surface area * shell thickness
        shellVolume = surfaceArea * const.glassShellThicknessM
        coreVolume = max(0.0, totalVolume - shellVolume)

        # Foam density based on construction type
        if self._params.foamType == 'eps':
            foamDensity = const.epsFoamDensity
        else:
            foamDensity = const.puFoamDensity

        self._boardMass = foamDensity * coreVolume + const.fiberglassDensity * shellVolume

        return self._boardMass

    def buoyancyForce(self, submergedVolume: float) -> float:

        '''

        Archimedes buoyancy force.

        F_b = rho_water * g * V_submerged

        Parameters:
        -----------
        submergedVolume : float
            Volume of board below waterline [m^3]

        Returns:
        --------
        float : Buoyancy force in Newtons (upward)

        '''

        return const.seawaterDensity * const.gravity * submergedVolume

    def findEquilibriumDraft(self, totalMassKg: float) -> float:

        '''

        Find the draft where buoyancy equals weight.

        Uses scipy.optimize.brentq on:
            f(draft) = rho * g * V_submerged(draft) - totalMass * g

        Parameters:
        -----------
        totalMassKg : float
            Combined mass of board + surfer (or board alone)

        Returns:
        --------
        float : Equilibrium draft in meters (height of waterline above
                the lowest point of the board)

        '''

        totalWeight = totalMassKg * const.gravity

        # Maximum possible draft is the total board height
        # (rocker + thickness)
        maxDraft = self._board.getMaxThicknessM() + 0.3  # generous upper bound

        def residual(draft: float) -> float:
            '''Residual: buoyancy - weight.'''
            subVol = self._board.computeSubmergedVolume(draft, trimAngleDeg=0.0)
            buoyancy = self.buoyancyForce(subVol)
            return buoyancy - totalWeight

        # Check if the board can support the weight at all
        # (fully submerged buoyancy must exceed weight)
        fullBuoyancy = self.buoyancyForce(self._getTotalVolume())
        if fullBuoyancy < totalWeight:
            # Board sinks: return max draft as an indicator
            return maxDraft

        try:
            equilibriumDraft = brentq(residual, 0.0, maxDraft, xtol=1e-6)
        except ValueError:
            # If brentq can't bracket, return an estimate
            equilibriumDraft = maxDraft * 0.5

        return equilibriumDraft

    def paddleDraftWithRider(self, riderMassKg: float = 75.0) -> float:

        '''

        Draft when paddling (board + rider weight supported by buoyancy).

        Parameters:
        -----------
        riderMassKg : float
            Rider mass in kg (default 75)

        Returns:
        --------
        float : Draft in meters

        '''

        boardMass = self.estimateBoardMass()
        return self.findEquilibriumDraft(boardMass + riderMassKg)

    def computeBuoyancyForce(
        self, draft: float, trimAngleDeg: float = 0.0
    ) -> ForceResult:

        '''

        Compute buoyancy force at a given draft and trim.

        Parameters:
        -----------
        draft : float
            Waterplane height in meters
        trimAngleDeg : float
            Pitch angle in degrees

        Returns:
        --------
        ForceResult : Buoyancy force result

        '''

        subVol = self._board.computeSubmergedVolume(draft, trimAngleDeg)
        force = self.buoyancyForce(subVol)
        return ForceResult(
            force=force,
            description=f'Buoyancy: {force:.1f} N (V_sub = {subVol * 1e6:.0f} cm^3)',
        )
