# -- Frictional Drag Model -- #

'''

Frictional drag using flat-plate analogy and ITTC 1957 friction line.

The ITTC 1957 model-ship correlation line is the standard method for
estimating skin friction drag on marine craft. It accounts for turbulent
boundary layer effects at high Reynolds numbers typical of surfing speeds.

References:
-----------
ITTC 1957: Cf = 0.075 / (log10(Re) - 2)^2
ITTC Recommended Procedures: https://ittc.info/media/2021/75-02-02-02.pdf
Falk et al. (2020) -- CFD surfboard drag methodology

Sean Bowman [02/03/2026]

'''

from __future__ import annotations

import math

from pysurf.Surfboard.SurfPhysics import constants as const
from pysurf.Surfboard.SurfPhysics.hydrodynamics.protocols import ForceResult


class FrictionDragModel:

    '''

    Frictional drag using the ITTC 1957 correlation line.

    At surfing speeds (2-8 m/s) with typical board lengths (1.5-2.7 m),
    Reynolds numbers are in the range 3e6 to 2e7, fully turbulent.

    '''

    def computeReynoldsNumber(self, speed: float, characteristicLength: float) -> float:

        '''

        Reynolds number Re = V * L / nu.

        Parameters:
        -----------
        speed : float
            Flow speed in m/s
        characteristicLength : float
            Characteristic length in meters (typically wetted length)

        Returns:
        --------
        float : Reynolds number (dimensionless)

        '''

        if speed <= 0.0 or characteristicLength <= 0.0:
            return 0.0
        return speed * characteristicLength / const.seawaterKinematicViscosity

    def frictionCoefficient(self, reynoldsNumber: float) -> float:

        '''

        Skin friction coefficient from ITTC 1957 or Blasius.

        ITTC 1957 (turbulent, Re > 1e5):
            Cf = 0.075 / (log10(Re) - 2)^2

        Blasius (laminar, Re < 5e5):
            Cf = 1.328 / sqrt(Re)

        Parameters:
        -----------
        reynoldsNumber : float
            Reynolds number

        Returns:
        --------
        float : Friction coefficient (dimensionless)

        '''

        if reynoldsNumber <= 0.0:
            return 0.0

        if reynoldsNumber < 5e5:
            # Laminar: Blasius solution
            return 1.328 / math.sqrt(reynoldsNumber)
        else:
            # Turbulent: ITTC 1957
            logRe = math.log10(reynoldsNumber)
            denominator = logRe - 2.0
            if abs(denominator) < 1e-10:
                return 0.0
            return 0.075 / (denominator * denominator)

    def computeFrictionDrag(
        self, speed: float, wettedArea: float, characteristicLength: float
    ) -> ForceResult:

        '''

        Compute frictional drag force.

        D_f = 0.5 * rho * V^2 * S_wet * Cf

        Parameters:
        -----------
        speed : float
            Board speed in m/s
        wettedArea : float
            Wetted surface area in m^2
        characteristicLength : float
            Wetted length in m (for Reynolds number calculation)

        Returns:
        --------
        ForceResult : Friction drag force

        '''

        if speed <= 0.0 or wettedArea <= 0.0:
            return ForceResult(force=0.0, description='Friction drag: 0 N (no speed/area)')

        re = self.computeReynoldsNumber(speed, characteristicLength)
        cf = self.frictionCoefficient(re)

        drag = 0.5 * const.seawaterDensity * speed * speed * wettedArea * cf

        return ForceResult(
            force=drag,
            description=f'Friction drag: {drag:.2f} N (Re = {re:.2e}, Cf = {cf:.6f})',
        )

    def computeFormDrag(
        self, speed: float, frontalArea: float, formFactor: float = 0.1
    ) -> ForceResult:

        '''

        Compute form (pressure) drag.

        D_p = 0.5 * rho * V^2 * A_frontal * Cd_form

        The form factor is empirical, typically 0.05-0.15 for streamlined
        bodies like surfboards. Default 0.1 is a moderate estimate.

        Parameters:
        -----------
        speed : float
            Board speed in m/s
        frontalArea : float
            Frontal (cross-sectional) area in m^2
        formFactor : float
            Form drag coefficient (default 0.1)

        Returns:
        --------
        ForceResult : Form drag force

        '''

        if speed <= 0.0 or frontalArea <= 0.0:
            return ForceResult(force=0.0, description='Form drag: 0 N')

        drag = 0.5 * const.seawaterDensity * speed * speed * frontalArea * formFactor

        return ForceResult(
            force=drag,
            description=f'Form drag: {drag:.2f} N (Cd = {formFactor:.3f})',
        )
