
# -- pySurf: Serre-Green-Naghdi Solver -- #

'''

The Serre-Green-Naghdi (SGN) equations in 1D: a free-surface model of water waves over a variable
bed, solved for the water depth h(x, t) and the depth-averaged velocity u(x, t).

SGN keeps the full nonlinearity of the shallow-water equations and adds the pressure that the
vertical acceleration of the water column carries, which gives the waves dispersion. With the
horizontal velocity uniform over the depth and the vertical velocity linear in height,

    w = u b_x - (z - b) u_x

the depth-integrated mass and momentum balances are

    h_t + (h u)_x = 0
    (h u)_t + (h u^2 + g h^2 / 2)_x + g h b_x + (h^2 Gamma / 2 + h^3 Psi / 3)_x + b_x (h Gamma + h^2 Psi / 2) = 0

    Gamma = b_x (u_t + u u_x) + u^2 b_xx
    Psi   = u_x^2 - u_xt - u u_xx

The last two terms of the momentum balance are the non-hydrostatic pressure integrated over the
depth and its push on the sloping bed. They contain u_t, so the velocity update is implicit.

The solution splits the momentum balance in two (Bonneton et al. 2011):

    hyperbolic part     the nonlinear shallow-water equations, as finite volumes: MUSCL slopes,
                        an HLL flux, and the hydrostatic reconstruction of Audusse et al. (2004),
                        which holds a lake at rest exactly and keeps the depth non-negative where
                        the water runs up a dry beach
    dispersive part     an extra acceleration phi, the difference between the SGN and the
                        shallow-water u_t, from a tridiagonal system: h phi + N(phi) = N(a) - N0,
                        where a = u u_x + g zeta_x is the shallow-water acceleration, N the
                        u_t-terms of the non-hydrostatic pressure and N0 the rest

and advances both together with the third-order strong-stability-preserving Runge-Kutta scheme.

Breaking is the switch of Tissier et al. (2012). A wave whose front is steeper than 30 degrees,
and strong enough to be a breaking bore by the termination criterion below, starts breaking; over a zone centered on its front, 2.5 roller lengths wide with the roller 2.9
times the wave height, the dispersive part is switched off. There the shallow-water equations
carry the front as a shock (a bore), which dissipates energy as a bore does. The wave stops
breaking when the bore's Froude number, from the depths behind and ahead of it, falls below 1.3 (set it to zero to keep a wave breaking once
it starts), and has stayed below it for 5 sqrt(h / g) (the transition time of Kennedy et al. 2000, which keeps a
bore at the threshold from switching off and on), unless the front has reached the waterline or
water too shallow to carry dispersion: there is nothing to switch back to, and on a plane beach a
broken wave breaks on to the shoreline.
These are the published values; Tissier et al. took the angle from Cienfuegos et al. (2010), who
fit it to laboratory breaking for their SGN model. The solver follows one front at a time: before
breaking, the one under the highest wet crest; once breaking, the same front from step to step.
That suits a single solitary wave; periodic waves need the dissipation-based front tracking Tissier
et al. describe.

The dispersive part is also off in water shallower than `dispersiveDepth` and in dry cells, where
it would act on a sheet a few cells thick at the waterline.

References:
-----------
Serre, F. (1953) -- Contribution a l'etude des ecoulements permanents et variables dans les canaux,
    La Houille Blanche 8, 374-388.
Green, A.E. & Naghdi, P.M. (1976) -- A derivation of equations for wave propagation in water of
    variable depth, J. Fluid Mech. 78(2), 237-246.
Audusse, E., Bouchut, F., Bristeau, M.-O., Klein, R. & Perthame, B. (2004) -- A fast and stable
    well-balanced scheme with hydrostatic reconstruction for shallow water flows, SIAM J. Sci.
    Comput. 25(6), 2050-2065.
Bonneton, P., Chazel, F., Lannes, D., Marche, F. & Tissier, M. (2011) -- A splitting approach for
    the fully nonlinear and weakly dispersive Green-Naghdi model, J. Comput. Phys. 230, 1479-1498.
Tissier, M., Bonneton, P., Marche, F., Chazel, F. & Lannes, D. (2012) -- A new approach to handle
    wave breaking in fully non-linear Boussinesq models, Coastal Engineering 67, 54-66.
Kennedy, A.B., Chen, Q., Kirby, J.T. & Dalrymple, R.A. (2000) -- Boussinesq modeling of wave
    transformation, breaking, and runup. I: 1D, J. Waterway, Port, Coastal, Ocean Eng. 126(1), 39-47.
Wei, G., Kirby, J.T. & Sinha, A. (1999) -- Generation of waves in Boussinesq models using a source
    function method, Coastal Engineering 36, 271-299.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

tests/testSgnSolver.py checks each part against a closed form: a lake at rest over a partly dry
beach stays at rest to rounding, volume is conserved to rounding, a dam break onto a dry bed with
the dispersive part off converges to Ritter's solution, a small standing wave oscillates at the
SGN linear frequency g k^2 d / (1 + (k d)^2 / 3), distinct from the shallow-water and Airy
frequencies, and the exact SGN solitary wave keeps its height and speed on a flat bed.
The measured errors: the standing-wave period within 0.01 percent at 80 cells per wavelength, the
solitary wave's height and speed within 1e-4 at 40 cells per depth and falling by four when the
cells halve, the dam break 0.12 percent in L1 at 0.05 m cells and first order, as a solution with
kinks allows. tests/testValidationSurfaceWave.py runs the solitary wave on a slope against Grilli
et al. (1997) and the non-breaking runup against Synolakis (1987); its results are recorded in
`pysurf.SurfaceBreakingWave`.

Sean Bowman [10/11/2026]

'''

# Standard library imports
from __future__ import annotations

import math
from dataclasses import dataclass

# Third-party imports
import numpy as np
from scipy.linalg import solve_banded
from scipy.signal import find_peaks

# Local imports
from pysurf import constants, waveTheory
from pysurf.bathymetry import Bathymetry
from pysurf.waveBreaking import WaveFront, waveFront

#----------------------------------------------------------------------#
# -- Configuration -- #
#----------------------------------------------------------------------#

@dataclass
class SgnConfig:

    '''

    Numerical and breaking parameters of an SGN run. SI units throughout.

    Parameters:
    -----------
    cellSize : float
        Grid spacing [m]
    gravity : float
        Gravitational acceleration [m/s^2]
    cflNumber : float
        Courant number on the shallow-water wave speed |u| + sqrt(g h)
    dryDepth : float
        Depth below which a cell is dry [m]
    dispersive : bool
        Include the dispersive part; False solves the nonlinear shallow-water equations
    dispersiveDepth : float
        Depth below which the dispersive part is off [m]
    breakingAngle : float
        Front slope at which a wave starts breaking [deg]; Tissier et al. (2012) use 30
    terminationFroude : float
        Bore Froude number below which a wave stops breaking; Tissier et al. (2012) use 1.3
    rollerLength : float
        Roller length over wave height, l_r / H; Haller and Catalan (2010) measured 2.9
    breakingZone : float
        Width of the zone without dispersion, in roller lengths; Tissier et al. (2012) use 2.5
    dispersionAlpha : float
        Dispersion parameter alpha; 1 is the original equations, 1.159 the improved dispersion of
        Bonneton et al. (2011)
    frontMinHeight : float
        Smallest crest-to-trough height that counts as a wave front, and the smallest climb back
        that ends one [m]; 0 takes 1 percent of the deepest still water
    spongeLeft, spongeRight : float
        Width of the absorbing layer at each end [m]; 0 for none
    spongeRate : float
        Largest damping rate in the absorbing layers, at the ends [1/s]

    '''

    cellSize: float
    gravity: float = constants.gravity
    cflNumber: float = 0.3
    dryDepth: float = 1e-6
    dispersive: bool = True
    dispersiveDepth: float = 0.0
    breakingAngle: float = 30.0
    terminationFroude: float = 1.3
    rollerLength: float = 2.9
    breakingZone: float = 2.5
    dispersionAlpha: float = 1.0
    frontMinHeight: float = 0.0
    spongeLeft: float = 0.0
    spongeRight: float = 0.0
    spongeRate: float = 0.0

#----------------------------------------------------------------------#
# -- Hyperbolic Part: Shallow-Water Finite Volumes -- #
#----------------------------------------------------------------------#

def mirrored(values: np.ndarray, ghosts: int, sign: float = 1.0) -> np.ndarray:

    '''

    The array extended by `ghosts` cells at each end, mirrored about the end faces: the reflection
    a solid wall imposes. Pass sign = -1 for a quantity that changes sign in the mirror, such as a
    velocity normal to the wall.

    '''

    left = sign * values[ghosts - 1::-1]
    right = sign * values[:-ghosts - 1:-1]

    return np.concatenate([left, values, right])

def limitedSlope(values: np.ndarray) -> np.ndarray:

    '''

    Monotonized central (MC) limited differences of an extended array, per cell, zero at the two
    end cells: minmod(2 a, 2 b, (a + b) / 2) of the backward and forward differences a and b.

    '''

    backward = np.diff(values)[:-1]
    forward = np.diff(values)[1:]
    central = 0.5 * (backward + forward)
    sameSign = (backward * forward) > 0.0
    magnitude = np.minimum(np.minimum(2.0 * np.abs(backward), 2.0 * np.abs(forward)), np.abs(central))
    slope = np.zeros_like(values)
    slope[1:-1] = np.where(sameSign, np.sign(central) * magnitude, 0.0)

    return slope

def hllFlux(depthLeft: np.ndarray, velocityLeft: np.ndarray, depthRight: np.ndarray, velocityRight: np.ndarray,
            gravity: float, dryDepth: float) -> tuple[np.ndarray, np.ndarray]:

    '''

    HLL flux of the shallow-water equations at faces, with the dry-bed wave speeds of Toro (2001)
    on a side that holds no water.

    Returns:
    --------
    tuple : (mass flux [m^2/s], momentum flux [m^3/s^2])

    '''

    hL, hR = depthLeft, depthRight
    dryL, dryR = hL <= dryDepth, hR <= dryDepth
    uL = np.where(dryL, 0.0, velocityLeft)
    uR = np.where(dryR, 0.0, velocityRight)
    cL, cR = np.sqrt(gravity * hL), np.sqrt(gravity * hR)

    speedL = np.minimum(uL - cL, uR - cR)
    speedR = np.maximum(uL + cL, uR + cR)
    speedL = np.where(dryL, uR - 2.0 * cR, np.where(dryR, uL - cL, speedL))
    speedR = np.where(dryL, uR + cR, np.where(dryR, uL + 2.0 * cL, speedR))

    massL, massR = hL * uL, hR * uR
    momentumL = hL * uL**2 + 0.5 * gravity * hL**2
    momentumR = hR * uR**2 + 0.5 * gravity * hR**2

    span = speedR - speedL
    safeSpan = np.where(span > 0.0, span, 1.0)
    massStar = (speedR * massL - speedL * massR + speedL * speedR * (hR - hL)) / safeSpan
    momentumStar = (speedR * momentumL - speedL * momentumR + speedL * speedR * (massR - massL)) / safeSpan

    mass = np.where(speedL >= 0.0, massL, np.where(speedR <= 0.0, massR, massStar))
    momentum = np.where(speedL >= 0.0, momentumL, np.where(speedR <= 0.0, momentumR, momentumStar))
    bothDry = dryL & dryR

    return np.where(bothDry, 0.0, mass), np.where(bothDry, 0.0, momentum)

def shallowWaterRates(depth: np.ndarray, discharge: np.ndarray, bed: np.ndarray, cellSize: float,
                      gravity: float, dryDepth: float) -> tuple[np.ndarray, np.ndarray]:

    '''

    Rates of change of depth and discharge from the nonlinear shallow-water equations, second-order
    finite volumes with the hydrostatic reconstruction of Audusse et al. (2004), solid walls at
    both ends.

    Depth, surface elevation and velocity are reconstructed linearly in each cell with MC-limited
    slopes, flat in a cell that is dry or next to a dry one. At each face the bed is raised to the
    higher of its two reconstructed values and the depths are cut down to it, so a wet cell next to
    a higher dry one sees no water flow into the bank; the cut depths enter the flux, and the
    hydrostatic pressure they leave out returns as a correction on each side of the face. Together
    with the centered bed-slope source, the momentum rate of a lake at rest is zero to rounding.

    Parameters:
    -----------
    depth, discharge : np.ndarray
        Cell-average depth h [m] and discharge h u [m^2/s]
    bed : np.ndarray
        Bed elevation at the cell centers [m]
    cellSize : float
        Grid spacing [m]
    gravity : float
        Gravitational acceleration [m/s^2]
    dryDepth : float
        Depth below which a cell is dry [m]

    Returns:
    --------
    tuple : (dh/dt [m/s], d(hu)/dt [m^2/s^2])

    '''

    g, dx = gravity, cellSize
    h = mirrored(depth, 2)
    q = mirrored(discharge, 2, -1.0)
    b = mirrored(bed, 2)
    wet = h > dryDepth
    u = np.where(wet, q / np.where(wet, h, 1.0), 0.0)
    surface = h + b

    # Flat reconstruction in and next to dry cells, so the waterline stays first order and positive
    flat = ~wet | ~np.roll(wet, 1) | ~np.roll(wet, -1)
    slopes = [np.where(flat, 0.0, limitedSlope(field)) for field in (h, surface, u)]
    sH, sZ, sU = slopes

    # Face j + 1/2 sits between extended cells j and j + 1, for j = 1 .. n + 1
    left, right = slice(1, -2), slice(2, -1)
    hL = np.maximum(h[left] + 0.5 * sH[left], 0.0)
    hR = np.maximum(h[right] - 0.5 * sH[right], 0.0)
    zL = surface[left] + 0.5 * sZ[left]
    zR = surface[right] - 0.5 * sZ[right]
    uL = u[left] + 0.5 * sU[left]
    uR = u[right] - 0.5 * sU[right]
    bL, bR = zL - hL, zR - hR

    bStar = np.maximum(bL, bR)
    hLStar = np.maximum(zL - bStar, 0.0)
    hRStar = np.maximum(zR - bStar, 0.0)
    massFlux, momentumFlux = hllFlux(hLStar, uL, hRStar, uR, g, dryDepth)

    # Real cell i has its left face at index i and its right face at index i + 1
    rightFace, leftFace = slice(1, None), slice(None, -1)
    momentumRight = momentumFlux[rightFace] + 0.5 * g * (hL[rightFace]**2 - hLStar[rightFace]**2)
    momentumLeft = momentumFlux[leftFace] + 0.5 * g * (hR[leftFace]**2 - hRStar[leftFace]**2)
    bedSource = -0.5 * g * (hR[leftFace] + hL[rightFace]) * (bL[rightFace] - bR[leftFace])

    depthRate = -(massFlux[rightFace] - massFlux[leftFace]) / dx
    dischargeRate = (-(momentumRight - momentumLeft) + bedSource) / dx

    return depthRate, dischargeRate

#----------------------------------------------------------------------#
# -- Dispersive Part -- #
#----------------------------------------------------------------------#

def dispersiveAcceleration(depth: np.ndarray, velocity: np.ndarray, bed: np.ndarray, cellSize: float,
                           gravity: float, active: np.ndarray, alpha: float = 1.0) -> np.ndarray:

    '''

    The dispersive acceleration phi of the SGN equations: the SGN u_t is the shallow-water u_t
    plus phi [m/s^2].

    With a = u u_x + g zeta_x, phi solves the tridiagonal system h phi + alpha N(phi) = N(a) - N0 with

        N(v) = (h^2 b_x v / 2 - h^3 v_x / 3)_x + h b_x^2 v - h^2 b_x v_x / 2
        N0   = (h^2 Gamma0 / 2 + h^3 Psi0 / 3)_x + b_x (h Gamma0 + h^2 Psi0 / 2)
        Gamma0 = b_x u u_x + u^2 b_xx,   Psi0 = u_x^2 - u u_xx

    the parts of the non-hydrostatic pressure with and without u_t. The dispersion parameter alpha
    scales the operator on the left only: written for phi, Bonneton et al.'s (2011) form
    (I + alpha h T(1 / h))[(h u)_t + (h u^2)_x + (alpha - 1) / alpha g h zeta_x] + g h zeta_x / alpha +
    h Q1 = 0 leaves the right side as it is at alpha = 1. Derivatives are second-order
    central differences; the divergence in N is taken across faces, which makes the operator
    symmetric and diagonally dominant on a flat bed. Walls at both ends mirror the fields, with phi
    and u changing sign. Cells outside `active` have phi = 0.

    Parameters:
    -----------
    depth, velocity, bed : np.ndarray
        Cell-center depth [m], depth-averaged velocity [m/s] and bed elevation [m]
    cellSize : float
        Grid spacing [m]
    gravity : float
        Gravitational acceleration [m/s^2]
    active : np.ndarray
        Boolean mask of the cells the dispersive part applies to
    alpha : float
        Dispersion parameter; 1 for the original equations

    '''

    dx = cellSize
    n = len(depth)
    phi = np.zeros(n)
    if not np.any(active):
        return phi

    h = mirrored(depth, 1)
    u = mirrored(velocity, 1, -1.0)
    b = mirrored(bed, 1)
    zeta = h + b

    def centralDifference(field: np.ndarray) -> np.ndarray:
        return (field[2:] - field[:-2]) / (2.0 * dx)

    def secondDifference(field: np.ndarray) -> np.ndarray:
        return (field[2:] - 2.0 * field[1:-1] + field[:-2]) / dx**2

    hc, uc = h[1:-1], u[1:-1]
    bx, bxx = centralDifference(b), secondDifference(b)
    ux, uxx = centralDifference(u), secondDifference(u)
    acceleration = uc * ux + gravity * centralDifference(zeta)

    gamma0 = bx * uc * ux + uc**2 * bxx
    psi0 = ux**2 - uc * uxx
    pressureMoment = mirrored(0.5 * hc**2 * gamma0 + hc**3 * psi0 / 3.0, 1)
    remainder = centralDifference(pressureMoment) + bx * (hc * gamma0 + 0.5 * hc**2 * psi0)

    # Coefficients of N(v) = (A v - B v_x)_x + C v - A v_x, with A and B averaged to the faces
    A = mirrored(0.5 * hc**2 * bx, 1, -1.0)
    B = mirrored(hc**3 / 3.0, 1)
    C = hc * bx**2
    faceA = 0.5 * (A[1:] + A[:-1])
    faceB = 0.5 * (B[1:] + B[:-1])
    aRight, aLeft = faceA[1:], faceA[:-1]
    bRight, bLeft = faceB[1:], faceB[:-1]

    lower = -aLeft / (2.0 * dx) - bLeft / dx**2 + A[1:-1] / (2.0 * dx)
    upper = aRight / (2.0 * dx) - bRight / dx**2 - A[1:-1] / (2.0 * dx)
    diagonalN = aRight / (2.0 * dx) + bRight / dx**2 - aLeft / (2.0 * dx) + bLeft / dx**2 + C

    # phi and a change sign across a wall: the ghost value is minus the end cell's
    diagonalN[0] -= lower[0]
    diagonalN[-1] -= upper[-1]
    aExtended = np.concatenate([[-acceleration[0]], acceleration, [-acceleration[-1]]])
    operatorOfA = lower * aExtended[:-2] + diagonalN * acceleration + upper * aExtended[2:]
    operatorOfA[0] = diagonalN[0] * acceleration[0] + upper[0] * acceleration[1]
    operatorOfA[-1] = lower[-1] * acceleration[-2] + diagonalN[-1] * acceleration[-1]

    rightHandSide = np.where(active, operatorOfA - remainder, 0.0)
    diagonal = np.where(active, hc + alpha * diagonalN, 1.0)
    lower = np.where(active, alpha * lower, 0.0)
    upper = np.where(active, alpha * upper, 0.0)

    banded = np.zeros((3, n))
    banded[0, 1:] = upper[:-1]
    banded[1, :] = diagonal
    banded[2, :-1] = lower[1:]
    phi = solve_banded((1, 1), banded, rightHandSide)

    return np.where(active, phi, 0.0)

def boreFroudeNumber(depthBehind: float, depthAhead: float) -> float:

    '''

    Froude number of a bore relative to the water ahead of it, from the depths on either side by
    mass and momentum conservation: Fr = sqrt(((2 h2 / h1 + 1)^2 - 1) / 8). Infinite onto a dry bed.

    '''

    if depthAhead <= 0.0:
        return math.inf

    return math.sqrt(((2.0 * depthBehind / depthAhead + 1.0)**2 - 1.0) / 8.0)

#----------------------------------------------------------------------#
# -- Wave Generation -- #
#----------------------------------------------------------------------#

class WaveSource:

    '''

    An internal wave generator: a mass source of Gaussian shape that radiates a regular wave of the
    requested height both ways from its center (Wei, Kirby and Sinha 1999).

    The source adds S(x, t) = A exp(-gamma (x - x_s)^2) sin(omega t) to the mass balance. For the
    linearized SGN equations over a flat bed of depth d, with b = d^2 / 3 and the dispersion function

        D(k) = g d k^2 (1 + (alpha - 1) b k^2) - omega^2 (1 + alpha b k^2)

    the wave radiated each way has amplitude a = omega (1 + alpha b k^2) S_hat(k) / |D'(k)| at the
    wavenumber where D vanishes, with S_hat the source's Fourier transform. That fixes A for a target
    height H = 2a. Without dispersion it reduces to a total source strength of 2 a c, the volume
    flux of two waves leaving at speed c. The source width follows Wei et al.: gamma = 80 /
    (delta^2 L^2), with delta = 0.5. The stroke ramps up over the first periods.

    Parameters:
    -----------
    height : float
        Wave height H [m]
    period : float
        Wave period T [s]
    depth : float
        Still water depth at the source d [m]
    centerX : float
        Source center x_s [m]
    alpha : float
        Dispersion parameter of the equations the source drives
    rampPeriods : float
        Periods over which the source ramps from zero
    widthShare : float
        Width parameter delta, as a share of the wavelength

    '''

    def __init__(self, height: float, period: float, depth: float, centerX: float, alpha: float = 1.0,
                 rampPeriods: float = 3.0, widthShare: float = 0.5, gravity: float = constants.gravity) -> None:

        self.height, self.period, self.depth, self.centerX = height, period, depth, centerX
        self.rampPeriods = rampPeriods
        self.angularFrequency = 2.0 * math.pi / period                                 # [rad/s]
        self.wavenumber = waveTheory.serreWavenumber(self.angularFrequency, depth, gravity, alpha)  # [rad/m]
        self.wavelength = 2.0 * math.pi / self.wavenumber                               # [m]
        self.gamma = 80.0 / (widthShare**2 * self.wavelength**2)                        # [1/m^2]

        k, omega, b = self.wavenumber, self.angularFrequency, depth**2 / 3.0
        slope = 2.0 * gravity * depth * k + 4.0 * gravity * depth * (alpha - 1.0) * b * k**3 - 2.0 * omega**2 * alpha * b * k
        transform = math.sqrt(math.pi / self.gamma) * math.exp(-k**2 / (4.0 * self.gamma))
        self.amplitude = 0.5 * height * abs(slope) / (omega * (1.0 + alpha * b * k**2) * transform)   # [m/s]

    def rate(self, x: np.ndarray, time: float) -> np.ndarray:

        '''The mass source at each position and time, d h / d t [m/s].'''

        ramp = min(1.0, time / (self.rampPeriods * self.period)) if self.rampPeriods > 0.0 else 1.0
        shape = np.exp(-self.gamma * (x - self.centerX)**2)

        return self.amplitude * ramp * math.sin(self.angularFrequency * time) * shape

@dataclass
class BreakingFront:

    '''

    A wave front the solver is treating as breaking.

    Parameters:
    -----------
    x : float
        Steepest point of the front [m]
    halfWidth : float
        Half the width of its zone without dispersion [m]
    started : float
        Time it started breaking [s]
    weakSince : float | None
        Time its bore Froude number fell below the termination value and has stayed there [s]

    '''

    x: float
    halfWidth: float
    started: float
    weakSince: float | None = None

#----------------------------------------------------------------------#
# -- Solver -- #
#----------------------------------------------------------------------#

class SgnSolver:

    '''

    Time stepping of the SGN equations over a bed profile, with the breaking switch.

    Parameters:
    -----------
    bathymetry : Bathymetry
        The bed and its still water level; the grid spans its full length, with a solid wall at
        each end
    config : SgnConfig
        Grid spacing, time step and breaking parameters
    source : WaveSource | None
        Internal wave generator

    '''

    def __init__(self, bathymetry: Bathymetry, config: SgnConfig, source: WaveSource | None = None) -> None:

        self.bathymetry = bathymetry
        self.config = config
        cells = max(4, int(round((bathymetry.endX - bathymetry.startX) / config.cellSize)))
        self.cellSize = (bathymetry.endX - bathymetry.startX) / cells      # [m]
        self.x = bathymetry.startX + (np.arange(cells) + 0.5) * self.cellSize
        self.bed = np.asarray(bathymetry.elevation(self.x), dtype = float)

        self.depth = np.maximum(bathymetry.stillWaterLevel - self.bed, 0.0)
        self.discharge = np.zeros(cells)
        self.time = 0.0                     # [s]
        self.steps = 0
        self.timeStep = 0.0                 # [s]

        self.breaking = False
        self.front: WaveFront | None = None
        self.breakingZone = np.zeros(cells, dtype = bool)
        self.breakingFronts: list[BreakingFront] = []
        # Fronts that started breaking in the last step, (x [m], still water depth [m], front height
        # [m]), and that stopped, (x [m], still water depth [m], bore Froude number)
        self.startedFronts: list[tuple[float, float, float]] = []
        self.stoppedFronts: list[tuple[float, float, float]] = []

        self.source = source
        self.stillDepth = np.maximum(bathymetry.stillWaterLevel - self.bed, 0.0)    # [m]
        self._frontMinHeight = config.frontMinHeight or 0.01 * float(np.max(self.stillDepth))
        self.spongeRate = self._spongeProfile()                                    # [1/s]

    # -- State -- #
    @property
    def wet(self) -> np.ndarray:
        '''Boolean mask of the cells that hold water.'''
        return self.depth > self.config.dryDepth

    @property
    def velocity(self) -> np.ndarray:
        '''Depth-averaged velocity u [m/s], zero in dry cells.'''
        wet = self.wet
        return np.where(wet, self.discharge / np.where(wet, self.depth, 1.0), 0.0)

    @property
    def surface(self) -> np.ndarray:
        '''Free-surface elevation zeta = b + h [m]; the bed in dry cells.'''
        return self.bed + self.depth

    @property
    def volume(self) -> float:
        '''Water volume per unit crest width [m^2].'''
        return float(np.sum(self.depth) * self.cellSize)

    @property
    def dispersiveCells(self) -> np.ndarray:
        '''Boolean mask of the cells the dispersive part applies to.'''
        config = self.config
        if not config.dispersive:
            return np.zeros_like(self.wet)
        return self.wet & (self.depth > config.dispersiveDepth) & ~self.breakingZone

    def setState(self, depth: np.ndarray, velocity: np.ndarray) -> None:

        '''

        Set the depth and velocity of every cell.

        Raises:
        -------
        ValueError : If either array does not match the grid, or a depth is negative.

        '''

        depth, velocity = np.asarray(depth, dtype = float), np.asarray(velocity, dtype = float)
        if depth.shape != self.x.shape or velocity.shape != self.x.shape:
            raise ValueError(f'Depth and velocity must have one value per cell ({self.x.size}).')
        if np.any(depth < 0.0):
            raise ValueError('Depth cannot be negative.')

        self.depth = depth.copy()
        self.discharge = np.where(depth > self.config.dryDepth, depth * velocity, 0.0)

    # -- Private Helper Methods -- #
    def _spongeProfile(self) -> np.ndarray:

        '''

        Damping rate of the absorbing layers: zero at their inner edge, rising with the square of the
        distance into the layer to the configured rate at the end of the domain [1/s].

        '''

        config, x = self.config, self.x
        rate = np.zeros_like(x)
        start, end = self.bathymetry.startX, self.bathymetry.endX
        if config.spongeLeft > 0.0:
            share = np.clip((start + config.spongeLeft - x) / config.spongeLeft, 0.0, 1.0)
            rate = np.maximum(rate, config.spongeRate * share**2)
        if config.spongeRight > 0.0:
            share = np.clip((x - (end - config.spongeRight)) / config.spongeRight, 0.0, 1.0)
            rate = np.maximum(rate, config.spongeRate * share**2)

        return rate

    def _rates(self, depth: np.ndarray, discharge: np.ndarray, active: np.ndarray,
               time: float) -> tuple[np.ndarray, np.ndarray]:

        '''

        Rates of depth and discharge: shallow-water finite volumes, plus h phi where dispersive, the
        wave source, and the absorbing layers relaxing the water toward rest at still water level.

        '''

        config = self.config
        depthRate, dischargeRate = shallowWaterRates(depth, discharge, self.bed, self.cellSize,
                                                     config.gravity, config.dryDepth)
        if np.any(active):
            wet = depth > config.dryDepth
            velocity = np.where(wet, discharge / np.where(wet, depth, 1.0), 0.0)
            phi = dispersiveAcceleration(depth, velocity, self.bed, self.cellSize, config.gravity,
                                         active & wet, config.dispersionAlpha)
            dischargeRate = dischargeRate + depth * phi
        if self.source is not None:
            depthRate = depthRate + self.source.rate(self.x, time)
        if config.spongeRate > 0.0:
            depthRate = depthRate - self.spongeRate * (depth - self.stillDepth)
            dischargeRate = dischargeRate - self.spongeRate * discharge

        return depthRate, dischargeRate

    def _clean(self, depth: np.ndarray, discharge: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        '''Clip rounding-level negative depths and remove the discharge of dry cells.'''

        depth = np.maximum(depth, 0.0)

        return depth, np.where(depth > self.config.dryDepth, discharge, 0.0)

    def _detectFronts(self, wet: np.ndarray, surface: np.ndarray) -> list[WaveFront]:

        '''

        Every wave front on the surface: one per crest that stands at least the minimum front height
        above its surroundings, running shoreward from it.

        '''

        if not np.any(wet):
            return []
        low = float(np.min(surface[wet])) - 1.0
        # Padded with low ends, so a crest that runs to a wall still counts as a peak
        level = np.concatenate([[low], np.where(wet, surface, low), [low]])
        crests, _ = find_peaks(level, prominence = self._frontMinHeight)
        crests = crests - 1
        fronts = []
        for crest in crests:
            front = waveFront(self.x, surface, wet, crestIndex = int(crest), minimumRise = self._frontMinHeight)
            if front is not None and surface[front.crestIndex] - surface[front.troughIndex] >= self._frontMinHeight:
                fronts.append(front)

        return fronts

    def _updateBreaking(self) -> None:

        '''

        Update which wave fronts are breaking, and the zone without dispersion around each.

        Every front is checked each step. A front already breaking, found again where it should have
        travelled, keeps breaking until its bore's Froude number has stayed below the
        termination value for 5 sqrt(h / g), the transition time Kennedy et al. (2000) give breaking,
        with h the still water depth under its crest: a bore at the threshold dips under it and back
        from step to step, and ending its breaking at the first dip would switch it off and on every
        few steps. A front that ends at the waterline, or stands over water too shallow to carry
        dispersion, has nothing to switch back to and keeps breaking. A new front starts breaking
        when it is steeper than the breaking angle and strong enough to pass the termination test.

        '''

        config = self.config
        wet, surface, x = self.wet, self.surface, self.x
        fronts = self._detectFronts(wet, surface)
        previous = self.breakingFronts
        current, started, stopped = [], [], []
        heights = [surface[front.crestIndex] - surface[front.troughIndex] for front in fronts]

        # Each breaking front claims the tallest front whose span, crest to trough, holds where the
        # breaking front should be now: carried shoreward at the long-wave speed under it. Matching
        # on the span rather than the steepest point keeps a front with its own wave when its
        # steepest point jumps as a shock forms, and away from a smaller wave travelling seaward.
        owner = {}
        for old in previous:
            predicted = old.x + math.sqrt(config.gravity * max(float(np.interp(old.x, x, self.depth)), 0.0)) * self.timeStep
            tolerance = max(old.halfWidth, 10.0 * self.cellSize)
            candidates = [i for i, front in enumerate(fronts) if i not in owner
                          and x[front.crestIndex] - tolerance <= predicted <= x[front.troughIndex] + tolerance]
            if candidates:
                owner[max(candidates, key = lambda i: heights[i])] = old

        for index, front in enumerate(fronts):
            steepestX = float(x[front.steepestIndex])
            halfWidth = max(0.5 * config.breakingZone * config.rollerLength * heights[index], self.cellSize)
            froude = boreFroudeNumber(self.depth[front.crestIndex], self.depth[front.troughIndex])
            stillDepth = float(self.stillDepth[front.crestIndex])

            if index in owner:
                old = owner[index]
                nextCell = min(front.troughIndex + 1, len(wet) - 1)
                weakSince = None
                if wet[nextCell] and stillDepth > config.dispersiveDepth and froude < config.terminationFroude:
                    weakSince = old.weakSince if old.weakSince is not None else self.time
                    if self.time - weakSince >= 5.0 * math.sqrt(stillDepth / config.gravity):
                        stopped.append((steepestX, stillDepth, froude))
                        continue
                current.append(BreakingFront(x = steepestX, halfWidth = halfWidth, started = old.started,
                                             weakSince = weakSince))
            elif front.angle >= config.breakingAngle and froude >= config.terminationFroude:
                current.append(BreakingFront(x = steepestX, halfWidth = halfWidth, started = self.time))
                started.append((steepestX, stillDepth, float(heights[index])))

        zone = np.zeros_like(wet)
        for breakingFront in current:
            zone |= np.abs(x - breakingFront.x) <= breakingFront.halfWidth
        self.breakingFronts, self.breakingZone = current, zone
        self.startedFronts, self.stoppedFronts = started, stopped
        self.breaking = bool(current)
        # The front under the highest crest, for callers following a single wave
        highest = waveFront(x, surface, wet)
        if highest is not None:
            self.front = highest

    # -- Public Methods -- #
    def stableTimeStep(self) -> float:

        '''CFL time step on the fastest shallow-water wave, |u| + sqrt(g h) [s].'''

        wet = self.wet
        if not np.any(wet):
            raise ValueError('No cell holds water.')
        speed = np.abs(self.velocity[wet]) + np.sqrt(self.config.gravity * self.depth[wet])

        return self.config.cflNumber * self.cellSize / float(np.max(speed))

    def step(self, timeStep: float | None = None) -> float:

        '''

        Advance one step with third-order SSP Runge-Kutta, then update the breaking state.

        The cells the dispersive part applies to are fixed for the step, so all three stages solve
        the same equations.

        Parameters:
        -----------
        timeStep : float | None
            Step to take [s]; the CFL step by default

        Returns:
        --------
        float : The step taken [s]

        '''

        dt = self.stableTimeStep() if timeStep is None else timeStep
        active = self.dispersiveCells
        h0, q0 = self.depth, self.discharge

        t0 = self.time
        rateH, rateQ = self._rates(h0, q0, active, t0)
        h1, q1 = self._clean(h0 + dt * rateH, q0 + dt * rateQ)
        rateH, rateQ = self._rates(h1, q1, active, t0 + dt)
        h2, q2 = self._clean(0.75 * h0 + 0.25 * (h1 + dt * rateH), 0.75 * q0 + 0.25 * (q1 + dt * rateQ))
        rateH, rateQ = self._rates(h2, q2, active, t0 + 0.5 * dt)
        h3, q3 = self._clean(h0 / 3.0 + 2.0 / 3.0 * (h2 + dt * rateH), q0 / 3.0 + 2.0 / 3.0 * (q2 + dt * rateQ))

        self.depth, self.discharge = h3, q3
        self.time += dt
        self.steps += 1
        self.timeStep = dt
        self._updateBreaking()

        return dt
