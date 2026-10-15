
# -- pySurf: Wave Theory -- #

'''

Closed-form water wave mechanics: the results a wave simulation is driven by and checked against.

Every result is a plain function of wave height, period or wavenumber, depth and slope, with no
simulation state, so it can be called from a solver, a wave maker, a figure or a test alike.

    Linear (Airy) waves     dispersion, wavelength, phase and group speed, kinematics, energy
    Solitary waves          first-order (Boussinesq) profile, celerity, decay, particle velocity;
                            the exact solitary wave of the Serre-Green-Naghdi equations
    Serre-Green-Naghdi      linear dispersion of the fully nonlinear, weakly dispersive equations
    Wave makers             Biesel piston transfer function
    Standing waves          rectangular-tank sloshing frequencies
    Breaking criteria       McCowan and Miche limits, the Iribarren number and Battjes' breaker
                            types, and Grilli's slope parameter for solitary waves on a slope

`WaveConditions` and `LinearWaveTheory` carry a periodic wave state for callers that prefer an
object; both delegate to the functions.

References:
-----------
Dean, R.G. & Dalrymple, R.A. (1991) -- Water Wave Mechanics for Engineers and Scientists,
    World Scientific. Linear theory (ch. 3-4), wave makers (ch. 6), solitary waves (ch. 11).
Biesel, F. (1951) -- Etude theorique d'un type d'appareil a houle, La Houille Blanche 6.
Battjes, J.A. (1974) -- Surf similarity, Proc. 14th Coastal Engineering Conf., 466-480.
Grilli, S.T., Svendsen, I.A. & Subramanya, R. (1997) -- Breaking criterion and characteristics
    for solitary waves on slopes, J. Waterway, Port, Coastal, Ocean Eng. 123(3), 102-112.
Serre, F. (1953) -- Contribution a l'etude des ecoulements permanents et variables dans les canaux,
    La Houille Blanche 8, 374-388.
Green, A.E. & Naghdi, P.M. (1976) -- A derivation of equations for wave propagation in water of
    variable depth, J. Fluid Mech. 78(2), 237-246.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

Linear theory is verified against its closed forms in tests/testWaveTheory.py: the dispersion
residual is at machine precision for periods of 6 to 14 s in depths of 0.1 to 1000 m, the
deep-water wavelength equals g T^2 / (2 pi), the shallow-water celerity equals sqrt(g d), and the
group-to-phase speed ratio reaches 1/2 and 1 in the two limits. The Biesel transfer function is
checked against its shallow (H/S = kd) and deep (H/S = 2) limits, and the sloshing frequency
against the shallow-water standing wave.

The solitary wave is the first-order solution: its celerity sqrt(g (d + H)) and decay
sqrt(3 H / (4 d^3)) are exact for that theory, and the tests check the profile integrates to the
closed-form volume 2 H / kappa. Higher-order solitary wave theory gives a slightly lower celerity
at large H/d; the first-order wave is the standard initial condition for numerical breaking
studies and sheds a small trailing tail as it adjusts.

The breaking criteria are empirical or semi-empirical and carry the scatter of the data they came
from. Grilli et al.'s slope parameter was derived from fully nonlinear potential flow computations
validated against laboratory measurements, on slopes from 1:100 to 1:8.

Sean Bowman [10/11/2026]

'''

# Standard library imports
from __future__ import annotations

import math
from dataclasses import dataclass

# Third-party imports
import numpy as np

# Local imports
from pysurf import constants

ArrayLike = np.ndarray | list | float

# McCowan (1894): a solitary wave breaks when H/d exceeds this ratio
mcCowanLimit: float = 0.78

# Miche (1944): a deep-water wave breaks when H/L exceeds this ratio
micheLimit: float = 1.0 / 7.0

#----------------------------------------------------------------------#
# -- Linear (Airy) Waves -- #
#----------------------------------------------------------------------#

def dispersionWavenumber(angularFrequency: ArrayLike, depth: ArrayLike, gravity: float = constants.gravity) -> np.ndarray:

    '''

    Solve the linear dispersion relation omega^2 = g k tanh(k d) for the wavenumber k.

    Newton iteration from the deep-water guess k0 = omega^2 / g converges to machine precision in
    under ten iterations for every depth, because the residual is monotone and convex in k.

    Parameters:
    -----------
    angularFrequency : ArrayLike
        omega [rad/s]
    depth : ArrayLike
        Still water depth d [m]
    gravity : float
        Gravitational acceleration [m/s^2]

    Returns:
    --------
    np.ndarray | float : Wavenumber k [rad/m], scalar for scalar input

    '''

    omega = np.asarray(angularFrequency, dtype = float)
    d = np.asarray(depth, dtype = float)
    k = omega**2 / gravity

    # The deep-water guess is too small in shallow water, where k d ~ omega d / sqrt(g d); the
    # larger of the two starts Newton on the convex side of the root
    k = np.maximum(k, omega / np.sqrt(gravity * d))

    for _ in range(60):
        tanhKd = np.tanh(k * d)
        residual = omega**2 - gravity * k * tanhKd
        derivative = -gravity * (tanhKd + k * d * (1.0 - tanhKd**2))
        step = -residual / derivative
        k = k + step
        if np.all(np.abs(step) <= 1e-14 * np.abs(k)):
            break

    return float(k) if k.ndim == 0 else k

def linearWavelength(period: ArrayLike, depth: ArrayLike, gravity: float = constants.gravity) -> np.ndarray:

    '''Wavelength L = 2 pi / k of a linear wave [m].'''

    return 2.0 * np.pi / dispersionWavenumber(2.0 * np.pi / np.asarray(period, dtype = float), depth, gravity)

def linearPhaseSpeed(period: ArrayLike, depth: ArrayLike, gravity: float = constants.gravity) -> np.ndarray:

    '''Phase speed c = omega / k of a linear wave [m/s].'''

    omega = 2.0 * np.pi / np.asarray(period, dtype = float)

    return omega / dispersionWavenumber(omega, depth, gravity)

def linearGroupSpeed(period: ArrayLike, depth: ArrayLike, gravity: float = constants.gravity) -> np.ndarray:

    '''

    Group speed cg = n c with n = (1 + 2 k d / sinh(2 k d)) / 2 [m/s].

    n is 1/2 in deep water and 1 in shallow water. Above 2 k d = 50 the sinh overflows double
    precision long after n has reached 1/2, so the limit is taken there.

    '''

    omega = 2.0 * np.pi / np.asarray(period, dtype = float)
    k = dispersionWavenumber(omega, depth, gravity)
    twoKd = 2.0 * k * np.asarray(depth, dtype = float)
    ratio = np.where(twoKd > 50.0, 0.5, 0.5 * (1.0 + twoKd / np.sinh(np.minimum(twoKd, 50.0))))

    return ratio * omega / k

def shallowWaterSpeed(depth: ArrayLike, gravity: float = constants.gravity) -> np.ndarray:

    '''Long-wave celerity sqrt(g d) [m/s].'''

    return np.sqrt(gravity * np.asarray(depth, dtype = float))

def deepWaterWavelength(period: ArrayLike, gravity: float = constants.gravity) -> np.ndarray:

    '''Deep-water wavelength L0 = g T^2 / (2 pi) [m].'''

    return gravity * np.asarray(period, dtype = float)**2 / (2.0 * np.pi)

#----------------------------------------------------------------------#
# -- Solitary Waves -- #
#----------------------------------------------------------------------#

def solitaryWaveSpeed(height: float, depth: float, gravity: float = constants.gravity) -> float:

    '''First-order solitary wave celerity c = sqrt(g (d + H)) [m/s].'''

    return math.sqrt(gravity * (depth + height))

def solitaryWaveDecay(height: float, depth: float) -> float:

    '''First-order solitary wave decay rate kappa = sqrt(3 H / (4 d^3)) [1/m].'''

    return math.sqrt(3.0 * height / (4.0 * depth**3))

def solitaryWaveProfile(x: ArrayLike, height: float, depth: float, crestX: float = 0.0) -> np.ndarray:

    '''

    Free-surface elevation above still water, eta = H sech^2(kappa (x - x0)) [m].

    Parameters:
    -----------
    x : ArrayLike
        Horizontal position [m]
    height : float
        Wave height H above still water [m]
    depth : float
        Still water depth d [m]
    crestX : float
        Crest position x0 [m]

    '''

    argument = solitaryWaveDecay(height, depth) * (np.asarray(x, dtype = float) - crestX)

    return height / np.cosh(argument)**2

def solitaryWaveVelocity(x: ArrayLike, height: float, depth: float, crestX: float = 0.0,
                         gravity: float = constants.gravity) -> np.ndarray:

    '''

    Depth-averaged horizontal particle velocity under a solitary wave, u = c eta / (d + eta) [m/s].

    This is the velocity that carries the wave's own volume flux c eta through a column of depth
    d + eta, which is exact for a wave of permanent form in a frame moving at c, and is the
    standard depth-uniform initial velocity for numerical solitary waves.

    '''

    eta = solitaryWaveProfile(x, height, depth, crestX)

    return solitaryWaveSpeed(height, depth, gravity) * eta / (depth + eta)

def solitaryWaveHalfLength(height: float, depth: float, fraction: float = 0.01) -> float:

    '''

    Distance from the crest at which the elevation falls to `fraction` of the height [m].

    sech^2(kappa x) = f gives kappa x = arccosh(1 / sqrt(f)). A domain that keeps the wave this far
    clear of a wall or a slope starts the wave undisturbed to that fraction.

    '''

    return math.acosh(1.0 / math.sqrt(fraction)) / solitaryWaveDecay(height, depth)

#----------------------------------------------------------------------#
# -- Serre-Green-Naghdi -- #
#----------------------------------------------------------------------#

def serreSolitaryWaveDecay(height: float, depth: float) -> float:

    '''

    Decay rate of the exact solitary wave of the Serre-Green-Naghdi equations,
    kappa = sqrt(3 H / (4 d^2 (d + H))) [1/m].

    The profile is the same sech^2 shape as the first-order wave and the celerity the same
    sqrt(g (d + H)); only the decay differs, smaller by sqrt(d / (d + H)), so the exact wave is
    wider. It is an exact solution of the equations at any height, which makes it the reference a
    Serre-Green-Naghdi solver is checked against.

    '''

    return math.sqrt(3.0 * height / (4.0 * depth**2 * (depth + height)))

def serreSolitaryWaveProfile(x: ArrayLike, height: float, depth: float, crestX: float = 0.0) -> np.ndarray:

    '''Free-surface elevation of the exact Serre-Green-Naghdi solitary wave, H sech^2(kappa (x - x0)) [m].'''

    argument = serreSolitaryWaveDecay(height, depth) * (np.asarray(x, dtype = float) - crestX)

    return height / np.cosh(argument)**2

def serreSolitaryWaveVelocity(x: ArrayLike, height: float, depth: float, crestX: float = 0.0,
                              gravity: float = constants.gravity) -> np.ndarray:

    '''

    Depth-averaged velocity of the exact Serre-Green-Naghdi solitary wave, u = c eta / (d + eta) [m/s].

    Exact for a wave of permanent form: in the frame moving at c the flow is steady and the volume
    flux h (u - c) = -c d is the same in every column.

    '''

    eta = serreSolitaryWaveProfile(x, height, depth, crestX)

    return solitaryWaveSpeed(height, depth, gravity) * eta / (depth + eta)

def serreSolitaryWaveHalfLength(height: float, depth: float, fraction: float = 0.01) -> float:

    '''Distance from the crest at which the exact Serre-Green-Naghdi wave falls to `fraction` of its height [m].'''

    return math.acosh(1.0 / math.sqrt(fraction)) / serreSolitaryWaveDecay(height, depth)

def serreSolitarySurfacePotential(x: np.ndarray, height: float, depth: float, crestX: float = 0.0,
                                  gravity: float = constants.gravity) -> np.ndarray:

    '''

    Velocity potential along the free surface of a solitary wave over a flat bed, from the exact
    SGN wave and the vertical structure potential flow gives its velocity [m^2/s].

    With the depth-averaged velocity ubar of the SGN wave and h = d + eta, irrotational flow over a
    flat bed has, to second order in depth over length,

        u_s = ubar - h^2 ubar_xx / 3       (horizontal velocity at the surface)
        w_s = -h ubar_x                    (vertical velocity at the surface)

    and the surface potential follows by integrating d phi / dx = u_s + w_s eta_x from the first
    point, where it is zero. It starts a fully nonlinear potential flow model with a wave close to
    its own solitary wave; the difference sheds as a small trailing tail.

    Parameters:
    -----------
    x : np.ndarray
        Increasing positions along the surface, finely spaced for the derivatives [m]
    height, depth : float
        Wave height H and still water depth d [m]
    crestX : float
        Crest position [m]

    '''

    x = np.asarray(x, dtype = float)
    eta = serreSolitaryWaveProfile(x, height, depth, crestX)
    mean = serreSolitaryWaveVelocity(x, height, depth, crestX, gravity)
    h = depth + eta
    meanX = np.gradient(mean, x)
    meanXX = np.gradient(meanX, x)
    slope = np.gradient(eta, x)
    integrand = (mean - h**2 * meanXX / 3.0) + (-h * meanX) * slope

    return np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(x))])

def serreAngularFrequency(wavenumber: ArrayLike, depth: float, gravity: float = constants.gravity,
                          alpha: float = 1.0) -> np.ndarray:

    '''

    Linear dispersion of the Serre-Green-Naghdi equations [rad/s]:

        omega^2 = g k^2 d (1 + (alpha - 1) (k d)^2 / 3) / (1 + alpha (k d)^2 / 3)

    alpha = 1 is the original equations, which match Airy's g k tanh(k d) to second order in k d:
    0.8 percent low in omega at k d = 1 and 13 percent low at k d = 3. Bonneton et al. (2011) take
    alpha = 1.159, which keeps the error under 1 percent out to k d = 3.

    '''

    kd2 = (np.asarray(wavenumber, dtype = float) * depth)**2
    k = np.asarray(wavenumber, dtype = float)

    return np.sqrt(gravity * k**2 * depth * (1.0 + (alpha - 1.0) * kd2 / 3.0) / (1.0 + alpha * kd2 / 3.0))

def serreWavenumber(angularFrequency: float, depth: float, gravity: float = constants.gravity, alpha: float = 1.0) -> float:

    '''

    The wavenumber of a linear SGN wave of a given angular frequency [rad/m]: the positive root of
    the dispersion relation, a quadratic in k^2.

    '''

    omega2 = angularFrequency**2
    b = depth**2 / 3.0
    quadratic = gravity * depth * (alpha - 1.0) * b
    linear = gravity * depth - omega2 * alpha * b
    if abs(quadratic) < 1e-15:
        k2 = omega2 / linear
    else:
        k2 = (-linear + math.sqrt(linear**2 + 4.0 * quadratic * omega2)) / (2.0 * quadratic)

    return math.sqrt(k2)

#----------------------------------------------------------------------#
# -- Wave Makers and Standing Waves -- #
#----------------------------------------------------------------------#

def pistonTransferFunction(kd: ArrayLike) -> np.ndarray:

    '''

    Biesel's piston wave-maker transfer function, H/S = 2 (cosh 2kd - 1) / (sinh 2kd + 2kd).

    H is the generated wave height and S the piston stroke (peak to peak). The ratio is kd in
    shallow water and 2 in deep water. Above 2kd = 50 cosh/sinh is 1 to double precision and
    both overflow, so the deep limit is taken there.

    '''

    twoKd = 2.0 * np.asarray(kd, dtype = float)
    bounded = np.minimum(twoKd, 50.0)
    ratio = 2.0 * (np.cosh(bounded) - 1.0) / (np.sinh(bounded) + bounded)

    return np.where(twoKd > 50.0, 2.0, ratio)

def sloshingFrequency(width: float, depth: float, mode: int = 1, gravity: float = constants.gravity) -> float:

    '''

    Natural angular frequency of the n-th sloshing mode in a rectangular tank [rad/s].

    omega_n^2 = g k_n tanh(k_n d) with k_n = n pi / W: the linear standing wave whose antinodes sit
    at the two walls.

    '''

    k = mode * math.pi / width

    return math.sqrt(gravity * k * math.tanh(k * depth))

#----------------------------------------------------------------------#
# -- Breaking Criteria -- #
#----------------------------------------------------------------------#

def isDepthLimitedBreaking(height: float, depth: float) -> bool:

    '''McCowan's limit: a wave breaks when H/d exceeds 0.78.'''

    return height / depth > mcCowanLimit

def isSteepnessLimitedBreaking(height: float, wavelength: float) -> bool:

    '''Miche's limit: a wave breaks when H/L exceeds 1/7.'''

    return wavelength > 0.0 and height / wavelength > micheLimit

def iribarrenNumber(slope: float, height: float, wavelength: float) -> float:

    '''

    Surf similarity (Iribarren) number xi = tan(beta) / sqrt(H / L).

    Parameters:
    -----------
    slope : float
        Beach slope tan(beta) [-]
    height : float
        Wave height [m]: offshore H0 or at the breaking point Hb
    wavelength : float
        Deep-water wavelength L0 [m]

    '''

    return slope / math.sqrt(height / wavelength)

def iribarrenBreakerType(xi: float, basis: str = 'offshore') -> str:

    '''

    Battjes' (1974) breaker type from the Iribarren number.

    With offshore wave height: spilling below 0.5, plunging to 3.3, surging or collapsing above.
    With breaking-point wave height: spilling below 0.4, plunging to 2.0, surging above.

    Parameters:
    -----------
    xi : float
        Iribarren number [-]
    basis : str
        'offshore' or 'breaking', the wave height xi was formed with

    Returns:
    --------
    str : 'spilling', 'plunging' or 'surging'

    '''

    thresholds = {'offshore': (0.5, 3.3), 'breaking': (0.4, 2.0)}
    if basis not in thresholds:
        raise ValueError(f'iribarrenBreakerType: basis must be \'offshore\' or \'breaking\', got {basis!r}.')
    lower, upper = thresholds[basis]

    if xi < lower:
        return 'spilling'
    if xi < upper:
        return 'plunging'
    return 'surging'

def solitarySlopeParameter(slope: float, height: float, depth: float) -> float:

    '''

    Grilli et al.'s (1997) slope parameter for a solitary wave, S0 = 1.521 tan(beta) / sqrt(H / d).

    H and d are the incident wave height and the still water depth on the flat bottom offshore of
    the slope. Larger S0 (steeper slope, smaller wave) moves the breaker from spilling toward
    surging.

    '''

    return 1.521 * slope / math.sqrt(height / depth)

def solitaryBreakerType(slopeParameter: float) -> str:

    '''

    Grilli et al.'s (1997) breaker type from the solitary slope parameter S0.

    Spilling below 0.025, plunging to 0.30, surging to 0.37, and no breaking above. The no-breaking
    bound is the same statement as H/d < 16.9 tan^2(beta), see `solitaryNonBreakingHeight`.

    Returns:
    --------
    str : 'spilling', 'plunging', 'surging' or 'none'

    '''

    if slopeParameter < 0.025:
        return 'spilling'
    if slopeParameter < 0.30:
        return 'plunging'
    if slopeParameter < 0.37:
        return 'surging'
    return 'none'

def solitaryNonBreakingHeight(slope: float) -> float:

    '''

    The largest H/d that does not break on a slope tan(beta): 16.9 tan^2(beta), equivalent to
    S0 = 0.37 since (1.521 / 0.37)^2 = 16.9.

    '''

    return (1.521 / 0.37)**2 * slope**2

#----------------------------------------------------------------------#
# -- Periodic Wave State -- #
#----------------------------------------------------------------------#

@dataclass
class WaveConditions:

    '''

    A periodic wave state.

    Parameters:
    -----------
    height : float
        Wave height H, trough to crest [m]
    period : float
        Wave period T [s]
    depth : float
        Still water depth d [m]
    direction : float
        Propagation direction, 0 = shore-normal [deg]

    '''

    height: float           # [m]
    period: float           # [s]
    depth: float            # [m]
    direction: float = 0.0  # [deg]

    @property
    def angularFrequency(self) -> float:
        '''Angular frequency omega = 2 pi / T [rad/s].'''
        return 2.0 * math.pi / self.period

    @property
    def amplitude(self) -> float:
        '''Wave amplitude a = H / 2 [m].'''
        return self.height / 2.0

    @classmethod
    def typicalBeachBreak(cls) -> WaveConditions:
        '''Chest-to-shoulder-high beach break: H = 1.5 m, T = 10 s, d = 2.5 m.'''
        return cls(height = 1.5, period = 10.0, depth = 2.5)

    @classmethod
    def smallDay(cls) -> WaveConditions:
        '''Knee-to-waist-high surf: H = 0.6 m, T = 8 s, d = 1.5 m.'''
        return cls(height = 0.6, period = 8.0, depth = 1.5)

    @classmethod
    def overhead(cls) -> WaveConditions:
        '''Overhead, well-organized swell: H = 2.0 m, T = 12 s, d = 3.0 m.'''
        return cls(height = 2.0, period = 12.0, depth = 3.0)

class LinearWaveTheory:

    '''

    Linear (Airy) wave theory evaluated for a `WaveConditions` state.

    Every method delegates to the module's functions. Pressure and energy depend on the water
    density; dispersion and kinematics do not.

    Parameters:
    -----------
    density : float
        Water density [kg/m^3], seawater by default

    '''

    def __init__(self, density: float = constants.seawaterDensity) -> None:

        self.density = density  # [kg/m^3]

    def solveDispersionRelation(self, omega: float, depth: float) -> float:
        '''Wavenumber k [rad/m] from omega^2 = g k tanh(k d).'''
        return dispersionWavenumber(omega, depth)

    def waveLength(self, waves: WaveConditions) -> float:
        '''Wavelength [m].'''
        return float(linearWavelength(waves.period, waves.depth))

    def waveSpeed(self, waves: WaveConditions) -> float:
        '''Phase speed [m/s].'''
        return float(linearPhaseSpeed(waves.period, waves.depth))

    def groupSpeed(self, waves: WaveConditions) -> float:
        '''Group speed [m/s].'''
        return float(linearGroupSpeed(waves.period, waves.depth))

    def surfaceElevation(self, x: float, t: float, waves: WaveConditions) -> float:
        '''Surface elevation eta = a cos(kx - omega t) [m].'''
        k = dispersionWavenumber(waves.angularFrequency, waves.depth)
        return waves.amplitude * math.cos(k * x - waves.angularFrequency * t)

    def velocityField(self, x: float, z: float, t: float, waves: WaveConditions) -> tuple[float, float]:

        '''

        Particle velocity (u, w) at height z (negative below still water) [m/s], with z clamped
        to the water column.

        '''

        omega, d = waves.angularFrequency, waves.depth
        k = dispersionWavenumber(omega, d)
        zClamped = max(-d, min(0.0, z))
        phase = k * x - omega * t
        u = waves.amplitude * omega * math.cosh(k * (zClamped + d)) / math.sinh(k * d) * math.cos(phase)
        w = waves.amplitude * omega * math.sinh(k * (zClamped + d)) / math.sinh(k * d) * math.sin(phase)

        return (u, w)

    def pressure(self, x: float, z: float, t: float, waves: WaveConditions) -> float:

        '''Hydrostatic plus dynamic pressure at height z (negative below still water) [Pa].'''

        omega, d = waves.angularFrequency, waves.depth
        k = dispersionWavenumber(omega, d)
        zClamped = max(-d, min(0.0, z))
        hydrostatic = -self.density * constants.gravity * zClamped
        dynamic = (self.density * constants.gravity * waves.amplitude * math.cosh(k * (zClamped + d))
                   / math.cosh(k * d) * math.cos(k * x - omega * t))

        return hydrostatic + dynamic

    def energyDensity(self, waves: WaveConditions) -> float:
        '''Energy per unit surface area E = rho g H^2 / 8 [J/m^2].'''
        return self.density * constants.gravity * waves.height**2 / 8.0

    def energyFlux(self, waves: WaveConditions) -> float:
        '''Energy flux per unit crest width E cg [W/m].'''
        return self.energyDensity(waves) * self.groupSpeed(waves)

    def isBroken(self, waves: WaveConditions) -> bool:
        '''Whether the wave exceeds either McCowan's or Miche's limit.'''
        return (isDepthLimitedBreaking(waves.height, waves.depth)
                or isSteepnessLimitedBreaking(waves.height, self.waveLength(waves)))
