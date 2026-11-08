
# -- pySurf: Weakly Compressible SPH Solver -- #

'''

Weakly compressible SPH (WCSPH) for free-surface water flows.

Water is treated as slightly compressible: pressure follows density through the Tait equation of
state, with an artificial speed of sound about ten times the fastest flow so density departs from
its reference by around one percent. That replaces the pressure Poisson equation of an
incompressible solver with an explicit update.

Each step:

    1. Pair every particle with its neighbors inside the kernel support (`pysurf.sphNeighbors`)
    2. Every `densityFilterInterval` steps, reset the fluid density with a Shepard filter
    3. Pressure from the Tait equation of state, negative pressure clamped to zero; with
       `wallPressure = 'extrapolated'` the boundary takes its pressure from the fluid around it
    4. Acceleration from the symmetric pressure gradient, Monaghan's artificial viscosity and
       gravity; with `viscosityLimiter` the viscosity sees each pair's velocity difference less
       its linear part
    5. Time step from the acoustic CFL and force conditions
    6. Kick the velocity, update the density from the kicked velocity, with the delta-SPH
       diffusive term when `densityDiffusion` is set, then drift the position with the
       XSPH-smoothed velocity
    7. With `particleShifting`, shift the fluid down its concentration gradient, along the free
       surface near it, carrying density and velocity to where each particle lands
    8. Enforce the solid boundaries (`pysurf.sphBoundaries`)

The density update skips pairs of two boundary particles: a wall moving past the bed, as a piston
does, compresses no water. An optional floor keeps the fluid density from falling far below rho0
in spray, where the pressure is zero anyway.

The physics is written as functions of arrays, one per equation, and `WcsphSolver` composes them.
Each function can be called and tested on its own. For large runs the solver can instead use
`pysurf.sphCompiled` (`backend = 'compiled'`): the same equations compiled with numba, summed per
particle over a cell-linked neighbor list across threads, with no per-pair arrays held in memory.
For millions of particles it can use `pysurf.sphGpu` (`backend = 'gpu'`): the same sums on a GPU
through NVIDIA Warp, in float32 with separations from float64 positions, while the host keeps the
state in float64 and integrates it.

Density is integrated from the continuity equation, d rho_i / dt = sum_j m_j v_ij . grad W_ij,
rather than summed from positions. A summed density reads low at the free surface, where half of
a particle's kernel support is empty air; the surface layer then carries no pressure and sinks
into the water below until its deficient sum reaches the reference density, compressing the top
of the column by most of a particle spacing and launching waves from nothing. The continuity
equation starts every particle at its true density and keeps it there. Its density drifts slowly
instead, which the periodic Shepard filter resets (Dalrymple & Rogers 2006 use every 30 steps).

`initialize` sets the starting density hydrostatic beneath a given free surface, fluid and
boundary alike, so the water starts in equilibrium with gravity rather than settling.

XSPH (Monaghan 1989) moves each particle with its velocity blended toward its fluid neighbors' mean,
which keeps particles ordered without a stability-limited time step. It changes only the velocity
particles are moved with. Writing the blended velocity back into the particle velocity would
diffuse momentum by roughly epsilon h^2 / dt every step, an effective viscosity of order 0.1 m^2/s
at typical resolution, which overdamps a sloshing tank to a creep.

The artificial viscosity acts between fluid particles; between fluid and boundary particles it is
scaled by `boundaryViscosityFactor`, zero by default, which makes the walls free-slip. The artificial
viscosity is a numerical stabilizer with an effective kinematic viscosity of alpha h c0 / 8 in 2D,
around a thousand times water's. Applied against resting wall particles it grows a boundary layer
sqrt(2 nu / omega) thick, 16 mm in a 100 mm deep sloshing tank at alpha = 0.1, where real water's
is under a millimeter; free-slip walls are the closer approximation, and the one the inviscid
references assume. On a solitary wave of H/d = 0.3 crossing a flat channel at 8 particles per
depth, free-slip walls cut the crest height lost in 3 s from 0.059 d to 0.052 d.

Six options replace parts of that scheme, and the scheme's kernel is a choice:

    densityDiffusion    the delta-SPH term of Molteni and Colagrossi (2009) in the continuity
                        equation, on the density less its hydrostatic part when 'hydrostatic', so
                        water at rest is left alone (`densityDiffusionRate`). It replaces the
                        Shepard filter.
    particleShifting    Lind et al. (2012) shifting with Sun et al.'s (2017) size and a free-surface
                        treatment (`shiftingDisplacement`), which keeps the particles evenly spread
                        in place of XSPH.
    wallPressure        'extrapolated' gives each boundary particle the kernel average of the
                        fluid's pressure carried hydrostatically down to it (Adami et al. 2012,
                        `extrapolatedWallPressure`). Integrated from the continuity equation, the
                        inner wall layers barely see a passing wave: under a solitary wave of
                        H/d = 0.3 the second layer's pressure stays 0.24 rho g d, its whole excess,
                        below the bed pressure.
    continuityCorrection
                        the kernel gradient correction of Bonet and Lok (1999) in the continuity
                        equation (`correctionMatrices`). Within 2h of a free surface the plain sum
                        reads a divergence-free flow that stretches the water vertically, as under
                        a steepening crest, as compression: a quarter of the strain rate one or two
                        spacings down, over a layer as deep as the kernel is wide. With c0 ten
                        times the speed of a solitary wave of H/d = 0.3, a one percent density
                        error carries 1.3 d of head, so the layer swells and opens voids. The
                        correction makes the sum exact for a linear velocity field.
    negativePressure    lets the fluid carry tension, where the scheme clamps the Tait pressure at
                        zero. The clamp suits the Shepard filter, whose free-surface densities read
                        low; with continuity density it leaves water near zero pressure, such as a
                        steep crest, with nothing to pull a gap shut.
    viscosityLimiter    reconstructs each fluid pair's velocities linearly to its midpoint from the
                        two particles' velocity gradients, under a slope limiter, and gives the
                        artificial viscosity the difference that is left (Frontiere et al. 2017;
                        Rosswog 2020; `limitedVelocityDifference`). Monaghan's term acts on any
                        approaching pair, so on a smooth flow it is a viscosity of alpha h c0 / 8
                        on the flow's own strain: under a steepening crest at 64 particles per
                        depth, 2.3e-3 m^2/s, more than half of the term's dissipation there. The
                        reconstruction removes the linear part, which leaves the term on the
                        particles' noise and on flow that varies faster than linearly across a
                        pair.
    kernelType          'cubicSpline' with h = 1.3 spacings, or 'wendlandC2' with h = 2.5 to 3
                        spacings. A fluid particle sliding over a wall of particles one spacing
                        apart feels sum_w grad W ripple along the wall: by 1.3% of its normal force
                        with the cubic spline at h = 1.3 spacings and by 0.05% with the Wendland at
                        2. Under a wave that ripple drags the water next to the bed. The overlap of
                        the kernels matters more: SPH's zeroth-order error over disordered
                        particles does not shrink with resolution at fixed h / spacing, and turns a
                        wave's energy into particle noise. A solitary wave of H/d = 0.3 loses 4.6%
                        of its height over 29 depths with the Wendland at h = 2 spacings, at 32 and
                        64 particles per depth alike, and its energy with it. With the continuity
                        correction and alpha = 0.02 it loses 2.3% and 2.5% at h = 2.5 spacings,
                        and 2.1% and 1.3% at 3, where the loss is the artificial viscosity's and
                        falls with h. On the GPU the wider kernel costs little in 2D, where its
                        longer step offsets the neighbors; in 3D, h = 2.5 and 3 spacings cost about
                        3.7 and 5.3 times the cubic spline per simulated second.

References:
-----------
Monaghan, J.J. (1989) -- On the problem of penetration in particle methods, J. Comput. Phys. 82
Monaghan, J.J. (1992) -- Smoothed particle hydrodynamics, Annu. Rev. Astron. Astrophys. 30
Monaghan, J.J. (1994) -- Simulating free surface flows with SPH, J. Comput. Phys. 110
Crespo, A.J.C., Gomez-Gesteira, M. & Dalrymple, R.A. (2007) -- Boundary conditions generated by
    dynamic particles in SPH methods, CMC 5(3)
Dalrymple, R.A. & Rogers, B.D. (2006) -- Numerical modeling of water waves with the SPH method,
    Coastal Engineering 53
Crespo, A.J.C. et al. (2015) -- DualSPHysics: open-source parallel CFD solver based on SPH,
    Comput. Phys. Commun. 187
Molteni, D. & Colagrossi, A. (2009) -- A simple procedure to improve the pressure evaluation in
    hydrodynamic context using the SPH, Comput. Phys. Commun. 180
Lind, S.J., Xu, R., Stansby, P.K. & Rogers, B.D. (2012) -- Incompressible smoothed particle
    hydrodynamics for free-surface flows: a generalised diffusion-based algorithm for stability
    and validations for impulsive flows and propagating waves, J. Comput. Phys. 231
Adami, S., Hu, X.Y. & Adams, N.A. (2012) -- A generalized wall boundary condition for smoothed
    particle hydrodynamics, J. Comput. Phys. 231
Sun, P.N., Colagrossi, A., Marrone, S. & Zhang, A.M. (2017) -- The delta-plus-SPH model: simple
    procedures for a further improvement of the SPH scheme, Comput. Methods Appl. Mech. Eng. 315
Bonet, J. & Lok, T.-S.L. (1999) -- Variational and momentum preservation aspects of smooth particle
    hydrodynamic formulations, Comput. Methods Appl. Mech. Eng. 180
Frontiere, N., Raskin, C.D. & Owen, J.M. (2017) -- CRKSPH: a conservative reproducing kernel
    smoothed particle hydrodynamics scheme, J. Comput. Phys. 332
Rosswog, S. (2020) -- The Lagrangian hydrodynamics code MAGMA2, Mon. Not. R. Astron. Soc. 498(3)

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

The equations are unit-tested in tests/testSphSolver.py: the equation of state against its closed
form and its inverse, pair forces against Newton's third law (total momentum conserved to rounding
in a gravity-free cloud), the artificial viscosity's sign and its zero for separating pairs, XSPH
against leaving the momentum untouched, the velocity gradient against a linear flow out to a free
surface (exact to rounding), the limited viscosity against acting on none of the pairs a pure strain
closes, against still damping noise laid over that strain (at least half what the plain term takes
from the noise alone, under a tenth of what it takes from both), against conserving linear and
angular momentum, and against any pair's dissipation going negative, the continuity rate against
zero for a rigid translation, the Shepard filter against returning a uniform density unchanged, and
a still tank initialized hydrostatically against staying at rest.

The solver is validated against the linear first-mode sloshing period of a rectangular tank in
tests/testValidationSloshing.py, and against the first-order solitary wave celerity in
tests/testValidationSolitary.py. Both are marked slow and run with `pytest -m slow`. Breaking on a
slope is validated through `pysurf.BreakingWave`.

Sean Bowman [02/05/2026]

'''

# Standard library imports
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

# Third-party imports
import numpy as np

# Local imports
from pysurf import constants
from pysurf.sphKernels import SphKernel, createKernel
from pysurf.sphNeighbors import neighborPairs
from pysurf.sphParticles import ParticleSystem
from pysurf.sphRefinement import splitParticles

# Tait exponent: 7 is the standard value for water
taitExponent: float = 7.0

# Softening in the artificial viscosity denominator, as a fraction of h^2, so coincident
# particles do not divide by zero
viscositySoftening: float = 0.01

# The limited artificial viscosity fades its reconstruction out for pairs closer than this fraction
# of h_ij, as exp(-((r / h - 0.3) / 0.2)^2), so particles pressed together keep the whole term that
# stops them interpenetrating: MAGMA2's eta_crit and eta_fold (Rosswog 2020). At h = 3 spacings a
# lattice's nearest neighbors sit at 0.33 h, outside the fade
limiterCloseDistance: float = 0.3
limiterFoldWidth: float = 0.2

# Particle shifting: the anti-clumping term's weight R and power n (Sun et al. 2017), and the free
# surface. A fluid particle is on it when its position divergence falls below 1.5 in 2D or 2.4 in 3D
# (Lee et al. 2008), or when its support is lopsided, h |grad C0| above a threshold set per kernel.
# On a lattice below a flat surface, layer by layer from the top:
#
#                                       div r 2D                div r 3D                h |grad C0|
#     cubic spline, h = 1.3 spacings   1.28, 1.90, 2.02        2.03, 2.88, 2.98        0.59, 0.07, 0
#     Wendland C2,  h = 2 spacings     1.18, 1.63, 1.94, 2.00  1.87, 2.57, 2.95, 3.00  0.68, 0.28, 0.04, 0
#
# The second test catches a surface particle beside a wall, whose wall neighbors lift its div r to
# the threshold (1.52 in a 2D corner with the cubic spline, 1.48 with the Wendland) while its support
# stays one-sided (h |grad C0| of 0.47 and 0.52). The Wendland's wider support leaves the layer below
# the surface lopsided too, at 0.28, so its threshold sits between that and the corner's
antiClumpingWeight: float = 0.2
antiClumpingPower: int = 4

# The continuity equation's kernel gradient correction applies where det M >= this to the power of
# the dimensions: 0.31 at a flat free surface in 2D with the Wendland at h = 3 spacings, 0.19 in 3D;
# a spray droplet with a few neighbors falls below it and keeps the uncorrected sum
correctionFloor: float = 0.25
freeSurfaceDivergence: dict = {2: 1.5, 3: 2.4}
freeSurfaceLopsidedness: dict = {'cubicSpline': 0.3, 'wendlandC2': 0.4}

# Next to the surface a particle whose own h |grad C| exceeds this does not shift: about half of its
# value one layer down on a lattice (0.071 in 2D, 0.058 in 3D), where the gradient is the surface's
# rather than the particles' disorder; two layers down it is zero
surfaceGradient: float = 0.03

# A in the shift's size A h |v_i| dt when each particle's speed sets it (Skillen et al. 2013)
localShiftingCoefficient: float = 2.0

#----------------------------------------------------------------------#
# -- Configuration and State -- #
#----------------------------------------------------------------------#

@dataclass
class SimulationConfig:

    '''

    Configuration for an SPH simulation. SI units throughout.

    Parameters:
    -----------
    domainMin, domainMax : np.ndarray
        Corners of the simulation domain [m]
    particleSpacing : float
        Lattice spacing of the initial particle arrangement [m]
    smoothingLengthRatio : float
        h / spacing, typically 1.2 to 1.5
    referenceDensity : float
        rho0 [kg/m^3]
    gravity : np.ndarray
        Gravity vector [m/s^2]
    endTime : float
        Simulated time to run [s]
    maxTimeStep : float
        Upper bound on the adaptive time step [s]
    outputInterval : float
        Simulated time between recorded frames [s]
    dimensions : int
        2 or 3
    kernelType : str
        'cubicSpline' or 'wendlandC2'
    speedOfSound : float | None
        Artificial speed of sound c0 [m/s]; None derives it from the domain height
    speedOfSoundFactor : float
        c0 / sqrt(g H) when c0 is derived
    alphaViscosity : float
        Monaghan artificial viscosity coefficient alpha
    boundaryViscosityFactor : float
        Multiplier on the artificial viscosity between fluid and boundary particles: 0 for free-slip
        walls, 1 to apply the full viscosity there
    xsphEpsilon : float
        XSPH blending factor epsilon, 0 to 1
    cflNumber : float
        Courant number for the adaptive time step
    densityFilterInterval : int
        Steps between Shepard density resets; 0 for none
    densityFloor : float
        Smallest fluid density, as a share of rho0; 0 for none. Below rho0 the pressure is already
        zero, so a floor changes no pressure: it keeps a spray droplet whose neighbors have flown
        apart from carrying a density near zero, which divides the artificial viscosity and the XSPH
        weight of every pair it joins
    initialSurface : float | Callable | None
        Free-surface elevation the starting density is hydrostatic beneath [m], as a level or a
        function of x; None starts every particle at the reference density
    periodicWidth : float | None
        In 3D, the length of a periodic y axis on [0, periodicWidth) [m]: particles leaving one side
        come back on the other, and pairs across the seam see each other at their nearest image;
        None for no periodic axis
    densityDiffusion : str
        Diffusive term in the continuity equation (delta-SPH): 'none'; 'molteni', the term of
        Molteni and Colagrossi (2009); or 'hydrostatic', the same term acting on the density less
        its hydrostatic part (`densityDiffusionRate`). It replaces the Shepard filter, so runs with
        it set `densityFilterInterval` to 0
    diffusionDelta : float
        The diffusive term's coefficient delta; 0.1 is the usual value
    particleShifting : bool
        Move each fluid particle a short way down its concentration gradient every step, along the
        free surface near it (`shiftingDisplacement`), so the particles stay evenly spread
    shiftingSpeed : str
        What sets the shift's size: 'reference', CFL Ma (2h)^2 for every particle (Sun et al. 2017),
        or 'local', A h |v_i| dt with A = 2 (Skillen et al. 2013), so water at rest does not shift
    shiftingMach : float
        Ma in the reference size, the flow's largest speed over c0: 0.1 when c0 is ten wave speeds
    wallPressure : str
        How boundary particles get their pressure: 'continuity', their density integrated from the
        continuity equation like the fluid's (Crespo et al. 2007); or 'extrapolated', taken each
        step from the fluid around them with the hydrostatic increment to the particle (Adami et
        al. 2012, `extrapolatedWallPressure`)
    negativePressure : bool
        Whether the fluid may carry tension. False clamps the Tait pressure at zero, as the
        Shepard-filtered scheme needs: there a free-surface particle's density reads low and its
        suction pulls particles into clumps. With the continuity density of delta-SPH the clamp
        leaves water near zero pressure, such as a steep crest, with nothing to pull a gap shut, and
        the near-surface particles drift apart. Boundary particles stay clamped either way
    continuityCorrection : bool
        Apply the kernel gradient correction (`correctionMatrices`) in the continuity equation, so
        a flow that stretches the water near a free surface, as under a steepening crest, does not
        read as compression there
    viscosityLimiter : bool
        Give the artificial viscosity each fluid pair's velocity difference less its linear part
        under a slope limiter (`limitedVelocityDifference`), so it damps the particles' noise and
        leaves a smooth flow's strain alone. Boundary pairs keep the whole difference
    backend : str
        'numpy', the array formulation of this module; 'compiled', the same equations compiled
        with numba and run across threads (`pysurf.sphCompiled`); or 'gpu', the same sums on Warp's
        preferred device, a CUDA GPU where present (`pysurf.sphGpu`). The compiled backend has the
        cubic spline kernel only

    '''

    domainMin: np.ndarray
    domainMax: np.ndarray
    particleSpacing: float = 0.005
    smoothingLengthRatio: float = 1.3
    referenceDensity: float = 1000.0
    gravity: np.ndarray = field(default_factory = lambda: np.array([0.0, -constants.gravity]))
    endTime: float = 3.0
    maxTimeStep: float = 1e-4
    outputInterval: float = 0.02
    dimensions: int = 2
    kernelType: str = 'cubicSpline'
    speedOfSound: float | None = None
    speedOfSoundFactor: float = 10.0
    alphaViscosity: float = 0.1
    boundaryViscosityFactor: float = 0.0
    xsphEpsilon: float = 0.5
    cflNumber: float = 0.25
    densityFilterInterval: int = 30
    densityFloor: float = 0.0
    initialSurface: object = None
    periodicWidth: float | None = None
    densityDiffusion: str = 'none'
    diffusionDelta: float = 0.1
    particleShifting: bool = False
    shiftingSpeed: str = 'reference'
    shiftingMach: float = 0.1
    wallPressure: str = 'continuity'
    negativePressure: bool = False
    continuityCorrection: bool = False
    viscosityLimiter: bool = False
    backend: str = 'numpy'

    @property
    def smoothingLength(self) -> float:
        '''Smoothing length h = ratio * spacing [m].'''
        return self.smoothingLengthRatio * self.particleSpacing

    @property
    def supportRadius(self) -> float:
        '''Kernel support radius 2h [m]; both kernels vanish beyond it.'''
        return 2.0 * self.smoothingLength

    @property
    def domainSize(self) -> np.ndarray:
        '''Domain extent along each axis [m].'''
        return np.asarray(self.domainMax) - np.asarray(self.domainMin)

    @classmethod
    def fromJson(cls, configPath: str) -> SimulationConfig:

        '''

        Load a configuration from the 'simulation', 'sph', 'fluid' and 'tank' sections of a JSON
        file.

        '''

        with open(configPath, 'r') as configFile:
            data = json.load(configFile)

        simSection = data.get('simulation', {})
        sphSection = data.get('sph', {})
        fluidSection = data.get('fluid', {})
        tankSection = data.get('tank', {})

        dimensions = simSection.get('dimensions', 2)
        gravityMagnitude = fluidSection.get('gravity', constants.gravity)
        gravityVector = np.zeros(dimensions)
        gravityVector[-1] = -gravityMagnitude

        tankWidth = tankSection.get('width', 0.5)
        tankHeight = tankSection.get('height', 0.3)
        if dimensions == 2:
            domainMax = np.array([tankWidth, tankHeight])
        else:
            domainMax = np.array([tankWidth, tankSection.get('depth', 0.2), tankHeight])

        return cls(
            domainMin = np.zeros(dimensions),
            domainMax = domainMax,
            particleSpacing = sphSection.get('particleSpacing', 0.005),
            smoothingLengthRatio = sphSection.get('smoothingLengthRatio', 1.3),
            referenceDensity = fluidSection.get('density', 1000.0),
            gravity = gravityVector,
            endTime = simSection.get('endTime', 3.0),
            maxTimeStep = sphSection.get('maxTimeStep', 1e-4),
            outputInterval = simSection.get('outputInterval', 0.02),
            dimensions = dimensions,
            kernelType = sphSection.get('kernelType', 'cubicSpline'),
        )

@dataclass
class SimulationState:

    '''

    Scalar diagnostics of one simulation instant.

    Parameters:
    -----------
    time : float
        Simulated time [s]
    step : int
        Steps taken
    dt : float
        Last time step [s]
    kineticEnergy, potentialEnergy : float
        Fluid energies [J], per unit depth [J/m] in 2D
    maxVelocity : float
        Largest fluid speed [m/s]
    maxDensityError : float
        Largest relative fluid density departure [-]; free-surface particles dominate it

    '''

    time: float
    step: int
    dt: float
    kineticEnergy: float
    potentialEnergy: float
    maxVelocity: float
    maxDensityError: float

    @property
    def totalEnergy(self) -> float:
        '''Kinetic plus potential energy [J], per unit depth [J/m] in 2D.'''
        return self.kineticEnergy + self.potentialEnergy

#----------------------------------------------------------------------#
# -- Equations -- #
#----------------------------------------------------------------------#

def speedOfSound(characteristicDepth: float, factor: float = 10.0, gravity: float = constants.gravity) -> float:

    '''

    Artificial speed of sound c0 = factor * sqrt(g H) [m/s].

    sqrt(g H) is the fastest long-wave speed a water depth H supports, and a free-surface flow's
    velocities scale with it. A factor of 10 puts the flow Mach number near 0.1, so density varies
    by about Ma^2 = 1 percent.

    '''

    return factor * math.sqrt(gravity * characteristicDepth)

def taitEquationOfState(density: np.ndarray, referenceDensity: float, soundSpeed: float,
                        exponent: float = taitExponent, clampNegative: bool = True) -> np.ndarray:

    '''

    Tait pressure p = B ((rho / rho0)^gamma - 1) with B = rho0 c0^2 / gamma [Pa].

    B is chosen so the slope dp/drho at rho0 is c0^2. Negative pressure is clamped to zero by
    default: a free-surface particle reads a low density because its kernel support is partly
    empty, and the suction that would produce pulls particles into clumps (tensile instability).

    '''

    stiffness = referenceDensity * soundSpeed**2 / exponent
    pressure = stiffness * ((np.asarray(density) / referenceDensity)**exponent - 1.0)

    return np.maximum(pressure, 0.0) if clampNegative else pressure

def taitDensity(pressure: np.ndarray, referenceDensity: float, soundSpeed: float,
                exponent: float = taitExponent) -> np.ndarray:

    '''The inverse of `taitEquationOfState`: the density that carries a pressure [kg/m^3].'''

    stiffness = referenceDensity * soundSpeed**2 / exponent

    return referenceDensity * (1.0 + np.asarray(pressure) / stiffness)**(1.0 / exponent)

def scatterAdd(target: np.ndarray, index: np.ndarray, values: np.ndarray) -> np.ndarray:

    '''

    Accumulate pair contributions into per-particle totals: target[index[k]] += values[k], with
    repeated indices summed. Returns the target, modified in place.

    Scalars go through np.add.at, which NumPy 2 runs on a fast path for 1D targets. Vectors go
    column by column through np.bincount: np.add.at into an (N, dim) array takes its slow
    general path, about seven times slower at SPH pair counts.

    '''

    if np.ndim(values) == 1:
        np.add.at(target, index, values)
    else:
        for column in range(values.shape[1]):
            target[:, column] += np.bincount(index, weights = values[:, column], minlength = target.shape[0])

    return target

def _pairMass(masses, index: np.ndarray):

    '''The mass of each pair member: an array, or the scalar itself when every particle has one mass.'''

    return masses[index] if np.ndim(masses) else masses

def summationDensity(masses: np.ndarray, i: np.ndarray, j: np.ndarray, kernelValues: np.ndarray,
                     selfKernel: float, count: int | None = None) -> np.ndarray:

    '''

    SPH density rho_i = sum_j m_j W_ij, the particle's own contribution included [kg/m^3].

    Parameters:
    -----------
    masses : np.ndarray | float
        Particle masses, or one mass shared by every particle
    i, j : np.ndarray
        Unique pair indices
    kernelValues : np.ndarray
        W for each pair
    selfKernel : float
        W(0, h), the particle's weight on itself
    count : int
        Number of particles, needed when `masses` is a scalar

    '''

    density = masses * selfKernel if np.ndim(masses) else np.full(count, masses * selfKernel)
    scatterAdd(density, i, _pairMass(masses, j) * kernelValues)
    scatterAdd(density, j, _pairMass(masses, i) * kernelValues)

    return density

def continuityRate(masses, velocities: np.ndarray, i: np.ndarray, j: np.ndarray,
                   kernelGradients: np.ndarray, count: int, active: np.ndarray | None = None,
                   corrections: np.ndarray | None = None) -> np.ndarray:

    '''

    The continuity equation in SPH form, d rho_i / dt = sum_j m_j (v_i - v_j) . grad W_ij [kg/(m^3 s)].

    v_ij . grad W_ij is the same for both members of a pair (both factors change sign), so each
    pair adds the same rate to i and j, weighted by the other's mass. A rigid translation has
    v_ij = 0 for every pair and leaves the density unchanged. Pairs outside the optional `active`
    mask add nothing: two solid particles sliding past each other, a piston past the bed, compress
    no water.

    With `corrections` (`correctionMatrices`), each member takes its own L grad W_ij, so the sum
    is exact for a linear velocity field wherever L applies, a free surface included.

    '''

    difference = velocities[i] - velocities[j]
    if corrections is None:
        rateI = rateJ = np.sum(difference * kernelGradients, axis = 1)
    else:
        rateI = np.einsum('pa,pab,pb->p', difference, corrections[i], kernelGradients)
        rateJ = np.einsum('pa,pab,pb->p', difference, corrections[j], kernelGradients)
    if active is not None:
        rateI, rateJ = np.where(active, rateI, 0.0), np.where(active, rateJ, 0.0)
    densityRate = np.zeros(count)
    scatterAdd(densityRate, i, _pairMass(masses, j) * rateI)
    scatterAdd(densityRate, j, _pairMass(masses, i) * rateJ)

    return densityRate

def correctionMatrices(masses, densities: np.ndarray, i: np.ndarray, j: np.ndarray, separation: np.ndarray,
                       kernelGradients: np.ndarray, isFluid: np.ndarray) -> np.ndarray:

    '''

    The kernel gradient correction of Bonet and Lok (1999), L_i = M_i^-1, with
    M_i = sum_j V_j (x_j - x_i) (x) grad_i W_ij and V_j = m_j / rho_j, shape (N, dim, dim).

    sum_j V_j (f_j - f_i) L_i grad_i W_ij is then exact for any linear f. M is the identity where
    the support is full. Within 2h of a free surface it loses the rows the missing neighbors would
    add, and loses more of its vertical row than its horizontal ones, so there the uncorrected
    continuity sum reads a divergence-free flow that stretches the water vertically as compression.
    L is the identity on the boundary and wherever det M falls below `correctionFloor` to the
    power of the dimensions, a particle with too few neighbors to set a gradient by.

    '''

    count, dimensions = len(densities), separation.shape[1]
    volumes = (masses if np.ndim(masses) else np.full(count, float(masses))) / np.asarray(densities)
    # Both members of a pair add -(x_i - x_j) (x) grad_i W_ij, weighted by the other's volume
    outer = -separation[:, :, None] * kernelGradients[:, None, :]
    matrices = np.zeros((count, dimensions, dimensions))
    for a in range(dimensions):
        for b in range(dimensions):
            column = np.zeros(count)
            scatterAdd(column, i, volumes[j] * outer[:, a, b])
            scatterAdd(column, j, volumes[i] * outer[:, a, b])
            matrices[:, a, b] = column

    corrections = np.broadcast_to(np.eye(dimensions), matrices.shape).copy()
    applies = isFluid & (np.linalg.det(matrices) >= correctionFloor**dimensions)
    corrections[applies] = np.linalg.inv(matrices[applies])

    return corrections

def hydrostaticDensityDifference(verticalSeparation: np.ndarray, meanDensity: np.ndarray, referenceDensity: float,
                                 soundSpeed: float, gravity: float, exponent: float = taitExponent) -> np.ndarray:

    '''

    The density difference rho_j - rho_i that water at rest holds between two particles z_i - z_j
    apart vertically [kg/m^3].

    At rest dp/dz = -rho g, and the Tait equation's local sound speed is c^2 = dp/drho =
    c0^2 (rho / rho0)^(gamma - 1), so drho/dz = -rho g / c^2. Taken at the pair's mean density, the
    difference is rho0 (g / c0^2) (z_i - z_j) (rho_bar / rho0)^(2 - gamma), exact to second order in
    the separation.

    '''

    ratio = np.asarray(meanDensity) / referenceDensity

    return referenceDensity * gravity / soundSpeed**2 * np.asarray(verticalSeparation) * ratio**(2.0 - exponent)

def densityDiffusionRate(masses, densities: np.ndarray, i: np.ndarray, j: np.ndarray, separation: np.ndarray,
                         distance: np.ndarray, kernelGradients: np.ndarray, count: int, smoothingLength: float,
                         soundSpeed: float, delta: float, referenceDensity: float, gravity: float | None = None,
                         active: np.ndarray | None = None) -> np.ndarray:

    '''

    The delta-SPH diffusive term of the continuity equation [kg/(m^3 s)]:

        D_i = delta h c0 sum_j 2 (rho_j - rho_i - rho^H_ji) (-r_ij . grad W_ij) / (r_ij^2 + 0.01 h^2) V_j

    with r_ij = r_i - r_j and V_j = m_j / rho_j. -r_ij . grad W_ij = -r dW/dr is positive, so each
    pair moves the two densities toward each other: a Laplacian of the density, which damps the
    high-wavenumber pressure noise of weakly compressible SPH (Molteni & Colagrossi 2009).

    Without gravity, rho^H_ji = 0 and this is Molteni and Colagrossi's term. That term is not
    consistent with water at rest: next to the free surface and the bed the kernel support is
    incomplete, its Laplacian of the hydrostatic density is not zero, and it drives a flow in still
    water (Antuono et al. 2010). With gravity, rho^H_ji is the hydrostatic part of the pair's
    difference (`hydrostaticDensityDifference`) and the term acts on the rest, the hydrostatic
    subtraction of Fourtakas et al. (2019) taken at the local sound speed.

    Parameters:
    -----------
    masses : np.ndarray | float
        Particle masses, or one shared mass [kg]
    densities : np.ndarray
        Particle densities [kg/m^3]
    i, j : np.ndarray
        Unique pair indices
    separation, distance : np.ndarray
        r_i - r_j and its length for each pair [m]
    kernelGradients : np.ndarray
        grad W_ij for each pair, shape (pairs, dim)
    count : int
        Number of particles
    smoothingLength : float | np.ndarray
        h [m], or each pair's h_ij
    soundSpeed, delta, referenceDensity : float
        c0 [m/s], the coefficient delta and rho0 [kg/m^3]
    gravity : float | None
        Gravity's magnitude along the last axis [m/s^2] for the hydrostatic subtraction; None for
        Molteni and Colagrossi's term
    active : np.ndarray | None
        Pairs that diffuse; the solver passes fluid-fluid pairs

    '''

    if active is not None:
        i, j = i[active], j[active]
        separation, distance, kernelGradients = separation[active], distance[active], kernelGradients[active]
        if np.ndim(smoothingLength):
            smoothingLength = smoothingLength[active]
    difference = densities[j] - densities[i]
    if gravity is not None:
        difference = difference - hydrostaticDensityDifference(separation[:, -1], 0.5 * (densities[i] + densities[j]),
                                                               referenceDensity, soundSpeed, gravity)
    outward = -np.sum(separation * kernelGradients, axis = 1)
    weight = 2.0 * delta * smoothingLength * soundSpeed * difference * outward \
        / (distance**2 + viscositySoftening * smoothingLength**2)
    volumes = (masses if np.ndim(masses) else np.full(count, masses)) / densities
    rate = np.zeros(count)
    scatterAdd(rate, i, volumes[j] * weight)
    scatterAdd(rate, j, -volumes[i] * weight)

    return rate

def shiftingFields(masses, densities: np.ndarray, i: np.ndarray, j: np.ndarray, separation: np.ndarray,
                   kernelValues: np.ndarray, kernelGradients: np.ndarray, count: int,
                   spacingKernel: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    '''

    The particle concentration's gradient, with and without the anti-clumping weight, the position
    divergence and the density gradient, over every pair, boundary particles included so a wall is
    not a surface.

        grad C_i    = sum_j V_j [1 + R (W_ij / W(dx))^n] grad W_ij      (R = 0.2, n = 4)
        grad C0_i   = sum_j V_j grad W_ij
        div r_i     = -sum_j V_j r_ij . grad W_ij
        grad rho_i  = sum_j (rho_j - rho_i) V_j grad W_ij

    The anti-clumping weight (Monaghan 2000; Sun et al. 2017) grows as two particles close on each
    other, so the shift parts pairs before they clump. grad C0 points into the water at a free
    surface, and -grad C0 / |grad C0| is the surface's outward normal. div r is the dimension
    inside the water and falls toward half of it on a free surface. The density gradient carries
    each particle's density with its shift (`shiftingDisplacement`).

    Parameters:
    -----------
    masses : np.ndarray | float
        Particle masses, or one shared mass [kg]
    densities : np.ndarray
        Particle densities [kg/m^3]
    i, j : np.ndarray
        Unique pair indices
    separation : np.ndarray
        r_i - r_j for each pair [m]
    kernelValues, kernelGradients : np.ndarray
        W and grad W_ij for each pair
    count : int
        Number of particles
    spacingKernel : float
        W at the particle spacing, W(dx)

    Returns:
    --------
    tuple : grad C and grad C0, shape (count, dim) [1/m], div r, shape (count,) [-], and grad rho,
        shape (count, dim) [kg/m^4]

    '''

    volumes = (masses if np.ndim(masses) else np.full(count, masses)) / densities
    weight = 1.0 + antiClumpingWeight * (kernelValues / spacingKernel)**antiClumpingPower
    dimensions = separation.shape[1]
    gradient, plain = np.zeros((count, dimensions)), np.zeros((count, dimensions))
    scatterAdd(gradient, i, (volumes[j] * weight)[:, None] * kernelGradients)
    scatterAdd(gradient, j, -(volumes[i] * weight)[:, None] * kernelGradients)
    scatterAdd(plain, i, volumes[j][:, None] * kernelGradients)
    scatterAdd(plain, j, -volumes[i][:, None] * kernelGradients)
    term = -np.sum(separation * kernelGradients, axis = 1)
    divergence = np.zeros(count)
    scatterAdd(divergence, i, volumes[j] * term)
    scatterAdd(divergence, j, volumes[i] * term)
    # (rho_j - rho_i) grad_i W_ij for i, and (rho_i - rho_j) grad_j W_ji = the same for j
    difference = (densities[j] - densities[i])[:, None] * kernelGradients
    densityGradient = np.zeros((count, dimensions))
    scatterAdd(densityGradient, i, volumes[j][:, None] * difference)
    scatterAdd(densityGradient, j, volumes[i][:, None] * difference)

    return gradient, plain, divergence, densityGradient

def freeSurface(plain: np.ndarray, divergence: np.ndarray, isFluid: np.ndarray, smoothingLength: float,
                kernelType: str = 'cubicSpline') -> np.ndarray:

    '''

    The fluid particles on the free surface: div r below 1.5 in 2D or 2.4 in 3D, or a lopsided
    support, h |grad C0| above the kernel's threshold, 0.3 for the cubic spline and 0.4 for the
    Wendland C2 (see `freeSurfaceLopsidedness`).

    '''

    lopsided = smoothingLength * np.linalg.norm(plain, axis = 1) > freeSurfaceLopsidedness[kernelType]

    return isFluid & ((divergence < freeSurfaceDivergence[plain.shape[1]]) | lopsided)

def shiftingDisplacement(gradient: np.ndarray, plain: np.ndarray, divergence: np.ndarray, densityGradient: np.ndarray,
                         i: np.ndarray, j: np.ndarray, kernelValues: np.ndarray, isFluid: np.ndarray, size: float,
                         smoothingLength: float, kernelType: str = 'cubicSpline') -> tuple[np.ndarray, np.ndarray]:

    '''

    The shift of each particle this step, after Lind et al. (2012) [m]: delta r_i = -size grad C_i,
    and the density change that carries with it. The size is CFL Ma (2h)^2, CFL the step's acoustic
    Courant number c0 dt / h, the same for every particle (Sun et al. 2017), or A h |v_i| dt with
    A = 2, each particle's own (Skillen et al. 2013).

    Down the concentration gradient particles move from crowded to sparse regions. At a free surface
    that gradient points into the water, and the shift would carry particles out of it, so:

        on the free surface (`freeSurface`)                         no shift
        with a surface particle within the support and a
            strong gradient, h |grad C| above 0.03                  no shift
        with a surface particle within the support otherwise        the shift along the surface,
                                                                    (I - n n) delta r
        elsewhere                                                   the full shift

    The surface's outward normal at a surface particle is -grad C0 / |grad C0|, and next to it n is
    the kernel-weighted mean of the surface normals within the support. A particle's own gradient
    would not do there: two layers down it is the particles' disorder, and a projection across it
    drops a random component of the shift. One layer down the gradient is the surface's, strong
    (`surfaceGradient`) and outward, a shift of 0.009 spacings a step in a 15 ppd tank, so any error
    in n leaks it along the surface: the dry wall particles above a tank's waterline tilt n by a
    few degrees at the corner, and a tilt of 2.6 degrees moves those particles toward the wall by
    4e-4 of a spacing every step. Those particles do not shift.

    Boundary particles do not shift. A shifted particle takes the density at its new position,
    rho_i + delta r_i . grad rho_i, the first-order Taylor correction of Lind et al. (2012) and
    Skillen et al. (2013), and the velocity there (`shiftedVelocityChange`).

    Parameters:
    -----------
    gradient, plain, divergence, densityGradient : np.ndarray
        grad C, grad C0, div r and grad rho (`shiftingFields`)
    i, j : np.ndarray
        Unique pair indices
    kernelValues : np.ndarray
        W for each pair
    isFluid : np.ndarray
        True for fluid particles
    size : float | np.ndarray
        The shift's size, one for every particle or each particle's own [m^2]
    smoothingLength : float
        h [m]
    kernelType : str
        'cubicSpline' or 'wendlandC2', for the free surface's lopsidedness threshold

    Returns:
    --------
    tuple : the shifts, shape (count, dim) [m], and the density changes, shape (count,) [kg/m^3]

    '''

    surface = freeSurface(plain, divergence, isFluid, smoothingLength, kernelType)
    length = np.linalg.norm(plain, axis = 1)
    surfaceNormal = np.zeros_like(plain)
    surfaceNormal[surface] = -plain[surface] / length[surface, None]
    # Each particle's kernel-weighted sum of its surface neighbors' normals
    normal = np.zeros_like(plain)
    scatterAdd(normal, i, kernelValues[:, None] * surfaceNormal[j])
    scatterAdd(normal, j, kernelValues[:, None] * surfaceNormal[i])
    vicinity = np.zeros(len(isFluid), dtype = bool)
    vicinity[i[surface[j]]] = True
    vicinity[j[surface[i]]] = True
    weight = np.linalg.norm(normal, axis = 1)
    normal = np.divide(normal, weight[:, None], out = np.zeros_like(normal), where = weight[:, None] > 0.0)
    shift = -np.reshape(size, (-1, 1)) * gradient
    shift[vicinity] -= np.sum(shift[vicinity] * normal[vicinity], axis = 1)[:, None] * normal[vicinity]
    strong = vicinity & (smoothingLength * np.linalg.norm(gradient, axis = 1) > surfaceGradient)
    shift[surface | strong | ~isFluid] = 0.0

    return shift, np.sum(shift * densityGradient, axis = 1)

def shiftedVelocityChange(masses, densities: np.ndarray, velocities: np.ndarray, i: np.ndarray, j: np.ndarray,
                          kernelGradients: np.ndarray, shift: np.ndarray, active: np.ndarray) -> np.ndarray:

    '''

    The velocity a shifted particle takes at its new position, to first order [m/s]:

        delta v_i = (delta r_i . grad) v_i = sum_j V_j (v_j - v_i) (grad W_ij . delta r_i)

    over fluid pairs, so a wall's resting particles do not drag the velocity there. A shift moves a
    particle across the flow; without this it carries its old velocity to where the flow is
    different, which diffuses momentum: the reference-sized shift without it takes 26 percent off
    a solitary wave's height over 30 depths at 32 particles per depth, against 10 percent with no
    shifting.

    Parameters:
    -----------
    masses : np.ndarray | float
        Particle masses, or one shared mass [kg]
    densities, velocities : np.ndarray
        Particle densities [kg/m^3] and velocities [m/s]
    i, j : np.ndarray
        Unique pair indices
    kernelGradients : np.ndarray
        grad W_ij for each pair
    shift : np.ndarray
        This step's shifts, shape (count, dim) [m]
    active : np.ndarray
        Pairs that carry velocity: the solver passes fluid-fluid pairs

    '''

    count = len(densities)
    i, j, gradients = i[active], j[active], kernelGradients[active]
    volumes = (masses if np.ndim(masses) else np.full(count, masses)) / densities
    difference = velocities[j] - velocities[i]
    change = np.zeros_like(velocities)
    scatterAdd(change, i, (volumes[j] * np.sum(gradients * shift[i], axis = 1))[:, None] * difference)
    scatterAdd(change, j, (volumes[i] * np.sum(gradients * shift[j], axis = 1))[:, None] * difference)

    return change

def shepardDensity(masses, densities: np.ndarray, i: np.ndarray, j: np.ndarray, kernelValues: np.ndarray,
                   selfKernel: float) -> np.ndarray:

    '''

    The Shepard-filtered density rho_i = sum_j m_j W_ij / sum_j (m_j / rho_j) W_ij [kg/m^3].

    The denominator is the SPH interpolant of one, so the ratio is a kernel average of the density
    corrected for an incomplete support: a uniform density comes back unchanged, at the free
    surface as well as inside the fluid.

    '''

    count = len(densities)
    numerator = summationDensity(masses, i, j, kernelValues, selfKernel, count)
    volume = (masses if np.ndim(masses) else np.full(count, masses)) / densities
    denominator = volume * selfKernel
    scatterAdd(denominator, i, volume[j] * kernelValues)
    scatterAdd(denominator, j, volume[i] * kernelValues)

    return numerator / denominator

def extrapolatedWallPressure(densities: np.ndarray, pressures: np.ndarray, i: np.ndarray, j: np.ndarray,
                             separation: np.ndarray, kernelValues: np.ndarray, isFluid: np.ndarray,
                             gravity: np.ndarray) -> np.ndarray:

    '''

    Each boundary particle's pressure from the fluid around it (Adami et al. 2012, eq. 27) [Pa]:

        p_w = (sum_f p_f W_wf + g . sum_f rho_f r_wf W_wf) / sum_f W_wf,   r_wf = r_w - r_f

    The first sum is the kernel average of the fluid's pressure; the second carries each fluid
    particle's pressure down to the wall particle through the hydrostatic increment, so a fluid at
    rest gives every wall particle its own hydrostatic pressure, however one-sided its support.
    Integrated from the continuity equation instead, a wall particle's density sees only the fluid
    on one side of it and never the flow's compensating vertical divergence: under a passing wave
    the inner layers keep the still-water pressure, and the fluid next to the bed loses the
    horizontal pressure gradient that drives it. Negative pressures are clamped to zero, as the
    fluid's are, and a boundary particle with no fluid neighbor gets zero. The fluid entries are
    returned as given.

    Parameters:
    -----------
    densities, pressures : np.ndarray
        Every particle's density [kg/m^3] and pressure [Pa], shape (N,)
    i, j : np.ndarray
        Pair indices, each pair once
    separation : np.ndarray
        r_i - r_j for each pair, shape (P, dim) [m]
    kernelValues : np.ndarray
        W_ij for each pair, shape (P,) [1/m^dim]
    isFluid : np.ndarray
        Fluid mask, shape (N,)
    gravity : np.ndarray
        Body acceleration less the wall's own acceleration, shape (dim,) [m/s^2]

    Returns:
    --------
    np.ndarray : the pressures with the boundary entries replaced, shape (N,) [Pa]

    '''

    count = len(pressures)
    # Pairs of a wall particle and a fluid particle, oriented wall first
    wallFirst = ~isFluid[i] & isFluid[j]
    fluidFirst = isFluid[i] & ~isFluid[j]
    wall = np.concatenate([i[wallFirst], j[fluidFirst]])
    water = np.concatenate([j[wallFirst], i[fluidFirst]])
    toWall = np.concatenate([separation[wallFirst], -separation[fluidFirst]])
    weights = np.concatenate([kernelValues[wallFirst], kernelValues[fluidFirst]])

    weight = np.zeros(count)
    total = np.zeros(count)
    scatterAdd(weight, wall, weights)
    scatterAdd(total, wall, weights * (pressures[water] + densities[water] * (toWall @ np.asarray(gravity))))

    result = pressures.copy()
    boundary = ~isFluid
    reached = boundary & (weight > 0.0)
    result[boundary] = 0.0
    result[reached] = np.maximum(total[reached] / weight[reached], 0.0)

    return result

def hydrostaticDensity(depthBelowSurface: np.ndarray, referenceDensity: float, soundSpeed: float,
                       gravity: float = constants.gravity) -> np.ndarray:

    '''

    The density that carries hydrostatic pressure rho0 g z at a depth z below the free surface,
    through the inverse Tait equation [kg/m^3]. Depths above the surface give the reference density.

    '''

    pressure = referenceDensity * gravity * np.maximum(np.asarray(depthBelowSurface, dtype = float), 0.0)

    return taitDensity(pressure, referenceDensity, soundSpeed)

def artificialViscosity(relativeVelocity: np.ndarray, separation: np.ndarray, distance: np.ndarray,
                        meanDensity: np.ndarray, smoothingLength: float, soundSpeed: float,
                        alpha: float, reconstructed: np.ndarray | None = None) -> np.ndarray:

    '''

    Monaghan's artificial viscosity term Pi_ij for each pair [m^5/(kg s^2)].

    Pi_ij = -alpha c0 mu_ij / rho_bar for approaching pairs (v_ij . r_ij < 0) and zero for
    separating ones, with mu_ij = h v_ij . r_ij / (r_ij^2 + 0.01 h^2). It is positive when
    particles approach, so it adds a repulsion that damps the approach, and it conserves linear
    and angular momentum because it acts along r_ij.

    With `reconstructed`, each pair's velocity difference less its linear part
    (`limitedVelocityDifference`), mu_ij takes that difference, and the term acts where both it and
    v_ij approach. Requiring both keeps every pair's dissipation, Pi_ij v_ij . grad W_ij, from
    going negative: the term never feeds energy into the flow.

    '''

    approach = np.sum(relativeVelocity * separation, axis = 1)
    acting = approach < 0.0
    if reconstructed is not None:
        approach = np.sum(reconstructed * separation, axis = 1)
        acting &= approach < 0.0
    mu = smoothingLength * approach / (distance**2 + viscositySoftening * smoothingLength**2)

    return np.where(acting, -alpha * soundSpeed * mu / meanDensity, 0.0)

def velocityGradients(masses, densities: np.ndarray, velocities: np.ndarray, i: np.ndarray, j: np.ndarray,
                      separation: np.ndarray, kernelGradients: np.ndarray, isFluid: np.ndarray) -> np.ndarray:

    '''

    Each fluid particle's velocity gradient G_i, G_ab = dv_a / dx_b, from its fluid neighbors
    [1/s], shape (N, dim, dim).

    G_i = S_i M_i^-1, with S_i = sum_j V_j (v_j - v_i) (x) grad_i W_ij and M_i = sum_j V_j
    (x_j - x_i) (x) grad_i W_ij, V_j = m_j / rho_j: for a linear field v = G x, S_i = G M_i, so the
    gradient is exact wherever M can be inverted, beside a free surface too. Only fluid neighbors
    count, so a resting wall does not read as shear next to it. G is zero on the boundary and
    wherever det M falls below `correctionFloor` to the power of the dimensions.

    '''

    count, dimensions = len(densities), separation.shape[1]
    volumes = (masses if np.ndim(masses) else np.full(count, float(masses))) / np.asarray(densities)
    fluidPairs = isFluid[i] & isFluid[j]
    i, j = i[fluidPairs], j[fluidPairs]
    gradient = kernelGradients[fluidPairs]
    # Both members of a pair add the same outer products, each weighted by the other's volume:
    # (v_i - v_j) (x) grad_j W_ji = (v_j - v_i) (x) grad_i W_ij, and likewise for the separation
    rise = (velocities[j] - velocities[i])[:, :, None] * gradient[:, None, :]
    offset = -separation[fluidPairs][:, :, None] * gradient[:, None, :]
    sums = np.zeros((count, dimensions, dimensions))
    matrices = np.zeros((count, dimensions, dimensions))
    for a in range(dimensions):
        for b in range(dimensions):
            for target, outer in ((sums, rise), (matrices, offset)):
                column = np.zeros(count)
                scatterAdd(column, i, volumes[j] * outer[:, a, b])
                scatterAdd(column, j, volumes[i] * outer[:, a, b])
                target[:, a, b] = column

    gradients = np.zeros((count, dimensions, dimensions))
    applies = isFluid & (np.linalg.det(matrices) >= correctionFloor**dimensions)
    gradients[applies] = sums[applies] @ np.linalg.inv(matrices[applies])

    return gradients

def limitedVelocityDifference(relativeVelocity: np.ndarray, separation: np.ndarray, distance: np.ndarray,
                              smoothingLength, gradientsI: np.ndarray, gradientsJ: np.ndarray) -> np.ndarray:

    '''

    Each pair's velocity difference less its linear part under a slope limiter [m/s]: v_ij -
    (phi_ij / 2) (G_i + G_j) r_ij, the difference of the two particles' velocities reconstructed
    to the pair's midpoint (Frontiere et al. 2017; Rosswog 2020).

    phi_ij = max(0, min(1, 4 A / (1 + A)^2)), A = (r . G_i r) / (r . G_j r), MAGMA2's limiter: one
    where the two gradients agree along the pair, falling to zero as they part and where they
    disagree in sign, as at a crest where the strain turns over, so there the term sees the whole
    v_ij. It is symmetric in i and j (A becomes 1 / A), so the term stays antisymmetric and
    conserves momentum. Pairs closer than `limiterCloseDistance` h fade the reconstruction out.

    Parameters:
    -----------
    relativeVelocity, separation : np.ndarray
        v_i - v_j and x_i - x_j for each pair, shape (pairs, dim)
    distance : np.ndarray
        |x_i - x_j| for each pair [m]
    smoothingLength : float | np.ndarray
        h, or each pair's h_ij [m]
    gradientsI, gradientsJ : np.ndarray
        Each pair's two velocity gradients (`velocityGradients`), shape (pairs, dim, dim) [1/s]

    '''

    rampI = np.einsum('pab,pb->pa', gradientsI, separation)
    rampJ = np.einsum('pab,pb->pa', gradientsJ, separation)
    slopeI = np.sum(separation * rampI, axis = 1)
    slopeJ = np.sum(separation * rampJ, axis = 1)
    agree = slopeI * slopeJ > 0.0
    ratio = np.divide(slopeI, slopeJ, out = np.zeros_like(slopeI), where = agree)
    limiter = np.where(agree, np.minimum(1.0, 4.0 * ratio / (1.0 + ratio)**2), 0.0)
    eta = distance / smoothingLength
    limiter *= np.where(eta < limiterCloseDistance, np.exp(-((eta - limiterCloseDistance) / limiterFoldWidth)**2), 1.0)

    return relativeVelocity - 0.5 * limiter[:, None] * (rampI + rampJ)

def pairAccelerations(masses: np.ndarray, densities: np.ndarray, pressures: np.ndarray, i: np.ndarray,
                      j: np.ndarray, kernelGradients: np.ndarray, viscosity: np.ndarray,
                      count: int) -> np.ndarray:

    '''

    Accelerations from the symmetric pressure gradient and the artificial viscosity [m/s^2].

    a_i = -sum_j m_j (p_i / rho_i^2 + p_j / rho_j^2 + Pi_ij) grad W_ij, with grad W_ij the
    gradient with respect to particle i. Every pair applies equal and opposite forces m_i a_i and
    m_j a_j, so total momentum is conserved to rounding.

    Parameters:
    -----------
    kernelGradients : np.ndarray
        grad W_ij for each pair, shape (pairs, dim)
    viscosity : np.ndarray
        Pi_ij for each pair
    count : int
        Number of particles

    '''

    coefficient = -(pressures[i] / densities[i]**2 + pressures[j] / densities[j]**2 + viscosity)
    accelerations = np.zeros((count, kernelGradients.shape[1]))
    scatterAdd(accelerations, i, (_pairMass(masses, j) * coefficient)[:, None] * kernelGradients)
    scatterAdd(accelerations, j, -(_pairMass(masses, i) * coefficient)[:, None] * kernelGradients)

    return accelerations

def xsphVelocity(velocities: np.ndarray, masses: np.ndarray, densities: np.ndarray, i: np.ndarray,
                 j: np.ndarray, kernelValues: np.ndarray, epsilon: float) -> np.ndarray:

    '''

    The XSPH transport velocity: v_i + epsilon sum_j (m_j / rho_bar_ij) (v_j - v_i) W_ij [m/s].

    The pairs passed in should be fluid-fluid pairs: a resting wall particle in the sum would drag
    the fluid's transport velocity toward zero at the wall. The result moves positions only; the
    particle velocities are not touched.

    '''

    weight = kernelValues / (0.5 * (densities[i] + densities[j]))
    difference = velocities[j] - velocities[i]
    correction = np.zeros_like(velocities)
    scatterAdd(correction, i, (_pairMass(masses, j) * weight)[:, None] * difference)
    scatterAdd(correction, j, -(_pairMass(masses, i) * weight)[:, None] * difference)

    return velocities + epsilon * correction

def cflTimeStep(smoothingLength: float, soundSpeed: float, maxSpeed: float, maxAcceleration: float,
                cflNumber: float, maxTimeStep: float) -> float:

    '''

    The adaptive time step: the smaller of the acoustic limit CFL h / (c0 + |v|max) and the force
    limit CFL sqrt(h / |a|max), and never more than maxTimeStep [s].

    '''

    timeStep = cflNumber * smoothingLength / (soundSpeed + maxSpeed)
    if maxAcceleration > 1e-12:
        timeStep = min(timeStep, cflNumber * math.sqrt(smoothingLength / maxAcceleration))

    return min(timeStep, maxTimeStep)

#----------------------------------------------------------------------#
# -- Solver -- #
#----------------------------------------------------------------------#

class WcsphSolver:

    '''

    The WCSPH time-stepping loop.

    Boundary particles take part in density and pressure but are never integrated; a wave maker
    may move them between steps by writing their positions and velocities directly.

    Parameters:
    -----------
    config : SimulationConfig
        Domain, resolution and numerical parameters
    kernel : SphKernel | None
        Smoothing kernel; the configured kernel type by default
    boundary : ContainerBoundary | BedBoundary | None
        Anything with an `enforce(particles)` method, called after each integration

    '''

    def __init__(self, config: SimulationConfig, kernel: SphKernel | None = None, boundary = None) -> None:

        self._config = config
        self._kernel = kernel or createKernel(config.kernelType, config.dimensions)
        self._boundary = boundary

        self._particles: ParticleSystem | None = None
        self._time: float = 0.0
        self._step: int = 0
        self._dt: float = config.maxTimeStep
        self._pairs = (np.array([], dtype = np.intp), np.array([], dtype = np.intp))
        self._pairLength: float | np.ndarray = config.smoothingLength  # [m], one, or each pair's h_ij
        self._gravity = abs(float(np.asarray(config.gravity)[-1]))     # [m/s^2]

        # Pairs across the seam count one image only: a particle must not reach two images of
        # another, which needs the period longer than twice the kernel's support
        if config.periodicWidth is not None and config.periodicWidth <= 2.0 * config.supportRadius:
            raise ValueError(f'The periodic width {config.periodicWidth} m must exceed twice the kernel support, '
                             f'{2.0 * config.supportRadius} m.')
        if config.backend not in ('numpy', 'compiled', 'gpu'):
            raise ValueError(f"Backend '{config.backend}' is not 'numpy', 'compiled' or 'gpu'.")
        if config.backend == 'compiled' and config.kernelType != 'cubicSpline':
            raise ValueError('The compiled backend has the cubic spline kernel only.')
        if config.shiftingSpeed not in ('reference', 'local'):
            raise ValueError(f"Shifting speed '{config.shiftingSpeed}' is not 'reference' or 'local'.")
        if config.densityDiffusion not in ('none', 'molteni', 'hydrostatic'):
            raise ValueError(f"Density diffusion '{config.densityDiffusion}' is not 'none', 'molteni' or 'hydrostatic'.")
        if config.wallPressure not in ('continuity', 'extrapolated'):
            raise ValueError(f"Wall pressure '{config.wallPressure}' is not 'continuity' or 'extrapolated'.")
        if (config.densityDiffusion != 'none' or config.particleShifting or config.wallPressure != 'continuity'
                or config.negativePressure or config.continuityCorrection
                or config.viscosityLimiter) and config.backend == 'compiled':
            raise ValueError('The compiled backend has no density diffusion, particle shifting, extrapolated wall '
                             'pressure, negative pressure, continuity correction or viscosity limiter; use the numpy '
                             'or gpu backend.')
        self._compiled = config.backend == 'compiled'
        self._gpu = None
        if self._compiled:
            from pysurf import sphCompiled
            self._sph = sphCompiled
            self._sigma = sphCompiled.cubicSplineNormalization(config.smoothingLength, config.dimensions)
            self._period = config.periodicWidth or 0.0
        elif config.backend == 'gpu':
            from pysurf import sphGpu
            self._gpu = sphGpu.PairSums(config.dimensions, config.smoothingLength, config.domainMin, config.domainMax,
                                        config.periodicWidth or 0.0, kernelType = config.kernelType)

        if config.speedOfSound is not None:
            self._speedOfSound = float(config.speedOfSound)
        else:
            self._speedOfSound = speedOfSound(float(config.domainSize[-1]), config.speedOfSoundFactor, self._gravity)
        if self._gpu is not None:
            self._gpu.setDiffusion(config.densityDiffusion, config.diffusionDelta, self._speedOfSound,
                                   config.referenceDensity, self._gravity)
            self._gpu.setShifting(config.particleShifting, config.shiftingMach, config.particleSpacing,
                                  self._speedOfSound, config.shiftingSpeed == 'local')
            self._gpu.setWallPressure(config.wallPressure == 'extrapolated')
            self._gpu.setNegativePressure(config.negativePressure)
            self._gpu.setContinuityCorrection(config.continuityCorrection)
            self._gpu.setViscosityLimiter(config.viscosityLimiter)

    # -- Private Helper Methods -- #
    def _updatePairs(self) -> None:

        '''Pair the particles at their current positions, with each pair's separation, distance and W.'''

        p = self._particles
        period = self._config.periodicWidth
        if self._gpu is not None:
            self._gpu.update(p.positions, p.masses, p.isFluid, p.lengths)
            return
        if self._compiled:
            self._offsets, self._neighbors = self._sph.neighborList(p.positions, self._config.supportRadius,
                                                                    self._period)
            return
        lengths = p.lengths
        reach = self._config.supportRadius if lengths is None else 2.0 * float(lengths.max())
        self._pairs = neighborPairs(p.positions, reach, period)
        i, j = self._pairs
        self._separation = p.positions[i] - p.positions[j]
        if period is not None:
            # The nearest image across the periodic seam
            self._separation[:, 1] -= period * np.round(self._separation[:, 1] / period)
        self._distance = np.linalg.norm(self._separation, axis = 1)
        if lengths is None:
            self._pairLength = self._config.smoothingLength
        else:
            # Particles of different sizes pair at the mean of their lengths, within its support
            pairLength = 0.5 * (lengths[i] + lengths[j])
            keep = self._distance < 2.0 * pairLength
            self._pairs = (i[keep], j[keep])
            i, j = self._pairs
            self._separation, self._distance, self._pairLength = self._separation[keep], self._distance[keep], pairLength[keep]
        self._kernelValues = self._kernel.evaluateBatch(self._distance, self._pairLength)
        self._boundaryPairs = ~(p.isFluid[i] & p.isFluid[j])
        self._fluidTouching = p.isFluid[i] | p.isFluid[j]

    def _filterDensity(self) -> None:

        '''Reset the fluid density to its Shepard-filtered value; the boundary keeps its own.'''

        p = self._particles
        if self._gpu is not None:
            filtered = self._gpu.shepardDensity(p.densities)
        elif self._compiled:
            filtered = self._sph.shepardDensity(p.positions, p.densities, p.masses, self._offsets, self._neighbors,
                                                self._config.smoothingLength, self._sigma, self._period)
        else:
            selfKernel = (self._kernel.evaluate(0.0, self._config.smoothingLength) if p.lengths is None
                          else self._kernel.evaluateBatch(np.zeros(p.nParticles), p.lengths))
            filtered = shepardDensity(self._masses, p.densities, *self._pairs, self._kernelValues, selfKernel)
        p.densities[p.isFluid] = filtered[p.isFluid]

    def _pressure(self, densities: np.ndarray, isFluid: np.ndarray) -> np.ndarray:

        '''Tait pressure, clamped at zero on the boundary and, unless `negativePressure`, in the fluid [Pa].'''

        config = self._config
        pressures = taitEquationOfState(densities, config.referenceDensity, self._speedOfSound,
                                        clampNegative = not config.negativePressure)
        if config.negativePressure:
            pressures[~isFluid] = np.maximum(pressures[~isFluid], 0.0)

        return pressures

    def _computeRates(self) -> None:

        '''Pressure from the current density, then the acceleration of the fluid; zero on the boundary.'''

        p, config = self._particles, self._config

        p.pressures = self._pressure(p.densities, p.isFluid)
        if config.wallPressure == 'extrapolated':
            boundary = ~p.isFluid
            if self._gpu is not None:
                p.pressures = self._gpu.wallPressure(p.densities, p.pressures, np.asarray(config.gravity, dtype = float),
                                                     config.referenceDensity, self._speedOfSound)
            else:
                p.pressures = extrapolatedWallPressure(p.densities, p.pressures, *self._pairs, self._separation,
                                                       self._kernelValues, p.isFluid, np.asarray(config.gravity))
            # The boundary carries the density of its pressure, which the pressure term divides by
            p.densities[boundary] = taitDensity(p.pressures[boundary], config.referenceDensity, self._speedOfSound)
        if self._gpu is not None:
            p.accelerations = self._gpu.pairAccelerations(p.velocities, p.densities, p.pressures, self._speedOfSound,
                                                          config.alphaViscosity, config.boundaryViscosityFactor,
                                                          np.asarray(config.gravity, dtype = float))
            return
        if self._compiled:
            p.accelerations = self._sph.pairAccelerations(
                p.positions, p.velocities, p.densities, p.pressures, p.masses, p.isFluid, self._offsets,
                self._neighbors, config.smoothingLength, self._sigma, self._speedOfSound, config.alphaViscosity,
                config.boundaryViscosityFactor, np.asarray(config.gravity, dtype = float), self._period)
            return
        i, j = self._pairs

        # Every pair is evaluated, boundary-boundary ones included: they are a few percent of the
        # total, and masking them out copies every pair array, which costs more than it saves
        gradients = self._kernel.gradientBatch(self._separation, self._distance, self._pairLength)
        relativeVelocity = p.velocities[i] - p.velocities[j]
        reconstructed = None
        if config.viscosityLimiter:
            velocityGradient = velocityGradients(self._masses, p.densities, p.velocities, i, j, self._separation,
                                                 gradients, p.isFluid)
            reconstructed = limitedVelocityDifference(relativeVelocity, self._separation, self._distance,
                                                      self._pairLength, velocityGradient[i], velocityGradient[j])
            reconstructed[self._boundaryPairs] = relativeVelocity[self._boundaryPairs]
        viscosity = artificialViscosity(relativeVelocity, self._separation, self._distance,
                                        0.5 * (p.densities[i] + p.densities[j]),
                                        self._pairLength, self._speedOfSound, config.alphaViscosity, reconstructed)
        if config.boundaryViscosityFactor != 1.0:
            viscosity[self._boundaryPairs] *= config.boundaryViscosityFactor

        accelerations = pairAccelerations(self._masses, p.densities, p.pressures, i, j, gradients,
                                          viscosity, p.nParticles)
        accelerations[p.isFluid] += np.asarray(config.gravity)
        accelerations[~p.isFluid] = 0.0
        p.accelerations = accelerations
        self._kernelGradients = gradients

    def _integrate(self, timeStep: float) -> None:

        '''

        Kick the fluid velocity, update every particle's density from the kicked velocity, then
        drift the fluid with the XSPH transport velocity.

        The density rate must see the kicked velocity. Pressure drives velocity and velocity
        divergence drives density, which is an acoustic oscillator; advancing both from old values
        is forward Euler on that oscillator, which grows without bound however small the step.
        Using the new velocity makes the pair symplectic Euler, which is stable under the acoustic
        CFL limit.

        '''

        p, config = self._particles, self._config
        fluid = p.isFluid
        p.velocities[fluid] += p.accelerations[fluid] * timeStep
        if self._compiled or self._gpu is not None:
            gpu, h = self._gpu, config.smoothingLength
            if gpu is not None:
                rate = gpu.continuityRate(p.velocities, p.densities)
            else:
                rate = self._sph.continuityRate(p.positions, p.velocities, p.masses, fluid, self._offsets,
                                                self._neighbors, h, self._sigma, self._period)
            p.densities += rate * timeStep
            if config.densityFloor > 0.0:
                p.densities[fluid] = np.maximum(p.densities[fluid], config.densityFloor * config.referenceDensity)
            transport = p.velocities
            if config.xsphEpsilon > 0.0 and gpu is not None:
                transport = gpu.xsphVelocity(p.velocities, p.densities, config.xsphEpsilon)
            elif config.xsphEpsilon > 0.0:
                transport = self._sph.xsphVelocity(p.positions, p.velocities, p.densities, p.masses, fluid,
                                                   self._offsets, self._neighbors, h, self._sigma, config.xsphEpsilon,
                                                   self._period)
            p.positions[fluid] += transport[fluid] * timeStep
            if config.particleShifting:
                shift, densityChange, velocityChange = gpu.shiftingDisplacement(p.densities, timeStep, p.velocities)
                p.positions[fluid] += shift[fluid]
                p.densities[fluid] += densityChange[fluid]
                p.velocities[fluid] += velocityChange[fluid]
            if config.periodicWidth is not None:
                p.positions[fluid, 1] = np.mod(p.positions[fluid, 1], config.periodicWidth)
            return
        i, j = self._pairs
        corrections = (correctionMatrices(self._masses, p.densities, i, j, self._separation, self._kernelGradients,
                                          fluid) if config.continuityCorrection else None)
        rate = continuityRate(self._masses, p.velocities, i, j, self._kernelGradients, p.nParticles,
                              self._fluidTouching, corrections)
        if config.densityDiffusion != 'none':
            # The diffusion sees the densities the pressures came from, before this step's update
            rate += densityDiffusionRate(self._masses, p.densities, i, j, self._separation, self._distance,
                                         self._kernelGradients, p.nParticles, self._pairLength,
                                         self._speedOfSound, config.diffusionDelta, config.referenceDensity,
                                         self._gravity if config.densityDiffusion == 'hydrostatic' else None,
                                         ~self._boundaryPairs)
        p.densities += rate * timeStep
        if config.densityFloor > 0.0:
            p.densities[fluid] = np.maximum(p.densities[fluid], config.densityFloor * config.referenceDensity)

        transport = p.velocities
        if config.xsphEpsilon > 0.0:
            i, j = self._pairs
            fluidPairs = fluid[i] & fluid[j]
            transport = xsphVelocity(p.velocities, self._masses, p.densities, i[fluidPairs], j[fluidPairs],
                                     self._kernelValues[fluidPairs], config.xsphEpsilon)
        p.positions[fluid] += transport[fluid] * timeStep
        if config.particleShifting:
            fields = shiftingFields(self._masses, p.densities, i, j, self._separation, self._kernelValues,
                                    self._kernelGradients, p.nParticles,
                                    self._kernel.evaluate(config.particleSpacing, config.smoothingLength))
            shift, densityChange = shiftingDisplacement(*fields, i, j, self._kernelValues, fluid,
                                                        self._shiftingSize(timeStep), config.smoothingLength,
                                                        config.kernelType)
            velocityChange = shiftedVelocityChange(self._masses, p.densities, p.velocities, i, j,
                                                   self._kernelGradients, shift, ~self._boundaryPairs)
            p.positions[fluid] += shift[fluid]
            p.densities[fluid] += densityChange[fluid]
            p.velocities[fluid] += velocityChange[fluid]
        if config.periodicWidth is not None:
            p.positions[fluid, 1] = np.mod(p.positions[fluid, 1], config.periodicWidth)

    def _shiftingSize(self, timeStep: float) -> float | np.ndarray:

        '''

        The shift's size this step [m^2]: CFL Ma (2h)^2 with CFL = c0 dt / h, or each particle's
        A h |v_i| dt.

        '''

        h = self._config.smoothingLength
        if self._config.shiftingSpeed == 'local':
            return localShiftingCoefficient * h * np.linalg.norm(self._particles.velocities, axis = 1) * timeStep
        return (self._speedOfSound * timeStep / h) * self._config.shiftingMach * (2.0 * h)**2

    # -- Public Methods -- #
    def initialize(self, particles: ParticleSystem) -> None:

        '''

        Take a particle system and set its starting density: hydrostatic beneath the configured
        initial surface, fluid and boundary alike, or the reference density without one.

        '''

        self._particles = particles
        self._time = 0.0
        self._step = 0

        # One shared mass, the usual case for a lattice-seeded run, lets every pair sum skip
        # gathering two mass arrays
        uniform = particles.masses.size > 0 and bool(np.all(particles.masses == particles.masses[0]))
        self._masses = float(particles.masses[0]) if uniform else particles.masses

        surface = self._config.initialSurface
        if surface is None:
            particles.densities = np.full(particles.nParticles, self._config.referenceDensity)
        else:
            x = particles.positions[:, 0]
            surfaceElevation = surface(x) if callable(surface) else np.full_like(x, float(surface))
            particles.densities = hydrostaticDensity(surfaceElevation - particles.positions[:, -1],
                                                     self._config.referenceDensity, self._speedOfSound, self._gravity)

        particles.pressures = self._pressure(particles.densities, particles.isFluid)
        self._updatePairs()

    def step(self) -> SimulationState:

        '''

        Advance one adaptive time step.

        Returns:
        --------
        SimulationState : Diagnostics after the step

        '''

        p, config = self._particles, self._config

        self._updatePairs()
        if config.densityFilterInterval and self._step > 0 and self._step % config.densityFilterInterval == 0:
            self._filterDensity()
        self._computeRates()

        fluidAccelerations = p.accelerations[p.isFluid]
        maxAcceleration = float(np.max(np.linalg.norm(fluidAccelerations, axis = 1))) if len(fluidAccelerations) else 0.0
        self._dt = cflTimeStep(self.shortestLength, self._speedOfSound, p.maxSpeed(), maxAcceleration,
                               config.cflNumber, config.maxTimeStep)

        self._integrate(self._dt)
        if self._boundary is not None:
            self._boundary.enforce(p)

        self._time += self._dt
        self._step += 1

        return self.currentState

    def advance(self, untilTime: float, maxSteps: int | None = None) -> SimulationState:

        '''

        Step until the simulated time reaches `untilTime`, or `maxSteps` steps.

        On the GPU backend, with a boundary the device can enforce (`sphGpu.PairSums.canEnforce`),
        every step runs on the device: the state goes up once, the steps run there, and the state
        comes back once, so the particle arrays hold the state at `untilTime` and nothing between.
        Anything that changes the particles between steps (a moving wall, particles added) needs
        `step`; a piston holds its position and speed through the call. Otherwise this is `step` in
        a loop.

        Parameters:
        -----------
        untilTime : float
            Simulated time to stop at or just past [s]
        maxSteps : int | None
            Most steps to take; None for no limit

        Returns:
        --------
        SimulationState : Diagnostics at the end

        '''

        p, config, gpu = self._particles, self._config, self._gpu
        lastStep = self._step + maxSteps if maxSteps is not None else None
        if gpu is None or not gpu.canEnforce(self._boundary):
            while self._time < untilTime and (lastStep is None or self._step < lastStep):
                self.step()
            return self.currentState

        gpu.setPhysics(config.referenceDensity, self._speedOfSound, config.alphaViscosity,
                       config.boundaryViscosityFactor, np.asarray(config.gravity, dtype = float), config.xsphEpsilon,
                       config.densityFloor * config.referenceDensity)
        gpu.push(p)
        gpu.setBoundary(self._boundary, p)
        interval = config.densityFilterInterval
        while self._time < untilTime and (lastStep is None or self._step < lastStep):
            maxSpeed, maxAcceleration = gpu.residentRates(bool(interval) and self._step > 0 and self._step % interval == 0)
            self._dt = cflTimeStep(self.shortestLength, self._speedOfSound, maxSpeed, maxAcceleration,
                                   config.cflNumber, config.maxTimeStep)
            gpu.residentIntegrate(self._dt)
            self._time += self._dt
            self._step += 1
        gpu.pull(p)

        return self.currentState

    def refine(self, mask: np.ndarray, lengthRatio: float = 1.0) -> int:

        '''

        Split the fluid particles `mask` selects, each into 2^dim daughters of half the spacing
        (`sphRefinement.splitParticles`), between steps. The boundary keeps
        its particles and their indices; a piston's mask grows with the particle arrays, and the
        boundary is enforced on the daughters.

        Parameters:
        -----------
        mask : np.ndarray
            Particles to split, shape (N,); boundary particles in it are left alone
        lengthRatio : float
            A daughter's smoothing length over its mother's: 1 keeps h, 0.5 keeps h / s

        Returns:
        --------
        int : How many particles split

        Raises:
        -------
        ValueError : On the compiled backend, or with particle shifting, which take one smoothing
            length throughout.

        '''

        config, p = self._config, self._particles
        if self._compiled:
            raise ValueError('The compiled backend takes one smoothing length throughout; refine on numpy or gpu.')
        if config.particleShifting:
            raise ValueError('Particle shifting takes one smoothing length throughout.')
        before = p.nParticles
        count = splitParticles(p, mask, config.referenceDensity, config.smoothingLength, lengthRatio)
        if count == 0:
            return 0
        self._masses = p.masses
        pistonMask = getattr(self._boundary, '_pistonMask', None)
        if pistonMask is not None:
            self._boundary._pistonMask = np.concatenate([pistonMask, np.zeros(p.nParticles - before, dtype = bool)])
        if self._boundary is not None:
            self._boundary.enforce(p)
        self._updatePairs()

        return count

    @property
    def shortestLength(self) -> float:
        '''The shortest smoothing length among the particles, which sets the time step [m].'''
        p = self._particles
        if p is None or p.lengths is None:
            return self._config.smoothingLength
        return float(p.lengths.min())

    @property
    def currentState(self) -> SimulationState:
        '''Diagnostics of the current instant.'''
        p = self._particles
        return SimulationState(
            time            = self._time,
            step            = self._step,
            dt              = self._dt,
            kineticEnergy   = p.kineticEnergy(),
            potentialEnergy = p.potentialEnergy(self._gravity),
            maxVelocity     = p.maxSpeed(),
            maxDensityError = p.maxDensityError(self._config.referenceDensity),
        )

    @property
    def particles(self) -> ParticleSystem:
        '''The particle system being advanced.'''
        return self._particles

    @property
    def config(self) -> SimulationConfig:
        '''The simulation configuration.'''
        return self._config

    @property
    def time(self) -> float:
        '''Simulated time [s].'''
        return self._time

    @property
    def speedOfSound(self) -> float:
        '''Artificial speed of sound c0 [m/s].'''
        return self._speedOfSound

    @property
    def eosStiffness(self) -> float:
        '''Tait stiffness B = rho0 c0^2 / gamma [Pa].'''
        return self._config.referenceDensity * self._speedOfSound**2 / taitExponent
