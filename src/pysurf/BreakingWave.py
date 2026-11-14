
# -- pySurf: Breaking Wave -- #

'''

A solitary wave running up a plane sand beach, shoaling, overturning and breaking, simulated in
2D with weakly compressible SPH.

`BreakingWave` carries the problem and delegates each part of it to a module:

    pysurf.bathymetry       the beach: a flat bottom, then a constant slope through the shoreline
    pysurf.waveMakers       the solitary wave laid in already moving, crest over the flat bottom
    pysurf.sphBoundaries    the sand as layers of fixed boundary particles, and a seaward wall
    pysurf.sphSolver        the WCSPH time stepping
    pysurf.waveBreaking     crest tracking, the breaking point and the runup
    pysurf.figures          the frames, and the transient as a GIF

    wave = BreakingWave(depth = 1.0, waveHeight = 0.3, beachSlope = 1 / 35)
    wave.run()
    wave.writeAnimation('breakingWave.gif')

Grilli et al.'s (1997) slope parameter S0 = 1.521 tan(beta) / sqrt(H / d) predicts the breaker type
before anything runs: spilling below 0.025, plunging to 0.30, surging to 0.37. The default case,
H/d = 0.3 on a 1:35 slope, has S0 = 0.079 and plunges.

The geometry scales with the depth: the wave starts with its crest far enough from the seaward wall
that the wall sees 1 percent of the wave height, the slope begins as far again past the crest, and
the beach runs on above still water far enough to hold the runup. The model has no physical
viscosity or surface tension, and at a fixed number of particles per depth its artificial viscosity
scales with the depth too, so the run is Froude-similar: every result divided by the depth, and
every time by sqrt(d / g), is the same at any depth.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

The 'deltaSph' scheme, with its limited artificial viscosity, measured on the GPU backend against
the phase 1 criteria of planning/gpuSphRoadmap.md, set before the runs except the still tank's; the
runs are scripted there, not yet in the test suite. The slope cases start from bed-fitted columns
(`bedFitted`), the default:

Validated at 64 and 128 particles per depth: the breaking point. Against Grilli et al. (1997) for
H/d = 0.3 on a 1:35 slope, Hb/d = 0.476 and hb/d = 0.345 (tabulated in Table 8 of Senthilkumar 2016,
arXiv:1601.06822), the model breaks at Hb/d = 0.486, 0.496 and 0.484 (+2.1, +4.3, +1.6 percent) and
hb/d = 0.278, 0.316 and 0.335 (-19.5, -8.4, -2.8 percent) at 32, 64 and 128 particles per depth,
against criteria of 10 and 15 percent: at 32 the wave breaks too far up the slope. The breaking
depth converges at first order, and Richardson extrapolation of the three estimates hb/d = 0.355
(+3.0 percent). From the square lattice's start (`bedFitted = False`) the wave breaks earlier and
deeper, at hb/d = 0.318, 0.352 and 0.355, inside the criteria at all three, and extrapolates to the
same depth. Before the slope the exact solitary wave of H/d = 0.3 loses 0.42 percent of its height
over 29 depths of flat bed at 32 particles per depth and 0.76 at 64, inside 2 percent.

Validated at 128 particles per depth: the overturn at touchdown. On the Surf Ranch soliton (H/d =
0.6, 0.0693 reef face) against Feddersen et al.'s (2024) DNS, run from the start, the jet lands 0.04
d short of the DNS's impact point, and the height (-0.9 percent), overturn area (-3), jet area (-11.5,
against a criterion of 20) and overturn angle (-3.6 degrees) pass. At 64 four pass; the overturn
angle, 5.2 degrees low, fails by 0.2. At 32 only the jet area passes: the jet lands 0.88 d long and
11 percent low over a tube 40 percent too large. The overturn area does not converge monotonically
(0.495, 0.308 and 0.341 H^2, against 0.352), so the pass holds at 128 and is not a converged result.
The wave breaks 2.5 and 2.1 percent below the boundary element model's height at 64 and 128 and
lands 0.12 and 0.04 s after it. From the square lattice's start four of the five pass at 32 and 64
and all five at 128.

Not validated: the surface before impact and what follows the touchdown. 0.1 sqrt(d / g) before the
boundary element impact, the boundary element surface lies up to 11, 15 and 14 spacings from the
particles' at 32, 64 and 128 particles per depth, against a criterion of 2. The energy lost in
active breaking on a 3 degree beach is 0.20 and 0.17 of the energy at breaking at 32 and 64
particles per depth, against Mostert and Deike's 0.31: 35 and 47 percent short of a 30 percent
criterion, and further off at the finer spacing. The scheme models no turbulence. Continued from
the boundary element model's jet (`SplashUp`'s hand-off, plain term) the scheme loses 0.23 and
0.18, where the hand-off with the Shepard scheme loses 0.30 at 48: the shortfall is the dissipation
after the jet lands, not the jet.

The next two paragraphs were measured from the square lattice's start.

The artificial viscosity. Monaghan's term acts on the crest's smooth strain as a viscosity of about
alpha c0 h / 8, which halves with the spacing, and the plunging lip is made of the top 0.03 d of the
water ahead of the wave. With the plain term (solverOptions = {'viscosityLimiter': False}), 0.1 s
before impact the crest is 4.7, 2.2 and 1.3 percent low at 64, 128 and 256 particles per depth and
the lip carries 52, 67 and 72 percent of the boundary element model's water; at 128 the overturn's
jet area fails, 25 percent small. At alpha 0 the lip carries 68 and 84 percent at 64 and 128, but
the jet's tip breaks up from the particles' noise. The limited term damps the noise and leaves the
strain alone: the lip carries 70 and 87 percent at 64 and 128 with the tip whole. Started from the
boundary element wave 0.18 s before impact, the plain term carries the lip within 0.6 spacings of
the boundary element model's.

Splitting the fluid near the crest (`refinement`, h kept), measured with the plain term, passes four
of the five overturn criteria at 64 particles per depth, the jet area 14 percent small, and two at
128, where the jet's tip stays whole and lands at the boundary element model's time, 0.11 d behind
its impact point; the overturn area stays 18 and 21 percent small. Started from the boundary element
wave 1.3 s before impact, a zone reaching the bed ('above' at most minus the depth there) matters
more than its levels: split once there, 64 particles per depth carries uniform 128's lip in a third
of the time, and split twice follows uniform 256 until 0.1 s before impact, where the lip thins to
near the kernel's reach; the default floor, 0.15 d over still water, holds the lip to 61 percent at
either level.

A tank of still water, after 20 sqrt(d / g), keeps an rms speed of 7.1×10^-4 sqrt(g d) and a fastest
of 3.2×10^-3 at 60 particles per depth, inside a criterion of 10^-3 on the rms and 10^-2 on the
fastest; at 30 the rms reaches 1.4×10^-3. That criterion replaced one of 10^-3 on the fastest
particle after the runs, which every setting failed.

Not validated: a jittered start (`startJitter`, 0 by default). Moved off bed-fitted columns by up to
0.15 spacings and read over six spacings of arc, the slope cases pass where the columns' do and at
two more: Grilli's breaking point at 32 particles per depth (Hb/d +6.2 to +9.3 percent, hb/d -13.7
to -14.7 over three seeds; -4.9 and -1.2 percent at 64 and 128) and the overturn angle at 64 (-3.7
degrees, all five criteria passing). Its particles never come to rest: the still tank keeps an rms
speed of 1.3×10^-2 and 9.8×10^-3 sqrt(g d) at 30 and 60 particles per depth, ten times the
criterion, and fails at both.

Not validated: a packed start (`startPacking`, False by default). Packing the 0.15-spacing jitter
returns the particles most of the way to their columns: the structure factor at the columns'
wavenumber rises from 0.74 of the particle count to 0.85 to 0.97 (1 on the columns). From it the
slope cases read as from the columns: Grilli's breaking point fails at 32 particles per depth
(hb/d -19.9 to -20.3 percent over three seeds) and passes at 64 and 128 (-7.0 and -2.4), and the
overturn passes four criteria at 64 and 128, the angle failing at 64 (-5.6 degrees) and the jet
area at 128 (-20.2 percent). The particles come to rest: the still tank passes at 60 particles per
depth (rms 6.6×10^-4 and 5.6×10^-4 sqrt(g d) on two seeds) and fails at 30. The solitary wave
loses 1.34 and 0.81 percent of its height at 32 and 64, against 0.50 and 0.76 from the columns.
A jitter of 0.45 spacings, packed, keeps less of the order and also comes to rest. On the ridge's
centre line it breaks no earlier than the packed 0.15 jitter at 64 particles per depth, where the
unpacked jitter breaks 0.05 to 0.09 sqrt(d / g) earlier, and recovers about half the jitter's gain
at 32 (one seed): the jittered start's earlier breaking goes mostly with its particles' motion.

The 'shepard' scheme:

Not validated: the breaking point. Against the same Grilli case, at 16 particles per depth the
model breaks at hb/d = 0.097 with Hb/d = 0.325: 72 percent too shallow and 32 percent too low
(tests/testValidationBreaking.py, an expected failure). Numerical damping drains the wave faster
than shoaling builds it over the 30 depths of slope, and the fixed particle size leaves 5.5
particles across the water column where Grilli's wave breaks.

Validated: the wave before the slope. Its celerity is within 0.5 percent of sqrt(g (d + H)) at 12
particles per depth and 0.3 percent at 16 (tests/testValidationSolitary.py), and the solver
reproduces the first-mode sloshing period within 2.2 percent (tests/testValidationSloshing.py). The
crest loses height as it travels, 6.5 percent per 11 depths at 16 particles per depth against 20
percent at 8, and that is the error that grows over a long slope.

References:
-----------
Grilli, S.T., Svendsen, I.A. & Subramanya, R. (1997) -- Breaking criterion and characteristics
    for solitary waves on slopes, J. Waterway, Port, Coastal, Ocean Eng. 123(3), 102-112.
Monaghan, J.J. & Kos, A. (1999) -- Solitary waves on a Cretan beach, J. Waterway, Port, Coastal,
    Ocean Eng. 125(3), 145-154.
Dalrymple, R.A. & Rogers, B.D. (2006) -- Numerical modeling of water waves with the SPH method,
    Coastal Engineering 53, 141-147.
Feddersen, F., Hanson, K., Mostert, W. & Fincham, A. (2024) -- Modelling wind-induced changes to
    overturning wave shape, J. Fluid Mech. 1000, A48.
Mostert, W. & Deike, L. (2020) -- Inertial energy dissipation in shallow-water breaking waves,
    J. Fluid Mech. 890, A12.

Sean Bowman [10/11/2026]

'''

# Standard library imports
from __future__ import annotations

import math
import time as timeModule
from dataclasses import dataclass

# Third-party imports
import numpy as np
from scipy.integrate import trapezoid

# Local imports
from pysurf import constants, figures, waveTheory
from pysurf.bathymetry import Bathymetry
from pysurf.bemSolver import FlowField, JetGeometry, jetGeometry
from pysurf.errors import InvalidInputError, NumericalInstabilityError
from pysurf.sphBoundaries import BedBoundary
from pysurf.sphHandoff import boundaryPressure, clearSurface, seedStillWater, seedWater
from pysurf.sphPacking import PackingResult, packPoints
from pysurf.sphParticles import ParticleSystem
from pysurf.sphRefinement import refinementLevels
from pysurf.sphSolver import SimulationConfig, WcsphSolver, taitDensity, taitEquationOfState
from pysurf.validation import applyRules, choiceRule, integerRule, numericRule
from pysurf.waveBreaking import BreakingTracker
from pysurf.waveMakers import exactSolitaryBoundaryPressure, exactSolitaryWaveFluid, solitaryWaveFluid

# h / s, the artificial viscosity, the start's random offset in spacings and whether the start is
# packed, each scheme's unless given
defaultSmoothingRatio = {'shepard': 1.3, 'deltaSph': 3.0}
defaultViscosity = {'shepard': 0.02, 'deltaSph': 0.01}
defaultJitter = {'shepard': 0.0, 'deltaSph': 0.0}
defaultPacking = {'shepard': False, 'deltaSph': False}

# The arc in spacings the face slope is read over, for a start on its lattice and one jittered off it
latticeFaceArc, jitteredFaceArc = 2.0, 6.0

# Inputs a breaking wave must satisfy before anything is built
inputRules = (
    numericRule('depth',             'Still water depth',   units = 'm', minimum = 0.0),
    numericRule('waveHeight',        'Wave height',         units = 'm', minimum = 0.0),
    numericRule('beachSlope',        'Beach slope',         minimum = 0.0, maximum = 1.0),
    integerRule('particlesPerDepth', 'Particles per depth', minimum = 4, exclusiveMinimum = False),
    numericRule('frameInterval',     'Frame interval',      units = 's', minimum = 0.0),
    numericRule('alphaViscosity',    'Artificial viscosity', minimum = 0.0, exclusiveMinimum = False),
    numericRule('runupAllowance',    'Runup allowance',     minimum = 0.0),
    choiceRule('scheme',             'SPH scheme',          choices = tuple(defaultSmoothingRatio)),
    numericRule('smoothingLengthRatio', 'Smoothing length over spacing', minimum = 1.0, maximum = 4.0),
    choiceRule('backend',            'Solver backend',      choices = ('numpy', 'compiled', 'gpu')),
    choiceRule('startingWave',       'Starting wave',       choices = ('exact', 'serre', 'still')),
    numericRule('startJitter',       'Start jitter',        minimum = 0.0, exclusiveMinimum = False, maximum = 0.5,
                exclusiveMaximum = True, note = 'two neighbors moved half a spacing toward each other would meet'),
    numericRule('faceArc',           'Face slope arc',      units = 'spacings', minimum = 1.0, exclusiveMinimum = False,
                maximum = 12.0, exclusiveMaximum = False),
)

# The crest-following refinement zone: splits, reach behind and ahead of the crest and floor over
# still water in depths, and the crest position it starts at (None for the toe)
refinementDefaults = {'levels': 1, 'behind': 1.0, 'ahead': 1.5, 'above': 0.15, 'fromX': None}
refinementRules = (
    integerRule('levels', 'Refinement levels', minimum = 1, maximum = 3, exclusiveMinimum = False),
    numericRule('behind', 'Refinement zone behind the crest', minimum = 0.0, exclusiveMinimum = False),
    numericRule('ahead',  'Refinement zone ahead of the crest', minimum = 0.0, exclusiveMinimum = False),
    numericRule('above',  'Refinement zone floor over still water', minimum = -10.0),
)

@dataclass
class WaveFrame:

    '''

    One recorded instant of the fluid.

    Parameters:
    -----------
    time : float
        Simulated time [s]
    positions : np.ndarray
        Fluid particle positions, shape (N, 2) [m]
    speeds : np.ndarray
        Fluid particle speeds [m/s]
    kineticEnergy, potentialEnergy : float
        Fluid energies per unit crest width [J/m]

    '''

    time: float
    positions: np.ndarray
    speeds: np.ndarray
    kineticEnergy: float
    potentialEnergy: float

class BreakingWave:

    '''

    A solitary wave breaking on a plane beach, or on a given bed.

    Parameters:
    -----------
    depth : float
        Still water depth over the flat bottom d [m]
    waveHeight : float
        Incident solitary wave height above still water H [m]
    beachSlope : float
        Beach slope tan(beta) [-]; for a given bed, the slope that sets Grilli's slope parameter
    particlesPerDepth : int
        Particles across the still water depth; sets the resolution
    frameInterval : float
        Simulated time between recorded frames [s]
    alphaViscosity : float | None
        Monaghan artificial viscosity coefficient; None takes the scheme's, 0.02 for 'shepard' and
        0.01 for 'deltaSph'
    runupAllowance : float
        How far above still water the beach extends, in depths
    endTime : float | None
        Simulated time to run [s]; None runs to 1.15 times the estimated arrival at the shoreline
    scheme : str
        'shepard': the cubic spline, a Shepard density reset every 30 steps, XSPH, boundary
        pressure from continuity, and the first-order solitary wave as the start. 'deltaSph': the
        Wendland C2 kernel, delta-SPH density diffusion in its hydrostatic form, the kernel
        gradient correction in the continuity equation, boundary pressure extrapolated from the
        water, no XSPH, the fluid density held at or above 0.9 rho0, the artificial viscosity
        limited to the particles' noise (`SimulationConfig.viscosityLimiter`), and the exact
        solitary wave with its pressure as the start
    smoothingLengthRatio : float | None
        h / s; None takes the scheme's, 1.3 for 'shepard' and 3.0 for 'deltaSph'
    backend : str
        Solver backend, 'numpy', 'compiled' or 'gpu' (`SimulationConfig.backend`); the compiled
        backend runs only the 'shepard' scheme
    solverOptions : dict | None
        Further `SimulationConfig` fields, such as `particleShifting`, applied over the scheme's;
        not the fields this class sets from its own inputs (spacing, kernel width, viscosity, sound
        speed, times, domain, backend)
    bathymetry : Bathymetry | None
        A bed to use instead of the plane beach, with its still water level at `depth` and flat at
        that depth from its start to at least a half length past the starting crest. A bed that
        ends below still water ends at a wall.
    startX : float | None
        Starting crest position [m]; by default the half length from the seaward wall, where the
        wall sees 1 percent of the wave height
    startingWave : str
        For the 'deltaSph' scheme, the solitary wave the water starts as: 'exact', the exact
        potential flow wave (`waveTheory.exactSolitaryFlow`); or 'serre', the starting state of
        `OverturningWave`, the Serre-Green-Naghdi wave's surface with its potential, whose flow
        inside the water the boundary element solution gives (`bemSolver.FlowField`). The second
        starts both models from one state, as a comparison between them needs; at H/d = 0.6 it
        carries 17 percent more water than the exact wave of the same height. 'still' starts the
        same tank with still water laid the same way (`bedFitted`), at rest and hydrostatic: the
        reference run whose drift `energyHistory` takes out of a wave's energies.
    refinement : dict | None
        Split the fluid in a zone that follows the crest (`WcsphSolver.refine`, keeping h): 'levels',
        how many times a particle there splits (1 to 3); 'behind' and 'ahead', the zone's reach
        from the tracked crest along x, and 'above', its floor over still water, in depths; 'fromX',
        the crest position the zone starts at [m], the toe by default. Applied at every breaking
        check; daughters stay split when the zone moves on. None for a uniform spacing.
    bedFitted : bool | None
        Lay the starting water in columns half a spacing clear of the bed
        (`sphParticles.bedFittedPoints`) rather than on the square lattice, whose lowest particles
        sit half a spacing to one and a half above a slope and fall into those steps from rest.
        Only the 'deltaSph' scheme lays bed-fitted columns. None takes the scheme's: bed-fitted
        columns for 'deltaSph', the lattice for 'shepard'
    startJitter : float | None
        Move each starting particle off its place by a seeded uniform random offset of up to this
        many spacings along each axis, reflected back into the water, with the start's flow and
        pressure at its final place (`sphParticles.jitterPoints`); 0 for none. A jitter of 0.15
        spacings keeps most of the columns' order and leaves the particles moving: the still tank
        fails, and the slope cases break earlier, on the ridge's centre line mostly from that
        motion. Only the 'deltaSph' scheme jitters its start. None takes the scheme's, 0 for both
    startSeed : int
        Seed of the start's random offsets
    startPacking : bool | None
        Relax the laid start before its flow and pressure are set (`sphPacking.packPoints`): each
        particle moves down the gradient of its kernel sum, between the boundary's fixed particles
        and below the starting surface, until that gradient, which sets the SPH pressure sum's
        error in water at rest, falls to 1e-4 / h in rms. A start jittered by 0.15 spacings packs
        most of the way back to its columns; a start on its lattice barely moves. Only the
        'deltaSph' scheme packs its start. None takes the scheme's, False for both
    faceArc : float | None
        For the 'deltaSph' scheme, the arc the face slope is read over, in spacings
        (`waveBreaking.contourFace`); None for 2 when the start lies on its lattice and 6, the
        kernel's support, when it is jittered, whose surface is rough at the particles' scale

    '''

    # -- Constructor -- #
    def __init__(self, depth: float = 1.0, waveHeight: float = 0.3, beachSlope: float = 1.0 / 35.0,
                 particlesPerDepth: int = 16, frameInterval: float = 0.1, alphaViscosity: float | None = None,
                 runupAllowance: float = 0.8, endTime: float | None = None, scheme: str = 'shepard',
                 smoothingLengthRatio: float | None = None, backend: str = 'numpy',
                 solverOptions: dict | None = None, bathymetry: Bathymetry | None = None,
                 startX: float | None = None, startingWave: str = 'exact', refinement: dict | None = None,
                 bedFitted: bool | None = None, startJitter: float | None = None, startSeed: int = 1,
                 faceArc: float | None = None, startPacking: bool | None = None) -> None:

        self.depth = depth                          # [m]
        self.waveHeight = waveHeight                # [m]
        self.beachSlope = beachSlope                # [-]
        self.particlesPerDepth = particlesPerDepth
        self.frameInterval = frameInterval          # [s]
        self.alphaViscosity = alphaViscosity if alphaViscosity is not None else defaultViscosity.get(scheme, 0.02)
        self.runupAllowance = runupAllowance
        self.endTime = endTime                      # [s]
        self.scheme = scheme
        self.smoothingLengthRatio = (smoothingLengthRatio if smoothingLengthRatio is not None
                                     else defaultSmoothingRatio.get(scheme, 1.3))
        self.backend = backend
        self.solverOptions = dict(solverOptions) if solverOptions is not None else {}
        self.customBathymetry = bathymetry
        self.startX = startX                        # [m]
        self.startingWave = startingWave
        self.refinement = ({**refinementDefaults, **refinement} if refinement is not None else None)
        self.bedFitted = bool(bedFitted) if bedFitted is not None else scheme == 'deltaSph'
        self.startJitter = float(startJitter if startJitter is not None else defaultJitter.get(scheme, 0.0))
        self.startSeed = int(startSeed)
        self.startPacking = bool(startPacking) if startPacking is not None else defaultPacking.get(scheme, False)
        self.faceArc = float(faceArc if faceArc is not None
                             else jitteredFaceArc if self.startJitter > 0.0 else latticeFaceArc)

        # Speed of sound as a multiple of the solitary wave celerity, and simulated time between
        # breaking checks in units of sqrt(d / g)
        self.soundSpeedFactor = 10.0
        self.trackInterval = 0.01

        self.bathymetry: Bathymetry | None = None
        self.boundary: BedBoundary | None = None
        self.particles: ParticleSystem | None = None
        self.solver: WcsphSolver | None = None
        self.tracker: BreakingTracker | None = None
        self.frames: list[WaveFrame] = []
        self.packing: PackingResult | None = None
        self.wallClockSeconds = 0.0                 # [s]
        self._fluidMass = 0.0                       # [kg/m]

        self._validateInputs()

    # -- Private Helper Methods -- #
    def _validateInputs(self) -> None:

        '''

        Check the inputs against the rule table, and that the wave is a wave this model can start.

        Raises:
        -------
        InvalidInputError : If any input is out of range.

        '''

        applyRules(self, inputRules)
        applyRules({'heightRatio': self.waveHeight / self.depth},
                   [numericRule('heightRatio', 'Wave height over depth', maximum = waveTheory.mcCowanLimit,
                                note = 'a solitary wave higher than this breaks before it reaches the slope')])
        if self.startX is not None and self.startX < self.halfLength:
            raise InvalidInputError('The starting crest must be at least a half length from the seaward wall.',
                                    parameterName = 'startX', value = self.startX)
        if self.startingWave != 'exact' and self.scheme != 'deltaSph':
            raise InvalidInputError("Only the 'deltaSph' scheme takes a starting wave other than its own.",
                                    parameterName = 'startingWave', value = self.startingWave)
        if self.bedFitted and self.scheme != 'deltaSph':
            raise InvalidInputError("Only the 'deltaSph' scheme lays its start in bed-fitted columns.",
                                    parameterName = 'bedFitted', value = self.bedFitted)
        if self.startJitter > 0.0 and self.scheme != 'deltaSph':
            raise InvalidInputError("Only the 'deltaSph' scheme jitters its start.", parameterName = 'startJitter',
                                    value = self.startJitter)
        if self.startPacking and self.scheme != 'deltaSph':
            raise InvalidInputError("Only the 'deltaSph' scheme packs its start.", parameterName = 'startPacking',
                                    value = self.startPacking)
        if self.refinement is not None:
            unknown = set(self.refinement) - set(refinementDefaults)
            if unknown:
                raise InvalidInputError(f'Unknown refinement settings: {sorted(unknown)}.', parameterName = 'refinement',
                                        value = self.refinement)
            applyRules(self.refinement, refinementRules)
            if self.backend == 'compiled':
                raise InvalidInputError('The compiled backend cannot split particles.', parameterName = 'refinement',
                                        value = self.refinement)
        bed = self.customBathymetry
        if bed is not None and abs(bed.stillWaterLevel - self.depth) > 1e-9 * self.depth:
            raise InvalidInputError('The bed must hold still water at the given depth.',
                                    parameterName = 'bathymetry', value = bed.stillWaterLevel)

    # -- Geometry -- #
    @property
    def spacing(self) -> float:
        '''Particle spacing [m].'''
        return self.depth / self.particlesPerDepth

    @property
    def boundaryLayers(self) -> int:
        '''Boundary particle layers: enough to fill the kernel's support, 2h, and never fewer than 3.'''
        return max(3, math.ceil(2.0 * self.smoothingLengthRatio - 1e-9))

    @property
    def halfLength(self) -> float:
        '''Distance from the crest at which the starting wave is 1 percent of its height [m].'''
        if self.startingWave == 'serre':
            return waveTheory.serreSolitaryWaveHalfLength(self.waveHeight, self.depth, 0.01)
        return waveTheory.solitaryWaveHalfLength(self.waveHeight, self.depth, 0.01)

    @property
    def crestX(self) -> float:
        '''Initial crest position: `startX`, or by default the half length from the seaward wall [m].'''
        return self.startX if self.startX is not None else self.halfLength

    @property
    def toeX(self) -> float:

        '''

        Toe of the slope [m]: for the plane beach a half length past the starting crest; for a given
        bed the last vertex at the offshore depth before the bed first rises.

        '''

        bed = self.customBathymetry
        if bed is None:
            return self.crestX + self.halfLength
        rising = np.flatnonzero(bed.depth(bed.x) < self.depth * (1.0 - 1e-6))
        return float(bed.x[max(int(rising[0]) - 1, 0)]) if rising.size else float(bed.x[-1])

    @property
    def shorelineX(self) -> float:
        '''Where still water meets the beach, or the end of a given bed that stays under water [m].'''
        bed = self.customBathymetry
        if bed is None:
            return self.toeX + self.depth / self.beachSlope
        try:
            return bed.shorelineX()
        except InvalidInputError:
            return bed.endX

    @property
    def slopeParameter(self) -> float:
        '''Grilli's solitary slope parameter S0 [-].'''
        return waveTheory.solitarySlopeParameter(self.beachSlope, self.waveHeight, self.depth)

    @property
    def predictedBreakerType(self) -> str:
        '''Breaker type Grilli's criterion predicts.'''
        return waveTheory.solitaryBreakerType(self.slopeParameter)

    def estimatedArrivalTime(self) -> float:

        '''

        Time for the crest to reach the shoreline, integrating dt = dx / sqrt(g (h(x) + H)) with
        the wave height held at its initial value [s]. Shoaling raises H and speeds the wave, so
        this overestimates the arrival slightly.

        '''

        x = np.linspace(self.crestX, self.shorelineX, 2001)
        bed = self.customBathymetry
        if bed is None:
            h = np.maximum(self.depth - np.maximum(x - self.toeX, 0.0) * self.beachSlope, 0.0)
        else:
            h = np.maximum(bed.depth(x), 0.0)
        celerity = np.sqrt(constants.gravity * (h + self.waveHeight))

        return float(trapezoid(1.0 / celerity, x))

    # -- Public Methods -- #
    def build(self) -> None:

        '''

        Lay out the beach, the boundary, the moving solitary wave, the solver and the tracker.

        '''

        d, s, rho = self.depth, self.spacing, constants.freshwaterDensity
        layers = self.boundaryLayers
        exact = self.scheme == 'deltaSph'
        wallTop = d + 3.0 * self.waveHeight

        if self.customBathymetry is None:
            endX = self.toeX + (1.0 + self.runupAllowance) * d / self.beachSlope
            self.bathymetry = Bathymetry.planeBeach(depth = d, toeX = self.toeX, slope = self.beachSlope, endX = endX)
            endWallTop = None
        else:
            self.bathymetry = self.customBathymetry
            endX = self.bathymetry.endX
            endWallTop = wallTop if self.bathymetry.elevation(endX) < d else None
        self.boundary = BedBoundary(self.bathymetry, s, nLayers = layers, wallTop = wallTop, endWallTop = endWallTop)

        field, tankEnd, level = None, self.bathymetry.endX, self.bathymetry.stillWaterLevel
        laying = dict(bedFitted = self.bedFitted, jitter = self.startJitter, seed = self.startSeed)
        pack = self._packer()
        if exact and self.startingWave == 'serre':
            field, tankEnd = self._potentialFlowStart()
            fluidPositions = seedWater(field, self.bathymetry, s, **laying)
            if tankEnd < self.bathymetry.endX - 1e-9:
                # Past the boundary element tank's end wall the water is still
                fluidPositions = np.vstack([fluidPositions, seedStillWater(self.bathymetry, s, tankEnd, **laying)])
            inTank = fluidPositions[:, 0] <= tankEnd
            if pack is not None:
                # The two waters pack as one; each point then takes the flow of the water it lies in
                surface = field.surface
                fluidPositions = pack(fluidPositions, lambda x: np.interp(x, surface[:, 0], surface[:, 1], right = level))
                inTank = fluidPositions[:, 0] <= tankEnd
                fluidPositions[inTank] = clearSurface(surface, fluidPositions[inTank], s)
            fluidVelocities = np.zeros_like(fluidPositions)
            fluidPressures = constants.gravity * (level - fluidPositions[:, 1])
            fluidVelocities[inTank] = field.velocity(fluidPositions[inTank])
            fluidPressures[inTank] = np.maximum(field.pressure(fluidPositions[inTank], fluidVelocities[inTank]), 0.0) / rho
        elif exact and self.startingWave == 'still':
            # The points `exactSolitaryWaveFluid` lays, below still water
            fluidPositions = seedStillWater(self.bathymetry, s, self.bathymetry.startX - s, **laying)
            if pack is not None:
                fluidPositions = pack(fluidPositions, lambda x: np.full(np.shape(x), level))
            fluidVelocities = np.zeros_like(fluidPositions)
            fluidPressures = constants.gravity * (level - fluidPositions[:, 1])
        elif exact:
            fluidPositions, fluidVelocities, fluidPressures = exactSolitaryWaveFluid(self.bathymetry, self.waveHeight,
                                                                                     self.crestX, s, pack = pack, **laying)
        else:
            fluidPositions, fluidVelocities = solitaryWaveFluid(self.bathymetry, self.waveHeight, self.crestX, s)
        self.particles = ParticleSystem.assemble(fluidPositions, self.boundary.positions(), s, rho, fluidVelocities)
        self._fluidMass = float(self.particles.masses[0])          # [kg/m], of a particle that has not split

        # The floor holds spray and the thin sheet of the runup, whose neighbors fly apart, off a
        # density near zero: the hydrostatic diffusion's expected difference grows as the pair's mean
        # density to the power 2 - gamma, so a pair near zero density drives its densities apart.
        # The limiter keeps the artificial viscosity off the crest's smooth strain, which at alpha
        # 0.01 would otherwise thin the plunging lip (see Validation status)
        schemeOptions = (dict(kernelType = 'wendlandC2', densityFilterInterval = 0, densityDiffusion = 'hydrostatic',
                              xsphEpsilon = 0.0, wallPressure = 'extrapolated', continuityCorrection = True,
                              densityFloor = 0.9, viscosityLimiter = True)
                         if exact else {})
        schemeOptions.update(self.solverOptions)
        soundSpeed = self.soundSpeedFactor * waveTheory.solitaryWaveSpeed(self.waveHeight, d)
        config = SimulationConfig(
            domainMin       = np.array([-(layers + 1.0) * s, -(layers + 1.0) * s]),
            domainMax       = np.array([endX + (layers + 1.0) * s, wallTop]),
            particleSpacing = s,
            smoothingLengthRatio = self.smoothingLengthRatio,
            referenceDensity = rho,
            gravity         = np.array([0.0, -constants.gravity]),
            endTime         = self.endTime if self.endTime is not None else 1.15 * self.estimatedArrivalTime(),
            maxTimeStep     = 1.0,
            outputInterval  = self.frameInterval,
            speedOfSound    = soundSpeed,
            alphaViscosity  = self.alphaViscosity,
            initialSurface  = lambda x: d + waveTheory.solitaryWaveProfile(x, self.waveHeight, d, self.crestX),
            backend         = self.backend,
            **schemeOptions,
        )
        self.solver = WcsphSolver(config, boundary = self.boundary)
        self.solver.initialize(self.particles)

        if exact:
            # The exact wave's pressure, carried by each particle's density; a boundary particle starts
            # with the pressure of the water just inside it
            p = self.particles
            if field is not None:
                wallPressures = boundaryPressure(field, self.bathymetry, p.positions[~p.isFluid], s,
                                                 tankEnd if tankEnd < self.bathymetry.endX - 1e-9 else None) / rho
            elif self.startingWave == 'still':
                wallPressures = self._stillWallPressure(p.positions[~p.isFluid])
            else:
                wallPressures = exactSolitaryBoundaryPressure(self.bathymetry, p.positions[~p.isFluid],
                                                              self.waveHeight, self.crestX, s, constants.gravity)
            p.densities[p.isFluid] = taitDensity(rho * fluidPressures, rho, soundSpeed)
            p.densities[~p.isFluid] = taitDensity(rho * wallPressures, rho, soundSpeed)
            p.pressures = taitEquationOfState(p.densities, rho, soundSpeed)

        # A lean offshore of the toe would be start-up noise, never breaking
        self.tracker = BreakingTracker(self.bathymetry, s, faceReach = d, searchFromX = self.toeX,
                                       faceMeasure = 'contour' if exact else 'cells', faceArc = self.faceArc)
        self.frames = []
        self._recordFrame()

    def _packer(self):

        '''

        The start's packing (`startPacking`) as a function of the laid points and the surface's
        elevation eta(x), returning the packed points and keeping the result in `packing`; None
        when the start is not packed.

        '''

        if not self.startPacking:
            return None
        fixed = self.boundary.positions()
        kernelType = self.solverOptions.get('kernelType', 'wendlandC2')

        def pack(points: np.ndarray, surface) -> np.ndarray:
            self.packing = packPoints(points, fixed, self.spacing, surface, self.bathymetry.elevation,
                                      smoothingLengthRatio = self.smoothingLengthRatio, kernelType = kernelType,
                                      backend = 'gpu' if self.backend == 'gpu' else 'numpy')
            return self.packing.positions

        return pack

    def _stillWallPressure(self, positions: np.ndarray) -> np.ndarray:

        '''

        The pressure over density each boundary particle starts with under still water
        [m^2/s^2], by the rule `exactSolitaryBoundaryPressure` follows: hydrostatic where the point
        half a spacing inside it lies below still water level, zero elsewhere.

        '''

        bed, s, level = self.bathymetry, self.spacing, self.bathymetry.stillWaterLevel
        reference = np.clip(positions[:, 0], bed.startX + 0.5 * s, bed.endX - 0.5 * s)
        insideHeight = np.maximum(positions[:, 1], bed.elevation(reference) + 0.5 * s)

        return np.where(insideHeight < level, constants.gravity * (level - positions[:, 1]), 0.0)

    def _potentialFlowStart(self) -> tuple[FlowField, float]:

        '''

        The flow of `OverturningWave`'s Serre-Green-Naghdi start on this bed, and where its tank
        ends [m]. On the plane beach that tank ends at a wall in water a tenth of the depth deep;
        a given bed is the tank.

        '''

        # A local import: the overturning wave is needed only for this start
        from pysurf.OverturningWave import OverturningWave

        start = OverturningWave(depth = self.depth, waveHeight = self.waveHeight, beachSlope = self.beachSlope,
                                bathymetry = self.customBathymetry, startX = self.crestX, startingWave = 'serre')
        start.build()
        solver = start.solver
        field = FlowField(solver.surface, solver.potential, solver.bed, solver.config.surfaceSpacing,
                          start.bathymetry.stillWaterLevel, constants.gravity, constants.freshwaterDensity)

        return field, float(start.bathymetry.endX)

    def _recordFrame(self) -> None:

        '''Store the fluid at the current instant.'''

        p, fluid = self.particles, self.particles.isFluid
        state = self.solver.currentState
        self.frames.append(WaveFrame(time = self.solver.time, positions = p.positions[fluid].copy(),
                                     speeds = np.linalg.norm(p.velocities[fluid], axis = 1),
                                     kineticEnergy = state.kineticEnergy, potentialEnergy = state.potentialEnergy))

    def run(self, endTime: float | None = None, progress: bool = True, stopAt: str = 'end') -> None:

        '''

        Step the simulation to `endTime`, recording frames and watching for breaking.

        The tracker reads the particles every `trackInterval` sqrt(d / g) of simulated time. The
        solver advances between those instants in one call, so on the GPU backend the state stays
        on the device in between.

        Parameters:
        -----------
        endTime : float | None
            Simulated time to stop at [s]; the configured end time by default
        progress : bool
            Print a line roughly every tenth of the run
        stopAt : str
            'end', 'breaking' to stop once the tracker's breaking point is final
            (`BreakingTracker.breakingFinal`), or 'touchdown' to stop once it has recorded the
            jet's touchdown (the 'deltaSph' scheme). The 'deltaSph' scheme reads the smoothed
            surface, where the touchdown fixes the breaking point, so there the two stop together

        Raises:
        -------
        NumericalInstabilityError : If any particle position stops being finite.

        '''

        if stopAt not in ('end', 'breaking', 'touchdown'):
            raise ValueError(f"stopAt '{stopAt}' is not 'end', 'breaking' or 'touchdown'.")
        if self.solver is None:
            self.build()

        solver, tracker = self.solver, self.tracker
        stopTime = endTime if endTime is not None else solver.config.endTime
        trackStep = self.trackInterval * math.sqrt(self.depth / constants.gravity)
        nextTrack = solver.time + trackStep
        nextFrame = solver.time + self.frameInterval
        nextReport = solver.time + stopTime / 10.0
        started = timeModule.perf_counter()

        if progress:
            print(f'BreakingWave: {self.particles.nFluid} fluid and {self.particles.nBoundary} boundary particles, '
                  f'S0 = {self.slopeParameter:.3f} ({self.predictedBreakerType}), running to {stopTime:.2f} s')

        # A hundred times the steps an interval takes at the acoustic time step: more means the step
        # has collapsed, which a diverging run does before its positions stop being finite
        config = solver.config
        stepLimit = max(1000, int(100.0 * min(trackStep, self.frameInterval) * solver.speedOfSound
                                  / (config.cflNumber * config.smoothingLength)))

        while solver.time < stopTime and not ((stopAt == 'breaking' and tracker.breakingFinal)
                                              or (stopAt == 'touchdown' and tracker.touchdownSurface is not None)):
            target = min(nextTrack, nextFrame, stopTime)
            state = solver.advance(target, maxSteps = stepLimit)
            if solver.time < target:
                raise NumericalInstabilityError('The time step collapsed.', variableName = 'dt', operation = 'WCSPH step',
                                                context = {'time': solver.time, 'step': state.step, 'dt': state.dt})

            if solver.time >= nextTrack:
                p = self.particles
                # Split particles count their own area in the smoothed surface
                volumes = p.masses[p.isFluid] / solver.config.referenceDensity if p.lengths is not None else None
                tracker.update(solver.time, p.positions[p.isFluid], volumes)
                nextTrack += trackStep
                if self.refinement is not None:
                    self._refineCrest()
            if solver.time >= nextFrame:
                if not np.all(np.isfinite(self.particles.positions)):
                    raise NumericalInstabilityError('Particle positions stopped being finite.',
                                                    variableName = 'positions', operation = 'WCSPH step',
                                                    context = {'time': solver.time, 'step': state.step})
                self._recordFrame()
                nextFrame += self.frameInterval
            if progress and solver.time >= nextReport:
                elapsed = timeModule.perf_counter() - started
                # On the contour face measure the breaking point stays provisional until touchdown
                broken = ('breaking recorded' if tracker.breakingFinal
                          else 'breaking provisional' if tracker.breakingPoint else 'not broken')
                print(f'  t = {solver.time:6.2f} s  step {state.step:7d}  dt {state.dt:.2e} s  '
                      f'wall {elapsed:7.1f} s  crest {tracker.crestHeight[-1] if tracker.crestHeight else 0.0:.3f} m  '
                      f'{broken}')
                nextReport += stopTime / 10.0

        self.wallClockSeconds += timeModule.perf_counter() - started

    def _refineCrest(self) -> int:

        '''

        Split the fluid in the zone that follows the tracked crest, a level at a time, until every
        particle in it has split `refinement['levels']` times. Returns how many particles split.

        '''

        settings, d = self.refinement, self.depth
        tracker, p = self.tracker, self.particles
        if not tracker.crestX:
            return 0
        crestX = tracker.crestX[-1]
        fromX = settings['fromX'] if settings['fromX'] is not None else self.toeX
        if crestX < fromX:
            return 0
        low, high = crestX - settings['behind'] * d, crestX + settings['ahead'] * d
        floor = self.bathymetry.stillWaterLevel + settings['above'] * d
        total = 0
        for _ in range(int(settings['levels'])):
            x, z = p.positions[:, 0], p.positions[:, -1]
            zone = p.isFluid & (x > low) & (x < high) & (z > floor)
            split = self.solver.refine(zone & (refinementLevels(p, self._fluidMass) < settings['levels']))
            total += split
            if split == 0:
                break

        return total

    @property
    def breakingPoint(self):
        '''Where the wave broke, or None.'''
        return self.tracker.breakingPoint if self.tracker is not None else None

    @property
    def overturn(self) -> JetGeometry | None:

        '''

        The jet and the tube under it on the surface just before touchdown (`bemSolver.jetGeometry`),
        or None before touchdown. The surface is resampled at half a spacing of arc length, and the
        jet tip and impact point must lie 0.3 d of arc apart, as for the boundary element model's
        nodes at a crest spacing of 0.02 d.

        '''

        surface = self.tracker.touchdownSurface if self.tracker is not None else None
        if surface is None:
            return None
        step = 0.5 * self.spacing
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(surface, axis = 0).T))])
        samples = np.arange(0.0, arc[-1], step)
        resampled = np.column_stack([np.interp(samples, arc, surface[:, 0]), np.interp(samples, arc, surface[:, 1])])

        return jetGeometry(resampled, gap = max(4, int(round(0.3 * self.depth / step))))

    def summary(self) -> dict:

        '''

        The breaking measures in dimensional and depth-normalized form.

        Returns:
        --------
        dict : slope parameter, breaker types, breaking index, Hb/d, hb/d, breaking position from
            the toe in depths, maximum runup over depth, particle counts and wall-clock time; once
            the jet has touched down ('deltaSph'), the touchdown time, where the jet's tip and its
            impact point lie, the impact height H_I (the highest point of the surface), the tube's
            area over d^2 and the closed pocket's, and the overturn measures of Feddersen et al.
            (2023, 2024) as `OverturningWave` reports them: A_O / H_I^2, A_J / H_I^2, W / L and
            theta_O

        '''

        point = self.breakingPoint
        result = {
            'slopeParameter': self.slopeParameter,
            'predictedBreakerType': self.predictedBreakerType,
            'particlesPerDepth': self.particlesPerDepth,
            'bedFitted': self.bedFitted,
            'startJitter': self.startJitter,
            'startSeed': self.startSeed,
            'startPacking': self.startPacking,
            'faceArc': self.faceArc,
            'fluidParticles': self.particles.nFluid if self.particles is not None else 0,
            'wallClockSeconds': self.wallClockSeconds,
            'broke': point is not None,
        }
        if self.packing is not None:
            result.update({
                'packingIterations': self.packing.iterations,
                'packingResidualBefore': self.packing.gradientBefore[0],
                'packingResidual': self.packing.gradientAfter[0],
                'packingLargestMove': self.packing.largestMove,
            'packingFrozen': self.packing.frozen,
            })
        if point is not None:
            result.update({
                'breakingTime': point.time,
                'breakingIndex': point.breakingIndex,
                'breakingHeightOverDepth': point.crestHeight / self.depth,
                'breakingDepthOverDepth': point.depth / self.depth,
                'breakingDistanceFromToe': (point.crestX - self.toeX) / self.depth,
                'breakingFaceDepthOverDepth': point.faceDepth / self.depth,
            })
        if self.tracker is not None and np.isfinite(self.tracker.maxRunup):
            result['maxRunupOverDepth'] = self.tracker.maxRunup / self.depth
        jet = self.overturn
        result['touchedDown'] = jet is not None
        if jet is not None:
            tracker, d = self.tracker, self.depth
            impactHeight = float(np.max(tracker.touchdownSurface[:, 1])) - self.bathymetry.stillWaterLevel
            pocket = tracker.touchdownPocket
            pocketArea = 0.5 * abs(float(np.dot(pocket[:, 0], np.roll(pocket[:, 1], -1))
                                         - np.dot(pocket[:, 1], np.roll(pocket[:, 0], -1))))
            result.update({
                'touchdownTime': tracker.touchdownTime,
                'jetTipX': float(jet.tip[0]),
                'impactX': float(jet.impactPoint[0]),
                'impactHeightOverDepth': impactHeight / d,
                'cavityAreaOverDepthSquared': jet.cavityArea / d**2,
                'pocketAreaOverDepthSquared': pocketArea / d**2,
                'overturnAreaOverImpactHeightSquared': jet.cavityArea / impactHeight**2,
                'jetAreaOverImpactHeightSquared': jet.jetArea / impactHeight**2,
                'overturnAspectRatio': jet.aspectRatio,
                'overturnAngle': jet.overturnAngle,
            })

        return result

    def startingWavePotentialEnergy(self) -> float:

        '''

        The starting wave's potential energy over still water, rho g int(eta^2 / 2) dx, from the
        surface it starts with [J/m].

        '''

        bed, d, H = self.bathymetry, self.depth, self.waveHeight
        if self.startingWave == 'still':
            return 0.0
        x = np.linspace(bed.startX, bed.endX, int((bed.endX - bed.startX) / (0.005 * d)) + 1)
        if self.scheme != 'deltaSph':
            eta = waveTheory.solitaryWaveProfile(x, H, d, self.crestX)
        elif self.startingWave == 'serre':
            eta = waveTheory.serreSolitaryWaveProfile(x, H, d, self.crestX)
        else:
            eta, _ = waveTheory.exactSolitarySurface(x, H, d, self.crestX)

        return float(constants.freshwaterDensity * constants.gravity * trapezoid(0.5 * eta**2, x))

    def energyHistory(self, reference: BreakingWave | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        '''

        Kinetic energy and the wave's potential energy over still water at each recorded frame
        [J/m]. The particles' potential energy differs from the wave's by a constant while the water
        keeps its mass: the still water's, and the excess volume's at still water level. The
        constant is taken at the start, where the wave's is known (`startingWavePotentialEnergy`).

        Weakly compressible water does not keep that constant. The square lattice leaves the rows
        over a sloping bed further from it than they settle; the water falls into those steps and
        compresses under its own weight, turning potential energy into elastic and kinetic energy
        the budget does not count: on a 3 degree beach, 17 percent of the energy of a solitary wave
        of H/d = 0.3 in the first half second at 32 particles per depth, and 10 percent at 64. A
        run of the same tank from still water (`startingWave = 'still'`), laid the same way,
        settles the same way, so its kinetic energy and the change in its potential energy, taken
        out here, leave the wave's. Bed-fitted columns (`bedFitted`) start the water at rest
        against the bed: on the same beach the still water loses 0.7 and 0.4 percent of that
        wave's energy in the first half second, and the reference takes out the drift that remains.

        Parameters:
        -----------
        reference : BreakingWave | None
            The still water run of the same tank, recorded to within half its frame interval of
            this one's end; None takes nothing out

        Returns:
        --------
        tuple : (time [s], kinetic energy [J/m], potential energy [J/m])

        Raises:
        -------
        ValueError : If the reference does not start from still water, lays its water differently,
            or ends before this run.

        '''

        times = np.array([frame.time for frame in self.frames])
        kinetic = np.array([frame.kineticEnergy for frame in self.frames])
        potential = np.array([frame.potentialEnergy for frame in self.frames])
        potential = potential - potential[0] + self.startingWavePotentialEnergy()
        if reference is not None:
            if reference.startingWave != 'still':
                raise ValueError("The reference run must start from still water (startingWave = 'still').")
            if (reference.bedFitted != self.bedFitted or reference.startJitter != self.startJitter
                    or reference.startPacking != self.startPacking):
                raise ValueError('The reference run must lay its water as this run does (bedFitted, startJitter, '
                                 'startPacking).')
            referenceTimes = np.array([frame.time for frame in reference.frames])
            # Each run overshoots its last frame by up to a step; half a frame is a gap, not a shorter run
            if referenceTimes[-1] < times[-1] - 0.5 * reference.frameInterval:
                raise ValueError(f'The reference run ends at {referenceTimes[-1]:.3f} s, before this one ({times[-1]:.3f} s).')
            referencePotential = np.array([frame.potentialEnergy for frame in reference.frames])
            kinetic = kinetic - np.interp(times, referenceTimes, [frame.kineticEnergy for frame in reference.frames])
            potential = potential - (np.interp(times, referenceTimes, referencePotential) - referencePotential[0])

        return times, kinetic, potential

    def breakingDissipation(self, reference: BreakingWave | None = None) -> dict:

        '''

        The energy lost in active breaking, as Mostert and Deike (2020) measure it and
        `SplashUp.breakingDissipation` reports it: from the moment the face turns vertical, t1, to
        the first moment after it when the kinetic energy has fallen to the potential, t2, the
        mechanical energy falls by dE, and the dissipation rate per unit crest width is
        epsilon = dE / (t2 - t1). Their inertial model gives (epsilon / epsilon0) (db / d0) =
        0.0281 (hb / d0)^3.152, epsilon0 = rho g^(3/2) d0^(5/2), hb the wave height at breaking and
        db the still water depth under the vertical face. Here the whole run is SPH and the face
        is read from the smoothed surface ('deltaSph'); the energies come from the recorded
        frames, so the frame interval sets how finely t2 is found.

        Parameters:
        -----------
        reference : BreakingWave | None
            The still water run of the same tank, whose drift `energyHistory` takes out; None
            takes nothing out, and the settling of the start then reads as energy lost

        Returns:
        --------
        dict : t1, t2 and their difference [s], the energy at breaking and at t2 [J/m], the share
            lost, epsilon [W/m], the scaled rate and the fit at the same hb and db, and whether t2
            was reached in the run

        Raises:
        -------
        ValueError : If the wave has not broken, or its face was read in cells, which gives no
            face depth.

        '''

        point = self.breakingPoint
        if point is None or not np.isfinite(point.faceDepth):
            raise ValueError('The dissipation needs a breaking point read from the smoothed surface.')
        times, kinetic, potential = self.energyHistory(reference)
        energy = kinetic + potential
        d, g, rho = self.depth, constants.gravity, constants.freshwaterDensity
        breakingEnergy = float(np.interp(point.time, times, energy))

        # The first crossing after breaking of kinetic down through potential energy
        surplus = kinetic - potential
        after = np.flatnonzero((times > point.time) & (surplus <= 0.0))
        reached = after.size > 0
        if reached:
            k = int(after[0])
            share = surplus[k - 1] / (surplus[k - 1] - surplus[k])
            endTime = float(times[k - 1] + share * (times[k] - times[k - 1]))
            endEnergy = float(energy[k - 1] + share * (energy[k] - energy[k - 1]))
        else:
            endTime, endEnergy = float(times[-1]), float(energy[-1])

        rate = (breakingEnergy - endEnergy) / (endTime - point.time)

        return {
            'breakingTime': point.time,
            'activeEndTime': endTime,
            'activeDuration': endTime - point.time,
            'activeDurationOverScale': (endTime - point.time) / math.sqrt(d / g),
            'startEnergy': float(energy[0]),
            'breakingEnergy': breakingEnergy,
            'activeEndEnergy': endEnergy,
            'energyShareLost': (breakingEnergy - endEnergy) / breakingEnergy,
            'dissipationRate': rate,
            'scaledDissipation': rate / (rho * g**1.5 * d**2.5) * (point.faceDepth / d),
            'scaledDissipationFit': 0.0281 * (point.crestHeight / d)**3.152,
            'breakingHeightOverDepth': point.crestHeight / d,
            'breakingFaceDepthOverDepth': point.faceDepth / d,
            'activeEndReached': reached,
        }

    def drawFrame(self, index: int = -1):

        '''A recorded frame as a Matplotlib figure; see `figures.drawBreakingWaveFrame`.'''

        return figures.drawBreakingWaveFrame(self, index)

    def writeAnimation(self, path: str, fps: int = 12, frameStep: int = 1, dpi: int = 90) -> str:

        '''Write the recorded transient as a GIF; see `figures.writeBreakingWaveAnimation`.'''

        return figures.writeBreakingWaveAnimation(self, path, fps = fps, dpi = dpi, frameStep = frameStep)
