
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

Not validated: the breaking point. The reference is the fully nonlinear potential flow computation
of Grilli et al. (1997) for H/d = 0.3 on a 1:35 slope, Hb/hb = 1.380 and Hb/d = 0.476 (tabulated in
Table 8 of Senthilkumar 2016, arXiv:1601.06822), which places breaking at hb/d = 0.345. At 16
particles per depth the model breaks at hb/d = 0.097 with Hb/d = 0.325: 72 percent too shallow and
32 percent too low (tests/testValidationBreaking.py, an expected failure). Numerical damping drains
the wave faster than shoaling builds it over the 30 depths of slope, and the fixed particle size
leaves 5.5 particles across the water column where Grilli's wave breaks. The front face does lean
forward of vertical by up to four spacings as the wave overturns, but where and how high it breaks
are not to be trusted.

Validated: the wave before the slope. Its celerity is within 0.5 percent of sqrt(g (d + H)) at 12
particles per depth and 0.3 percent at 16 (tests/testValidationSolitary.py), and the solver
reproduces the first-mode sloshing period within 2.2 percent (tests/testValidationSloshing.py). The
crest loses height as it travels, 6.5 percent per 11 depths at 16 particles per depth against 20
percent at 8, and that is the error that grows over a long slope.

Post-breaking flow (the bore and the runup tongue) has no reference here; at a few particles across
the jet it is a qualitative picture of where the water goes, not a measurement. Closing the
breaking gap takes higher resolution or a density-diffusion term (delta-SPH) in place of the
Shepard filter, rerun against the same Grilli case.

References:
-----------
Grilli, S.T., Svendsen, I.A. & Subramanya, R. (1997) -- Breaking criterion and characteristics
    for solitary waves on slopes, J. Waterway, Port, Coastal, Ocean Eng. 123(3), 102-112.
Monaghan, J.J. & Kos, A. (1999) -- Solitary waves on a Cretan beach, J. Waterway, Port, Coastal,
    Ocean Eng. 125(3), 145-154.
Dalrymple, R.A. & Rogers, B.D. (2006) -- Numerical modeling of water waves with the SPH method,
    Coastal Engineering 53, 141-147.

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
from pysurf.errors import NumericalInstabilityError
from pysurf.sphBoundaries import BedBoundary
from pysurf.sphParticles import ParticleSystem
from pysurf.sphSolver import SimulationConfig, WcsphSolver
from pysurf.validation import applyRules, integerRule, numericRule
from pysurf.waveBreaking import BreakingTracker
from pysurf.waveMakers import solitaryWaveFluid

# Inputs a breaking wave must satisfy before anything is built
inputRules = (
    numericRule('depth',             'Still water depth',   units = 'm', minimum = 0.0),
    numericRule('waveHeight',        'Wave height',         units = 'm', minimum = 0.0),
    numericRule('beachSlope',        'Beach slope',         minimum = 0.0, maximum = 1.0),
    integerRule('particlesPerDepth', 'Particles per depth', minimum = 4, exclusiveMinimum = False),
    numericRule('frameInterval',     'Frame interval',      units = 's', minimum = 0.0),
    numericRule('alphaViscosity',    'Artificial viscosity', minimum = 0.0, exclusiveMinimum = False),
    numericRule('runupAllowance',    'Runup allowance',     minimum = 0.0),
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

    A solitary wave breaking on a plane beach.

    Parameters:
    -----------
    depth : float
        Still water depth over the flat bottom d [m]
    waveHeight : float
        Incident solitary wave height above still water H [m]
    beachSlope : float
        Beach slope tan(beta) [-]
    particlesPerDepth : int
        Particles across the still water depth; sets the resolution
    frameInterval : float
        Simulated time between recorded frames [s]
    alphaViscosity : float
        Monaghan artificial viscosity coefficient
    runupAllowance : float
        How far above still water the beach extends, in depths
    endTime : float | None
        Simulated time to run [s]; None runs to 1.15 times the estimated arrival at the shoreline

    '''

    # -- Constructor -- #
    def __init__(self, depth: float = 1.0, waveHeight: float = 0.3, beachSlope: float = 1.0 / 35.0,
                 particlesPerDepth: int = 16, frameInterval: float = 0.1, alphaViscosity: float = 0.02,
                 runupAllowance: float = 0.8, endTime: float | None = None) -> None:

        self.depth = depth                          # [m]
        self.waveHeight = waveHeight                # [m]
        self.beachSlope = beachSlope                # [-]
        self.particlesPerDepth = particlesPerDepth
        self.frameInterval = frameInterval          # [s]
        self.alphaViscosity = alphaViscosity
        self.runupAllowance = runupAllowance
        self.endTime = endTime                      # [s]

        # Speed of sound as a multiple of the solitary wave celerity, and steps between breaking checks
        self.soundSpeedFactor = 10.0
        self.trackInterval = 5

        self.bathymetry: Bathymetry | None = None
        self.boundary: BedBoundary | None = None
        self.particles: ParticleSystem | None = None
        self.solver: WcsphSolver | None = None
        self.tracker: BreakingTracker | None = None
        self.frames: list[WaveFrame] = []
        self.wallClockSeconds = 0.0                 # [s]

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

    # -- Geometry -- #
    @property
    def spacing(self) -> float:
        '''Particle spacing [m].'''
        return self.depth / self.particlesPerDepth

    @property
    def crestX(self) -> float:
        '''Initial crest position, where the seaward wall sees 1 percent of the wave height [m].'''
        return waveTheory.solitaryWaveHalfLength(self.waveHeight, self.depth, 0.01)

    @property
    def toeX(self) -> float:
        '''Toe of the slope, the same distance again past the crest [m].'''
        return 2.0 * self.crestX

    @property
    def shorelineX(self) -> float:
        '''Where still water meets the beach [m].'''
        return self.toeX + self.depth / self.beachSlope

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
        h = np.maximum(self.depth - np.maximum(x - self.toeX, 0.0) * self.beachSlope, 0.0)
        celerity = np.sqrt(constants.gravity * (h + self.waveHeight))

        return float(trapezoid(1.0 / celerity, x))

    # -- Public Methods -- #
    def build(self) -> None:

        '''

        Lay out the beach, the boundary, the moving solitary wave, the solver and the tracker.

        '''

        d, s = self.depth, self.spacing
        endX = self.toeX + (1.0 + self.runupAllowance) * d / self.beachSlope

        self.bathymetry = Bathymetry.planeBeach(depth = d, toeX = self.toeX, slope = self.beachSlope, endX = endX)
        self.boundary = BedBoundary(self.bathymetry, s, nLayers = 3, wallTop = d + 3.0 * self.waveHeight)

        fluidPositions, fluidVelocities = solitaryWaveFluid(self.bathymetry, self.waveHeight, self.crestX, s)
        self.particles = ParticleSystem.assemble(fluidPositions, self.boundary.positions(), s,
                                                 constants.freshwaterDensity, fluidVelocities)

        config = SimulationConfig(
            domainMin       = np.array([0.0, 0.0]),
            domainMax       = np.array([endX, d + 3.0 * self.waveHeight]),
            particleSpacing = s,
            referenceDensity = constants.freshwaterDensity,
            gravity         = np.array([0.0, -constants.gravity]),
            endTime         = self.endTime if self.endTime is not None else 1.15 * self.estimatedArrivalTime(),
            maxTimeStep     = 1.0,
            outputInterval  = self.frameInterval,
            speedOfSound    = self.soundSpeedFactor * waveTheory.solitaryWaveSpeed(self.waveHeight, d),
            alphaViscosity  = self.alphaViscosity,
            initialSurface  = lambda x: d + waveTheory.solitaryWaveProfile(x, self.waveHeight, d, self.crestX),
        )
        self.solver = WcsphSolver(config, boundary = self.boundary)
        self.solver.initialize(self.particles)

        # A lean offshore of the toe would be start-up noise, never breaking
        self.tracker = BreakingTracker(self.bathymetry, s, faceReach = d, searchFromX = self.toeX)
        self.frames = []
        self._recordFrame()

    def _recordFrame(self) -> None:

        '''Store the fluid at the current instant.'''

        p, fluid = self.particles, self.particles.isFluid
        state = self.solver.currentState
        self.frames.append(WaveFrame(time = self.solver.time, positions = p.positions[fluid].copy(),
                                     speeds = np.linalg.norm(p.velocities[fluid], axis = 1),
                                     kineticEnergy = state.kineticEnergy, potentialEnergy = state.potentialEnergy))

    def run(self, endTime: float | None = None, progress: bool = True) -> None:

        '''

        Step the simulation to `endTime`, recording frames and watching for breaking.

        Parameters:
        -----------
        endTime : float | None
            Simulated time to stop at [s]; the configured end time by default
        progress : bool
            Print a line roughly every tenth of the run

        Raises:
        -------
        NumericalInstabilityError : If any particle position stops being finite.

        '''

        if self.solver is None:
            self.build()

        solver, tracker = self.solver, self.tracker
        stopTime = endTime if endTime is not None else solver.config.endTime
        nextFrame = solver.time + self.frameInterval
        nextReport = solver.time + stopTime / 10.0
        started = timeModule.perf_counter()

        if progress:
            print(f'BreakingWave: {self.particles.nFluid} fluid and {self.particles.nBoundary} boundary particles, '
                  f'S0 = {self.slopeParameter:.3f} ({self.predictedBreakerType}), running to {stopTime:.2f} s')

        while solver.time < stopTime:
            state = solver.step()

            if state.step % self.trackInterval == 0:
                tracker.update(solver.time, self.particles.positions[self.particles.isFluid])
            if solver.time >= nextFrame:
                if not np.all(np.isfinite(self.particles.positions)):
                    raise NumericalInstabilityError('Particle positions stopped being finite.',
                                                    variableName = 'positions', operation = 'WCSPH step',
                                                    context = {'time': solver.time, 'step': state.step})
                self._recordFrame()
                nextFrame += self.frameInterval
            if progress and solver.time >= nextReport:
                elapsed = timeModule.perf_counter() - started
                broken = 'breaking recorded' if tracker.breakingPoint else 'not broken'
                print(f'  t = {solver.time:6.2f} s  step {state.step:7d}  dt {state.dt:.2e} s  '
                      f'wall {elapsed:7.1f} s  crest {tracker.crestHeight[-1] if tracker.crestHeight else 0.0:.3f} m  '
                      f'{broken}')
                nextReport += stopTime / 10.0

        self.wallClockSeconds += timeModule.perf_counter() - started

    @property
    def breakingPoint(self):
        '''Where the wave broke, or None.'''
        return self.tracker.breakingPoint if self.tracker is not None else None

    def summary(self) -> dict:

        '''

        The breaking measures in dimensional and depth-normalized form.

        Returns:
        --------
        dict : slope parameter, breaker types, breaking index, Hb/d, hb/d, breaking position from
            the toe in depths, maximum runup over depth, particle counts and wall-clock time

        '''

        point = self.breakingPoint
        result = {
            'slopeParameter': self.slopeParameter,
            'predictedBreakerType': self.predictedBreakerType,
            'particlesPerDepth': self.particlesPerDepth,
            'fluidParticles': self.particles.nFluid if self.particles is not None else 0,
            'wallClockSeconds': self.wallClockSeconds,
            'broke': point is not None,
        }
        if point is not None:
            result.update({
                'breakingTime': point.time,
                'breakingIndex': point.breakingIndex,
                'breakingHeightOverDepth': point.crestHeight / self.depth,
                'breakingDepthOverDepth': point.depth / self.depth,
                'breakingDistanceFromToe': (point.crestX - self.toeX) / self.depth,
            })
        if self.tracker is not None and np.isfinite(self.tracker.maxRunup):
            result['maxRunupOverDepth'] = self.tracker.maxRunup / self.depth

        return result

    def drawFrame(self, index: int = -1):

        '''A recorded frame as a Matplotlib figure; see `figures.drawBreakingWaveFrame`.'''

        return figures.drawBreakingWaveFrame(self, index)

    def writeAnimation(self, path: str, fps: int = 12, frameStep: int = 1, dpi: int = 90) -> str:

        '''Write the recorded transient as a GIF; see `figures.writeBreakingWaveAnimation`.'''

        return figures.writeBreakingWaveAnimation(self, path, fps = fps, dpi = dpi, frameStep = frameStep)
