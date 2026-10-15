
# -- pySurf: Surface Breaking Wave -- #

'''

A solitary wave running up a plane sand beach, shoaling and breaking, solved for its free surface
with the Serre-Green-Naghdi equations.

`SurfaceBreakingWave` poses the same problem as `pysurf.BreakingWave`, with a free-surface model in
place of particles. Each part is a module:

    pysurf.bathymetry       the beach: a flat bottom, then a constant slope through the shoreline
    pysurf.waveTheory       the exact SGN solitary wave, laid in already moving
    pysurf.sgnSolver        the SGN equations, the shallow-water bore and the breaking switch
    pysurf.waveBreaking     crest tracking, the breaking point and the runup
    pysurf.figures          the frames, and the transient as a GIF

    wave = SurfaceBreakingWave(depth = 1.0, waveHeight = 0.3, beachSlope = 1 / 35)
    wave.run()
    wave.writeAnimation('surfaceWave.gif')

The surface is a single-valued profile, so the wave cannot overturn. It steepens until its front
passes the breaking angle, then the shallow-water equations carry the front as a bore up the beach.
In exchange for the lip, the model has no particle noise and needs only cells along the beach: the
default case runs in about a minute, against 40 minutes for the SPH model.

The geometry follows `BreakingWave`, scaled with the depth: the crest starts where the seaward wall
sees 1 percent of the wave height, the slope begins as far again past the crest, and the beach runs
on above still water far enough to hold the runup. The exact SGN wave is wider than the first-order
one, so the same rule puts the toe 14 percent further out. The model has no viscosity and is
Froude-similar.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

Validated against Grilli et al. (1997) for H/d = 0.3 on a 1:35 slope (tests/testValidationSurfaceWave.py),
with the published 30 degree breaking angle and no tuning:

    quantity    Grilli      SGN, 50 cells per depth     error
    Hb/d        0.476       0.454                       -4.5 %
    hb/d        0.345       0.352                       +2.0 %
    Hb/hb       1.380       1.291                       -6.5 %

The result is converged in resolution (Hb/d 0.452, 0.454, 0.455 at 25, 50 and 100 cells per depth).
The criterion is the sensitivity: Grilli's breaking point is where the face turns vertical, which a
single-valued surface cannot reach, and a 25 or 35 degree angle moves hb/d to 0.390 or 0.321, about
10 percent either way. The non-breaking runup of H/d = 0.019 on a 1:19.85 slope is R/d = 0.0874
against Synolakis' (1987) 0.0890, -1.8 percent.

The solver is verified against closed forms in tests/testSgnSolver.py: lake at rest, Ritter's dam
break, the SGN standing-wave frequency, and the exact solitary wave, which keeps its height and
speed to 1e-4 at 40 cells per depth with second-order convergence.

Not validated: the bore after breaking and the runup of a breaking wave. The shallow-water bore
carries mass and momentum correctly and dissipates energy as a shock, but the runup tongue has no
bed friction; it climbs as a sheet millimeters thick and still moving fast when the run ends, so
the maximum runup the summary reports for a breaking wave is not a measurement.

References:
-----------
Grilli, S.T., Svendsen, I.A. & Subramanya, R. (1997) -- Breaking criterion and characteristics
    for solitary waves on slopes, J. Waterway, Port, Coastal, Ocean Eng. 123(3), 102-112.
Tissier, M., Bonneton, P., Marche, F., Chazel, F. & Lannes, D. (2012) -- A new approach to handle
    wave breaking in fully non-linear Boussinesq models, Coastal Engineering 67, 54-66.
Synolakis, C.E. (1987) -- The runup of solitary waves, J. Fluid Mech. 185, 523-545.

Sean Bowman [10/11/2026]

'''

# Standard library imports
from __future__ import annotations

import time as timeModule
from dataclasses import dataclass

# Third-party imports
import numpy as np
from scipy.integrate import trapezoid

# Local imports
from pysurf import constants, figures, waveTheory
from pysurf.bathymetry import Bathymetry
from pysurf.errors import NumericalInstabilityError
from pysurf.sgnSolver import SgnConfig, SgnSolver
from pysurf.validation import applyRules, integerRule, numericRule
from pysurf.waveBreaking import SurfaceBreakingTracker

# Inputs a surface breaking wave must satisfy before anything is built
inputRules = (
    numericRule('depth',          'Still water depth',  units = 'm', minimum = 0.0),
    numericRule('waveHeight',     'Wave height',        units = 'm', minimum = 0.0),
    numericRule('beachSlope',     'Beach slope',        minimum = 0.0, maximum = 1.0),
    integerRule('cellsPerDepth',  'Cells per depth',    minimum = 8, exclusiveMinimum = False),
    numericRule('frameInterval',  'Frame interval',     units = 's', minimum = 0.0),
    numericRule('runupAllowance', 'Runup allowance',    minimum = 0.0),
    numericRule('breakingAngle',  'Breaking angle',     units = 'deg', minimum = 0.0, maximum = 90.0),
)

@dataclass
class SurfaceFrame:

    '''

    One recorded instant of the free surface.

    Parameters:
    -----------
    time : float
        Simulated time [s]
    surface : np.ndarray
        Free-surface elevation at the cell centers [m]
    velocity : np.ndarray
        Depth-averaged velocity [m/s]
    depth : np.ndarray
        Water depth [m]
    breakingZone : np.ndarray
        Boolean mask of the cells carried by the shallow-water equations as a breaking front

    '''

    time: float
    surface: np.ndarray
    velocity: np.ndarray
    depth: np.ndarray
    breakingZone: np.ndarray

class SurfaceBreakingWave:

    '''

    A solitary wave breaking on a plane beach, solved with the Serre-Green-Naghdi equations.

    Parameters:
    -----------
    depth : float
        Still water depth over the flat bottom d [m]
    waveHeight : float
        Incident solitary wave height above still water H [m]
    beachSlope : float
        Beach slope tan(beta) [-]
    cellsPerDepth : int
        Grid cells per still water depth; sets the resolution
    frameInterval : float
        Simulated time between recorded frames [s]
    runupAllowance : float
        How far above still water the beach extends, in depths
    endTime : float | None
        Simulated time to run [s]; None runs to 1.15 times the estimated arrival at the shoreline
    breakingAngle : float
        Front slope at which the wave starts breaking [deg]

    '''

    # -- Constructor -- #
    def __init__(self, depth: float = 1.0, waveHeight: float = 0.3, beachSlope: float = 1.0 / 35.0,
                 cellsPerDepth: int = 50, frameInterval: float = 0.05, runupAllowance: float = 0.8,
                 endTime: float | None = None, breakingAngle: float = 30.0) -> None:

        self.depth = depth                          # [m]
        self.waveHeight = waveHeight                # [m]
        self.beachSlope = beachSlope                # [-]
        self.cellsPerDepth = cellsPerDepth
        self.frameInterval = frameInterval          # [s]
        self.runupAllowance = runupAllowance
        self.endTime = endTime                      # [s]
        self.breakingAngle = breakingAngle          # [deg]

        # Water shallower than this fraction of the depth is carried without dispersion
        self.dispersiveDepthRatio = 0.05
        # The waterline is the tip of the sheet at least this fraction of the depth thick, so a
        # film a few molecules thick does not set the runup
        self.waterlineDepthRatio = 1e-3

        self.bathymetry: Bathymetry | None = None
        self.solver: SgnSolver | None = None
        self.tracker: SurfaceBreakingTracker | None = None
        self.frames: list[SurfaceFrame] = []
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

    def _recordFrame(self) -> None:

        '''Store the surface at the current instant.'''

        solver = self.solver
        self.frames.append(SurfaceFrame(time = solver.time, surface = solver.surface.copy(),
                                        velocity = solver.velocity.copy(), depth = solver.depth.copy(),
                                        breakingZone = solver.breakingZone.copy()))

    # -- Geometry -- #
    @property
    def cellSize(self) -> float:
        '''Grid spacing [m].'''
        return self.depth / self.cellsPerDepth

    @property
    def crestX(self) -> float:
        '''Initial crest position, where the seaward wall sees 1 percent of the wave height [m].'''
        return waveTheory.serreSolitaryWaveHalfLength(self.waveHeight, self.depth, 0.01)

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
        the wave height held at its initial value [s].

        '''

        x = np.linspace(self.crestX, self.shorelineX, 2001)
        h = np.maximum(self.depth - np.maximum(x - self.toeX, 0.0) * self.beachSlope, 0.0)

        return float(trapezoid(1.0 / np.sqrt(constants.gravity * (h + self.waveHeight)), x))

    # -- Public Methods -- #
    def build(self) -> None:

        '''Lay out the beach, the grid, the moving solitary wave and the tracker.'''

        d, H = self.depth, self.waveHeight
        endX = self.toeX + (1.0 + self.runupAllowance) * d / self.beachSlope
        self.bathymetry = Bathymetry.planeBeach(depth = d, toeX = self.toeX, slope = self.beachSlope, endX = endX)

        # A broken solitary wave on a plane beach breaks on to the shoreline, as the broken waves
        # did on Tissier et al.'s plane beach, so the bore-strength termination is off
        config = SgnConfig(cellSize = self.cellSize, dryDepth = 1e-5 * d, dispersiveDepth = self.dispersiveDepthRatio * d,
                           breakingAngle = self.breakingAngle, terminationFroude = 0.0)
        self.solver = SgnSolver(self.bathymetry, config)

        x, bed = self.solver.x, self.solver.bed
        eta = waveTheory.serreSolitaryWaveProfile(x, H, d, self.crestX)
        velocity = waveTheory.serreSolitaryWaveVelocity(x, H, d, self.crestX)
        depth = np.maximum(d + eta - bed, 0.0)
        # The wave is laid over the flat bottom; ahead of it the water starts at rest
        self.solver.setState(depth, np.where(x < self.toeX, velocity, 0.0))

        self.tracker = SurfaceBreakingTracker(self.bathymetry, searchFromX = self.toeX)
        self.frames = []
        self._recordFrame()

    def run(self, endTime: float | None = None, progress: bool = True) -> None:

        '''

        Step the simulation to `endTime`, recording frames and watching for breaking.

        Parameters:
        -----------
        endTime : float | None
            Simulated time to stop at [s]; the configured or estimated end time by default
        progress : bool
            Print a line roughly every tenth of the run

        Raises:
        -------
        NumericalInstabilityError : If the depth or velocity stops being finite.

        '''

        if self.solver is None:
            self.build()

        solver, tracker = self.solver, self.tracker
        if endTime is None:
            endTime = self.endTime if self.endTime is not None else 1.15 * self.estimatedArrivalTime()
        nextFrame = solver.time + self.frameInterval
        nextReport = solver.time + endTime / 10.0
        waterlineDepth = self.waterlineDepthRatio * self.depth
        started = timeModule.perf_counter()

        if progress:
            print(f'SurfaceBreakingWave: {solver.x.size} cells, S0 = {self.slopeParameter:.3f} '
                  f'({self.predictedBreakerType}), running to {endTime:.2f} s')

        while solver.time < endTime:
            solver.step()
            tracker.update(solver.time, solver.x, solver.surface, solver.depth > waterlineDepth, solver.breaking)

            if solver.time >= nextFrame:
                if not (np.all(np.isfinite(solver.depth)) and np.all(np.isfinite(solver.discharge))):
                    raise NumericalInstabilityError('Depth or discharge stopped being finite.',
                                                    variableName = 'depth', operation = 'SGN step',
                                                    context = {'time': solver.time, 'step': solver.steps})
                self._recordFrame()
                nextFrame += self.frameInterval
            if progress and solver.time >= nextReport:
                elapsed = timeModule.perf_counter() - started
                state = 'breaking' if solver.breaking else ('broken' if tracker.breakingPoint else 'not broken')
                print(f'  t = {solver.time:6.2f} s  step {solver.steps:6d}  dt {solver.timeStep:.2e} s  '
                      f'wall {elapsed:6.1f} s  crest {tracker.crestHeight[-1]:.3f} m  {state}')
                nextReport += endTime / 10.0

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
        dict : slope parameter, breaker type, breaking index, Hb/d, hb/d, breaking position from
            the toe in depths, front angle at breaking, maximum runup over depth, grid size and
            wall-clock time

        '''

        point = self.breakingPoint
        result = {
            'slopeParameter': self.slopeParameter,
            'predictedBreakerType': self.predictedBreakerType,
            'cellsPerDepth': self.cellsPerDepth,
            'cells': self.solver.x.size if self.solver is not None else 0,
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
                'frontAngleAtBreaking': point.frontAngle,
            })
        if self.tracker is not None and np.isfinite(self.tracker.maxRunup):
            result['maxRunupOverDepth'] = self.tracker.maxRunup / self.depth

        return result

    def drawFrame(self, index: int = -1):

        '''A recorded frame as a Matplotlib figure; see `figures.drawSurfaceWaveFrame`.'''

        return figures.drawSurfaceWaveFrame(self, index)

    def writeAnimation(self, path: str, fps: int = 12, frameStep: int = 1, dpi: int = 90) -> str:

        '''Write the recorded transient as a GIF; see `figures.writeSurfaceWaveAnimation`.'''

        return figures.writeSurfaceWaveAnimation(self, path, fps = fps, dpi = dpi, frameStep = frameStep)
