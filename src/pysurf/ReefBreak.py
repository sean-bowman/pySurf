
# -- pySurf: Reef Break -- #

'''

Regular swell breaking over a reef in open water, solved for its free surface with the
Serre-Green-Naghdi equations.

`ReefBreak` drives `pysurf.sgnSolver` with a train of regular waves instead of a single solitary
wave, over a bed of any profile; by default a reef break from `Bathymetry.reefBreak`: a flat bottom
offshore, a steep reef face, a shallow reef flat, a deeper lagoon behind it and a beach. Each part
is a module:

    pysurf.bathymetry       the reef profile
    pysurf.sgnSolver        the SGN equations with improved dispersion, the internal wave source,
                            the absorbing layer, and breaking that starts and stops front by front
    pysurf.figures          the frames, and the transient as a GIF

    reef = ReefBreak(depth = 10.0, waveHeight = 1.5, period = 12.0)
    reef.run()
    reef.summary()
    reef.writeAnimation('reefBreak.gif')

The waves come from a mass source offshore of the reef (Wei, Kirby and Sinha 1999), which radiates
the requested height both ways; the wave running seaward is absorbed in a layer at the offshore end.
A wave steepening on the reef face starts breaking when its front passes 30 degrees and becomes a
bore across the reef flat; if the bore weakens in the deeper lagoon, its Froude number falls below
1.3 and it stops breaking (Tissier et al. 2012). The dispersion parameter is Bonneton et al.'s
(2011) 1.159, which holds the linear phase speed within 1 percent of Airy's out to k d = 3, the
range of swell in the deeper water seaward of a reef.

The surface is single-valued, so the waves do not overturn; `pysurf.OverturningWave` gives the
shape of the plunging lip over a reef profile.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

The wave source, the absorbing layer and the improved dispersion are checked against linear theory
in tests/testSgnSolver.py: the radiated height within 0.4 percent of the target at 120 cells per
wavelength (1 percent at 80), reflection from the absorbing layer near 0.1 percent, and the standing
wave period with alpha = 1.159 within 0.2 percent of the alpha-SGN value. The breaking switch is
validated for a solitary wave on a plane beach against Grilli et al. (1997) through
`pysurf.SurfaceBreakingWave`.

On the default reef (1.5 m, 12 s waves; 10 m offshore, 1:5 face, 2.5 m reef flat, 6 m lagoon) every
wave shoals to about 2.1 m on the face and starts breaking at the reef edge over 3.2 m of still
water, stops breaking in the lagoon with a bore Froude number of 1.11, and breaks again as a shore
break in 1.2 m of water, the same every period.

Not validated: breaking over a reef. Run in Blenkinsopp and Chaplin's (2008) flume by
`pysurf.ReefOverturn`, the same solver gets the incident height within 3.5 percent but reflects 1.5
to 2.1 times what the flume reflected and passes 11 to 44 percent more height over the reef: its
breaking dissipates too little on a steep reef face. Fringing-reef flume data such as Yao et al.
(2012) would test it at a scale nearer the sea's.

References:
-----------
Wei, G., Kirby, J.T. & Sinha, A. (1999) -- Generation of waves in Boussinesq models using a source
    function method, Coastal Engineering 36, 271-299.
Bonneton, P., Chazel, F., Lannes, D., Marche, F. & Tissier, M. (2011) -- A splitting approach for
    the fully nonlinear and weakly dispersive Green-Naghdi model, J. Comput. Phys. 230, 1479-1498.
Tissier, M., Bonneton, P., Marche, F., Chazel, F. & Lannes, D. (2012) -- A new approach to handle
    wave breaking in fully non-linear Boussinesq models, Coastal Engineering 67, 54-66.

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
from pysurf.sgnSolver import SgnConfig, SgnSolver, WaveSource
from pysurf.validation import applyRules, numericRule

# Inputs a reef break must satisfy before anything is built
inputRules = (
    numericRule('depth',         'Offshore depth',    units = 'm', minimum = 0.0),
    numericRule('waveHeight',    'Wave height',       units = 'm', minimum = 0.0),
    numericRule('period',        'Wave period',       units = 's', minimum = 0.0),
    numericRule('cellSize',      'Cell size',         units = 'm', minimum = 0.0),
    numericRule('periods',       'Periods to run',    minimum = 0.0),
    numericRule('frameInterval', 'Frame interval',    units = 's', minimum = 0.0),
)

@dataclass
class ReefFrame:

    '''

    One recorded instant of the surface over the reef.

    Parameters:
    -----------
    time : float
        Simulated time [s]
    surface, velocity, depth : np.ndarray
        Free-surface elevation [m], depth-averaged velocity [m/s] and depth [m] at the cell centers
    breakingZone : np.ndarray
        Boolean mask of the cells carried as breaking fronts

    '''

    time: float
    surface: np.ndarray
    velocity: np.ndarray
    depth: np.ndarray
    breakingZone: np.ndarray

@dataclass
class BreakingEvent:

    '''

    A wave front starting or stopping breaking.

    Parameters:
    -----------
    time : float
        Simulated time [s]
    kind : str
        'start' or 'stop'
    x : float
        Steepest point of the front [m]
    stillDepth : float
        Still water depth under the front's crest [m]
    value : float
        Front height for a start [m]; bore Froude number for a stop

    '''

    time: float
    kind: str
    x: float
    stillDepth: float
    value: float

class ReefBreak:

    '''

    Regular swell breaking over a reef, solved with the SGN equations.

    Parameters:
    -----------
    depth : float
        Still water depth offshore d [m]
    waveHeight : float
        Height of the waves the source sends toward the reef H [m]
    period : float
        Wave period T [s]
    bathymetry : Bathymetry | None
        Bed profile, flat at depth d over the first 2.5 wavelengths for the absorbing layer and the
        source; None builds the default reef
    cellSize : float | None
        Grid spacing [m]; a twentieth of the offshore depth by default
    periods : float
        Wave periods to run after the first waves reach the reef
    frameInterval : float
        Simulated time between recorded frames [s]
    breakingAngle : float
        Front slope at which a wave starts breaking [deg]
    terminationFroude : float
        Bore Froude number below which a wave stops breaking

    '''

    # -- Constructor -- #
    def __init__(self, depth: float = 10.0, waveHeight: float = 1.5, period: float = 12.0,
                 bathymetry: Bathymetry | None = None, cellSize: float | None = None, periods: float = 10.0,
                 frameInterval: float = 0.25, breakingAngle: float = 30.0, terminationFroude: float = 1.3) -> None:

        self.depth = depth                          # [m]
        self.waveHeight = waveHeight                # [m]
        self.period = period                        # [s]
        self.cellSize = cellSize if cellSize is not None else depth / 20.0     # [m]
        self.periods = periods
        self.frameInterval = frameInterval          # [s]
        self.breakingAngle = breakingAngle          # [deg]
        self.terminationFroude = terminationFroude
        self.customBathymetry = bathymetry

        self.alpha = 1.159
        omega = 2.0 * np.pi / period
        self.wavelength = 2.0 * np.pi / waveTheory.serreWavenumber(omega, depth, alpha = self.alpha)   # [m]

        self.bathymetry: Bathymetry | None = None
        self.solver: SgnSolver | None = None
        self.frames: list[ReefFrame] = []
        self.events: list[BreakingEvent] = []
        self.wallClockSeconds = 0.0                 # [s]

        self._validateInputs()

    # -- Private Helper Methods -- #
    def _validateInputs(self) -> None:

        '''

        Check the inputs against the rule table.

        Raises:
        -------
        InvalidInputError : If any input is out of range.

        '''

        applyRules(self, inputRules)

    def _recordFrame(self) -> None:

        '''Store the surface at the current instant.'''

        solver = self.solver
        self.frames.append(ReefFrame(time = solver.time, surface = solver.surface.copy(), velocity = solver.velocity.copy(),
                                     depth = solver.depth.copy(), breakingZone = solver.breakingZone.copy()))

    # -- Geometry -- #
    @property
    def spongeWidth(self) -> float:
        '''Width of the absorbing layer at the offshore end, one and a half wavelengths [m].'''
        return 1.5 * self.wavelength

    @property
    def sourceX(self) -> float:
        '''Center of the wave source, half a wavelength inside the absorbing layer's edge [m].'''
        return self.spongeWidth + 0.5 * self.wavelength

    def defaultBathymetry(self) -> Bathymetry:

        '''

        The default reef: the toe a wavelength past the source, a 1:5 face up to a reef flat a
        quarter of the offshore depth deep and half a wavelength long, a lagoon at 60 percent of the
        offshore depth for a wavelength, and a 1:20 beach rising 30 percent of the depth above still
        water.

        '''

        d = self.depth

        return Bathymetry.reefBreak(depth = d, toeX = self.sourceX + self.wavelength, faceSlope = 0.2,
                                    reefDepth = 0.25 * d, reefLength = 0.5 * self.wavelength, lagoonDepth = 0.6 * d,
                                    lagoonLength = self.wavelength, beachSlope = 0.05, beachHeight = 0.3 * d)

    # -- Public Methods -- #
    def build(self) -> None:

        '''Lay out the reef, the grid, the wave source and the absorbing layer.'''

        d = self.depth
        self.bathymetry = self.customBathymetry if self.customBathymetry is not None else self.defaultBathymetry()
        config = SgnConfig(cellSize = self.cellSize, dryDepth = 1e-5 * d, dispersiveDepth = 0.05 * d,
                           breakingAngle = self.breakingAngle, terminationFroude = self.terminationFroude,
                           dispersionAlpha = self.alpha, spongeLeft = self.spongeWidth,
                           spongeRate = 4.0 * 2.0 * np.pi / self.period)
        source = WaveSource(self.waveHeight, self.period, d, self.sourceX, alpha = self.alpha)
        self.solver = SgnSolver(self.bathymetry, config, source = source)
        self.frames, self.events = [], []
        self._recordFrame()

    def arrivalTime(self) -> float:

        '''Time for the first waves to travel from the source to the end of the reef flat at the group speed [s].'''

        x = np.linspace(self.sourceX, self.bathymetry.x[3], 500)
        depth = np.maximum(self.bathymetry.depth(x), 0.05 * self.depth)
        speed = waveTheory.linearGroupSpeed(self.period, depth)

        return float(trapezoid(1.0 / speed, x))

    def run(self, endTime: float | None = None, progress: bool = True) -> None:

        '''

        Step the simulation, recording frames and every start and stop of breaking.

        Parameters:
        -----------
        endTime : float | None
            Simulated time to stop at [s]; by default the ramp, the arrival at the reef and the
            configured number of periods
        progress : bool
            Print a line roughly every tenth of the run

        Raises:
        -------
        NumericalInstabilityError : If the depth or discharge stops being finite.

        '''

        if self.solver is None:
            self.build()

        solver = self.solver
        if endTime is None:
            endTime = solver.source.rampPeriods * self.period + self.arrivalTime() + self.periods * self.period
        nextFrame = solver.time + self.frameInterval
        nextReport = solver.time + endTime / 10.0
        started = timeModule.perf_counter()

        if progress:
            print(f'ReefBreak: {solver.x.size} cells, H = {self.waveHeight} m, T = {self.period} s, '
                  f'L = {self.wavelength:.1f} m, running to {endTime:.1f} s')

        while solver.time < endTime:
            solver.step()
            for x, stillDepth, height in solver.startedFronts:
                self.events.append(BreakingEvent(solver.time, 'start', x, stillDepth, height))
            for x, stillDepth, froude in solver.stoppedFronts:
                self.events.append(BreakingEvent(solver.time, 'stop', x, stillDepth, froude))

            if solver.time >= nextFrame:
                if not (np.all(np.isfinite(solver.depth)) and np.all(np.isfinite(solver.discharge))):
                    raise NumericalInstabilityError('Depth or discharge stopped being finite.',
                                                    variableName = 'depth', operation = 'SGN step',
                                                    context = {'time': solver.time, 'step': solver.steps})
                self._recordFrame()
                nextFrame += self.frameInterval
            if progress and solver.time >= nextReport:
                elapsed = timeModule.perf_counter() - started
                print(f'  t = {solver.time:6.1f} s  step {solver.steps:6d}  wall {elapsed:6.1f} s  '
                      f'breaking fronts {len(solver.breakingFronts)}  starts {sum(e.kind == "start" for e in self.events)}')
                nextReport += endTime / 10.0

        self.wallClockSeconds += timeModule.perf_counter() - started

    def summary(self, settledPeriods: float | None = None) -> dict:

        '''

        Where the waves start and stop breaking over the last periods of the run, grouped into
        breaking zones: starts within a quarter wavelength of each other belong to one zone (the
        reef edge, a shore break behind a lagoon), and each zone is reported on its own.

        Parameters:
        -----------
        settledPeriods : float | None
            Periods at the end of the run to average over; half the configured periods by default

        Returns:
        --------
        dict : the wavelength, the grid size, the wall-clock time, and per breaking zone, seaward
            first, the number of starts per period, the mean position and still water depth, and the
            mean front height and breaker index (height over depth) when breaking starts; then the
            mean position and depth where breaking stops

        '''

        window = (settledPeriods if settledPeriods is not None else 0.5 * self.periods) * self.period
        endTime = self.solver.time if self.solver is not None else 0.0
        recent = [event for event in self.events if event.time >= endTime - window]
        result = {'wavelength': self.wavelength, 'cells': self.solver.x.size if self.solver is not None else 0,
                  'wallClockSeconds': self.wallClockSeconds, 'zones': [], 'stops': []}

        for kind, key in (('start', 'zones'), ('stop', 'stops')):
            events = sorted((event for event in recent if event.kind == kind), key = lambda event: event.x)
            groups, group = [], []
            for event in events:
                if group and event.x - group[-1].x > 0.25 * self.wavelength:
                    groups.append(group)
                    group = []
                group.append(event)
            if group:
                groups.append(group)
            for members in groups:
                x = np.array([event.x for event in members])
                depth = np.array([event.stillDepth for event in members])
                value = np.array([event.value for event in members])
                record = {'x': float(np.mean(x)), 'stillDepth': float(np.mean(depth)),
                          'perPeriod': len(members) * self.period / window}
                if kind == 'start':
                    record.update({'frontHeight': float(np.mean(value)),
                                   'breakerIndex': float(np.mean(value / np.maximum(depth, 1e-9)))})
                else:
                    record['boreFroude'] = float(np.mean(value))
                result[key].append(record)

        return result

    def drawFrame(self, index: int = -1):

        '''A recorded frame as a Matplotlib figure; see `figures.drawReefFrame`.'''

        return figures.drawReefFrame(self, index)

    def writeAnimation(self, path: str, fps: int = 12, frameStep: int = 1, dpi: int = 90,
                       lastPeriods: float | None = None) -> str:

        '''Write the recorded transient as a GIF; see `figures.writeReefAnimation`.'''

        return figures.writeReefAnimation(self, path, fps = fps, dpi = dpi, frameStep = frameStep,
                                          lastPeriods = lastPeriods)
