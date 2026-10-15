
# -- pySurf: Overturning Wave -- #

'''

A solitary wave shoaling over a bed profile until its crest curls over and the jet lands: the shape
of a plunging breaker, from fully nonlinear potential flow.

`OverturningWave` poses the problem of `SurfaceBreakingWave` and `BreakingWave` with a model whose
free surface can fold, so the lip and the tube it encloses are part of the solution. Each part is a
module:

    pysurf.bathymetry       the bed: a plane beach, a reef, or any profile, ending at a wall
    pysurf.waveTheory       the starting wave: the exact SGN solitary wave's surface and potential
    pysurf.bemSolver        the boundary element solution and the Lagrangian surface
    pysurf.figures          the profile sequence of the overturn, and the transient as a GIF

    wave = OverturningWave(depth = 1.0, waveHeight = 0.3, beachSlope = 1 / 35)
    wave.run()
    wave.summary()
    wave.writeAnimation('overturningWave.gif')

The run ends when the jet touches down, or at the end time. The flow is inviscid and irrotational,
which holds well up to touchdown and not after: what follows (the splash-up, the tube collapsing)
needs a model with a broken free surface, such as SPH.

The beach ends in a vertical wall in water a tenth of the offshore depth deep, shoreward of where
the wave breaks, because a free surface meeting a sloping bed (a moving shoreline) is a singular
corner the boundary element method handles poorly; the wave breaks before its front reaches the
wall. A reef profile from `Bathymetry.reefBreak` ends at a wall at the back of the reef flat.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

Breaking point: validated against Grilli et al. (1997) for H/d = 0.3 on a 1:35 slope, with the
breaking point defined as theirs is, the front face first vertical
(tests/testValidationOverturning.py):

    quantity    Grilli      boundary elements, crest spacing 0.02 d     error
    Hb/d        0.476       0.504                                       +5.8 %
    hb/d        0.345       0.359                                       +4.1 %
    Hb/hb       1.380       1.402                                       +1.6 %

The breaking point is converged in resolution (Hb/d 0.504 at crest spacings of 0.02 d and 0.014 d,
hb/d 0.359 at both). The difference from Grilli is the starting wave: theirs is the exact potential
flow solitary wave, this one the exact SGN wave with potential flow's vertical velocity structure,
which adjusts by about 1 percent in height over the first depths of travel. The solver itself is
checked in tests/testBemSolver.py against an exact harmonic function and the Airy standing-wave
period.

Overturn at jet impact: validated against the two-phase DNS of Feddersen et al. (2024) for the Surf
Ranch soliton, a0/h0 = 0.6 over a 0.0693 slope up to a flat 0.371 h0 deep, measured as they measure
it, with the criteria set before the comparison was run:

    quantity      DNS       boundary elements, 0.02 d / 0.014 d     error           criterion
    impact x/h0   42.05     41.85 / 41.85                           -0.20 h0        0.5 h0
    H_I/h0        0.64      0.654 / 0.655                           +2.2 %          5 %
    A_O/H_I^2     0.352     0.349 / 0.350                           -0.9 %          15 %
    A_J/H_I^2     0.22      0.195 / 0.192                           -12 %           20 %
    theta_O       33 deg    30.0 / 30.1 deg                         -3.0 deg        5 deg
    W/L           0.300     0.385 / 0.386                           +28 %, fails    15 %

W/L fails against the DNS but matches the field-scale measurement of Feddersen et al. (2023), 0.38
at zero wind, and Pick and Feddersen's (2026) potential flow gives 0.41 at these conditions: on
this measure the DNS stands apart from both. The field-scale overturn area at zero wind, 0.31, is
11 percent below the model's 0.349.

Across plane slopes the overturn follows the potential flow scaling of Pick and Feddersen (2026) in
five cases from 1:30 to 1:15 and H0/h0 = 0.2 to 0.6: A_O/H_I^2 within 6 percent of their fit, W/L
within 6 percent, theta_O within 2.3 degrees and Hb/hb within 4.1 percent, all inside three times
the fit's RMSE. The jet area A_J/H_I^2 is 2 to 19 percent above the fit, outside three RMSE in
three of the five. Against the DNS the same measure is 12 percent low; it grows fast in the last
instants before impact, which each study fixes in its own way.

The overturn is converged in resolution: on the 1:35 slope the tube is 0.255 d long, 0.279 d high
and 0.0350 d^2 in area at a crest spacing of 0.02 d, against 0.258 d, 0.282 d and 0.0350 d^2 at
0.014 d, and the Surf Ranch measures agree to 1.5 percent between the two spacings.

Not validated: a solitary wave over a reef. Blenkinsopp and Chaplin (2008) measured overturns over
a 1:10 reef under periodic waves; `pysurf.ReefOverturn` runs them, handing this model waves from an
SGN train, and fails against the flume because the train dissipates too little in breaking.

References:
-----------
Grilli, S.T., Svendsen, I.A. & Subramanya, R. (1997) -- Breaking criterion and characteristics for
    solitary waves on slopes, J. Waterway, Port, Coastal, Ocean Eng. 123(3), 102-112.
Blenkinsopp, C.E. & Chaplin, J.R. (2008) -- The effect of relative crest submergence on wave
    breaking over submerged slopes, Coastal Engineering 55(12), 967-974.
Feddersen, F., Fincham, A.M., Brodie, K.L., Young, A.P., Spydell, M.S., Grimes, D.J., Pieszka, M. &
    Hanson, K. (2023) -- Cross-shore wind-induced changes to field-scale overturning wave shape,
    J. Fluid Mech. 958, A4.
Feddersen, F., Hanson, K., Mostert, W. & Fincham, A. (2024) -- Modelling wind-induced changes to
    overturning wave shape, J. Fluid Mech. 1000, A48.
Pick, K. & Feddersen, F. (2026) -- Scaling the shape of shoaling and overturning solitary waves,
    J. Fluid Mech. 1040, A8.
Longuet-Higgins, M.S. (1982) -- Parametric solutions for breaking waves, J. Fluid Mech. 121, 403-424.

Sean Bowman [10/11/2026]

'''

# Standard library imports
from __future__ import annotations

import time as timeModule
from dataclasses import dataclass, field

# Third-party imports
import numpy as np

# Local imports
from pysurf import constants, figures, waveTheory
from pysurf.bathymetry import Bathymetry
from pysurf.bemSolver import (BemConfig, BemSolver, arcLength, crestIndex, crestSpan, frontSlopeMinimum, jetGap,
                              jetGeometry, selfCrossings, trimLoop)
from pysurf.errors import InvalidInputError
from pysurf.validation import applyRules, numericRule

# Inputs an overturning wave must satisfy before anything is built
inputRules = (
    numericRule('depth',               'Still water depth',      units = 'm', minimum = 0.0),
    numericRule('waveHeight',          'Wave height',            units = 'm', minimum = 0.0),
    numericRule('beachSlope',          'Beach slope',            minimum = 0.0, maximum = 1.0),
    numericRule('wallDepthRatio',      'Wall depth over depth',  minimum = 0.0, maximum = 1.0),
    numericRule('crestSpacingRatio',   'Crest spacing',          minimum = 0.0, maximum = 0.2),
    numericRule('surfaceSpacingRatio', 'Surface spacing',        minimum = 0.0, maximum = 0.5),
    numericRule('frameInterval',       'Frame interval',         units = 's', minimum = 0.0),
)

@dataclass
class SurfaceSnapshot:

    '''

    One recorded instant of the free surface.

    Parameters:
    -----------
    time : float
        Simulated time [s]
    surface : np.ndarray
        Free-surface nodes from wall to wall, shape (N, 2) [m]
    speed : np.ndarray
        Water speed at each node [m/s]
    potential : np.ndarray | None
        phi at each node [m^2/s], from which the flow inside the water can be rebuilt

    '''

    time: float
    surface: np.ndarray
    speed: np.ndarray
    potential: np.ndarray | None = None

@dataclass
class CrestTrack:

    '''

    Time history of the highest point of the surface.

    Parameters:
    -----------
    times, crestX, crestHeight : list
        Simulated time [s], crest position [m] and crest height above still water [m]

    '''

    times: list = field(default_factory = list)
    crestX: list = field(default_factory = list)
    crestHeight: list = field(default_factory = list)

@dataclass
class OverturningPoint:

    '''

    Where the wave breaks: its front face first vertical.

    Parameters:
    -----------
    time : float
        Simulated time [s]
    crestX : float
        Crest position [m]
    crestHeight : float
        Crest elevation above still water, Hb [m]
    depth : float
        Still water depth under the crest, hb [m]
    faceX : float
        Where the face is vertical [m]
    faceDepth : float
        Still water depth where the face is vertical [m], the breaking depth of Mostert and Deike
        (2020)

    '''

    time: float
    crestX: float
    crestHeight: float
    depth: float
    faceX: float = 0.0
    faceDepth: float = 0.0

    @property
    def breakingIndex(self) -> float:
        '''Hb / hb [-].'''
        return self.crestHeight / self.depth if self.depth > 0.0 else float('inf')

class OverturningWave:

    '''

    A solitary wave overturning over a bed profile, solved with fully nonlinear potential flow.

    Parameters:
    -----------
    depth : float
        Still water depth over the flat bottom d [m]
    waveHeight : float
        Incident solitary wave height above still water H [m]
    beachSlope : float
        Plane beach slope tan(beta) [-]; ignored when `bathymetry` is given
    bathymetry : Bathymetry | None
        A bed profile to use instead of the plane beach; flat at depth d from x = 0 to at least a
        half length past the starting crest
    wallDepthRatio : float
        Still water depth at the shoreward wall of the plane beach, over d
    crestSpacingRatio : float
        Free-surface node spacing around the crest, over d
    surfaceSpacingRatio : float
        Free-surface node spacing elsewhere, over d
    frameInterval : float
        Simulated time between recorded frames [s]; a fifth of this once the face is steeper than
        45 degrees, so the overturn, which lasts a fraction of a second, is recorded in slow motion
    endTime : float | None
        Simulated time to stop at if the jet has not landed [s]
    startX : float | None
        Starting crest position [m]; by default the wave's half length, where the seaward wall sees
        1 percent of the wave height. It may not be less than that.
    crestWindowRatio : float
        Scale on the stretch of surface kept at the crest spacing, two depths behind the crest and
        one and a half ahead; below one for a wave small against the depth, whose crest needs fine
        nodes over a shorter stretch
    initialState : tuple | None
        A free surface and its potential to start from instead of the solitary wave, as (surface
        nodes from wall to wall, shape (N, 2) [m], potential [m^2/s]); it needs `bathymetry`, and
        `waveHeight` then only scales the run
    crestRange : tuple | None
        Stretch of x, as (low, high) [m], in which the crest, the breaking face and the jet are
        sought; None for the whole tank. A starting surface may hold other waves, such as the bores
        behind a reef, that rise higher against a wall than the wave to follow.

    '''

    # -- Constructor -- #
    def __init__(self, depth: float = 1.0, waveHeight: float = 0.3, beachSlope: float = 1.0 / 35.0,
                 bathymetry: Bathymetry | None = None, wallDepthRatio: float = 0.1, crestSpacingRatio: float = 0.02,
                 surfaceSpacingRatio: float = 0.08, frameInterval: float = 0.05, endTime: float | None = None,
                 startX: float | None = None, crestWindowRatio: float = 1.0, initialState: tuple | None = None,
                 crestRange: tuple[float, float] | None = None) -> None:

        self.depth = depth                              # [m]
        self.waveHeight = waveHeight                    # [m]
        self.beachSlope = beachSlope                    # [-]
        self.customBathymetry = bathymetry
        self.wallDepthRatio = wallDepthRatio
        self.crestSpacingRatio = crestSpacingRatio
        self.surfaceSpacingRatio = surfaceSpacingRatio
        self.frameInterval = frameInterval              # [s]
        self.endTime = endTime                          # [s]
        self.startX = startX                            # [m]
        self.crestWindowRatio = crestWindowRatio
        self.initialState = initialState
        self.crestRange = crestRange                    # [m]

        self.bathymetry: Bathymetry | None = None
        self.solver: BemSolver | None = None
        self.frames: list[SurfaceSnapshot] = []
        self.breakingPoint: OverturningPoint | None = None
        self.touchedDown = False
        self.trimmedLoops = 0
        # Crossings closer than this many nodes along the surface are loops at the jet tip
        self.loopNodes = 15
        self.stopReason = ''
        self.tracker = CrestTrack()
        self.wallClockSeconds = 0.0                     # [s]

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
        applyRules({'heightRatio': self.waveHeight / self.depth},
                   [numericRule('heightRatio', 'Wave height over depth', maximum = waveTheory.mcCowanLimit,
                                note = 'a solitary wave higher than this breaks before it reaches the slope')])
        if self.initialState is not None and self.customBathymetry is None:
            raise InvalidInputError('A starting surface needs the bed it lies over.', parameterName = 'bathymetry')
        if self.startX is not None and self.startX < self.halfLength:
            raise InvalidInputError(f'Starting crest position {self.startX} m is closer to the seaward wall than the '
                                    f'wave\'s half length, {self.halfLength:.3f} m.', parameterName = 'startX',
                                    value = self.startX, validRange = f'>= {self.halfLength:.3f} m')

    def _crest(self) -> tuple[int, float, float]:

        '''Index, position and height above still water of the highest surface node in the crest range.'''

        surface = self.solver.surface
        index = crestIndex(surface, self.crestRange)

        return index, float(surface[index, 0]), float(surface[index, 1] - self.bathymetry.stillWaterLevel)

    def _crestRegion(self) -> tuple[int, int]:

        '''

        Node range from a depth behind the crest to four depths ahead of it, along the arc length: far
        enough ahead to take in the jet, the tube under it and the face it lands on, which for a
        large wave lie more than two depths along the surface from the crest.

        '''

        surface = self.solver.surface
        s = arcLength(surface)
        index, _, _ = self._crest()
        start = int(np.searchsorted(s, s[index] - self.depth))
        stop = int(np.searchsorted(s, s[index] + 4.0 * self.depth))

        return max(start, 0), min(stop, len(s))

    def _recordFrame(self) -> None:

        '''Store the surface at the current instant.'''

        solver = self.solver
        speed = (np.hypot(solver.velocity[:, 0], solver.velocity[:, 1]) if solver.velocity is not None
                 and len(solver.velocity) == len(solver.surface) else np.zeros(len(solver.surface)))
        self.frames.append(SurfaceSnapshot(time = solver.time, surface = solver.surface.copy(), speed = speed,
                                           potential = solver.potential.copy()))

    # -- Geometry -- #
    @property
    def halfLength(self) -> float:
        '''Distance from the crest at which the solitary wave falls to 1 percent of its height [m].'''
        return waveTheory.serreSolitaryWaveHalfLength(self.waveHeight, self.depth, 0.01)

    @property
    def crestX(self) -> float:
        '''Starting crest position: `startX`, or by default the half length from the seaward wall [m].'''
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
        '''Where still water would meet a plane beach; beyond the wall, which closes the tank first [m].'''
        return self.toeX + self.depth / self.beachSlope

    @property
    def slopeParameter(self) -> float:
        '''Grilli's solitary slope parameter S0 for the plane beach [-].'''
        return waveTheory.solitarySlopeParameter(self.beachSlope, self.waveHeight, self.depth)

    @property
    def predictedBreakerType(self) -> str:
        '''Breaker type Grilli's criterion predicts for the plane beach.'''
        return waveTheory.solitaryBreakerType(self.slopeParameter)

    # -- Public Methods -- #
    def build(self) -> None:

        '''Lay out the bed, the boundary element solver and the solitary wave on the surface.'''

        d, H = self.depth, self.waveHeight
        if self.customBathymetry is not None:
            self.bathymetry = self.customBathymetry
        else:
            wallX = self.toeX + (1.0 - self.wallDepthRatio) * d / self.beachSlope
            self.bathymetry = Bathymetry.planeBeach(depth = d, toeX = self.toeX, slope = self.beachSlope, endX = wallX)

        window = self.crestWindowRatio * d
        config = BemConfig(surfaceSpacing = self.surfaceSpacingRatio * d, crestSpacing = self.crestSpacingRatio * d,
                           crestBehind = 2.0 * window, crestAhead = 1.5 * window, transition = window,
                           bedSpacing = 0.25 * d, farSpacing = 0.2 * d, nearDistance = 8.0 * d,
                           crestRange = self.crestRange)
        self.solver = BemSolver(self.bathymetry, config)

        bathymetry = self.bathymetry
        if self.initialState is not None:
            surface, potential = self.initialState
        else:
            x = np.linspace(bathymetry.startX, bathymetry.endX,
                            int((bathymetry.endX - bathymetry.startX) / (0.005 * d)) + 1)
            elevation = waveTheory.serreSolitaryWaveProfile(x, H, d, self.crestX)
            surface = np.column_stack([x, bathymetry.stillWaterLevel + elevation])
            potential = waveTheory.serreSolitarySurfacePotential(x, H, d, self.crestX)
        self.solver.setSurface(surface, potential, regrid = True)

        self.frames, self.tracker = [], CrestTrack()
        self.breakingPoint, self.touchedDown, self.stopReason = None, False, ''
        self._recordFrame()

    def run(self, endTime: float | None = None, progress: bool = True) -> None:

        '''

        Step the wave until the jet lands, the run fails, or `endTime`.

        Parameters:
        -----------
        endTime : float | None
            Simulated time to stop at [s]; the configured end time, or 40 sqrt(d / g), by default
        progress : bool
            Print a line every recorded second of simulated time

        '''

        if self.solver is None:
            self.build()

        solver = self.solver
        stopTime = endTime if endTime is not None else (self.endTime if self.endTime is not None
                                                        else 40.0 * np.sqrt(self.depth / constants.gravity))
        nextFrame = solver.time + self.frameInterval
        nextReport = solver.time + 1.0
        previousSlope, previousTime = 1.0, solver.time
        steepSlope = float(np.cos(np.radians(45.0)))
        started = timeModule.perf_counter()

        if progress:
            print(f'OverturningWave: {len(solver.surface)} surface and {len(solver.bed)} bed nodes, running to '
                  f'{stopTime:.2f} s or touchdown')

        while solver.time < stopTime:
            lastSurface, lastPotential, lastTime = solver.surface.copy(), solver.potential.copy(), solver.time
            try:
                solver.step()
            except (np.linalg.LinAlgError, ValueError, MemoryError) as error:
                solver.surface, solver.potential, solver.time = lastSurface, lastPotential, lastTime
                self.stopReason = f'solver failed: {error}'
                break
            if not (np.all(np.isfinite(solver.surface)) and np.all(np.isfinite(solver.potential))):
                solver.surface, solver.potential, solver.time = lastSurface, lastPotential, lastTime
                self.stopReason = 'the surface stopped being finite'
                break

            index, crestX, crestHeight = self._crest()
            self.tracker.times.append(solver.time)
            self.tracker.crestX.append(crestX)
            self.tracker.crestHeight.append(crestHeight)
            start, stop = self._crestRegion()
            slope = frontSlopeMinimum(solver.surface[start:stop]) if stop - start > 4 else 1.0

            # The breaking point: the front face first vertical, interpolated between the two steps
            if self.breakingPoint is None and slope <= 0.0:
                share = previousSlope / (previousSlope - slope) if previousSlope > slope else 1.0
                time = previousTime + share * (solver.time - previousTime)
                region = solver.surface[start:stop]
                lean = np.gradient(region[:, 0]) / np.hypot(*np.gradient(region, axis = 0).T)
                faceX = float(region[int(np.argmin(lean)), 0])
                self.breakingPoint = OverturningPoint(time = time, crestX = crestX, crestHeight = crestHeight,
                                                      depth = float(self.bathymetry.depth(crestX)), faceX = faceX,
                                                      faceDepth = float(self.bathymetry.depth(faceX)))
            previousSlope, previousTime = slope, solver.time

            if self.breakingPoint is not None:
                crossings = selfCrossings(solver.surface, start, stop)
                # A crossing within a few nodes, around a loop a few crest spacings across, is the jet
                # tip winding into itself: trim it and go on. Anything larger is the jet reaching the
                # face, and trimming it would cut the jet off.
                loopLimit = 4.0 * self.crestSpacingRatio * self.depth

                def isTouchdown(first: int, second: int) -> bool:
                    loop = solver.surface[first + 1:second + 1]
                    return second - first >= self.loopNodes or float(np.hypot(*np.ptp(loop, axis = 0))) > loopLimit

                if any(isTouchdown(first, second) for first, second in crossings):
                    solver.surface, solver.potential, solver.time = lastSurface, lastPotential, lastTime
                    self.touchedDown = True
                    self.stopReason = 'the jet touched down'
                    self._recordFrame()
                    break
                for first, second in sorted(crossings, reverse = True):
                    solver.surface, solver.potential = trimLoop(solver.surface, solver.potential, first, second)
                    self.trimmedLoops += 1

                # The jet lands when it comes within one element of the face, the closest the boundary
                # solution resolves; the surface there is still sound, so the run keeps it
                span = crestSpan(solver.surface, self.crestRange)
                approach = jetGap(solver.surface[span], min(stop, span.stop) - span.start)
                if approach is not None and approach[0] < approach[1]:
                    self.touchedDown = True
                    self.stopReason = 'the jet touched down'
                    self._recordFrame()
                    break

            if solver.time >= nextFrame:
                self._recordFrame()
                interval = self.frameInterval / 5.0 if slope < steepSlope else self.frameInterval
                nextFrame = solver.time + interval
            if progress and solver.time >= nextReport:
                elapsed = timeModule.perf_counter() - started
                state = 'overturning' if self.breakingPoint else 'shoaling'
                print(f'  t = {solver.time:6.2f} s  step {solver.steps:5d}  dt {solver.timeStep:.2e} s  '
                      f'{len(solver.surface):4d} nodes  wall {elapsed:6.1f} s  crest {crestHeight:.3f} m  {state}')
                nextReport += 1.0
        else:
            self.stopReason = 'end time reached'

        self.wallClockSeconds += timeModule.perf_counter() - started

    def summary(self) -> dict:

        '''

        The breaking measures in dimensional and depth-normalized form, and how the run ended.

        Returns:
        --------
        dict : breaking index, Hb/d, hb/d, breaking time and position; when the jet has landed, where
            it lands, the impact height H_I (the highest point of the surface then), the tube under
            the jet (area, length, height and their ratio), and the overturn measures of Feddersen
            et al. (2023, 2024): A_O / H_I^2, A_J / H_I^2, the aspect ratio W / L and angle theta_O
            of the fitted Longuet-Higgins curve, and the fit's rms error over L; whether the jet
            landed, the stop reason, node counts and wall-clock time

        '''

        point = self.breakingPoint
        result = {
            'slopeParameter': self.slopeParameter,
            'predictedBreakerType': self.predictedBreakerType,
            'surfaceNodes': len(self.solver.surface) if self.solver is not None else 0,
            'endTime': self.solver.time if self.solver is not None else 0.0,
            'touchedDown': self.touchedDown,
            'stopReason': self.stopReason,
            'trimmedLoops': self.trimmedLoops,
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
                'breakingFaceDepthOverDepth': point.faceDepth / self.depth,
            })
        # The tube is measured only once the jet has landed; a run that stopped short has no tube,
        # and one that failed may end on a surface no longer fit to measure
        jet = self.jet if self.touchedDown else None
        if jet is not None:
            impactHeight = (float(self.solver.surface[crestIndex(self.solver.surface, self.crestRange), 1])
                            - self.bathymetry.stillWaterLevel)
            result.update({
                'jetTipX': float(jet.tip[0]),
                'impactX': float(jet.impactPoint[0]),
                'impactHeightOverDepth': impactHeight / self.depth,
                'cavityAreaOverDepthSquared': jet.cavityArea / self.depth**2,
                'tubeLengthOverDepth': jet.tubeLength / self.depth,
                'tubeHeightOverDepth': jet.tubeHeight / self.depth,
                'tubeRatio': jet.tubeRatio,
                'overturnAreaOverImpactHeightSquared': jet.cavityArea / impactHeight**2,
                'jetAreaOverImpactHeightSquared': jet.jetArea / impactHeight**2,
                'overturnAspectRatio': jet.aspectRatio,
                'overturnAngle': jet.overturnAngle,
                'overturnFitErrorOverLength': jet.fitError / jet.overturnLength,
            })

        return result

    @property
    def jet(self):
        '''The jet and the tube under it at the last state, or None before the surface overhangs.'''
        if self.solver is None or self.solver.surface is None:
            return None
        return jetGeometry(self.solver.surface[crestSpan(self.solver.surface, self.crestRange)])

    def drawFrame(self, index: int = -1):

        '''A recorded frame as a Matplotlib figure; see `figures.drawOverturningFrame`.'''

        return figures.drawOverturningFrame(self, index)

    def drawProfiles(self, count: int = 8):

        '''The surface at evenly spaced instants through the overturn; see `figures.drawOverturningProfiles`.'''

        return figures.drawOverturningProfiles(self, count)

    def drawOverturn(self):

        '''The overturn at the last state with its fitted curve and measures; see `figures.drawOverturnShape`.'''

        return figures.drawOverturnShape(self)

    def writeAnimation(self, path: str, fps: int = 12, frameStep: int = 1, dpi: int = 90) -> str:

        '''Write the recorded transient as a GIF; see `figures.writeOverturningAnimation`.'''

        return figures.writeOverturningAnimation(self, path, fps = fps, dpi = dpi, frameStep = frameStep)
