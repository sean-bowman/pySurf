
# -- pySurf: A Solitary Wave Breaking over a 3D Bed -- #

'''

A solitary wave breaking over a bed of any shape, run in SPH on its own: `BreakingWave`'s
low-dissipation scheme in a rectangular tank over a height field.

    tank        the bed (`bathymetry.HeightField`) with free-slip walls on its bounds
                (`sphBoundaries.HeightFieldBoundary`)
    start       the exact potential flow solitary wave of height H in water `depth` deep, its crest
                at `crestX` and straight along y (`waveMakers.exactSolitaryWaveFluid3D`), with its
                pressure; each boundary particle starts with the pressure of the water just inside
                it
    scheme      `BreakingWave`'s 'deltaSph': Wendland C2 at h = 3 spacings, hydrostatic delta-SPH
                density diffusion, the kernel gradient correction in the continuity equation, wall
                pressure extrapolated from the water, alpha 0.01 with the limited artificial
                viscosity, fluid density held at 0.9 rho0 or above; every wall free-slip
    sections    vertical planes y = constant on which a `waveBreaking.BreakingTracker` reads the
                breaking point from the smoothed surface ('contour'): the fluid within
                `sectionRows` / 2 spacings of the plane, each particle standing for
                s^3 / (rows s) of the plane's area, so the plane holds as many particles per area
                as a 2D run
    snapshots   the whole fluid's positions and velocities at `recordTimes`, and each section's
                fluid every `frameInterval`

`velocity` gives the flow at any points at a snapshot's time: the Shepard-normalized Wendland
average of the fluid around each point within the kernel's support.

A side wall disturbs the flow beside it. Over a flat bed at 32 particles per depth, 0.3 s after
the start of a solitary wave of H/d = 0.3 between walls 1 d apart, the top of the particle row next
to each wall stands 13 percent lower at the crest than mid-tank, the second row 7 percent lower,
the third 1 percent, and the rows beyond within 0.3 percent. Still water between the same walls
settles alike beside them and mid-tank (at 16 particles per depth). The likely cause, not yet
confirmed, is that the wall's particles carry no velocity of their own, so the continuity sum of a
particle sliding along the wall misses part of the divergence the wall's side of its kernel would
carry. A plane of symmetry modeled as a wall would carry the same error onto a section measured on
it, so the model has none: a symmetric case runs in the full tank.

Run on the GPU backend: with the height-field bed the device enforces the boundary, so the state
stays on the device between the trackers' readings. Gravity is -z, x is the direction of travel.

References:
-----------
Guyenne, P. & Grilli, S.T. (2006) -- Numerical study of three-dimensional overturning waves in
    shallow water, J. Fluid Mech. 547, 361-388.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

The scheme is `BreakingWave`'s, whose 2D validation status that module states. In 3D the model is
compared with the sloping ridge of Guyenne and Grilli (2006), with the criteria of
tests/testValidationRidgeOverturn.py, run at 32 particles per depth in the full tank outside the
test suite (about 5 GPU hours a ridge). The breaking point passes for all five ridges, k = 0.10
to 0.50: time +3.2 to +4.8 percent (limit 5), crest position +0.35 to +0.47 h0 (limit 0.5), crest
height -0.6 to +1.5 percent (limit 5). The crest arrives on time, within 0.14 h0 and 2 percent of
the reference at its breaking time, and the face turns vertical 0.25 to 0.41 sqrt(h0/g) late, so
time and position sit near the edges of their bands. The velocity at k = 0.25 fails: the largest
horizontal speed at z = -0.2 h0 lies 4.1 percent above 0.651 sqrt(g h0) (limit 10), and the
largest spanwise velocity is 0.125 of the largest streamwise one against 0.085 (+47 percent, limit
30). No run reaches the jet's touchdown before the face runs into water too shallow to read.
tests/testBreakingWave3D.py checks the build: over a flat bed the start is the 2D exact wave in
every lattice row, with the 2D rule's boundary pressures; a section reads its own two rows; and
after a short run on the GPU the flow mid-depth under the crest lies within 5 percent of the exact
wave's (2 percent at 8 and 16 particles per depth).

Sean Bowman [11/10/2026]

'''

# Standard library imports
from __future__ import annotations

import math
import time as timeModule
from dataclasses import dataclass

# Third-party imports
import numpy as np
from scipy.spatial import cKDTree

# Local imports
from pysurf import constants, waveTheory
from pysurf.bathymetry import HeightField
from pysurf.errors import InvalidInputError, NumericalInstabilityError
from pysurf.sphBoundaries import HeightFieldBoundary
from pysurf.sphParticles import ParticleSystem
from pysurf.sphSolver import SimulationConfig, WcsphSolver, taitDensity, taitEquationOfState
from pysurf.waveBreaking import BreakingTracker
from pysurf.waveMakers import exactSolitaryBoundaryPressure3D, exactSolitaryWaveFluid3D

@dataclass
class SectionFrame:

    '''One section's fluid at an instant: positions [m] and velocities [m/s], shape (N, 3).'''

    time: float
    positions: np.ndarray
    velocities: np.ndarray

class BreakingWave3D:

    '''

    A solitary wave breaking over a height-field bed in 3D SPH.

    Parameters:
    -----------
    field : HeightField
        The bed, its still water level and its bounds, which are the tank's walls
    waveHeight : float
        Solitary wave height H [m]
    crestX : float
        Starting crest position [m]
    depth : float | None
        Still water depth of the starting wave [m]; None for the depth at (startX, startY)
    particlesPerDepth : int
        depth / particle spacing
    sections : tuple | None
        y of each plane a breaking tracker reads [m]; the first is the one `breakingPoint` and
        `stopAt` follow. None for the tank's middle
    sectionRows : int
        Particle rows a section takes: those within rows / 2 spacings of its plane
    searchFromX : float | None
        A crest offshore of this position cannot register as breaking [m]; None for `crestX`
    recordTimes : tuple
        Times at which the whole fluid's state is kept [s]
    frameInterval : float
        Simulated time between section frames [s]
    trackInterval : float
        Simulated time between the trackers' readings [sqrt(depth / g)]
    endTime : float | None
        Simulated time to run to [s]; None for the time the wave takes to reach the end wall
    wallTop : float | None
        Top of the tank's walls [m]; None for still water plus twice the wave height
    smoothingLengthRatio : float
        h / spacing
    alphaViscosity : float
        Artificial viscosity coefficient
    soundSpeedFactor : float
        c0 over the solitary wave's speed
    backend : str
        The solver's backend: 'gpu', or 'numpy' for small tests
    solverOptions : dict | None
        `SimulationConfig` fields that replace the scheme's

    '''

    def __init__(self, field: HeightField, waveHeight: float, crestX: float, depth: float | None = None,
                 particlesPerDepth: int = 32, sections: tuple | None = None,
                 sectionRows: int = 2, searchFromX: float | None = None, recordTimes: tuple = (),
                 frameInterval: float = 0.1, trackInterval: float = 0.01, endTime: float | None = None,
                 wallTop: float | None = None, smoothingLengthRatio: float = 3.0, alphaViscosity: float = 0.01,
                 soundSpeedFactor: float = 10.0, backend: str = 'gpu', solverOptions: dict | None = None) -> None:

        self.field = field
        self.waveHeight = float(waveHeight)                 # [m]
        self.crestX = float(crestX)                         # [m]
        self.depth = float(depth if depth is not None
                           else field.stillWaterLevel - field.elevation(field.startX, field.startY))  # [m]
        self.particlesPerDepth = int(particlesPerDepth)
        middle = 0.5 * (field.startY + field.endY)
        self.sections = tuple(float(y) for y in (sections if sections is not None else (middle,)))
        self.sectionRows = int(sectionRows)
        self.searchFromX = float(searchFromX if searchFromX is not None else crestX)    # [m]
        self.recordTimes = tuple(sorted(float(t) for t in recordTimes))                # [s]
        self.frameInterval = float(frameInterval)           # [s]
        self.trackInterval = float(trackInterval)           # [sqrt(d / g)]
        self.endTime = endTime                              # [s]
        self.wallTop = float(wallTop if wallTop is not None else field.stillWaterLevel + 2.0 * waveHeight)  # [m]
        self.smoothingLengthRatio = float(smoothingLengthRatio)
        self.alphaViscosity = float(alphaViscosity)
        self.soundSpeedFactor = float(soundSpeedFactor)
        self.backend = backend
        self.solverOptions = dict(solverOptions or {})
        self._validateInputs()

        self.boundary: HeightFieldBoundary | None = None
        self.particles: ParticleSystem | None = None
        self.solver: WcsphSolver | None = None
        self.trackers: dict[float, BreakingTracker] = {}
        self.sectionFrames: dict[float, list[SectionFrame]] = {}
        self.snapshots: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        self.wallClockSeconds = 0.0

    # -- Private Helper Methods -- #
    def _validateInputs(self) -> None:

        '''

        Raises:
        -------
        InvalidInputError : If the wave, the resolution or a section does not fit the tank.

        '''

        field = self.field
        if not (0.0 < self.waveHeight < self.depth):
            raise InvalidInputError('The wave height must be positive and below the depth.', parameterName = 'waveHeight',
                                    value = self.waveHeight)
        if not (field.startX < self.crestX < field.endX):
            raise InvalidInputError('The crest must start inside the tank.', parameterName = 'crestX', value = self.crestX)
        if self.particlesPerDepth < 2 or self.sectionRows < 1:
            raise InvalidInputError('Particles per depth must be at least 2 and section rows at least 1.',
                                    parameterName = 'particlesPerDepth', value = self.particlesPerDepth)
        for y in self.sections:
            if not (field.startY - 1e-9 <= y <= field.endY + 1e-9):
                raise InvalidInputError('Every section must lie inside the tank.', parameterName = 'sections', value = y)

    def _sectionMask(self, y: float) -> np.ndarray:

        '''The fluid particles a section takes, as a mask over the fluid.'''

        p = self.particles
        across = p.positions[p.isFluid, 1]

        return np.abs(across - y) < 0.5 * self.sectionRows * self.spacing

    def _readSections(self) -> None:

        '''Feed each section's fluid, in its (x, z) plane, to its tracker.'''

        p, s = self.particles, self.spacing
        fluid = p.positions[p.isFluid]
        for y, tracker in self.trackers.items():
            members = fluid[self._sectionMask(y)]
            tracker.update(self.solver.time, members[:, [0, 2]], np.full(len(members), s**2 / self.sectionRows))

    def _recordSections(self) -> None:

        '''Keep each section's fluid at this instant.'''

        p = self.particles
        positions, velocities = p.positions[p.isFluid], p.velocities[p.isFluid]
        for y in self.sections:
            mask = self._sectionMask(y)
            self.sectionFrames[y].append(SectionFrame(self.solver.time, positions[mask].copy(), velocities[mask].copy()))

    # -- Public Methods -- #
    @property
    def spacing(self) -> float:
        '''Particle spacing [m].'''
        return self.depth / self.particlesPerDepth

    @property
    def boundaryLayers(self) -> int:
        '''Boundary particle layers: enough to fill the kernel's support, 2h, and never fewer than 3.'''
        return max(3, math.ceil(2.0 * self.smoothingLengthRatio - 1e-9))

    @property
    def smoothingLength(self) -> float:
        '''h [m].'''
        return self.smoothingLengthRatio * self.spacing

    @property
    def timeScale(self) -> float:
        '''sqrt(depth / g) [s].'''
        return math.sqrt(self.depth / constants.gravity)

    @property
    def tracker(self) -> BreakingTracker | None:
        '''The first section's tracker.'''
        return self.trackers.get(self.sections[0])

    @property
    def breakingPoint(self):
        '''Where the wave broke on the first section, or None.'''
        return self.tracker.breakingPoint if self.tracker is not None else None

    def build(self) -> None:

        '''Lay out the tank, the boundary, the starting wave, the solver and the trackers.'''

        field, d, s, H = self.field, self.depth, self.spacing, self.waveHeight
        rho, level, layers = constants.freshwaterDensity, field.stillWaterLevel, self.boundaryLayers

        self.boundary = HeightFieldBoundary(field, s, nLayers = layers, wallTop = self.wallTop)
        floor = self.boundary.latticeFloor
        fluidPositions, fluidVelocities, fluidPressures = exactSolitaryWaveFluid3D(field, H, d, self.crestX, s, floor)
        self.particles = ParticleSystem.assemble(fluidPositions, self.boundary.positions(), s, rho, fluidVelocities)

        # `BreakingWave`'s deltaSph scheme (see its build for the reasons behind each setting)
        schemeOptions = dict(kernelType = 'wendlandC2', densityFilterInterval = 0, densityDiffusion = 'hydrostatic',
                             xsphEpsilon = 0.0, wallPressure = 'extrapolated', continuityCorrection = True,
                             densityFloor = 0.9, viscosityLimiter = True)
        schemeOptions.update(self.solverOptions)
        soundSpeed = self.soundSpeedFactor * waveTheory.solitaryWaveSpeed(H, d)
        margin = (layers + 2.0) * s
        travel = (field.endX - self.crestX) / waveTheory.solitaryWaveSpeed(H, d)
        config = SimulationConfig(
            domainMin       = np.array([field.startX - margin, field.startY - margin, floor - margin]),
            domainMax       = np.array([field.endX + margin, field.endY + margin, self.wallTop]),
            particleSpacing = s,
            smoothingLengthRatio = self.smoothingLengthRatio,
            referenceDensity = rho,
            gravity         = np.array([0.0, 0.0, -constants.gravity]),
            endTime         = self.endTime if self.endTime is not None else travel,
            maxTimeStep     = 1.0,
            outputInterval  = self.frameInterval,
            dimensions      = 3,
            speedOfSound    = soundSpeed,
            alphaViscosity  = self.alphaViscosity,
            initialSurface  = lambda x: level + waveTheory.solitaryWaveProfile(x, H, d, self.crestX),
            backend         = self.backend,
            **schemeOptions,
        )
        self.solver = WcsphSolver(config, boundary = self.boundary)
        self.solver.initialize(self.particles)

        # The exact wave's pressure, carried by each particle's density
        p = self.particles
        wallPressures = exactSolitaryBoundaryPressure3D(field, p.positions[~p.isFluid], H, d, self.crestX, s,
                                                        constants.gravity)
        p.densities[p.isFluid] = taitDensity(rho * fluidPressures, rho, soundSpeed)
        p.densities[~p.isFluid] = taitDensity(rho * wallPressures, rho, soundSpeed)
        p.pressures = taitEquationOfState(p.densities, rho, soundSpeed)

        self.trackers = {y: BreakingTracker(field.section(y), s, faceReach = d, searchFromX = self.searchFromX,
                                            faceMeasure = 'contour') for y in self.sections}
        self.sectionFrames = {y: [] for y in self.sections}
        self.snapshots = {}
        self._recordSections()
        if any(abs(t) < 1e-12 for t in self.recordTimes):
            self.snapshots[0.0] = (p.positions[p.isFluid].copy(), p.velocities[p.isFluid].copy())

    def run(self, endTime: float | None = None, progress: bool = True, stopAt: str = 'end') -> None:

        '''

        Step the simulation to `endTime`, reading the sections and keeping the snapshots.

        The trackers read the sections every `trackInterval` sqrt(depth / g); the solver advances
        between those instants, and the record times, in one call, so on the GPU backend the state
        stays on the device in between.

        Parameters:
        -----------
        endTime : float | None
            Simulated time to stop at [s]; the configured end time by default
        progress : bool
            Print a line roughly every tenth of the run
        stopAt : str
            'end', or 'breaking' to stop once the first section's breaking point is final
            (`BreakingTracker.breakingFinal`: the jet has touched down)

        Raises:
        -------
        NumericalInstabilityError : If the time step collapses or a position stops being finite.

        '''

        if stopAt not in ('end', 'breaking'):
            raise ValueError(f"stopAt '{stopAt}' is not 'end' or 'breaking'.")
        if self.solver is None:
            self.build()

        solver, tracker = self.solver, self.tracker
        stopTime = endTime if endTime is not None else solver.config.endTime
        trackStep = self.trackInterval * self.timeScale
        nextTrack = solver.time + trackStep
        nextFrame = solver.time + self.frameInterval
        pending = [t for t in self.recordTimes if t > solver.time + 1e-12]
        nextReport = solver.time + stopTime / 10.0
        started = timeModule.perf_counter()

        if progress:
            print(f'BreakingWave3D: {self.particles.nFluid} fluid and {self.particles.nBoundary} boundary particles, '
                  f'running to {stopTime:.3f} s')

        # A hundred times the steps an interval takes at the acoustic time step: more means the step
        # has collapsed, which a diverging run does before its positions stop being finite
        config = solver.config
        stepLimit = max(1000, int(100.0 * min(trackStep, self.frameInterval) * solver.speedOfSound
                                  / (config.cflNumber * config.smoothingLength)))

        while solver.time < stopTime and not (stopAt == 'breaking' and tracker.breakingFinal):
            target = min([nextTrack, nextFrame, stopTime] + pending[:1])
            state = solver.advance(target, maxSteps = stepLimit)
            if solver.time < target:
                raise NumericalInstabilityError('The time step collapsed.', variableName = 'dt', operation = 'WCSPH step',
                                                context = {'time': solver.time, 'step': state.step, 'dt': state.dt})

            p = self.particles
            if solver.time >= nextTrack:
                self._readSections()
                nextTrack += trackStep
            if solver.time >= nextFrame:
                if not np.all(np.isfinite(p.positions)):
                    raise NumericalInstabilityError('Particle positions stopped being finite.', variableName = 'positions',
                                                    operation = 'WCSPH step', context = {'time': solver.time, 'step': state.step})
                self._recordSections()
                nextFrame += self.frameInterval
            while pending and solver.time >= pending[0] - 1e-12:
                self.snapshots[pending.pop(0)] = (p.positions[p.isFluid].copy(), p.velocities[p.isFluid].copy())
            if progress and solver.time >= nextReport:
                elapsed = timeModule.perf_counter() - started
                broken = ('breaking recorded' if tracker.breakingFinal
                          else 'breaking provisional' if tracker.breakingPoint else 'not broken')
                crest = tracker.crestHeight[-1] if tracker.crestHeight else float('nan')
                print(f'  t = {solver.time:6.3f} s  step {state.step:7d}  dt {state.dt:.2e} s  wall {elapsed:8.1f} s  '
                      f'crest {crest:.3f} m  {broken}', flush = True)
                nextReport += stopTime / 10.0

        self.wallClockSeconds += timeModule.perf_counter() - started

    def velocity(self, points: np.ndarray, time: float) -> np.ndarray:

        '''

        The flow at points at a snapshot's time [m/s], shape (N, 3): the Shepard-normalized Wendland
        average, at the smoothing length, of the fluid within the kernel's support of each point.
        NaN where no fluid lies within reach.

        Raises:
        -------
        KeyError : If no snapshot was kept at `time`.

        '''

        key = min(self.snapshots, key = lambda t: abs(t - time)) if self.snapshots else None
        if key is None or abs(key - time) > 1e-9:
            raise KeyError(f'No snapshot at t = {time} s; recordTimes holds {sorted(self.snapshots)}.')
        positions, velocities = self.snapshots[key]
        h = self.smoothingLength

        points = np.atleast_2d(np.asarray(points, dtype = float))
        tree = cKDTree(positions)
        result = np.full((len(points), 3), np.nan)
        for index, neighbors in enumerate(tree.query_ball_point(points, 2.0 * h)):
            if not neighbors:
                continue
            q = np.linalg.norm(positions[neighbors] - points[index], axis = 1) / h
            weights = np.maximum(1.0 - 0.5 * q, 0.0)**4 * (2.0 * q + 1.0)
            if weights.sum() > 0.0:
                result[index] = weights @ velocities[neighbors] / weights.sum()

        return result

    def summary(self) -> dict:

        '''Each section's breaking point and touchdown, and the run's size and cost.'''

        sections = {}
        for y, tracker in self.trackers.items():
            point = tracker.breakingPoint
            sections[y] = {
                'broke': point is not None,
                'breakingFinal': tracker.breakingFinal,
                'breakingTime': point.time if point is not None else None,
                'breakingCrestX': point.crestX if point is not None else None,
                'breakingCrestHeight': point.crestHeight if point is not None else None,
                'touchedDown': tracker.touchdownSurface is not None,
                'touchdownTime': tracker.touchdownTime if tracker.touchdownSurface is not None else None,
            }

        return {'particlesPerDepth': self.particlesPerDepth, 'fluidParticles': self.particles.nFluid,
                'boundaryParticles': self.particles.nBoundary, 'time': self.solver.time,
                'wallClockSeconds': self.wallClockSeconds, 'sections': sections}
