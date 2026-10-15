
# -- pySurf: Splash-Up -- #

'''

A plunging breaker carried through the jet's impact into the splash-up: the boundary element
overturn of `OverturningWave` handed to weakly compressible SPH in a window around the jet.

Potential flow takes the wave up to the instant the jet lands, and no further: after it the free
surface encloses the tube, the water is no longer one region bounded by one curve, and the impact
makes vorticity. SPH has no such limit. `SplashUp` takes the potential flow's state, at the jet's
impact or at any recorded instant before it, and lets SPH carry it on. Each part is a module:

    pysurf.bemSolver        the flow inside the water: velocity, and pressure from phi_t
    pysurf.sphHandoff       the window, the particles, their starting state, the piston
    pysurf.sphBoundaries    the bed and the walls as boundary particles
    pysurf.sphSolver        the WCSPH time stepping
    pysurf.figures          the frames, and the transient as a GIF

    wave = OverturningWave(depth = 1.0, waveHeight = 0.3, beachSlope = 1 / 35)
    wave.run()
    splash = SplashUp(wave, particlesPerDepth = 64)
    splash.run()
    splash.writeAnimation('splashUp.gif')

The window runs from a few depths behind the crest to a few ahead of where the jet lands. Behind
it, the wave is represented by a piston moving with the potential flow's depth-averaged velocity;
ahead of it, the water is still at rest. The SPH water is single-phase: the air in the tube has no
pressure and does not resist the tube's collapse, as the air in a real tube does.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

The potential flow handed over is checked in tests/testBemSolver.py against exact harmonic fields,
its velocity to 2e-4 k and its pressure to 1.5e-4 rho u^2 at 80 nodes, and the hand-off itself in
tests/testSphHandoff.py: still water starts at rest and hydrostatic, the tube under a jet starts
empty, and a solitary wave's depth-averaged velocity is c eta / (d + eta) to 3 percent.

The continuation is checked against the potential flow it continues, which is validated up to the
jet's impact (`pysurf.OverturningWave`, against Grilli et al. 1997 and the DNS of Feddersen et al.
2024). On the Surf Ranch soliton, handed to SPH 0.10 s (0.31 sqrt(d / g)) before the jet lands, the
SPH water arrives at the impact with the potential flow's surface (`surfaceMismatch`,
tests/testValidationOverturning.py):

    particles per depth     gap from that surface to the nearest particle      water outside it
                            rms           largest                              largest
    32                      0.15          0.42 spacings                        none
    48                      0.19          0.66                                 none
    64                      0.14          0.39                                 0.05 spacings

The lattice alone leaves gaps of 0.2 spacings rms at the hand-off, so the SPH flow adds nothing
measurable over the overturn. That is a cross-check against another model, validated as it is.

The flow after the impact is checked against the energy Mostert and Deike (2020) measured breaking
waves lose in two-phase DNS (`breakingDissipation`, tests/testValidationSplashUp.py). Over the active
breaking time, from the face turning vertical to the kinetic energy falling to the potential, their
plunging case (H0/h0 = 0.3 on a 3 degree beach) loses 0.31 of its energy, and across slopes of 2 to 5
degrees their dissipation rate collapses onto (epsilon / epsilon0) (db / d0) = 0.0281 (hb / d0)^3.152.
The same case here, with the window's offshore wall standing still eight depths behind the crest
and the beach running eight depths past the shoreline, at 48 particles per depth:

    quantity                                        SPH         Mostert and Deike     error
    share of energy lost in active breaking         0.277       0.31                  -11 %
    (epsilon / epsilon0) (db / d0)                   0.00202     0.00217 (fit)         -7 %
    active breaking time, t sqrt(g / d)             8.85        8.35                  +6 %

The criteria, set before the comparison was run, allowed 30 percent. At 64 particles per depth the
share lost is 0.279 and the scaled rate 0.00200, both within 1 percent of the values at 48. The SPH
model's dissipation is its artificial viscosity, not resolved turbulence, so the agreement holds
for this breaker and is not a model of turbulence.

Between 64 and 96 particles per depth the Surf Ranch splash-up's peak height agrees to 0.5 percent
(1.324 and 1.331 H0) and its timing to 10 percent, and the broken water's leading edge advances at
speeds 7 percent apart; at 32 the secondary splash-up is not resolved. Not validated: the shape of
the splash-up itself, for which no profiles of a solitary wave's splash-up were found. The water is
single-phase: the air in the tube neither cushions its collapse nor escapes as it does in a real
breaker, and spray, the share of particles with fewer than four neighbors, is a property of the
resolution rather than of the sea.

References:
-----------
Feddersen, F., Hanson, K., Mostert, W. & Fincham, A. (2024) -- Modelling wind-induced changes to
    overturning wave shape, J. Fluid Mech. 1000, A48.
Mostert, W. & Deike, L. (2020) -- Inertial energy dissipation in shallow-water breaking waves,
    J. Fluid Mech. 890, A12.

Sean Bowman [10/12/2026]

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
from pysurf import constants, figures
from pysurf.bemSolver import FlowField, jetGeometry
from pysurf.errors import InvalidInputError, NumericalInstabilityError
from pysurf.sphBoundaries import BedBoundary
from pysurf.sphHandoff import (PistonDrive, boundaryPressure, handoffSoundSpeed, seedStillWater, seedWater,
                               windowBathymetry)
from pysurf.sphParticles import ParticleSystem
from pysurf.sphSolver import SimulationConfig, WcsphSolver, taitDensity, taitEquationOfState
from pysurf.validation import applyRules, integerRule, numericRule

# Inputs a splash-up must satisfy before anything is built
inputRules = (
    integerRule('particlesPerDepth', 'Particles per depth', minimum = 4, exclusiveMinimum = False),
    numericRule('behindDepths',      'Window behind the crest', minimum = 0.0),
    numericRule('aheadDepths',       'Window ahead of the impact', minimum = 0.0),
    numericRule('alphaViscosity',    'Artificial viscosity', minimum = 0.0, exclusiveMinimum = False),
    numericRule('soundSpeedFactor',  'Speed of sound factor', minimum = 0.0),
)

@dataclass
class SplashFrame:

    '''

    One recorded instant of the SPH water.

    Parameters:
    -----------
    time : float
        Simulated time, on the overturning wave's clock [s]
    positions : np.ndarray
        Fluid particle positions, shape (N, 2) [m]
    speeds : np.ndarray
        Fluid particle speeds [m/s]
    kineticEnergy : float
        Fluid kinetic energy per unit crest width [J/m]
    potentialEnergy : float
        Fluid potential energy over still water filling the window, per unit crest width [J/m]
    pistonX : float
        Position of the offshore wall [m]
    crestHeight : float
        Highest water above still water level, spray left out [m]
    crestX : float
        Where the highest water is [m]
    frontX : float
        The most shoreward water more than a tenth of the wave height above still water, spray left
        out [m]
    sprayShare : float
        Share of the fluid particles in spray [-]

    '''

    time: float
    positions: np.ndarray
    speeds: np.ndarray
    kineticEnergy: float
    pistonX: float
    potentialEnergy: float = 0.0
    crestHeight: float = 0.0
    crestX: float = 0.0
    frontX: float = 0.0
    sprayShare: float = 0.0

def waterMeasures(positions: np.ndarray, spacing: float, stillWaterLevel: float, waveHeight: float,
                  neighbors: int = 4) -> dict:

    '''

    The crest and the leading edge of the water, with spray left out. A particle is spray when
    fewer than `neighbors` others lie within one and a half spacings of it: a droplet, or a strand
    a particle thick, rather than the body of the water or a sheet of it.

    Returns:
    --------
    dict : crestHeight [m], crestX [m], frontX [m] (the most shoreward water more than a tenth of
        the wave height above still water) and sprayShare [-]

    '''

    counts = cKDTree(positions).query_ball_point(positions, 1.5 * spacing, return_length = True) - 1
    water = positions[counts >= neighbors]
    highest = int(np.argmax(water[:, 1]))
    raised = water[water[:, 1] > stillWaterLevel + 0.1 * waveHeight, 0]

    return {
        'crestHeight': float(water[highest, 1] - stillWaterLevel),
        'crestX': float(water[highest, 0]),
        'frontX': float(np.max(raised)) if raised.size else float('nan'),
        'sprayShare': float(np.mean(counts < neighbors)),
    }

class SplashUp:

    '''

    The splash-up of a plunging breaker, from a boundary element overturn handed to SPH.

    Parameters:
    -----------
    wave : OverturningWave
        A run overturning wave
    particlesPerDepth : int
        Particles across the offshore still water depth; sets the resolution
    behindDepths : float
        Window behind the crest, in offshore depths
    aheadDepths : float
        Window ahead of where the jet lands, in offshore depths; the run ends early when the water's
        leading edge comes within half a depth of the wall there
    duration : float | None
        Simulated time to run after the hand-off [s]; 3 sqrt(d / g) by default
    frameInterval : float | None
        Simulated time between recorded frames [s]; a sixtieth of the duration by default
    alphaViscosity : float
        Monaghan artificial viscosity coefficient
    soundSpeedFactor : float
        Speed of sound over the solitary wave speed sqrt(g (d + H))
    frameIndex : int | None
        Recorded frame of the overturning wave to hand off from; the last state, at the jet's
        impact, by default
    piston : bool
        Drive the offshore wall with the potential flow behind it; without, the wall stands still,
        which closes the window's energy budget when it stands far enough behind the wave
    bed : Bathymetry | None
        The bed under the SPH water, the overturning wave's by default; it may run on past the end
        of the potential flow's tank, up a beach, with still water filling it there

    '''

    # -- Constructor -- #
    def __init__(self, wave, particlesPerDepth: int = 64, behindDepths: float = 3.0, aheadDepths: float = 6.0,
                 duration: float | None = None, frameInterval: float | None = None, alphaViscosity: float = 0.02,
                 soundSpeedFactor: float = 10.0, frameIndex: int | None = None, piston: bool = True,
                 bed = None) -> None:

        self.wave = wave
        self.particlesPerDepth = particlesPerDepth
        self.behindDepths = behindDepths
        self.aheadDepths = aheadDepths
        scale = math.sqrt(wave.depth / constants.gravity)
        self.duration = duration if duration is not None else 3.0 * scale       # [s]
        self.frameInterval = frameInterval if frameInterval is not None else self.duration / 60.0   # [s]
        self.alphaViscosity = alphaViscosity
        self.soundSpeedFactor = soundSpeedFactor
        # Spray leaves the water as droplets of a few particles, whose densities would otherwise
        # fall toward zero as they separate
        self.densityFloor = 0.9
        self.frameIndex = frameIndex
        self.usePiston = piston
        self.bed = bed

        self.field: FlowField | None = None
        self.bathymetry = None
        self.boundary: BedBoundary | None = None
        self.particles: ParticleSystem | None = None
        self.solver: WcsphSolver | None = None
        self.piston: PistonDrive | None = None
        self.handoffTime = 0.0                      # [s]
        self.handoffSurface: np.ndarray | None = None
        self.crestX = 0.0                           # [m], at the hand-off
        self.impactX = 0.0                          # [m], where the jet lands or will
        self.crestSpeed = 0.0                       # [m/s], the wave's speed before it broke
        self.frames: list[SplashFrame] = []
        self.stopReason = ''
        self.wallClockSeconds = 0.0                 # [s]

        self._validateInputs()

    # -- Private Helper Methods -- #
    def _validateInputs(self) -> None:

        '''

        Check the inputs against the rule table, and that the wave has a state to hand off.

        Raises:
        -------
        InvalidInputError : If any input is out of range, or the wave has not run.

        '''

        applyRules(self, inputRules)
        if getattr(self.wave, 'solver', None) is None or self.wave.solver.surface is None:
            raise InvalidInputError('The overturning wave has not been run; there is no state to hand off.',
                                    parameterName = 'wave')
        if self.frameIndex is not None and getattr(self.wave.frames[self.frameIndex], 'potential', None) is None:
            raise InvalidInputError(f'Frame {self.frameIndex} of the overturning wave has no potential recorded.',
                                    parameterName = 'frameIndex', value = self.frameIndex)

    def _handoffState(self) -> tuple[np.ndarray, np.ndarray, float]:

        '''The free surface, its potential and the time the hand-off starts from.'''

        if self.frameIndex is None:
            solver = self.wave.solver
            return solver.surface, solver.potential, solver.time
        frame = self.wave.frames[self.frameIndex]

        return frame.surface, frame.potential, frame.time

    def _crestSpeed(self) -> float:

        '''

        The wave's speed: the crest's mean speed over the last depth-scale time before it broke,
        or the solitary wave speed at the hand-off crest's depth when it has not.

        '''

        wave = self.wave
        times, crestX = np.asarray(wave.tracker.times), np.asarray(wave.tracker.crestX)
        end = wave.breakingPoint.time if wave.breakingPoint is not None else self.handoffTime
        recent = (times > end - math.sqrt(wave.depth / constants.gravity)) & (times <= end)
        if np.count_nonzero(recent) >= 3:
            return float(np.polyfit(times[recent], crestX[recent], 1)[0])
        localDepth = float(wave.bathymetry.depth(self.crestX))

        return math.sqrt(constants.gravity * (localDepth + wave.waveHeight))

    def _recordFrame(self) -> None:

        '''Store the fluid at the current instant.'''

        p, fluid = self.particles, self.particles.isFluid
        positions = p.positions[fluid]
        measures = waterMeasures(positions, self.spacing, self.bathymetry.stillWaterLevel, self.wave.waveHeight)
        potentialEnergy = p.potentialEnergy(self.wave.solver.config.gravity) - self._restPotentialEnergy()
        self.frames.append(SplashFrame(time = self.handoffTime + self.solver.time, positions = positions.copy(),
                                       speeds = np.linalg.norm(p.velocities[fluid], axis = 1),
                                       kineticEnergy = p.kineticEnergy(), pistonX = self.boundary.wallX,
                                       potentialEnergy = potentialEnergy, **measures))

    def _restPotentialEnergy(self) -> float:

        '''

        Potential energy of still water filling the window from the offshore wall to the shoreward
        one, up to still water level, per unit crest width [J/m]: the reference the water's
        potential energy is measured from.

        '''

        bed, level = self.bathymetry, self.bathymetry.stillWaterLevel
        x = np.linspace(self.boundary.wallX, bed.endX, 2001)
        under = np.minimum(bed.elevation(x), level)
        column = 0.5 * (level**2 - under**2)

        return (constants.freshwaterDensity * self.wave.solver.config.gravity
                * float(np.sum(0.5 * (column[1:] + column[:-1]) * np.diff(x))))

    # -- Geometry -- #
    @property
    def depth(self) -> float:
        '''Offshore still water depth [m].'''
        return self.wave.depth

    @property
    def spacing(self) -> float:
        '''Particle spacing [m].'''
        return self.depth / self.particlesPerDepth

    # -- Public Methods -- #
    def build(self) -> None:

        '''

        Hand the potential flow to SPH: the flow field, the window, the particles with their
        velocity and pressure, the boundary with its pressure, the piston and the solver.

        '''

        wave, d, s = self.wave, self.depth, self.spacing
        rho = constants.freshwaterDensity
        surface, potential, self.handoffTime = self._handoffState()
        self.handoffSurface = surface.copy()
        solver = wave.solver
        self.field = FlowField(surface, potential, solver.bed, solver.config.surfaceSpacing,
                               wave.bathymetry.stillWaterLevel, solver.config.gravity, rho)

        self.crestX = float(surface[np.argmax(surface[:, 1]), 0])
        jet = jetGeometry(surface)
        self.impactX = float(jet.impactPoint[0]) if jet is not None else self.crestX + d
        bed = self.bed if self.bed is not None else wave.bathymetry
        tankEnd = wave.bathymetry.endX
        backX = max(self.crestX - self.behindDepths * d, wave.bathymetry.startX)
        frontX = min(self.impactX + self.aheadDepths * d, bed.endX)
        self.bathymetry = windowBathymetry(bed, backX, frontX)

        fluid = seedWater(self.field, self.bathymetry, s)
        velocity = self.field.velocity(fluid)
        pressure = np.maximum(self.field.pressure(fluid, velocity), 0.0)
        if frontX > tankEnd:
            # Past the potential flow's tank the water has not been reached by the wave yet
            still = seedStillWater(self.bathymetry, s, tankEnd)
            fluid = np.vstack([fluid, still])
            velocity = np.vstack([velocity, np.zeros_like(still)])
            pressure = np.concatenate([pressure, rho * solver.config.gravity
                                       * (self.bathymetry.stillWaterLevel - still[:, 1])])
        # Inside the water the speed cannot exceed the largest on its boundary by much; a particle
        # that does sat on the boundary, where the representation of the flow is singular
        limit = 2.0 * float(np.max(np.hypot(*self.field.surfaceVelocity.T)))
        if np.max(np.hypot(*velocity.T)) > limit:
            raise NumericalInstabilityError('A particle started faster than any water on the free surface.',
                                            variableName = 'velocity', operation = 'potential flow hand-off',
                                            context = {'limit': limit, 'speed': float(np.max(np.hypot(*velocity.T)))})

        wallTop = wave.bathymetry.stillWaterLevel + 2.5 * wave.waveHeight
        self.boundary = BedBoundary(self.bathymetry, s, nLayers = 3, wallTop = wallTop, endWallTop = wallTop)
        boundaryPositions = self.boundary.positions()
        pressure = np.concatenate([pressure, boundaryPressure(self.field, self.bathymetry, boundaryPositions, s,
                                                              tankEnd if frontX > tankEnd else None)])

        soundSpeed = handoffSoundSpeed(d, wave.waveHeight, self.soundSpeedFactor)
        self.particles = ParticleSystem.assemble(fluid, boundaryPositions, s, rho, velocity)
        config = SimulationConfig(
            domainMin        = np.array([backX - 4.0 * s, float(np.min(self.bathymetry.z)) - 4.0 * s]),
            domainMax        = np.array([frontX + 4.0 * s, wallTop]),
            particleSpacing  = s,
            referenceDensity = rho,
            gravity          = np.array([0.0, -solver.config.gravity]),
            endTime          = self.duration,
            maxTimeStep      = 1.0,
            outputInterval   = self.frameInterval,
            speedOfSound     = soundSpeed,
            alphaViscosity   = self.alphaViscosity,
            densityFloor     = self.densityFloor,
        )
        self.solver = WcsphSolver(config, boundary = self.boundary)
        self.solver.initialize(self.particles)
        # The potential flow's pressure, carried by each particle's density
        self.particles.densities = taitDensity(pressure, rho, soundSpeed)
        self.particles.pressures = taitEquationOfState(self.particles.densities, rho, soundSpeed)

        self.crestSpeed = self._crestSpeed()
        if self.usePiston:
            self.boundary.attachPiston(self.particles)
            self.piston = PistonDrive(self.field, wave.bathymetry, backX, self.crestSpeed, self.duration)
            self.boundary.movePiston(self.particles, backX, self.piston.velocity(backX, 0.0))

        self.frames = []
        self._recordFrame()

    def run(self, endTime: float | None = None, progress: bool = True) -> None:

        '''

        Step the SPH water from the hand-off, moving the piston and recording frames.

        Parameters:
        -----------
        endTime : float | None
            Simulated time after the hand-off to stop at [s]; the duration by default
        progress : bool
            Print a line roughly every tenth of the run

        Raises:
        -------
        NumericalInstabilityError : If any particle position stops being finite.

        '''

        if self.solver is None:
            self.build()

        solver, boundary, piston = self.solver, self.boundary, self.piston
        stopTime = endTime if endTime is not None else self.duration
        nextFrame = solver.time + self.frameInterval
        nextReport = solver.time + stopTime / 10.0
        started = timeModule.perf_counter()

        if progress:
            print(f'SplashUp: {self.particles.nFluid} fluid and {self.particles.nBoundary} boundary particles, '
                  f'hand-off at t = {self.handoffTime:.3f} s, running {stopTime:.3f} s')

        pistonVelocity = piston.velocity(boundary.wallX, solver.time) if piston is not None else 0.0
        while solver.time < stopTime:
            state = solver.step()
            if piston is not None:
                # The step ran with the piston at its old velocity; move it on, then take the new one
                pistonX = boundary.wallX + pistonVelocity * state.dt
                pistonVelocity = piston.velocity(pistonX, solver.time)
                boundary.movePiston(self.particles, pistonX, pistonVelocity)

            if solver.time >= nextFrame:
                if not np.all(np.isfinite(self.particles.positions)):
                    raise NumericalInstabilityError('Particle positions stopped being finite.',
                                                    variableName = 'positions', operation = 'SPH splash-up',
                                                    context = {'time': solver.time, 'step': state.step})
                self._recordFrame()
                nextFrame += self.frameInterval
                # Past this the water would run up the window's wall, which is not in the sea
                if self.frames[-1].frontX > self.bathymetry.endX - 0.5 * self.depth:
                    self.stopReason = 'the water reached the shoreward wall'
                    break
            if progress and solver.time >= nextReport:
                elapsed = timeModule.perf_counter() - started
                print(f'  t = {self.handoffTime + solver.time:6.3f} s  step {state.step:6d}  dt {state.dt:.2e} s  '
                      f'wall {elapsed:7.1f} s  max speed {state.maxVelocity:.2f} m/s  '
                      f'density error {state.maxDensityError:.3f}')
                nextReport += stopTime / 10.0
        else:
            self.stopReason = 'end time reached'

        self.wallClockSeconds += timeModule.perf_counter() - started

    def surfaceMismatch(self, surface: np.ndarray, region: tuple[float, float] | None = None) -> dict:

        '''

        How far the SPH water's surface lies from a free surface, in particle spacings, at the
        current instant: from each surface node in `region` to the nearest particle, less the half
        spacing a particle sits inside a full surface; and from each particle outside the water
        that surface bounds to the surface. Over the overturn, against the potential flow at the
        same instant, it measures how faithfully SPH carries the potential flow on.

        Parameters:
        -----------
        surface : np.ndarray
            Free-surface nodes from wall to wall, shape (N, 2) [m]
        region : tuple | None
            x-range compared [m]; a depth behind the hand-off crest to a depth past the impact by
            default

        Returns:
        --------
        dict : rms and maximum of each distance over the spacing, and the share of particles in
            the region outside the water

        '''

        s, d = self.spacing, self.depth
        low, high = region if region is not None else (self.crestX - d, self.impactX + d)
        everywhere = self.particles.positions[self.particles.isFluid]
        positions = everywhere[(everywhere[:, 0] > low + s) & (everywhere[:, 0] < high - s)]
        inside = (surface[:, 0] > low) & (surface[:, 0] < high)
        first, last = int(np.argmax(inside)), len(inside) - 1 - int(np.argmax(inside[::-1]))
        nodes = surface[first:last + 1]

        # Every particle is a candidate neighbor, so a node near the region's end finds its own
        missing = np.maximum(cKDTree(everywhere).query(nodes)[0] - 0.5 * s, 0.0) / s

        # The water the surface bounds, closed by verticals at the region's ends, where the surface
        # is one curve, and the bed of the window under it
        ends = [np.interp(low, surface[:first + 1, 0], surface[:first + 1, 1]),
                np.interp(high, surface[last:, 0], surface[last:, 1])]
        bedX = np.linspace(high, low, 50)
        polygon = np.vstack([[low, ends[0]], nodes, [high, ends[1]],
                             np.column_stack([bedX, self.bathymetry.elevation(bedX) - s])])
        from matplotlib.path import Path
        outside = ~Path(polygon).contains_points(positions)
        segments = np.stack([nodes[:-1], nodes[1:]], axis = 1)
        stray = np.zeros(int(np.count_nonzero(outside)))
        if stray.size:
            p = positions[outside]
            a, ab = segments[:, 0], segments[:, 1] - segments[:, 0]
            share = np.clip(np.einsum('ijk,jk->ij', p[:, None, :] - a[None], ab) / np.sum(ab**2, axis = 1), 0.0, 1.0)
            closest = a[None] + share[..., None] * ab[None]
            stray = np.min(np.hypot(*(p[:, None, :] - closest).transpose(2, 0, 1)), axis = 1) / s

        return {
            'missingRms': float(np.sqrt(np.mean(missing**2))),
            'missingMax': float(np.max(missing)),
            'strayRms': float(np.sqrt(np.mean(stray**2))) if stray.size else 0.0,
            'strayMax': float(np.max(stray)) if stray.size else 0.0,
            'strayShare': float(stray.size / len(positions)),
        }

    def energyHistory(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        '''

        Kinetic and potential energy of the whole wave from the moment it broke [J/m]: the potential
        flow's, over its whole tank, up to the hand-off, and after it the potential flow's at the
        hand-off carried on by the change in the SPH water's.

        The SPH window's changes stand for the whole wave's when its offshore wall stands still and
        far enough behind the wave that the water outside the window keeps its energy; the
        lattice's own offset from the potential flow's energy then drops out with the difference.

        Returns:
        --------
        tuple : (time [s], kinetic energy [J/m], potential energy [J/m]), in time order

        Raises:
        -------
        ValueError : If the offshore wall is a piston, which feeds the window energy from outside.

        '''

        if self.usePiston:
            raise ValueError('The energy history needs the offshore wall standing still: build with piston = False.')

        wave, solver = self.wave, self.wave.solver
        point = wave.breakingPoint
        start = point.time if point is not None else self.handoffTime
        rows = []
        for frame in wave.frames:
            if start - 1e-9 <= frame.time < self.handoffTime - 1e-9 and frame.potential is not None:
                field = FlowField(frame.surface, frame.potential, solver.bed, solver.config.surfaceSpacing,
                                  wave.bathymetry.stillWaterLevel, solver.config.gravity, constants.freshwaterDensity)
                rows.append((frame.time, *field.energy()))
        kinetic0, potential0 = self.field.energy()
        first = self.frames[0]
        for frame in self.frames:
            rows.append((frame.time, kinetic0 + frame.kineticEnergy - first.kineticEnergy,
                         potential0 + frame.potentialEnergy - first.potentialEnergy))
        rows = np.array(sorted(rows))

        return rows[:, 0], rows[:, 1], rows[:, 2]

    def breakingDissipation(self) -> dict:

        '''

        The energy lost in active breaking, as Mostert and Deike (2020) measure it: from the moment
        the face turns vertical, t1, to the first moment after it when the kinetic energy has fallen
        to the potential, t2, the mechanical energy falls by dE, and the dissipation rate per unit
        crest width is epsilon = dE / (t2 - t1). Their inertial model scales it with the wave height
        at breaking hb and the still water depth at the vertical face db; across their DNS of solitary
        waves on slopes of 2 to 5 degrees, (epsilon / epsilon0) (db / d0) = 0.0281 (hb / d0)^3.152,
        epsilon0 = rho g^(3/2) d0^(5/2).

        Returns:
        --------
        dict : t1, t2 and their difference [s], the energy at breaking and at t2 [J/m], the share
            lost, epsilon [W/m], the scaled rate and the fit at the same hb and db, and whether t2
            was reached in the run

        '''

        wave = self.wave
        point = wave.breakingPoint
        times, kinetic, potential = self.energyHistory()
        energy = kinetic + potential
        d, g, rho = wave.depth, wave.solver.config.gravity, constants.freshwaterDensity
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
        scaled = rate / (rho * g**1.5 * d**2.5) * (point.faceDepth / d)

        return {
            'breakingTime': point.time,
            'activeEndTime': endTime,
            'activeDuration': endTime - point.time,
            'activeDurationOverScale': (endTime - point.time) / math.sqrt(d / g),
            'breakingEnergy': breakingEnergy,
            'activeEndEnergy': endEnergy,
            'energyShareLost': (breakingEnergy - endEnergy) / breakingEnergy,
            'dissipationRate': rate,
            'scaledDissipation': scaled,
            'scaledDissipationFit': 0.0281 * (point.crestHeight / d)**3.152,
            'breakingHeightOverDepth': point.crestHeight / d,
            'breakingFaceDepthOverDepth': point.faceDepth / d,
            'activeEndReached': reached,
        }

    def summary(self) -> dict:

        '''

        The hand-off and the run in figures.

        Returns:
        --------
        dict : hand-off time, crest and impact positions, window, particle counts and spacing,
            speed of sound, crest speed driving the piston, the piston's travel, kinetic energy at
            the hand-off and at the end, the largest fluid speed, and wall-clock time

        '''

        result = {
            'handoffTime': self.handoffTime,
            'crestX': self.crestX,
            'impactX': self.impactX,
            'particlesPerDepth': self.particlesPerDepth,
            'wallClockSeconds': self.wallClockSeconds,
        }
        if self.solver is not None:
            result.update({
                'windowBackX': self.bathymetry.startX,
                'windowFrontX': self.bathymetry.endX,
                'fluidParticles': self.particles.nFluid,
                'boundaryParticles': self.particles.nBoundary,
                'spacing': self.spacing,
                'soundSpeed': self.solver.speedOfSound,
                'crestSpeed': self.crestSpeed,
                'pistonTravel': self.boundary.wallX - self.bathymetry.startX,
                'endTime': self.handoffTime + self.solver.time,
                'stopReason': self.stopReason,
                'kineticEnergyAtHandoff': self.frames[0].kineticEnergy,
                'kineticEnergyAtEnd': self.frames[-1].kineticEnergy,
                'maxSpeed': float(max(np.max(frame.speeds) for frame in self.frames)),
            })

        return result

    def drawFrame(self, index: int = -1):

        '''A recorded frame as a Matplotlib figure; see `figures.drawSplashFrame`.'''

        return figures.drawSplashFrame(self, index)

    def writeAnimation(self, path: str, fps: int = 12, frameStep: int = 1, dpi: int = 90) -> str:

        '''Write the recorded transient as a GIF; see `figures.writeSplashAnimation`.'''

        return figures.writeSplashAnimation(self, path, fps = fps, dpi = dpi, frameStep = frameStep)
