
# -- pySurf: Peeling Wave -- #

'''

A plunging breaker peeling along a reef whose depth contours run obliquely to its crest, in three
dimensions: one two-dimensional boundary element overturn, handed to SPH plane by plane.

The bed is a two-dimensional profile f(x) sheared across the span, z_b(x, y) = f(x - y tan(alpha)),
alpha the angle between the contours and the incident crest, which runs along y. The incident
solitary wave is straight-crested and travels unchanged at c over the flat offshore bed: the exact
potential flow solitary wave (`OverturningWave(startingWave = 'exact')`) does, and an approximate
one that is still adjusting does not. Translating the whole problem by y along the contours and by
y tan(alpha) / c in time then leaves it unchanged, so the flow in every cross-section y = constant
is the flow in the section y = 0, shifted by y tan(alpha) along x and delayed by y tan(alpha) / c.
Strip theory takes the y = 0 section to be the two-dimensional run (`pysurf.sliceStack`): each
section satisfies Laplace's equation in (x, z) and the stacked flow misses
d2phi/dy2 = tan^2(alpha) (d/dx + d/dt / c)^2 phi, which is zero for any part of the flow that
travels unchanged at c, so the incident wave is exact and the error lives in the breaking region,
of order tan^2(alpha). The spanwise velocity of the stacked flow is v = dphi/dy =
-tan(alpha) (u + phi_t / c).

The SPH lattice is a set of planes y_k = (k + 1/2) s. Plane k is plane 0 delayed by
delta = s tan(alpha) / c, so the jet lands in plane k at t = k delta: the breaker peels along y at
c / tan(alpha). Each plane is handed to SPH at its own impact, so every overturn is the boundary
element model's and SPH carries only what potential flow cannot. Ahead of the newest SPH plane,
`bufferPlanes` planes of particles move with the potential flow at their own phase, carrying its
velocity (v included) and pressure; the SPH water sees them as its neighbor across the peel. A
buffer plane joins the SPH water at its impact, and a new one is seeded behind the buffer.

Every buffer plane carries the same water through the same phases, shifted along x and y. `build`
therefore follows one template plane through them once: lattice points laid over the water
`bufferPlanes` lags before the impact and carried by the potential flow to it, with their velocity
and pressure at every half lag, and the bed's pressure beneath. Each buffer plane takes the
template's state at its phase, interpolated in time between half lags. The boundary element state
is needed only over those last lags: `build` restarts the two-dimensional run from a frame
recorded shortly before its impact and records the surface every half lag to the impact.

Each plane's offshore wall is a piston driven as `SplashUp` drives its own, delayed by the plane's
lag. The side walls at y = 0 and at the far end are free-slip. On a reef that stays oblique the
water at y = 0 continues into older broken water; the wall there stops its spanwise flow, and that
disturbance spreads toward the peel. On a triangular reef, z_b = f(x - |y| tan(alpha)), the wave
reaches the apex at y = 0 first and peels both ways; for flow that stays symmetric the wall is the
symmetry plane, so a run is that reef's half y > 0, and mirroring the frames in y gives the whole.
Strip theory is least accurate at the apex, where the bed's contours turn: it hands the planes
beside the wall a spanwise velocity that the three-dimensional flow would bring to zero there.
Far enough from the wall the flow is steady in a frame moving with the peel, so
every plane's history as a function of the time since its own impact is the same: `planeHistory`
gives each plane's energy against that age, and where the histories collapse the wall has not
reached. A plane's energy is that of the particles it was laid with, followed wherever they move:
binned by position instead, a lattice plane that drifts half a spacing sideways crosses into the
next bin all at once.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

Not validated: no measurement or three-dimensional simulation of a solitary wave breaking on an
oblique reef was found to compare with. What is checked:

    the two-dimensional overturn         tests/testValidationOverturning.py (Grilli et al. 1997,
                                         Feddersen et al. 2024)
    the two-dimensional splash-up        tests/testValidationSplashUp.py (Mostert & Deike 2020)
    strip theory on a curved bed         tests/testValidationRidgeOverturn.py (Guyenne & Grilli
                                         2006): slices reproduce the breaking point and the
                                         in-plane speed where the bed focuses the wave mildly, but
                                         overstate the spanwise velocity there by 70 percent
    the three-dimensional SPH path       tests/testValidationExtrudedSplashUp.py: extruded across a
                                         periodic channel, the 3D water stays planar and repeats
                                         the 2D splash-up

Because strip theory's spanwise velocity is the least certain part of the state handed over,
`spanwiseScale` multiplies it; runs at 1 and 0 bracket how much the result depends on it.

The Surf Ranch reef with its contours at 30 degrees, from the exact solitary wave of H0/h0 = 0.6,
at 32 particles per depth for 1.6 sqrt(h0 / g) (111 planes, 470,000 fluid particles, 68 minutes on
four threads): the incident wave crosses the flat bed at 3.9125 m/s against the exact 3.9144. The
newest half of the planes that reach 0.25, 0.5, 0.8 and 1.2 sqrt(h0 / g) after their impact agree
in energy within 0.6, 0.8, 1.2 and 1.7 percent, with the spanwise velocity or without: the peel is
steady. Without spanwise velocity
their median is the two-dimensional splash-up's within 0.1 percent at every age; with it, within
0.1 percent to 0.8 sqrt(h0 / g) and 1.2 percent above at 1.2. With it the water drifts toward the
broken side by 0.027 h0 by 0.8 sqrt(h0 / g), and the top of the water moves by under 0.01 h0. At
16 particles per depth the plane energies agree with these within 0.7 percent. The wall at y = 0
feeds the oldest planes up to 9.5 percent more energy with the spanwise velocity; without it the
oldest plane loses at most 1.1 percent.
`stripResidual` reports the stacked flow's divergence, d v / d y, against the in-plane divergence:
the part of the strip-theory state that is not incompressible, which the SPH water feels as a
density disturbance when each plane joins it. The water is single-phase.

Sean Bowman [10/31/2026]

'''

# Standard library imports
from __future__ import annotations

import math
import time as timeModule
from dataclasses import dataclass

# Third-party imports
import numpy as np

# Local imports
from pysurf import constants
from pysurf.bemSolver import FlowField, jetGeometry
from pysurf.errors import InvalidInputError, NumericalInstabilityError
from pysurf.OverturningWave import OverturningWave
from pysurf.SplashUp import waterMeasures
from pysurf.sphBoundaries import BedBoundary, ShearedBedBoundary
from pysurf.sphHandoff import PistonDrive, boundaryPressure, handoffSoundSpeed, seedWater, windowBathymetry
from pysurf.sphParticles import ParticleSystem
from pysurf.sphSolver import SimulationConfig, WcsphSolver, taitDensity, taitEquationOfState
from pysurf.validation import applyRules, integerRule, numericRule

# Inputs a peeling wave must satisfy before anything is built
inputRules = (
    numericRule('obliqueAngle',      'Contour angle to the crest', units = 'deg', minimum = 0.0, maximum = 60.0),
    integerRule('particlesPerDepth', 'Particles per depth', minimum = 4, exclusiveMinimum = False),
    integerRule('bufferPlanes',      'Buffer planes', minimum = 3, exclusiveMinimum = False),
    numericRule('behindDepths',      'Window behind the crest', minimum = 0.0),
    numericRule('aheadDepths',       'Window ahead of the impact', minimum = 0.0),
    numericRule('spanwiseScale',     'Spanwise velocity scale', minimum = 0.0, exclusiveMinimum = False),
)

@dataclass
class PeelFrame:

    '''

    One recorded instant of the SPH water.

    Parameters:
    -----------
    time : float
        Time since the jet landed in plane 0 [s]
    positions : np.ndarray
        SPH fluid particle positions, shape (N, 3), single precision [m]
    speeds : np.ndarray
        SPH fluid particle speeds, single precision [m/s]
    activePlanes : int
        Planes handed to SPH so far
    planeKinetic, planePotential : np.ndarray
        Kinetic energy, and potential energy over still water filling the plane's window, of the
        water each plane was laid with, wherever it has since moved, per unit crest width [J/m]
    crestHeight : float
        Highest water above still water level, spray left out [m]

    '''

    time: float
    positions: np.ndarray
    speeds: np.ndarray
    activePlanes: int
    planeKinetic: np.ndarray
    planePotential: np.ndarray
    crestHeight: float = 0.0

class PeelingWave:

    '''

    A plunging breaker peeling along an oblique reef, in three-dimensional SPH.

    Parameters:
    -----------
    wave : OverturningWave
        The two-dimensional overturn on the reef's profile, started from the exact solitary wave
        and run to the jet's impact
    obliqueAngle : float
        Angle between the reef's depth contours and the incident crest [deg]
    particlesPerDepth : int
        Particles across the offshore still water depth
    duration : float | None
        Simulated time after the jet lands in plane 0 [s]; 1.6 sqrt(d / g) by default, the time
        the two-dimensional splash-up takes to reach its peak
    frameInterval : float | None
        Simulated time between recorded frames [s]; a fortieth of the duration by default
    behindDepths, aheadDepths : float
        Each plane's window behind the crest and ahead of where the jet lands, in offshore depths
    bufferPlanes : int
        Planes moving with the potential flow ahead of the newest SPH plane; at least the kernel
        support across, 2.6 spacings
    spanwiseScale : float
        Factor on strip theory's spanwise velocity in the state handed to SPH: 1 as strip theory
        gives it, 0 for none
    alphaViscosity : float
        Monaghan artificial viscosity coefficient
    soundSpeedFactor : float
        Speed of sound over the solitary wave speed sqrt(g (d + H))
    backend : str
        The SPH solver's backend, 'compiled', 'gpu' or 'numpy'

    '''

    # -- Constructor -- #
    def __init__(self, wave: OverturningWave, obliqueAngle: float = 30.0, particlesPerDepth: int = 32,
                 duration: float | None = None, frameInterval: float | None = None, behindDepths: float = 3.0,
                 aheadDepths: float = 4.0, bufferPlanes: int = 3, spanwiseScale: float = 1.0,
                 alphaViscosity: float = 0.02, soundSpeedFactor: float = 10.0, backend: str = 'compiled') -> None:

        self.wave = wave
        self.obliqueAngle = obliqueAngle
        self.particlesPerDepth = particlesPerDepth
        scale = math.sqrt(wave.depth / constants.gravity)
        self.duration = duration if duration is not None else 1.6 * scale                   # [s]
        self.frameInterval = frameInterval if frameInterval is not None else self.duration / 40.0     # [s]
        self.behindDepths = behindDepths
        self.aheadDepths = aheadDepths
        self.bufferPlanes = bufferPlanes
        self.spanwiseScale = spanwiseScale
        self.alphaViscosity = alphaViscosity
        self.soundSpeedFactor = soundSpeedFactor
        self.backend = backend
        self.densityFloor = 0.9
        self.soundSpeed = handoffSoundSpeed(wave.depth, wave.waveHeight, soundSpeedFactor)    # [m/s]

        self.incidentSpeed = 0.0                    # [m/s], c
        self.restart: OverturningWave | None = None
        self.impactTime = 0.0                       # [s], when the jet lands in plane 0, on the wave's clock
        self.fields: list[FlowField] = []           # the flow at phases impact - m delta / 2
        self.window = None
        self.boundary: ShearedBedBoundary | None = None
        self.particles: ParticleSystem | None = None
        self.solver: WcsphSolver | None = None
        self.piston: PistonDrive | None = None
        self.planes = 0
        self.activePlanes = 0
        self.frames: list[PeelFrame] = []
        self.stopReason = ''
        self.wallClockSeconds = 0.0                 # [s]
        self._template: dict = {}
        self._buffers: dict[int, np.ndarray] = {}   # buffer plane -> its particle indices
        self._planeOf = np.zeros(0, dtype = np.intp)  # the plane each particle was laid in; -1 for the boundary
        self._pistonVelocity = np.zeros(0)
        self._profile: np.ndarray | None = None
        self._mass = 0.0
        self._halfSteps = 0

        self._validateInputs()

    # -- Private Helper Methods -- #
    def _validateInputs(self) -> None:

        '''Check the inputs against the rule table, and that the wave has landed its jet.'''

        applyRules(self, inputRules)
        if getattr(self.wave, 'solver', None) is None or not self.wave.touchedDown:
            raise InvalidInputError('The overturning wave must be run until its jet lands.', parameterName = 'wave')

    def _measureIncidentSpeed(self) -> float:

        '''The wave's speed over the flat offshore bed, from its crest track before it reaches the slope [m/s].'''

        wave = self.wave
        times, crestX = np.asarray(wave.tracker.times), np.asarray(wave.tracker.crestX)
        flat = (crestX > wave.crestX + 0.5 * wave.depth) & (crestX < wave.toeX - wave.halfLength)
        if np.count_nonzero(flat) < 5:
            raise InvalidInputError('The wave runs too short a flat bed to measure its speed before the slope.',
                                    parameterName = 'wave')

        return float(np.polyfit(times[flat], crestX[flat], 1)[0])

    def _crestSpeed(self) -> float:

        '''The wave's speed in the last depth-scale time before it broke, as `SplashUp` takes it [m/s].'''

        wave = self.wave
        times, crestX = np.asarray(wave.tracker.times), np.asarray(wave.tracker.crestX)
        end = wave.breakingPoint.time
        recent = (times > end - math.sqrt(wave.depth / constants.gravity)) & (times <= end)

        return float(np.polyfit(times[recent], crestX[recent], 1)[0])

    def _restartToImpact(self) -> None:

        '''

        Rerun the overturn from a frame shortly before the jet lands, recording the surface every
        half plane lag, and keep the flow at the last recorded instant before the jet lands and at
        the 2 `bufferPlanes` half lags before it.

        '''

        wave, half = self.wave, 0.5 * self.planeLag
        needed = (2 * self.bufferPlanes + 4) * half
        frames = [frame for frame in wave.frames
                  if frame.potential is not None and frame.time <= wave.solver.time - needed]
        if not frames:
            raise InvalidInputError('The overturning wave recorded no frame early enough to restart from.',
                                    parameterName = 'wave')
        start = frames[-1]
        span = wave.solver.time - start.time + 4.0 * half
        record = np.arange(int(span / half) + 1) * half
        restart = OverturningWave(depth = wave.depth, waveHeight = wave.waveHeight, bathymetry = wave.bathymetry,
                                  crestSpacingRatio = wave.crestSpacingRatio,
                                  surfaceSpacingRatio = wave.surfaceSpacingRatio,
                                  crestWindowRatio = wave.crestWindowRatio,
                                  initialState = (start.surface, start.potential), crestRange = wave.crestRange,
                                  frameInterval = 10.0 * span, recordTimes = record, endTime = span)
        restart.run(progress = False)
        if not restart.touchedDown:
            raise NumericalInstabilityError('The restarted overturn did not land its jet.',
                                            variableName = 'touchedDown', operation = 'peeling wave restart',
                                            context = {'stopReason': restart.stopReason})
        recorded = [time for time in record if restart.frameAt(time) is not None]
        last = len(recorded) - 1
        if last < 2 * self.bufferPlanes:
            raise NumericalInstabilityError('The restart recorded too few instants before the jet landed.',
                                            variableName = 'recorded', operation = 'peeling wave restart',
                                            context = {'recorded': len(recorded)})
        solver = restart.solver
        self.restart = restart
        self.impactTime = start.time + recorded[last]
        self.fields = []
        for m in range(2 * self.bufferPlanes + 1):
            frame = restart.frameAt(recorded[last - m])
            self.fields.append(FlowField(frame.surface, frame.potential, solver.bed, solver.config.surfaceSpacing,
                                         solver.stillWaterLevel, solver.config.gravity, constants.freshwaterDensity))

    def _evaluate(self, phase: int, local: np.ndarray, velocity: np.ndarray | None = None,
                  density: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:

        '''

        The stacked flow at points in the two-dimensional run's (x, z) at a recorded phase: velocity
        (u, v, w) and the density that carries its pressure. Points outside that phase's water keep
        the values passed in, or rest and the reference density.

        '''

        flow = self.fields[phase]
        result = np.zeros((len(local), 3)) if velocity is None else velocity.copy()
        weight = np.full(len(local), constants.freshwaterDensity) if density is None else density.copy()
        wet = flow.inside(local)
        if np.any(wet):
            planar = flow.velocity(local[wet])
            rate = flow.potentialRate(local[wet])
            result[wet, 0] = planar[:, 0]
            result[wet, 1] = -self.spanwiseScale * self.shear * (planar[:, 0] + rate / self.incidentSpeed)
            result[wet, 2] = planar[:, 1]
            pressure = np.maximum(flow.pressure(local[wet], planar), 0.0)
            weight[wet] = taitDensity(pressure, constants.freshwaterDensity, self.soundSpeed)

        return result, weight

    def _buildTemplate(self) -> None:

        '''

        Follow one plane's water from `bufferPlanes` lags before the impact to the impact: lattice
        points over the water then, carried by the potential flow with a trapezoidal step each half
        lag, and their velocity and density at every half lag, with the bed's density beneath.

        '''

        top, half = 2 * self.bufferPlanes, 0.5 * self.planeLag
        positions = seedWater(self.fields[top], self.window, self.spacing)
        velocity, density = self._evaluate(top, positions)
        limit = 2.0 * float(np.max(np.hypot(*self.fields[0].surfaceVelocity.T)))
        template = {'positions': {top: positions}, 'velocities': {top: velocity}, 'densities': {top: density},
                    'bed': {}}
        for phase in range(top, 0, -1):
            predicted = positions + half * velocity[:, [0, 2]]
            ahead, _ = self._evaluate(phase - 1, predicted, velocity)
            positions = positions + 0.25 * self.planeLag * (velocity[:, [0, 2]] + ahead[:, [0, 2]])
            velocity, density = self._evaluate(phase - 1, positions, velocity, density)
            template['positions'][phase - 1] = positions
            template['velocities'][phase - 1] = velocity
            template['densities'][phase - 1] = density
        if np.max(np.linalg.norm(template['velocities'][0], axis = 1)) > limit:
            raise NumericalInstabilityError('A particle reached the impact faster than any water on the free surface.',
                                            variableName = 'velocity', operation = 'peeling wave hand-off',
                                            context = {'limit': limit})
        for phase in range(top + 1):
            pressure = boundaryPressure(self.fields[phase], self.window, self._profile, self.spacing)
            template['bed'][phase] = taitDensity(pressure, constants.freshwaterDensity, self.soundSpeed)
        self._template = template

    def _templateState(self, plane: int, phase: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        '''

        The template's positions in a plane, its velocity and its density at a phase counted in
        half lags before the impact, interpolated linearly between the recorded half lags.

        '''

        template = self._template
        upper = min(int(math.ceil(phase)), 2 * self.bufferPlanes)
        lower = max(upper - 1, 0)
        share = float(np.clip(upper - phase, 0.0, 1.0)) if upper > lower else 0.0
        local = (1.0 - share) * template['positions'][upper] + share * template['positions'][lower]
        velocity = (1.0 - share) * template['velocities'][upper] + share * template['velocities'][lower]
        density = (1.0 - share) * template['densities'][upper] + share * template['densities'][lower]
        y = self.planeY(plane)
        positions = np.column_stack([local[:, 0] + self.shear * y, np.full(len(local), y), local[:, 1]])

        return positions, velocity, density

    def _rowBoundary(self, row: int) -> np.ndarray:

        '''Indices of a row's bed and wall particles in the particle system.'''

        size = len(self._profile)
        return self.boundary._start + row * size + np.arange(size)

    def _placeBuffer(self, plane: int, phase: float) -> None:

        '''Put a buffer plane's particles, and the bed beneath it, in the template's state at a phase.'''

        positions, velocity, density = self._templateState(plane, phase)
        members = self._buffers[plane]
        p = self.particles
        p.positions[members] = positions
        p.velocities[members] = velocity
        p.densities[members] = density
        p.densities[self._rowBoundary(plane)] = self._template['bed'][min(int(round(phase)), 2 * self.bufferPlanes)]

    def _seedBuffer(self, plane: int) -> None:

        '''Lay a plane's particles as a buffer, `bufferPlanes` lags before its impact.'''

        phase = 2 * self.bufferPlanes
        positions, velocity, density = self._templateState(plane, phase)
        self._buffers[plane] = self.particles.append(positions, velocity, density, self._mass, isFluid = False)
        self._planeOf = np.concatenate([self._planeOf, np.full(len(positions), plane)])
        self.particles.densities[self._rowBoundary(plane)] = self._template['bed'][phase]

    def _activate(self, plane: int) -> None:

        '''Hand a buffer plane to SPH at its impact and set its offshore wall moving.'''

        self._placeBuffer(plane, 0.0)
        members = self._buffers.pop(plane)
        self.particles.isFluid[members] = True
        self.activePlanes = plane + 1
        wallX = self.boundary.wallX[plane]
        self._pistonVelocity[plane] = self.piston.velocity(wallX - self.shear * self.planeY(plane), 0.0)
        self.boundary.movePiston(self.particles, plane, wallX, self._pistonVelocity[plane])

    def _setSideWallPressure(self) -> None:

        '''Give the side wall at y = 0 the pressure of plane 0's water beside it.'''

        layers, rows, size = self.boundary.nLayers, self.boundary.rows, len(self._profile)
        sides = self.boundary.positions()[rows * size:]
        count = layers * (len(sides) // (2 * layers))
        local = np.column_stack([sides[:count, 0] - self.shear * sides[:count, 1], sides[:count, 2]])
        pressure = boundaryPressure(self.fields[0], self.window, local, self.spacing)
        members = self.boundary._start + rows * size + np.arange(count)
        self.particles.densities[members] = taitDensity(pressure, constants.freshwaterDensity, self.soundSpeed)

    def _restPotentialEnergy(self, row: int) -> float:

        '''Potential energy of still water filling a row's window, per unit crest width [J/m].'''

        bed, level = self.window, self.window.stillWaterLevel
        x = np.linspace(self.boundary.wallX[row] - self.shear * self.planeY(row), bed.endX, 2001)
        under = np.minimum(bed.elevation(x), level)
        column = 0.5 * (level**2 - under**2)

        return constants.freshwaterDensity * constants.gravity * float(np.sum(0.5 * (column[1:] + column[:-1]) * np.diff(x)))

    def _recordFrame(self) -> None:

        '''Store the SPH water at the current instant.'''

        p, s, g = self.particles, self.spacing, constants.gravity
        fluid = p.isFluid
        positions, velocities, masses = p.positions[fluid], p.velocities[fluid], p.masses[fluid]
        planes = self._planeOf[fluid]
        kinetic = np.bincount(planes, weights = 0.5 * masses * np.sum(velocities**2, axis = 1), minlength = self.planes) / s
        potential = np.bincount(planes, weights = masses * g * positions[:, 2], minlength = self.planes) / s
        for plane in range(self.activePlanes):
            potential[plane] -= self._restPotentialEnergy(plane)
        potential[self.activePlanes:] = 0.0
        # The buffer is water too: counted as neighbors, a plane next to it is not taken for spray
        water = np.vstack([positions] + [p.positions[members] for members in self._buffers.values()])
        crest = waterMeasures(water, s, self.window.stillWaterLevel, self.wave.waveHeight)['crestHeight']
        self.frames.append(PeelFrame(time = self.solver.time, positions = positions.astype(np.float32),
                                     speeds = np.linalg.norm(velocities, axis = 1).astype(np.float32),
                                     activePlanes = self.activePlanes, planeKinetic = kinetic, planePotential = potential,
                                     crestHeight = crest))

    # -- Geometry -- #
    @property
    def depth(self) -> float:
        '''Offshore still water depth [m].'''
        return self.wave.depth

    @property
    def spacing(self) -> float:
        '''Particle spacing [m].'''
        return self.depth / self.particlesPerDepth

    @property
    def shear(self) -> float:
        '''tan(alpha): how far the contours advance along x per unit y [-].'''
        return math.tan(math.radians(self.obliqueAngle))

    @property
    def planeLag(self) -> float:
        '''delta = s tan(alpha) / c, the delay of each plane behind the one before it [s].'''
        speed = self.incidentSpeed if self.incidentSpeed > 0.0 else self._measureIncidentSpeed()
        return self.spacing * self.shear / speed

    @property
    def peelSpeed(self) -> float:
        '''c / tan(alpha), the speed the jet's landing moves along the crest [m/s].'''
        return self.incidentSpeed / self.shear

    def planeY(self, plane: int) -> float:
        '''Spanwise position of a lattice plane [m].'''
        return (plane + 0.5) * self.spacing

    # -- Public Methods -- #
    def build(self) -> None:

        '''

        Restart the overturn to its impact, lay out the channel and its bed, follow the template
        plane to the impact, hand plane 0 to SPH and seed the buffer planes ahead of it.

        '''

        wave, d, s = self.wave, self.depth, self.spacing
        rho = constants.freshwaterDensity
        self.incidentSpeed = self._measureIncidentSpeed()
        self._restartToImpact()
        impact = self.fields[0]

        crestX = float(impact.surface[np.argmax(impact.surface[:, 1]), 0])
        jet = jetGeometry(impact.surface)
        impactX = float(jet.impactPoint[0]) if jet is not None else crestX + d
        backX = max(crestX - self.behindDepths * d, wave.bathymetry.startX)
        frontX = min(impactX + self.aheadDepths * d, wave.bathymetry.endX)
        self.window = windowBathymetry(wave.bathymetry, backX, frontX)

        self.planes = int(math.ceil(self.duration / self.planeLag)) + self.bufferPlanes + 1
        width = self.planes * s
        wallTop = wave.bathymetry.stillWaterLevel + 2.5 * wave.waveHeight
        self.boundary = ShearedBedBoundary(self.window, s, width, self.shear, nLayers = 3, wallTop = wallTop,
                                           endWallTop = wallTop)
        boundaryPositions = self.boundary.positions()
        self._profile = BedBoundary(self.window, s, nLayers = 3, wallTop = wallTop,
                                    endWallTop = wallTop)._profilePositions()
        self._mass = rho * s**3
        self._buildTemplate()

        positions, velocity, density = self._templateState(0, 0.0)
        self.particles = ParticleSystem.assemble(positions, boundaryPositions, s, rho, velocity)
        self.boundary.attach(len(positions))
        self._planeOf = np.concatenate([np.zeros(len(positions), dtype = np.intp),
                                        np.full(len(boundaryPositions), -1, dtype = np.intp)])
        bottom = float(np.min(self.window.z)) - 4.0 * s
        config = SimulationConfig(
            domainMin        = np.array([backX - 4.0 * s, -4.0 * s, bottom]),
            domainMax        = np.array([frontX + self.shear * width + 4.0 * s, width + 4.0 * s, wallTop]),
            particleSpacing  = s,
            referenceDensity = rho,
            gravity          = np.array([0.0, 0.0, -constants.gravity]),
            endTime          = self.duration,
            maxTimeStep      = 1.0,
            outputInterval   = self.frameInterval,
            dimensions       = 3,
            speedOfSound     = self.soundSpeed,
            alphaViscosity   = self.alphaViscosity,
            densityFloor     = self.densityFloor,
            backend          = self.backend,
        )
        self.solver = WcsphSolver(config, boundary = self.boundary)
        self.solver.initialize(self.particles)
        self.particles.densities[:] = rho
        self.particles.densities[:len(positions)] = density
        self.particles.densities[self._rowBoundary(0)] = self._template['bed'][0]
        self._buffers = {}
        for plane in range(1, self.bufferPlanes + 1):
            self._seedBuffer(plane)
            self._placeBuffer(plane, 2.0 * plane)
        self._setSideWallPressure()
        self.particles.pressures = taitEquationOfState(self.particles.densities, rho, self.soundSpeed)

        self.piston = PistonDrive(impact, wave.bathymetry, backX, self._crestSpeed(), self.duration)
        self._pistonVelocity = np.zeros(self.planes)
        self.activePlanes = 1
        wallX = self.boundary.wallX[0]
        self._pistonVelocity[0] = self.piston.velocity(wallX - self.shear * self.planeY(0), 0.0)
        self.boundary.movePiston(self.particles, 0, wallX, self._pistonVelocity[0])
        self._halfSteps = 0
        self.frames = []
        self._recordFrame()

    def run(self, progress: bool = True) -> None:

        '''

        Step the SPH water for the duration: each step moves the active planes' pistons and puts
        the buffer planes in the template's state at their phase; every lag a plane joins the SPH
        water and a new buffer plane is seeded.

        Raises:
        -------
        NumericalInstabilityError : If any particle position stops being finite.

        '''

        if self.solver is None:
            self.build()

        solver, boundary = self.solver, self.boundary
        lag = self.planeLag
        nextFrame = solver.time + self.frameInterval
        nextReport = solver.time + self.duration / 10.0
        started = timeModule.perf_counter()
        if progress:
            print(f'PeelingWave: {self.planes} planes {self.spacing:.4f} m apart, one every {lag:.4e} s, '
                  f'peeling at {self.peelSpeed:.2f} m/s; {self.particles.nFluid} fluid particles to start')

        while solver.time < self.duration:
            state = solver.step()
            dt, p = state.dt, self.particles
            # Each active plane's wall follows the flow behind it, delayed by the plane's lag
            for row in range(self.activePlanes):
                pistonX = boundary.wallX[row] + self._pistonVelocity[row] * dt
                self._pistonVelocity[row] = self.piston.velocity(pistonX - self.shear * self.planeY(row),
                                                                 solver.time - row * lag)
                boundary.movePiston(p, row, pistonX, self._pistonVelocity[row])
            # Planes whose jet has landed join the SPH water, and new buffer planes are seeded
            while self.activePlanes < self.planes and solver.time >= self.activePlanes * lag:
                plane = self.activePlanes
                self._activate(plane)
                if plane + self.bufferPlanes < self.planes:
                    self._seedBuffer(plane + self.bufferPlanes)
            for plane in self._buffers:
                self._placeBuffer(plane, 2.0 * (plane - solver.time / lag))

            if solver.time >= nextFrame:
                if not np.all(np.isfinite(self.particles.positions)):
                    raise NumericalInstabilityError('Particle positions stopped being finite.',
                                                    variableName = 'positions', operation = 'peeling wave',
                                                    context = {'time': solver.time, 'step': state.step})
                self._recordFrame()
                nextFrame += self.frameInterval
            if progress and solver.time >= nextReport:
                elapsed = timeModule.perf_counter() - started
                print(f'  t = {solver.time:6.3f} s  step {state.step:6d}  dt {dt:.2e} s  wall {elapsed:7.1f} s  '
                      f'planes {self.activePlanes:4d}  fluid {self.particles.nFluid:8d}  '
                      f'max speed {state.maxVelocity:.2f} m/s')
                nextReport += self.duration / 10.0
        else:
            self.stopReason = 'end time reached'

        self.wallClockSeconds += timeModule.perf_counter() - started

    def planeHistory(self) -> tuple[np.ndarray, np.ndarray]:

        '''

        Each plane's mechanical energy per unit crest width against the time since its own impact.

        Returns:
        --------
        tuple[np.ndarray, np.ndarray] : ages [s] and energies [J/m], shape (frames, planes), NaN
            before a plane's impact

        '''

        lag = self.planeLag
        times = np.array([frame.time for frame in self.frames])
        ages = times[:, None] - lag * np.arange(self.planes)[None, :]
        energy = np.array([frame.planeKinetic + frame.planePotential for frame in self.frames])
        energy[ages < 0.0] = np.nan

        return ages, energy

    def stripResidual(self, points: int = 4000) -> dict:

        '''

        The stacked flow's spanwise divergence against its in-plane divergence at the jet's impact.

        d v / d y = tan^2(alpha) (u_x + 2 u_t / c + phi_tt / c^2), with the time derivatives taken
        across one lag; the in-plane divergence scale is |d u / d x|, which d w / d z balances.
        The ratio is the share of the strip-theory state that is not incompressible.

        Returns:
        --------
        dict : rms and largest |d v / d y| over the largest |d u / d x|, over random points in the
            water within two depths of the crest

        '''

        impact, before = self.fields[0], self.fields[2]
        lag = self.planeLag
        rng = np.random.default_rng(0)
        crestX = float(impact.surface[np.argmax(impact.surface[:, 1]), 0])
        low = np.array([crestX - 2.0 * self.depth, float(np.min(self.window.z))])
        high = np.array([crestX + 2.0 * self.depth, float(np.max(impact.surface[:, 1]))])
        candidates = rng.uniform(low, high, size = (4 * points, 2))
        candidates = candidates[impact.inside(candidates) & before.inside(candidates)][:points]
        step = 0.25 * self.spacing
        forward, backward = candidates + np.array([step, 0.0]), candidates - np.array([step, 0.0])
        ux = (impact.velocity(forward)[:, 0] - impact.velocity(backward)[:, 0]) / (2.0 * step)
        ut = (impact.velocity(candidates)[:, 0] - before.velocity(candidates)[:, 0]) / lag
        phiTT = (impact.potentialRate(candidates) - before.potentialRate(candidates)) / lag
        c = self.incidentSpeed
        divergence = self.shear**2 * (ux + 2.0 * ut / c + phiTT / c**2)
        scale = float(np.max(np.abs(ux)))

        return {'rms': float(np.sqrt(np.mean(divergence**2))) / scale,
                'largest': float(np.max(np.abs(divergence))) / scale, 'points': int(len(candidates))}
