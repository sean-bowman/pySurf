
# -- pySurf: Wave Makers -- #

'''

The ways a wave enters an SPH simulation.

    PistonWaveMaker     a moving wall: boundary particles on the offshore face oscillate
                        horizontally, with the stroke sized by Biesel's transfer function so the
                        generated wave has the requested height
    solitaryWaveFluid   an initial condition: fluid particles laid under a first-order solitary
                        wave already in motion, with the velocity field that carries it
    exactSolitaryWaveFluid
                        the same under the exact potential flow solitary wave, with its pressure;
                        `exactSolitaryWaveFluid3D` lays it straight across a 3D bed. Both lay the
                        particles on the square lattice or, with `bedFitted`, in columns half a
                        spacing clear of the bed (`sphParticles.bedFittedPoints`); with `jitter`
                        each particle then moves off its place at random
                        (`sphParticles.jitterPoints`)

A periodic wave train needs the piston and a long run-in for the generated waves to settle. A
single solitary wave needs neither: it starts as a wave of permanent form, travels unchanged over
a flat bottom, and shoals and breaks on a slope, which is why it is the standard benchmark for
breaking.

References:
-----------
Biesel, F. (1951) -- Etude theorique d'un type d'appareil a houle, La Houille Blanche 6.
Madsen, O.S. (1971) -- On the generation of long waves, J. Geophys. Res. 76(36), 8672-8683.
Dean, R.G. & Dalrymple, R.A. (1991) -- Water Wave Mechanics for Engineers and Scientists, ch. 6
    (wave makers) and ch. 11 (solitary waves).

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

tests/testWaveMakers.py checks the piston stroke against the shallow and deep limits of
`waveTheory.pistonTransferFunction`, and the paddle velocity against the derivative of its
displacement. It checks the solitary initial condition against its definition: the particles fill
the profile to within the lattice spacing and carry the wave's volume, and the velocity field
satisfies continuity, the bed condition and the kinematic free-surface condition. The wave it launches is validated against the solitary celerity in
tests/testValidationSolitary.py. The 3D exact wave over a flat height field matches the 2D one's
particles, velocities and pressures in every lattice row across the tank. A jittered start keeps
the ordered start's particle count, lies inside the water, and carries the exact flow at each
particle's final place, in 2D and 3D; `exactSurfaceElevation` matches the surface the flow reads.

Sean Bowman [10/11/2026]

'''

# Standard library imports
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

# Third-party imports
import numpy as np

# Local imports
from pysurf import waveTheory
from pysurf.bathymetry import Bathymetry, HeightField
from pysurf.sphParticles import ParticleSystem, bedFittedPoints, fillRegion, jitterPoints, latticePoints

#----------------------------------------------------------------------#
# -- Piston Wave Maker -- #
#----------------------------------------------------------------------#

@dataclass
class WaveMakerConfig:

    '''

    A piston wave maker's target wave and start-up.

    Parameters:
    -----------
    waveHeight : float
        Target wave height H [m]
    wavePeriod : float
        Wave period T [s]
    waterDepth : float
        Still water depth at the paddle d [m]
    rampCycles : int
        Wave periods over which the stroke ramps from zero
    useSecondOrder : bool
        Add Madsen's second-order stroke, which suppresses the free second harmonic a purely
        sinusoidal paddle radiates

    '''

    waveHeight: float
    wavePeriod: float
    waterDepth: float
    rampCycles: int = 3
    useSecondOrder: bool = True

class PistonWaveMaker:

    '''

    A piston paddle made of boundary particles that move horizontally as a rigid wall.

    X(t) = (S / 2) sin(omega t) ramp(t), with S = H / (H/S)_Biesel; with the second-order
    correction, S2 = S k H / 4 adds (S2 / 2) sin(2 omega t).

    Parameters:
    -----------
    config : WaveMakerConfig
        Target wave and start-up

    '''

    def __init__(self, config: WaveMakerConfig) -> None:

        self._config = config
        self._omega = 2.0 * math.pi / config.wavePeriod
        self._k = float(waveTheory.dispersionWavenumber(self._omega, config.waterDepth))
        self._stroke = config.waveHeight / float(waveTheory.pistonTransferFunction(self._k * config.waterDepth))
        self._stroke2 = 0.25 * self._stroke * self._k * config.waveHeight if config.useSecondOrder else 0.0

        self._paddleMask: np.ndarray | None = None
        self._restX: np.ndarray | None = None

    @property
    def stroke(self) -> float:
        '''Piston stroke S, peak to peak [m].'''
        return self._stroke

    @property
    def wavenumber(self) -> float:
        '''Wavenumber k [rad/m].'''
        return self._k

    @property
    def wavelength(self) -> float:
        '''Wavelength L = 2 pi / k [m].'''
        return 2.0 * math.pi / self._k

    @property
    def phaseSpeed(self) -> float:
        '''Phase speed c = omega / k [m/s].'''
        return self._omega / self._k

    def _ramp(self, time: float) -> float:

        '''Linear start-up ramp from 0 to 1 over the configured number of periods.'''

        rampTime = self._config.rampCycles * self._config.wavePeriod

        return min(1.0, time / rampTime) if rampTime > 0.0 else 1.0

    def initializeBoundaryParticles(self, particles: ParticleSystem, pistonX: float, tolerance: float = 0.05) -> None:

        '''

        Take as the paddle every boundary particle behind the piston face and within `tolerance`
        of it. Bed particles in front of the face stay put.

        Raises:
        -------
        ValueError : If no boundary particle is near the face.

        '''

        x = particles.positions[:, 0]
        self._paddleMask = (~particles.isFluid) & (x < pistonX) & (x > pistonX - tolerance)
        self._restX = particles.positions[self._paddleMask, 0].copy()

        if not np.any(self._paddleMask):
            raise ValueError(f'No boundary particles found near piston at x={pistonX} (tolerance={tolerance}).')

    def getDisplacement(self, time: float) -> float:

        '''Paddle displacement from rest [m].'''

        phase = self._omega * time
        displacement = 0.5 * self._stroke * math.sin(phase)
        if self._stroke2 > 0.0:
            displacement += 0.5 * self._stroke2 * math.sin(2.0 * phase)

        return displacement * self._ramp(time)

    def getVelocity(self, time: float) -> float:

        '''Paddle velocity [m/s], the derivative of the displacement with the ramp held fixed.'''

        phase = self._omega * time
        velocity = 0.5 * self._stroke * self._omega * math.cos(phase)
        if self._stroke2 > 0.0:
            velocity += self._stroke2 * self._omega * math.cos(2.0 * phase)

        return velocity * self._ramp(time)

    def updateBoundaryMotion(self, particles: ParticleSystem, time: float) -> None:

        '''

        Move the paddle particles to their position at `time` and give them its velocity.

        Raises:
        -------
        RuntimeError : If the paddle particles were never identified.

        '''

        if self._paddleMask is None:
            raise RuntimeError('Wave maker not initialized. Call initializeBoundaryParticles first.')

        particles.positions[self._paddleMask, 0] = self._restX + self.getDisplacement(time)
        particles.velocities[self._paddleMask] = 0.0
        particles.velocities[self._paddleMask, 0] = self.getVelocity(time)

#----------------------------------------------------------------------#
# -- Solitary Wave Initial Condition -- #
#----------------------------------------------------------------------#

def solitaryWaveVelocityField(x: np.ndarray, y: np.ndarray, height: float, depth: float, crestX: float,
                              bedElevation: np.ndarray) -> np.ndarray:

    '''

    The velocity of a first-order solitary wave at points (x, y), shape (N, 2) [m/s].

    The horizontal velocity is depth-uniform, u = c eta / (d + eta). The vertical velocity follows
    from continuity, du/dx + dw/dy = 0 with w = 0 on the bed:

        w = -(y - z_b) du/dx = -(y - z_b) c d eta' / (d + eta)^2

    At the free surface this equals -c d eta' / (d + eta), which is exactly the kinematic condition
    eta_t + u eta_x for a wave translating at c, so the surface starts moving with the wave.

    Parameters:
    -----------
    x, y : np.ndarray
        Point positions [m]
    height : float
        Wave height above still water H [m]
    depth : float
        Still water depth d [m]
    crestX : float
        Crest position [m]
    bedElevation : np.ndarray
        Bed elevation z_b under each point [m]

    '''

    celerity = waveTheory.solitaryWaveSpeed(height, depth)
    kappa = waveTheory.solitaryWaveDecay(height, depth)
    eta = waveTheory.solitaryWaveProfile(x, height, depth, crestX)
    etaSlope = -2.0 * kappa * eta * np.tanh(kappa * (x - crestX))

    u = celerity * eta / (depth + eta)
    w = -(y - bedElevation) * celerity * depth * etaSlope / (depth + eta)**2

    return np.column_stack([u, w])

def solitaryWaveFluid(bathymetry: Bathymetry, height: float, crestX: float, spacing: float) -> tuple[np.ndarray, np.ndarray]:

    '''

    Fluid particles under a solitary wave over a bed, and their velocities.

    Lattice points are kept from half a spacing above the bed up to the wave's free surface,
    SWL + eta(x); the boundary's first layer sits half a spacing below the bed, so the fluid starts
    one spacing from it. The depth that sets the wave's shape and speed is the still water depth at
    the crest, so the crest should sit over the flat part of the bed with the wave clear of the slope.

    Parameters:
    -----------
    bathymetry : Bathymetry
        The bed, with its still water level
    height : float
        Wave height above still water H [m]
    crestX : float
        Crest position [m]
    spacing : float
        Lattice spacing [m]

    Returns:
    --------
    tuple : (positions, velocities), each shaped (N, 2)

    '''

    depth = float(bathymetry.depth(crestX))
    swl = bathymetry.stillWaterLevel

    def inside(points: np.ndarray) -> np.ndarray:
        bed = bathymetry.elevation(points[:, 0])
        surface = swl + waveTheory.solitaryWaveProfile(points[:, 0], height, depth, crestX)
        # The tolerance keeps the lattice row exactly half a spacing above a flat bed
        return (points[:, 1] >= bed + (0.5 - 1e-9) * spacing) & (points[:, 1] < surface)

    lower = [bathymetry.startX, float(np.min(bathymetry.z))]
    upper = [bathymetry.endX, swl + height + spacing]
    positions = fillRegion(inside, lower, upper, spacing)

    velocities = solitaryWaveVelocityField(positions[:, 0], positions[:, 1], height, depth, crestX,
                                           bathymetry.elevation(positions[:, 0]))

    return positions, velocities

def exactSurfaceElevation(x: np.ndarray, height: float, depth: float, crestX: float) -> np.ndarray:

    '''

    The exact solitary wave's surface above still water at positions x [m], read as
    `waveTheory.exactSolitaryFlow` reads it (its third return) without evaluating the flow: along
    the conformal map's top level, and zero beyond the wave's interval.

    '''

    levelX, levelY, _, _ = waveTheory.exactSolitaryInterior(round(height / depth, 15))
    xq = (np.asarray(x, dtype = float) - crestX) / depth
    surface = np.interp(xq, levelX[-1], levelY[-1])
    still = (xq <= levelX[:, 0].max()) | (xq >= levelX[:, -1].min())

    return np.where(still, 0.0, surface) * depth

def exactSolitaryWaveFluid(bathymetry: Bathymetry, height: float, crestX: float, spacing: float,
                           bedFitted: bool = False, jitter: float = 0.0, seed: int = 1,
                           pack: Callable | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    '''

    Fluid particles under the exact potential flow solitary wave over a bed, with their velocities
    and pressures (`waveTheory.exactSolitaryFlow`).

    Lattice points are kept from half a spacing above the bed up to the wave's free surface, as in
    `solitaryWaveFluid`; with `bedFitted`, the points of `sphParticles.bedFittedPoints` instead,
    whose lowest point sits half a spacing above the bed in every column, so the bed's pressure
    holds the water over a slope from the start. With `jitter`, every point then moves off its
    place at random (`sphParticles.jitterPoints`), reflected back into the water across the walls,
    the line half a spacing above the bed and the wave's surface; with `pack`, the points then
    move as it moves them (`sphPacking.packPoints`). The velocity is the fully
    nonlinear wave's, from its conformal map, and the pressure is unsteady Bernoulli's for a wave
    of permanent form, both at each particle's final place, so the water starts in the state the
    wave carries rather than settling into it from a hydrostatic one. The flow is that of a flat
    bed of the depth at the crest; the crest should sit over the flat part with the wave clear of
    any slope.

    Parameters:
    -----------
    bathymetry : Bathymetry
        The bed, with its still water level
    height : float
        Wave height above still water H [m]
    crestX : float
        Crest position [m]
    spacing : float
        Lattice spacing [m]
    bedFitted : bool
        Lay the particles in bed-fitted columns rather than on the square lattice
    jitter : float
        Largest random offset of each particle from its place along each axis, in spacings; zero
        for none
    seed : int
        Seed of the offsets
    pack : Callable | None
        Moves the laid points before the flow is evaluated: called with the points and the
        surface's elevation eta(x) [m], it returns the points' new places (`BreakingWave`'s
        `startPacking`); None leaves them

    Returns:
    --------
    tuple : positions and velocities, each shaped (N, 2) [m, m/s], and pressures over density,
        shape (N,) [m^2/s^2]

    '''

    depth = float(bathymetry.depth(crestX))
    level = bathymetry.stillWaterLevel

    def inside(points: np.ndarray) -> np.ndarray:
        bed = bathymetry.elevation(points[:, 0])
        _, _, surface = waveTheory.exactSolitaryFlow(points, height, depth, crestX, level)
        # The tolerance keeps the lattice row exactly half a spacing above a flat bed
        return (points[:, 1] >= bed + (0.5 - 1e-9) * spacing) & (points[:, 1] < level + surface)

    lower = [bathymetry.startX, float(np.min(bathymetry.z))]
    upper = [bathymetry.endX, level + height + spacing]
    if bedFitted:
        candidates = bedFittedPoints(lower, upper, spacing, bathymetry.elevation)
        positions = candidates[inside(candidates)]
    else:
        positions = fillRegion(inside, lower, upper, spacing)
    surface = lambda x: level + exactSurfaceElevation(x, height, depth, crestX)
    if jitter > 0.0:
        positions = jitterPoints(positions, spacing, jitter, seed, [bathymetry.startX], [bathymetry.endX],
                                 bathymetry.elevation, surface)
    if pack is not None:
        positions = pack(positions, surface)
    velocities, pressures, _ = waveTheory.exactSolitaryFlow(positions, height, depth, crestX, level)

    return positions, velocities, np.maximum(pressures, 0.0)

def exactSolitaryBoundaryPressure(bathymetry: Bathymetry, positions: np.ndarray, height: float, crestX: float,
                                  spacing: float, gravity: float = 9.80665) -> np.ndarray:

    '''

    The pressure over density each boundary particle starts with under the exact solitary wave
    [m^2/s^2], as `sphHandoff.boundaryPressure` sets it after a hand-off: that of the water at the
    nearest point half a spacing inside it (above the bed, or in from a wall), plus the hydrostatic
    increment down to the particle; zero where that point is above the surface.

    '''

    depth = float(bathymetry.depth(crestX))
    level = bathymetry.stillWaterLevel
    reference = np.column_stack([np.clip(positions[:, 0], bathymetry.startX + 0.5 * spacing,
                                         bathymetry.endX - 0.5 * spacing), positions[:, 1]])
    reference[:, 1] = np.maximum(reference[:, 1], bathymetry.elevation(reference[:, 0]) + 0.5 * spacing)
    _, pressure, surface = waveTheory.exactSolitaryFlow(reference, height, depth, crestX, level, gravity)
    wet = reference[:, 1] < level + surface

    return np.where(wet, np.maximum(pressure, 0.0) + gravity * (reference[:, 1] - positions[:, 1]), 0.0)

def exactFlowInChunks(points: np.ndarray, height: float, depth: float, crestX: float, stillWaterLevel: float,
                      gravity: float = 9.80665, chunk: int = 50000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    '''

    `waveTheory.exactSolitaryFlow` over many points, a chunk at a time: each call holds every level
    of the wave's conformal map for every point it is given, about 3 kB a point.

    '''

    points = np.asarray(points, dtype = float)
    velocities, pressures, surface = np.zeros((len(points), 2)), np.zeros(len(points)), np.zeros(len(points))
    for start in range(0, len(points), chunk):
        part = slice(start, start + chunk)
        velocities[part], pressures[part], surface[part] = waveTheory.exactSolitaryFlow(
            points[part], height, depth, crestX, stillWaterLevel, gravity)

    return velocities, pressures, surface

def exactSolitaryWaveFluid3D(field: HeightField, height: float, depth: float, crestX: float, spacing: float,
                             floor: float | None = None, bedFitted: bool = False, jitter: float = 0.0,
                             seed: int = 1, pack: Callable | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    '''

    Fluid particles under the exact potential flow solitary wave, its crest straight along y, over
    a 3D bed, with their velocities and pressures.

    The wave is `exactSolitaryWaveFluid`'s, the flow of a flat bed `depth` deep, the same in every
    vertical (x, z) plane. Lattice points inside the field's bounds are kept from half a spacing
    above the bed up to the wave's surface; with `bedFitted`, the points of
    `sphParticles.bedFittedPoints` instead, half a spacing above the bed in every column. With
    `jitter`, every point then moves off its place at random along x, y and z
    (`sphParticles.jitterPoints`), reflected back into the water across the side and end walls,
    the line half a spacing above the bed and the wave's surface; with `pack`, the points then move
    as it moves them (`sphPacking.packPoints`). Either way the flow is evaluated at each point's
    final place. The spanwise velocity is zero. Where the bed rises under the wave the flow
    does not meet the bed's slope, as in the boundary element model's exact start, which takes the
    same flat-bed wave over a slope.

    Parameters:
    -----------
    field : HeightField
        The bed, with its still water level and bounds
    height : float
        Wave height above still water H [m]
    depth : float
        Still water depth of the flat-bed wave [m]
    crestX : float
        Crest position [m]
    spacing : float
        Lattice spacing [m]
    floor : float | None
        The level the lattice's heights are counted from [m]; None for the bed's lowest point
    bedFitted : bool
        Lay the particles in bed-fitted columns rather than on the square lattice
    jitter : float
        Largest random offset of each particle from its place along each axis, in spacings; zero
        for none
    seed : int
        Seed of the offsets
    pack : Callable | None
        Moves the laid points before the flow is evaluated: called with the points and the
        surface's elevation eta(x, y) [m], it returns the points' new places (`BreakingWave3D`'s
        `startPacking`); None leaves them

    Returns:
    --------
    tuple : positions and velocities, each shaped (N, 3) [m, m/s], and pressures over density,
        shape (N,) [m^2/s^2]

    '''

    s, level = spacing, field.stillWaterLevel
    floor = field.lowestElevation() if floor is None else floor
    startX, endX, startY, endY = field.bounds

    if jitter > 0.0 or pack is not None:
        # Every point moves, so the flow is evaluated at each one rather than once per (x, z)
        upper = [endX, endY, level + height + s]
        if bedFitted:
            points = bedFittedPoints([startX, startY, floor], upper, s, field.elevation)
        else:
            points = latticePoints([startX, startY, floor], upper, s)
            # The tolerance keeps the lattice row exactly half a spacing above a flat bed
            points = points[points[:, 2] >= field.elevation(points[:, 0], points[:, 1]) + (0.5 - 1e-9) * s]
        points = points[points[:, 2] < level + exactSurfaceElevation(points[:, 0], height, depth, crestX)]
        surface = lambda x, y: level + exactSurfaceElevation(x, height, depth, crestX)
        points = jitterPoints(points, s, jitter, seed, [startX, startY], [endX, endY],
                              lambda x, y: np.maximum(field.elevation(x, y), floor), surface)
        if pack is not None:
            points = pack(points, surface)
        velocities, pressures, _ = exactFlowInChunks(points[:, [0, 2]], height, depth, crestX, level)
        flow = np.column_stack([velocities[:, 0], np.zeros(len(points)), velocities[:, 1]])
        return points, flow, np.maximum(pressures, 0.0)

    if bedFitted:
        # Each column's heights follow its own bed, so the flow is evaluated at every distinct
        # (x, z); over a bed flat across y the rows share theirs
        points = bedFittedPoints([startX, startY, floor], [endX, endY, level + height + s], s, field.elevation)
        planar, inverse = np.unique(points[:, [0, 2]], axis = 0, return_inverse = True)
        velocities, pressures, surface = exactFlowInChunks(planar, height, depth, crestX, level)
        inverse = inverse.reshape(-1)
        wet = points[:, 2] < level + surface[inverse]
        index = inverse[wet]
        flow = np.column_stack([velocities[index, 0], np.zeros(len(index)), velocities[index, 1]])
        return points[wet], flow, np.maximum(pressures[index], 0.0)

    # The flow on one (x, z) plane of the lattice, then that plane laid at every row across
    plane = latticePoints([startX, floor], [endX, level + height + s], s)
    velocities, pressures, surface = exactFlowInChunks(plane, height, depth, crestX, level)
    wet = plane[:, 1] < level + surface
    plane, velocities, pressures = plane[wet], velocities[wet], np.maximum(pressures[wet], 0.0)
    rows = np.arange(startY + 0.5 * s, endY, s)

    x, z = np.tile(plane[:, 0], rows.size), np.tile(plane[:, 1], rows.size)
    y = np.repeat(rows, len(plane))
    # The tolerance keeps the lattice row exactly half a spacing above a flat bed
    keep = z >= field.elevation(x, y) + (0.5 - 1e-9) * s
    count = len(plane)
    index = np.tile(np.arange(count), rows.size)[keep]
    positions = np.column_stack([x[keep], y[keep], z[keep]])
    flow = np.column_stack([velocities[index, 0], np.zeros(len(index)), velocities[index, 1]])

    return positions, flow, pressures[index]

def exactSolitaryBoundaryPressure3D(field: HeightField, positions: np.ndarray, height: float, depth: float,
                                    crestX: float, spacing: float, gravity: float = 9.80665) -> np.ndarray:

    '''

    The pressure over density each boundary particle of a 3D tank starts with under the exact
    solitary wave [m^2/s^2], by `exactSolitaryBoundaryPressure`'s rule: that of the water at the
    point half a spacing inside it (above the bed, and in from the walls), plus the hydrostatic
    increment down to the particle; zero where that point is above the surface.

    '''

    startX, endX, startY, endY = field.bounds
    level = field.stillWaterLevel
    referenceX = np.clip(positions[:, 0], startX + 0.5 * spacing, endX - 0.5 * spacing)
    referenceY = np.clip(positions[:, 1], startY + 0.5 * spacing, endY - 0.5 * spacing)
    referenceZ = np.maximum(positions[:, 2], field.elevation(referenceX, referenceY) + 0.5 * spacing)
    _, pressure, surface = exactFlowInChunks(np.column_stack([referenceX, referenceZ]), height, depth, crestX, level,
                                             gravity)
    wet = referenceZ < level + surface

    return np.where(wet, np.maximum(pressure, 0.0) + gravity * (referenceZ - positions[:, 2]), 0.0)
