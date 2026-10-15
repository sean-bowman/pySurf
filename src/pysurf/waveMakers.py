
# -- pySurf: Wave Makers -- #

'''

The ways a wave enters an SPH simulation.

    PistonWaveMaker     a moving wall: boundary particles on the offshore face oscillate
                        horizontally, with the stroke sized by Biesel's transfer function so the
                        generated wave has the requested height
    solitaryWaveFluid   an initial condition: fluid particles laid under a first-order solitary
                        wave already in motion, with the velocity field that carries it

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
tests/testValidationSolitary.py.

Sean Bowman [10/11/2026]

'''

# Standard library imports
from __future__ import annotations

import math
from dataclasses import dataclass

# Third-party imports
import numpy as np

# Local imports
from pysurf import waveTheory
from pysurf.bathymetry import Bathymetry
from pysurf.sphParticles import ParticleSystem, fillRegion

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
