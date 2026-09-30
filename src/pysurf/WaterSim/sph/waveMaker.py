# -- Piston-Type Wave Maker Boundary -- #

'''
Piston-type wave-maker boundary condition for generating waves in SPH.

Uses the Biesel transfer function to relate piston stroke to wave amplitude,
with optional second-order Madsen correction to eliminate parasitic long waves.

Theory:
-------
For a piston moving with displacement X(t) = S/2 * sin(omega*t):
    H/S = 2 * (cosh(2kd) - 1) / (sinh(2kd) + 2kd)

where:
    H = target wave height [m]
    S = piston stroke (peak-to-peak displacement) [m]
    k = wavenumber [rad/m]
    d = water depth [m]

The wave-maker boundary particles move horizontally (x-direction) following
the prescribed piston motion. This implements the Dynamic Boundary Condition
(DBC) where boundary particles satisfy SPH equations but follow prescribed motion.

References:
-----------
Biesel (1951) - Wave maker theory
Madsen (1971) - On the generation of long waves
DualSPHysics - SPH formulation v5.0

Sean Bowman [02/06/2026]
'''

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from pysurf.Surfboard.SurfPhysics.waves.waveConditions import WaveConditions
from pysurf.Surfboard.SurfPhysics.waves.linearWaveTheory import LinearWaveTheory
from pysurf.WaterSim.sph.particles import ParticleSystem


#--------------------------------------------------------------------#
# -- Configuration -- #
#--------------------------------------------------------------------#

@dataclass
class WaveMakerConfig:
    '''
    Configuration for piston-type wave-maker.

    Parameters:
    -----------
    waveHeight : float
        Target wave height H [m]
    wavePeriod : float
        Wave period T [s]
    waterDepth : float
        Still water depth d [m]
    rampCycles : int
        Number of wave cycles for amplitude ramp-up (default 3)
    useSecondOrder : bool
        Apply Madsen second-order correction (default True)
    '''
    waveHeight: float
    wavePeriod: float
    waterDepth: float
    rampCycles: int = 3
    useSecondOrder: bool = True


#--------------------------------------------------------------------#
# -- Wave Maker Protocol -- #
#--------------------------------------------------------------------#

class WaveMaker(Protocol):
    '''Protocol for wave-maker boundary conditions.'''

    def initializeBoundaryParticles(
        self,
        particles: ParticleSystem,
        pistonX: float,
        tolerance: float,
    ) -> None:
        '''Identify wave-maker boundary particles.'''
        ...

    def updateBoundaryMotion(
        self,
        particles: ParticleSystem,
        time: float,
    ) -> None:
        '''Update wave-maker particle positions and velocities.'''
        ...


#--------------------------------------------------------------------#
# -- Piston Wave Maker -- #
#--------------------------------------------------------------------#

class PistonWaveMaker:
    '''
    Piston-type wave-maker boundary condition.

    Generates waves by prescribing horizontal motion of boundary
    particles on one face of the domain. Uses the Dynamic Boundary
    Condition: boundary particles satisfy the same SPH equations
    but follow prescribed motion.

    The piston oscillates in the x-direction (wave propagation axis).

    Parameters:
    -----------
    config : WaveMakerConfig
        Wave-maker configuration
    '''

    def __init__(self, config: WaveMakerConfig) -> None:
        self._config = config
        self._waveTheory = LinearWaveTheory()

        # Angular frequency
        self._omega = 2.0 * math.pi / config.wavePeriod

        # Solve dispersion relation for wavenumber
        self._k = self._waveTheory.solveDispersionRelation(
            self._omega, config.waterDepth
        )

        # Compute required piston stroke using Biesel transfer function
        self._stroke = self._computeStroke()

        # Second-order correction amplitude (Madsen)
        if config.useSecondOrder:
            kH = self._k * config.waveHeight
            self._stroke2 = 0.25 * self._stroke * kH
        else:
            self._stroke2 = 0.0

        # Store initial x-positions for wave-maker particles
        self._boundaryXInit: np.ndarray | None = None
        self._waveMakerMask: np.ndarray | None = None

    @property
    def stroke(self) -> float:
        '''Piston stroke (peak-to-peak displacement) [m].'''
        return self._stroke

    @property
    def wavenumber(self) -> float:
        '''Wavenumber k [rad/m].'''
        return self._k

    @property
    def wavelength(self) -> float:
        '''Wavelength L = 2*pi/k [m].'''
        return 2.0 * math.pi / self._k

    @property
    def phaseSpeed(self) -> float:
        '''Phase speed c = omega/k [m/s].'''
        return self._omega / self._k

    def _computeStroke(self) -> float:
        '''
        Compute required piston stroke from Biesel transfer function.

        H/S = 2 * (cosh(2kd) - 1) / (sinh(2kd) + 2kd)

        In deep water (kd >> 1): H/S -> 1
        In shallow water (kd << 1): H/S -> kd

        Returns:
        --------
        float : Piston stroke S [m]
        '''
        k = self._k
        d = self._config.waterDepth
        H = self._config.waveHeight

        kd2 = 2.0 * k * d

        # Avoid overflow for very deep water
        if kd2 > 50.0:
            # Deep water limit: H/S -> 1
            return H

        cosh2kd = math.cosh(kd2)
        sinh2kd = math.sinh(kd2)

        transferFunction = 2.0 * (cosh2kd - 1.0) / (sinh2kd + kd2)

        if transferFunction < 1e-12:
            return H  # Fallback

        return H / transferFunction

    def initializeBoundaryParticles(
        self,
        particles: ParticleSystem,
        pistonX: float,
        tolerance: float = 0.05,
    ) -> None:
        '''
        Identify and store initial positions of wave-maker boundary particles.

        Finds all boundary particles whose x-coordinate is within tolerance
        of the piston face position.

        Parameters:
        -----------
        particles : ParticleSystem
            Particle system containing boundary particles
        pistonX : float
            X-coordinate of the piston face [m]
        tolerance : float
            Distance tolerance for identifying piston particles [m]
        '''
        # Find boundary particles near the piston face
        isBoundary = ~particles.isFluid
        nearPiston = np.abs(particles.positions[:, 0] - pistonX) < tolerance
        self._waveMakerMask = isBoundary & nearPiston
        self._boundaryXInit = particles.positions[self._waveMakerMask, 0].copy()

        nPistonParticles = np.sum(self._waveMakerMask)
        if nPistonParticles == 0:
            raise ValueError(
                f'No boundary particles found near piston at x={pistonX} '
                f'(tolerance={tolerance}). Check domain bounds.'
            )

    def updateBoundaryMotion(
        self,
        particles: ParticleSystem,
        time: float,
    ) -> None:
        '''
        Update wave-maker boundary particle positions and velocities.

        First-order motion:
            X(t) = (S/2) * sin(omega*t) * ramp(t)
            V(t) = (S/2) * omega * cos(omega*t) * ramp(t)

        With optional second-order correction (Madsen):
            X2(t) = (S2/2) * sin(2*omega*t) * ramp(t)
            V2(t) = S2 * omega * cos(2*omega*t) * ramp(t)

        Parameters:
        -----------
        particles : ParticleSystem
            Particle system to update
        time : float
            Current simulation time [s]
        '''
        if self._waveMakerMask is None or self._boundaryXInit is None:
            raise RuntimeError(
                'Wave-maker not initialized. Call initializeBoundaryParticles first.'
            )

        omega = self._omega
        S = self._stroke

        # Ramp function: linear ramp over rampCycles periods
        rampTime = self._config.rampCycles * self._config.wavePeriod
        if rampTime > 0.0:
            ramp = min(1.0, time / rampTime)
        else:
            ramp = 1.0

        # First-order piston displacement and velocity
        phase = omega * time
        displacement = 0.5 * S * math.sin(phase) * ramp
        velocity = 0.5 * S * omega * math.cos(phase) * ramp

        # Second-order correction (Madsen)
        if self._config.useSecondOrder and self._stroke2 > 0.0:
            S2 = self._stroke2
            displacement += 0.5 * S2 * math.sin(2.0 * phase) * ramp
            velocity += S2 * omega * math.cos(2.0 * phase) * ramp

        # Update positions and velocities for wave-maker particles
        particles.positions[self._waveMakerMask, 0] = self._boundaryXInit + displacement
        particles.velocities[self._waveMakerMask, 0] = velocity
        # Keep y, z velocities at zero for piston motion
        particles.velocities[self._waveMakerMask, 1] = 0.0
        if particles.positions.shape[1] == 3:
            particles.velocities[self._waveMakerMask, 2] = 0.0

    def getDisplacement(self, time: float) -> float:
        '''
        Get current piston displacement (for diagnostics).

        Parameters:
        -----------
        time : float
            Current time [s]

        Returns:
        --------
        float : Piston displacement from rest position [m]
        '''
        rampTime = self._config.rampCycles * self._config.wavePeriod
        ramp = min(1.0, time / rampTime) if rampTime > 0.0 else 1.0

        phase = self._omega * time
        displacement = 0.5 * self._stroke * math.sin(phase) * ramp

        if self._config.useSecondOrder and self._stroke2 > 0.0:
            displacement += 0.5 * self._stroke2 * math.sin(2.0 * phase) * ramp

        return displacement

    def getVelocity(self, time: float) -> float:
        '''
        Get current piston velocity (for diagnostics).

        Parameters:
        -----------
        time : float
            Current time [s]

        Returns:
        --------
        float : Piston velocity [m/s]
        '''
        rampTime = self._config.rampCycles * self._config.wavePeriod
        ramp = min(1.0, time / rampTime) if rampTime > 0.0 else 1.0

        phase = self._omega * time
        velocity = 0.5 * self._stroke * self._omega * math.cos(phase) * ramp

        if self._config.useSecondOrder and self._stroke2 > 0.0:
            velocity += self._stroke2 * self._omega * math.cos(2.0 * phase) * ramp

        return velocity
