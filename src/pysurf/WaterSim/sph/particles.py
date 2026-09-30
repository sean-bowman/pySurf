# -- SPH Particle System -- #

'''

Dataclass representing the SPH particle system state.

Stores positions, velocities, accelerations, densities, pressures,
and masses as contiguous NumPy arrays for vectorized operations.
Both fluid and boundary particles are stored together, distinguished
by the isFluid boolean mask.

Sean Bowman [02/05/2026]

'''

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass
class ParticleSystem:

    '''

    SPH particle system state.

    All arrays have shape (nParticles, nDimensions) for vector
    quantities and (nParticles,) for scalar quantities.
    Fluid and boundary particles coexist in the same arrays,
    distinguished by the isFluid mask.

    Parameters:
    -----------
    positions : np.ndarray
        Particle positions [m], shape (N, dim)
    velocities : np.ndarray
        Particle velocities [m/s], shape (N, dim)
    accelerations : np.ndarray
        Particle accelerations [m/s^2], shape (N, dim)
    densities : np.ndarray
        Particle densities [kg/m^3], shape (N,)
    pressures : np.ndarray
        Particle pressures [Pa], shape (N,)
    masses : np.ndarray
        Particle masses [kg], shape (N,)
    isFluid : np.ndarray
        Boolean mask: True for fluid, False for boundary, shape (N,)

    '''

    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    densities: np.ndarray
    pressures: np.ndarray
    masses: np.ndarray
    isFluid: np.ndarray

    @property
    def nParticles(self) -> int:

        '''Total number of particles (fluid + boundary).'''
        return self.positions.shape[0]

    @property
    def nFluid(self) -> int:

        '''Number of fluid particles.'''
        return int(np.sum(self.isFluid))

    @property
    def nBoundary(self) -> int:

        '''Number of boundary particles.'''
        return self.nParticles - self.nFluid

    @property
    def dimensions(self) -> int:

        '''Number of spatial dimensions (2 or 3).'''
        return self.positions.shape[1]

    @property
    def fluidMask(self) -> np.ndarray:

        '''Boolean mask for fluid particles.'''
        return self.isFluid

    @property
    def boundaryMask(self) -> np.ndarray:

        '''Boolean mask for boundary particles.'''
        return ~self.isFluid

    def kineticEnergy(self) -> float:

        '''

        Total kinetic energy of fluid particles.

        KE = (1/2) * sum_i m_i * |v_i|^2

        Returns:
        --------
        float : Kinetic energy [J]

        '''
        fluidVels = self.velocities[self.isFluid]
        fluidMasses = self.masses[self.isFluid]
        speedsSq = np.sum(fluidVels * fluidVels, axis=1)
        return 0.5 * np.sum(fluidMasses * speedsSq)

    def potentialEnergy(self, gravity: float = 9.81) -> float:

        '''

        Total gravitational potential energy of fluid particles.

        PE = sum_i m_i * g * y_i

        Uses the y-coordinate (index 1) as the vertical axis.
        Reference level is y = 0.

        Parameters:
        -----------
        gravity : float
            Gravitational acceleration magnitude [m/s^2]

        Returns:
        --------
        float : Potential energy [J]

        '''
        fluidPositions = self.positions[self.isFluid]
        fluidMasses = self.masses[self.isFluid]
        # y-coordinate is vertical in 2D (index 1)
        heights = fluidPositions[:, 1]
        return np.sum(fluidMasses * gravity * heights)

    def maxSpeed(self) -> float:

        '''

        Maximum velocity magnitude among fluid particles.

        Returns:
        --------
        float : Maximum speed [m/s]

        '''
        fluidVels = self.velocities[self.isFluid]
        if len(fluidVels) == 0:
            return 0.0
        speeds = np.linalg.norm(fluidVels, axis=1)
        return float(np.max(speeds))

    def maxDensityError(self, referenceDensity: float) -> float:

        '''

        Maximum relative density error among fluid particles.

        Returns max |rho_i - rho_0| / rho_0

        Parameters:
        -----------
        referenceDensity : float
            Reference density rho_0 [kg/m^3]

        Returns:
        --------
        float : Maximum relative density error (dimensionless)

        '''
        fluidDensities = self.densities[self.isFluid]
        if len(fluidDensities) == 0:
            return 0.0
        errors = np.abs(fluidDensities - referenceDensity) / referenceDensity
        return float(np.max(errors))

    @classmethod
    def createUniform(
        cls,
        domainMin: np.ndarray,
        domainMax: np.ndarray,
        spacing: float,
        referenceDensity: float,
    ) -> ParticleSystem:

        '''

        Create a uniform particle grid filling a rectangular domain.

        Particles are placed on a regular grid with the given spacing.
        All particles are marked as fluid. Particle mass is computed
        from the reference density and spacing.

        Parameters:
        -----------
        domainMin : np.ndarray
            Lower corner of the fluid domain [m]
        domainMax : np.ndarray
            Upper corner of the fluid domain [m]
        spacing : float
            Inter-particle spacing [m]
        referenceDensity : float
            Reference density [kg/m^3]

        Returns:
        --------
        ParticleSystem : Initialized particle system with all fluid particles

        '''
        dimensions = len(domainMin)

        # Create grid coordinates along each axis
        # Offset by half spacing so particles aren't on the boundary
        axes = []
        for d in range(dimensions):
            coords = np.arange(
                domainMin[d] + spacing / 2.0,
                domainMax[d],
                spacing,
            )
            axes.append(coords)

        # Create meshgrid and flatten to particle positions
        if dimensions == 2:
            xx, yy = np.meshgrid(axes[0], axes[1], indexing='xy')
            positions = np.column_stack([xx.ravel(), yy.ravel()])
        else:
            xx, yy, zz = np.meshgrid(axes[0], axes[1], axes[2], indexing='xy')
            positions = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        nParticles = positions.shape[0]

        # Particle mass: m = rho * V_particle
        # In 2D, V = spacing^2; in 3D, V = spacing^3
        particleVolume = spacing ** dimensions
        mass = referenceDensity * particleVolume

        return cls(
            positions=positions,
            velocities=np.zeros((nParticles, dimensions)),
            accelerations=np.zeros((nParticles, dimensions)),
            densities=np.full(nParticles, referenceDensity),
            pressures=np.zeros(nParticles),
            masses=np.full(nParticles, mass),
            isFluid=np.ones(nParticles, dtype=bool),
        )
