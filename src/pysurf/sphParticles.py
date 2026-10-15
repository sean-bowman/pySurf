
# -- pySurf: SPH Particles -- #

'''

The SPH particle state, and the functions that seed it.

`ParticleSystem` stores positions, velocities, accelerations, densities, pressures and masses as
contiguous arrays, fluid and boundary particles together, told apart by the `isFluid` mask. The
energies and diagnostics a run reports are its methods.

Seeding is two steps every scenario shares: `fillRegion` lays a square lattice over a bounding
box and keeps the points a predicate accepts (below a free surface and above a bed, say), and
`ParticleSystem.assemble` joins fluid and boundary positions into one state with masses set from
the lattice spacing.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

Nothing here is a model beyond bookkeeping. tests/testSphParticles.py checks the lattice count and
spacing, that particle mass times count reproduces the filled volume at the reference density, and
the kinetic and potential energies against hand-computed values.

Sean Bowman [02/05/2026]

'''

# Standard library imports
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# Third-party imports
import numpy as np

@dataclass
class ParticleSystem:

    '''

    SPH particle state. Vector quantities are shaped (N, dim), scalars (N,).

    Parameters:
    -----------
    positions : np.ndarray
        Particle positions [m]
    velocities : np.ndarray
        Particle velocities [m/s]
    accelerations : np.ndarray
        Particle accelerations [m/s^2]
    densities : np.ndarray
        Particle densities [kg/m^3]
    pressures : np.ndarray
        Particle pressures [Pa]
    masses : np.ndarray
        Particle masses [kg]; per unit depth [kg/m] in 2D
    isFluid : np.ndarray
        True for fluid particles, False for boundary particles

    '''

    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    densities: np.ndarray
    pressures: np.ndarray
    masses: np.ndarray
    isFluid: np.ndarray

    # -- Constructors -- #
    @classmethod
    def assemble(cls, fluidPositions: np.ndarray, boundaryPositions: np.ndarray, spacing: float,
                 referenceDensity: float, fluidVelocities: np.ndarray | None = None) -> ParticleSystem:

        '''

        One state from fluid and boundary positions laid on a lattice of the given spacing.

        Every particle carries the mass of its lattice cell at the reference density,
        rho0 * spacing^dim, and starts at the reference density with zero pressure. Boundary
        particles start at rest.

        Parameters:
        -----------
        fluidPositions : np.ndarray
            Fluid particle positions, shape (Nf, dim) [m]
        boundaryPositions : np.ndarray
            Boundary particle positions, shape (Nb, dim) [m]
        spacing : float
            Lattice spacing the positions were laid on [m]
        referenceDensity : float
            rho0 [kg/m^3]
        fluidVelocities : np.ndarray, optional
            Initial fluid velocities, shape (Nf, dim) [m/s]; zero when omitted

        '''

        fluidPositions = np.asarray(fluidPositions, dtype = float)
        boundaryPositions = np.asarray(boundaryPositions, dtype = float).reshape(-1, fluidPositions.shape[1])
        nFluid, nBoundary = len(fluidPositions), len(boundaryPositions)
        nTotal, dimensions = nFluid + nBoundary, fluidPositions.shape[1]

        velocities = np.zeros((nTotal, dimensions))
        if fluidVelocities is not None:
            velocities[:nFluid] = fluidVelocities

        return cls(
            positions     = np.vstack([fluidPositions, boundaryPositions]),
            velocities    = velocities,
            accelerations = np.zeros((nTotal, dimensions)),
            densities     = np.full(nTotal, referenceDensity),
            pressures     = np.zeros(nTotal),
            masses        = np.full(nTotal, referenceDensity * spacing**dimensions),
            isFluid       = np.concatenate([np.ones(nFluid, dtype = bool), np.zeros(nBoundary, dtype = bool)]),
        )

    @classmethod
    def createUniform(cls, domainMin: np.ndarray, domainMax: np.ndarray, spacing: float,
                      referenceDensity: float) -> ParticleSystem:

        '''A box of fluid particles filling [domainMin, domainMax] on a lattice, with no boundary.'''

        positions = latticePoints(domainMin, domainMax, spacing)

        return cls.assemble(positions, np.empty((0, len(domainMin))), spacing, referenceDensity)

    # -- Counts -- #
    @property
    def nParticles(self) -> int:
        '''Total number of particles, fluid and boundary.'''
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
        '''Number of spatial dimensions.'''
        return self.positions.shape[1]

    @property
    def fluidMask(self) -> np.ndarray:
        '''Boolean mask of the fluid particles.'''
        return self.isFluid

    @property
    def boundaryMask(self) -> np.ndarray:
        '''Boolean mask of the boundary particles.'''
        return ~self.isFluid

    # -- Diagnostics -- #
    def kineticEnergy(self) -> float:

        '''Kinetic energy of the fluid, sum of m |v|^2 / 2 [J], per unit depth [J/m] in 2D.'''

        fluid = self.isFluid

        return float(0.5 * np.sum(self.masses[fluid] * np.sum(self.velocities[fluid]**2, axis = 1)))

    def potentialEnergy(self, gravity: float) -> float:

        '''

        Gravitational potential energy of the fluid above the zero of the vertical axis [J], per
        unit depth [J/m] in 2D. The vertical axis is the last coordinate.

        '''

        fluid = self.isFluid

        return float(np.sum(self.masses[fluid] * gravity * self.positions[fluid, -1]))

    def maxSpeed(self) -> float:

        '''Largest fluid particle speed [m/s].'''

        fluid = self.isFluid
        if not np.any(fluid):
            return 0.0

        return float(np.max(np.linalg.norm(self.velocities[fluid], axis = 1)))

    def maxDensityError(self, referenceDensity: float) -> float:

        '''

        Largest relative density departure among fluid particles, max |rho - rho0| / rho0.

        Free-surface particles read low by construction, since their kernel support is partly
        empty, so this reads tens of percent even at rest; the interior density error is what
        measures compressibility.

        '''

        fluid = self.isFluid
        if not np.any(fluid):
            return 0.0

        return float(np.max(np.abs(self.densities[fluid] - referenceDensity)) / referenceDensity)

#----------------------------------------------------------------------#
# -- Seeding -- #
#----------------------------------------------------------------------#

def latticePoints(lower, upper, spacing: float) -> np.ndarray:

    '''

    A square lattice of cell centers filling the box [lower, upper], half a spacing in from each
    face.

    Parameters:
    -----------
    lower, upper : array-like
        Box corners, length 2 or 3 [m]
    spacing : float
        Lattice spacing [m]

    Returns:
    --------
    np.ndarray : Points, shape (N, dim)

    '''

    axes = [np.arange(low + spacing / 2.0, high, spacing) for low, high in zip(lower, upper)]
    grids = np.meshgrid(*axes, indexing = 'xy')

    return np.column_stack([grid.ravel() for grid in grids])

def fillRegion(inside: Callable[[np.ndarray], np.ndarray], lower, upper, spacing: float) -> np.ndarray:

    '''

    The lattice points of a bounding box that a region predicate accepts.

    Parameters:
    -----------
    inside : Callable
        Takes points shaped (N, dim) and returns a boolean mask of those inside the region
    lower, upper : array-like
        Bounding box of the region [m]
    spacing : float
        Lattice spacing [m]

    Returns:
    --------
    np.ndarray : Accepted points, shape (M, dim)

    '''

    points = latticePoints(lower, upper, spacing)

    return points[np.asarray(inside(points), dtype = bool)]
