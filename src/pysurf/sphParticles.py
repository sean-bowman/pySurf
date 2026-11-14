
# -- pySurf: SPH Particles -- #

'''

The SPH particle state, and the functions that seed it.

`ParticleSystem` stores positions, velocities, accelerations, densities, pressures and masses as
contiguous arrays, fluid and boundary particles together, told apart by the `isFluid` mask. The
energies and diagnostics a run reports are its methods.

Seeding is two steps every scenario shares: `fillRegion` lays a square lattice over a bounding
box and keeps the points a predicate accepts (below a free surface and above a bed, say), and
`ParticleSystem.assemble` joins fluid and boundary positions into one state with masses set from
the lattice spacing. Over a sloping bed `bedFittedPoints` lays columns that stand on the bed, and
`jitterPoints` moves laid points off their lattice, so the water starts without ordered rows.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

Nothing here is a model beyond bookkeeping. tests/testSphParticles.py checks the lattice count and
spacing, that particle mass times count reproduces the filled volume at the reference density, and
the kinetic and potential energies against hand-computed values. It checks that jittered points
keep their count, stay inside the walls, the floor's line and the ceiling, move no more than the
amplitude along a horizontal axis, and repeat with their seed.

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
    lengths : np.ndarray | None
        Each particle's smoothing length h [m], once particles have split
        (`pysurf.sphRefinement`); None while every particle has the solver's

    '''

    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    densities: np.ndarray
    pressures: np.ndarray
    masses: np.ndarray
    isFluid: np.ndarray
    lengths: np.ndarray | None = None

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

    # -- Growth -- #
    def append(self, positions: np.ndarray, velocities: np.ndarray, densities: np.ndarray, mass: float,
               isFluid: bool = True, length: float | None = None) -> np.ndarray:

        '''

        Add particles at the end of the state, all of one mass and kind, with zero acceleration and
        zero pressure until the solver next sets them.

        Parameters:
        -----------
        positions, velocities : np.ndarray
            Shape (M, dim) [m], [m/s]
        densities : np.ndarray
            Shape (M,) [kg/m^3]
        mass : float
            Mass of each new particle [kg]
        isFluid : bool
            Whether the new particles are fluid
        length : float | None
            Smoothing length of each new particle [m]; needed once the particles carry their own

        Returns:
        --------
        np.ndarray : Indices of the new particles

        '''

        count = len(positions)
        start = self.nParticles
        self.positions = np.vstack([self.positions, positions])
        self.velocities = np.vstack([self.velocities, velocities])
        self.accelerations = np.vstack([self.accelerations, np.zeros_like(positions)])
        self.densities = np.concatenate([self.densities, densities])
        self.pressures = np.concatenate([self.pressures, np.zeros(count)])
        self.masses = np.concatenate([self.masses, np.full(count, mass)])
        self.isFluid = np.concatenate([self.isFluid, np.full(count, isFluid)])
        if self.lengths is not None:
            if length is None:
                raise ValueError('These particles carry their own smoothing lengths; give the new ones theirs.')
            self.lengths = np.concatenate([self.lengths, np.full(count, float(length))])

        return np.arange(start, start + count)

    def smoothingLengths(self, reference: float) -> np.ndarray:

        '''Every particle's smoothing length [m]: its own, or `reference` while they share one.'''

        return self.lengths if self.lengths is not None else np.full(self.nParticles, float(reference))

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

def bedFittedPoints(lower, upper, spacing: float, floor: Callable[..., np.ndarray]) -> np.ndarray:

    '''

    Columns of points standing on a floor: at `latticePoints`' horizontal positions in the box
    [lower, upper], each column from half a spacing above the floor up to the box's top, one
    spacing apart.

    Next to a sloping bed the square lattice leaves its lowest point anywhere from half a spacing
    to one and a half above the bed, in steps along the slope. The bed's pressure does not hold
    that arrangement at rest: still water laid on it falls into the steps and keeps moving. These
    columns keep the gap at half a spacing everywhere and leave the uneven gap at the top, where a
    free surface carries no pressure. Over a flat floor at the box's bottom they are the lattice's
    points to rounding, in column order.

    Parameters:
    -----------
    lower, upper : array-like
        Box corners, length 2 or 3 [m]; the last coordinate is vertical
    spacing : float
        Point spacing [m]
    floor : Callable
        The floor's elevation [m], from the horizontal coordinates, one array each: floor(x) in
        2D, floor(x, y) in 3D. A floor below the box's bottom counts as the bottom.

    Returns:
    --------
    np.ndarray : Points, shape (N, dim)

    '''

    lower, upper = np.asarray(lower, dtype = float), np.asarray(upper, dtype = float)
    axes = [np.arange(low + spacing / 2.0, high, spacing) for low, high in zip(lower[:-1], upper[:-1])]
    horizontal = [grid.ravel() for grid in np.meshgrid(*axes, indexing = 'ij')]
    bottom = np.maximum(np.asarray(floor(*horizontal), dtype = float), lower[-1]) + spacing / 2.0

    # Counted as np.arange counts, so a column on the box's bottom holds as many points as the lattice's
    counts = np.maximum(np.ceil((upper[-1] - bottom) / spacing), 0.0).astype(int)
    column = np.repeat(np.arange(bottom.size), counts)
    layer = np.arange(column.size) - np.repeat(np.cumsum(counts) - counts, counts)

    return np.column_stack([coordinate[column] for coordinate in horizontal] + [bottom[column] + layer * spacing])

def jitterPoints(points: np.ndarray, spacing: float, amplitude: float, seed: int, lower, upper,
                 floor: Callable[..., np.ndarray], ceiling: Callable[..., np.ndarray] | None = None) -> np.ndarray:

    '''

    Points moved off their lattice: each by a seeded uniform random offset of up to `amplitude`
    spacings along every axis, then reflected back across any face it crossed. The faces are the
    walls half a spacing in from the box [lower, upper] along each horizontal axis, the line half a
    spacing above the floor, and the ceiling (a free surface). A point the reflections leave outside,
    in water thinner than its offset, keeps its place.

    Water laid in ordered rows keeps them: under a steepening wave the rows buckle together and the
    face kinks where they end, which delays or advances the breaking point at a coarse spacing.
    Moving each point off the lattice removes the order and keeps every point, so the water's volume
    is the ordered start's. No point moves more than `amplitude` spacings along a horizontal axis;
    vertically, a reflection from a sloping floor or surface adds up to twice that face's rise over
    the point's horizontal move.

    Parameters:
    -----------
    points : np.ndarray
        Points inside the water, shape (N, dim); the last coordinate is vertical [m]
    spacing : float
        Point spacing [m]
    amplitude : float
        Largest offset along each axis, in spacings; zero returns the points unchanged
    seed : int
        Seed of the random offsets
    lower, upper : array-like
        The walls along each horizontal axis, length dim - 1 [m]
    floor : Callable
        The floor's elevation [m], from the horizontal coordinates, one array each: floor(x) in
        2D, floor(x, y) in 3D
    ceiling : Callable | None
        The free surface's elevation [m], called as `floor` is; None for no surface to reflect from

    Returns:
    --------
    np.ndarray : The moved points, shape (N, dim)

    '''

    points = np.asarray(points, dtype = float)
    if amplitude <= 0.0:
        return points.copy()
    moved = points + np.random.default_rng(seed).uniform(-amplitude, amplitude, points.shape) * spacing
    horizontal = points.shape[1] - 1
    lower, upper = np.asarray(lower, dtype = float), np.asarray(upper, dtype = float)

    def reflect(values: np.ndarray, low: np.ndarray | float, high: np.ndarray | float) -> np.ndarray:
        values = np.where(values < low, 2.0 * low - values, values)
        return np.where(values > high, 2.0 * high - values, values)

    for axis in range(horizontal):
        moved[:, axis] = reflect(moved[:, axis], lower[axis] + 0.5 * spacing, upper[axis] - 0.5 * spacing)
    bottom = np.asarray(floor(*moved[:, :horizontal].T), dtype = float) + 0.5 * spacing
    top = (np.asarray(ceiling(*moved[:, :horizontal].T), dtype = float) if ceiling is not None
           else np.full(len(moved), np.inf))
    # The surface excludes its own height, as the seeding functions' tests do
    moved[:, -1] = reflect(moved[:, -1], bottom, top - 1e-9 * spacing)

    walls = np.all((moved[:, :horizontal] >= lower + (0.5 - 1e-9) * spacing)
                   & (moved[:, :horizontal] <= upper - (0.5 - 1e-9) * spacing), axis = 1)
    stray = ~walls | (moved[:, -1] < bottom - 1e-9 * spacing) | (moved[:, -1] >= top)
    moved[stray] = points[stray]

    return moved

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
