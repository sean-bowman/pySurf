
# -- pySurf: SPH Particle Packing -- #

'''

Laid particles relaxed into an arrangement the SPH pressure sum holds at rest, by the packing
algorithm of Colagrossi et al. (2012).

At rest the SPH pressure gradient of particle i, sum_j (p_i + p_j) grad W_ij V_j, is to first
order Gamma_i grad p + 2 p_i grad Gamma_i, where Gamma_i = sum_j W_ij V_j and
grad Gamma_i = sum_j grad W_ij V_j. Inside water laid on a lattice grad Gamma is zero; inside water
whose particles sit off it, it is not, and at depth, where p is large, 2 p grad Gamma sets the
particles moving where the water should rest. The algorithm moves every particle down grad Gamma
under a fictitious uniform pressure p0, with a linear damping:

    du_i/dt = -beta grad Gamma_i - nu u_i,    dr_i/dt = u_i,    beta = 2 p0 / rho0

with nu = alpha sqrt(beta) / V0^(1/d) and the step dt = V0^(1/d) / sqrt(beta), V0 the volume
each particle holds. The particles stop where grad Gamma vanishes: spread as evenly as the kernel
can tell. A lattice is such an arrangement, and the packing restores order rather than removing
it: particles jittered off a lattice by a fraction of a spacing return most of the way to it
(measured in `pysurf.BreakingWave`).

    volume      V0 = s^d, the volume of a lattice point, so the packed water keeps the masses
                `ParticleSystem.assemble` gives it
    walls, bed  the run's own boundary particles, fixed, so the water packs against what the run
                will hold it with. Where the bed bends its layers leave gaps, and a particle that
                enters one meets the bed's even lattice, which holds it nowhere; given the bed's
                elevation, a point is held a quarter spacing clear of it, as the run holds it
                (`sphBoundaries.BedBoundary.enforce`), its velocity into the bed removed
    surface     a free surface leaves the support of a particle near it half empty, and grad Gamma
                would push the particle out. Colagrossi et al. close the surface with fixed
                particles. Here the space above the known surface counts as full: Gamma_i gains
                Phi(r_i), the kernel's integral over that space, and by the divergence theorem
                grad Phi = integral over the surface of W n dA, with n dA = (-grad eta, 1) dx (dy),
                summed over the surface table's nodes (`surfaceGradient`). Phi depends on a
                particle's place alone, so the packing descends one energy and comes to rest.
                Boundary particles above the surface leave the fixed set, since Phi fills that
                space. A particle that crosses the surface is reflected back across it, its
                vertical velocity reversed. Images of the particles mirrored across the surface
                would carry the water's own arrangement above it, but no mirror of a sloped or
                curved surface keeps distances, and the forces across it then lose their
                symmetry: under the crest of a solitary wave of H/h = 0.6 the residual of
                vertically mirrored images grows instead of falling, and particles stream along
                the surface without end
    shallows    where the water is shallower than the support and a spacing, a particle at the
                water's tip can be pushed along the surface into the dry bed, and once inside the
                bed's even lattice nothing holds it back. Given the bed's elevation, points in such
                water keep their laid places and count as fixed
    stop        when the residual, the rms of h |grad Gamma| (Phi included) over the water, falls below
                `tolerance` after the first `minimumIterations` steps, or after `iterations` steps.
                It is read every 50 steps, from the sums that step moved the points by

The surface and the bed are read from tables at half a spacing (`SurfaceTable`), linearly
interpolated.

Two backends run the same steps: 'numpy', with a k-d tree each step, for tests and small sets;
'gpu', in NVIDIA Warp on its own hash grid, with the float64 positions split into a float32 point
and remainder as `pysurf.sphGpu` splits them.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

Not a physical model: a relaxation of the particles' places before a run. tests/testSphPacking.py
checks that the surface table reads plane and bilinear surfaces exactly; that grad Phi matches a
direct quadrature of the kernel's gradient over the space above a solitary wave's crest, and on a
tilted plane in 2D and 3D equals the flat surface's term turned to the plane's normal; that
grad Gamma vanishes inside an even lattice, and a lattice under a flat surface starts within the
packing's tolerance and barely moves; that jittered still water under a flat and a sloping
surface, and jittered water under the crest of a solitary wave of H/h = 0.6, pack to a residual at
least ten times smaller, every point inside the walls, above the floor and below the surface; that
points in the shallows keep their places; and that the device backend moves the points as the
array sum does. What a packed start does to a run is measured in `pysurf.BreakingWave`.

References:
-----------
Colagrossi, A., Bouscasse, B., Antuono, M. & Marrone, S. (2012) -- Particle packing algorithm for
    SPH schemes, Comput. Phys. Commun. 183, 1641-1653. The algorithm, its damping and time step
    as given in the conference version, Bouscasse, B., Antuono, M., Marrone, S. & Colagrossi, A.
    (2011) -- Particle initialization through a novel packing algorithm, 6th SPHERIC Workshop,
    Hamburg, 109-116.

Sean Bowman [11/13/2026]

'''

# Standard library imports
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

# Third-party imports
import numpy as np
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree

# Local imports
from pysurf.sphKernels import createKernel
from pysurf.sphNeighbors import neighborPairs

checkInterval = 50          # steps between readings of the residual, the stopping test
bedClearance = 0.25         # [spacings] the least height of a point above the bed, as `BedBoundary.enforce` holds it

@dataclass
class SurfaceTable:

    '''

    A free surface's elevation on a regular grid over the horizontal axes, read by linear
    (bilinear in 3D) interpolation and held at its edge value beyond the grid.

    Parameters:
    -----------
    start : np.ndarray
        The grid's first node along each horizontal axis [m]
    step : np.ndarray
        Node spacing along each horizontal axis [m]
    values : np.ndarray
        Elevation at the nodes [m], shape (nx,) in 2D or (nx, ny) in 3D

    '''

    start: np.ndarray
    step: np.ndarray
    values: np.ndarray

    @classmethod
    def sample(cls, surface: Callable[..., np.ndarray], lower, upper, step: float) -> SurfaceTable:

        '''The table of `surface` (eta(x) in 2D, eta(x, y) in 3D) over the box [lower, upper] at `step`.'''

        lower, upper = np.atleast_1d(np.asarray(lower, dtype = float)), np.atleast_1d(np.asarray(upper, dtype = float))
        axes = [low + step * np.arange(max(int(math.ceil((high - low) / step)) + 1, 2)) for low, high in zip(lower, upper)]
        grids = np.meshgrid(*axes, indexing = 'ij')
        values = np.asarray(surface(*grids), dtype = float) * np.ones(grids[0].shape)

        return cls(lower.copy(), np.full(len(lower), float(step)), values)

    def elevation(self, *coordinates: np.ndarray) -> np.ndarray:

        '''The surface's elevation at the horizontal coordinates, one array each [m].'''

        weights, corners = [], []
        for axis, coordinate in enumerate(coordinates):
            u = np.clip((np.asarray(coordinate, dtype = float) - self.start[axis]) / self.step[axis], 0.0,
                        self.values.shape[axis] - 1.0)
            low = np.minimum(np.floor(u).astype(int), self.values.shape[axis] - 2)
            corners.append(low)
            weights.append(u - low)
        if len(coordinates) == 1:
            (low,), (w,) = corners, weights
            return (1.0 - w) * self.values[low] + w * self.values[low + 1]
        (lowX, lowY), (wx, wy) = corners, weights
        v = self.values

        return ((1.0 - wx) * ((1.0 - wy) * v[lowX, lowY] + wy * v[lowX, lowY + 1])
                + wx * ((1.0 - wy) * v[lowX + 1, lowY] + wy * v[lowX + 1, lowY + 1]))

    def slopes(self) -> list:

        '''grad eta at the nodes, one array per horizontal axis, by central differences (one-sided at the edges).'''

        slopes = np.gradient(self.values, *self.step)

        return [slopes] if self.values.ndim == 1 else list(slopes)

    def elements(self) -> tuple:

        '''

        The surface's nodes, shape (K, dim), and each node's outward normal times its share of
        horizontal area, (-grad eta, 1) dx (dy), shape (K, dim) [m^(dim - 1)].

        '''

        axes = [self.start[axis] + self.step[axis] * np.arange(count) for axis, count in enumerate(self.values.shape)]
        grids = np.meshgrid(*axes, indexing = 'ij')
        nodes = np.column_stack([grid.ravel() for grid in grids] + [self.values.ravel()])
        normals = np.column_stack([-slope.ravel() for slope in self.slopes()] + [np.ones(self.values.size)])

        return nodes, normals * float(np.prod(self.step))

    def lowest(self, reach: float) -> SurfaceTable:

        '''

        The table of the lowest node within `reach` and a node of each node, along every horizontal
        axis. Read at any point, it lies at or below every node within `reach` of the point, so a
        point more than `reach` below it has no node within `reach`.

        '''

        size = [2 * (int(math.ceil(reach / step)) + 1) + 1 for step in self.step]

        return SurfaceTable(self.start.copy(), self.step.copy(), minimum_filter(self.values, size = size, mode = 'nearest'))

@dataclass
class PackingResult:

    '''

    The packed points and how the packing went.

    Parameters:
    -----------
    positions : np.ndarray
        The packed points, shape (N, dim) [m]
    iterations : int
        Steps taken
    kineticEnergy : float
        The specific kinetic energy at the last step, mean |u|^2 / 2 over beta [-]
    gradientBefore, gradientAfter : tuple
        rms and largest h |grad Gamma| over the points, before and after [-]
    largestMove : float
        The largest distance a point moved, in spacings
    frozen : int
        Points that kept their places, in water shallower than the support and a spacing

    '''

    positions: np.ndarray
    iterations: int
    kineticEnergy: float
    gradientBefore: tuple
    gradientAfter: tuple
    largestMove: float
    frozen: int = 0

def packingGradient(points: np.ndarray, others: np.ndarray, volume: float, smoothingLength: float,
                    kernelType: str = 'wendlandC2', shares: np.ndarray | None = None) -> np.ndarray:

    '''

    grad Gamma_i = sum_j grad W_ij V_j for each of `points`, over the points and `others` [1/m],
    shape (N, dim): the array sum the device kernel reproduces.

    Parameters:
    -----------
    points : np.ndarray
        The points the sum is taken for, shape (N, dim) [m]
    others : np.ndarray
        Further points in the sums that take none themselves, the fixed particles, shape (M, dim) [m]
    volume : float
        V0 [m^dim]
    smoothingLength : float
        h [m]; the support is 2h
    kernelType : str
        'wendlandC2' or 'cubicSpline'
    shares : np.ndarray | None
        Each of `others`' volume as a share of V0, shape (M,); every one whole when None

    '''

    count, dims = points.shape
    others = np.asarray(others, dtype = float).reshape(-1, dims)
    everything = np.vstack([points, others])
    volumes = volume * np.concatenate([np.ones(count), np.ones(len(others)) if shares is None else np.asarray(shares)])
    i, j = neighborPairs(everything, 2.0 * smoothingLength)
    separation = everything[i] - everything[j]
    distance = np.linalg.norm(separation, axis = 1)
    slopes = createKernel(kernelType, dims).gradientBatch(separation, distance, smoothingLength)
    gradient = np.zeros((len(everything), dims))
    # grad_i W_ij = -grad_j W_ij: each pair adds to i with j's volume and takes from j with i's
    for axis in range(dims):
        gradient[:, axis] = (np.bincount(i, volumes[j] * slopes[:, axis], len(everything))
                             - np.bincount(j, volumes[i] * slopes[:, axis], len(everything)))

    return gradient[:count]

def surfaceGradient(points: np.ndarray, table: SurfaceTable, smoothingLength: float, kernelType: str = 'wendlandC2',
                    lowest: SurfaceTable | None = None) -> np.ndarray:

    '''

    grad Phi at each point [1/m], shape (N, dim): Phi is the kernel's integral over the space above
    the surface, and grad Phi = sum over the table's nodes of W(|r - node|) (-grad eta, 1) dx (dy),
    the divergence theorem's surface integral. It points up, toward the surface; the packing moves
    a point down it.

    Parameters:
    -----------
    points : np.ndarray
        Shape (N, dim) [m]
    table : SurfaceTable
        The surface
    smoothingLength : float
        h [m]; the support is 2h
    kernelType : str
        'wendlandC2' or 'cubicSpline'
    lowest : SurfaceTable | None
        `table.lowest(2h)`, to pass over points deeper than the support; made here when None

    '''

    points = np.asarray(points, dtype = float)
    count, dims = points.shape
    support = 2.0 * smoothingLength
    lowest = lowest if lowest is not None else table.lowest(support)
    gradient = np.zeros((count, dims))
    near = np.flatnonzero(points[:, -1] >= lowest.elevation(*points[:, :-1].T) - support)
    if len(near) == 0:
        return gradient
    nodes, normals = table.elements()
    found = cKDTree(nodes).query_ball_point(points[near], support)
    rows = np.repeat(np.arange(len(near)), [len(entry) for entry in found])
    columns = np.concatenate([np.asarray(entry, dtype = int) for entry in found])
    distance = np.linalg.norm(points[near][rows] - nodes[columns], axis = 1)
    inside = distance < support
    rows, columns = rows[inside], columns[inside]
    values = createKernel(kernelType, dims).evaluateBatch(distance[inside], smoothingLength)
    for axis in range(dims):
        gradient[near, axis] = np.bincount(rows, values * normals[columns, axis], len(near))

    return gradient

def packPoints(points: np.ndarray, fixed: np.ndarray, spacing: float, surface: Callable[..., np.ndarray],
               floor: Callable[..., np.ndarray] | None = None, smoothingLengthRatio: float = 3.0,
               kernelType: str = 'wendlandC2', damping: float = 0.05,
               iterations: int = 4000, minimumIterations: int = 100, tolerance: float = 1e-4,
               backend: str = 'numpy', device: str | None = None) -> PackingResult:

    '''

    Points relaxed down grad Gamma + grad Phi between fixed particles and below a free surface (see
    the module docstring).

    Parameters:
    -----------
    points : np.ndarray
        The water's points, shape (N, dim); the last coordinate is vertical [m]
    fixed : np.ndarray
        The boundary's particles, which do not move, shape (M, dim) [m]; those above the surface
        take no part
    spacing : float
        Particle spacing s; V0 = s^dim [m]
    surface : Callable
        The free surface's elevation [m], from the horizontal coordinates, one array each:
        eta(x) in 2D, eta(x, y) in 3D
    floor : Callable | None
        The bed's elevation [m], called as `surface` is; points in water shallower than the
        support and a spacing keep their places, and the rest stay a quarter spacing above it.
        None packs every point and holds none
    smoothingLengthRatio : float
        h / s
    kernelType : str
        'wendlandC2' or 'cubicSpline'
    damping : float
        alpha, the damping per step: nu dt
    iterations : int
        Most steps to take
    minimumIterations : int
        Fewest steps before the residual may stop the packing
    tolerance : float
        The rms of h |grad Gamma| over the water below which the packing stops
    backend : str
        'numpy' or 'gpu'
    device : str | None
        The Warp device for the 'gpu' backend, 'cuda:0' or 'cpu'; None for Warp's preferred device

    Returns:
    --------
    PackingResult : The packed points, the steps taken, and the residuals before and after, over
        the points that move

    '''

    points = np.asarray(points, dtype = float)
    fixed = np.asarray(fixed, dtype = float).reshape(-1, points.shape[1])
    if backend not in ('numpy', 'gpu'):
        raise ValueError(f"Packing backend '{backend}' is not 'numpy' or 'gpu'.")
    if len(points) == 0:
        return PackingResult(points.copy(), 0, 0.0, (0.0, 0.0), (0.0, 0.0), 0.0)
    h = smoothingLengthRatio * spacing
    support = 2.0 * h
    lower, upper = points.min(axis = 0) - support, points.max(axis = 0) + support
    table = SurfaceTable.sample(surface, lower[:-1] - spacing, upper[:-1] + spacing, 0.5 * spacing)
    bed = (SurfaceTable.sample(floor, lower[:-1] - spacing, upper[:-1] + spacing, 0.5 * spacing) if floor is not None
           else None)
    horizontal = points[:, :-1].T
    top = table.elevation(*horizontal)
    thin = (top - np.asarray(floor(*horizontal), dtype = float) < support + spacing if floor is not None
            else np.zeros(len(points), dtype = bool))
    # Only the fixed particles the water can reach take part; the shallows' points join them
    near = np.all((fixed >= lower - spacing) & (fixed <= upper + spacing), axis = 1)
    fixed, shares = _belowSurface(np.vstack([fixed[near], points[thin]]), table, spacing)
    movable = points[~thin]
    positions = points.copy()
    if len(movable) == 0:
        return PackingResult(positions, 0, 0.0, (0.0, 0.0), (0.0, 0.0), 0.0, int(thin.sum()))
    settings = dict(volume = spacing**points.shape[1], h = h, spacing = spacing, kernelType = kernelType,
                    damping = damping, iterations = iterations, minimumIterations = minimumIterations,
                    tolerance = tolerance)
    if backend == 'numpy':
        packed, steps, energy, before, after = _packNumpy(movable, fixed, shares, table, bed, **settings)
    else:
        from pysurf.sphPackingGpu import packOnDevice
        packed, steps, energy, before, after = packOnDevice(movable, fixed, shares, table, bed, device = device, **settings)
    positions[~thin] = packed
    moved = float(np.max(np.linalg.norm(positions - points, axis = 1))) / spacing

    return PackingResult(positions, steps, energy, before, after, moved, int(thin.sum()))

def residual(gradient: np.ndarray, h: float) -> tuple:

    '''rms and largest h |grad Gamma| [-].'''

    size = h * np.linalg.norm(gradient, axis = 1)

    return float(np.sqrt(np.mean(size**2))), float(size.max())

def _belowSurface(fixed: np.ndarray, table: SurfaceTable, spacing: float) -> tuple:

    '''

    The fixed particles as the packing counts them, and the share of V0 each holds: a particle
    stands for its cell, and where the surface cuts the cell only the part below it counts, at that
    part's centre, so a wall meets Phi without a gap, which would draw points in, or an overlap.
    Cells wholly above the surface drop out.

    '''

    base = fixed[:, -1] - 0.5 * spacing
    share = np.clip((table.elevation(*fixed[:, :-1].T) - base) / spacing, 0.0, 1.0)
    # A share within rounding of zero is a cell resting on a surface laid along the cell edges
    keep = share > 1e-9
    counted = fixed[keep].copy()
    share, base = share[keep], base[keep]
    counted[:, -1] = np.where(share < 1.0, base + 0.5 * share * spacing, counted[:, -1])

    return counted, share

def _packNumpy(points: np.ndarray, fixed: np.ndarray, shares: np.ndarray, table: SurfaceTable, bed: SurfaceTable | None,
               volume: float, h: float, spacing: float, kernelType: str, damping: float, iterations: int,
               minimumIterations: int, tolerance: float) -> tuple:

    '''The packing's steps on the host; returns positions, steps, kinetic energy and the two residuals.'''

    positions = points.copy()
    velocities = np.zeros_like(positions)
    beta = 1.0                                  # [m^2/s^2], sets only the pseudo-time's scale
    timeStep = spacing / math.sqrt(beta)        # [s]
    nu = damping / timeStep                     # [1/s]
    lowest = table.lowest(2.0 * h)

    def gradient() -> np.ndarray:
        return (packingGradient(positions, fixed, volume, h, kernelType, shares)
                + surfaceGradient(positions, table, h, kernelType, lowest))

    current = gradient()
    before = residual(current, h)
    energy, step = 0.0, 0
    for step in range(1, iterations + 1):
        velocities += timeStep * (-beta * current - nu * velocities)
        positions += timeStep * velocities
        _reflectBelow(positions, velocities, table)
        if bed is not None:
            _holdAbove(positions, velocities, bed, bedClearance * spacing)
        if step % checkInterval == 0 or step == iterations:
            energy = float(np.mean(np.sum(velocities**2, axis = 1))) / (2.0 * beta)
            if step >= minimumIterations and residual(current, h)[0] < tolerance:
                current = gradient()
                break
        current = gradient()

    return positions, step, energy, before, residual(current, h)

def _holdAbove(positions: np.ndarray, velocities: np.ndarray, bed: SurfaceTable, clearance: float) -> None:

    '''A point closer to the bed than `clearance` returns to that height, its velocity into the bed removed.'''

    bottom = bed.elevation(*positions[:, :-1].T) + clearance
    under = positions[:, -1] < bottom
    positions[under, -1] = bottom[under]
    velocities[under, -1] = np.maximum(velocities[under, -1], 0.0)

def _reflectBelow(positions: np.ndarray, velocities: np.ndarray, table: SurfaceTable) -> None:

    '''A point above the surface returns across it, its vertical velocity reversed.'''

    top = table.elevation(*positions[:, :-1].T)
    above = positions[:, -1] > top
    positions[above, -1] = 2.0 * top[above] - positions[above, -1]
    velocities[above, -1] = -velocities[above, -1]
