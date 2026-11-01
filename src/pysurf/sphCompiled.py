
# -- pySurf: Compiled SPH Kernels -- #

'''

The per-step sums of `pysurf.sphSolver`, compiled with numba and run across threads, for runs too
large for the array formulation: three-dimensional water with hundreds of thousands of particles.

    neighborList        every particle's neighbors inside the support, from a cell-linked list,
                        as compressed rows: neighbors of i are neighbors[offsets[i]:offsets[i + 1]]
    pairAccelerations   pressure gradient, artificial viscosity and gravity on the fluid
    continuityRate      d rho / dt for every particle
    xsphVelocity        the XSPH transport velocity of the fluid
    shepardDensity      the Shepard-filtered density

The equations are those of `pysurf.sphSolver`, term for term, with the cubic spline kernel. The
array formulation visits each pair once and adds its contribution to both members; here every
particle sums over its own neighbors, so each pair is evaluated from both sides. That costs twice
the arithmetic but no two threads ever write to the same particle, and no per-pair arrays are
stored, so memory grows with the particle count alone. Sums run in a different order than the
array formulation's, which changes results by rounding only.

A periodic axis is handled as in `sphNeighbors.neighborPairs`: distances along it are taken to the
nearest image, which needs the period longer than twice the support.

The number of threads is numba's: NUMBA_NUM_THREADS, or `numba.set_num_threads`.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

Nothing here is a new model. tests/testSphCompiled.py checks every function against its array
counterpart in `pysurf.sphSolver` and `pysurf.sphNeighbors` on random particle clouds in 2D and
3D, with and without a periodic axis, to rounding, and a short solver run against the array
backend.

Sean Bowman [10/31/2026]

'''

# Standard library imports
from __future__ import annotations

import math

# Third-party imports
import numpy as np
from numba import njit, prange

viscositySoftening = 0.01           # [-] as in sphSolver: mu_ij's denominator r^2 + 0.01 h^2

def cubicSplineNormalization(smoothingLength: float, dimensions: int) -> float:

    '''The cubic spline's sigma: 10 / (7 pi h^2) in 2D, 1 / (pi h^3) in 3D.'''

    h = smoothingLength
    return 10.0 / (7.0 * math.pi * h * h) if dimensions == 2 else 1.0 / (math.pi * h * h * h)

@njit(cache = True, inline = 'always')
def _kernel(r: float, h: float, sigma: float) -> tuple[float, float]:

    '''The cubic spline W(r) and dW/dr.'''

    q = r / h
    if q < 1.0:
        return sigma * (1.0 - 1.5 * q * q + 0.75 * q * q * q), sigma * (-3.0 * q + 2.25 * q * q) / h
    if q < 2.0:
        rest = 2.0 - q
        return sigma * 0.25 * rest * rest * rest, sigma * (-0.75 * rest * rest) / h
    return 0.0, 0.0

@njit(cache = True, inline = 'always')
def _separation(positions: np.ndarray, i: int, j: int, period: float, periodicAxis: int) -> tuple:

    '''r_i - r_j as (dx, dy, dz), dz zero in 2D, the periodic component to its nearest image.'''

    dx = positions[i, 0] - positions[j, 0]
    dy = positions[i, 1] - positions[j, 1]
    dz = positions[i, 2] - positions[j, 2] if positions.shape[1] == 3 else 0.0
    if period > 0.0:
        if periodicAxis == 0:
            dx -= period * np.round(dx / period)
        elif periodicAxis == 1:
            dy -= period * np.round(dy / period)
        else:
            dz -= period * np.round(dz / period)

    return dx, dy, dz

@njit(cache = True)
def _scan(i: int, positions: np.ndarray, radiusSquared: float, period: float, periodicAxis: int,
          cellOf: np.ndarray, start: np.ndarray, order: np.ndarray, cells: np.ndarray, steps: np.ndarray,
          stepCount: np.ndarray, neighbors: np.ndarray, write: int, fill: bool) -> int:

    '''Visit the particles of i's cell and the adjacent ones; count its neighbors, and write them when `fill`.'''

    home = cellOf[i]
    c2 = home % cells[2]
    c1 = (home // cells[2]) % cells[1]
    c0 = home // (cells[2] * cells[1])
    found = 0
    for a in range(stepCount[0]):
        n0 = c0 + steps[0, a]
        if period > 0.0 and periodicAxis == 0:
            n0 = n0 % cells[0]
        elif n0 < 0 or n0 >= cells[0]:
            continue
        for b in range(stepCount[1]):
            n1 = c1 + steps[1, b]
            if period > 0.0 and periodicAxis == 1:
                n1 = n1 % cells[1]
            elif n1 < 0 or n1 >= cells[1]:
                continue
            for e in range(stepCount[2]):
                n2 = c2 + steps[2, e]
                if period > 0.0 and periodicAxis == 2:
                    n2 = n2 % cells[2]
                elif n2 < 0 or n2 >= cells[2]:
                    continue
                cell = (n0 * cells[1] + n1) * cells[2] + n2
                for k in range(start[cell], start[cell + 1]):
                    j = order[k]
                    if j == i:
                        continue
                    dx, dy, dz = _separation(positions, i, j, period, periodicAxis)
                    if dx * dx + dy * dy + dz * dz < radiusSquared:
                        if fill:
                            neighbors[write + found] = j
                        found += 1

    return found

@njit(parallel = True, cache = True)
def neighborList(positions: np.ndarray, radius: float, period: float = 0.0,
                 periodicAxis: int = 1) -> tuple[np.ndarray, np.ndarray]:

    '''

    Every particle's neighbors closer than `radius`, itself excluded, as compressed rows.

    Particles are sorted into cubic cells one radius across (along a periodic axis, the period
    divided into whole cells at least that size), and each particle tests the particles of its own
    and the adjacent cells.

    Parameters:
    -----------
    positions : np.ndarray
        Particle positions, shape (N, dim) [m]
    radius : float
        Search radius [m]
    period : float
        Length of the periodic axis [m]; 0 for none
    periodicAxis : int
        Index of the periodic axis

    Returns:
    --------
    tuple[np.ndarray, np.ndarray] : (offsets, neighbors), offsets shape (N + 1,), neighbors of
        particle i at neighbors[offsets[i]:offsets[i + 1]]

    '''

    count, dim = positions.shape
    lower = np.empty(3)
    cells = np.ones(3, dtype = np.int64)
    size = np.full(3, radius)
    for axis in range(dim):
        low, high = positions[0, axis], positions[0, axis]
        for i in range(count):
            low = min(low, positions[i, axis])
            high = max(high, positions[i, axis])
        if period > 0.0 and axis == periodicAxis:
            lower[axis] = 0.0
            cells[axis] = max(int(period // radius), 1)
            size[axis] = period / cells[axis]
        else:
            lower[axis] = low
            cells[axis] = int((high - low) // radius) + 1

    # Cell of every particle, then a counting sort by cell
    cellOf = np.empty(count, dtype = np.int64)
    for i in prange(count):
        index = 0
        for axis in range(3):
            c = 0
            if axis < dim:
                value = positions[i, axis] - lower[axis]
                if period > 0.0 and axis == periodicAxis:
                    value = value - period * math.floor(value / period)
                c = min(max(int(value / size[axis]), 0), cells[axis] - 1)
            index = index * cells[axis] + c
        cellOf[i] = index
    totalCells = cells[0] * cells[1] * cells[2]
    start = np.zeros(totalCells + 1, dtype = np.int64)
    for i in range(count):
        start[cellOf[i] + 1] += 1
    for c in range(totalCells):
        start[c + 1] += start[c]
    fill = start[:-1].copy()
    order = np.empty(count, dtype = np.int64)
    for i in range(count):
        order[fill[cellOf[i]]] = i
        fill[cellOf[i]] += 1

    # The distinct neighboring cell offsets along each axis: a periodic axis of one or two cells
    # would otherwise visit the same cell twice
    steps = np.zeros((3, 3), dtype = np.int64)
    stepCount = np.ones(3, dtype = np.int64)
    for axis in range(3):
        if axis >= dim:
            continue
        if period > 0.0 and axis == periodicAxis and cells[axis] < 3:
            stepCount[axis] = cells[axis]
            for k in range(cells[axis]):
                steps[axis, k] = k
        else:
            stepCount[axis] = 3
            steps[axis, 0], steps[axis, 1], steps[axis, 2] = -1, 0, 1

    # Count each particle's neighbors, then lay them out in rows and fill them
    counts = np.zeros(count, dtype = np.int64)
    scratch = np.empty(0, dtype = np.int32)
    for i in prange(count):
        counts[i] = _scan(i, positions, radius * radius, period, periodicAxis, cellOf, start, order, cells,
                          steps, stepCount, scratch, 0, False)
    offsets = np.zeros(count + 1, dtype = np.int64)
    for i in range(count):
        offsets[i + 1] = offsets[i] + counts[i]
    neighbors = np.empty(offsets[count], dtype = np.int32)
    for i in prange(count):
        _scan(i, positions, radius * radius, period, periodicAxis, cellOf, start, order, cells, steps, stepCount,
              neighbors, offsets[i], True)

    return offsets, neighbors

@njit(parallel = True, cache = True)
def pairAccelerations(positions: np.ndarray, velocities: np.ndarray, densities: np.ndarray,
                      pressures: np.ndarray, masses: np.ndarray, isFluid: np.ndarray, offsets: np.ndarray,
                      neighbors: np.ndarray, h: float, sigma: float, soundSpeed: float, alpha: float,
                      boundaryViscosityFactor: float, gravity: np.ndarray, period: float = 0.0,
                      periodicAxis: int = 1) -> np.ndarray:

    '''

    a_i = -sum_j m_j (p_i / rho_i^2 + p_j / rho_j^2 + Pi_ij) grad W_ij + g on the fluid, zero on
    the boundary, with Monaghan's Pi_ij scaled by `boundaryViscosityFactor` for pairs with a
    boundary particle [m/s^2].

    '''

    count, dim = positions.shape
    result = np.zeros((count, dim))
    softening = viscositySoftening * h * h
    for i in prange(count):
        if not isFluid[i]:
            continue
        own = pressures[i] / (densities[i] * densities[i])
        ax, ay, az = 0.0, 0.0, 0.0
        for k in range(offsets[i], offsets[i + 1]):
            j = neighbors[k]
            dx, dy, dz = _separation(positions, i, j, period, periodicAxis)
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            if r < 1e-12:
                continue
            _, slope = _kernel(r, h, sigma)
            factor = slope / r
            ux = velocities[i, 0] - velocities[j, 0]
            uy = velocities[i, 1] - velocities[j, 1]
            uz = velocities[i, 2] - velocities[j, 2] if dim == 3 else 0.0
            approach = ux * dx + uy * dy + uz * dz
            viscosity = 0.0
            if approach < 0.0:
                mu = h * approach / (r * r + softening)
                viscosity = -alpha * soundSpeed * mu / (0.5 * (densities[i] + densities[j]))
                if not isFluid[j]:
                    viscosity *= boundaryViscosityFactor
            coefficient = -(own + pressures[j] / (densities[j] * densities[j]) + viscosity) * masses[j] * factor
            ax += coefficient * dx
            ay += coefficient * dy
            az += coefficient * dz
        result[i, 0] = ax + gravity[0]
        result[i, 1] = ay + gravity[1]
        if dim == 3:
            result[i, 2] = az + gravity[2]

    return result

@njit(parallel = True, cache = True)
def continuityRate(positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray, isFluid: np.ndarray,
                   offsets: np.ndarray, neighbors: np.ndarray, h: float, sigma: float, period: float = 0.0,
                   periodicAxis: int = 1) -> np.ndarray:

    '''d rho_i / dt = sum_j m_j (v_i - v_j) . grad W_ij over pairs with a fluid member [kg/(m^3 s)].'''

    count, dim = positions.shape
    rate = np.zeros(count)
    for i in prange(count):
        total = 0.0
        for k in range(offsets[i], offsets[i + 1]):
            j = neighbors[k]
            if not (isFluid[i] or isFluid[j]):
                continue
            dx, dy, dz = _separation(positions, i, j, period, periodicAxis)
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            if r < 1e-12:
                continue
            _, slope = _kernel(r, h, sigma)
            ux = velocities[i, 0] - velocities[j, 0]
            uy = velocities[i, 1] - velocities[j, 1]
            uz = velocities[i, 2] - velocities[j, 2] if dim == 3 else 0.0
            total += masses[j] * (ux * dx + uy * dy + uz * dz) * slope / r
        rate[i] = total

    return rate

@njit(parallel = True, cache = True)
def xsphVelocity(positions: np.ndarray, velocities: np.ndarray, densities: np.ndarray, masses: np.ndarray,
                 isFluid: np.ndarray, offsets: np.ndarray, neighbors: np.ndarray, h: float, sigma: float,
                 epsilon: float, period: float = 0.0, periodicAxis: int = 1) -> np.ndarray:

    '''v_i + epsilon sum_j (m_j / rho_bar_ij) (v_j - v_i) W_ij over fluid pairs; boundary rows unchanged [m/s].'''

    count, dim = positions.shape
    result = velocities.copy()
    for i in prange(count):
        if not isFluid[i]:
            continue
        cx, cy, cz = 0.0, 0.0, 0.0
        for k in range(offsets[i], offsets[i + 1]):
            j = neighbors[k]
            if not isFluid[j]:
                continue
            dx, dy, dz = _separation(positions, i, j, period, periodicAxis)
            value, _ = _kernel(math.sqrt(dx * dx + dy * dy + dz * dz), h, sigma)
            weight = masses[j] * value / (0.5 * (densities[i] + densities[j]))
            cx += weight * (velocities[j, 0] - velocities[i, 0])
            cy += weight * (velocities[j, 1] - velocities[i, 1])
            if dim == 3:
                cz += weight * (velocities[j, 2] - velocities[i, 2])
        result[i, 0] += epsilon * cx
        result[i, 1] += epsilon * cy
        if dim == 3:
            result[i, 2] += epsilon * cz

    return result

@njit(parallel = True, cache = True)
def shepardDensity(positions: np.ndarray, densities: np.ndarray, masses: np.ndarray, offsets: np.ndarray,
                   neighbors: np.ndarray, h: float, sigma: float, period: float = 0.0,
                   periodicAxis: int = 1) -> np.ndarray:

    '''rho_i = sum_j m_j W_ij / sum_j (m_j / rho_j) W_ij, the particle itself included [kg/m^3].'''

    count = positions.shape[0]
    result = np.empty(count)
    selfWeight = sigma
    for i in prange(count):
        numerator = masses[i] * selfWeight
        denominator = masses[i] / densities[i] * selfWeight
        for k in range(offsets[i], offsets[i + 1]):
            j = neighbors[k]
            dx, dy, dz = _separation(positions, i, j, period, periodicAxis)
            value, _ = _kernel(math.sqrt(dx * dx + dy * dy + dz * dz), h, sigma)
            numerator += masses[j] * value
            denominator += masses[j] / densities[j] * value
        result[i] = numerator / denominator

    return result
