
# -- pySurf: GPU SPH Kernels -- #

'''

The SPH sums and time step of `pysurf.sphSolver` on a GPU, written in NVIDIA Warp, for runs of
millions of particles: the same four sums as `pysurf.sphCompiled`, term for term, with the cubic
spline kernel.

    PairSums            device copies of the particle state, a hash grid over the particles, the
                        sums (pair accelerations, the continuity rate, the XSPH transport velocity
                        and the Shepard-filtered density), and the device-resident time step

The solver uses the device in one of two ways:

    step        the host keeps the state and does the time step, the integration and the
                boundaries; each step `update` copies the positions, masses and fluid flags up and
                rebuilds the hash grid, and each sum copies up the fields it reads and copies its
                result back. Every model runs this way, whatever it changes between steps.
    advance     the state goes up once (`push`), every step runs on the device (`residentRates`,
                then `residentIntegrate` with the time step the host computes from the two maxima
                they return), and the state comes back once (`pull`). The boundary must be one the
                device can enforce (`canEnforce`): a container, or a bed, whose piston then holds
                its position and speed until the state comes back.

The state is float64 on the device as on the host: positions, velocities and densities are
integrated in float64, and the pair sums run in float32. A float32 coordinate 30 m from the
origin rounds to 2e-6 m, 6e-5 of a 3 cm separation, and the pressure terms of a pair are a hundred
times gravity and cancel to it; so each float64 position is split into a float32 point and a float32
remainder, and a separation is the difference of the points, exact in float32 for two nearby
points (Sterbenz's lemma), plus the difference of the remainders: as accurate as a float64
difference rounded to float32, with no float64 arithmetic, which runs at 1/32 of the float32 rate
on consumer GPUs (`separationPrecision = 'float64'`). `separationPrecision = 'float32'` takes
separations from the points alone, for comparison. A density change of one step can be smaller
than a float32 density's rounding at 1000 kg/m^3, 6e-5, and would be lost every step, so the
integration stays float64.

Every particle sums over its own neighbors, as in `pysurf.sphCompiled`: each pair is evaluated
from both sides, and no two threads write to the same particle. Threads visit particles in the
hash grid's cell order, so neighboring threads read neighboring particles, and each candidate is
first tested on its float32 distance, so the precise separation is taken for neighbors only. The
grid's cells are one support across, and its dimensions cover the domain the solver is given, so
no two occupied cells share a hash; particles outside that domain are still found, at the cost of
extra candidates.

A periodic axis is y in 3D, on [0, period): a particle within a support of the seam searches a
second time from its image across it. The period must exceed twice the support, as the solver
requires, so no neighbor is reached from both.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

Nothing here is a new model. tests/testSphGpu.py checks every sum against its array counterpart in
`pysurf.sphSolver` on random particle clouds in 2D and 3D, with and without a periodic axis, to
float32 rounding, on Warp's CPU device and on CUDA where present; the bed enforcement against
`BedBoundary.enforce` to 1e-12 in 2D and 3D; a solver run on this backend against the compiled
backend; the device-resident steps against the host steps in a tank and over a bed; and a still
tank at rest. tests/testValidationGpuBackend.py repeats the CPU backends' validation cases on it.

Sean Bowman [11/05/2026]

'''

# Standard library imports
from __future__ import annotations

import math

# Third-party imports
import numpy as np
import warp as wp

# Local imports
from pysurf.sphBoundaries import BedBoundary, ContainerBoundary

wp.config.quiet = True

viscositySoftening = wp.constant(0.01)          # [-] as in sphSolver: mu_ij's denominator r^2 + 0.01 h^2
candidateMargin = wp.constant(1.0001)           # [-] on r^2: float32 distances near 30 m are good to 1e-4 of a support
taitExponent = 7.0                              # as in sphSolver
maxGridCells = 2**25                            # cap on the hash grid's cells, 256 MB of cell ranges

# The device arrays of a particle system, about 133 bytes a particle: the float64 state, its float32
# copies for the pair sums, and their results
layout = {'positions': wp.vec3d, 'velocities64': wp.vec3d, 'densities64': wp.float64, 'points': wp.vec3,
          'lows': wp.vec3, 'velocities': wp.vec3, 'densities': float, 'pressures': float, 'masses': float,
          'fluid': wp.uint8, 'accelerations': wp.vec3, 'vector': wp.vec3, 'scalar': float}

#----------------------------------------------------------------------#
# -- Device Functions -- #
#----------------------------------------------------------------------#

@wp.func
def cubicValue(r: float, h: float, sigma: float) -> float:

    '''The cubic spline W(r).'''

    q = r / h
    if q < 1.0:
        return sigma * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
    if q < 2.0:
        rest = 2.0 - q
        return sigma * 0.25 * rest * rest * rest
    return 0.0

@wp.func
def cubicSlope(r: float, h: float, sigma: float) -> float:

    '''The cubic spline dW/dr.'''

    q = r / h
    if q < 1.0:
        return sigma * (-3.0 * q + 2.25 * q * q) / h
    if q < 2.0:
        rest = 2.0 - q
        return sigma * (-0.75 * rest * rest) / h
    return 0.0

@wp.func
def imageCount(point: wp.vec3, period: float, support: float) -> int:

    '''Searches a particle makes: one, or two within a support of the periodic seam.'''

    if period > 0.0:
        if point[1] < support or point[1] > period - support:
            return 2
    return 1

@wp.func
def imageShift(point: wp.vec3, period: float, image: int) -> float:

    '''The y shift of a search: none for the first, across the seam for the second [m].'''

    if image == 0:
        return 0.0
    if point[1] < 0.5 * period:
        return period
    return -period

@wp.func
def separation(coarse: wp.vec3, pointI: wp.vec3, pointJ: wp.vec3, lowI: wp.vec3, lowJ: wp.vec3, offset: wp.vec3,
               precise: int) -> wp.vec3:

    '''

    r_i - r_j, r_i shifted by `offset`. Precise, from each float64 position split into a float32
    point and a float32 remainder: two nearby points differ exactly in float32 (Sterbenz), so the
    separation keeps the float64 positions' digits. Otherwise the float32 `coarse` separation of the
    points.

    '''

    if precise != 0:
        return (pointI - pointJ) + offset + (lowI - lowJ)
    return coarse

@wp.func
def single(value: wp.vec3d) -> wp.vec3:

    '''A float64 vector as float32.'''

    return wp.vec3(wp.float32(value[0]), wp.float32(value[1]), wp.float32(value[2]))

@wp.func
def double(value: wp.vec3) -> wp.vec3d:

    '''A float32 vector as float64.'''

    return wp.vec3d(wp.float64(value[0]), wp.float64(value[1]), wp.float64(value[2]))

#----------------------------------------------------------------------#
# -- Pair Sums -- #
#----------------------------------------------------------------------#

@wp.kernel
def accelerationKernel(grid: wp.uint64, points: wp.array(dtype = wp.vec3), lows: wp.array(dtype = wp.vec3),
                       precise: int, period: float, velocities: wp.array(dtype = wp.vec3),
                       densities: wp.array(dtype = float), pressures: wp.array(dtype = float),
                       masses: wp.array(dtype = float), fluid: wp.array(dtype = wp.uint8), h: float, sigma: float,
                       soundSpeed: float, alpha: float, boundaryFactor: float, gravity: wp.vec3,
                       result: wp.array(dtype = wp.vec3)):

    '''a_i = -sum_j m_j (p_i / rho_i^2 + p_j / rho_j^2 + Pi_ij) grad W_ij + g on the fluid, zero on the boundary.'''

    i = wp.hash_grid_point_id(grid, wp.tid())
    if fluid[i] == wp.uint8(0):
        result[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    support = 2.0 * h
    limit = candidateMargin * support * support
    softening = viscositySoftening * h * h
    pointI = points[i]
    lowI = lows[i]
    velocityI = velocities[i]
    densityI = densities[i]
    own = pressures[i] / (densityI * densityI)
    total = wp.vec3(0.0, 0.0, 0.0)
    for image in range(imageCount(pointI, period, support)):
        offset = wp.vec3(0.0, imageShift(pointI, period, image), 0.0)
        origin = pointI + offset
        query = wp.hash_grid_query(grid, origin, support)
        j = int(0)
        while wp.hash_grid_query_next(query, j):
            pointJ = points[j]
            coarse = origin - pointJ
            if wp.dot(coarse, coarse) >= limit or j == i:
                continue
            d = separation(coarse, pointI, pointJ, lowI, lows[j], offset, precise)
            r = wp.length(d)
            if r >= support or r < 1.0e-12:
                continue
            densityJ = densities[j]
            approach = wp.dot(velocityI - velocities[j], d)
            viscosity = float(0.0)
            if approach < 0.0:
                mu = h * approach / (r * r + softening)
                viscosity = -alpha * soundSpeed * mu / (0.5 * (densityI + densityJ))
                if fluid[j] == wp.uint8(0):
                    viscosity = viscosity * boundaryFactor
            coefficient = -(own + pressures[j] / (densityJ * densityJ) + viscosity) * masses[j]
            total += coefficient * cubicSlope(r, h, sigma) / r * d
    result[i] = total + gravity

@wp.kernel
def continuityKernel(grid: wp.uint64, points: wp.array(dtype = wp.vec3), lows: wp.array(dtype = wp.vec3),
                     precise: int, period: float, velocities: wp.array(dtype = wp.vec3),
                     masses: wp.array(dtype = float), fluid: wp.array(dtype = wp.uint8), h: float, sigma: float,
                     result: wp.array(dtype = float)):

    '''d rho_i / dt = sum_j m_j (v_i - v_j) . grad W_ij over pairs with a fluid member.'''

    i = wp.hash_grid_point_id(grid, wp.tid())
    support = 2.0 * h
    limit = candidateMargin * support * support
    pointI = points[i]
    lowI = lows[i]
    velocityI = velocities[i]
    boundaryI = fluid[i] == wp.uint8(0)
    total = float(0.0)
    for image in range(imageCount(pointI, period, support)):
        offset = wp.vec3(0.0, imageShift(pointI, period, image), 0.0)
        origin = pointI + offset
        query = wp.hash_grid_query(grid, origin, support)
        j = int(0)
        while wp.hash_grid_query_next(query, j):
            pointJ = points[j]
            coarse = origin - pointJ
            if wp.dot(coarse, coarse) >= limit or j == i:
                continue
            if boundaryI and fluid[j] == wp.uint8(0):
                continue
            d = separation(coarse, pointI, pointJ, lowI, lows[j], offset, precise)
            r = wp.length(d)
            if r >= support or r < 1.0e-12:
                continue
            total += masses[j] * wp.dot(velocityI - velocities[j], d) * cubicSlope(r, h, sigma) / r
    result[i] = total

@wp.kernel
def xsphKernel(grid: wp.uint64, points: wp.array(dtype = wp.vec3), lows: wp.array(dtype = wp.vec3),
               precise: int, period: float, velocities: wp.array(dtype = wp.vec3), densities: wp.array(dtype = float),
               masses: wp.array(dtype = float), fluid: wp.array(dtype = wp.uint8), h: float, sigma: float,
               epsilon: float, result: wp.array(dtype = wp.vec3)):

    '''v_i + epsilon sum_j (m_j / rho_bar_ij) (v_j - v_i) W_ij over fluid pairs; boundary rows unchanged.'''

    i = wp.hash_grid_point_id(grid, wp.tid())
    velocityI = velocities[i]
    if fluid[i] == wp.uint8(0):
        result[i] = velocityI
        return
    support = 2.0 * h
    limit = candidateMargin * support * support
    pointI = points[i]
    lowI = lows[i]
    densityI = densities[i]
    correction = wp.vec3(0.0, 0.0, 0.0)
    for image in range(imageCount(pointI, period, support)):
        offset = wp.vec3(0.0, imageShift(pointI, period, image), 0.0)
        origin = pointI + offset
        query = wp.hash_grid_query(grid, origin, support)
        j = int(0)
        while wp.hash_grid_query_next(query, j):
            pointJ = points[j]
            coarse = origin - pointJ
            if wp.dot(coarse, coarse) >= limit or j == i:
                continue
            if fluid[j] == wp.uint8(0):
                continue
            d = separation(coarse, pointI, pointJ, lowI, lows[j], offset, precise)
            r = wp.length(d)
            if r >= support:
                continue
            weight = masses[j] * cubicValue(r, h, sigma) / (0.5 * (densityI + densities[j]))
            correction += weight * (velocities[j] - velocityI)
    result[i] = velocityI + epsilon * correction

@wp.kernel
def shepardKernel(grid: wp.uint64, points: wp.array(dtype = wp.vec3), lows: wp.array(dtype = wp.vec3),
                  precise: int, period: float, densities: wp.array(dtype = float), masses: wp.array(dtype = float),
                  h: float, sigma: float, result: wp.array(dtype = float)):

    '''rho_i = sum_j m_j W_ij / sum_j (m_j / rho_j) W_ij, the particle itself included.'''

    i = wp.hash_grid_point_id(grid, wp.tid())
    support = 2.0 * h
    limit = candidateMargin * support * support
    pointI = points[i]
    lowI = lows[i]
    numerator = masses[i] * sigma
    denominator = masses[i] / densities[i] * sigma
    for image in range(imageCount(pointI, period, support)):
        offset = wp.vec3(0.0, imageShift(pointI, period, image), 0.0)
        origin = pointI + offset
        query = wp.hash_grid_query(grid, origin, support)
        j = int(0)
        while wp.hash_grid_query_next(query, j):
            pointJ = points[j]
            coarse = origin - pointJ
            if wp.dot(coarse, coarse) >= limit or j == i:
                continue
            d = separation(coarse, pointI, pointJ, lowI, lows[j], offset, precise)
            r = wp.length(d)
            if r >= support:
                continue
            value = cubicValue(r, h, sigma)
            numerator += masses[j] * value
            denominator += masses[j] / densities[j] * value
    result[i] = numerator / denominator

#----------------------------------------------------------------------#
# -- Device-Resident Step -- #
#----------------------------------------------------------------------#

@wp.kernel
def toPoints(positions: wp.array(dtype = wp.vec3d), points: wp.array(dtype = wp.vec3),
             lows: wp.array(dtype = wp.vec3)):

    '''Each float64 position as a float32 point, for the hash grid, and the float32 remainder.'''

    i = wp.tid()
    point = single(positions[i])
    points[i] = point
    lows[i] = single(positions[i] - double(point))

@wp.kernel
def toSingle(velocities: wp.array(dtype = wp.vec3d), densities: wp.array(dtype = wp.float64),
             velocities32: wp.array(dtype = wp.vec3), densities32: wp.array(dtype = float)):

    '''Float32 copies of the velocities and densities, for the pair sums.'''

    i = wp.tid()
    velocities32[i] = single(velocities[i])
    densities32[i] = wp.float32(densities[i])

@wp.kernel
def applyShepard(filtered: wp.array(dtype = float), fluid: wp.array(dtype = wp.uint8),
                 densities: wp.array(dtype = wp.float64), densities32: wp.array(dtype = float)):

    '''The fluid takes its Shepard-filtered density; the boundary keeps its own.'''

    i = wp.tid()
    if fluid[i] != wp.uint8(0):
        densities[i] = wp.float64(filtered[i])
        densities32[i] = filtered[i]

@wp.kernel
def equationOfState(densities: wp.array(dtype = wp.float64), referenceDensity: wp.float64, stiffness: wp.float64,
                    pressures: wp.array(dtype = float)):

    '''Tait pressure B ((rho / rho0)^7 - 1), negative pressure clamped to zero.'''

    i = wp.tid()
    ratio = densities[i] / referenceDensity
    square = ratio * ratio
    pressure = stiffness * (square * square * square * ratio - wp.float64(1.0))
    pressures[i] = wp.float32(wp.max(pressure, wp.float64(0.0)))

@wp.kernel
def fluidMaxima(velocities: wp.array(dtype = wp.vec3d), accelerations: wp.array(dtype = wp.vec3),
                fluid: wp.array(dtype = wp.uint8), maxima: wp.array(dtype = wp.float64)):

    '''The largest fluid speed and acceleration, into maxima[0] and maxima[1].'''

    i = wp.tid()
    if fluid[i] == wp.uint8(0):
        return
    wp.atomic_max(maxima, 0, wp.length(velocities[i]))
    wp.atomic_max(maxima, 1, wp.float64(wp.length(accelerations[i])))

@wp.kernel
def kick(velocities: wp.array(dtype = wp.vec3d), accelerations: wp.array(dtype = wp.vec3),
         fluid: wp.array(dtype = wp.uint8), timeStep: wp.float64, velocities32: wp.array(dtype = wp.vec3)):

    '''v += a dt on the fluid.'''

    i = wp.tid()
    velocity = velocities[i]
    if fluid[i] != wp.uint8(0):
        velocity = velocity + double(accelerations[i]) * timeStep
        velocities[i] = velocity
    velocities32[i] = single(velocity)

@wp.kernel
def advanceDensity(densities: wp.array(dtype = wp.float64), rates: wp.array(dtype = float),
                   fluid: wp.array(dtype = wp.uint8), timeStep: wp.float64, floor: wp.float64,
                   densities32: wp.array(dtype = float)):

    '''rho += d rho / dt dt for every particle, the fluid's held at the floor when one is set.'''

    i = wp.tid()
    density = densities[i] + wp.float64(rates[i]) * timeStep
    if floor > wp.float64(0.0) and fluid[i] != wp.uint8(0):
        density = wp.max(density, floor)
    densities[i] = density
    densities32[i] = wp.float32(density)

@wp.kernel
def drift(positions: wp.array(dtype = wp.vec3d), transport: wp.array(dtype = wp.vec3),
          velocities: wp.array(dtype = wp.vec3d), useTransport: int, fluid: wp.array(dtype = wp.uint8),
          timeStep: wp.float64, period: wp.float64):

    '''x += u dt on the fluid, u the XSPH transport velocity or the velocity; y wrapped onto the period.'''

    i = wp.tid()
    if fluid[i] == wp.uint8(0):
        return
    velocity = velocities[i]
    if useTransport != 0:
        velocity = double(transport[i])
    x = positions[i] + velocity * timeStep
    if period > wp.float64(0.0):
        x = wp.vec3d(x[0], x[1] - period * wp.floor(x[1] / period), x[2])
    positions[i] = x

@wp.kernel
def enforceContainer(positions: wp.array(dtype = wp.vec3d), velocities: wp.array(dtype = wp.vec3d),
                     fluid: wp.array(dtype = wp.uint8), lower: wp.vec3d, upper: wp.vec3d, dimensions: int,
                     openTop: int):

    '''`ContainerBoundary.enforce`: fluid outside the box returns to its face, with no velocity through it.'''

    i = wp.tid()
    if fluid[i] == wp.uint8(0):
        return
    x = positions[i]
    v = velocities[i]
    zero = wp.float64(0.0)
    for axis in range(dimensions):
        if x[axis] < lower[axis]:
            x[axis] = lower[axis]
            v[axis] = wp.max(v[axis], zero)
        if axis < dimensions - 1 or openTop == 0:
            if x[axis] > upper[axis]:
                x[axis] = upper[axis]
                v[axis] = wp.min(v[axis], zero)
    positions[i] = x
    velocities[i] = v

@wp.func
def bedSegment(x: wp.float64, bedX: wp.array(dtype = wp.float64)) -> int:

    '''The bed segment k holding x, bedX[k] <= x < bedX[k + 1], clipped to the first and last.'''

    low = int(0)
    high = bedX.shape[0] - 1
    while high - low > 1:
        middle = (low + high) // 2
        if bedX[middle] <= x:
            low = middle
        else:
            high = middle
    return low

@wp.kernel
def enforceBed(positions: wp.array(dtype = wp.vec3d), velocities: wp.array(dtype = wp.vec3d),
               fluid: wp.array(dtype = wp.uint8), bedX: wp.array(dtype = wp.float64),
               bedZ: wp.array(dtype = wp.float64), vertical: int, clearance: wp.float64, sideWalls: int,
               width: wp.float64, offshoreWall: int, wallX: wp.float64, pistonSpeed: wp.float64,
               endLimit: wp.float64):

    '''

    `BedBoundary.enforce`: fluid below the bed returns to a quarter spacing above it with no
    velocity into it; fluid past a side wall, the offshore wall or the shoreward end returns to it.

    '''

    i = wp.tid()
    if fluid[i] == wp.uint8(0):
        return
    x = positions[i]
    v = velocities[i]
    zero = wp.float64(0.0)
    one = wp.float64(1.0)

    # Bed, as np.interp and the segment slope give it
    k = bedSegment(x[0], bedX)
    run = bedX[k + 1] - bedX[k]
    rise = bedZ[k + 1] - bedZ[k]
    share = wp.clamp((x[0] - bedX[k]) / run, zero, one)
    bed = bedZ[k] + share * rise
    if x[vertical] < bed + clearance:
        x[vertical] = bed + clearance
        slope = rise / run
        length = wp.sqrt(one + slope * slope)
        normalX = -slope / length
        normalZ = one / length
        inward = v[0] * normalX + v[vertical] * normalZ
        if inward < zero:
            v[0] = v[0] - inward * normalX
            v[vertical] = v[vertical] - inward * normalZ

    # Side walls of a 3D channel
    if sideWalls != 0:
        if x[1] < clearance:
            x[1] = clearance
            v[1] = wp.max(v[1], zero)
        if x[1] > width - clearance:
            x[1] = width - clearance
            v[1] = wp.min(v[1], zero)

    # Offshore wall, at the piston's position and speed, and the shoreward end
    if offshoreWall != 0:
        if x[0] < wallX + clearance:
            x[0] = wallX + clearance
            v[0] = wp.max(v[0], pistonSpeed)
    if x[0] > endLimit:
        x[0] = endLimit
        v[0] = wp.min(v[0], zero)

    positions[i] = x
    velocities[i] = v

#----------------------------------------------------------------------#
# -- Host Interface -- #
#----------------------------------------------------------------------#

def cubicSplineNormalization(smoothingLength: float, dimensions: int) -> float:

    '''The cubic spline's sigma: 10 / (7 pi h^2) in 2D, 1 / (pi h^3) in 3D.'''

    h = smoothingLength
    return 10.0 / (7.0 * math.pi * h * h) if dimensions == 2 else 1.0 / (math.pi * h * h * h)

def gridDimensions(domainMin: np.ndarray, domainMax: np.ndarray, cellSize: float, dimensions: int) -> tuple:

    '''Hash grid cells along x, y and z covering the domain, one support across, capped at `maxGridCells`.'''

    extent = np.zeros(3)
    extent[:dimensions] = np.asarray(domainMax, dtype = float) - np.asarray(domainMin, dtype = float)
    cells = np.maximum(np.ceil(extent / cellSize).astype(int) + 3, 1)
    if dimensions == 2:
        cells[2] = 1
    while int(np.prod(cells)) > maxGridCells:
        cells = np.maximum((cells * 0.8).astype(int), 1)

    return tuple(int(count) for count in cells)

def padded(values: np.ndarray) -> np.ndarray:

    '''Vectors of 2 or 3 components as 3, the third zero for 2.'''

    values = np.asarray(values, dtype = float)
    if values.shape[1] == 3:
        return np.ascontiguousarray(values)
    result = np.zeros((len(values), 3))
    result[:, :values.shape[1]] = values

    return result

class PairSums:

    '''

    The SPH pair sums and the time step of one particle system on a Warp device.

    For host steps, call `update` once per step, after the positions move, then any of the sums:
    every sum takes host arrays and returns a host array, float64, shaped as the compiled
    backend's. For device steps, set the physics (`setPhysics`), copy the state up (`push`), set
    the boundary (`setBoundary`), call `residentRates` and `residentIntegrate` each step, and copy
    the state back (`pull`).

    Parameters:
    -----------
    dimensions : int
        2 or 3
    smoothingLength : float
        h [m]; the support is 2h
    domainMin, domainMax : np.ndarray
        Corners of the region the particles occupy [m], which sizes the hash grid
    period : float
        Length of a periodic y axis in 3D [m]; 0 for none
    separationPrecision : str
        'float64' to take pair separations from the float64 positions' point and remainder,
        'float32' from the float32 points alone
    device : str | None
        Warp device, 'cuda:0' or 'cpu'; None for Warp's preferred device

    '''

    def __init__(self, dimensions: int, smoothingLength: float, domainMin: np.ndarray, domainMax: np.ndarray,
                 period: float = 0.0, separationPrecision: str = 'float64', device: str | None = None) -> None:

        if separationPrecision not in ('float64', 'float32'):
            raise ValueError(f"Separation precision '{separationPrecision}' is not 'float64' or 'float32'.")
        if period > 0.0 and dimensions != 3:
            raise ValueError('A periodic axis needs three dimensions: it is y.')
        self.dimensions = dimensions
        self.smoothingLength = float(smoothingLength)                    # [m]
        self.sigma = cubicSplineNormalization(smoothingLength, dimensions)
        self.period = float(period)                                      # [m]
        self.precise = 1 if separationPrecision == 'float64' else 0
        self.device = wp.get_device(device)
        self.grid = wp.HashGrid(*gridDimensions(domainMin, domainMax, 2.0 * smoothingLength, dimensions),
                                device = self.device)
        self.count = 0
        self.physics: dict = {}
        self._enforcement: tuple | None = None
        self._capacity = 0
        self._device: dict[str, wp.array] = {}
        self._staging: dict[str, wp.array] = {}
        self._maxima = wp.zeros(2, dtype = wp.float64, device = self.device)

    # -- Private Helper Methods -- #
    def _reserve(self, count: int) -> None:

        '''Device arrays for at least `count` particles; staging arrays are made again when first used.'''

        if count <= self._capacity:
            return
        capacity = max(count, int(1.25 * self._capacity))
        self._device = {name: wp.zeros(capacity, dtype = dtype, device = self.device) for name, dtype in layout.items()}
        self._staging = {}
        self._capacity = capacity

    def _stage(self, name: str) -> wp.array:

        '''The pinned host array a field passes through, made on first use: only host steps need them.'''

        if name not in self._staging:
            self._staging[name] = wp.zeros(self._capacity, dtype = layout[name], device = 'cpu',
                                           pinned = self.device.is_cuda)
        return self._staging[name]

    def _upload(self, name: str, values: np.ndarray) -> wp.array:

        '''Copy a host field into its device array through the staging array; returns the device array.'''

        count = self.count
        staging = self._stage(name)
        view = staging.numpy()
        if view.ndim == 2:
            view[:count, :self.dimensions] = values
        else:
            view[:count] = values
        wp.copy(self._device[name], staging, count = count)

        return self._device[name]

    def _download(self, name: str) -> np.ndarray:

        '''Copy a device result back to the host as float64, shape (N, dim) or (N,).'''

        staging = self._stage(name)
        wp.copy(staging, self._device[name], count = self.count)
        wp.synchronize_device(self.device)
        values = staging.numpy()[:self.count]

        return values[:, :self.dimensions].astype(float) if values.ndim == 2 else values.astype(float)

    def _launch(self, kernel, inputs: list) -> None:

        wp.launch(kernel, dim = self.count, inputs = inputs, device = self.device)

    def _rebuild(self) -> None:

        '''Float32 points from the float64 positions, and the hash grid over them.'''

        d = self._device
        self._launch(toPoints, [d['positions'], d['points'], d['lows']])
        self.grid.build(d['points'][:self.count], 2.0 * self.smoothingLength)

    @property
    def _geometry(self) -> list:
        '''The pair kernels' leading arguments: grid, points, remainders, precision flag, period.'''
        d = self._device
        return [self.grid.id, d['points'], d['lows'], self.precise, self.period]

    # -- Host Steps -- #
    def update(self, positions: np.ndarray, masses: np.ndarray, isFluid: np.ndarray) -> None:

        '''

        Copy the positions, masses and fluid flags to the device and rebuild the hash grid.

        Parameters:
        -----------
        positions : np.ndarray
            Particle positions, shape (N, dim) [m]
        masses : np.ndarray
            Particle masses, shape (N,) [kg]
        isFluid : np.ndarray
            True for fluid particles, shape (N,)

        '''

        self.count = len(positions)
        self._reserve(self.count)
        if self.count == 0:
            return
        self._upload('positions', positions)
        self._upload('masses', masses)
        self._upload('fluid', np.asarray(isFluid).view(np.uint8))
        self._rebuild()

    def pairAccelerations(self, velocities: np.ndarray, densities: np.ndarray, pressures: np.ndarray,
                          soundSpeed: float, alpha: float, boundaryViscosityFactor: float,
                          gravity: np.ndarray) -> np.ndarray:

        '''Pressure gradient, artificial viscosity and gravity on the fluid, zero on the boundary [m/s^2].'''

        g = np.zeros(3)
        g[:self.dimensions] = gravity
        self._launch(accelerationKernel, self._geometry + [
            self._upload('velocities', velocities), self._upload('densities', densities),
            self._upload('pressures', pressures), self._device['masses'], self._device['fluid'],
            self.smoothingLength, self.sigma, soundSpeed, alpha, boundaryViscosityFactor, wp.vec3(*g),
            self._device['vector']])

        return self._download('vector')

    def continuityRate(self, velocities: np.ndarray) -> np.ndarray:

        '''d rho / dt for every particle [kg/(m^3 s)].'''

        self._launch(continuityKernel, self._geometry + [
            self._upload('velocities', velocities), self._device['masses'], self._device['fluid'],
            self.smoothingLength, self.sigma, self._device['scalar']])

        return self._download('scalar')

    def xsphVelocity(self, velocities: np.ndarray, densities: np.ndarray, epsilon: float) -> np.ndarray:

        '''The XSPH transport velocity of the fluid; boundary rows unchanged [m/s].'''

        self._launch(xsphKernel, self._geometry + [
            self._upload('velocities', velocities), self._upload('densities', densities), self._device['masses'],
            self._device['fluid'], self.smoothingLength, self.sigma, epsilon, self._device['vector']])

        return self._download('vector')

    def shepardDensity(self, densities: np.ndarray) -> np.ndarray:

        '''The Shepard-filtered density of every particle [kg/m^3].'''

        self._launch(shepardKernel, self._geometry + [
            self._upload('densities', densities), self._device['masses'], self.smoothingLength, self.sigma,
            self._device['scalar']])

        return self._download('scalar')

    # -- Device Steps -- #
    def canEnforce(self, boundary) -> bool:

        '''Whether the device can enforce this boundary: none, a `ContainerBoundary` or a `BedBoundary`.'''

        return boundary is None or isinstance(boundary, (ContainerBoundary, BedBoundary))

    def setBoundary(self, boundary, particles) -> None:

        '''

        The boundary the device steps enforce, as it stands: a piston holds its position and speed
        until the state comes back to the host.

        Parameters:
        -----------
        boundary : ContainerBoundary | BedBoundary | None
            The solver's boundary
        particles : ParticleSystem
            The particle state, for the piston's speed

        '''

        d = self._device
        if boundary is None:
            self._enforcement = None
        elif isinstance(boundary, ContainerBoundary):
            lower, upper = np.zeros(3), np.zeros(3)
            lower[:self.dimensions] = boundary.containerMin
            upper[:self.dimensions] = boundary.containerMax
            self._enforcement = (enforceContainer, [d['positions'], d['velocities64'], d['fluid'], wp.vec3d(*lower),
                                                    wp.vec3d(*upper), boundary.dimensions,
                                                    1 if boundary.openTop else 0])
        elif isinstance(boundary, BedBoundary):
            bed, clearance = boundary.bathymetry, 0.25 * boundary.spacing
            mask = boundary._pistonMask
            pistonSpeed = float(particles.velocities[mask, 0][0]) if mask is not None and mask.any() else 0.0
            sideWalls = boundary.width is not None and not boundary.periodic
            endLimit = bed.endX - clearance if boundary.endWallTop is not None else bed.endX
            self._enforcement = (enforceBed, [
                d['positions'], d['velocities64'], d['fluid'], wp.array(bed.x, dtype = wp.float64, device = self.device),
                wp.array(bed.z, dtype = wp.float64, device = self.device), self.dimensions - 1, clearance,
                1 if sideWalls else 0, float(boundary.width or 0.0), 1 if boundary.wallTop is not None else 0,
                float(boundary.wallX), pistonSpeed, float(endLimit)])
        else:
            raise ValueError(f'The device cannot enforce a {type(boundary).__name__}; step from the host.')

    def setPhysics(self, referenceDensity: float, soundSpeed: float, alpha: float, boundaryViscosityFactor: float,
                   gravity: np.ndarray, xsphEpsilon: float, densityFloor: float) -> None:

        '''

        The constants of the device step.

        Parameters:
        -----------
        referenceDensity : float
            rho0 [kg/m^3]
        soundSpeed : float
            c0 [m/s]
        alpha : float
            Artificial viscosity coefficient
        boundaryViscosityFactor : float
            Multiplier on the artificial viscosity between fluid and boundary particles
        gravity : np.ndarray
            Gravity vector, shape (dim,) [m/s^2]
        xsphEpsilon : float
            XSPH blending factor; 0 moves the fluid with its velocity
        densityFloor : float
            Smallest fluid density [kg/m^3]; 0 for none

        '''

        g = np.zeros(3)
        g[:self.dimensions] = gravity
        self.physics = {'referenceDensity': float(referenceDensity), 'soundSpeed': float(soundSpeed),
                        'stiffness': float(referenceDensity * soundSpeed**2 / taitExponent), 'alpha': float(alpha),
                        'boundaryViscosityFactor': float(boundaryViscosityFactor), 'gravity': wp.vec3(*g),
                        'xsphEpsilon': float(xsphEpsilon), 'densityFloor': float(densityFloor)}

    def push(self, particles) -> None:

        '''Copy a particle system's state to the device.'''

        self.update(particles.positions, particles.masses, particles.isFluid)
        if self.count == 0:
            return
        d = self._device
        wp.copy(d['velocities64'], wp.array(padded(particles.velocities), dtype = wp.vec3d, device = 'cpu'),
                count = self.count)
        wp.copy(d['densities64'], wp.array(np.asarray(particles.densities, dtype = float), dtype = wp.float64,
                                           device = 'cpu'), count = self.count)
        wp.synchronize_device(self.device)

    def pull(self, particles) -> None:

        '''Copy the device state back into a particle system, in place.'''

        if self.count == 0:
            return
        d, dim, count = self._device, self.dimensions, self.count
        wp.synchronize_device(self.device)
        particles.positions[:] = d['positions'].numpy()[:count, :dim]
        particles.velocities[:] = d['velocities64'].numpy()[:count, :dim]
        particles.densities[:] = d['densities64'].numpy()[:count]
        particles.pressures[:] = d['pressures'].numpy()[:count]
        particles.accelerations[:] = d['accelerations'].numpy()[:count, :dim]

    def residentRates(self, filterDensity: bool) -> tuple[float, float]:

        '''

        The first half of a device step: rebuild the grid, filter the density if due, then the
        pressure and the accelerations.

        Returns:
        --------
        tuple[float, float] : the largest fluid speed [m/s] and acceleration [m/s^2]

        '''

        d, physics = self._device, self.physics
        self._rebuild()
        self._launch(toSingle, [d['velocities64'], d['densities64'], d['velocities'], d['densities']])
        if filterDensity:
            self._launch(shepardKernel, self._geometry + [d['densities'], d['masses'], self.smoothingLength, self.sigma,
                                                          d['scalar']])
            self._launch(applyShepard, [d['scalar'], d['fluid'], d['densities64'], d['densities']])
        self._launch(equationOfState, [d['densities64'], physics['referenceDensity'], physics['stiffness'],
                                       d['pressures']])
        self._launch(accelerationKernel, self._geometry + [
            d['velocities'], d['densities'], d['pressures'], d['masses'], d['fluid'], self.smoothingLength, self.sigma,
            physics['soundSpeed'], physics['alpha'], physics['boundaryViscosityFactor'], physics['gravity'],
            d['accelerations']])
        self._maxima.zero_()
        self._launch(fluidMaxima, [d['velocities64'], d['accelerations'], d['fluid'], self._maxima])
        maxima = self._maxima.numpy()

        return float(maxima[0]), float(maxima[1])

    def residentIntegrate(self, timeStep: float) -> None:

        '''

        The second half of a device step: kick the fluid velocity, advance every density with the
        kicked velocity, drift the fluid with the XSPH transport velocity, and enforce the boundary
        set by `setBoundary`.

        '''

        d, physics = self._device, self.physics
        epsilon = physics['xsphEpsilon']
        self._launch(kick, [d['velocities64'], d['accelerations'], d['fluid'], timeStep, d['velocities']])
        self._launch(continuityKernel, self._geometry + [d['velocities'], d['masses'], d['fluid'], self.smoothingLength,
                                                         self.sigma, d['scalar']])
        self._launch(advanceDensity, [d['densities64'], d['scalar'], d['fluid'], timeStep, physics['densityFloor'],
                                      d['densities']])
        if epsilon > 0.0:
            self._launch(xsphKernel, self._geometry + [d['velocities'], d['densities'], d['masses'], d['fluid'],
                                                       self.smoothingLength, self.sigma, epsilon, d['vector']])
        self._launch(drift, [d['positions'], d['vector'], d['velocities64'], 1 if epsilon > 0.0 else 0, d['fluid'],
                             timeStep, self.period])
        self.enforceBoundary()

    def enforceBoundary(self) -> None:

        '''Enforce the boundary set by `setBoundary` on the device state.'''

        if self._enforcement is not None:
            kernel, inputs = self._enforcement
            self._launch(kernel, inputs)
