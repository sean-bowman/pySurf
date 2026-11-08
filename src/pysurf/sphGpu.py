
# -- pySurf: GPU SPH Kernels -- #

'''

The SPH sums and time step of `pysurf.sphSolver` on a GPU, written in NVIDIA Warp, for runs of
millions of particles: the sums of the array formulation, term for term, with the cubic spline or
the Wendland C2 kernel.

    PairSums            device copies of the particle state, a hash grid over the particles, the
                        sums (pair accelerations, the continuity rate with its optional diffusive
                        term, the XSPH transport velocity, the Shepard-filtered density, the
                        extrapolated wall pressure and the particle shift), and the
                        device-resident time step

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

Each particle carries its own smoothing length, none longer than the one the sums are made with,
which sizes the grid's cells: a particle searches out to its own h plus the longest, and each pair
takes the mean of its two lengths, h_ij = (h_i + h_j) / 2, with the kernel's normalization at h_ij
(`pysurf.sphRefinement`). With one length throughout the sums are the uniform ones. Particle
shifting takes one length throughout.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

Nothing here is a new model. tests/testSphGpu.py checks every sum against its array counterpart in
`pysurf.sphSolver` on random particle clouds in 2D and 3D, with and without a periodic axis, to
float32 rounding, on Warp's CPU device and on CUDA where present, the velocity gradients and the
limited artificial viscosity among them, and tests/testSphRefinement.py
the sums the deltaSph scheme uses on clouds of mixed smoothing lengths; the bed enforcement against
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
from pysurf.sphSolver import (antiClumpingPower, antiClumpingWeight, correctionFloor, freeSurfaceDivergence, freeSurfaceLopsidedness,
                              limiterCloseDistance, limiterFoldWidth, localShiftingCoefficient, surfaceGradient)

wp.config.quiet = True

viscositySoftening = wp.constant(0.01)          # [-] as in sphSolver: mu_ij's denominator r^2 + 0.01 h^2
closeDistance = wp.constant(limiterCloseDistance)   # [-] of h_ij, where the limited viscosity's reconstruction fades
foldWidth = wp.constant(limiterFoldWidth)           # [-] of h_ij, the fade's width
candidateMargin = wp.constant(1.0001)           # [-] on r^2: float32 distances near 30 m are good to 1e-4 of a support
taitExponent = 7.0                              # as in sphSolver
maxGridCells = 2**25                            # cap on the hash grid's cells, 256 MB of cell ranges

# The device arrays of a particle system, about 141 bytes a particle: the float64 state, its float32
# copies for the pair sums, and their results
layout = {'positions': wp.vec3d, 'velocities64': wp.vec3d, 'densities64': wp.float64, 'points': wp.vec3,
          'lows': wp.vec3, 'velocities': wp.vec3, 'densities': float, 'deviations': float, 'pressures': float,
          'masses': float, 'lengths': float, 'fluid': wp.uint8, 'accelerations': wp.vec3, 'vector': wp.vec3,
          'scalar': float}
# The shifting fields, about 68 bytes a particle more, made only for a run that shifts
shiftingLayout = {'gradients': wp.vec3, 'normals': wp.vec3, 'divergence': float, 'densityGradients': wp.vec3,
                  'shifts': wp.vec3, 'densityChanges': float, 'velocityChanges': wp.vec3}
diffusionModes = {'none': 0, 'molteni': 1, 'hydrostatic': 2}

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
def wendlandValue(r: float, h: float, sigma: float) -> float:

    '''The Wendland C2 W(r) = sigma (1 - q/2)^4 (2q + 1), q = r / h < 2.'''

    q = r / h
    if q < 2.0:
        rest = 1.0 - 0.5 * q
        square = rest * rest
        return sigma * square * square * (2.0 * q + 1.0)
    return 0.0

@wp.func
def wendlandSlope(r: float, h: float, sigma: float) -> float:

    '''The Wendland C2 dW/dr = -5 sigma q (1 - q/2)^3 / h.'''

    q = r / h
    if q < 2.0:
        rest = 1.0 - 0.5 * q
        return -5.0 * sigma * q * rest * rest * rest / h
    return 0.0

@wp.func
def kernelValue(r: float, h: float, sigma: float, kind: int) -> float:

    '''W(r) of the kernel `kind` names: 0 the cubic spline, 1 the Wendland C2.'''

    if kind == 1:
        return wendlandValue(r, h, sigma)
    return cubicValue(r, h, sigma)

@wp.func
def kernelSlope(r: float, h: float, sigma: float, kind: int) -> float:

    '''dW/dr of the kernel `kind` names: 0 the cubic spline, 1 the Wendland C2.'''

    if kind == 1:
        return wendlandSlope(r, h, sigma)
    return cubicSlope(r, h, sigma)

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

@wp.func
def normalization(unit: float, h: float, dims: int) -> float:

    '''The kernel's sigma at smoothing length h: its sigma at h = 1 over h^dims.'''

    if dims == 2:
        return unit / (h * h)
    return unit / (h * h * h)

#----------------------------------------------------------------------#
# -- Pair Sums -- #
#----------------------------------------------------------------------#

@wp.kernel
def accelerationKernel(grid: wp.uint64, points: wp.array(dtype = wp.vec3), lows: wp.array(dtype = wp.vec3),
                       precise: int, period: float, kind: int, velocities: wp.array(dtype = wp.vec3),
                       densities: wp.array(dtype = float), pressures: wp.array(dtype = float),
                       masses: wp.array(dtype = float), fluid: wp.array(dtype = wp.uint8),
                       lengths: wp.array(dtype = float), maxLength: float, sigmaUnit: float, dims: int,
                       soundSpeed: float, alpha: float, boundaryFactor: float, gravity: wp.vec3,
                       gradients: wp.array(dtype = wp.mat33), limited: int, result: wp.array(dtype = wp.vec3)):

    '''

    a_i = -sum_j m_j (p_i / rho_i^2 + p_j / rho_j^2 + Pi_ij) grad W_ij + g on the fluid, zero on the
    boundary, each pair at h_ij = (h_i + h_j) / 2. With `limited`, a fluid pair's Pi_ij takes its
    velocity difference less its linear part from the velocity `gradients`
    (`sphSolver.limitedVelocityDifference`) and acts where both differences approach.

    '''

    i = wp.hash_grid_point_id(grid, wp.tid())
    if fluid[i] == wp.uint8(0):
        result[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    lengthI = lengths[i]
    reach = lengthI + maxLength
    limit = candidateMargin * reach * reach
    pointI = points[i]
    lowI = lows[i]
    velocityI = velocities[i]
    densityI = densities[i]
    own = pressures[i] / (densityI * densityI)
    gradientI = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if limited != 0:
        gradientI = gradients[i]
    total = wp.vec3(0.0, 0.0, 0.0)
    for image in range(imageCount(pointI, period, reach)):
        offset = wp.vec3(0.0, imageShift(pointI, period, image), 0.0)
        origin = pointI + offset
        query = wp.hash_grid_query(grid, origin, reach)
        j = int(0)
        while wp.hash_grid_query_next(query, j):
            pointJ = points[j]
            coarse = origin - pointJ
            if wp.dot(coarse, coarse) >= limit or j == i:
                continue
            d = separation(coarse, pointI, pointJ, lowI, lows[j], offset, precise)
            r = wp.length(d)
            h = 0.5 * (lengthI + lengths[j])
            if r >= 2.0 * h or r < 1.0e-12:
                continue
            densityJ = densities[j]
            approach = wp.dot(velocityI - velocities[j], d)
            viscosity = float(0.0)
            if approach < 0.0 and limited != 0 and fluid[j] != wp.uint8(0):
                rampI = gradientI * d
                rampJ = gradients[j] * d
                slopeI = wp.dot(d, rampI)
                slopeJ = wp.dot(d, rampJ)
                limiter = float(0.0)
                if slopeI * slopeJ > 0.0:
                    ratio = slopeI / slopeJ
                    limiter = wp.min(1.0, 4.0 * ratio / ((1.0 + ratio) * (1.0 + ratio)))
                    eta = r / h
                    if eta < closeDistance:
                        fade = (eta - closeDistance) / foldWidth
                        limiter = limiter * wp.exp(-fade * fade)
                approach = wp.dot(velocityI - velocities[j] - (0.5 * limiter) * (rampI + rampJ), d)
            if approach < 0.0:
                mu = h * approach / (r * r + viscositySoftening * h * h)
                viscosity = -alpha * soundSpeed * mu / (0.5 * (densityI + densityJ))
                if fluid[j] == wp.uint8(0):
                    viscosity = viscosity * boundaryFactor
            coefficient = -(own + pressures[j] / (densityJ * densityJ) + viscosity) * masses[j]
            total += coefficient * kernelSlope(r, h, normalization(sigmaUnit, h, dims), kind) / r * d
    result[i] = total + gravity

@wp.func
def hydrostaticDifference(vertical: float, meanDeviation: float, referenceDensity: float, gravityOverSound: float) -> float:

    '''

    `sphSolver.hydrostaticDensityDifference`: rho0 (g / c0^2) (z_i - z_j) (rho_bar / rho0)^(2 - gamma),
    gamma = 7, with rho_bar / rho0 = 1 + the mean deviation over rho0.

    '''

    ratio = 1.0 + meanDeviation / referenceDensity
    square = ratio * ratio
    return referenceDensity * gravityOverSound * vertical / (square * square * ratio)

@wp.kernel
def continuityKernel(grid: wp.uint64, points: wp.array(dtype = wp.vec3), lows: wp.array(dtype = wp.vec3),
                     precise: int, period: float, kind: int, velocities: wp.array(dtype = wp.vec3),
                     masses: wp.array(dtype = float), fluid: wp.array(dtype = wp.uint8),
                     lengths: wp.array(dtype = float), maxLength: float, sigmaUnit: float, dims: int,
                     diffusion: int, diffusionFactor: float, densities: wp.array(dtype = float),
                     deviations: wp.array(dtype = float), referenceDensity: float, gravityOverSound: float,
                     vertical: int, corrections: wp.array(dtype = wp.mat33), corrected: int,
                     result: wp.array(dtype = float)):

    '''

    d rho_i / dt = sum_j m_j (v_i - v_j) . grad W_ij over pairs with a fluid member, and with
    `diffusion` the delta-SPH term over fluid pairs (`sphSolver.densityDiffusionRate`): 1 for
    Molteni and Colagrossi's, 2 with the hydrostatic part subtracted. `diffusionFactor` is
    2 delta c0, which each pair takes times its h_ij; `deviations` are rho - rho0, so neighbors'
    differences keep their digits. With `corrected`, grad W_ij becomes L_i grad W_ij
    (`correctionKernel`). Each pair is taken at h_ij = (h_i + h_j) / 2.

    '''

    i = wp.hash_grid_point_id(grid, wp.tid())
    lengthI = lengths[i]
    reach = lengthI + maxLength
    limit = candidateMargin * reach * reach
    pointI = points[i]
    lowI = lows[i]
    velocityI = velocities[i]
    boundaryI = fluid[i] == wp.uint8(0)
    diffuses = diffusion != 0 and not boundaryI
    deviationI = float(0.0)
    if diffuses:
        deviationI = deviations[i]
    correction = wp.identity(n = 3, dtype = float)
    if corrected != 0:
        correction = corrections[i]
    total = float(0.0)
    for image in range(imageCount(pointI, period, reach)):
        offset = wp.vec3(0.0, imageShift(pointI, period, image), 0.0)
        origin = pointI + offset
        query = wp.hash_grid_query(grid, origin, reach)
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
            h = 0.5 * (lengthI + lengths[j])
            if r >= 2.0 * h or r < 1.0e-12:
                continue
            slope = kernelSlope(r, h, normalization(sigmaUnit, h, dims), kind)
            if corrected != 0:
                total += masses[j] * wp.dot(velocityI - velocities[j], correction * d) * slope / r
            else:
                total += masses[j] * wp.dot(velocityI - velocities[j], d) * slope / r
            if diffuses and fluid[j] != wp.uint8(0):
                deviationJ = deviations[j]
                difference = deviationJ - deviationI
                if diffusion == 2:
                    difference -= hydrostaticDifference(d[vertical], 0.5 * (deviationI + deviationJ), referenceDensity,
                                                        gravityOverSound)
                total += diffusionFactor * h * masses[j] / densities[j] * difference * (-slope * r) \
                    / (r * r + viscositySoftening * h * h)
    result[i] = total

@wp.kernel
def correctionKernel(grid: wp.uint64, points: wp.array(dtype = wp.vec3), lows: wp.array(dtype = wp.vec3),
                     precise: int, period: float, kind: int, masses: wp.array(dtype = float),
                     densities: wp.array(dtype = float), fluid: wp.array(dtype = wp.uint8),
                     lengths: wp.array(dtype = float), maxLength: float, sigmaUnit: float, dims: int,
                     planar: int, floor: float, result: wp.array(dtype = wp.mat33)):

    '''

    `sphSolver.correctionMatrices`: L_i = M_i^-1, M_i = sum_j V_j (x_j - x_i) (x) grad_i W_ij, the
    identity on the boundary and where det M < `floor`. In 2D (`planar`) the unused third axis
    carries a one on M's diagonal. Each pair is taken at h_ij = (h_i + h_j) / 2.

    '''

    i = wp.hash_grid_point_id(grid, wp.tid())
    identity = wp.identity(n = 3, dtype = float)
    if fluid[i] == wp.uint8(0):
        result[i] = identity
        return
    lengthI = lengths[i]
    reach = lengthI + maxLength
    limit = candidateMargin * reach * reach
    pointI = points[i]
    lowI = lows[i]
    total = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for image in range(imageCount(pointI, period, reach)):
        offset = wp.vec3(0.0, imageShift(pointI, period, image), 0.0)
        origin = pointI + offset
        query = wp.hash_grid_query(grid, origin, reach)
        j = int(0)
        while wp.hash_grid_query_next(query, j):
            pointJ = points[j]
            coarse = origin - pointJ
            if wp.dot(coarse, coarse) >= limit or j == i:
                continue
            d = separation(coarse, pointI, pointJ, lowI, lows[j], offset, precise)
            r = wp.length(d)
            h = 0.5 * (lengthI + lengths[j])
            if r >= 2.0 * h or r < 1.0e-12:
                continue
            slope = kernelSlope(r, h, normalization(sigmaUnit, h, dims), kind)
            total += (-masses[j] / densities[j] * slope / r) * wp.outer(d, d)
    if planar != 0:
        total += wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    if wp.determinant(total) >= floor:
        result[i] = wp.inverse(total)
    else:
        result[i] = identity

@wp.kernel
def velocityGradientKernel(grid: wp.uint64, points: wp.array(dtype = wp.vec3), lows: wp.array(dtype = wp.vec3),
                           precise: int, period: float, kind: int, velocities: wp.array(dtype = wp.vec3),
                           masses: wp.array(dtype = float), densities: wp.array(dtype = float),
                           fluid: wp.array(dtype = wp.uint8), lengths: wp.array(dtype = float), maxLength: float,
                           sigmaUnit: float, dims: int, planar: int, floor: float,
                           result: wp.array(dtype = wp.mat33)):

    '''

    `sphSolver.velocityGradients`: G_i = S_i M_i^-1 over fluid neighbors, S_i = sum_j V_j
    (v_j - v_i) (x) grad_i W_ij and M_i = sum_j V_j (x_j - x_i) (x) grad_i W_ij; zero on the
    boundary and where det M < `floor`. In 2D (`planar`) the unused third axis carries a one on M's
    diagonal. Each pair is taken at h_ij = (h_i + h_j) / 2.

    '''

    i = wp.hash_grid_point_id(grid, wp.tid())
    zero = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if fluid[i] == wp.uint8(0):
        result[i] = zero
        return
    lengthI = lengths[i]
    reach = lengthI + maxLength
    limit = candidateMargin * reach * reach
    pointI = points[i]
    lowI = lows[i]
    velocityI = velocities[i]
    rises = zero
    offsets = zero
    for image in range(imageCount(pointI, period, reach)):
        offset = wp.vec3(0.0, imageShift(pointI, period, image), 0.0)
        origin = pointI + offset
        query = wp.hash_grid_query(grid, origin, reach)
        j = int(0)
        while wp.hash_grid_query_next(query, j):
            pointJ = points[j]
            coarse = origin - pointJ
            if wp.dot(coarse, coarse) >= limit or j == i or fluid[j] == wp.uint8(0):
                continue
            d = separation(coarse, pointI, pointJ, lowI, lows[j], offset, precise)
            r = wp.length(d)
            h = 0.5 * (lengthI + lengths[j])
            if r >= 2.0 * h or r < 1.0e-12:
                continue
            weight = masses[j] / densities[j] * kernelSlope(r, h, normalization(sigmaUnit, h, dims), kind) / r
            # grad_i W_ij = slope d / r, and x_j - x_i = -d
            rises += weight * wp.outer(velocities[j] - velocityI, d)
            offsets += -weight * wp.outer(d, d)
    if planar != 0:
        offsets += wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    if wp.determinant(offsets) >= floor:
        result[i] = rises * wp.inverse(offsets)
    else:
        result[i] = zero

@wp.kernel
def xsphKernel(grid: wp.uint64, points: wp.array(dtype = wp.vec3), lows: wp.array(dtype = wp.vec3),
               precise: int, period: float, kind: int, velocities: wp.array(dtype = wp.vec3), densities: wp.array(dtype = float),
               masses: wp.array(dtype = float), fluid: wp.array(dtype = wp.uint8),
               lengths: wp.array(dtype = float), maxLength: float, sigmaUnit: float, dims: int,
               epsilon: float, result: wp.array(dtype = wp.vec3)):

    '''v_i + epsilon sum_j (m_j / rho_bar_ij) (v_j - v_i) W_ij over fluid pairs; boundary rows unchanged.'''

    i = wp.hash_grid_point_id(grid, wp.tid())
    velocityI = velocities[i]
    if fluid[i] == wp.uint8(0):
        result[i] = velocityI
        return
    lengthI = lengths[i]
    reach = lengthI + maxLength
    limit = candidateMargin * reach * reach
    pointI = points[i]
    lowI = lows[i]
    densityI = densities[i]
    correction = wp.vec3(0.0, 0.0, 0.0)
    for image in range(imageCount(pointI, period, reach)):
        offset = wp.vec3(0.0, imageShift(pointI, period, image), 0.0)
        origin = pointI + offset
        query = wp.hash_grid_query(grid, origin, reach)
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
            h = 0.5 * (lengthI + lengths[j])
            if r >= 2.0 * h:
                continue
            value = kernelValue(r, h, normalization(sigmaUnit, h, dims), kind)
            weight = masses[j] * value / (0.5 * (densityI + densities[j]))
            correction += weight * (velocities[j] - velocityI)
    result[i] = velocityI + epsilon * correction

@wp.kernel
def shepardKernel(grid: wp.uint64, points: wp.array(dtype = wp.vec3), lows: wp.array(dtype = wp.vec3),
                  precise: int, period: float, kind: int, densities: wp.array(dtype = float), masses: wp.array(dtype = float),
                  lengths: wp.array(dtype = float), maxLength: float, sigmaUnit: float, dims: int, result: wp.array(dtype = float)):

    '''rho_i = sum_j m_j W_ij / sum_j (m_j / rho_j) W_ij, the particle itself included, at h_ij = (h_i + h_j) / 2.'''

    i = wp.hash_grid_point_id(grid, wp.tid())
    lengthI = lengths[i]
    reach = lengthI + maxLength
    limit = candidateMargin * reach * reach
    pointI = points[i]
    lowI = lows[i]
    own = normalization(sigmaUnit, lengthI, dims)
    numerator = masses[i] * own
    denominator = masses[i] / densities[i] * own
    for image in range(imageCount(pointI, period, reach)):
        offset = wp.vec3(0.0, imageShift(pointI, period, image), 0.0)
        origin = pointI + offset
        query = wp.hash_grid_query(grid, origin, reach)
        j = int(0)
        while wp.hash_grid_query_next(query, j):
            pointJ = points[j]
            coarse = origin - pointJ
            if wp.dot(coarse, coarse) >= limit or j == i:
                continue
            d = separation(coarse, pointI, pointJ, lowI, lows[j], offset, precise)
            r = wp.length(d)
            h = 0.5 * (lengthI + lengths[j])
            if r >= 2.0 * h:
                continue
            value = kernelValue(r, h, normalization(sigmaUnit, h, dims), kind)
            numerator += masses[j] * value
            denominator += masses[j] / densities[j] * value
    result[i] = numerator / denominator

@wp.kernel
def wallPressureKernel(grid: wp.uint64, points: wp.array(dtype = wp.vec3), lows: wp.array(dtype = wp.vec3),
                       precise: int, period: float, kind: int, fluid: wp.array(dtype = wp.uint8),
                       densities: wp.array(dtype = float), pressures: wp.array(dtype = float),
                       lengths: wp.array(dtype = float), maxLength: float, sigmaUnit: float, dims: int, gravity: wp.vec3, referenceDensity: wp.float64, stiffness: wp.float64,
                       densities64: wp.array(dtype = wp.float64)):

    '''

    `sphSolver.extrapolatedWallPressure` on each boundary particle, with the density that carries
    the pressure. Only boundary entries are written and only fluid entries read.

    '''

    i = wp.hash_grid_point_id(grid, wp.tid())
    if fluid[i] != wp.uint8(0):
        return
    lengthI = lengths[i]
    reach = lengthI + maxLength
    limit = candidateMargin * reach * reach
    pointI = points[i]
    lowI = lows[i]
    weight = float(0.0)
    total = float(0.0)
    for image in range(imageCount(pointI, period, reach)):
        offset = wp.vec3(0.0, imageShift(pointI, period, image), 0.0)
        origin = pointI + offset
        query = wp.hash_grid_query(grid, origin, reach)
        j = int(0)
        while wp.hash_grid_query_next(query, j):
            if fluid[j] == wp.uint8(0):
                continue
            pointJ = points[j]
            coarse = origin - pointJ
            if wp.dot(coarse, coarse) >= limit:
                continue
            d = separation(coarse, pointI, pointJ, lowI, lows[j], offset, precise)
            r = wp.length(d)
            h = 0.5 * (lengthI + lengths[j])
            if r >= 2.0 * h:
                continue
            value = kernelValue(r, h, normalization(sigmaUnit, h, dims), kind)
            weight += value
            total += value * (pressures[j] + densities[j] * wp.dot(gravity, d))
    pressure = float(0.0)
    if weight > 0.0:
        pressure = wp.max(total / weight, 0.0)
    pressures[i] = pressure
    density = referenceDensity * wp.pow(wp.float64(1.0) + wp.float64(pressure) / stiffness, wp.float64(1.0 / 7.0))
    densities64[i] = density
    densities[i] = wp.float32(density)

@wp.kernel
def shiftingFieldsKernel(grid: wp.uint64, points: wp.array(dtype = wp.vec3), lows: wp.array(dtype = wp.vec3),
                         precise: int, period: float, kind: int, densities: wp.array(dtype = float),
                         deviations: wp.array(dtype = float), masses: wp.array(dtype = float),
                         fluid: wp.array(dtype = wp.uint8), h: float, sigma: float, spacingKernel: float,
                         antiClumping: float, lopsidedness: float, gradients: wp.array(dtype = wp.vec3),
                         normals: wp.array(dtype = wp.vec3), divergence: wp.array(dtype = float),
                         densityGradients: wp.array(dtype = wp.vec3)):

    '''

    `sphSolver.shiftingFields` on the fluid, over every neighbor: grad C with the anti-clumping
    weight, grad C0 without it, div r and grad rho, its differences from the deviations rho - rho0.
    div r marks the free surface for `shiftKernel`: a lopsided support (`sphSolver.freeSurface`)
    sets it to -1, and a boundary particle's is set high, so no boundary particle reads as one.

    '''

    i = wp.hash_grid_point_id(grid, wp.tid())
    zero = wp.vec3(0.0, 0.0, 0.0)
    if fluid[i] == wp.uint8(0):
        gradients[i] = zero
        normals[i] = zero
        divergence[i] = 1.0e6
        densityGradients[i] = zero
        return
    support = 2.0 * h
    limit = candidateMargin * support * support
    pointI = points[i]
    lowI = lows[i]
    deviationI = deviations[i]
    gradient = zero
    plain = zero
    densityGradient = zero
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
            d = separation(coarse, pointI, pointJ, lowI, lows[j], offset, precise)
            r = wp.length(d)
            if r >= support or r < 1.0e-12:
                continue
            volume = masses[j] / densities[j]
            slope = kernelSlope(r, h, sigma, kind)
            ratio = kernelValue(r, h, sigma, kind) / spacingKernel
            square = ratio * ratio
            weighted = volume * slope / r * d
            plain += weighted
            gradient += (1.0 + antiClumping * square * square) * weighted
            densityGradient += (deviations[j] - deviationI) * weighted
            total -= volume * slope * r
    gradients[i] = gradient
    normals[i] = plain
    if h * wp.length(plain) > lopsidedness:
        total = -1.0
    divergence[i] = total
    densityGradients[i] = densityGradient

@wp.kernel
def shiftKernel(grid: wp.uint64, points: wp.array(dtype = wp.vec3), lows: wp.array(dtype = wp.vec3), precise: int,
                period: float, kind: int, fluid: wp.array(dtype = wp.uint8), divergence: wp.array(dtype = float),
                gradients: wp.array(dtype = wp.vec3), normals: wp.array(dtype = wp.vec3),
                densityGradients: wp.array(dtype = wp.vec3), velocities: wp.array(dtype = wp.vec3),
                densities: wp.array(dtype = float), masses: wp.array(dtype = float), h: float, sigma: float,
                threshold: float, surfaceGradient: float, size: float, local: int, shifts: wp.array(dtype = wp.vec3),
                densityChanges: wp.array(dtype = float), velocityChanges: wp.array(dtype = wp.vec3)):

    '''

    `sphSolver.shiftingDisplacement`: no shift on the free surface or the boundary, the shift along
    the surface with a surface particle within the support, across the normal of
    `sphSolver.shiftingDisplacement`, the full shift elsewhere, and the density and velocity changes
    that go with it (`sphSolver.shiftedVelocityChange`).

    '''

    i = wp.hash_grid_point_id(grid, wp.tid())
    zero = wp.vec3(0.0, 0.0, 0.0)
    if fluid[i] == wp.uint8(0) or divergence[i] < threshold:
        shifts[i] = zero
        densityChanges[i] = 0.0
        velocityChanges[i] = zero
        return
    support = 2.0 * h
    limit = candidateMargin * support * support
    pointI = points[i]
    lowI = lows[i]
    vicinity = int(0)
    normal = zero
    for image in range(imageCount(pointI, period, support)):
        offset = wp.vec3(0.0, imageShift(pointI, period, image), 0.0)
        origin = pointI + offset
        query = wp.hash_grid_query(grid, origin, support)
        j = int(0)
        while wp.hash_grid_query_next(query, j):
            pointJ = points[j]
            coarse = origin - pointJ
            if wp.dot(coarse, coarse) >= limit or j == i or divergence[j] >= threshold:
                continue
            d = separation(coarse, pointI, pointJ, lowI, lows[j], offset, precise)
            r = wp.length(d)
            if r < support:
                vicinity = 1
                plain = normals[j]
                length = wp.length(plain)
                if length > 0.0:
                    normal -= kernelValue(r, h, sigma, kind) / length * plain
    scale = size
    if local != 0:
        scale = size * wp.length(velocities[i])
    shift = -scale * gradients[i]
    if vicinity != 0:
        if h * wp.length(gradients[i]) > surfaceGradient:
            shift = zero
        weight = wp.length(normal)
        if weight > 0.0:
            normal = normal / weight
            shift = shift - wp.dot(shift, normal) * normal
    shifts[i] = shift
    densityChanges[i] = wp.dot(shift, densityGradients[i])

    # The velocity at the new position: sum_j V_j (v_j - v_i) (grad W_ij . delta r_i) over fluid neighbors
    velocityI = velocities[i]
    change = zero
    if wp.length(shift) > 0.0:
        for image in range(imageCount(pointI, period, support)):
            offset = wp.vec3(0.0, imageShift(pointI, period, image), 0.0)
            origin = pointI + offset
            query = wp.hash_grid_query(grid, origin, support)
            j = int(0)
            while wp.hash_grid_query_next(query, j):
                pointJ = points[j]
                coarse = origin - pointJ
                if wp.dot(coarse, coarse) >= limit or j == i or fluid[j] == wp.uint8(0):
                    continue
                d = separation(coarse, pointI, pointJ, lowI, lows[j], offset, precise)
                r = wp.length(d)
                if r >= support or r < 1.0e-12:
                    continue
                slope = kernelSlope(r, h, sigma, kind)
                change += masses[j] / densities[j] * slope / r * wp.dot(d, shift) * (velocities[j] - velocityI)
    velocityChanges[i] = change

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
def toDeviation(densities: wp.array(dtype = wp.float64), referenceDensity: wp.float64,
                deviations: wp.array(dtype = float)):

    '''rho - rho0 in float32, taken in float64, for the diffusive term's differences.'''

    i = wp.tid()
    deviations[i] = wp.float32(densities[i] - referenceDensity)

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
                    fluid: wp.array(dtype = wp.uint8), tension: int, pressures: wp.array(dtype = float)):

    '''Tait pressure B ((rho / rho0)^7 - 1), negative pressure clamped to zero except in the fluid with `tension`.'''

    i = wp.tid()
    ratio = densities[i] / referenceDensity
    square = ratio * ratio
    pressure = stiffness * (square * square * square * ratio - wp.float64(1.0))
    if tension == 0 or fluid[i] == wp.uint8(0):
        pressure = wp.max(pressure, wp.float64(0.0))
    pressures[i] = wp.float32(pressure)

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
          velocities: wp.array(dtype = wp.vec3d), useTransport: int, shifts: wp.array(dtype = wp.vec3),
          densityChanges: wp.array(dtype = float), velocityChanges: wp.array(dtype = wp.vec3), useShifts: int,
          densities: wp.array(dtype = wp.float64), fluid: wp.array(dtype = wp.uint8), timeStep: wp.float64,
          period: wp.float64):

    '''

    x += u dt on the fluid, u the XSPH transport velocity or the velocity, plus the shift and its
    density and velocity changes when shifting; y wrapped onto the period.

    '''

    i = wp.tid()
    if fluid[i] == wp.uint8(0):
        return
    velocity = velocities[i]
    if useTransport != 0:
        velocity = double(transport[i])
    x = positions[i] + velocity * timeStep
    if useShifts != 0:
        x = x + double(shifts[i])
        densities[i] = densities[i] + wp.float64(densityChanges[i])
        velocities[i] = velocities[i] + double(velocityChanges[i])
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

def wendlandNormalization(smoothingLength: float, dimensions: int) -> float:

    '''The Wendland C2's sigma: 7 / (4 pi h^2) in 2D, 21 / (16 pi h^3) in 3D.'''

    h = smoothingLength
    return 7.0 / (4.0 * math.pi * h * h) if dimensions == 2 else 21.0 / (16.0 * math.pi * h * h * h)

def kernelNormalization(kernelType: str, smoothingLength: float, dimensions: int) -> float:

    '''The sigma of 'cubicSpline' or 'wendlandC2'.'''

    if kernelType == 'wendlandC2':
        return wendlandNormalization(smoothingLength, dimensions)
    return cubicSplineNormalization(smoothingLength, dimensions)

def hostKernelValue(kernelType: str, r: float, smoothingLength: float, dimensions: int) -> float:

    '''W(r) on the host, as `kernelValue` gives it on the device.'''

    if kernelType == 'wendlandC2':
        q = r / smoothingLength
        return wendlandNormalization(smoothingLength, dimensions) * max(1.0 - 0.5 * q, 0.0)**4 * (2.0 * q + 1.0)
    return cubicSplineValue(r, smoothingLength, dimensions)

def cubicSplineValue(r: float, smoothingLength: float, dimensions: int) -> float:

    '''The cubic spline W(r) on the host, as `cubicValue` gives it on the device.'''

    q = r / smoothingLength
    sigma = cubicSplineNormalization(smoothingLength, dimensions)
    if q < 1.0:
        return sigma * (1.0 - 1.5 * q * q + 0.75 * q**3)
    return sigma * 0.25 * max(2.0 - q, 0.0)**3

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
    kernelType : str
        'cubicSpline' or 'wendlandC2', both supported on 2h
    device : str | None
        Warp device, 'cuda:0' or 'cpu'; None for Warp's preferred device

    '''

    def __init__(self, dimensions: int, smoothingLength: float, domainMin: np.ndarray, domainMax: np.ndarray,
                 period: float = 0.0, separationPrecision: str = 'float64', device: str | None = None,
                 kernelType: str = 'cubicSpline') -> None:

        if separationPrecision not in ('float64', 'float32'):
            raise ValueError(f"Separation precision '{separationPrecision}' is not 'float64' or 'float32'.")
        if kernelType not in ('cubicSpline', 'wendlandC2'):
            raise ValueError(f"Kernel '{kernelType}' is not 'cubicSpline' or 'wendlandC2'.")
        if period > 0.0 and dimensions != 3:
            raise ValueError('A periodic axis needs three dimensions: it is y.')
        self.dimensions = dimensions
        self.smoothingLength = float(smoothingLength)                    # [m]
        self.kernelType = kernelType
        self.kind = 1 if kernelType == 'wendlandC2' else 0
        self.sigma = kernelNormalization(kernelType, smoothingLength, dimensions)
        self.sigmaUnit = kernelNormalization(kernelType, 1.0, dimensions)    # sigma h^dims
        self.maxLength = self.smoothingLength                            # [m], sizes the grid's cells
        self.minLength = self.smoothingLength                            # [m], sets the time step
        self.uniformLengths = True
        self.period = float(period)                                      # [m]
        self.precise = 1 if separationPrecision == 'float64' else 0
        self.device = wp.get_device(device)
        self.grid = wp.HashGrid(*gridDimensions(domainMin, domainMax, 2.0 * smoothingLength, dimensions),
                                device = self.device)
        self.count = 0
        self.physics: dict = {}
        self.diffusion = {'mode': 0, 'factor': 0.0, 'referenceDensity': 1000.0, 'gravityOverSound': 0.0}
        self.shifting = {'enabled': False, 'mach': 0.0, 'spacingKernel': 1.0, 'soundSpeed': 1.0,
                         'threshold': freeSurfaceDivergence[dimensions], 'local': False}
        self.extrapolateWalls = False
        self.negativePressure = False
        self.correctContinuity = False
        self.limitViscosity = False
        self._noCorrections = wp.zeros(1, dtype = wp.mat33, device = self.device)
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
        fields = {**layout, **shiftingLayout} if self.shifting['enabled'] else dict(layout)
        if self.correctContinuity:
            fields['corrections'] = wp.mat33
        if self.limitViscosity:
            fields['gradients'] = wp.mat33
        self._device = {name: wp.zeros(capacity, dtype = dtype, device = self.device) for name, dtype in fields.items()}
        self._staging = {}
        self._capacity = capacity

    def _stage(self, name: str) -> wp.array:

        '''The pinned host array a field passes through, made on first use: only host steps need them.'''

        if name not in self._staging:
            dtype = layout[name] if name in layout else shiftingLayout[name]
            self._staging[name] = wp.zeros(self._capacity, dtype = dtype, device = 'cpu', pinned = self.device.is_cuda)
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
        '''The pair kernels' leading arguments: grid, points, remainders, precision flag, period, kernel.'''
        d = self._device
        return [self.grid.id, d['points'], d['lows'], self.precise, self.period, self.kind]

    @property
    def _lengthInputs(self) -> list:
        '''The converted pair kernels' smoothing-length arguments: each particle's h, the largest, sigma h^dims, dims.'''
        return [self._device['lengths'], self.maxLength, self.sigmaUnit, self.dimensions]

    @property
    def _diffusionInputs(self) -> list:
        '''The continuity kernel's diffusion and correction arguments, after the kernel constants.'''
        d, diffusion = self._device, self.diffusion
        corrections = d['corrections'] if self.correctContinuity else self._noCorrections
        return [diffusion['mode'], diffusion['factor'], d['densities'], d['deviations'], diffusion['referenceDensity'],
                diffusion['gravityOverSound'], self.dimensions - 1, corrections, int(self.correctContinuity)]

    def _correct(self) -> None:

        '''This step's correction matrices into 'corrections', from the float32 densities on the device.'''

        d = self._device
        self._launch(correctionKernel, self._geometry + [d['masses'], d['densities'], d['fluid']] + self._lengthInputs
                     + [1 if self.dimensions == 2 else 0, correctionFloor**self.dimensions, d['corrections']])

    def setContinuityCorrection(self, enabled: bool) -> None:

        '''Whether the continuity sum takes the kernel gradient correction (`sphSolver.correctionMatrices`).'''

        self.correctContinuity = bool(enabled)
        if enabled and self._capacity and 'corrections' not in self._device:
            self._device['corrections'] = wp.zeros(self._capacity, dtype = wp.mat33, device = self.device)

    def _velocityGradients(self) -> None:

        '''This step's velocity gradients into 'gradients', from the float32 velocities and densities on the device.'''

        d = self._device
        self._launch(velocityGradientKernel, self._geometry + [d['velocities'], d['masses'], d['densities'], d['fluid']]
                     + self._lengthInputs + [1 if self.dimensions == 2 else 0, correctionFloor**self.dimensions,
                                             d['gradients']])

    @property
    def _viscosityInputs(self) -> list:
        '''The acceleration kernel's limiter arguments: the velocity gradients and the flag.'''
        gradients = self._device['gradients'] if self.limitViscosity else self._noCorrections
        return [gradients, int(self.limitViscosity)]

    def setViscosityLimiter(self, enabled: bool) -> None:

        '''Whether the artificial viscosity takes the limited velocity difference (`sphSolver.limitedVelocityDifference`).'''

        self.limitViscosity = bool(enabled)
        if enabled and self._capacity and 'gradients' not in self._device:
            self._device['gradients'] = wp.zeros(self._capacity, dtype = wp.mat33, device = self.device)

    def velocityGradients(self, velocities: np.ndarray, densities: np.ndarray) -> np.ndarray:

        '''G for every particle at the current positions, shape (N, dim, dim) (`sphSolver.velocityGradients`).'''

        if not self.limitViscosity:
            raise ValueError('The viscosity limiter is off; see setViscosityLimiter.')
        self._upload('velocities', velocities)
        self._upload('densities', densities)
        self._velocityGradients()
        wp.synchronize_device(self.device)
        dim = self.dimensions

        return self._device['gradients'].numpy()[:self.count, :dim, :dim].astype(float)

    def correctionMatrices(self, densities: np.ndarray) -> np.ndarray:

        '''L for every particle at the current positions, shape (N, dim, dim) (`sphSolver.correctionMatrices`).'''

        if not self.correctContinuity:
            raise ValueError('The continuity correction is off; see setContinuityCorrection.')
        self._upload('densities', densities)
        self._correct()
        wp.synchronize_device(self.device)
        dim = self.dimensions

        return self._device['corrections'].numpy()[:self.count, :dim, :dim].astype(float)

    def setDiffusion(self, mode: str, delta: float, soundSpeed: float, referenceDensity: float, gravity: float) -> None:

        '''

        The delta-SPH term the continuity sum carries (`sphSolver.densityDiffusionRate`).

        Parameters:
        -----------
        mode : str
            'none', 'molteni' or 'hydrostatic'
        delta : float
            The coefficient delta
        soundSpeed : float
            c0 [m/s]
        referenceDensity : float
            rho0 [kg/m^3]
        gravity : float
            Gravity's magnitude along the last axis [m/s^2]

        '''

        # Each pair takes the factor times its own h_ij
        self.diffusion = {'mode': diffusionModes[mode], 'factor': 2.0 * delta * soundSpeed,
                          'referenceDensity': float(referenceDensity), 'gravityOverSound': gravity / soundSpeed**2}

    def setWallPressure(self, extrapolated: bool) -> None:

        '''

        Whether the device steps take each boundary particle's pressure from the fluid around it
        (`sphSolver.extrapolatedWallPressure`) rather than from its integrated density.

        '''

        self.extrapolateWalls = bool(extrapolated)

    def setNegativePressure(self, enabled: bool) -> None:

        '''Whether the device steps let the fluid carry negative pressure (`SimulationConfig.negativePressure`).'''

        self.negativePressure = bool(enabled)

    def setShifting(self, enabled: bool, mach: float, spacing: float, soundSpeed: float, local: bool = False) -> None:

        '''

        Particle shifting (`sphSolver.shiftingDisplacement`) for every step that follows.

        Parameters:
        -----------
        enabled : bool
            Whether the fluid shifts
        mach : float
            Ma in the shift's size CFL Ma (2h)^2
        spacing : float
            Particle spacing dx, for the anti-clumping weight's W(dx) [m]
        soundSpeed : float
            c0, for the step's Courant number c0 dt / h [m/s]
        local : bool
            Size each particle's shift by its own speed, A h |v_i| dt, rather than CFL Ma (2h)^2

        '''

        self.shifting = {'enabled': bool(enabled), 'mach': float(mach),
                         'spacingKernel': hostKernelValue(self.kernelType, spacing, self.smoothingLength,
                                                          self.dimensions),
                         'soundSpeed': float(soundSpeed), 'threshold': freeSurfaceDivergence[self.dimensions],
                         'local': bool(local)}
        if enabled and not self.uniformLengths:
            raise ValueError('Particle shifting needs one smoothing length for every particle.')
        if enabled and self._capacity and 'shifts' not in self._device:
            for name, dtype in shiftingLayout.items():
                self._device[name] = wp.zeros(self._capacity, dtype = dtype, device = self.device)

    def _shift(self, timeStep: float) -> None:

        '''This step's shifts into the 'shifts' array, from the positions of the last grid build.'''

        d, shifting, h = self._device, self.shifting, self.smoothingLength
        if shifting['local']:
            size = localShiftingCoefficient * h * timeStep
        else:
            size = (shifting['soundSpeed'] * timeStep / h) * shifting['mach'] * (2.0 * h)**2
        if antiClumpingPower != 4:
            raise ValueError('The device anti-clumping weight is written for the power 4.')
        self._launch(shiftingFieldsKernel, self._geometry + [
            d['densities'], d['deviations'], d['masses'], d['fluid'], h, self.sigma, shifting['spacingKernel'],
            antiClumpingWeight, freeSurfaceLopsidedness[self.kernelType], d['gradients'], d['normals'], d['divergence'],
            d['densityGradients']])
        self._launch(shiftKernel, self._geometry + [d['fluid'], d['divergence'], d['gradients'], d['normals'],
                                                    d['densityGradients'], d['velocities'], d['densities'],
                                                    d['masses'], h, self.sigma, shifting['threshold'], surfaceGradient,
                                                    size, 1 if shifting['local'] else 0, d['shifts'],
                                                    d['densityChanges'], d['velocityChanges']])

    def shiftingDisplacement(self, densities: np.ndarray, timeStep: float,
                             velocities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        '''

        This step's shift of every particle, zero on the boundary and the free surface [m], and the
        density [kg/m^3] and velocity [m/s] changes that go with it.

        '''

        self._upload('velocities', velocities)
        self._upload('densities', densities)
        self._upload('deviations', np.asarray(densities, dtype = float) - self.diffusion['referenceDensity'])
        self._shift(timeStep)

        return self._download('shifts'), self._download('densityChanges'), self._download('velocityChanges')

    # -- Host Steps -- #
    def update(self, positions: np.ndarray, masses: np.ndarray, isFluid: np.ndarray,
               lengths: np.ndarray | None = None) -> None:

        '''

        Copy the positions, masses, fluid flags and smoothing lengths to the device and rebuild the
        hash grid.

        Parameters:
        -----------
        positions : np.ndarray
            Particle positions, shape (N, dim) [m]
        masses : np.ndarray
            Particle masses, shape (N,) [kg]
        isFluid : np.ndarray
            True for fluid particles, shape (N,)
        lengths : np.ndarray | None
            Each particle's smoothing length h, shape (N,) [m], none longer than the one the sums
            were made with, which sizes the grid's cells; None for that one throughout

        Raises:
        -------
        ValueError : If a smoothing length exceeds the grid's, or lengths differ while shifting.

        '''

        self.count = len(positions)
        self._reserve(self.count)
        if self.count == 0:
            return
        if lengths is None:
            lengths = np.full(self.count, self.smoothingLength)
        lengths = np.asarray(lengths, dtype = float)
        if lengths.max() > self.maxLength * (1.0 + 1e-9):
            raise ValueError(f'A smoothing length of {lengths.max()} m exceeds the {self.maxLength} m the hash grid '
                             f'is built for.')
        self.minLength = float(lengths.min())
        self.uniformLengths = bool(np.all(lengths == lengths[0]))
        if self.shifting['enabled'] and not self.uniformLengths:
            raise ValueError('Particle shifting needs one smoothing length for every particle.')
        self._upload('positions', positions)
        self._upload('masses', masses)
        self._upload('lengths', lengths)
        self._upload('fluid', np.asarray(isFluid).view(np.uint8))
        self._rebuild()

    def pairAccelerations(self, velocities: np.ndarray, densities: np.ndarray, pressures: np.ndarray,
                          soundSpeed: float, alpha: float, boundaryViscosityFactor: float,
                          gravity: np.ndarray) -> np.ndarray:

        '''Pressure gradient, artificial viscosity and gravity on the fluid, zero on the boundary [m/s^2].'''

        g = np.zeros(3)
        g[:self.dimensions] = gravity
        uploaded = [self._upload('velocities', velocities), self._upload('densities', densities),
                    self._upload('pressures', pressures)]
        if self.limitViscosity:
            self._velocityGradients()
        self._launch(accelerationKernel, self._geometry + uploaded + [self._device['masses'], self._device['fluid']]
                     + self._lengthInputs + [soundSpeed, alpha, boundaryViscosityFactor, wp.vec3(*g)]
                     + self._viscosityInputs + [self._device['vector']])

        return self._download('vector')

    def continuityRate(self, velocities: np.ndarray, densities: np.ndarray | None = None) -> np.ndarray:

        '''d rho / dt for every particle, with the diffusive term `setDiffusion` set [kg/(m^3 s)].'''

        if self.diffusion['mode'] != 0 or self.correctContinuity:
            if densities is None:
                raise ValueError('The diffusive term and the correction need the densities.')
            self._upload('densities', densities)
            self._upload('deviations', np.asarray(densities, dtype = float) - self.diffusion['referenceDensity'])
        if self.correctContinuity:
            self._correct()
        self._launch(continuityKernel, self._geometry + [
            self._upload('velocities', velocities), self._device['masses'], self._device['fluid']]
            + self._lengthInputs + self._diffusionInputs + [self._device['scalar']])

        return self._download('scalar')

    def xsphVelocity(self, velocities: np.ndarray, densities: np.ndarray, epsilon: float) -> np.ndarray:

        '''The XSPH transport velocity of the fluid; boundary rows unchanged [m/s].'''

        self._launch(xsphKernel, self._geometry + [
            self._upload('velocities', velocities), self._upload('densities', densities), self._device['masses'],
            self._device['fluid']] + self._lengthInputs + [epsilon, self._device['vector']])

        return self._download('vector')

    def wallPressure(self, densities: np.ndarray, pressures: np.ndarray, gravity: np.ndarray, referenceDensity: float,
                     soundSpeed: float) -> np.ndarray:

        '''Every pressure with the boundary's taken from the fluid (`sphSolver.extrapolatedWallPressure`) [Pa].'''

        g = np.zeros(3)
        g[:self.dimensions] = gravity
        self._launch(wallPressureKernel, self._geometry + [
            self._device['fluid'], self._upload('densities', densities), self._upload('pressures', pressures)]
            + self._lengthInputs + [wp.vec3(*g), float(referenceDensity),
            float(referenceDensity * soundSpeed**2 / taitExponent), self._device['densities64']])

        return self._download('pressures')

    def shepardDensity(self, densities: np.ndarray) -> np.ndarray:

        '''The Shepard-filtered density of every particle [kg/m^3].'''

        self._launch(shepardKernel, self._geometry + [
            self._upload('densities', densities), self._device['masses']] + self._lengthInputs + [
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

        self.update(particles.positions, particles.masses, particles.isFluid, particles.lengths)
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
            self._launch(shepardKernel, self._geometry + [d['densities'], d['masses']] + self._lengthInputs
                         + [d['scalar']])
            self._launch(applyShepard, [d['scalar'], d['fluid'], d['densities64'], d['densities']])
        self._launch(equationOfState, [d['densities64'], physics['referenceDensity'], physics['stiffness'], d['fluid'],
                                       int(self.negativePressure), d['pressures']])
        if self.extrapolateWalls:
            self._launch(wallPressureKernel, self._geometry + [d['fluid'], d['densities'], d['pressures']]
                         + self._lengthInputs + [physics['gravity'], physics['referenceDensity'], physics['stiffness'],
                                                 d['densities64']])
        if self.limitViscosity:
            self._velocityGradients()
        self._launch(accelerationKernel, self._geometry + [d['velocities'], d['densities'], d['pressures'], d['masses'],
                                                           d['fluid']] + self._lengthInputs + [
            physics['soundSpeed'], physics['alpha'], physics['boundaryViscosityFactor'], physics['gravity']]
                     + self._viscosityInputs + [d['accelerations']])
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
        if self.diffusion['mode'] != 0:
            self._launch(toDeviation, [d['densities64'], self.diffusion['referenceDensity'], d['deviations']])
        if self.correctContinuity:
            self._correct()
        self._launch(continuityKernel, self._geometry + [d['velocities'], d['masses'], d['fluid']] + self._lengthInputs
                     + self._diffusionInputs + [d['scalar']])
        self._launch(advanceDensity, [d['densities64'], d['scalar'], d['fluid'], timeStep, physics['densityFloor'],
                                      d['densities']])
        if epsilon > 0.0:
            self._launch(xsphKernel, self._geometry + [d['velocities'], d['densities'], d['masses'], d['fluid']]
                         + self._lengthInputs + [epsilon, d['vector']])
        shifting = self.shifting['enabled']
        if shifting:
            self._launch(toDeviation, [d['densities64'], self.diffusion['referenceDensity'], d['deviations']])
            self._shift(timeStep)
        self._launch(drift, [d['positions'], d['vector'], d['velocities64'], 1 if epsilon > 0.0 else 0,
                             d['shifts'] if shifting else d['vector'], d['densityChanges'] if shifting else d['scalar'],
                             d['velocityChanges'] if shifting else d['vector'], 1 if shifting else 0,
                             d['densities64'], d['fluid'], timeStep, self.period])
        self.enforceBoundary()

    def enforceBoundary(self) -> None:

        '''Enforce the boundary set by `setBoundary` on the device state.'''

        if self._enforcement is not None:
            kernel, inputs = self._enforcement
            self._launch(kernel, inputs)
