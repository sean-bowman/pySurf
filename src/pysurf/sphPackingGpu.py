
# -- pySurf: SPH Particle Packing on a GPU -- #

'''

The steps of `pysurf.sphPacking` on a Warp device, for the millions of particles a 3D start holds.

The device holds every point in one float64 array: the water's points first, then the fixed
particles. Each step splits the positions into float32 points and remainders and rebuilds the hash
grid over them (`sphGpu.toPoints`), sums grad Gamma for the water's points, adds the surface's term
grad Phi from the surface table's nodes within the support of each point (the table is regular, so
each point loops over its own window of nodes, and a table of the lowest nearby node passes over
points deeper than the support), and advances the points in float64, below the surface and clear
of the bed. Every `checkInterval` steps the kinetic energy and the residual come back to the host.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

Nothing here is a new model. tests/testSphPacking.py checks, on Warp's CPU device and on CUDA
where present, that one step moves every point as the 'numpy' backend's array sum
(`sphPacking.packingGradient`) does, to 1e-5 spacings in 2D and 3D, with the same residual, and
that 300 steps agree to 1e-3 spacings, under a flat surface, a sloping one and a solitary
wave's crest.

Sean Bowman [11/13/2026]

'''

# Standard library imports
from __future__ import annotations

import math

# Third-party imports
import numpy as np
import warp as wp

# Local imports
from pysurf.sphGpu import (candidateMargin, double, gridDimensions, kernelNormalization, kernelSlope, kernelValue,
                            separation, toPoints)
from pysurf.sphPacking import SurfaceTable, bedClearance, checkInterval, residual

@wp.func
def surfaceElevation(table: wp.array(dtype = wp.float64), start: wp.vec2d, step: wp.vec2d, nx: int, ny: int,
                     point: wp.vec3d, dims: int) -> wp.float64:

    '''The surface table's elevation under `point`: linear along x in 2D, bilinear in (x, y) in 3D, held at its edges.'''

    one = wp.float64(1.0)
    u = wp.clamp((point[0] - start[0]) / step[0], wp.float64(0.0), wp.float64(nx - 1))
    ix = wp.min(int(wp.floor(u)), nx - 2)
    wx = u - wp.float64(ix)
    if dims == 2:
        return (one - wx) * table[ix] + wx * table[ix + 1]
    v = wp.clamp((point[1] - start[1]) / step[1], wp.float64(0.0), wp.float64(ny - 1))
    iy = wp.min(int(wp.floor(v)), ny - 2)
    wy = v - wp.float64(iy)
    low = (one - wy) * table[ix * ny + iy] + wy * table[ix * ny + iy + 1]
    high = (one - wy) * table[(ix + 1) * ny + iy] + wy * table[(ix + 1) * ny + iy + 1]

    return (one - wx) * low + wx * high

@wp.func
def withVertical(point: wp.vec3d, value: wp.float64, dims: int) -> wp.vec3d:

    '''`point` with its vertical coordinate, the last of `dims`, replaced by `value`.'''

    if dims == 2:
        return wp.vec3d(point[0], value, point[2])
    return wp.vec3d(point[0], point[1], value)

@wp.kernel
def surfaceTermKernel(positions: wp.array(dtype = wp.vec3d), table: wp.array(dtype = wp.float64), start: wp.vec2d,
                      step: wp.vec2d, nx: int, ny: int, dims: int, slopeX: wp.array(dtype = wp.float64),
                      slopeY: wp.array(dtype = wp.float64), lowest: wp.array(dtype = wp.float64), kind: int, h: float,
                      sigma: float, result: wp.array(dtype = wp.vec3)):

    '''

    Adds grad Phi = sum over the surface table's nodes of W (-grad eta, 1) dx (dy) to each water
    point's grad Gamma, as `sphPacking.surfaceGradient` sums it. Node offsets come from the point's
    float64 place in the table's index space, so they keep float64 accuracy in float32.

    '''

    i = wp.tid()
    point = positions[i]
    support = 2.0 * h
    vertical = point[dims - 1]
    if vertical < surfaceElevation(lowest, start, step, nx, ny, point, dims) - wp.float64(support):
        return
    u = (point[0] - start[0]) / step[0]
    reach = wp.float64(support) / step[0]
    firstX = wp.max(int(wp.ceil(u - reach)), 0)
    lastX = wp.min(int(wp.floor(u + reach)), nx - 1)
    v = wp.float64(0.0)
    firstY = int(0)
    lastY = int(0)
    area = float(step[0])
    if dims == 3:
        v = (point[1] - start[1]) / step[1]
        reachY = wp.float64(support) / step[1]
        firstY = wp.max(int(wp.ceil(v - reachY)), 0)
        lastY = wp.min(int(wp.floor(v + reachY)), ny - 1)
        area = float(step[0] * step[1])
    total = wp.vec3(0.0, 0.0, 0.0)
    for ix in range(firstX, lastX + 1):
        dx = float((wp.float64(ix) - u) * step[0])
        for iy in range(firstY, lastY + 1):
            node = ix * ny + iy
            dz = float(table[node] - vertical)
            dy = float(0.0)
            if dims == 3:
                dy = float((wp.float64(iy) - v) * step[1])
            r = wp.sqrt(dx * dx + dy * dy + dz * dz)
            if r < support:
                weight = kernelValue(r, h, sigma, kind) * area
                if dims == 2:
                    total += weight * wp.vec3(-float(slopeX[node]), 1.0, 0.0)
                else:
                    total += weight * wp.vec3(-float(slopeX[node]), -float(slopeY[node]), 1.0)
    result[i] = result[i] + total

@wp.kernel
def packingGradientKernel(grid: wp.uint64, points: wp.array(dtype = wp.vec3), lows: wp.array(dtype = wp.vec3),
                          precise: int, kind: int, movable: int, h: float, sigma: float,
                          volumes: wp.array(dtype = float), result: wp.array(dtype = wp.vec3)):

    '''grad Gamma_i = sum_j V_j grad W_ij over every neighbor, for the water's points (the first `movable`).'''

    i = wp.hash_grid_point_id(grid, wp.tid())
    if i >= movable:
        return
    support = 2.0 * h
    limit = candidateMargin * support * support
    pointI = points[i]
    lowI = lows[i]
    offset = wp.vec3(0.0, 0.0, 0.0)
    total = wp.vec3(0.0, 0.0, 0.0)
    query = wp.hash_grid_query(grid, pointI, support)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        pointJ = points[j]
        coarse = pointI - pointJ
        if wp.dot(coarse, coarse) >= limit or j == i:
            continue
        d = separation(coarse, pointI, pointJ, lowI, lows[j], offset, precise)
        r = wp.length(d)
        if r >= support or r < 1.0e-12:
            continue
        total += volumes[j] * kernelSlope(r, h, sigma, kind) / r * d
    result[i] = total

@wp.kernel
def packingStepKernel(positions: wp.array(dtype = wp.vec3d), velocities: wp.array(dtype = wp.vec3d),
                      gradients: wp.array(dtype = wp.vec3), beta: wp.float64, nu: wp.float64, timeStep: wp.float64,
                      table: wp.array(dtype = wp.float64), start: wp.vec2d, step: wp.vec2d, nx: int, ny: int, dims: int,
                      bed: wp.array(dtype = wp.float64), bedStart: wp.vec2d, bedStep: wp.vec2d, bedNx: int, bedNy: int,
                      holdAbove: int, clearance: wp.float64, accumulate: int, h: wp.float64,
                      sums: wp.array(dtype = wp.float64)):

    '''

    One step of du/dt = -beta grad Gamma - nu u, dr/dt = u for a water point; a point that crosses
    the surface returns across it with its vertical velocity reversed, and with `holdAbove` a
    point closer to the bed than `clearance` returns to that height, its velocity into the bed
    removed. With `accumulate`, adds |u|^2 and (h |grad Gamma|)^2 into `sums`.

    '''

    i = wp.tid()
    gradient = double(gradients[i])
    velocity = velocities[i] + timeStep * (-beta * gradient - nu * velocities[i])
    position = positions[i] + timeStep * velocity
    eta = surfaceElevation(table, start, step, nx, ny, position, dims)
    if position[dims - 1] > eta:
        position = withVertical(position, wp.float64(2.0) * eta - position[dims - 1], dims)
        velocity = withVertical(velocity, -velocity[dims - 1], dims)
    if holdAbove != 0:
        bottom = surfaceElevation(bed, bedStart, bedStep, bedNx, bedNy, position, dims) + clearance
        if position[dims - 1] < bottom:
            position = withVertical(position, bottom, dims)
            velocity = withVertical(velocity, wp.max(velocity[dims - 1], wp.float64(0.0)), dims)
    positions[i] = position
    velocities[i] = velocity
    if accumulate != 0:
        wp.atomic_add(sums, 0, wp.dot(velocity, velocity))
        wp.atomic_add(sums, 1, h * h * wp.dot(gradient, gradient))

def padded(values: np.ndarray) -> np.ndarray:

    '''Points of 2 or 3 coordinates as 3, the third zero for 2.'''

    result = np.zeros((len(values), 3))
    result[:, :values.shape[1]] = values

    return result

def tableInputs(table: SurfaceTable, device) -> list:

    '''A surface table's kernel arguments: its values, first node, node spacing and node counts along x and y.'''

    values = wp.array(table.values.ravel(), dtype = wp.float64, device = device)
    start = wp.vec2d(*(list(table.start) + [0.0])[:2])
    step = wp.vec2d(*(list(table.step) + [1.0])[:2])
    nx, ny = table.values.shape[0], (table.values.shape[1] if table.values.ndim == 2 else 1)

    return [values, start, step, nx, ny]

def packOnDevice(points: np.ndarray, fixed: np.ndarray, shares: np.ndarray, table: SurfaceTable, bed: SurfaceTable | None,
                 volume: float, h: float, spacing: float, kernelType: str, damping: float, iterations: int,
                 minimumIterations: int, tolerance: float, device: str | None = None) -> tuple:

    '''

    `sphPacking._packNumpy`'s steps on a Warp device; returns positions, steps, kinetic energy and
    the residuals before and after.

    '''

    device = wp.get_device(device)
    count, dims = points.shape
    everything = np.vstack([points, fixed])
    total = len(everything)
    support = 2.0 * h
    lower, upper = everything.min(axis = 0) - support, everything.max(axis = 0) + support
    grid = wp.HashGrid(*gridDimensions(lower, upper, support, dims), device = device)

    positions = wp.array(padded(everything), dtype = wp.vec3d, device = device)
    velocities = wp.zeros(count, dtype = wp.vec3d, device = device)
    pointArray = wp.zeros(total, dtype = wp.vec3, device = device)
    lowArray = wp.zeros(total, dtype = wp.vec3, device = device)
    gradients = wp.zeros(count, dtype = wp.vec3, device = device)
    sums = wp.zeros(2, dtype = wp.float64, device = device)
    volumes = wp.array(volume * np.concatenate([np.ones(count), shares]), dtype = float, device = device)
    surfaceInputs = tableInputs(table, device) + [dims]
    slopes = table.slopes()
    slopeInputs = [wp.array(slopes[0].ravel(), dtype = wp.float64, device = device),
                   wp.array(slopes[-1].ravel(), dtype = wp.float64, device = device),
                   wp.array(table.lowest(2.0 * h).values.ravel(), dtype = wp.float64, device = device)]
    # Without a bed the surface's table stands in, unread
    bedInputs = tableInputs(bed if bed is not None else table, device) + [1 if bed is not None else 0,
                                                                          bedClearance * spacing]
    sigma = kernelNormalization(kernelType, h, dims)
    kind = 1 if kernelType == 'wendlandC2' else 0
    beta = 1.0                                  # [m^2/s^2], sets only the pseudo-time's scale
    timeStep = spacing / math.sqrt(beta)        # [s]
    nu = damping / timeStep                     # [1/s]

    def sumGradients() -> None:
        wp.launch(toPoints, dim = total, inputs = [positions, pointArray, lowArray], device = device)
        grid.build(pointArray, support)
        wp.launch(packingGradientKernel, dim = total, inputs = [grid.id, pointArray, lowArray, 1, kind, count, h, sigma,
                                                                volumes, gradients], device = device)
        wp.launch(surfaceTermKernel, dim = count, inputs = [positions] + surfaceInputs + slopeInputs
                  + [kind, h, sigma, gradients], device = device)

    sumGradients()
    before = residual(gradients.numpy()[:, :dims].astype(float), h)
    energy, step = 0.0, 0
    for step in range(1, iterations + 1):
        check = step % checkInterval == 0 or step == iterations
        if check:
            sums.zero_()
        wp.launch(packingStepKernel, dim = count, inputs = [positions, velocities, gradients, beta, nu, timeStep]
                  + surfaceInputs + bedInputs + [1 if check else 0, h, sums], device = device)
        sumGradients()
        if check:
            totals = sums.numpy()
            energy = float(totals[0]) / (2.0 * beta * count)
            if step >= minimumIterations and math.sqrt(float(totals[1]) / count) < tolerance:
                break
    after = residual(gradients.numpy()[:, :dims].astype(float), h)
    packed = positions.numpy()[:count, :dims].astype(float)

    return packed, step, energy, before, after
