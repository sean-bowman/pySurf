
# -- pySurf: SPH Particle Packing Tests -- #

'''

Checks of pysurf.sphPacking: the surface table reads a plane surface exactly; the surface's term
grad Phi matches a direct quadrature of the kernel's gradient over the space above a solitary
wave's crest, and on a tilted plane equals the flat surface's term turned to the plane's normal; a
lattice under a flat surface moves its rows near the surface by about a tenth of a spacing;
jittered still water, and jittered water under the crest of a solitary wave of H/h = 0.6, pack to
a residual at least ten times smaller, every point kept between the floor and the surface; points
in the shallows keep their places; the packing stops once the residual reaches its tolerance; and
the Warp backend, on its CPU device and on CUDA where present, moves the points as the numpy
backend does.

Sean Bowman [11/13/2026]

'''

import numpy as np
import pytest

from pysurf.sphBoundaries import ContainerBoundary
from pysurf.sphKernels import createKernel
from pysurf.sphPacking import SurfaceTable, packPoints, packingGradient, surfaceGradient
from pysurf.sphParticles import jitterPoints, latticePoints

spacing, width, depth = 0.01, 0.3, 0.16
crestHeight = 0.6 * depth
decay = np.sqrt(0.75 * crestHeight / depth**3)

def flat(x: np.ndarray) -> np.ndarray:
    return depth + 0.0 * x

def sloping(x: np.ndarray) -> np.ndarray:
    return depth - 0.1 * (x - 0.5 * width)

def crest(x: np.ndarray) -> np.ndarray:
    '''A solitary wave's sech^2 crest of H/h = 0.6 at the middle of the container: slopes to 0.31.'''
    return depth + crestHeight / np.cosh(decay * (x - 0.5 * width))**2

def stillWater(surface = flat, seed: int = 4) -> tuple:

    '''A lattice of water in a container 6 layers deep, below `surface`, and the same water jittered by 0.15 spacings.'''

    points = latticePoints([0.0, 0.0], [width, depth + crestHeight + 0.02], spacing)
    points = points[points[:, 1] < surface(points[:, 0])]
    jittered = jitterPoints(points, spacing, 0.15, seed, [0.0], [width], lambda x: 0.0 * x, surface)
    fixed = ContainerBoundary([0.0, 0.0], [width, 2.0 * depth], spacing, nLayers = 6).positions()

    return points, jittered, fixed

def testTheSurfaceTableReadsAPlaneExactly():

    '''

    A plane surface and a bilinear one are read exactly between the nodes, and held at the edge
    value beyond them; a plane's slopes are read exactly at every node.

    '''

    rng = np.random.default_rng(1)
    table = SurfaceTable.sample(lambda x: 0.3 + 0.1 * x, [0.0], [2.0], 0.05)
    x = rng.uniform(0.0, 2.0, 50)
    assert table.elevation(x) == pytest.approx(0.3 + 0.1 * x, abs = 1e-12)
    assert table.elevation(np.array([3.0])) == pytest.approx(table.elevation(np.array([table.start[0] + 40 * 0.05])))
    assert table.slopes()[0] == pytest.approx(0.1, abs = 1e-12)

    bilinear = lambda x, y: 0.2 + 0.1 * x - 0.05 * y + 0.02 * x * y
    table3 = SurfaceTable.sample(bilinear, [0.0, 0.0], [2.0, 1.0], 0.1)
    x, y = rng.uniform(0.0, 2.0, 50), rng.uniform(0.0, 1.0, 50)
    assert table3.elevation(x, y) == pytest.approx(bilinear(x, y), abs = 1e-12)

def testTheSurfaceTermMatchesAQuadratureOverTheSpaceAbove():

    '''

    Under the crest, on its flank and in front of it, half a spacing to four spacings deep, grad Phi
    from the surface's nodes matches the kernel's gradient summed over cells of s/40 filling the
    space above the crest, to 0.2 percent.

    '''

    h = 3.0 * spacing
    support = 2.0 * h
    kernel = createKernel('wendlandC2', 2)
    table = SurfaceTable.sample(crest, [-0.1], [width + 0.1], 0.5 * spacing)
    cell = spacing / 40.0
    for x0, below in ((0.5 * width, 0.5), (0.07, 0.5), (0.07, 2.0), (0.2, 4.0)):
        point = np.array([x0, crest(np.array([x0]))[0] - below * spacing])
        offsets = np.arange(-support, support, cell) + 0.5 * cell
        x, z = np.meshgrid(point[0] + offsets, point[1] + offsets, indexing = 'ij')
        above = z > crest(x)
        separation = point - np.column_stack([x[above], z[above]])
        distance = np.linalg.norm(separation, axis = 1)
        inside = distance < support
        reference = kernel.gradientBatch(separation[inside], distance[inside], h).sum(axis = 0) * cell**2
        term = surfaceGradient(point[None, :], table, h)[0]

        assert np.linalg.norm(term - reference) < 2e-3 * np.linalg.norm(reference)

def testATiltedPlaneTurnsTheFlatTerm():

    '''

    At a third of a spacing to three spacings along the normal below a plane tilted by slopes of
    0.2 and 0.6 in 2D, and by (0.3, -0.2) in 3D, grad Phi is the flat surface's term at the same
    depth turned to the plane's normal, to 1e-4 of its size.

    '''

    h = 3.0 * spacing
    flat2 = SurfaceTable.sample(flat, [-0.2], [0.5], 0.5 * spacing)
    flat3 = SurfaceTable.sample(lambda x, y: depth + 0.0 * x, [-0.2, -0.2], [0.5, 0.5], 0.5 * spacing)
    for below in (0.3, 1.0, 3.0):
        d = below * spacing
        size = surfaceGradient(np.array([[0.15, depth - d]]), flat2, h)[0, 1]
        for slope in (0.2, 0.6):
            plane = SurfaceTable.sample(lambda x: depth + slope * (x - 0.15), [-0.2], [0.5], 0.5 * spacing)
            normal = np.array([-slope, 1.0]) / np.hypot(slope, 1.0)
            term = surfaceGradient((np.array([0.15, depth]) - d * normal)[None, :], plane, h)[0]
            assert np.linalg.norm(term - size * normal) < 1e-4 * size

        size = surfaceGradient(np.array([[0.15, 0.15, depth - d]]), flat3, h)[0, 2]
        plane = SurfaceTable.sample(lambda x, y: depth + 0.3 * (x - 0.15) - 0.2 * (y - 0.15), [-0.2, -0.2], [0.5, 0.5],
                                    0.5 * spacing)
        normal = np.array([-0.3, 0.2, 1.0]) / np.sqrt(1.13)
        term = surfaceGradient((np.array([0.15, 0.15, depth]) - d * normal)[None, :], plane, h)[0]
        assert np.linalg.norm(term - size * normal) < 1e-4 * size

def testALatticeUnderAFlatSurfaceBarelyMoves():

    '''

    Still water on its lattice, 16 rows under a flat surface between the container's layers. The
    lattice's sum against the continuum above the surface leaves its rows near the surface a
    residual of 3e-3 (rms); 100 steps take it under 3e-4 by moving those rows less than a fifth of
    a spacing.

    '''

    points, _, fixed = stillWater()
    result = packPoints(points, fixed, spacing, flat, iterations = 100, minimumIterations = 100)

    assert result.gradientBefore[0] < 5e-3
    assert result.gradientAfter[0] < 3e-4
    assert result.largestMove < 0.2
    assert result.iterations == 100

def testJitteredWaterPacks():

    '''

    The jittered water packs under a flat and a sloping surface in 600 steps, and under the crest
    of a solitary wave of H/h = 0.6 in 400: the residual falls more than ten times (under the crest
    under 1e-4, a hundredth of where it started), every point stays inside the walls, above the
    floor and below the surface, and the lowest points stand within a fifth of a spacing of the
    lattice's half spacing above the floor.

    '''

    for surface, steps in ((flat, 600), (sloping, 600), (crest, 400)):
        _, jittered, fixed = stillWater(surface)
        result = packPoints(jittered, fixed, spacing, surface, lambda x: 0.0 * x, iterations = steps,
                            minimumIterations = steps)
        x, z = result.positions.T

        assert result.gradientAfter[0] < 0.1 * result.gradientBefore[0]
        assert len(result.positions) == len(jittered) and result.frozen == 0
        assert np.all((x > 0.25 * spacing) & (x < width - 0.25 * spacing))
        assert np.all((z > 0.25 * spacing) & (z < surface(x)))
        assert abs(z.min() - 0.5 * spacing) < 0.2 * spacing
        if surface is crest:
            assert result.gradientAfter[0] < min(1e-4, 0.01 * result.gradientBefore[0])

def testThePackingStopsAtItsToleranceAndLeavesTheShallows():

    '''

    With a tolerance of 1e-3 the packing stops at a reading, every 50 steps, well before its 4000;
    points where the floor stands within the support and a spacing of the surface keep their
    places exactly, and the rest move.

    '''

    _, jittered, fixed = stillWater()
    result = packPoints(jittered, fixed, spacing, flat, tolerance = 1e-3)
    assert result.iterations % 50 == 0 and result.iterations < 4000

    shelf = lambda x: np.where(x > 0.2, 0.12, 0.0)
    shallow = jittered[:, 0] > 0.2
    result = packPoints(jittered, fixed, spacing, flat, shelf, iterations = 50, minimumIterations = 50)
    assert result.frozen == np.count_nonzero(shallow)
    assert np.all(result.positions[shallow] == jittered[shallow])
    assert np.all(np.any(result.positions[~shallow] != jittered[~shallow], axis = 1))

def testTheArraySumIsZeroInsideAnEvenLattice():

    '''grad Gamma of a lattice point six spacings or more inside a larger lattice is zero to rounding.'''

    lattice = latticePoints([0.0, 0.0], [0.4, 0.4], spacing)
    inner = np.all(np.abs(lattice - 0.2) < 0.2 - 6.5 * spacing, axis = 1)
    gradient = packingGradient(lattice[inner], lattice[~inner], spacing**2, 3.0 * spacing)

    assert np.max(np.abs(gradient)) * 3.0 * spacing < 1e-12

wp = pytest.importorskip('warp')
devices = ['cpu'] + (['cuda:0'] if wp.get_cuda_device_count() > 0 else [])

@pytest.mark.parametrize('device', devices)
def testTheDeviceMovesThePointsAsTheArraysDo(device):

    '''

    Jittered water under a sloping surface and under the crest: one step on the device moves every
    point as the array sum does, to 1e-5 spacings, with the same residual; after 300 steps the two
    agree to 1e-3 spacings, the float32 sums' rounding grown through the steps.

    '''

    for surface in (sloping, crest):
        _, jittered, fixed = stillWater(surface)
        for steps, tolerance in ((1, 1e-5), (300, 1e-3)):
            settings = dict(iterations = steps, minimumIterations = steps)
            reference = packPoints(jittered, fixed, spacing, surface, **settings)
            result = packPoints(jittered, fixed, spacing, surface, backend = 'gpu', device = device, **settings)

            assert np.max(np.abs(result.positions - reference.positions)) < tolerance * spacing
            assert result.gradientBefore[0] == pytest.approx(reference.gradientBefore[0], rel = 1e-5)
            assert result.iterations == reference.iterations

@pytest.mark.parametrize('device', devices)
def testTheDeviceMatchesTheArraysIn3D(device):

    '''

    A jittered 3D box of water in its container at h = 2 s, under a flat surface and a tilted one:
    one step on the device follows the array sum to 1e-5 spacings.

    '''

    s, side, height = 0.02, 0.2, 0.16
    for top in (lambda x, y: height + 0.0 * x, lambda x, y: height + 0.1 * (x - 0.1) - 0.05 * (y - 0.1)):
        points = latticePoints([0.0, 0.0, 0.0], [side, side, height + 0.02], s)
        points = points[points[:, 2] < top(points[:, 0], points[:, 1])]
        jittered = jitterPoints(points, s, 0.15, 3, [0.0, 0.0], [side, side], lambda x, y: 0.0 * x, top)
        fixed = ContainerBoundary([0.0, 0.0, 0.0], [side, side, 2.0 * height], s, nLayers = 4).positions()
        settings = dict(smoothingLengthRatio = 2.0, iterations = 1, minimumIterations = 1)
        reference = packPoints(jittered, fixed, s, top, **settings)
        result = packPoints(jittered, fixed, s, top, backend = 'gpu', device = device, **settings)

        assert np.max(np.abs(result.positions - reference.positions)) < 1e-5 * s
        assert result.gradientBefore[0] == pytest.approx(reference.gradientBefore[0], rel = 1e-5)
