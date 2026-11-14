
# -- pySurf: 3D Breaking Wave Tests -- #

'''

Checks of pysurf.BreakingWave3D and its start: over a flat bed the 3D exact solitary wave is the 2D
one laid in every lattice row across the tank, particles, velocities and pressures, and its
boundary particles start with the 2D rule's pressure; laid in bed-fitted columns over the ridge it
stands half a spacing above the bed and carries the 2D exact flow, and jittered off those columns
it keeps its particles inside the water with the flow at each one's place; on the GPU a short run reads its section from
the two rows around the plane at every tracker interval, and its snapshot gives the flow mid-depth
under the crest within 5 percent of the exact wave's.

Sean Bowman [11/10/2026]

'''

import math

import numpy as np
import pytest

from pysurf import constants, waveTheory
from pysurf.bathymetry import Bathymetry, HeightField
from pysurf.errors import InvalidInputError
from pysurf.sphBoundaries import HeightFieldBoundary
from pysurf.BreakingWave3D import BreakingWave3D
from pysurf.waveMakers import (exactFlowInChunks, exactSolitaryBoundaryPressure, exactSolitaryBoundaryPressure3D,
                               exactSolitaryWaveFluid, exactSolitaryWaveFluid3D)

spacing, height, crestX = 0.125, 0.3, 2.5

@pytest.fixture
def flat() -> HeightField:

    return HeightField.fromFunction(lambda x, y: 0.0 * x, (0.0, 6.0, 0.0, 0.5), 1.0 / 32.0, stillWaterLevel = 1.0)

def testAFlatStartIsTheTwoDimensionalWaveInEveryRow(flat):

    positions, velocities, pressures = exactSolitaryWaveFluid3D(flat, height, 1.0, crestX, spacing)
    plane, planeVelocities, planePressures = exactSolitaryWaveFluid(Bathymetry([0.0, 6.0], [0.0, 0.0], 1.0), height,
                                                                    crestX, spacing)
    rows = np.arange(0.5 * spacing, 0.5, spacing)

    assert len(positions) == rows.size * len(plane)
    for y in rows:
        row = np.abs(positions[:, 1] - y) < 1e-9
        assert positions[row][:, [0, 2]] == pytest.approx(plane, abs = 1e-12)
        assert velocities[row][:, [0, 2]] == pytest.approx(planeVelocities, abs = 1e-12)
        assert np.all(velocities[row][:, 1] == 0.0)
        assert pressures[row] == pytest.approx(planePressures, abs = 1e-12)

def testTheBoundaryStartsWithTheTwoDimensionalRule(flat):

    points = HeightFieldBoundary(flat, spacing, nLayers = 3, wallTop = 1.6).positions()
    bed = points[(points[:, 0] > 0.0) & (points[:, 0] < 6.0) & (points[:, 1] > 0.0) & (points[:, 1] < 0.5)]
    pressures = exactSolitaryBoundaryPressure3D(flat, bed, height, 1.0, crestX, spacing, constants.gravity)
    planar = exactSolitaryBoundaryPressure(Bathymetry([0.0, 6.0], [0.0, 0.0], 1.0), bed[:, [0, 2]], height, crestX,
                                           spacing, constants.gravity)

    assert np.all(bed[:, 2] < 0.0)
    assert pressures == pytest.approx(planar, abs = 1e-12)

def testABedFittedStartFollowsTheRidge():

    '''

    The exact wave of H/h0 = 0.6 over the ridge of k = 0.25 at 8 particles per depth, in bed-fitted
    columns: every column's lowest particle stands half a spacing above the bed, offshore of the
    toe the particles, flow and pressure are the square lattice's, and every particle carries the
    2D exact flow at its (x, z) with no spanwise velocity. `BreakingWave3D` lays its start this
    way by default.

    '''

    field, s = HeightField.ridge(0.25), 0.125
    positions, velocities, pressures = exactSolitaryWaveFluid3D(field, 0.6, 1.0, 5.785, s, bedFitted = True)
    lattice = exactSolitaryWaveFluid3D(field, 0.6, 1.0, 5.785, s)

    columns, inverse = np.unique(positions[:, :2], axis = 0, return_inverse = True)
    lowest = np.full(len(columns), np.inf)
    np.minimum.at(lowest, inverse.reshape(-1), positions[:, 2])
    assert lowest == pytest.approx(field.elevation(columns[:, 0], columns[:, 1]) + 0.5 * s)

    def offshore(start: tuple) -> list:
        keep = start[0][:, 0] < 5.0
        order = np.lexsort(start[0][keep].T[::-1])
        return [part[keep][order] for part in start]

    for fittedPart, latticePart in zip(offshore((positions, velocities, pressures)), offshore(lattice)):
        assert fittedPart == pytest.approx(latticePart, abs = 1e-12)

    sample = np.arange(0, len(positions), 97)
    flow, pressure, _ = exactFlowInChunks(positions[sample][:, [0, 2]], 0.6, 1.0, 5.785, field.stillWaterLevel)
    assert velocities[sample][:, [0, 2]] == pytest.approx(flow, abs = 1e-12)
    assert pressures[sample] == pytest.approx(np.maximum(pressure, 0.0), abs = 1e-12)
    assert np.all(velocities[:, 1] == 0.0)

    slope = HeightField.fromFunction(lambda x, y: 0.1 * x, (0.0, 2.0, 0.0, 0.5), 1.0 / 32.0, stillWaterLevel = 1.0)
    wave = BreakingWave3D(slope, 0.2, 1.0, depth = 1.0, particlesPerDepth = 4, backend = 'numpy', startJitter = 0.0)
    wave.build()
    fluid = wave.particles.positions[wave.particles.isFluid]
    assert np.min(fluid[:, 2] - slope.elevation(fluid[:, 0], fluid[:, 1])) == pytest.approx(0.5 * wave.spacing)

def testAJitteredStartKeepsItsWaterAndCarriesTheFlowWhereItLies():

    '''

    The bed-fitted start over the ridge, every particle moved by up to 0.15 spacings along x, y and
    z: as many particles, inside the walls, half a spacing or more above the bed and below the
    surface, each with the 2D exact flow at its own (x, z) and no spanwise velocity. A jittered
    `BreakingWave3D` reads its face over six spacings of arc, an ordered one over two.

    '''

    field, s = HeightField.ridge(0.25), 0.125
    fitted = exactSolitaryWaveFluid3D(field, 0.6, 1.0, 5.785, s, bedFitted = True)[0]
    positions, velocities, pressures = exactSolitaryWaveFluid3D(field, 0.6, 1.0, 5.785, s, bedFitted = True,
                                                                jitter = 0.15, seed = 2)
    flow, pressure, surface = exactFlowInChunks(positions[:, [0, 2]], 0.6, 1.0, 5.785, field.stillWaterLevel)
    startX, endX, startY, endY = field.bounds

    assert len(positions) == len(fitted)
    assert np.all((positions[:, 0] >= startX + 0.5 * s - 1e-12) & (positions[:, 0] <= endX - 0.5 * s + 1e-12)
                  & (positions[:, 1] >= startY + 0.5 * s - 1e-12) & (positions[:, 1] <= endY - 0.5 * s + 1e-12))
    assert np.all(positions[:, 2] >= field.elevation(positions[:, 0], positions[:, 1]) + 0.5 * s - 1e-12)
    assert np.all(positions[:, 2] < field.stillWaterLevel + surface)
    assert velocities[:, [0, 2]] == pytest.approx(flow, abs = 1e-12)
    assert pressures == pytest.approx(np.maximum(pressure, 0.0), abs = 1e-12)
    assert np.all(velocities[:, 1] == 0.0)

    moved = []

    def pack(points: np.ndarray, surface) -> np.ndarray:
        moved.append(surface(points[:, 0], points[:, 1]) - points[:, 2])
        return points - np.array([0.0, 0.0, 0.1 * s])

    packed, packedVelocities, _ = exactSolitaryWaveFluid3D(field, 0.6, 1.0, 5.785, s, bedFitted = True, jitter = 0.15,
                                                           seed = 2, pack = pack)
    packedFlow, _, _ = exactFlowInChunks(packed[:, [0, 2]], 0.6, 1.0, 5.785, field.stillWaterLevel)
    assert packed == pytest.approx(positions - np.array([0.0, 0.0, 0.1 * s]), abs = 1e-12)
    assert len(moved) == 1 and np.all(moved[0] > 0.0)
    assert packedVelocities[:, [0, 2]] == pytest.approx(packedFlow, abs = 1e-12)

    slope = HeightField.fromFunction(lambda x, y: 0.1 * x, (0.0, 2.0, 0.0, 0.5), 1.0 / 32.0, stillWaterLevel = 1.0)
    assert BreakingWave3D(slope, 0.2, 1.0, depth = 1.0, startJitter = 0.15, backend = 'numpy').faceArc == 6.0
    assert BreakingWave3D(slope, 0.2, 1.0, depth = 1.0, startJitter = 0.0, backend = 'numpy').faceArc == 2.0
    with pytest.raises(InvalidInputError):
        BreakingWave3D(slope, 0.2, 1.0, depth = 1.0, startJitter = 0.5, backend = 'numpy')

wp = pytest.importorskip('warp')
cuda = pytest.mark.skipif(wp.get_cuda_device_count() == 0, reason = 'needs a CUDA device')

@cuda
def testAShortRunReadsItsSectionAndCarriesTheWave():

    from pysurf.BreakingWave3D import BreakingWave3D

    # The crest a half length from the offshore wall, so the wall does not cut the wave's tail
    field = HeightField.fromFunction(lambda x, y: 0.0 * x, (0.0, 10.0, 0.0, 1.0), 1.0 / 32.0, stillWaterLevel = 1.0)
    start, snapshotTime = 6.5, 0.1
    wave = BreakingWave3D(field, height, start, particlesPerDepth = 8, recordTimes = (snapshotTime,), endTime = 0.2)
    wave.build()
    s = wave.spacing

    # The section takes the two rows either side of the tank's middle
    section = wave.particles.positions[wave.particles.isFluid][wave._sectionMask(0.5)]
    assert np.unique(np.round(section[:, 1], 9)).tolist() == pytest.approx([0.5 - 0.5 * s, 0.5 + 0.5 * s])

    wave.run(progress = False)
    assert len(wave.tracker.times) == pytest.approx(0.2 / (0.01 * math.sqrt(1.0 / constants.gravity)), abs = 1)
    assert len(wave.sectionFrames[0.5]) == 3

    # Mid-depth under the crest the flow is the exact wave's
    crest = start + waveTheory.solitaryWaveSpeed(height, 1.0) * snapshotTime
    flow = wave.velocity(np.array([[crest, 0.5, 0.5]]), snapshotTime)[0]
    exact, _, _ = waveTheory.exactSolitaryFlow(np.array([[crest, 0.5]]), height, 1.0, crest, 1.0)
    assert flow[0] == pytest.approx(exact[0, 0], rel = 0.05)
    assert abs(flow[1]) < 1e-3 * flow[0]
