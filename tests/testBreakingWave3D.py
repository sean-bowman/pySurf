
# -- pySurf: 3D Breaking Wave Tests -- #

'''

Checks of pysurf.BreakingWave3D and its start: over a flat bed the 3D exact solitary wave is the 2D
one laid in every lattice row across the tank, particles, velocities and pressures, and its
boundary particles start with the 2D rule's pressure; on the GPU a short run reads its section from
the two rows around the plane at every tracker interval, and its snapshot gives the flow mid-depth
under the crest within 5 percent of the exact wave's.

Sean Bowman [11/10/2026]

'''

import math

import numpy as np
import pytest

from pysurf import constants, waveTheory
from pysurf.bathymetry import Bathymetry, HeightField
from pysurf.sphBoundaries import HeightFieldBoundary
from pysurf.waveMakers import (exactSolitaryBoundaryPressure, exactSolitaryBoundaryPressure3D, exactSolitaryWaveFluid,
                               exactSolitaryWaveFluid3D)

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
