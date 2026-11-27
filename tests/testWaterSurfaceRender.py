
# -- pySurf: Water Surface Rendering Tests -- #

'''

Checks of tools/rendering/waterSurface.py against closed forms and against the particles' own
resolution s, with the surface's kernel support R = 3 s, the normals' R_n = 6 s and a grid of half
a spacing.

    sphere      one particle of eight spacings' volume: its field's 0.5 level is the sphere of the
                radius where 8 s^3 W(r) = 0.5, to a tenth of a spacing
    slab        a flat slab whose top layer lies half a spacing under z = 0: drawn at z = 0 within
                a quarter of a spacing, its normals within 5 degrees of vertical, clear of its
                edges, with and without the wider normal field
    calibration roadmap item 5's gate on the exact solitary wave of H/h0 = 0.3 in bed-fitted
                columns at 32 per depth, seen from 25 degrees above still water across the crest;
                R_n = 6 s was chosen on this case, so it calibrates rather than validates
    held out    the exact solitary wave of H/h0 = 0.5 from a second camera, 40 degrees above the
                horizontal at azimuth -60 degrees, set before any run with R_n

The bounds of the last two, at every fluid pixel more than 4 pixels from the silhouette and clear
of the tank's ends and sides: the surface within one spacing of the exact surface, its median
offset within a quarter of a spacing; normals within 10 degrees of the exact surface's at 95
percent of those pixels and within 15 degrees at all. Bed-fitted columns put the free surface 0 to
1 spacing above each column's top particle, so a surface half a spacing above the top particles
errs by up to half a spacing either way.

Sean Bowman [11/26/2026]

'''

import math
import sys
from pathlib import Path

import numpy as np
import pytest

warp = pytest.importorskip('warp')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'rendering'))

from waterSurface import Camera, Look, WaterSurfaceRenderer

from pysurf.bathymetry import HeightField
from pysurf.waveMakers import exactSolitaryWaveFluid3D, exactSurfaceElevation

# The solitary-wave gates gather millions of grid points; Warp's CPU device would take many minutes
cuda = pytest.mark.skipif(warp.get_cuda_device_count() == 0, reason = 'needs a CUDA device')

def levelRadius(volume: float, support: float, level: float = 0.5) -> float:

    '''The radius where one particle's field volume W(r) falls to `level`, by bisection.'''

    sigma = 21.0 / (2.0 * math.pi * support**3)
    value = lambda q: volume * sigma * (1.0 - q)**4 * (1.0 + 4.0 * q) - level
    low, high = 0.0, 1.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if value(middle) > 0.0 else (low, middle)

    return 0.5 * (low + high) * support

def solitaryWaveReading(height: float, camera: Camera, spacing: float, normalRadius: float | None) -> dict:

    '''

    The exact solitary wave of `height` / h0 in bed-fitted columns over a flat bed 8 h0 by 1 h0,
    crest at x = 4 h0, drawn by `camera` at R = 3 s: its interior pixels' offsets from the exact
    surface along its normal [spacings] and their normals' angles from the exact normal [deg].

    '''

    bed = HeightField.fromFunction(lambda x, y: 0.0 * x, (0.0, 8.0, 0.0, 1.0), spacing, stillWaterLevel = 1.0)
    crestX = 4.0
    positions, _, _ = exactSolitaryWaveFluid3D(bed, height, 1.0, crestX, spacing, bedFitted = True)
    # The whole tank and the kernels' reach beyond it, so no face of the box cuts the water
    look = Look(supportRadius = 3.0 * spacing, cellSize = 0.5 * spacing, normalRadius = normalRadius)
    renderer = WaterSurfaceRenderer(camera, look, (-0.3, 8.3, -0.3, 1.3, -0.3, 1.8))
    surface = renderer.render(positions, np.full(len(positions), spacing**3))
    hits, found = surface['hits'], surface['found']

    # Pixels more than 4 from the silhouette, and clear of the tank's ends and sides
    interior = found.copy()
    for shift in range(1, 5):
        for axis in (0, 1):
            for sign in (1, -1):
                interior &= np.roll(found, sign * shift, axis = axis)
    x, y, z = (np.nan_to_num(hits[..., a]) for a in range(3))
    interior &= (x > 0.5) & (x < 7.5) & (y > 0.1) & (y < 0.9)
    # Distance to the exact surface along its normal, from the vertical gap and the surface's slope
    eta = lambda at: 1.0 + exactSurfaceElevation(at, height, 1.0, crestX)
    slope = (eta(x + 1e-5) - eta(x - 1e-5)) / 2e-5
    gap = (z - eta(x)) / np.sqrt(1.0 + slope**2) / spacing
    exactNormal = np.stack([-slope, np.zeros_like(slope), np.ones_like(slope)], axis = -1)
    exactNormal /= np.linalg.norm(exactNormal, axis = -1, keepdims = True)
    angle = np.degrees(np.arccos(np.clip(np.sum(surface['normals'] * exactNormal, axis = -1), -1.0, 1.0)))
    gap, angle = gap[interior], angle[interior]
    text = (f'gap / s min {gap.min():+.3f} median {np.median(gap):+.3f} max {gap.max():+.3f}; normal deg 50th '
            f'{np.percentile(angle, 50):.2f} 95th {np.percentile(angle, 95):.2f} max {angle.max():.2f}; {len(gap)} px')

    return {'gap': gap, 'angle': angle, 'text': text}

def assertWithinTheGate(reading: dict) -> None:
    gap, angle, text = reading['gap'], reading['angle'], reading['text']
    assert len(gap) > 20000, text
    assert np.abs(gap).max() < 1.0, text
    assert abs(np.median(gap)) < 0.25, text
    assert np.percentile(angle, 95) < 10.0, text
    assert angle.max() < 15.0, text

def testOneParticleDrawsTheSphereOfItsLevel():

    spacing = 0.05
    support = 3.0 * spacing
    centre = np.array([0.013, -0.007, 0.021])
    radius = levelRadius(8.0 * spacing**3, support)
    camera = Camera.orbit(centre, distance = 0.6, elevation = 20.0, azimuth = -70.0, width = 160, height = 160)
    reach = support + spacing
    box = (centre[0] - reach, centre[0] + reach, centre[1] - reach, centre[1] + reach, centre[2] - reach,
           centre[2] + reach)
    renderer = WaterSurfaceRenderer(camera, Look(supportRadius = support, cellSize = 0.5 * spacing), box,
                                    device = 'cpu')
    surface = renderer.render(centre[None, :], np.array([8.0 * spacing**3]))
    found = surface['found']
    distance = np.linalg.norm(surface['hits'][found] - centre, axis = -1)

    # Most of the sphere's disc, about r / d * focal pixels in radius, must draw
    assert found.sum() > 0.8 * math.pi * (radius / 0.6 * camera.focal)**2
    assert np.abs(distance - radius).max() < 0.1 * spacing, (np.abs(distance - radius).max() / spacing, radius / spacing)

@pytest.mark.parametrize('normalSpacings', [None, 6.0])
def testAFlatSlabIsDrawnAtItsSurface(normalSpacings):

    '''Eight layers whose top lies half a spacing under z = 0, seen from 35 degrees.'''

    spacing = 0.05
    x, y, z = np.meshgrid(np.arange(-1.0, 1.0, spacing) + 0.5 * spacing, np.arange(-1.0, 1.0, spacing) + 0.5 * spacing,
                          -spacing * (0.5 + np.arange(8)))
    positions = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    camera = Camera.orbit([0.0, 0.0, 0.0], distance = 4.0, elevation = 35.0, azimuth = -90.0, width = 320, height = 256)
    look = Look(supportRadius = 3.0 * spacing, cellSize = 0.5 * spacing,
                normalRadius = None if normalSpacings is None else normalSpacings * spacing)
    renderer = WaterSurfaceRenderer(camera, look, (-1.4, 1.4, -1.4, 1.4, -0.8, 0.4), device = 'cpu')
    surface = renderer.render(positions, np.full(len(positions), spacing**3))
    hits, found = surface['hits'], surface['found']
    # Clear of the edges by the wider kernel's reach
    inside = found & (np.abs(np.nan_to_num(hits[..., 0])) < 0.7) & (np.abs(np.nan_to_num(hits[..., 1])) < 0.7)
    tilt = np.degrees(np.arccos(np.clip(surface['normals'][inside, 2], -1.0, 1.0)))

    assert inside.sum() > 10000
    assert np.abs(hits[inside, 2]).max() < 0.25 * spacing, np.abs(hits[inside, 2]).max() / spacing
    assert tilt.max() < 5.0, tilt.max()

@cuda
def testTheCalibrationWaveIsDrawnOnItsSurface():

    '''Roadmap item 5's gate, the case R_n = 6 s was chosen on: calibration, not validation.'''

    spacing = 1.0 / 32
    camera = Camera.orbit([4.0, 0.5, 1.15], distance = 5.0, elevation = 25.0, azimuth = -90.0, width = 768,
                          height = 450, fieldOfView = 20.0)
    assertWithinTheGate(solitaryWaveReading(0.3, camera, spacing, 6.0 * spacing))

@cuda
def testTheHeldOutWaveIsDrawnOnItsSurface():

    '''The held-out case, set before any run with R_n: H/h0 = 0.5 from a second camera, the gate's bounds.'''

    spacing = 1.0 / 32
    camera = Camera.orbit([4.0, 0.5, 1.25], distance = 5.0, elevation = 40.0, azimuth = -60.0, width = 768,
                          height = 450, fieldOfView = 20.0)
    assertWithinTheGate(solitaryWaveReading(0.5, camera, spacing, 6.0 * spacing))
