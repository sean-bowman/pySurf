
# -- pySurf: Whitewater Tests -- #

'''

Checks of tools/rendering/whitewater.py, Ihmsen et al.'s (2012) diffuse particles, against closed
forms and symmetry. Phi maps to 0..1 as eq. 1 states. Two particles closing at speed u a spacing
apart give each other v_diff = 2 u (1 - s/h_w); moving apart, zero. A lattice slab flags its top
layer as surface and nothing a kernel's reach below it; at rest it spawns nothing, and its flat top
has no curvature, while a ball dilating as v = a x traps no air and its surface curves and moves
along its normals, its curvature sum near the continuum estimate's order. A point in a
block's middle counts as a bubble, half a spacing above its top as foam, three spacings above as
spray. Spray with no drag follows the Euler-Cromer sum exactly; foam rides a uniform stream at its
speed; a bubble in still water rises at buoyancy g times the relaxation time. Spawned particles
fill the cylinder of the volume radius along v dt uniformly, their count the expected one on
average. One splatted particle deposits its projected area in pixels.

Sean Bowman [11/26/2026]

'''

import math
import sys
from pathlib import Path

import numpy as np
import pytest

warp = pytest.importorskip('warp')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'rendering'))

from waterSurface import Camera
from whitewater import Whitewater, WhitewaterSettings, bubble, clampPotential, foam, gravity, spray

spacing = 0.05

def lattice(nx: int, ny: int, nz: int, top: float = 0.0) -> np.ndarray:
    '''A cubic lattice of spacing s, centred on x = y = 0, its top layer at z = top.'''
    x, y, z = np.meshgrid((np.arange(nx) - 0.5 * (nx - 1)) * spacing, (np.arange(ny) - 0.5 * (ny - 1)) * spacing,
                          top - np.arange(nz) * spacing, indexing = 'ij')
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

def whitewater(box: tuple = (-2.0, 2.0, -2.0, 2.0, -2.0, 2.0), **options) -> Whitewater:
    return Whitewater(WhitewaterSettings(spacing = spacing, **options), box, device = 'cpu')

def testPhiMapsOntoZeroToOne():
    values = clampPotential(np.array([0.0, 5.0, 12.5, 20.0, 40.0]), (5.0, 20.0))
    assert values == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])

def testTwoClosingParticlesTrapAir():

    u = 3.0
    positions = np.array([[0.0, 0.0, 0.0], [spacing, 0.0, 0.0]])
    volumes = np.full(2, spacing**3)
    closing = whitewater().potentials(positions, np.array([[0.5 * u, 0, 0], [-0.5 * u, 0, 0]]), volumes, 0.01)
    parting = whitewater().potentials(positions, np.array([[-0.5 * u, 0, 0], [0.5 * u, 0, 0]]), volumes, 0.01)

    # |v_ij| (1 - vhat . xhat)(1 - r / h_w) = u * 2 * (1 - 1/2)
    assert closing['trappedAir'] == pytest.approx([u, u], rel = 1e-5)
    assert parting['trappedAir'] == pytest.approx([0.0, 0.0], abs = 1e-6)

def testASlabFlagsItsTopLayerAndSpawnsNothingAtRest():

    # Wide and deep enough that the middle lies beyond the normals' Gaussian reach (4 sigma, 6.2 s) of the sides and bottom
    positions = lattice(32, 32, 14)
    measures = whitewater().potentials(positions, np.zeros_like(positions), np.full(len(positions), spacing**3), 0.01)
    x, y, z = measures['positions'].T
    middle = (np.abs(x) < 0.25) & (np.abs(y) < 0.25)
    top = middle & (z > -0.5 * spacing)
    deep = middle & (z < -2.5 * spacing) & (z > -4.5 * spacing)

    assert measures['surface'][top].all()
    assert not measures['surface'][deep].any()
    assert np.all(measures['spawnCount'] == 0.0)
    assert np.abs(measures['curvature'][top]).max() < 1e-4

def testAnExpandingBallIsACrestMovingOutward():

    '''

    Dilating as v = a x, so every relative velocity lies along its pair's separation and no air
    is trapped; its curvature sum lies under the published tau_wc, so the test sets its own.

    '''

    grid = lattice(17, 17, 17, top = 8 * spacing)
    radius = 6.0 * spacing
    positions = grid[np.linalg.norm(grid, axis = 1) <= radius]
    measures = whitewater(waveCrest = (0.05, 0.2)).potentials(positions, 8.0 / radius * positions,
                                                              np.full(len(positions), spacing**3), 0.01)
    surface = measures['surface']
    assert np.abs(measures['trappedAir']).max() < 1e-4

    # Over a sphere of radius R_s, 1 - n_i . n_j = d^2 / 2 R_s^2; summed with weight 1 - d/h_w over a
    # surface layer about 1.5 particles thick, sum = 2 pi 1.5 / (2 R_s^2 s^2) int d^3 (1 - d/h_w) dd = 0.105
    assert surface.sum() > 100
    assert np.median(measures['curvature'][surface]) > 0.05
    assert measures['aligned'][surface].mean() > 0.8
    assert measures['spawnCount'][surface].sum() > 0.0

def testClassesFollowTheNeighbourCount():

    fluid = lattice(20, 20, 20)
    ww = whitewater()
    ww.positions = np.array([[0.0, 0.0, -10 * spacing], [0.0, 0.0, 0.5 * spacing], [0.0, 0.0, 3.0 * spacing]],
                            np.float32)
    ww.velocities = np.zeros((3, 3), np.float32)
    ww.lifetime = np.full(3, 10.0, np.float32)
    ww.kinds = np.full(3, foam, np.int32)
    counts = ww.advance(fluid, np.zeros_like(fluid), np.full(len(fluid), spacing**3), 1e-6)

    assert list(ww.kinds) == [bubble, foam, spray], counts
    assert counts[0] == pytest.approx(4.0 * math.pi / 3.0 * 8.0, rel = 0.1)

def testSprayFollowsTheEulerCromerSum():

    fluid = lattice(4, 4, 4, top = -1.5)
    ww = whitewater(sprayDrag = 0.0)
    start, launch = np.array([0.1, 0.2, 0.3]), np.array([1.0, -0.5, 2.0])
    ww.positions, ww.velocities = start[None].astype(np.float32), launch[None].astype(np.float32)
    ww.lifetime, ww.kinds = np.full(1, 1.0, np.float32), np.full(1, spray, np.int32)
    dt, steps = 0.004, 50
    for _ in range(steps):
        ww.advance(fluid, np.zeros_like(fluid), np.full(len(fluid), spacing**3), dt)
    exact = start + steps * dt * launch - np.array([0.0, 0.0, gravity]) * dt**2 * steps * (steps + 1) / 2.0

    assert ww.positions[0] == pytest.approx(exact, abs = 1e-5)
    assert ww.lifetime[0] == pytest.approx(1.0)

def testFoamRidesAUniformStream():

    fluid = lattice(20, 20, 8)
    stream = np.array([1.2, -0.4, 0.1])
    ww = whitewater()
    start = np.array([0.0, 0.0, 0.5 * spacing])
    ww.positions, ww.velocities = start[None].astype(np.float32), np.zeros((1, 3), np.float32)
    ww.lifetime, ww.kinds = np.full(1, 2.0, np.float32), np.full(1, foam, np.int32)
    ww.advance(fluid, np.tile(stream, (len(fluid), 1)), np.full(len(fluid), spacing**3), 0.01)

    assert ww.kinds[0] == foam
    assert ww.positions[0] == pytest.approx(start + 0.01 * stream, abs = 1e-6)
    assert ww.velocities[0] == pytest.approx(stream, abs = 1e-5)
    assert ww.lifetime[0] == pytest.approx(1.99)

def testABubbleRisesAtItsTerminalSpeed():

    fluid = lattice(16, 16, 30, top = 0.75)
    ww = whitewater()
    ww.positions, ww.velocities = np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32)
    ww.lifetime, ww.kinds = np.full(1, 1.0, np.float32), np.full(1, bubble, np.int32)
    for _ in range(200):
        ww.advance(fluid, np.zeros_like(fluid), np.full(len(fluid), spacing**3), 0.004)

    assert ww.kinds[0] == bubble
    assert ww.velocities[0] == pytest.approx([0.0, 0.0, 0.5 * gravity * 0.05], abs = 1e-4)

def testSpawnedParticlesFillTheirCylinder():

    ww = whitewater()
    velocity, dt = np.array([2.0, 1.0, -0.5]), 0.01
    measures = {'positions': np.zeros((1, 3)), 'velocities': velocity[None], 'volumes': np.full(1, spacing**3),
                'spawnCount': np.array([40000.4]), 'generation': np.array([0.5])}
    spawned = ww.spawn(measures, dt)
    speed = np.linalg.norm(velocity)
    along = spawned @ (velocity / speed)
    across = np.linalg.norm(spawned - along[:, None] * velocity / speed, axis = 1)

    assert len(spawned) in (40000, 40001)
    assert along.min() >= 0.0 and along.max() <= speed * dt + 1e-12
    assert across.max() <= 0.5 * spacing + 1e-12
    # Uniform in the disc and along the axis
    assert np.mean(across**2) == pytest.approx(0.5 * (0.5 * spacing)**2, rel = 0.02)
    assert np.mean(along) == pytest.approx(0.5 * speed * dt, rel = 0.02)
    assert ww.lifetime == pytest.approx(np.full(len(spawned), 2.5))

    counts = whitewater().spawn({**measures, 'positions': np.zeros((20000, 3)), 'velocities': np.tile(velocity, (20000, 1)),
                                 'volumes': np.full(20000, spacing**3), 'spawnCount': np.full(20000, 0.3),
                                 'generation': np.zeros(20000)}, dt)
    assert len(counts) == pytest.approx(6000, rel = 0.05)

def testOneSplatDepositsItsProjectedArea():

    ww = whitewater(radii = (0.0, 2.0, 0.0))
    camera = Camera(eye = [0.0, -5.0, 0.0], target = [0.0, 0.0, 0.0], width = 64, height = 48)
    ww.positions = np.array([[0.01, 0.0, 0.02]], np.float32)
    ww.kinds = np.full(1, foam, np.int32)
    ww.lifetime = np.ones(1, np.float32)
    front, behind = ww.splat(camera, np.full((48, 64), np.inf), spacing, 3.0)
    depth = 5.0
    radius = 2.0 * spacing * camera.focal / depth                                    # [px]

    assert front.sum() == pytest.approx(math.pi * radius**2, rel = 1e-4)
    assert behind.sum() == 0.0
