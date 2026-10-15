
# -- pySurf: SPH Boundary Tests -- #

'''

Checks of pysurf.sphBoundaries: the boundary particles sit where they should (behind each face, one
spacing from the fluid lattice, the stated number of layers deep), and enforcement returns stray
fluid to the water side and removes only the velocity into the solid.

Sean Bowman [10/11/2026]

'''

import numpy as np
import pytest

from pysurf.bathymetry import Bathymetry
from pysurf.sphBoundaries import BedBoundary, ContainerBoundary
from pysurf.sphParticles import ParticleSystem, latticePoints

s = 0.02    # [m]

#----------------------------------------------------------------------#
# -- Container -- #
#----------------------------------------------------------------------#

def testTheContainerShellIsOutsideTheInterior():

    boundary = ContainerBoundary(np.array([0.0, 0.0]), np.array([0.4, 0.3]), s, nLayers = 3)
    points = boundary.positions()
    inside = np.all((points > boundary.containerMin) & (points < boundary.containerMax), axis = 1)

    assert not np.any(inside)
    assert points[:, 0].min() == pytest.approx(-2.5 * s)
    assert points[:, 0].max() == pytest.approx(0.4 + 2.5 * s)
    assert points[:, 1].min() == pytest.approx(-2.5 * s)

def testAnOpenTopHasNoLidAndAClosedTopDoes():

    openTop = ContainerBoundary(np.array([0.0, 0.0]), np.array([0.4, 0.3]), s).positions()
    closed = ContainerBoundary(np.array([0.0, 0.0]), np.array([0.4, 0.3]), s, openTop = False).positions()
    overTheFluid = lambda points: np.sum((points[:, 1] > 0.3) & (points[:, 0] > 0.0) & (points[:, 0] < 0.4))

    assert overTheFluid(openTop) == 0
    assert overTheFluid(closed) == 3 * 20

def testTheFluidLatticeSitsOneSpacingFromTheWalls():

    boundary = ContainerBoundary(np.array([0.0, 0.0]), np.array([0.4, 0.3]), s)
    fluid = latticePoints([0.0, 0.0], [0.4, 0.2], s)
    wall = boundary.positions()
    gaps = np.min(np.linalg.norm(fluid[:, None, :] - wall[None, :, :], axis = 2), axis = 1)

    assert gaps.min() == pytest.approx(s)

def testContainerEnforcementClampsAndStopsOutwardMotion():

    boundary = ContainerBoundary(np.array([0.0, 0.0]), np.array([0.4, 0.3]), s)
    fluid = np.array([[-0.01, 0.1], [0.2, -0.02], [0.2, 0.5], [0.2, 0.1]])
    particles = ParticleSystem.assemble(fluid, np.zeros((0, 2)), s, 1000.0,
                                        np.array([[-1.0, 0.5], [0.3, -2.0], [0.0, 1.0], [0.2, 0.2]]))
    boundary.enforce(particles)

    assert particles.positions[0] == pytest.approx([0.0, 0.1])
    assert particles.velocities[0] == pytest.approx([0.0, 0.5])
    assert particles.positions[1] == pytest.approx([0.2, 0.0])
    assert particles.velocities[1] == pytest.approx([0.3, 0.0])
    # The open top lets water rise freely
    assert particles.positions[2] == pytest.approx([0.2, 0.5])
    assert particles.velocities[3] == pytest.approx([0.2, 0.2])

#----------------------------------------------------------------------#
# -- Bed -- #
#----------------------------------------------------------------------#

@pytest.fixture
def beach() -> Bathymetry:

    return Bathymetry.planeBeach(depth = 0.3, toeX = 1.0, slope = 0.1, endX = 5.0)

def testBedLayersSitBelowTheBedAlongItsNormal(beach):

    boundary = BedBoundary(beach, s, nLayers = 3)
    points = boundary.positions()
    below = beach.elevation(points[:, 0]) - points[:, 1]

    # Distance below the bed measured along the normal is (layer + 1/2) spacing
    normalDistance = below / np.sqrt(1.0 + beach.slope(points[:, 0])**2)
    layers = normalDistance / s - 0.5

    assert np.all(below > 0.0)
    assert np.allclose(layers[(points[:, 0] > 0.1) & (points[:, 0] < 0.9)], np.round(layers[(points[:, 0] > 0.1) & (points[:, 0] < 0.9)]), atol = 1e-9)
    assert normalDistance.max() == pytest.approx(2.5 * s, rel = 0.02)

def testTheOffshoreWallClosesTheCorner(beach):

    boundary = BedBoundary(beach, s, nLayers = 3, wallTop = 0.5)
    wall = boundary.positions()
    behind = wall[wall[:, 0] < beach.startX]

    assert behind[:, 0].min() == pytest.approx(-2.5 * s)
    assert behind[:, 1].min() < -2.0 * s
    assert behind[:, 1].max() < 0.5 and behind[:, 1].max() > 0.5 - s

def testTheBedExtrudesTo3DWithSideWalls(beach):

    width = 0.1
    points = BedBoundary(beach, s, nLayers = 2, wallTop = 0.5, width = width).positions()

    assert points.shape[1] == 3
    assert np.any(points[:, 1] < 0.0) and np.any(points[:, 1] > width)
    assert not np.any((points[:, 1] > 0.0) & (points[:, 1] < width) & (points[:, 2] > beach.elevation(points[:, 0]) + 1e-9)
                      & (points[:, 0] > beach.startX))

def testBedEnforcementLiftsFluidAndRemovesOnlyTheInwardVelocity(beach):

    boundary = BedBoundary(beach, s, wallTop = 0.5)
    x = 3.0
    bedZ = float(beach.elevation(x))
    normal = beach.normal(x)
    tangent = np.array([normal[1], -normal[0]])
    velocity = 0.7 * tangent - 0.4 * normal
    particles = ParticleSystem.assemble(np.array([[x, bedZ - 0.01], [-0.05, 0.1], [6.0, 0.6]]), np.zeros((0, 2)), s, 1000.0,
                                        np.vstack([velocity, [-1.0, 0.0], [1.0, 0.0]]))
    boundary.enforce(particles)

    assert particles.positions[0, 1] == pytest.approx(bedZ + 0.25 * s)
    assert particles.velocities[0] == pytest.approx(0.7 * tangent)
    assert particles.positions[1, 0] == pytest.approx(0.25 * s)
    assert particles.velocities[1, 0] == 0.0
    assert particles.positions[2, 0] == pytest.approx(beach.endX)
    assert particles.velocities[2, 0] == 0.0

def testFluidAboveTheBedIsLeftAlone(beach):

    boundary = BedBoundary(beach, s, wallTop = 0.5)
    particles = ParticleSystem.assemble(np.array([[2.0, 0.25]]), np.zeros((0, 2)), s, 1000.0, np.array([[0.3, -0.5]]))
    boundary.enforce(particles)

    assert particles.positions[0] == pytest.approx([2.0, 0.25])
    assert particles.velocities[0] == pytest.approx([0.3, -0.5])
