
# -- pySurf: SPH Particle Tests -- #

'''

Checks of pysurf.sphParticles: lattice counts and placement, region fills, the masses and initial
state of an assembled system, and the energy diagnostics against hand sums.

Sean Bowman [10/11/2026]

'''

import numpy as np
import pytest

from pysurf.sphParticles import ParticleSystem, fillRegion, latticePoints

def testTheLatticeFillsTheBoxHalfASpacingInFromEachFace():

    points = latticePoints([0.0, 0.0], [1.0, 0.5], 0.1)

    assert len(points) == 10 * 5
    assert points.min(axis = 0) == pytest.approx([0.05, 0.05])
    assert points.max(axis = 0) == pytest.approx([0.95, 0.45])

def testTheLatticeWorksIn3D():

    assert latticePoints([0.0, 0.0, 0.0], [0.2, 0.3, 0.4], 0.1).shape == (2 * 3 * 4, 3)

def testFillRegionKeepsOnlyAcceptedPoints():

    inside = lambda points: np.hypot(points[:, 0] - 0.5, points[:, 1] - 0.5) < 0.3
    points = fillRegion(inside, [0.0, 0.0], [1.0, 1.0], 0.01)

    assert np.all(inside(points))
    # A disk of radius 0.3 on a 0.01 lattice: area / spacing^2 to within the perimeter cells
    assert len(points) == pytest.approx(np.pi * 0.3**2 / 0.01**2, rel = 0.01)

def testAssembleOrdersFluidFirstWithCellMasses():

    fluid = latticePoints([0.0, 0.0], [0.2, 0.2], 0.05)
    boundary = np.array([[-0.025, 0.025], [-0.025, 0.075]])
    velocities = np.tile([0.3, -0.1], (len(fluid), 1))
    particles = ParticleSystem.assemble(fluid, boundary, 0.05, 1000.0, velocities)

    assert (particles.nFluid, particles.nBoundary, particles.dimensions) == (16, 2, 2)
    assert particles.masses == pytest.approx(np.full(18, 1000.0 * 0.05**2))
    assert particles.densities == pytest.approx(np.full(18, 1000.0))
    assert np.all(particles.isFluid[:16]) and not np.any(particles.isFluid[16:])
    assert particles.velocities[:16] == pytest.approx(velocities)
    assert particles.velocities[16:] == pytest.approx(np.zeros((2, 2)))

def testAnEmptyBoundaryIsAccepted():

    particles = ParticleSystem.createUniform(np.array([0.0, 0.0]), np.array([0.1, 0.1]), 0.05, 1000.0)

    assert (particles.nFluid, particles.nBoundary) == (4, 0)

def testEnergyDiagnosticsAreHandSums():

    fluid = np.array([[0.0, 1.0], [0.0, 2.0]])
    boundary = np.array([[0.0, 5.0]])
    particles = ParticleSystem.assemble(fluid, boundary, 0.1, 1000.0, np.array([[3.0, 4.0], [0.0, 0.0]]))
    particles.velocities[2] = [100.0, 0.0]
    mass = 1000.0 * 0.1**2

    assert particles.kineticEnergy() == pytest.approx(0.5 * mass * 25.0)
    assert particles.potentialEnergy(9.81) == pytest.approx(mass * 9.81 * 3.0)
    assert particles.maxSpeed() == pytest.approx(5.0)

def testDensityErrorIsTheLargestFluidDeparture():

    particles = ParticleSystem.createUniform(np.array([0.0, 0.0]), np.array([0.1, 0.1]), 0.05, 1000.0)
    particles.densities[:] = [1000.0, 990.0, 1020.0, 1005.0]

    assert particles.maxDensityError(1000.0) == pytest.approx(0.02)
