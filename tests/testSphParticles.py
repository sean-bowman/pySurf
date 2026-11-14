
# -- pySurf: SPH Particle Tests -- #

'''

Checks of pysurf.sphParticles: lattice counts and placement, region fills, bed-fitted columns (half
a spacing over a sloping floor in 2D and 3D, the lattice over a flat one), the masses and initial
state of an assembled system, and the energy diagnostics against hand sums.

Sean Bowman [10/11/2026]

'''

import numpy as np
import pytest

from pysurf.sphParticles import ParticleSystem, bedFittedPoints, fillRegion, jitterPoints, latticePoints

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

def lowestInEachColumn(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    '''Each column's horizontal position and its lowest point's height.'''

    columns, inverse = np.unique(points[:, :-1], axis = 0, return_inverse = True)
    lowest = np.full(len(columns), np.inf)
    np.minimum.at(lowest, inverse.reshape(-1), points[:, -1])

    return columns, lowest

def testBedFittedColumnsStandHalfASpacingAboveASlopingFloor():

    spacing = 0.05
    points = bedFittedPoints([0.0, 0.0], [1.0, 1.0], spacing, lambda x: 0.1 + 0.3 * x)
    columns, lowest = lowestInEachColumn(points)

    # The lattice's 20 columns, each from half a spacing over the floor, a spacing apart, ending within a spacing of the top
    assert columns[:, 0] == pytest.approx(np.arange(0.025, 1.0, spacing))
    assert lowest == pytest.approx(0.1 + 0.3 * columns[:, 0] + 0.5 * spacing)
    for x in columns[:, 0]:
        z = np.sort(points[points[:, 0] == x, 1])
        assert np.diff(z) == pytest.approx(spacing)
        assert 1.0 - spacing <= z[-1] < 1.0

def testBedFittedColumnsFollowAFloorIn3D():

    spacing = 0.1
    floor = lambda x, y: 0.2 * x + 0.1 * y**2
    points = bedFittedPoints([0.0, -0.5, 0.0], [1.0, 0.5, 1.0], spacing, floor)
    columns, lowest = lowestInEachColumn(points)

    assert len(columns) == 10 * 10
    assert lowest == pytest.approx(floor(columns[:, 0], columns[:, 1]) + 0.5 * spacing)

@pytest.mark.parametrize('lower, upper', [([0.0, 0.0], [1.0, 0.5]), ([0.0, 0.0, 0.0], [0.2, 0.3, 0.4])])
def testBedFittedColumnsOnAFlatFloorAreTheLattice(lower, upper):

    def ordered(points: np.ndarray) -> np.ndarray:
        return points[np.lexsort(points.T[::-1])]

    fitted = bedFittedPoints(lower, upper, 0.1, lambda *horizontal: np.zeros_like(horizontal[0]))
    lattice = latticePoints(lower, upper, 0.1)

    assert fitted.shape == lattice.shape
    assert ordered(fitted) == pytest.approx(ordered(lattice), abs = 1e-12)

@pytest.mark.parametrize('dimension', [2, 3])
def testJitteredPointsStayInTheWaterWithinTheirAmplitude(dimension):

    '''

    Bed-fitted columns over a floor sloping across the box, under a ceiling a few spacings above
    it in places, moved by up to 0.3 spacings: every point stays, inside the walls, above the
    floor's line and below the ceiling, within 0.3 spacings of its place along each horizontal
    axis, and vertically within that plus twice the steepest face's rise (0.6 here) over the
    horizontal move; the same seed gives the same points, another seed others.

    '''

    spacing, amplitude = 0.05, 0.3
    if dimension == 2:
        floor = lambda x: 0.4 * x
        ceiling = lambda x: 0.5 + 0.1 * np.sin(6.0 * x)
        lower, upper = [0.0], [1.0]
    else:
        floor = lambda x, y: 0.4 * x + 0.1 * y
        ceiling = lambda x, y: 0.5 + 0.1 * np.sin(6.0 * x) + 0.0 * y
        lower, upper = [0.0, 0.0], [1.0, 0.4]
    laid = bedFittedPoints(lower + [0.0], upper + [0.7], spacing, floor)
    laid = laid[laid[:, -1] < ceiling(*laid[:, :-1].T)]
    moved = jitterPoints(laid, spacing, amplitude, 7, lower, upper, floor, ceiling)
    horizontal = moved[:, :-1]

    assert moved.shape == laid.shape
    shift = np.abs(moved - laid)
    assert np.all(shift[:, :-1] <= amplitude * spacing + 1e-12)
    rise = 2.0 * 0.6 * np.sum(shift[:, :-1], axis = 1)
    assert np.all(shift[:, -1] <= amplitude * spacing + rise + 1e-12)
    assert np.all((horizontal >= np.array(lower) + 0.5 * spacing - 1e-12)
                  & (horizontal <= np.array(upper) - 0.5 * spacing + 1e-12))
    assert np.all(moved[:, -1] >= floor(*horizontal.T) + 0.5 * spacing - 1e-12)
    assert np.all(moved[:, -1] < ceiling(*horizontal.T))
    # Nearly every point moves; the few left in place sit in water thinner than their offset
    assert np.count_nonzero(np.all(moved == laid, axis = 1)) < 0.01 * len(laid)
    assert jitterPoints(laid, spacing, amplitude, 7, lower, upper, floor, ceiling) == pytest.approx(moved, abs = 0.0)
    assert not np.allclose(jitterPoints(laid, spacing, amplitude, 8, lower, upper, floor, ceiling), moved)
    assert jitterPoints(laid, spacing, 0.0, 7, lower, upper, floor, ceiling) == pytest.approx(laid, abs = 0.0)

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

def testAppendedParticlesJoinTheStateAtTheEnd():

    particles = ParticleSystem.assemble(np.zeros((3, 3)), np.ones((2, 3)), 0.1, 1000.0)
    members = particles.append(np.full((4, 3), 2.0), np.ones((4, 3)), np.full(4, 1010.0), 0.5, isFluid = False)

    assert members.tolist() == [5, 6, 7, 8]
    assert particles.nParticles == 9 and particles.nFluid == 3
    assert particles.densities[members] == pytest.approx(1010.0)
    assert particles.masses[members] == pytest.approx(0.5)
    assert particles.accelerations.shape == (9, 3) and particles.pressures.shape == (9,)
