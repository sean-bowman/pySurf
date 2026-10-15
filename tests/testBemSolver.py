
# -- pySurf: Boundary Element Solver Tests -- #

'''

Checks of pysurf.bemSolver against closed forms.

The closed-form element integrals are checked against numerical quadrature, including a point at
an element's end and a point a millimetre from its middle. The boundary solution is checked against
the exact harmonic function phi = cosh(k (z + d)) cos(k x) in a rectangle, with zero flux on the
bed and walls: the surface flux and the bed potential converge at second order away from the
corners. The time stepping is checked against the Airy period of a small standing wave at k d = 1
and k d = 3, where the SGN equations are 0.8 and 13 percent off, and against conservation of volume
and energy. The geometry helpers are checked on constructed surfaces: a crossing, an overhang and
the tube under it.

Sean Bowman [10/11/2026]

'''

import math

import numpy as np
import pytest
from scipy.integrate import quad

from pysurf import constants
from pysurf.bathymetry import Bathymetry
from pysurf.bemSolver import (BemConfig, BemSolver, FlowField, elementIntegrals, foldDistance, frontSlopeMinimum,
                              interiorPotential, interiorVelocity, jetGap, jetGeometry, regridSurface, solveBoundary,
                              solveNeumann, surfaceCrossesItself, windingShare)

g = constants.gravity

#----------------------------------------------------------------------#
# -- Integrals and Boundary Solution -- #
#----------------------------------------------------------------------#

def quadratureIntegrals(point, start, end):

    '''The four element integrals by adaptive quadrature, split at the point's foot on the element.'''

    length = np.linalg.norm(end - start)
    tangent = (end - start) / length
    normal = np.array([tangent[1], -tangent[0]])
    u0, v0 = (point - start) @ tangent, (point - start) @ normal
    breaks = [u0] if 0.0 < u0 < length else None

    def r(u):
        return start + u * tangent - point

    def green(u):
        return -math.log(max(np.linalg.norm(r(u)), 1e-300)) / (2.0 * math.pi)

    def greenNormal(u):
        vector = r(u)
        return -(vector @ normal) / (vector @ vector) / (2.0 * math.pi)

    onLine = abs(v0) < 1e-14
    shapes = (lambda u: 1.0 - u / length, lambda u: u / length)
    G = [quad(lambda u, N = N: N(u) * green(u), 0.0, length, points = breaks, limit = 400)[0] for N in shapes]
    H = [0.0, 0.0] if onLine else [quad(lambda u, N = N: N(u) * greenNormal(u), 0.0, length, points = breaks,
                                        limit = 400)[0] for N in shapes]

    return G + H

def testElementIntegralsMatchQuadrature():

    rng = np.random.default_rng(0)
    starts = rng.uniform(-1.0, 1.0, (4, 2))
    ends = starts + rng.uniform(-0.6, 0.6, (4, 2))
    midpoint = 0.5 * (starts[3] + ends[3])
    points = np.vstack([rng.uniform(-1.5, 1.5, (5, 2)), starts[:2], ends[2:3], midpoint + [1e-3, -2e-3]])
    Ga, Gb, Ha, Hb = elementIntegrals(points, starts, ends)

    for m, point in enumerate(points):
        for e in range(len(starts)):
            reference = quadratureIntegrals(point, starts[e], ends[e])
            assert [Ga[m, e], Gb[m, e], Ha[m, e], Hb[m, e]] == pytest.approx(reference, abs = 1e-9)

def rectangleErrors(nodes: int) -> tuple[float, float]:

    '''Surface flux error away from the corners and bed potential error, for phi = cosh(k (z + d)) cos(k x).'''

    width, depth = 2.0, 1.0
    k = math.pi / width
    x = np.linspace(0.0, width, nodes + 1)
    surface = np.column_stack([x, np.zeros_like(x)])
    bed = np.column_stack([x, np.full_like(x, -depth)])
    solution = solveBoundary(surface, math.cosh(k * depth) * np.cos(k * x), bed, width / nodes)
    flux = k * math.sinh(k * depth) * np.cos(k * x)
    fluxError = np.max(np.abs(solution.normalDerivative - flux)[2:-2]) / (k * math.sinh(k * depth))
    bedError = np.max(np.abs(solution.bedPotential - np.cos(k * x)))

    return fluxError, bedError

def testTheBoundarySolutionConvergesToAnExactHarmonicFunction():

    coarse, fine = rectangleErrors(40), rectangleErrors(80)

    assert fine[0] < 2e-4 and fine[1] < 2e-4
    assert coarse[0] / fine[0] > 3.0
    assert coarse[1] / fine[1] > 3.0

#----------------------------------------------------------------------#
# -- Time Stepping -- #
#----------------------------------------------------------------------#

def standingWave(kd: float, periods: float = 2.3) -> dict:

    '''A small standing wave in a closed flat basin a half wavelength long, run for a few periods.'''

    depth, amplitude = 1.0, 0.002
    k = kd / depth
    width = math.pi / k
    basin = Bathymetry([0.0, width], [0.0, 0.0], stillWaterLevel = depth)
    spacing = width / 40
    solver = BemSolver(basin, BemConfig(surfaceSpacing = spacing, crestSpacing = spacing, crestBehind = 0.0,
                                        crestAhead = 0.0, transition = 1.0, bedSpacing = spacing, regridInterval = 0))
    x = np.linspace(0.0, width, 41)
    solver.setSurface(np.column_stack([x, depth + amplitude * np.cos(k * x)]), np.zeros_like(x), regrid = False)
    airy = 2.0 * math.pi / math.sqrt(g * k * math.tanh(kd))
    volume, energy = solver.volume(), sum(solver.energy())

    times, levels = [], []
    while solver.time < periods * airy:
        solver.step()
        times.append(solver.time)
        levels.append(solver.surface[0, 1] - depth)
    times, levels = np.asarray(times), np.asarray(levels)
    up = np.flatnonzero((levels[:-1] < 0.0) & (levels[1:] >= 0.0))
    crossings = times[up] - levels[up] * (times[up + 1] - times[up]) / (levels[up + 1] - levels[up])

    return {'period': float(np.mean(np.diff(crossings))), 'airy': airy, 'volumeChange': solver.volume() - volume,
            'energyChange': sum(solver.energy()) / energy - 1.0}

def rectangleField(nodes: int) -> tuple:

    '''The boundary solution for phi = cosh(k (z + d)) cos(k x) in a 2 x 1 rectangle, with k and the node spacing.'''

    width, depth = 2.0, 1.0
    k = math.pi / width
    x = np.linspace(0.0, width, nodes + 1)
    surface = np.column_stack([x, np.zeros_like(x)])
    bed = np.column_stack([x, np.full_like(x, -depth)])

    return surface, math.cosh(k * depth) * np.cos(k * x), bed, k, width / nodes

def testTheInteriorPotentialAndVelocityMatchAnExactHarmonicField():

    # Points from one element below the surface to near the bed, clear of the walls; the error is
    # second order in the node spacing, 1.4e-4 in phi and 2e-4 k in velocity at 80 nodes
    surface, potential, bed, k, spacing = rectangleField(80)
    solution = solveBoundary(surface, potential, bed, spacing)
    x, z = np.meshgrid(np.linspace(0.3, 1.7, 8), np.linspace(-0.95, -spacing, 6))
    points = np.column_stack([x.ravel(), z.ravel()])
    exact = np.cosh(k * (points[:, 1] + 1.0)) * np.cos(k * points[:, 0])
    velocity = np.column_stack([-k * np.cosh(k * (points[:, 1] + 1.0)) * np.sin(k * points[:, 0]),
                                k * np.sinh(k * (points[:, 1] + 1.0)) * np.cos(k * points[:, 0])])

    assert interiorPotential(solution, points) == pytest.approx(exact, abs = 3e-4)
    assert np.max(np.hypot(*(interiorVelocity(solution, points) - velocity).T)) < 5e-4 * k

def testTheSurfacePotentialFollowsFromHowTheSurfaceMoves():

    # A linear progressive wave eta = a cos(k x) rising at a omega sin(k x), between walls where eta
    # crosses zero and u vanishes at every depth: the surface potential is (g a / omega) sin(k x)
    gravity, depth, length, amplitude, nodes = 9.81, 1.0, 4.0, 1e-5, 160
    k = 2.0 * math.pi / length
    omega = math.sqrt(gravity * k * math.tanh(k * depth))
    x = np.linspace(0.25 * length, 2.25 * length, nodes + 1)
    surface = np.column_stack([x, amplitude * np.cos(k * x)])
    bed = np.column_stack([x, np.full_like(x, -depth)])
    rise = amplitude * omega * np.sin(k * x)
    slope = -amplitude * k * np.sin(k * x)
    phi = solveNeumann(surface, rise / np.sqrt(1.0 + slope**2), bed, 2.0 * length / nodes).surfacePotential
    exact = gravity * amplitude / omega * np.sin(k * x)

    assert np.max(np.abs((phi - phi.mean()) - (exact - exact.mean()))) < 1.5e-3 * gravity * amplitude / omega

def testTheWindingShareIsOneInsideAndZeroOutside():

    surface, potential, bed, _, spacing = rectangleField(20)
    solution = solveBoundary(surface, potential, bed, spacing)
    points = np.array([[1.0, -0.5], [0.05, -0.95], [2.5, -0.5], [1.0, 0.3], [1.0, -1.2]])

    assert windingShare(solution, points) == pytest.approx([1.0, 1.0, 0.0, 0.0, 0.0], abs = 1e-9)

def testThePressureMatchesTheExactUnsteadyField():

    # With zero pressure on the flat surface, phi_t there is -|u|^2 / 2 = -k^2 (sinh^2 kd + sin^2 kx) / 2,
    # and the harmonic field with that data and no flux through the bed and walls is
    # -k^2 (sinh^2 kd + 1/2) / 2 + k^2 cosh(2 k (z + d)) cos(2 k x) / (4 cosh(2 k d))
    gravity, density = 9.81, 1000.0
    surface, potential, bed, k, spacing = rectangleField(80)
    field = FlowField(surface, potential, bed, spacing, 0.0, gravity, density)
    x, z = np.meshgrid(np.linspace(0.3, 1.7, 8), np.linspace(-0.9, -2.0 * spacing, 6))
    x, z = x.ravel(), z.ravel()
    speedSquared = k**2 * (np.cosh(k * (z + 1.0))**2 * np.sin(k * x)**2 + np.sinh(k * (z + 1.0))**2 * np.cos(k * x)**2)
    rate = (-0.5 * k**2 * (math.sinh(k)**2 + 0.5)
            + 0.25 * k**2 * np.cosh(2.0 * k * (z + 1.0)) / math.cosh(2.0 * k) * np.cos(2.0 * k * x))
    pressure = -density * (rate + 0.5 * speedSquared + gravity * z)
    scale = density * k**2 * math.cosh(k)**2

    assert np.max(np.abs(field.pressure(np.column_stack([x, z])) - pressure)) < 3e-4 * scale

@pytest.mark.parametrize('kd', [1.0, 3.0])
def testAStandingWaveOscillatesAtTheAiryPeriod(kd):

    '''Within 0.2 percent at 40 nodes per half wavelength; it reads -0.03 and -0.11 percent.'''

    result = standingWave(kd)

    assert result['period'] == pytest.approx(result['airy'], rel = 0.002)

def testVolumeAndEnergyAreConserved():

    result = standingWave(1.0)

    assert abs(result['volumeChange']) < 1e-7
    assert abs(result['energyChange']) < 1e-4

#----------------------------------------------------------------------#
# -- Surface Geometry -- #
#----------------------------------------------------------------------#

def testRegriddingKeepsTheShapeAndTheEnds():

    x = np.linspace(0.0, 10.0, 401)
    surface = np.column_stack([x, 1.0 + 0.3 / np.cosh(x - 5.0)**2])
    potential = np.sin(x)
    newSurface, newPotential = regridSurface(surface, potential, lambda s: np.full_like(s, 0.1))

    assert newSurface[0, 0] == 0.0 and newSurface[-1, 0] == 10.0
    assert newSurface[:, 1] == pytest.approx(1.0 + 0.3 / np.cosh(newSurface[:, 0] - 5.0)**2, abs = 1e-5)
    assert newPotential == pytest.approx(np.sin(newSurface[:, 0]), abs = 1e-5)

def plungingSurface() -> np.ndarray:

    '''

    A surface that rises to a crest, curls into a jet around a tube, and runs down a face that passes
    under the jet tip. The top of the jet is an arc of radius 0.25 about (4.5, 1.25) from 160 degrees
    round to -20, the underside an arc of radius 0.15 about the same center from -20 degrees round to
    270, and the face a straight line from there to (6.0, 1.0) in elements 0.05 long.

    '''

    outer = np.radians(np.linspace(160.0, -20.0, 73))
    inner = np.radians(np.linspace(-20.0, 270.0, 117))
    back = np.column_stack([np.linspace(0.0, 4.265, 50, endpoint = False), np.linspace(1.0, 1.335, 50, endpoint = False)])
    jet = np.column_stack([4.5 + 0.25 * np.cos(outer), 1.25 + 0.25 * np.sin(outer)])
    tube = np.column_stack([4.5 + 0.15 * np.cos(inner), 1.25 + 0.15 * np.sin(inner)])
    face = np.column_stack([np.linspace(4.5, 6.0, 31)[1:], np.linspace(1.1, 1.0, 31)[1:]])

    return np.vstack([back, jet, tube, face])

def testAnOverhangIsFoundAndItsTubeMeasured():

    # The jet node nearest the face is the end of the jet's top, at -20 degrees; the face node at
    # x = 4.75 is nearest it and closes the tube, whose back is at x = 4.35 and top at z = 1.40
    flat = np.column_stack([np.linspace(0.0, 1.0, 20), np.ones(20)])
    surface = plungingSurface()
    jet = jetGeometry(surface)
    tip = [4.5 + 0.25 * np.cos(np.radians(-20.0)), 1.25 + 0.25 * np.sin(np.radians(-20.0))]

    assert jetGeometry(flat) is None
    assert frontSlopeMinimum(flat) == pytest.approx(1.0)
    assert frontSlopeMinimum(surface) < 0.0
    assert jet.tip == pytest.approx(tip, abs = 1e-9)
    assert jet.impactPoint == pytest.approx([4.75, 1.1 - 0.1 * 0.25 / 1.5], abs = 1e-9)
    assert jet.tubeLength == pytest.approx(0.40, abs = 1e-9)
    assert jet.tubeHeight == pytest.approx(1.40 - (1.1 - 0.1 * 0.25 / 1.5), abs = 1e-9)
    assert jet.cavityArea > 0.0

def testATubeMeasureIgnoresANodeWiggle():

    # A sawtooth a thousandth of the tube across flips dx/ds at single nodes around the jet
    surface = plungingSurface()
    wiggled = surface.copy()
    wiggled[50:240, 0] += 0.001 * (-1.0) ** np.arange(190)
    clean, rough = jetGeometry(surface), jetGeometry(wiggled)

    assert rough.tubeLength == pytest.approx(clean.tubeLength, rel = 0.02)
    assert rough.tubeHeight == pytest.approx(clean.tubeHeight, rel = 0.02)
    assert rough.cavityArea == pytest.approx(clean.cavityArea, rel = 0.02)

def testTheGapFromJetToFaceIsMeasured():

    # The closest approach is the last node of the jet's top to the face line below it
    surface = plungingSurface()
    tip = np.array([4.5 + 0.25 * np.cos(np.radians(-20.0)), 1.25 + 0.25 * np.sin(np.radians(-20.0))])
    direction = np.array([1.5, -0.1]) / np.hypot(1.5, 0.1)
    offset = tip - np.array([4.5, 1.1])
    expected = abs(offset[0] * direction[1] - offset[1] * direction[0])
    gap, element = jetGap(surface, len(surface))
    flat = np.column_stack([np.linspace(0.0, 1.0, 20), np.ones(20)])

    assert gap == pytest.approx(expected, abs = 1e-9)
    assert element == pytest.approx(np.hypot(1.5, 0.1) / 30.0, abs = 1e-9)
    assert jetGap(flat, len(flat)) is None

def testAFoldIsMeasuredAcrossTheTubeMouth():

    # The last node of the jet's top (index 122) is nearest the face node at x = 4.75 across the mouth
    # of the tube; the round tip and the tube's own curve are not folds
    surface = plungingSurface()
    fold = foldDistance(surface)
    tip = np.array([4.5 + 0.25 * np.cos(np.radians(-20.0)), 1.25 + 0.25 * np.sin(np.radians(-20.0))])
    flat = np.column_stack([np.linspace(0.0, 1.0, 20), np.ones(20)])

    assert fold[122] == pytest.approx(np.hypot(4.75 - tip[0], 1.1 - 0.1 * 0.25 / 1.5 - tip[1]), abs = 1e-9)
    assert np.all(np.isinf(foldDistance(flat)))

def testACrossingIsDetected():

    crossing = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, -0.5], [0.4, -1.0]])
    straight = np.column_stack([np.linspace(0.0, 1.0, 10), np.zeros(10)])

    assert surfaceCrossesItself(crossing, 0, len(crossing), gap = 2)
    assert not surfaceCrossesItself(straight, 0, len(straight))
