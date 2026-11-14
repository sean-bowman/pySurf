
# -- pySurf: Potential Flow to SPH Hand-off Tests -- #

'''

Tests of pysurf.sphHandoff and pysurf.SplashUp: still water hands off at rest with hydrostatic
pressure on the fluid and the boundary alike, a plunging surface is filled with particles that
leave the tube under the jet empty, bed-fitted seeds stand half a spacing above a slope, the piston reads the flow behind it as a wave translating
unchanged, the depth-averaged velocity of a solitary wave is c eta / (d + eta), and a short
splash-up from a solitary wave builds, runs, keeps its particles and writes a GIF.

Sean Bowman [10/12/2026]

'''

import math

import numpy as np
import pytest
from PIL import Image

from pysurf import constants
from pysurf.bathymetry import Bathymetry
from pysurf.bemSolver import FlowField
from pysurf.errors import InvalidInputError
from pysurf.OverturningWave import OverturningWave
from pysurf.sphBoundaries import BedBoundary
from pysurf.sphHandoff import (PistonDrive, boundaryPressure, clearSurface, columnVelocity, nearestSurfacePoint, seedStillWater,
                               seedWater, windowBathymetry)
from pysurf.SplashUp import SplashUp

def stillTank(width: float = 4.0, depth: float = 1.0, nodes: int = 80) -> tuple:

    '''Still water in a flat tank: the flow field and the bed.'''

    x = np.linspace(0.0, width, nodes + 1)
    surface = np.column_stack([x, np.full_like(x, depth)])
    bed = np.column_stack([x, np.zeros_like(x)])
    field = FlowField(surface, np.zeros_like(x), bed, width / nodes, depth)

    return field, Bathymetry([0.0, width], [0.0, 0.0], depth)

def testStillWaterHandsOffAtRestAndHydrostatic():

    field, bed = stillTank()
    spacing = 0.05
    window = windowBathymetry(bed, 1.0, 3.0)
    fluid = seedWater(field, window, spacing)
    boundary = BedBoundary(window, spacing, nLayers = 3, wallTop = 1.5, endWallTop = 1.5).positions()
    rho, g = constants.freshwaterDensity, constants.gravity

    # Two depths of water at a twentieth of a depth is 40 columns of 20 rows, the walls and bed half a spacing clear
    assert len(fluid) == 40 * 20
    assert np.max(np.abs(field.velocity(fluid))) < 1e-9
    assert field.pressure(fluid) == pytest.approx(rho * g * (1.0 - fluid[:, 1]), abs = 1e-6 * rho * g)
    wet = boundary[:, 1] < 1.0
    assert boundaryPressure(field, window, boundary, spacing)[wet] == pytest.approx(rho * g * (1.0 - boundary[wet, 1]),
                                                                                    abs = 1e-6 * rho * g)

def testAPlungingSurfaceIsFilledAndItsTubeLeftEmpty():

    # A jet over a tube of radius 0.15 about (4.5, 1.25), between flat water at z = 1 out to the walls
    outer = np.radians(np.linspace(160.0, -20.0, 73))
    inner = np.radians(np.linspace(-20.0, 270.0, 117))
    back = np.column_stack([np.linspace(0.0, 4.265, 120, endpoint = False), np.linspace(1.0, 1.335, 120, endpoint = False)])
    jet = np.column_stack([4.5 + 0.25 * np.cos(outer), 1.25 + 0.25 * np.sin(outer)])
    tube = np.column_stack([4.5 + 0.15 * np.cos(inner), 1.25 + 0.15 * np.sin(inner)])
    face = np.column_stack([np.linspace(4.5, 8.0, 101)[1:], np.linspace(1.1, 1.0, 101)[1:]])
    surface = np.vstack([back, jet, tube, face])
    bed = np.column_stack([np.linspace(0.0, 8.0, 81), np.zeros(81)])
    field = FlowField(surface, np.zeros(len(surface)), bed, 0.1, 1.0)
    fluid = seedWater(field, Bathymetry([0.0, 8.0], [0.0, 0.0], 1.0), 0.02)
    radius = np.hypot(fluid[:, 0] - 4.5, fluid[:, 1] - 1.25)

    assert not np.any(radius < 0.15 - 0.01)
    assert np.count_nonzero((radius > 0.16) & (radius < 0.24) & (fluid[:, 1] > 1.25)) > 50
    assert np.min(nearestSurfacePoint(surface, fluid)[0]) >= 0.25 * 0.02 - 1e-12

def testBedFittedSeedsStandHalfASpacingAboveASlope():

    # Still water over a bed flat to x = 2, then 1:8, both as the potential flow's tank and as a window beyond it
    spacing = 0.05
    x = np.linspace(0.0, 4.0, 81)
    bedLine = np.maximum(x - 2.0, 0.0) / 8.0
    field = FlowField(np.column_stack([x, np.ones_like(x)]), np.zeros_like(x), np.column_stack([x, bedLine]), 0.05, 1.0)
    bed = Bathymetry([0.0, 2.0, 4.0], [0.0, 0.0, 0.25], 1.0)

    def gaps(points: np.ndarray) -> np.ndarray:
        columns, inverse = np.unique(points[:, 0], return_inverse = True)
        lowest = np.full(columns.size, np.inf)
        np.minimum.at(lowest, inverse, points[:, 1])
        return lowest - bed.elevation(columns)

    for fitted, lattice in ((seedWater(field, bed, spacing, bedFitted = True), seedWater(field, bed, spacing)),
                            (seedStillWater(bed, spacing, -1.0, bedFitted = True), seedStillWater(bed, spacing, -1.0))):
        assert gaps(fitted) == pytest.approx(0.5 * spacing)
        assert gaps(lattice).max() > spacing
        # The same points over the flat bed, and water up to still water level only
        flatFitted, flatLattice = fitted[fitted[:, 0] < 2.0], lattice[lattice[:, 0] < 2.0]
        assert flatFitted[np.lexsort(flatFitted.T[::-1])] == pytest.approx(flatLattice[np.lexsort(flatLattice.T[::-1])],
                                                                             abs = 1e-12)
        assert np.all(fitted[:, 1] < 1.0)

def testJitteredSeedsStayInTheWater():

    '''

    The plunging surface of the tube test and still water over a slope, seeded in bed-fitted
    columns and moved by up to 0.15 spacings: as many points as the ordered seeds, none in the tube
    or above the surface, all half a spacing or more above the bed.

    '''

    outer = np.radians(np.linspace(160.0, -20.0, 73))
    inner = np.radians(np.linspace(-20.0, 270.0, 117))
    back = np.column_stack([np.linspace(0.0, 4.265, 120, endpoint = False), np.linspace(1.0, 1.335, 120, endpoint = False)])
    jet = np.column_stack([4.5 + 0.25 * np.cos(outer), 1.25 + 0.25 * np.sin(outer)])
    tube = np.column_stack([4.5 + 0.15 * np.cos(inner), 1.25 + 0.15 * np.sin(inner)])
    face = np.column_stack([np.linspace(4.5, 8.0, 101)[1:], np.linspace(1.1, 1.0, 101)[1:]])
    surface = np.vstack([back, jet, tube, face])
    field = FlowField(surface, np.zeros(len(surface)), np.column_stack([np.linspace(0.0, 8.0, 81), np.zeros(81)]), 0.1, 1.0)
    flat = Bathymetry([0.0, 8.0], [0.0, 0.0], 1.0)
    ordered = seedWater(field, flat, 0.02, bedFitted = True)
    moved = seedWater(field, flat, 0.02, bedFitted = True, jitter = 0.15, seed = 4)

    assert len(moved) == len(ordered)
    assert np.all(field.inside(moved))
    assert not np.any(np.hypot(moved[:, 0] - 4.5, moved[:, 1] - 1.25) < 0.15 - 0.01)
    assert np.all(moved[:, 1] >= 0.5 * 0.02 - 1e-12)
    assert np.count_nonzero(np.all(moved == ordered, axis = 1)) < 0.05 * len(ordered)

    sloping = Bathymetry([0.0, 2.0, 4.0], [0.0, 0.0, 0.25], 1.0)
    still = seedStillWater(sloping, 0.05, -1.0, bedFitted = True)
    stillMoved = seedStillWater(sloping, 0.05, -1.0, bedFitted = True, jitter = 0.15, seed = 4)
    assert len(stillMoved) == len(still)
    assert np.all(stillMoved[:, 1] < 1.0)
    assert np.all(stillMoved[:, 1] >= sloping.elevation(stillMoved[:, 0]) + 0.5 * 0.05 - 1e-12)

def testPointsNearTheSurfaceMoveInToTheClearance():

    '''

    Under a flat surface at z = 1, points closer than a quarter spacing to it move straight down
    to a quarter spacing below it; points deeper keep their places.

    '''

    surface = np.column_stack([np.linspace(0.0, 4.0, 41), np.ones(41)])
    points = np.array([[1.03, 0.999], [2.07, 0.99], [3.01, 0.9], [1.55, 0.5]])
    cleared = clearSurface(surface, points, 0.1)

    assert cleared[:2] == pytest.approx(np.array([[1.03, 0.975], [2.07, 0.975]]), abs = 1e-12)
    assert np.all(cleared[2:] == points[2:])
    assert len(clearSurface(surface, np.zeros((0, 2)), 0.1)) == 0

def testThePistonReadsTheFlowBehindAsATranslatingWave():

    field, bed = stillTank()
    piston = PistonDrive(field, bed, 3.0, crestSpeed = 2.0, duration = 1.0)
    piston.stationVelocity = np.linspace(0.0, 1.0, len(piston.stationX))

    assert piston.stationX[0] == pytest.approx(1.0) and piston.stationX[-1] == pytest.approx(3.0)
    assert piston.velocity(3.0, 0.0) == pytest.approx(1.0)
    assert piston.velocity(3.0, 0.5) == pytest.approx(0.5)

@pytest.fixture(scope = 'module')
def solitaryWave() -> OverturningWave:

    wave = OverturningWave(crestSpacingRatio = 0.04, surfaceSpacingRatio = 0.1)
    wave.build()

    return wave

def testTheColumnVelocityOfASolitaryWaveCarriesItsMass(solitaryWave):

    # A wave of permanent form at speed c carries mass c eta through every column, so its
    # depth-averaged velocity is c eta / (d + eta); the starting wave is close to permanent.
    # Columns are taken behind the crest, where a piston reads them
    wave = solitaryWave
    field = FlowField.fromSolver(wave.solver)
    x = wave.crestX + np.array([-2.0, -1.5, -1.0, -0.5, 0.0])
    eta = np.interp(x, wave.solver.surface[:, 0], wave.solver.surface[:, 1]) - 1.0
    celerity = math.sqrt(constants.gravity * (1.0 + wave.waveHeight))

    assert columnVelocity(field, wave.bathymetry, x) == pytest.approx(celerity * eta / (1.0 + eta), rel = 0.03)

def testASplashUpNeedsARunWave():

    with pytest.raises(InvalidInputError):
        SplashUp(OverturningWave())

@pytest.fixture(scope = 'module')
def shortSplash(solitaryWave) -> SplashUp:

    # The window reaches past the water a solitary wave raises ahead of its crest, so the run is not
    # stopped for reaching the wall
    splash = SplashUp(solitaryWave, particlesPerDepth = 10, behindDepths = 2.0, aheadDepths = 5.0, duration = 0.05,
                      frameInterval = 0.01)
    splash.run(progress = False)

    return splash

def testAShortSplashUpKeepsItsParticlesAndItsEnergy(shortSplash):

    summary = shortSplash.summary()
    frames = shortSplash.frames

    assert len(frames) >= 5
    assert all(len(frame.positions) == summary['fluidParticles'] for frame in frames)
    assert summary['kineticEnergyAtEnd'] == pytest.approx(summary['kineticEnergyAtHandoff'], rel = 0.1)
    assert summary['pistonTravel'] > 0.0
    assert summary['stopReason'] == 'end time reached'

def testAShortSplashUpDrawsAndWritesAsAGif(shortSplash, tmp_path):

    assert len(shortSplash.drawFrame(-1).axes) == 3
    path = shortSplash.writeAnimation(str(tmp_path / 'splash.gif'), fps = 4)

    with Image.open(path) as gif:
        assert gif.n_frames >= len(shortSplash.frames)

def testAStillWallClosesTheEnergyBudgetOfAnUnbrokenWave():

    # A solitary wave in a flat tank 30 depths long, whose SPH bed runs five depths further, where
    # still water fills it at rest, 50 columns of 10 rows; behind a wall that stands still, the
    # unbroken wave's energy changes only by the SPH model's damping
    tank = Bathymetry([0.0, 30.0], [0.0, 0.0], 1.0)
    wave = OverturningWave(bathymetry = tank, crestSpacingRatio = 0.04, surfaceSpacingRatio = 0.1)
    wave.build()
    bed = Bathymetry([0.0, 35.0], [0.0, 0.0], 1.0)
    splash = SplashUp(wave, particlesPerDepth = 10, behindDepths = 6.0, aheadDepths = 60.0, piston = False, bed = bed,
                      duration = 0.1, frameInterval = 0.02)
    splash.build()
    beyond = splash.particles.isFluid & (splash.particles.positions[:, 0] > tank.endX)
    splash.run(progress = False)
    _, kinetic, potential = splash.energyHistory()
    energy = kinetic + potential

    assert np.count_nonzero(beyond) == 50 * 10
    assert np.max(np.abs(splash.frames[0].speeds[-np.count_nonzero(beyond):])) == 0.0
    assert energy[-1] == pytest.approx(energy[0], rel = 0.02)
