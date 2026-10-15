
# -- pySurf: Wave Breaking Measure Tests -- #

'''

Checks of pysurf.waveBreaking on constructed particle arrangements whose answer is known by
construction: a face sloping back has negative lean, a vertical face none, an overhanging lip a
lean equal to its overhang; still water ahead of the wave, a face running past the search window and
particle rows lifted under the crest do not count as lean; the tracker records breaking only once
the lean persists; and the surface profile and waterline match the arrangement they were built
from.

Sean Bowman [10/11/2026]

'''

import numpy as np
import pytest

from pysurf.bathymetry import Bathymetry
from pysurf.sphParticles import ParticleSystem, fillRegion, latticePoints
from pysurf.waveBreaking import (BreakingDetector, BreakingTracker, SurfaceBreakingTracker, freeSurfaceProfile,
                                 frontFaceLean, profileCrest, waterlineX, waveFront)

s = 0.05            # [m] particle spacing
swl = 1.0           # [m] still water level

def column(x0: float, x1: float, z0: float, z1: float) -> np.ndarray:

    '''Lattice particles filling a rectangle.'''

    return latticePoints([x0, z0], [x1, z1], s)

def crestWithFace(faceX) -> np.ndarray:

    '''Still water to z = 1, and a crest from x = 0 whose face position at height z is faceX(z).'''

    water = column(-1.0, 3.0, 0.0, swl)
    crest = fillRegion(lambda p: (p[:, 0] > 0.0) & (p[:, 0] < faceX(p[:, 1])), [-1.0, swl], [3.0, swl + 0.4], s)

    return np.vstack([water, crest])

def testAFaceSlopingBackHasNegativeLean():

    positions = crestWithFace(lambda z: 1.0 - 2.0 * (z - swl))

    assert frontFaceLean(positions, 0.0, swl, s, reach = 2.0) < -s

def testAVerticalFaceHasNoLean():

    positions = crestWithFace(lambda z: np.full_like(z, 1.0))

    assert frontFaceLean(positions, 0.0, swl, s, reach = 2.0) == pytest.approx(0.0, abs = 1e-12)

def testALipLeansByItsOverhang():

    '''A vertical face up to z = 1.2, then a lip reaching 0.3 m further shoreward.'''

    positions = crestWithFace(lambda z: np.where(z < swl + 0.2, 1.0, 1.3))

    assert frontFaceLean(positions, 0.0, swl, s, reach = 2.0) == pytest.approx(0.3, abs = 1e-9)

def testStillWaterAheadDoesNotMaskTheFace():

    '''The still water runs to x = 3 at every height below SWL; only water above SWL is the face.'''

    positions = crestWithFace(lambda z: np.full_like(z, 1.0))

    assert np.max(positions[:, 0]) > 2.9
    assert frontFaceLean(positions, 0.0, swl, s, reach = 2.0) <= 0.0

def testAFaceRunningPastTheWindowIsNotLean():

    '''

    A long gentle front runs past the reach in its lower bands. Jitter on the particles cut by the
    window edge must not read as lean.

    '''

    rng = np.random.default_rng(5)
    positions = crestWithFace(lambda z: 4.0 - 8.0 * (z - swl))
    positions += rng.uniform(-0.3 * s, 0.3 * s, size = positions.shape)

    assert frontFaceLean(positions, 0.0, swl, s, reach = 1.0) < 0.5 * s

def testRowsLiftedUnderTheCrestAreNotLean():

    '''

    Under a wave every particle row rises most beneath the crest, so deeper rows climb into the
    bands above still water near the crest. Built here as a lattice stretched by z (1 + eta / d)
    under a steep sech^2 crest, its face sloping back at up to 43 degrees, with particle jitter. A
    face taken as the most shoreward particle per band reads up to 1.4 spacings of lean on these
    arrangements.

    '''

    for seed in range(20):
        positions = latticePoints([-3.0, 0.0], [4.0, swl], s)
        positions[:, 1] *= 1.0 + 0.3 / np.cosh(4.0 * positions[:, 0])**2 / swl
        positions += np.random.default_rng(seed).uniform(-0.3 * s, 0.3 * s, size = positions.shape)

        assert frontFaceLean(positions, 0.0, swl, s, reach = 1.0) < s

def testTooFewBandsGiveMinusInfinity():

    assert frontFaceLean(column(-1.0, 3.0, 0.0, swl), 0.0, swl, s, reach = 2.0) == -np.inf

def testTheProfileIsTheTopOfTheWater():

    positions = np.vstack([column(0.0, 1.0, 0.0, 0.5), column(1.0, 2.0, 0.0, 0.8)])
    centers, elevation = freeSurfaceProfile(positions, 0.0, 3.0, s)

    assert centers[0] == pytest.approx(0.5 * s)
    assert elevation[centers < 1.0] == pytest.approx(np.full(np.sum(centers < 1.0), 0.5))
    assert elevation[(centers > 1.0) & (centers < 2.0)] == pytest.approx(np.full(20, 0.8))
    assert np.all(np.isnan(elevation[centers > 2.0]))

def testTheWaterlineIsTheTipOfTheSheet():

    beach = Bathymetry([0.0, 10.0], [0.0, 1.0], stillWaterLevel = 0.5)
    sheet = fillRegion(lambda p: (p[:, 1] > beach.elevation(p[:, 0])) & (p[:, 1] < 0.5), [0.0, 0.0], [10.0, 0.6], s)
    droplet = np.array([[8.0, 0.82]])
    tip = waterlineX(np.vstack([sheet, droplet]), beach, wetHeight = 2.0 * s)

    assert tip == pytest.approx(beach.shorelineX(), abs = 2.0 * s / 0.1)
    assert tip < 6.0

def testTheTrackerRecordsTheCrestAndBreaking():

    beach = Bathymetry([-1.0, 3.0, 20.0], [0.0, 0.0, 1.7], stillWaterLevel = swl)
    tracker = BreakingTracker(beach, s, faceReach = 2.0, searchFromX = -0.5)

    vertical = crestWithFace(lambda z: np.full_like(z, 1.0))
    lip = crestWithFace(lambda z: np.where(z < swl + 0.2, 1.0, 1.3))

    tracker.update(0.0, vertical)
    assert tracker.breakingPoint is None

    # A lip seen once, then gone, is jitter: the streak resets
    tracker.update(0.1, lip)
    tracker.update(0.2, vertical)
    for time in (0.3, 0.4):
        tracker.update(time, lip)
    assert tracker.breakingPoint is None

    tracker.update(0.5, lip)
    point = tracker.breakingPoint

    assert point is not None and point.time == 0.3
    assert point.lean == pytest.approx(0.3, abs = 1e-9)
    assert point.crestHeight == pytest.approx(0.4, abs = s)
    assert point.breakingIndex == pytest.approx(point.crestHeight / point.depth)
    assert len(tracker.times) == 6

def testTheKinematicDetectorFiresOnFastParticles():

    positions = np.array([[0.0, 0.5], [0.05, 0.5], [5.0, 0.5]])
    particles = ParticleSystem.assemble(positions, np.zeros((0, 2)), s, 1000.0,
                                        np.array([[2.0, 0.0], [2.0, 0.0], [0.1, 0.0]]))
    detector = BreakingDetector(waveSpeed = 2.0, breakerType = 'plunging')

    events = detector.detectBreaking(particles, 1.0)

    assert len(events) == 1 and events[0].nParticles == 2
    assert events[0].intensity == pytest.approx(1.0)
    assert detector.detectBreaking(particles, 1.1) == []
    assert detector.getStatistics()['breakerType'] == 'plunging'

#----------------------------------------------------------------------#
# -- Surface Profile Measures -- #
#----------------------------------------------------------------------#

def testTheProfileCrestFindsASech2CrestBetweenGridPoints():

    x = np.linspace(0.0, 10.0, 201)
    eta = 0.3 / np.cosh(0.5 * (x - 4.013))**2

    assert profileCrest(x, eta) == pytest.approx(4.013, abs = 0.002)

def testTheProfileCrestOfABoreIsItsHighestPoint():

    '''A plateau rising toward a cliff is no parabola; the vertex would land past the cliff.'''

    x = np.linspace(0.0, 10.0, 201)
    eta = np.where(x < 6.0, 0.2 + 0.02 * x, 0.0)

    assert profileCrest(x, eta) == pytest.approx(x[np.argmax(eta)])

def testTheWaveFrontRunsFromTheCrestToTheTroughAhead():

    x = np.linspace(0.0, 10.0, 501)
    surface = 1.0 + 0.3 / np.cosh(1.5 * (x - 4.0))**2 - 0.05 * np.exp(-((x - 7.0) / 0.3)**2)
    front = waveFront(x, surface, np.ones_like(x, dtype = bool))
    steepestSlope = 0.3 * 1.5 * 4.0 / (3.0 * np.sqrt(3.0))

    assert x[front.crestIndex] == pytest.approx(4.0, abs = 0.02)
    # The trough is the foot of the front, within 2 percent of the drop of the dip's bottom
    assert x[front.troughIndex] == pytest.approx(6.9, abs = 0.05)
    assert front.slope == pytest.approx(steepestSlope, rel = 0.01)
    assert front.angle == pytest.approx(np.degrees(np.arctan(steepestSlope)), rel = 0.01)

def testAFlatProfileHasNoFront():

    x = np.linspace(0.0, 1.0, 11)

    assert waveFront(x, np.ones_like(x), np.ones_like(x, dtype = bool)) is None
    assert waveFront(x, np.ones_like(x), np.zeros_like(x, dtype = bool)) is None

def testTheSurfaceTrackerRecordsTheBreakingPointOnce():

    beach = Bathymetry([0.0, 10.0, 30.0], [0.0, 0.0, 2.0], stillWaterLevel = 1.0)
    tracker = SurfaceBreakingTracker(beach, searchFromX = 5.0)
    x = np.linspace(0.0, 30.0, 601)
    bed = beach.elevation(x)

    def update(time, crest, breaking):
        surface = np.maximum(1.0 + 0.3 / np.cosh(0.6 * (x - crest))**2, bed)
        tracker.update(time, x, surface, surface > bed + 1e-9, breaking)

    update(0.0, 3.0, True)
    assert tracker.breakingPoint is None
    update(1.0, 15.0, True)
    update(2.0, 16.0, True)
    point = tracker.breakingPoint

    assert point.time == 1.0
    assert point.crestX == pytest.approx(15.0, abs = 0.01)
    assert point.depth == pytest.approx(0.5, abs = 0.01)
    assert point.breakingIndex == pytest.approx(point.crestHeight / point.depth)
    assert tracker.waterline[0] == pytest.approx(20.0, abs = 0.1)
    assert tracker.maxRunup == pytest.approx(0.0, abs = 0.01)

def testARippleOnTheFaceDoesNotEndTheFront():

    '''A 1 cm ripple partway down a 0.4 m bore face is part of the front, not its trough.'''

    x = np.linspace(0.0, 10.0, 1001)
    face = 1.0 + 0.4 * np.clip((6.0 - x) / 0.5, 0.0, 1.0)
    surface = face + 0.01 * np.exp(-((x - 5.75) / 0.03)**2)
    front = waveFront(x, surface, np.ones_like(x, dtype = bool))

    assert x[front.troughIndex] == pytest.approx(5.99, abs = 0.011)
    assert surface[front.troughIndex] == pytest.approx(1.0, abs = 0.008)
