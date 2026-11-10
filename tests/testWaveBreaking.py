
# -- pySurf: Wave Breaking Measure Tests -- #

'''

Checks of pysurf.waveBreaking on constructed particle arrangements whose answer is known by
construction: a face sloping back has negative lean, a vertical face none, an overhanging lip a
lean equal to its overhang; still water ahead of the wave, a face running past the search window and
particle rows lifted under the crest do not count as lean; the tracker records breaking only once
the lean persists; the surface profile and waterline match the arrangement they were built
from; and the smoothed fluid-fraction surface stands at the top of the water's cells and sorts its
closed contours into pockets of air and drops of water; the face read from that surface is
level, vertical or overhanging as built, in deep or shallow water, whatever voids lie under the
crest; the jet's touchdown is the tube closing under a lip, not a void opening in the crest nor a
pocket under a face that has sloped back; and the breaking point read from that surface is the start
of the face's last vertical run before touchdown, or of its longest without one.

Sean Bowman [10/11/2026]

'''

import numpy as np
import pytest

from pysurf.bathymetry import Bathymetry
from pysurf.sphParticles import ParticleSystem, fillRegion, latticePoints
from pysurf.sphRefinement import splitParticles
from pysurf.waveBreaking import (BreakingDetector, BreakingTracker, ParticleSurface, SurfaceBreakingTracker,
                                 airShare, contourFace, freeSurfaceProfile, frontFaceLean, particleSurface, profileCrest,
                                 waterlineX, waveFront)

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

def testTheParticleSurfaceSortsTheSurfacePocketsAndDrops():

    '''A block of water with a hole in it and a drop above it, cut by a window whose lower edge lies in the water.'''

    water = column(0.0, 1.0, 0.0, 0.5)
    # The hole empties the cells of 0.3 < x < 0.7 and 0.15 < z < 0.35; the drop fills 0.4 < x < 0.55, 0.7 < z < 0.85
    hole = (np.abs(water[:, 0] - 0.5) < 0.2 - 1e-9) & (np.abs(water[:, 1] - 0.25) < 0.1 - 1e-9)
    drop = column(0.4, 0.55, 0.7, 0.85)
    result = particleSurface(np.vstack([water[~hole], drop]), s, np.array([-0.2, 0.1]), np.array([1.2, 1.0]))

    # The surface runs up the left side, across the top and down the right, starting at its left end
    assert result.surface[0, 0] < result.surface[-1, 0]
    top = result.surface[np.abs(result.surface[:, 0] - 0.5) < 0.3]
    assert np.allclose(top[:, 1], 0.5, atol = 0.05 * s)
    # Each contour stands on the edges of the cells; it rounds their corners
    assert len(result.pockets) == 1 and len(result.drops) == 1
    pocket, droplet = result.pockets[0], result.drops[0]
    assert np.allclose([pocket[:, 0].min(), pocket[:, 0].max(), pocket[:, 1].min(), pocket[:, 1].max()],
                       [0.3, 0.7, 0.15, 0.35], atol = 0.05 * s)
    assert ParticleSurface.area(pocket) == pytest.approx(0.4 * 0.2, rel = 0.1)
    assert ParticleSurface.area(droplet) == pytest.approx(0.15 * 0.15, rel = 0.25)

flatBed = Bathymetry([-3.0, 4.0], [0.0, 0.0], stillWaterLevel = swl)

@pytest.mark.parametrize('face, slope', [(lambda z: 1.0 - 0.5 * (z - swl), 0.5 / np.sqrt(1.25)),
                                         (lambda z: np.full_like(z, 1.0), 0.0),
                                         (lambda z: np.where(z < swl + 0.2, 1.0, 1.3), -1.0)])
def testTheContourFaceReadsTheSlopeOfTheFace(face, slope):

    '''A face sloping back at dx/dz = -0.5, a vertical face, and a lip: dx/ds 0.447, 0 and -1.'''

    result = contourFace(crestWithFace(face), s, 0.5, flatBed, reach = 1.0, windowDepth = 0.1)

    assert result.crestHeight == pytest.approx(0.4, abs = 0.05 * s)
    assert result.slope == pytest.approx(slope, abs = 0.02)
    # The face built at x = 1 - 0.5 (z - SWL), 1 and 1 to 1.3 for the lip, between z = SWL and SWL + 0.4
    assert 0.75 < result.faceX < 1.35

def testTheContourFaceReadsAFaceInShallowWater():

    '''

    A vertical face over water 0.3 deep, less than the window depth of 0.2 and three spacings: the
    window's lower edge rises to 0.075 below still water, and the face reads vertical.

    '''

    shallowBed = Bathymetry([-3.0, 4.0], [0.7, 0.7], stillWaterLevel = swl)
    water = column(-1.0, 3.0, 0.7, swl)
    crest = fillRegion(lambda p: (p[:, 0] > 0.0) & (p[:, 0] < 1.0), [-1.0, swl], [3.0, swl + 0.4], s)
    result = contourFace(np.vstack([water, crest]), s, 0.5, shallowBed, reach = 1.0, windowDepth = 0.2)

    assert result.window is not None
    assert result.crestHeight == pytest.approx(0.4, abs = 0.05 * s)
    assert result.slope == pytest.approx(0.0, abs = 0.02)
    # Water 0.24 deep leaves a window 0.045 deep, less than a spacing: there is none
    assert contourFace(np.vstack([water, crest]), s, 0.5, Bathymetry([-3.0, 4.0], [0.76, 0.76], stillWaterLevel = swl),
                       reach = 1.0, windowDepth = 0.2).window is None

def testAVoidUnderTheCrestIsNotAFace():

    '''

    Two rows emptied under a crest whose face slopes back, as the tensile voids of a scheme without
    shifting open there: the band inside the void ends its run at the void and reads as lean, while
    the smoothed surface keeps the void as a pocket apart from the surface.

    '''

    positions = crestWithFace(lambda z: 1.0 - 0.5 * (z - swl))
    void = (positions[:, 1] > swl + 0.1) & (positions[:, 1] < swl + 0.2) & (positions[:, 0] > 0.1) & (positions[:, 0] < 0.6)
    positions = positions[~void]

    assert frontFaceLean(positions, 0.0, swl, s, reach = 1.0) > 10.0 * s
    assert contourFace(positions, s, 0.5, flatBed, reach = 1.0, windowDepth = 0.1).slope == pytest.approx(0.447, abs = 0.02)
    assert len(particleSurface(positions, s, np.array([-0.5, 0.9]), np.array([2.0, 2.0])).pockets) == 1

def testTheTrackerRecordsTouchdownWhenTheTubeCloses():

    '''

    A lip over open air is not a touchdown; the same lip with its tip joined to the water below
    closes a 0.25 by 0.2 pocket, and the tracker keeps the open surface before it.

    '''

    lip = crestWithFace(lambda z: np.where(z < swl + 0.2, 1.0, 1.3))
    tip = column(1.25, 1.3, swl, swl + 0.2)
    tracker = BreakingTracker(flatBed, s, faceReach = 1.0, faceMeasure = 'contour')

    tracker.update(1.0, lip)
    assert tracker.touchdownSurface is None
    tracker.update(2.0, np.vstack([lip, tip]))

    assert tracker.touchdownTime == pytest.approx(1.5)
    # The kept surface overhangs: x runs back along the lip's underside
    assert np.any(np.diff(tracker.touchdownSurface[:, 0]) < -0.5 * s)
    assert ParticleSurface.area(tracker.touchdownPocket) == pytest.approx(0.25 * 0.2, rel = 0.15)

def testAVoidInTheWaterIsNotATouchdown():

    '''

    A 0.4 by 0.1 void opening inside the crest of the lip is water the last surface held, so it is
    not a touchdown; the tube closing later is, and the tracker keeps the tube, not the void, as the
    pocket.

    '''

    lip = crestWithFace(lambda z: np.where(z < swl + 0.2, 1.0, 1.3))
    inVoid = (lip[:, 0] > 0.1) & (lip[:, 0] < 0.5) & (lip[:, 1] > swl + 0.25) & (lip[:, 1] < swl + 0.35)
    voided = lip[~inVoid]
    tip = column(1.25, 1.3, swl, swl + 0.2)
    tracker = BreakingTracker(flatBed, s, faceReach = 1.0, faceMeasure = 'contour')

    tracker.update(1.0, lip)
    tracker.update(2.0, voided)
    assert tracker.touchdownSurface is None
    window = particleSurface(voided, s, np.array([-0.5, 0.9]), np.array([2.0, 2.0]))
    assert len(window.pockets) == 1 and airShare(window.pockets[0], tracker._lastSurface, s) == 0.0

    tracker.update(3.0, np.vstack([voided, tip]))
    assert tracker.touchdownTime == pytest.approx(2.5)
    # The tube lies between the face at x = 1 and the tip at 1.25
    assert 1.0 < tracker.touchdownPocket[:, 0].mean() < 1.25

def testTheBreakingPointIsTheLastVerticalRunBeforeTouchdown():

    '''

    On the smoothed surface, the tube closing under the lip fixes the breaking point at the start of
    the vertical run in progress, though an earlier run that sloped back lasted longer; a face
    sloping back and standing again afterwards does not move it.

    '''

    sloping = crestWithFace(lambda z: 1.0 - 0.5 * (z - swl))
    lip = crestWithFace(lambda z: np.where(z < swl + 0.2, 1.0, 1.3))
    tip = column(1.25, 1.3, swl, swl + 0.2)
    tracker = BreakingTracker(flatBed, s, faceReach = 1.0, faceMeasure = 'contour')

    tracker.update(0.0, sloping)
    assert tracker.breakingPoint is None
    for time in (1.0, 2.0, 3.0):
        tracker.update(time, lip)
    tracker.update(4.0, sloping)
    tracker.update(5.0, lip)
    assert tracker.breakingPoint.time == 1.0 and not tracker.breakingFinal
    tracker.update(6.0, np.vstack([lip, tip]))
    tracker.update(7.0, sloping)
    tracker.update(8.0, lip)

    assert tracker.breakingFinal and tracker.touchdownTime == pytest.approx(5.5)
    assert tracker.breakingPoint.time == 5.0

def testWithoutTouchdownTheBreakingPointIsTheLongestVerticalRun():

    '''

    With no tube closing, the breaking point is the start of the longest vertical run, not the
    runs before it or the one after it.

    '''

    sloping = crestWithFace(lambda z: 1.0 - 0.5 * (z - swl))
    lip = crestWithFace(lambda z: np.where(z < swl + 0.2, 1.0, 1.3))
    tracker = BreakingTracker(flatBed, s, faceReach = 1.0, faceMeasure = 'contour')

    for time, face in enumerate([lip, sloping, lip, lip, lip, sloping, lip, sloping]):
        tracker.update(float(time), face)

    assert tracker.breakingPoint.time == 2.0 and not tracker.breakingFinal

def testATubeUnderAFaceThatSlopedBackIsNotATouchdown():

    '''A pocket closing over air, an update after the face read as sloping back, does not land a jet.'''

    sloping = crestWithFace(lambda z: 1.0 - 0.5 * (z - swl))
    lip = crestWithFace(lambda z: np.where(z < swl + 0.2, 1.0, 1.3))
    tip = column(1.25, 1.3, swl, swl + 0.2)
    tracker = BreakingTracker(flatBed, s, faceReach = 1.0, faceMeasure = 'contour')

    tracker.update(0.0, lip)
    tracker.update(1.0, sloping)
    tracker.update(2.0, np.vstack([lip, tip]))

    assert tracker.touchdownSurface is None and not tracker.breakingFinal

def testSplitParticlesCountTheirOwnAreaInTheSurface():

    # A still block 0.5 m deep with its right half split: weighted by area the surface stays at the
    # top of the water; counted as whole cells the split half reads four times as full and rises
    s = 0.02
    particles = ParticleSystem.assemble(latticePoints(np.zeros(2), np.array([1.0, 0.5]), s), np.empty((0, 2)), s, 1000.0)
    splitParticles(particles, particles.positions[:, 0] > 0.5, 1000.0, 1.3 * s)
    lower, upper = np.array([0.1, 0.3]), np.array([0.9, 0.7])
    weighted = particleSurface(particles.positions, s, lower, upper, volumes = particles.masses / 1000.0).surface
    counted = particleSurface(particles.positions, s, lower, upper).surface

    assert np.abs(weighted[:, 1] - 0.5).max() < 0.1 * s
    assert counted[counted[:, 0] > 0.6, 1].min() > 0.5 + 0.5 * s

