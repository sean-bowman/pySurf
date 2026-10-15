
# -- Surfboard Cross Sections -- #

'''

The cross sections `Surfboard.generateGeometry` builds: their layout, how they meet the planform
and the rockers, the checks on their settings, how closely they reproduce the sections of the
reference thruster STL the exponents were fitted to, and the closed surface `exportGeometry`
writes from them.

The STL runs along +z from the tail (low z) to the nose, with its width along x and its deck
toward +y; it carries three fins, which cut as loops of their own. Each of its sections is scaled
to its half-width and its centerline thickness, the board's to theirs, and compared by the
distance from each STL point to the board's contour, in units of the local thickness. That is a
check of the calibration, not a validation: the exponents came from this STL. The tolerances
guard the fit, 1 percent rms and 2.5 percent at worst from 0.2 to 0.9 of the length, 2.5 and 4.5
percent toward the tips, where the sections are smallest and the STL's tail carries a slight deck
concave the two-exponent contour does not have.

Sean Bowman [10/15/2026]

'''

import functools
from pathlib import Path

import numpy as np
import pytest

from pysurf.errors import GeometryError, InvalidInputError
from pysurf.Surfboard import Surfboard

referenceStl = Path(__file__).resolve().parents[1] / 'assets' / 'referenceMaterial' / 'referenceThruster.stl'

@functools.lru_cache(maxsize = None)
def generatedBoard() -> Surfboard:

    '''The default board with its geometry generated.'''

    board = Surfboard()
    board.generateGeometry()
    return board

@functools.lru_cache(maxsize = None)
def referenceMesh():

    '''The reference thruster as a trimesh mesh.'''

    trimesh = pytest.importorskip('trimesh')
    return trimesh.load(str(referenceStl))

def polygonArea(points: np.ndarray) -> float:

    '''Area enclosed by a closed polygon, by the shoelace formula.'''

    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def referenceSection(fraction: float) -> tuple[np.ndarray, np.ndarray, float]:

    '''

    The STL's section at a fraction of its length, as lateral position over the half-width and
    height above the bottom centerline over the centerline thickness, with the half-width over
    the thickness.

    '''

    mesh = referenceMesh()
    low, high = mesh.bounds[0, 2], mesh.bounds[1, 2]
    section = mesh.section(plane_origin = [0.0, 0.0, low + fraction * (high - low)], plane_normal = [0.0, 0.0, 1.0])
    loop = max(section.discrete, key = polygonArea)

    # Resample the loop evenly along its length, so every stretch of contour counts alike
    closed = np.vstack([loop, loop[:1]])
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(closed, axis = 0).T))])
    samples = np.linspace(0.0, arc[-1], 1500, endpoint = False)
    x, y = np.interp(samples, arc, closed[:, 0]), np.interp(samples, arc, closed[:, 1])

    center, halfWidth = 0.5 * (x.max() + x.min()), 0.5 * (x.max() - x.min())
    eta = np.abs(x - center) / halfWidth
    middle = eta < 0.02
    deck, bottom = y[middle].max(), y[middle].min()

    return eta, (y - bottom) / (deck - bottom), halfWidth / (deck - bottom)

def sectionDistances(board: Surfboard, fraction: float) -> np.ndarray:

    '''Distance from each STL point to the board's scaled contour at the same station [local thickness].'''

    eta, height, aspect = referenceSection(fraction)
    lateral = np.concatenate([np.linspace(0.0, 1.0, 2000), 1.0 - np.logspace(-7.0, -1.0, 300)])
    deck, bottom = board.crossSectionShape(fraction, lateral)
    onDeck = height > board.railApexFrac

    distances = np.empty_like(eta)
    for mask, contour in ((onDeck, deck), (~onDeck, bottom)):
        gaps = np.hypot((eta[mask, None] - lateral[None, :]) * aspect, height[mask, None] - contour[None, :])
        distances[mask] = gaps.min(axis = 1)

    return distances

#----------------------------------------------------------------------#
# -- Layout -- #
#----------------------------------------------------------------------#

def testEveryStationHasADeckAndABottomContour():

    board = generatedBoard()
    data = board.geometryData
    shape = (board.resolution, board.numCrossSectionPoints)

    for key in ('deckX', 'deckY', 'deckZ', 'bottomX', 'bottomY', 'bottomZ'):
        assert data[key].shape == shape and np.isfinite(data[key]).all(), key

def testTheSectionsMeetTheRockersAndThePlanform():

    data = generatedBoard().geometryData
    halfWidth = np.interp(data['xRockerTop'], data['xBoardOutline'], data['rBoardOutline'])
    railHeight = data['rRockerBottom'] + 0.5 * (data['rRockerTop'] - data['rRockerBottom'])

    assert data['deckZ'][:, 0] == pytest.approx(data['rRockerTop'], abs = 1e-12)
    assert data['bottomZ'][:, 0] == pytest.approx(data['rRockerBottom'], abs = 1e-12)
    assert data['deckY'][:, -1] == pytest.approx(halfWidth, abs = 1e-12)
    assert data['deckZ'][:, -1] == pytest.approx(railHeight, abs = 1e-12)
    assert data['bottomZ'][:, -1] == pytest.approx(railHeight, abs = 1e-12)

def testTheSectionsStayInsideTheSideView():

    data = generatedBoard().geometryData

    assert (data['deckZ'] <= data['rRockerTop'][:, None] + 1e-12).all()
    assert (data['bottomZ'] >= data['rRockerBottom'][:, None] - 1e-12).all()
    assert (np.diff(data['deckZ'], axis = 1) <= 1e-12).all()
    assert (np.diff(data['bottomZ'], axis = 1) >= -1e-12).all()

def testAnExponentOfTwoGivesAnEllipse():

    board = Surfboard()
    board.deckFlatness = [2.0] * len(board.sectionStations)
    board.bottomFlatness = [2.0] * len(board.sectionStations)
    eta = np.linspace(0.0, 1.0, 11)
    deck, bottom = board.crossSectionShape(0.5, eta)

    assert deck == pytest.approx(0.5 + 0.5 * np.sqrt(1.0 - eta**2))
    assert bottom == pytest.approx(0.5 - 0.5 * np.sqrt(1.0 - eta**2))

#----------------------------------------------------------------------#
# -- Settings -- #
#----------------------------------------------------------------------#

@pytest.mark.parametrize('field, value', [
    ('railApexFrac', 1.2), ('deckFlatness', [0.5] * 13), ('bottomFlatness', [4.0] * 12),
    ('sectionStations', [0.1, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]),
])
def testInvalidSectionSettingsAreRefusedByName(field, value):

    board = Surfboard()
    setattr(board, field, value)

    with pytest.raises(InvalidInputError) as raised:
        board._validateInputs()

    assert raised.value.getContext()['parameterName'] == field

def testABoardWithNoThicknessIsRefused():

    board = Surfboard()
    board.noseTopRockerFt = 0.2

    with pytest.raises(GeometryError):
        board.generateGeometry()

#----------------------------------------------------------------------#
# -- Export -- #
#----------------------------------------------------------------------#

@pytest.mark.parametrize('tailShape', ['round', 'squash'])
def testTheExportIsAClosedOutwardSurfaceOfTheBoardsVolume(tailShape, tmp_path):

    # A round tail closes to a line; a squash tail and the nose tip take caps
    trimesh = pytest.importorskip('trimesh')
    board = Surfboard(tailShape = tailShape)
    board.resolution = 200
    board.numCrossSectionPoints = 25
    board.generateGeometry()
    mesh = trimesh.load(board.exportGeometry(str(tmp_path / 'board')))

    data = board.geometryData
    area = 2.0 * np.trapezoid(data['deckZ'] - data['bottomZ'], data['deckY'], axis = 1)
    volume = np.trapezoid(area, data['deckX'][:, 0]) * 12.0**3

    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.volume == pytest.approx(volume, rel = 0.005)
    assert mesh.extents[0] == pytest.approx(12.0 * board.totalLength)
    # The file holds float32, and the widest point of the outline falls between stations
    assert mesh.extents[1] == pytest.approx(2.0 * 12.0 * data['rBoardOutline'].max(), rel = 1e-4)

def testABoardWithoutSectionsCannotBeExported(tmp_path):

    with pytest.raises(GeometryError):
        Surfboard().exportGeometry(str(tmp_path / 'board.stl'))

#----------------------------------------------------------------------#
# -- Calibration Against the Reference STL -- #
#----------------------------------------------------------------------#

@pytest.mark.parametrize('fraction', [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
def testTheSectionsReproduceTheReferenceThruster(fraction):

    distances = sectionDistances(Surfboard(), fraction)

    assert np.sqrt(np.mean(distances**2)) < 0.010 and distances.max() < 0.025, \
        (np.sqrt(np.mean(distances**2)), distances.max())

@pytest.mark.parametrize('fraction', [0.05, 0.1, 0.95])
def testTheSectionsNearTheTipsFollowTheReferenceThruster(fraction):

    distances = sectionDistances(Surfboard(), fraction)

    assert np.sqrt(np.mean(distances**2)) < 0.025 and distances.max() < 0.045, \
        (np.sqrt(np.mean(distances**2)), distances.max())
