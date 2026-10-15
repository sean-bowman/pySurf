
# -- pySurf: Figure Tests -- #

'''

Checks on pysurf.figures and pysurf.palette.

The display switch is checked against the environment. Every board figure is built from a real
board through both backends without a window opening, and an extractor handed a board that has not
reached its stage returns None rather than raising. Board profiles come from the outline and
rocker generators directly, so these tests do not depend on the cross-section step.

The palette is checked against its own contract: nineteen named colors in both modes, colormaps
built only from palette colors, and the categorical order the palette defines.

Sean Bowman [10/10/2026]

'''

import os

import matplotlib.colors as mcolors
import numpy as np
import pytest

from pysurf import figures, palette
from pysurf.errors import GeometryError
from pysurf.Surfboard import Surfboard

def profiledBoard() -> Surfboard:

    '''A default board with its outline and rocker profiles generated.'''

    board = Surfboard()
    board.resolution = 200
    xBottom, rBottom, cpBottom = board._generateRockerProfile('bottom')
    xTop, rTop, cpTop = board._generateRockerProfile('top')
    xOutline, rOutline, cpOutline = board._generateNoseToTailProfile()
    board.geometryData = {
        'xRockerBottom': xBottom, 'rRockerBottom': rBottom, 'cpRockerBottom': cpBottom,
        'xRockerTop': xTop, 'rRockerTop': rTop, 'cpRockerTop': cpTop,
        'xBoardOutline': xOutline, 'rBoardOutline': rOutline, 'cpOutline': cpOutline,
    }
    return board

def withCrossSections(board: Surfboard, nLateral: int = 21) -> Surfboard:

    '''Add simple elliptical cross sections in the layout the cross-section step produces.'''

    data = board.geometryData
    eta = np.linspace(0.0, 1.0, nLateral)
    x = data['xRockerTop']
    halfWidth = np.interp(x, data['xBoardOutline'], data['rBoardOutline'])
    middle = 0.5 * (data['rRockerTop'] + data['rRockerBottom'])
    halfThickness = 0.5 * (data['rRockerTop'] - data['rRockerBottom'])
    data['deckX'] = np.tile(x[:, None], (1, nLateral))
    data['bottomX'] = data['deckX'].copy()
    data['deckY'] = halfWidth[:, None] * eta[None, :]
    data['bottomY'] = data['deckY'].copy()
    data['deckZ'] = middle[:, None] + halfThickness[:, None] * np.sqrt(1.0 - eta[None, :]**2)
    data['bottomZ'] = middle[:, None] - halfThickness[:, None] * np.sqrt(1.0 - eta[None, :]**2)
    return board

#----------------------------------------------------------------------#
# -- The Display Switch -- #
#----------------------------------------------------------------------#

@pytest.mark.parametrize('value', ['1', 'true', 'True', 'on', 'yes'])
def testHeadlessIsOnForAnythingMeaningful(value, monkeypatch):

    monkeypatch.setenv('PYSURF_HEADLESS', value)

    assert figures.headlessPlots() is True

@pytest.mark.parametrize('value', ['', '0', 'false', 'False', 'off', '  '])
def testHeadlessIsOffForTheValuesThatMeanOff(value, monkeypatch):

    monkeypatch.setenv('PYSURF_HEADLESS', value)

    assert figures.headlessPlots() is False

def testShowFigureDoesNothingWhenHeadless(monkeypatch):

    class Recorder:
        shown = False
        def show(self):
            Recorder.shown = True

    monkeypatch.setenv('PYSURF_HEADLESS', '1')
    figures.showFigure(Recorder())

    assert Recorder.shown is False

#----------------------------------------------------------------------#
# -- Extractors -- #
#----------------------------------------------------------------------#

def testTheProfileDescriptionCarriesTheBoardInFeet():

    data = figures.boardProfileFigure(profiledBoard())

    assert data.totalLength == pytest.approx(6.0)
    assert data.maxWidth == pytest.approx(20.0 / 12.0)
    assert data.midpointThickness == pytest.approx(2.5 / 12.0)
    assert 2.0 * data.widePointHalfWidth == pytest.approx(data.maxWidth, rel = 0.05)
    assert data.noseRocker > data.tailRocker > 0.0
    # Two cubics per section, two sections per profile; two rockers
    assert len(data.outlineControlPoints) == 4
    assert len(data.rockerControlPoints) == 8

def testTheReferenceImagesAreFoundAndPlacedInFeet():

    '''The thruster photographs live in assets/referenceMaterial beside src/.'''

    data = figures.boardProfileFigure(profiledBoard())

    assert set(data.referenceImages) == {'top', 'side', 'bottom'}
    assert all(os.path.exists(path) for path in data.referenceImages.values())
    left, right, bottom, top = data.referenceExtents['top']
    assert (left, right) == pytest.approx((0.0, 6.0))
    assert top - bottom == pytest.approx(20.0 / 12.0)

def testExtractorsReturnNoneBeforeTheirStageHasRun():

    board = Surfboard()

    assert figures.boardProfileFigure(board) is None
    assert figures.crossSectionFigure(profiledBoard()) is None
    assert figures.boardSurfaceFigure(profiledBoard()) is None

def testEveryRendererHandedNoneReturnsNone():

    assert figures.plotlyBoardProfiles(None) is None
    assert figures.plotlyCrossSections(None) is None
    assert figures.drawBoardProfileFigure(Surfboard()) is None
    assert figures.drawCrossSectionFigure(profiledBoard()) is None

def testCrossSectionsSitOnTheirOwnFloor():

    data = figures.crossSectionFigure(withCrossSections(profiledBoard()), nStations = 5)

    assert len(data.stations) == 5
    assert np.all(np.diff(data.stations) > 0.0)
    assert 0.0 < data.stations[0] and data.stations[-1] < 6.0
    for deckZ, bottomZ in zip(data.deckZ, data.bottomZ):
        assert min(deckZ.min(), bottomZ.min()) == pytest.approx(0.0, abs = 1e-12)

#----------------------------------------------------------------------#
# -- Renderers -- #
#----------------------------------------------------------------------#

def testTheMatplotlibProfileFigureHasThreeViews():

    figure = figures.drawBoardProfileFigure(profiledBoard())

    assert [axes.get_ylabel() for axes in figure.axes] == ['Half-width [ft]', 'Height [ft]', 'Half-width [ft]']
    assert all(axes.images for axes in figure.axes)

def testThePlotlyProfileFigureHasThreeTrueScaleViewsWithReferences():

    figure = figures.plotlyBoardProfiles(figures.boardProfileFigure(profiledBoard()))

    assert len(figure.layout.images) == 3
    assert figure.layout.yaxis.scaleanchor == 'x'
    assert figure.layout.yaxis2.scaleanchor == 'x2'
    assert figure.layout.yaxis3.scaleanchor == 'x3'
    assert figure.layout.paper_bgcolor == palette.colors()['bg']

def testTheProfileCalloutsQuoteTheBoardInInches():

    figure = figures.plotlyBoardProfiles(figures.boardProfileFigure(profiledBoard()))
    texts = [annotation.text for annotation in figure.layout.annotations]

    assert any(text.startswith('wide point') and text.endswith(' in') for text in texts)
    assert '6\'0" x 20.00" x 2.50"' in figure.layout.title.text

def testCrossSectionsDrawInBothBackends():

    board = withCrossSections(profiledBoard())

    assert len(figures.drawCrossSectionFigure(board).axes[0].lines) == 4 * 5
    assert len(figures.plotlyCrossSections(figures.crossSectionFigure(board)).data) == 2 * 5

def testTheSurfaceFigureMirrorsDeckAndBottom():

    figure = figures.boardSurfaceFigure(withCrossSections(profiledBoard()))

    assert len(figure.data) == 4

def testPlotGeometryRunsHeadless():

    withCrossSections(profiledBoard()).plotGeometry(show3D = True)

def testPlotGeometryRefusesABoardWithoutProfiles():

    with pytest.raises(GeometryError):
        Surfboard().plotGeometry()

def testExportWritesEveryFigureThatHasData(tmp_path):

    written = figures.exportInteractiveFigures(withCrossSections(profiledBoard()), str(tmp_path))

    assert sorted(os.path.basename(path) for path in written) == sorted(figures.interactiveFigureBuilders)
    assert all(os.path.getsize(path) > 0 for path in written)

def testExportSkipsFiguresWithoutData(tmp_path):

    written = figures.exportInteractiveFigures(profiledBoard(), str(tmp_path))

    assert [os.path.basename(path) for path in written] == ['boardProfilesInteractive.html']

#----------------------------------------------------------------------#
# -- Palette -- #
#----------------------------------------------------------------------#

def testLightIsTheDefaultMode():

    assert palette.figureMode() == 'light'
    assert palette.colors()['bg'] == '#F7F1E6'
    assert palette.colors()['accent'] == '#4FBDB3'

@pytest.mark.parametrize('mode', palette.modes)
def testEveryModeNamesTheSameNineteenColors(mode):

    assert set(palette.colors(mode)) == set(palette.colors('light'))
    assert len(palette.colors(mode)) == 19

@pytest.mark.parametrize('mode', palette.modes)
def testEveryColormapStopIsAPaletteColor(mode):

    named = set(palette.colors(mode).values())
    for name, stops in palette.colormapStops[mode].items():
        assert set(stops) <= named, name

@pytest.mark.parametrize('mode', palette.modes)
def testTheSequentialRampsRunMonotonicallyInLightness(mode):

    '''A magnitude has to read as one direction, so lightness must not double back along the ramp.'''

    def lightness(color):
        red, green, blue = mcolors.to_rgb(color)
        return 0.2126 * red + 0.7152 * green + 0.0722 * blue

    for name in ('ocean', 'sunset'):
        values = np.array([lightness(stop) for stop in palette.colormapStops[mode][name]])
        steps = np.diff(values)
        assert np.all(steps > 0.0) or np.all(steps < 0.0), name

def testTheCategoricalOrderStartsOnGreen():

    named = palette.colors()

    assert palette.categories() == [named[key] for key in ('green', 'accent', 'red', 'yellow', 'blue',
                                                            'orange', 'purple', 'cyan')]

def testTheFigureModeSwitchesAndRefusesAnUnknownMode():

    original = palette.figureMode()
    try:
        palette.setFigureMode('dark')
        assert palette.colors() == palette.palettes['dark']
        assert figures.plotlyCrossSections(figures.crossSectionFigure(withCrossSections(profiledBoard()))) \
            .layout.paper_bgcolor == '#16232A'
        with pytest.raises(ValueError):
            palette.setFigureMode('sepia')
    finally:
        palette.setFigureMode(original)
