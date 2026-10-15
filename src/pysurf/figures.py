
# -- pySurf: Figures -- #

'''

Every plotting concern in one place: what a figure shows, and how both backends draw it.

pySurf draws every board figure twice. Matplotlib renders the static figure a design session
looks at and a document embeds. Plotly renders the interactive companion written to HTML, which
pans, zooms and reads values off a curve far better than a static image can. Both are required
dependencies.

Rather than write each figure twice, the extractors pull the arrays and labels off a Surfboard
into a plain dataclass that neither library appears in, and each backend's renderers draw from
that one description. A figure is described once, and the two renderings cannot drift. An
extractor returns None when the stage its figure shows has not run, and every renderer handed
None returns None, so a caller asks for a figure without first checking what the board has built.

    boardProfileFigure  -> plotlyBoardProfiles, drawBoardProfileFigure   planform, rocker, bottom
    crossSectionFigure  -> plotlyCrossSections, drawCrossSectionFigure   station cross sections
    boardSurfaceFigure                                                   3D deck and bottom
    breakingWaveFigure  -> drawBreakingWaveFrame, writeBreakingWaveAnimation,
                           plotlyBreakingWaveAnimation                   a wave breaking on a beach

The 3D surface has no Matplotlib counterpart, so it skips the extractor-then-renderer split and
builds the plotly figure directly off a Surfboard. The modules that generate geometry call in here
rather than building a figure themselves, so what a module computes and what it draws stay in two
different files.

Every display goes through `showFigure`, which does nothing when the environment variable
`PYSURF_HEADLESS` is set. `exportInteractiveFigures` writes the HTML companion of every figure a
board can produce. Colors come from `pysurf.palette`: Surfy Pastels, light by default.

Sean Bowman [10/10/2026]

'''

# Standard library imports
import json
import math
import os
from dataclasses import dataclass, field

# Third-party imports
import numpy as np
import matplotlib

def headlessPlots() -> bool:

    '''

    Whether figures should be built and left alone rather than put in front of somebody.

    A design session draws a lot of figures, and every one of them opens a window or a browser tab
    that has to be closed by hand. Setting the environment variable `PYSURF_HEADLESS` suppresses
    the display of all of them without suppressing the figures themselves: matplotlib windows
    never open, plotly writes its HTML and does not launch a browser, and every file that would
    have been produced is still produced.

    This is a property of where pySurf is running rather than of the board being designed, so it
    belongs to the shell or the harness that started the run, not to a board attribute.

    Returns:
    --------
    bool : True when figures should not be displayed.

    '''

    return os.environ.get('PYSURF_HEADLESS', '').strip().lower() not in ('', '0', 'false', 'off')

# Selected before pyplot is imported, so the choice sticks rather than having to be forced back
# afterwards. This covers matplotlib only; a plotly figure ignores the matplotlib backend and
# opens a browser tab of its own, which is what `showFigure` below is for.
if headlessPlots():
    matplotlib.use('Agg', force = True)

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
from plotly.offline import plot as _writePlotlyFigure

# Local imports
from pysurf import constants, palette as figurePalette
from pysurf.bemSolver import crestIndex
from pysurf.units import convert

# Reference material lives at the repository root, beside src/
repositoryRoot = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
referenceDirectory = os.path.join(repositoryRoot, 'assets', 'referenceMaterial')

# The thruster reference board: three orthographic photographs and the physical dimensions their
# pixel axes are calibrated to
referenceImageFiles = {
    'top':    'thrusterReferenceImage_pointedNose_roundTail_topView.png',
    'side':   'thrusterReferenceImage_pointedNose_roundTail_sideView.png',
    'bottom': 'thrusterReferenceImage_pointedNose_roundTail_bottomView.png',
}
referenceDimensionsFile = 'thrusterReferenceImageDimensions.json'

# Opacity of a reference photograph under the generated curves
referenceAlpha = 0.45

def showFigure(figure) -> None:

    '''

    Put a figure in front of somebody, unless pySurf is running headless.

    Every display of a figure goes through here rather than calling `.show()` directly. A plotly
    figure is the case that matters: it ignores the matplotlib backend entirely, and its browser
    renderer starts a local web server and opens a tab against it, which is why suppressing it is
    a separate problem from selecting a non-interactive backend.

    Parameters:
    -----------
    figure : plotly.graph_objects.Figure | matplotlib.figure.Figure
        Anything with a `show` method.

    '''

    if headlessPlots():
        return

    figure.show()

def _color(name: str) -> str:

    '''

    A color of the mode figures are being drawn in, by its name in `pysurf.palette`. Read at draw
    time rather than bound at import, so a change of mode reaches every figure drawn after it.

    '''

    return figurePalette.colors()[name]

def _rgba(name: str, alpha: float) -> str:

    '''A palette color as a plotly rgba string at the given opacity, for translucent fills.'''

    hexColor = _color(name).lstrip('#')
    red, green, blue = (int(hexColor[i:i + 2], 16) for i in (0, 2, 4))

    return f'rgba({red},{green},{blue},{alpha})'

def _feetInches(lengthFt: float) -> str:

    '''A board length the way it is quoted: 6.0 ft is 6\'0".'''

    feet = int(np.floor(lengthFt + 1e-9))
    inches = convert(lengthFt - feet, 'ft', 'in')

    return f'{feet}\'{inches:.0f}"'

def _dimensionText(data) -> str:

    '''Length x width x thickness the way a board is quoted: 6\'0" x 20.00" x 2.50".'''

    width = convert(data.maxWidth, 'ft', 'in')
    thickness = convert(data.midpointThickness, 'ft', 'in')

    return f'{_feetInches(data.totalLength)} x {width:.2f}" x {thickness:.2f}"'

def _callouts(data) -> dict:

    '''The profile callouts both renderers place, in the inches a board is quoted in.'''

    wideWidth = convert(2.0 * data.widePointHalfWidth, 'ft', 'in')
    noseRocker = convert(data.noseRocker, 'ft', 'in')
    tailRocker = convert(data.tailRocker, 'ft', 'in')

    return {
        'wide': f'wide point {wideWidth:.2f} in',
        'nose': f'nose rocker {noseRocker:.2f} in',
        'tail': f'tail rocker {tailRocker:.2f} in',
    }

#----------------------------------------------------------------------#
# -- Figure Descriptions -- #
#----------------------------------------------------------------------#

@dataclass
class BoardProfileFigure:

    '''

    The three views a shaper works from: the planform from above, the rocker from the side, and
    the planform again from below, each against its reference photograph.

    Control polygons are lists of cubic Bezier control-point dicts (p1 to p4). Reference extents
    are [left, right, bottom, top] in feet, so both renderers place a photograph identically.

    '''

    xOutline: np.ndarray                                    # [ft], tail to nose
    halfWidth: np.ndarray                                   # [ft]
    xDeck: np.ndarray                                       # [ft]
    deckRocker: np.ndarray                                  # [ft]
    xBottom: np.ndarray                                     # [ft]
    bottomRocker: np.ndarray                                # [ft]
    totalLength: float = 0.0                                # [ft]
    maxWidth: float = 0.0                                   # [ft], specified
    midpointThickness: float = 0.0                          # [ft], specified
    widePointX: float = 0.0                                 # [ft], from the tail
    widePointHalfWidth: float = 0.0                         # [ft], generated
    noseRocker: float = 0.0                                 # [ft], bottom at the nose tip
    tailRocker: float = 0.0                                 # [ft], bottom at the tail tip
    outlineControlPoints: list = field(default_factory = list)
    rockerControlPoints: list = field(default_factory = list)
    showControlPoints: bool = True
    referenceImages: dict = field(default_factory = dict)   # view -> image path
    referenceExtents: dict = field(default_factory = dict)  # view -> [left, right, bottom, top], [ft]
    title: str = 'Board profiles'

@dataclass
class CrossSectionFigure:

    '''

    Deck and bottom contours of a few stations along the board. Each contour is the right half of
    the section, shifted so the section's lowest point sits at zero: sections are compared by
    shape the way a shaper lays templates on one baseline, and the rocker that separates them
    vertically is the side view's job.

    '''

    stations: np.ndarray                                    # [ft], from the tail
    deckY: list                                             # [ft], one array per station
    deckZ: list                                             # [ft]
    bottomY: list                                           # [ft]
    bottomZ: list                                           # [ft]
    title: str = 'Board cross sections'

#----------------------------------------------------------------------#
# -- Extractors -- #
#----------------------------------------------------------------------#

def _asArray(value) -> np.ndarray:

    '''

    Coerce a geometry entry to a float array, tolerating placeholders a stage leaves before it
    has run.

    '''

    try:
        return np.asarray(value, dtype = float)
    except (TypeError, ValueError):
        return np.array([])

def _flattenControlPoints(sectionControlPoints: dict) -> list:

    '''

    The control polygons of a profile as one flat list. Each profile section ('tail', 'nose')
    holds a list of cubics, so the result is every cubic of both sections in order.

    '''

    return [cubic for section in ('tail', 'nose') for cubic in sectionControlPoints.get(section, [])]

def _referenceOverlays(totalLength: float, maxWidth: float) -> tuple:

    '''

    The reference photographs that exist, and where each lands in feet.

    The images are calibrated to the dimensions in the JSON beside them. Without the JSON the
    board's own dimensions stand in, so the images still land at roughly the right scale. The top
    view spans the reference length and width; the side and bottom views span the reference length
    and keep their own pixel aspect ratio.

    Returns:
    --------
    tuple : (view -> path, view -> [left, right, bottom, top] in feet)

    '''

    from PIL import Image

    referenceLength, referenceWidth = totalLength, maxWidth
    dimensionsPath = os.path.join(referenceDirectory, referenceDimensionsFile)
    if os.path.exists(dimensionsPath):
        with open(dimensionsPath, 'r') as dimensionsFile:
            referenceDimensions = json.load(dimensionsFile)
        referenceLength = referenceDimensions.get('totalLength', totalLength)
        referenceWidth = convert(referenceDimensions.get('maxWidth', convert(maxWidth, 'ft', 'in')), 'in', 'ft')

    images, extents = {}, {}
    for view, fileName in referenceImageFiles.items():
        imagePath = os.path.join(referenceDirectory, fileName)
        if not os.path.exists(imagePath):
            continue
        with Image.open(imagePath) as image:
            width, height = image.size
        viewHeight = referenceLength * height / width
        images[view] = imagePath
        if view == 'top':
            extents[view] = [0.0, referenceLength, -referenceWidth / 2.0, referenceWidth / 2.0]
        elif view == 'side':
            extents[view] = [0.0, referenceLength, 0.0, viewHeight]
        else:
            extents[view] = [0.0, referenceLength, -viewHeight / 2.0, viewHeight / 2.0]

    return images, extents

def boardProfileFigure(board):

    '''

    Profile views description, or None when the outline and rocker have not been generated.

    '''

    data = board.geometryData
    if data is None:
        return None

    xOutline = _asArray(data.get('xBoardOutline', []))
    halfWidth = _asArray(data.get('rBoardOutline', []))
    xDeck, deck = _asArray(data.get('xRockerTop', [])), _asArray(data.get('rRockerTop', []))
    xBottom, bottom = _asArray(data.get('xRockerBottom', [])), _asArray(data.get('rRockerBottom', []))
    if xOutline.size == 0 or halfWidth.size != xOutline.size or xBottom.size == 0 or xDeck.size == 0:
        return None

    totalLength = float(board.totalLength)                     # [ft]
    maxWidth = convert(board.maxWidth, 'in', 'ft')             # [ft]
    images, extents = _referenceOverlays(totalLength, maxWidth)

    rockerControlPoints = []
    for key in ('cpRockerBottom', 'cpRockerTop'):
        if key in data:
            rockerControlPoints += _flattenControlPoints(data[key])

    wideIndex = int(np.argmax(halfWidth))

    return BoardProfileFigure(
        xOutline             = xOutline,
        halfWidth            = halfWidth,
        xDeck                = xDeck,
        deckRocker           = deck,
        xBottom              = xBottom,
        bottomRocker         = bottom,
        totalLength          = totalLength,
        maxWidth             = maxWidth,
        midpointThickness    = convert(board.midpointThickness, 'in', 'ft'),
        widePointX           = float(xOutline[wideIndex]),
        widePointHalfWidth   = float(halfWidth[wideIndex]),
        noseRocker           = float(bottom[-1]),
        tailRocker           = float(bottom[0]),
        outlineControlPoints = _flattenControlPoints(data['cpOutline']) if 'cpOutline' in data else [],
        rockerControlPoints  = rockerControlPoints,
        showControlPoints    = bool(board.showControlPoints),
        referenceImages      = images,
        referenceExtents     = extents,
    )

def crossSectionFigure(board, nStations: int = 5):

    '''

    Cross-section description of a few stations spaced evenly between the tips, or None when the
    board has no cross sections.

    Reads the 'deckX', 'deckY', 'deckZ', 'bottomY' and 'bottomZ' arrays the cross-section step
    adds to the geometry data, each shaped (stations, lateral points).

    '''

    data = board.geometryData
    keys = ('deckX', 'deckY', 'deckZ', 'bottomY', 'bottomZ')
    if data is None or not all(key in data for key in keys):
        return None

    deckX = _asArray(data['deckX'])
    if deckX.ndim != 2 or deckX.shape[0] < nStations + 2:
        return None

    indices = np.linspace(0, deckX.shape[0] - 1, nStations + 2).round().astype(int)[1:-1]

    deckY, deckZ, bottomY, bottomZ = [], [], [], []
    for i in indices:
        stationDeckZ, stationBottomZ = _asArray(data['deckZ'][i]), _asArray(data['bottomZ'][i])
        floor = min(np.nanmin(stationDeckZ), np.nanmin(stationBottomZ))
        deckY.append(_asArray(data['deckY'][i]))
        bottomY.append(_asArray(data['bottomY'][i]))
        deckZ.append(stationDeckZ - floor)
        bottomZ.append(stationBottomZ - floor)

    return CrossSectionFigure(stations = deckX[indices, 0], deckY = deckY, deckZ = deckZ,
                              bottomY = bottomY, bottomZ = bottomZ)

#----------------------------------------------------------------------#
# -- Plotly Renderers -- #
#----------------------------------------------------------------------#

def _baseLayout(title: str, xLabel: str, yLabel: str) -> dict:

    '''

    Layout in the active figure mode: the panel in the surface color on a sand page, axes ruled
    in the border color, and the legend boxed to match.

    '''

    return dict(
        title = dict(text = title, font = dict(color = _color('text'), size = 16)),
        paper_bgcolor = _color('bg'),
        plot_bgcolor = _color('surface'),
        font = dict(color = _color('text'), family = 'Segoe UI, Inter, sans-serif'),
        xaxis = dict(title = xLabel, gridcolor = _color('border'), zerolinecolor = _color('border'),
                     linecolor = _color('border')),
        yaxis = dict(title = yLabel, gridcolor = _color('border'), zerolinecolor = _color('border'),
                     linecolor = _color('border')),
        legend = dict(bgcolor = _color('surface'), bordercolor = _color('border'), borderwidth = 1),
        margin = dict(l = 70, r = 30, t = 70, b = 60),
    )

def _plotlyControlPolygon(figure, cubic: dict, row: int, sign: float, showlegend: bool) -> None:

    '''

    One cubic's control polygon on a subplot: tangent lines dashed from each end point to its
    handle, end points filled, handles hollow.

    '''

    xs = [cubic[key][0] for key in ('p1', 'p2', 'p3', 'p4')]
    ys = [sign * cubic[key][1] for key in ('p1', 'p2', 'p3', 'p4')]
    control = _color('orange')

    for start, stop in ((0, 1), (2, 3)):
        figure.add_trace(go.Scatter(x = [xs[start], xs[stop]], y = [ys[start], ys[stop]], mode = 'lines',
                                    line = dict(color = control, width = 1, dash = 'dash'),
                                    legendgroup = 'control', showlegend = False, hoverinfo = 'skip'),
                         row = row, col = 1)
    figure.add_trace(go.Scatter(x = [xs[0], xs[3]], y = [ys[0], ys[3]], mode = 'markers',
                                marker = dict(color = control, size = 8, symbol = 'circle'),
                                name = 'Bezier control points', legendgroup = 'control', showlegend = showlegend,
                                hovertemplate = 'end point<br>x %{x:.3f} ft<br>%{y:.3f} ft<extra></extra>'),
                     row = row, col = 1)
    figure.add_trace(go.Scatter(x = [xs[1], xs[2]], y = [ys[1], ys[2]], mode = 'markers',
                                marker = dict(color = control, size = 8, symbol = 'square-open', line = dict(width = 2)),
                                legendgroup = 'control', showlegend = False,
                                hovertemplate = 'handle<br>x %{x:.3f} ft<br>%{y:.3f} ft<extra></extra>'),
                     row = row, col = 1)

def plotlyBoardProfiles(data: BoardProfileFigure):

    '''

    Interactive planform, rocker and bottom views on a shared length axis, each at true scale
    against its reference photograph, or None without a description.

    '''

    if data is None:
        return None

    from PIL import Image
    from plotly.subplots import make_subplots

    figure = make_subplots(rows = 3, cols = 1, shared_xaxes = True, vertical_spacing = 0.07,
                           subplot_titles = ('Planform (top view)', 'Rocker (side view)', 'Planform (bottom view)'))

    stroke = _color('accentText')

    # Reference photographs go down first, under every trace
    for view, row in (('top', 1), ('side', 2), ('bottom', 3)):
        if view in data.referenceImages:
            left, right, bottom, top = data.referenceExtents[view]
            with Image.open(data.referenceImages[view]) as image:
                source = image.copy()
            figure.add_layout_image(dict(source = source, x = left, y = top, sizex = right - left, sizey = top - bottom,
                                         xanchor = 'left', yanchor = 'top', sizing = 'stretch',
                                         opacity = referenceAlpha, layer = 'below'),
                                    row = row, col = 1)

    # Planform on the top and bottom views, filled between the rails
    for row in (1, 3):
        figure.add_trace(go.Scatter(x = data.xOutline, y = data.halfWidth, mode = 'lines', name = 'Outline',
                                    line = dict(color = stroke, width = 2), showlegend = row == 1,
                                    legendgroup = 'outline',
                                    hovertemplate = 'x %{x:.3f} ft<br>half-width %{y:.3f} ft<extra></extra>'),
                         row = row, col = 1)
        figure.add_trace(go.Scatter(x = data.xOutline, y = -data.halfWidth, mode = 'lines', fill = 'tonexty',
                                    fillcolor = _rgba('accent', 0.12),
                                    line = dict(color = stroke, width = 2), showlegend = False,
                                    legendgroup = 'outline', hoverinfo = 'skip'),
                         row = row, col = 1)

    # Rocker: deck and bottom
    figure.add_trace(go.Scatter(x = data.xDeck, y = data.deckRocker, mode = 'lines', name = 'Deck rocker',
                                line = dict(color = stroke, width = 2, dash = 'dot'),
                                hovertemplate = 'x %{x:.3f} ft<br>deck %{y:.3f} ft<extra></extra>'),
                     row = 2, col = 1)
    figure.add_trace(go.Scatter(x = data.xBottom, y = data.bottomRocker, mode = 'lines', name = 'Bottom rocker',
                                line = dict(color = stroke, width = 2),
                                hovertemplate = 'x %{x:.3f} ft<br>bottom %{y:.3f} ft<extra></extra>'),
                     row = 2, col = 1)

    if data.showControlPoints:
        first = True
        for cubic in data.outlineControlPoints:
            for row in (1, 3):
                for sign in (1.0, -1.0):
                    _plotlyControlPolygon(figure, cubic, row, sign, showlegend = first)
                    first = False
        for cubic in data.rockerControlPoints:
            _plotlyControlPolygon(figure, cubic, 2, 1.0, showlegend = False)

    # Callouts in the board's own units: width in inches at the wide point, rocker at each tip
    muted = dict(color = _color('textMuted'), size = 11)
    callouts = _callouts(data)
    # Each callout sits just inside its curve, clear of the subplot titles and the other traces
    figure.add_annotation(x = data.widePointX, y = data.widePointHalfWidth, text = callouts['wide'],
                          showarrow = False, yshift = -14, font = muted, row = 1, col = 1)
    # The nose callout sits on the baseline under the kick, the one place near the tip no curve crosses
    figure.add_annotation(x = data.xBottom[-1], y = 0.0, text = callouts['nose'],
                          showarrow = False, xshift = -8, yshift = 10, xanchor = 'right', font = muted, row = 2, col = 1)
    figure.add_annotation(x = data.xBottom[0], y = data.tailRocker, text = callouts['tail'],
                          showarrow = False, xshift = 8, yshift = -14, xanchor = 'left', font = muted, row = 2, col = 1)

    layout = _baseLayout(f'{data.title}   ({_dimensionText(data)})', '', '')
    for key in ('xaxis', 'yaxis'):
        layout.pop(key)
    figure.update_layout(**layout, height = 900)

    figure.update_xaxes(gridcolor = _color('border'), zerolinecolor = _color('border'), linecolor = _color('border'))
    figure.update_yaxes(gridcolor = _color('border'), zerolinecolor = _color('border'), linecolor = _color('border'))
    figure.update_xaxes(title_text = 'Length from tail [ft]', row = 3, col = 1)
    for row, label, anchor in ((1, 'Half-width [ft]', 'x'), (2, 'Height [ft]', 'x2'), (3, 'Half-width [ft]', 'x3')):
        figure.update_yaxes(title_text = label, scaleanchor = anchor, scaleratio = 1, row = row, col = 1)
    for annotation in figure.layout.annotations:
        if annotation.text in ('Planform (top view)', 'Rocker (side view)', 'Planform (bottom view)'):
            annotation.font.color = _color('text')

    return figure

def _stationColors(count: int) -> list:

    '''

    Per-station colors, stepping along the ocean ramp from the station nearest the tail to the
    one nearest the nose, so a station's color says where along the board it sits. The low fifth
    of the ramp is held back so the first station still clears the page.

    '''

    positions = list(np.linspace(0.2, 1.0, max(int(count), 2)))

    return sample_colorscale(figurePalette.plotlyColorscale('ocean'), positions)

def plotlyCrossSections(data: CrossSectionFigure):

    '''

    Interactive station cross sections, mirrored about the centerline at true scale, or None
    without a description.

    '''

    if data is None:
        return None

    figure = go.Figure()
    colors = _stationColors(len(data.stations))

    for index, station in enumerate(data.stations):

        name = f'x = {station:.2f} ft'
        for y, z in ((data.deckY[index], data.deckZ[index]), (data.bottomY[index], data.bottomZ[index])):
            figure.add_trace(go.Scatter(x = np.concatenate([-y[::-1], y]), y = np.concatenate([z[::-1], z]),
                                        mode = 'lines', name = name, legendgroup = name,
                                        showlegend = y is data.deckY[index],
                                        line = dict(color = colors[index], width = 2),
                                        hovertemplate = name + '<br>y %{x:.3f} ft<br>z %{y:.3f} ft<extra></extra>'))

    layout = _baseLayout(data.title, 'Width from centerline [ft]', 'Height above section floor [ft]')
    layout['yaxis']['scaleanchor'] = 'x'
    layout['yaxis']['scaleratio'] = 1
    figure.update_layout(**layout)
    return figure

#----------------------------------------------------------------------#
# -- Matplotlib Renderers -- #
#----------------------------------------------------------------------#

def applyMplStyle() -> None:

    '''Themed rcParams for every Matplotlib figure pySurf draws, the debug plots included.'''

    plt.rcParams.update(figurePalette.matplotlibStyle())
    plt.rcParams.update({
        'axes.grid': True, 'grid.alpha': 0.8, 'font.size': 9,
        'axes.titlesize': 11, 'axes.titleweight': 'bold', 'legend.framealpha': 0.0,
    })

def _drawControlPolygon(axes, cubic: dict, mirrorY: bool = False) -> None:

    '''

    Draw the four control points and two tangent lines of one cubic Bezier.

    p1 and p4 are the on-curve end points (filled circles); p2 and p3 are the interior handles
    (hollow squares). mirrorY negates y for the mirrored lower half of the planform.

    '''

    sign = -1.0 if mirrorY else 1.0
    color = _color('orange')

    p1x, p1y = cubic['p1'][0], sign * cubic['p1'][1]
    p2x, p2y = cubic['p2'][0], sign * cubic['p2'][1]
    p3x, p3y = cubic['p3'][0], sign * cubic['p3'][1]
    p4x, p4y = cubic['p4'][0], sign * cubic['p4'][1]

    # Tangent lines: dashed connection from each end point to its interior handle
    axes.plot([p1x, p2x], [p1y, p2y], '--', lw = 1.0, color = color, zorder = 5)
    axes.plot([p3x, p4x], [p3y, p4y], '--', lw = 1.0, color = color, zorder = 5)

    # On-curve end points: filled circles
    axes.scatter([p1x, p4x], [p1y, p4y], marker = 'o', s = 40, zorder = 6, color = color)

    # Interior handles: hollow squares
    axes.scatter([p2x, p3x], [p2y, p3y], marker = 's', s = 35, zorder = 6,
                 facecolors = 'none', edgecolors = color, linewidths = 1.5)

def drawBoardProfileFigure(board):

    '''

    Planform, rocker and bottom views on a shared length axis, with the reference photographs
    overlaid for direct shape comparison.

    Reference image axes are remapped from pixel space to physical coordinates [ft] using the
    reference dimensions in thrusterReferenceImageDimensions.json; adjust the JSON to dial in the
    alignment between the images and the generated profiles.

    Returns:
    --------
    matplotlib.figure.Figure or None
        None when the outline and rocker have not been generated.

    '''

    data = boardProfileFigure(board)
    if data is None:
        return None

    applyMplStyle()
    figure, (axTop, axSide, axBottom) = plt.subplots(3, 1, sharex = True, constrained_layout = True, figsize = (12, 8))
    stroke, fill = _color('accentText'), _color('accent')

    for view, axes in (('top', axTop), ('side', axSide), ('bottom', axBottom)):
        if view in data.referenceImages:
            axes.imshow(plt.imread(data.referenceImages[view]), extent = data.referenceExtents[view],
                        aspect = 'equal', alpha = referenceAlpha, zorder = 0)

    # -- Planform on the top and bottom views -- #
    for axes, title in ((axTop, 'Planform (top view)'), (axBottom, 'Planform (bottom view)')):
        axes.plot(data.xOutline,  data.halfWidth, color = stroke, lw = 1.8)
        axes.plot(data.xOutline, -data.halfWidth, color = stroke, lw = 1.8)
        axes.fill_between(data.xOutline, data.halfWidth, -data.halfWidth, color = fill, alpha = 0.12)
        axes.set_ylabel('Half-width [ft]')
        axes.set_title(title)

    callouts = _callouts(data)
    # Each callout sits just inside its curve, clear of the axes titles and the other lines
    axTop.annotate(callouts['wide'], (data.widePointX, data.widePointHalfWidth), textcoords = 'offset points',
                   xytext = (0, -8), ha = 'center', va = 'top', color = _color('textMuted'), fontsize = 8)

    # -- Rocker on the side view -- #
    axSide.plot(data.xBottom, data.bottomRocker, color = stroke, lw = 1.8, label = 'bottom')
    axSide.plot(data.xDeck, data.deckRocker, color = stroke, lw = 1.4, ls = ':', label = 'deck')
    axSide.axhline(0.0, color = _color('textMuted'), lw = 0.8, ls = '--')
    # The nose callout sits on the baseline under the kick, the one place near the tip no curve crosses
    axSide.annotate(callouts['nose'], (data.xBottom[-1], 0.0), textcoords = 'offset points',
                    xytext = (-8, 4), ha = 'right', va = 'bottom', color = _color('textMuted'), fontsize = 8)
    axSide.annotate(callouts['tail'], (data.xBottom[0], data.tailRocker), textcoords = 'offset points',
                    xytext = (8, -8), ha = 'left', va = 'top', color = _color('textMuted'), fontsize = 8)
    axSide.set_ylabel('Height [ft]')
    axSide.set_title('Rocker (side view)')
    axSide.legend(loc = 'upper center')

    # -- Bezier control point overlay -- #
    if data.showControlPoints:
        for cubic in data.outlineControlPoints:
            for axes in (axTop, axBottom):
                _drawControlPolygon(axes, cubic, mirrorY = False)
                _drawControlPolygon(axes, cubic, mirrorY = True)
        for cubic in data.rockerControlPoints:
            _drawControlPolygon(axSide, cubic)

    axBottom.set_xlabel('Length from tail [ft]')
    figure.suptitle(f'{data.title}   ({_dimensionText(data)})   reference overlay at {referenceAlpha:.0%} alpha',
                    color = _color('text'), fontsize = 11)
    return figure

def drawCrossSectionFigure(board):

    '''

    Station cross sections in the YZ plane, mirrored about the centerline at true scale.

    Returns:
    --------
    matplotlib.figure.Figure or None
        None when the board has no cross sections.

    '''

    data = crossSectionFigure(board)
    if data is None:
        return None

    applyMplStyle()
    figure, axes = plt.subplots(figsize = (11, 4.2))
    stationColors = figurePalette.colormap('ocean')(np.linspace(0.2, 1.0, max(len(data.stations), 2)))

    for index, station in enumerate(data.stations):
        color = stationColors[index]
        for y, z in ((data.deckY[index], data.deckZ[index]), (data.bottomY[index], data.bottomZ[index])):
            axes.plot(y, z, color = color, lw = 1.6, label = f'x = {station:.2f} ft' if y is data.deckY[index] else None)
            axes.plot(-y, z, color = color, lw = 1.6)

    axes.set_aspect('equal', adjustable = 'box')
    axes.set_xlabel('Width from centerline [ft]')
    axes.set_ylabel('Height above section floor [ft]')
    axes.set_title(data.title)
    axes.legend(loc = 'upper right', fontsize = 8)
    return figure

#----------------------------------------------------------------------#
# -- 3D Renderers, Plotly Only -- #
#----------------------------------------------------------------------#

def _solid(color: str) -> list:

    '''A one-color plotly colorscale, for a surface drawn flat.'''

    return [[0, color], [1, color]]

def _sceneLayout() -> dict:

    '''

    Layout for a 3D plotly scene in the active figure mode: page and font colors, and each axis
    box filled with the panel color and ruled in the border color.

    '''

    named = figurePalette.colors()
    axis = dict(backgroundcolor = named['surface'], gridcolor = named['border'],
                zerolinecolor = named['border'], color = named['textMuted'])

    return dict(paper_bgcolor = named['bg'], font = dict(color = named['text'], family = 'Segoe UI, Inter, sans-serif'),
                scene = dict(xaxis = axis, yaxis = dict(axis), zaxis = dict(axis)))

def boardSurfaceFigure(board):

    '''

    The deck in seafoam and the bottom in sand-gold, each mirrored across the centerline, in one
    3D view at true scale.

    Returns:
    --------
    plotly.graph_objects.Figure or None
        None when the board has no deck and bottom surface grids.

    '''

    data = board.geometryData
    keys = ('deckX', 'deckY', 'deckZ', 'bottomX', 'bottomY', 'bottomZ')
    if data is None or not all(key in data for key in keys):
        return None

    figure = go.Figure()
    for surface, color in (('deck', _color('accent')), ('bottom', _color('yellow'))):
        x, y, z = _asArray(data[f'{surface}X']), _asArray(data[f'{surface}Y']), _asArray(data[f'{surface}Z'])
        for sign in (1.0, -1.0):
            figure.add_trace(go.Surface(x = x, y = sign * y, z = z, colorscale = _solid(color), opacity = 0.9,
                                        showscale = False, name = surface.capitalize(),
                                        hovertemplate = 'x %{x:.3f} ft<br>y %{y:.3f} ft<br>z %{z:.3f} ft<extra></extra>'))

    figure.update_layout(
        scene = dict(xaxis_title = 'Length from tail [ft]', yaxis_title = 'Width [ft]',
                     zaxis_title = 'Height [ft]'),
        title = {'text': 'Board surface', 'x': 0.5, 'xanchor': 'center', 'y': 0.9, 'yanchor': 'top'},
        scene_aspectmode = 'data', showlegend = False, margin = dict(l = 0, r = 0, t = 60, b = 0))
    figure.update_layout(**_sceneLayout())
    return figure

#----------------------------------------------------------------------#
# -- Breaking Wave Figures -- #
#----------------------------------------------------------------------#

@dataclass
class BreakingWaveFigure:

    '''

    One instant of a solitary wave on a beach: the water as particles colored by speed, the sand,
    the still water line, and where the wave is in its life.

    The close-up window follows the crest at true scale and closes in as the water shoals, its size
    in proportion to the local water column, so the breaker fills the frame; the overview shows the
    whole beach with its vertical scale stretched, since a beach is far longer than it is deep.

    '''

    time: float                                             # [s]
    positions: np.ndarray                                   # [m], fluid particles, shape (N, 2)
    speeds: np.ndarray                                      # [m/s]
    bedX: np.ndarray                                        # [m], bed vertices
    bedZ: np.ndarray                                        # [m]
    stillWaterLevel: float                                  # [m]
    depth: float                                            # [m], offshore still water depth
    waveHeight: float                                       # [m], incident wave height
    beachSlope: float                                       # [-]
    slopeParameter: float                                   # [-], Grilli's S0
    predictedBreakerType: str
    spacing: float                                          # [m], particle spacing
    crestX: float                                           # [m]
    crestHeight: float                                      # [m], above still water
    windowX: tuple = (0.0, 1.0)                             # [m], close-up span
    windowZ: tuple = (0.0, 1.0)                             # [m], close-up height range
    speedScale: float = 1.0                                 # [m/s], top of the color scale
    viewX: tuple = (0.0, 1.0)                               # [m], overview span
    breakingX: float | None = None                          # [m], breaking point, once reached
    breakingIndex: float | None = None                      # [-], Hb/hb
    breakingDepthRatio: float | None = None                 # [-], hb/d
    phase: str = 'shoaling'
    title: str = 'Solitary wave breaking on a beach'

def _crestTrack(wave) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    '''

    The crest position over time, averaged over a quarter of a second and never moving back toward
    the sea, so the close-up window glides instead of jumping when the highest point changes, with
    the crest height beside it.

    The crest is held behind the waterline: once the wave runs up the beach, the highest water can
    be a thin sheet far up the sand, and the window should stay on the water's leading edge.

    '''

    tracker = wave.tracker
    if tracker is None or not tracker.times:
        return np.array([0.0]), np.array([wave.crestX]), np.array([wave.waveHeight])

    times = np.asarray(tracker.times)
    raw = np.asarray(tracker.crestX, dtype = float)
    waterline = np.asarray(getattr(tracker, 'waterline', []), dtype = float)
    if waterline.size == raw.size:
        raw = np.where(np.isfinite(waterline), np.minimum(raw, waterline), raw)
    samples = max(1, int(round(0.25 / max(np.median(np.diff(times)), 1e-9)))) if len(times) > 1 else 1
    if samples > 1 and len(raw) > samples:
        # Edge-padded moving average, so the ends are not pulled toward zero
        padded = np.pad(raw, (samples // 2, samples - 1 - samples // 2), mode = 'edge')
        raw = np.convolve(padded, np.ones(samples) / samples, mode = 'valid')
    crestX = np.maximum.accumulate(raw)

    return times, crestX, np.asarray(tracker.crestHeight)

def _waveView(wave, time: float, windowDepths: float, closestZoom: float, breakingLabel: str) -> dict:

    '''

    What a frame of a wave on a beach shows besides the water itself, for either model: the crest,
    the close-up window, the overview span, the phase and the breaking point once reached.

    The close-up window follows the crest at true scale and closes in as the water shoals, its size
    in proportion to the water column under the crest, crest included, between `closestZoom` and
    one. It keeps the same height-to-width ratio at every zoom, so the true-scale axes keep their
    shape, and its floor follows the bed at its offshore edge.

    '''

    times, track, heights = _crestTrack(wave)
    crestX = float(np.interp(time, times, track))
    crestHeight = float(np.interp(time, times, heights))

    d, bathymetry = wave.depth, wave.bathymetry
    columnOffshore = d + 1.6 * wave.waveHeight
    column = float(bathymetry.depth(crestX)) + 1.6 * wave.waveHeight
    zoom = float(np.clip(column / columnOffshore, closestZoom, 1.0))
    halfWindow = 0.5 * windowDepths * d * zoom
    center = float(np.clip(crestX, bathymetry.startX + halfWindow, bathymetry.endX - halfWindow))
    zSpan = (columnOffshore + 0.3 * d) * zoom
    zLow = float(bathymetry.elevation(center - halfWindow)) - 0.15 * d * zoom
    viewEnd = min(bathymetry.endX, wave.shorelineX + 0.5 * (bathymetry.endX - wave.shorelineX))

    point = wave.breakingPoint
    broken = point is not None and time >= point.time
    if not broken:
        phase = 'shoaling' if crestX > wave.toeX else 'approaching'
    else:
        phase = breakingLabel if time < point.time + 1.5 * math.sqrt(d / constants.gravity) else 'broken'

    return {
        'bedX': bathymetry.x, 'bedZ': bathymetry.z, 'stillWaterLevel': bathymetry.stillWaterLevel,
        'depth': d, 'waveHeight': wave.waveHeight, 'beachSlope': wave.beachSlope,
        'slopeParameter': wave.slopeParameter, 'predictedBreakerType': wave.predictedBreakerType,
        'crestX': crestX, 'crestHeight': crestHeight,
        'windowX': (center - halfWindow, center + halfWindow), 'windowZ': (zLow, zLow + zSpan),
        'speedScale': 1.2 * math.sqrt(constants.gravity * d), 'viewX': (bathymetry.startX, viewEnd),
        'breakingX': point.crestX if broken else None,
        'breakingIndex': point.breakingIndex if broken else None,
        'breakingDepthRatio': point.depth / d if broken else None,
        'phase': phase,
    }

def breakingWaveFigure(wave, frameIndex: int = -1, windowDepths: float = 5.0, closestZoom: float = 0.45):

    '''

    Description of one recorded frame, or None when the wave has no frames yet.

    Parameters:
    -----------
    wave : pysurf.BreakingWave
        A built (and usually run) breaking wave
    frameIndex : int
        Which recorded frame
    windowDepths : float
        Width of the close-up window offshore, in depths
    closestZoom : float
        Smallest fraction of its offshore size the close-up window closes in to

    '''

    if not getattr(wave, 'frames', None):
        return None

    frame = wave.frames[frameIndex]
    view = _waveView(wave, frame.time, windowDepths, closestZoom, 'overturning')

    return BreakingWaveFigure(time = frame.time, positions = frame.positions, speeds = frame.speeds,
                              spacing = wave.spacing, **view)

def _waveCaption(data: BreakingWaveFigure) -> str:

    '''The status line under the title: time, crest and phase, and the breaking point once reached.'''

    nondimensionalTime = data.time * math.sqrt(constants.gravity / data.depth)
    caption = (f't = {data.time:5.2f} s   (t sqrt(g/d) = {nondimensionalTime:5.1f})   '
               f'crest H/d = {data.crestHeight / data.depth:4.2f}   {data.phase}')
    if data.breakingIndex is not None:
        caption += f'   breaks at hb/d = {data.breakingDepthRatio:.2f}, Hb/hb = {data.breakingIndex:.2f}'

    return caption

def _waveTitle(data: BreakingWaveFigure) -> str:

    '''The figure title: the incident wave, the slope, and the breaker Grilli's criterion predicts.'''

    return (f'{data.title}   H/d = {data.waveHeight / data.depth:.2f}, slope 1:{1.0 / data.beachSlope:.0f}, '
            f'S0 = {data.slopeParameter:.3f} ({data.predictedBreakerType})')

def _waterStops() -> tuple:

    '''

    The ocean colormap without its lightest stop, so water at rest still reads as water against the
    sand-white page (and, in dark mode, against the night-ocean ground).

    '''

    return figurePalette.colormapStops[figurePalette.figureMode()]['ocean'][1:]

def _waterColormap():

    '''The particle speed colormap as a Matplotlib colormap.'''

    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list('surfWater', list(_waterStops()))

def _particleMarkerArea(axes, spacing: float, xSpan: float, zSpan: float) -> float:

    '''

    Scatter marker area [pt^2] that draws each particle about one spacing across at the axes' scale,
    so the particles read as a continuous body of water. On stretched axes the geometric mean of
    the two scales is used: the horizontal scale alone draws specks, the vertical alone blobs.

    '''

    box = axes.get_window_extent()
    pointsPerPixel = 72.0 / axes.figure.dpi
    pointsPerMeter = np.sqrt(box.width / xSpan * box.height / zSpan) * pointsPerPixel

    return (0.9 * spacing * pointsPerMeter)**2

def _drawBeach(axes, data, floor: float, levelOnTop: bool = False) -> None:

    '''The sand down to `floor`, the bed line and the still water line.'''

    sandX = np.concatenate([data.bedX, [data.bedX[-1], data.bedX[0]]])
    sandZ = np.concatenate([data.bedZ, [floor, floor]])
    axes.fill(sandX, sandZ, color = _color('yellow'), zorder = 1)
    axes.plot(data.bedX, data.bedZ, color = _color('orange'), lw = 1.0, zorder = 3)
    axes.axhline(data.stillWaterLevel, color = _color('textMuted'), lw = 0.8, ls = '--', zorder = 4 if levelOnTop else 2)

def _drawWaveAxes(axes, data: BreakingWaveFigure, xRange: tuple, zRange: tuple, markerSize: float):

    '''

    The sand, still water line and particles on one set of axes; returns the particle scatter so an
    animation can move it.

    '''

    _drawBeach(axes, data, zRange[0] - data.depth)
    scatter = axes.scatter(data.positions[:, 0], data.positions[:, 1], c = data.speeds, s = markerSize,
                           cmap = _waterColormap(), vmin = 0.0, vmax = data.speedScale,
                           linewidths = 0, zorder = 3)
    axes.set_xlim(*xRange)
    axes.set_ylim(*zRange)
    axes.grid(False)

    return scatter

def drawBreakingWaveFrame(wave, frameIndex: int = -1):

    '''

    One recorded frame: the whole beach above, stretched vertically, and a true-scale close-up of the
    crest below.

    Returns:
    --------
    matplotlib.figure.Figure or None
        None when the wave has no frames.

    '''

    data = breakingWaveFigure(wave, frameIndex)
    if data is None:
        return None

    applyMplStyle()
    figure, (overview, closeUp) = plt.subplots(2, 1, figsize = (11, 6.6), constrained_layout = True,
                                               gridspec_kw = {'height_ratios': [1.0, 1.6]})
    artists = _layoutWaveFrame(figure, overview, closeUp, data)
    _updateWaveFrame(artists, data)

    return figure

def _layoutWaveFrame(figure, overview, closeUp, data: BreakingWaveFigure) -> dict:

    '''

    The parts of a wave frame that stay put between frames, and handles on the parts that move.

    '''

    d = data.depth
    zTop = data.stillWaterLevel + 1.6 * data.waveHeight + 0.15 * d
    zRange = (-0.15 * d, zTop)
    overviewScatter = _drawWaveAxes(overview, data, data.viewX, zRange, 1.0)
    overview.set_ylabel('Height [m]')
    overview.set_title('Whole beach (vertical scale stretched)')

    closeScatter = _drawWaveAxes(closeUp, data, data.windowX, data.windowZ, 6.0)
    # Equal scales on fixed limits: the axes box takes the shape of the window, which keeps its
    # height-to-width ratio as it zooms
    closeUp.set_aspect('equal', adjustable = 'box')
    closeUp.set_xlabel('Distance [m]')
    closeUp.set_ylabel('Height [m]')
    closeUp.set_title('Crest, true scale')

    window = plt.Rectangle((data.windowX[0], data.windowZ[0]), data.windowX[1] - data.windowX[0],
                           data.windowZ[1] - data.windowZ[0], fill = False, edgecolor = _color('accentText'),
                           lw = 1.0, zorder = 4)
    overview.add_patch(window)
    marker, = overview.plot([], [], marker = 'v', color = _color('red'), markersize = 8, ls = 'none', zorder = 5)

    colorbar = figure.colorbar(closeScatter, ax = [overview, closeUp], location = 'right', shrink = 0.8, pad = 0.01)
    colorbar.set_label('Particle speed [m/s]')
    figure.suptitle(_waveTitle(data), color = _color('text'), fontsize = 11)
    caption = closeUp.text(0.01, 0.97, '', transform = closeUp.transAxes, va = 'top', ha = 'left', fontsize = 9,
                           color = _color('text'))

    # Marker sizes depend on the settled axes boxes, which exist only after a layout pass
    figure.canvas.draw()
    zSpan = zRange[1] - zRange[0]
    overviewScatter.set_sizes([_particleMarkerArea(overview, data.spacing, data.viewX[1] - data.viewX[0], zSpan)])

    return {'overview': overviewScatter, 'closeUp': closeScatter, 'closeAxes': closeUp, 'window': window,
            'marker': marker, 'caption': caption, 'zTop': zTop}

def _updateWaveFrame(artists: dict, data: BreakingWaveFigure) -> None:

    '''Move the particles, the close-up window, the breaking marker and the caption to this frame.'''

    for key in ('overview', 'closeUp'):
        artists[key].set_offsets(data.positions)
        artists[key].set_array(data.speeds)

    closeAxes = artists['closeAxes']
    closeAxes.set_xlim(*data.windowX)
    closeAxes.set_ylim(*data.windowZ)
    width, height = data.windowX[1] - data.windowX[0], data.windowZ[1] - data.windowZ[0]
    artists['closeUp'].set_sizes([_particleMarkerArea(closeAxes, data.spacing, width, height)])
    artists['window'].set_xy((data.windowX[0], data.windowZ[0]))
    artists['window'].set_width(width)
    artists['window'].set_height(height)
    if data.breakingX is not None:
        artists['marker'].set_data([data.breakingX], [artists['zTop'] - 0.05 * data.depth])
    else:
        artists['marker'].set_data([], [])
    artists['caption'].set_text(_waveCaption(data))

def writeBreakingWaveAnimation(wave, path: str, fps: int = 12, dpi: int = 90, frameStep: int = 1,
                               colors: int | None = 64) -> str:

    '''

    Write the recorded transient as an animated GIF.

    The figure is laid out once and each frame only moves the particles, the window and the text,
    so writing a few hundred frames takes seconds per hundred rather than minutes. The frames are
    then mapped onto one shared palette, which roughly halves the file: the per-frame palettes the
    writer chooses differ frame to frame and compress poorly.

    Parameters:
    -----------
    wave : pysurf.BreakingWave
        A run breaking wave
    path : str
        Output path, ending in .gif
    fps : int
        Frames per second of the GIF
    dpi : int
        Resolution of each frame
    frameStep : int
        Use every n-th recorded frame
    colors : int | None
        Size of the shared palette; None keeps the writer's per-frame palettes

    Returns:
    --------
    str : The path written

    '''

    from matplotlib.animation import PillowWriter

    indices = list(range(0, len(wave.frames), max(1, int(frameStep))))
    if not indices:
        raise ValueError('The wave has no recorded frames to animate: run it first.')

    first = breakingWaveFigure(wave, indices[0])
    applyMplStyle()
    figure, (overview, closeUp) = plt.subplots(2, 1, figsize = (11, 6.6), constrained_layout = True,
                                               gridspec_kw = {'height_ratios': [1.0, 1.6]})
    artists = _layoutWaveFrame(figure, overview, closeUp, first)

    writer = PillowWriter(fps = fps)
    with writer.saving(figure, path, dpi):
        for index in indices:
            _updateWaveFrame(artists, breakingWaveFigure(wave, index))
            writer.grab_frame()
    plt.close(figure)

    if colors:
        _sharePalette(path, colors)

    return path

def _sharePalette(path: str, colors: int) -> None:

    '''

    Rewrite a GIF with every frame mapped onto one palette, chosen from a strip of frames spread
    through the animation so the colors of every phase are in it.

    '''

    from PIL import Image

    with Image.open(path) as gif:
        frames, durations = [], []
        for index in range(gif.n_frames):
            gif.seek(index)
            frames.append(gif.convert('RGB'))
            durations.append(gif.info.get('duration', 100))

    samples = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, min(6, len(frames)))]
    strip = Image.new('RGB', (samples[0].width, samples[0].height * len(samples)))
    for row, sample in enumerate(samples):
        strip.paste(sample, (0, row * sample.height))
    palette = strip.quantize(colors = colors, method = Image.Quantize.MEDIANCUT)

    mapped = [frame.quantize(palette = palette, dither = Image.Dither.NONE) for frame in frames]
    mapped[0].save(path, save_all = True, append_images = mapped[1:], duration = durations, loop = 0,
                   optimize = True, disposal = 1)

def plotlyBreakingWaveAnimation(wave, maxFrames: int = 60, maxParticles: int = 6000):

    '''

    The recorded transient as an interactive plotly animation with a time slider, or None when the
    wave has no frames.

    Frames and particles are thinned to the given counts so the HTML stays a few megabytes.

    '''

    if not getattr(wave, 'frames', None):
        return None

    indices = np.unique(np.linspace(0, len(wave.frames) - 1, min(maxFrames, len(wave.frames))).astype(int))
    first = breakingWaveFigure(wave, int(indices[0]))
    stride = max(1, len(first.positions) // maxParticles)
    stops = _waterStops()
    colorscale = [[index / (len(stops) - 1), color] for index, color in enumerate(stops)]

    def particleTrace(data: BreakingWaveFigure):
        return go.Scattergl(x = data.positions[::stride, 0], y = data.positions[::stride, 1], mode = 'markers',
                            marker = dict(size = 3, color = data.speeds[::stride], colorscale = colorscale, cmin = 0.0,
                                          cmax = data.speedScale, colorbar = dict(title = 'Speed [m/s]')),
                            hovertemplate = 'x %{x:.2f} m<br>z %{y:.2f} m<br>%{marker.color:.2f} m/s<extra></extra>',
                            name = 'Water')

    floor = -0.15 * first.depth
    sand = go.Scatter(x = np.concatenate([first.bedX, [first.bedX[-1], first.bedX[0]]]),
                      y = np.concatenate([first.bedZ, [floor - first.depth, floor - first.depth]]),
                      fill = 'toself', fillcolor = _color('yellow'), line = dict(color = _color('orange'), width = 1),
                      mode = 'lines', hoverinfo = 'skip', name = 'Sand')

    frames = []
    for index in indices:
        data = breakingWaveFigure(wave, int(index))
        frames.append(go.Frame(data = [particleTrace(data)], traces = [1], name = f'{data.time:.2f}',
                               layout = dict(title_text = f'{_waveTitle(data)}<br><sub>{_waveCaption(data)}</sub>')))

    figure = go.Figure(data = [sand, particleTrace(first)], frames = frames)
    layout = _baseLayout(f'{_waveTitle(first)}<br><sub>{_waveCaption(first)}</sub>', 'Distance [m]', 'Height [m]')
    layout['xaxis']['range'] = list(first.viewX)
    layout['yaxis']['range'] = [floor, first.stillWaterLevel + 1.6 * first.waveHeight + 0.15 * first.depth]
    layout['showlegend'] = False
    layout['updatemenus'] = [dict(type = 'buttons', x = 0.0, y = -0.12, xanchor = 'left', showactive = False,
                                  buttons = [dict(label = 'Play', method = 'animate',
                                                  args = [None, dict(frame = dict(duration = 80, redraw = True),
                                                                     fromcurrent = True)])])]
    layout['sliders'] = [dict(x = 0.08, y = -0.12, len = 0.9, currentvalue = dict(prefix = 't = ', suffix = ' s'),
                              steps = [dict(label = frame.name, method = 'animate',
                                            args = [[frame.name], dict(mode = 'immediate', frame = dict(duration = 0,
                                                                                                         redraw = True))])
                                       for frame in frames])]
    layout['margin'] = dict(l = 70, r = 30, t = 90, b = 110)
    figure.update_layout(**layout)

    return figure

#----------------------------------------------------------------------#
# -- Surface Wave Figures -- #
#----------------------------------------------------------------------#

@dataclass
class SurfaceWaveFigure:

    '''

    One instant of a solitary wave on a beach from a free-surface model: the water column between
    the bed and the surface, colored by depth-averaged speed and whitened where the wave is
    breaking, over the sand and the still water line.

    The close-up and the overview are framed as in `BreakingWaveFigure`.

    '''

    time: float                                             # [s]
    x: np.ndarray                                           # [m], cell centers
    cellSize: float                                         # [m]
    surface: np.ndarray                                     # [m], free-surface elevation
    speeds: np.ndarray                                      # [m/s], depth-averaged speed
    breakingZone: np.ndarray                                # cells carried as a breaking front
    bedX: np.ndarray                                        # [m], bed vertices
    bedZ: np.ndarray                                        # [m]
    stillWaterLevel: float                                  # [m]
    depth: float                                            # [m], offshore still water depth
    waveHeight: float                                       # [m], incident wave height
    beachSlope: float                                       # [-]
    slopeParameter: float                                   # [-], Grilli's S0
    predictedBreakerType: str
    crestX: float                                           # [m]
    crestHeight: float                                      # [m], above still water
    windowX: tuple = (0.0, 1.0)                             # [m], close-up span
    windowZ: tuple = (0.0, 1.0)                             # [m], close-up height range
    speedScale: float = 1.0                                 # [m/s], top of the color scale
    viewX: tuple = (0.0, 1.0)                               # [m], overview span
    breakingX: float | None = None                          # [m], breaking point, once reached
    breakingIndex: float | None = None                      # [-], Hb/hb
    breakingDepthRatio: float | None = None                 # [-], hb/d
    phase: str = 'shoaling'
    title: str = 'Solitary wave on a beach, Serre-Green-Naghdi'

def surfaceWaveFigure(wave, frameIndex: int = -1, windowDepths: float = 5.0, closestZoom: float = 0.45):

    '''

    Description of one recorded frame of a free-surface wave, or None when it has no frames yet.

    Parameters:
    -----------
    wave : pysurf.SurfaceBreakingWave
        A built (and usually run) surface breaking wave
    frameIndex : int
        Which recorded frame
    windowDepths : float
        Width of the close-up window offshore, in depths
    closestZoom : float
        Smallest fraction of its offshore size the close-up window closes in to

    '''

    if not getattr(wave, 'frames', None):
        return None

    frame = wave.frames[frameIndex]
    view = _waveView(wave, frame.time, windowDepths, closestZoom, 'breaking')

    return SurfaceWaveFigure(time = frame.time, x = wave.solver.x, cellSize = wave.solver.cellSize,
                             surface = frame.surface, speeds = np.abs(frame.velocity),
                             breakingZone = frame.breakingZone, **view)

def _waterColumns(data: SurfaceWaveFigure, cellsPerColumn: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    '''

    The water as quadrilateral columns from the bed up to the surface, with the speed and the
    breaking state of each.

    Corners sit on the column edges, with the bed and surface interpolated there, so the columns
    join into one smooth body that thins to nothing at the waterline. `cellsPerColumn` merges
    neighboring cells into one column, taking their mean speed, for an overview where single cells
    are smaller than a pixel.

    Returns:
    --------
    tuple : (corners shaped (columns, 4, 2) [m], speed [m/s], breaking flag)

    '''

    x, half = data.x, 0.5 * data.cellSize
    starts = np.arange(0, len(x), max(1, int(cellsPerColumn)))
    edges = np.append(x[starts] - half, x[-1] + half)
    bed = np.interp(edges, data.bedX, data.bedZ)
    top = np.maximum(np.interp(edges, x, data.surface), bed)

    corners = np.empty((len(starts), 4, 2))
    corners[:, 0, 0], corners[:, 0, 1] = edges[:-1], bed[:-1]
    corners[:, 1, 0], corners[:, 1, 1] = edges[1:], bed[1:]
    corners[:, 2, 0], corners[:, 2, 1] = edges[1:], top[1:]
    corners[:, 3, 0], corners[:, 3, 1] = edges[:-1], top[:-1]

    counts = np.diff(np.append(starts, len(x)))
    speed = np.add.reduceat(data.speeds, starts) / counts
    breaking = np.maximum.reduceat(data.breakingZone.astype(int), starts) > 0

    return corners, speed, breaking

def _waterColors(speed: np.ndarray, breaking: np.ndarray, speedScale: float, foamShare: float = 0.6) -> np.ndarray:

    '''Face colors of the water columns: the speed colormap, blended toward foam where the wave breaks.'''

    from matplotlib.colors import to_rgba

    colors = _waterColormap()(np.clip(speed / speedScale, 0.0, 1.0))
    # Foam is the lightest tone of the mode: the panel color in light mode, the text color in dark
    foam = np.array(to_rgba(_color('surface' if figurePalette.figureMode() == 'light' else 'text')))
    colors[breaking] = (1.0 - foamShare) * colors[breaking] + foamShare * foam

    return colors

def _drawSurfaceAxes(axes, data: SurfaceWaveFigure, xRange: tuple, zRange: tuple, cellsPerColumn: int) -> dict:

    '''The sand, the water columns and the surface line on one set of axes, with handles to move them.'''

    from matplotlib.collections import PolyCollection

    _drawBeach(axes, data, zRange[0] - data.depth, levelOnTop = True)
    corners, speed, breaking = _waterColumns(data, cellsPerColumn)
    # Face-colored edges wide enough to cover the antialiasing seams between neighboring columns
    water = PolyCollection(corners, facecolors = _waterColors(speed, breaking, data.speedScale),
                           edgecolors = 'face', linewidths = 0.8, zorder = 2)
    axes.add_collection(water)
    wet = data.surface > np.interp(data.x, data.bedX, data.bedZ) + 1e-6 * data.depth
    line, = axes.plot(data.x, np.where(wet, data.surface, np.nan), color = _color('accentText'), lw = 1.1, zorder = 5)
    axes.set_xlim(*xRange)
    axes.set_ylim(*zRange)
    axes.grid(False)

    return {'water': water, 'line': line, 'cellsPerColumn': cellsPerColumn}

def _moveSurfaceAxes(handles: dict, data: SurfaceWaveFigure) -> None:

    '''Move the water columns and the surface line to this frame.'''

    corners, speed, breaking = _waterColumns(data, handles['cellsPerColumn'])
    handles['water'].set_verts(corners)
    handles['water'].set_facecolor(_waterColors(speed, breaking, data.speedScale))
    wet = data.surface > np.interp(data.x, data.bedX, data.bedZ) + 1e-6 * data.depth
    handles['line'].set_ydata(np.where(wet, data.surface, np.nan))

def _layoutSurfaceFrame(figure, overview, closeUp, data: SurfaceWaveFigure) -> dict:

    '''

    The parts of a surface-wave frame that stay put between frames, and handles on the parts that
    move.

    '''

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch

    d = data.depth
    zTop = data.stillWaterLevel + 1.6 * data.waveHeight + 0.15 * d
    zRange = (-0.15 * d, zTop)
    # About one column per two pixels across the overview
    overviewColumns = max(1, int(len(data.x) / 600))
    overviewHandles = _drawSurfaceAxes(overview, data, data.viewX, zRange, overviewColumns)
    overview.set_ylabel('Height [m]')
    overview.set_title('Whole beach (vertical scale stretched)')

    closeHandles = _drawSurfaceAxes(closeUp, data, data.windowX, data.windowZ, 1)
    # Equal scales on fixed limits: the axes box takes the shape of the window
    closeUp.set_aspect('equal', adjustable = 'box')
    closeUp.set_xlabel('Distance [m]')
    closeUp.set_ylabel('Height [m]')
    closeUp.set_title('Crest, true scale')
    foam = _waterColors(np.zeros(1), np.ones(1, dtype = bool), data.speedScale)[0]
    closeUp.legend(handles = [Patch(facecolor = foam, edgecolor = _color('border'), label = 'Breaking front (bore)')],
                   loc = 'upper right', fontsize = 8, framealpha = 0.9)

    window = plt.Rectangle((data.windowX[0], data.windowZ[0]), data.windowX[1] - data.windowX[0],
                           data.windowZ[1] - data.windowZ[0], fill = False, edgecolor = _color('accentText'),
                           lw = 1.0, zorder = 6)
    overview.add_patch(window)
    marker, = overview.plot([], [], marker = 'v', color = _color('red'), markersize = 8, ls = 'none', zorder = 7)

    speedScale = ScalarMappable(norm = Normalize(0.0, data.speedScale), cmap = _waterColormap())
    colorbar = figure.colorbar(speedScale, ax = [overview, closeUp], location = 'right', shrink = 0.8, pad = 0.01)
    colorbar.set_label('Depth-averaged speed [m/s]')
    figure.suptitle(_waveTitle(data), color = _color('text'), fontsize = 11)
    caption = closeUp.text(0.01, 0.97, '', transform = closeUp.transAxes, va = 'top', ha = 'left', fontsize = 9,
                           color = _color('text'), zorder = 8)

    return {'overview': overviewHandles, 'closeUp': closeHandles, 'closeAxes': closeUp, 'window': window,
            'marker': marker, 'caption': caption, 'zTop': zTop}

def _updateSurfaceFrame(artists: dict, data: SurfaceWaveFigure) -> None:

    '''Move the water, the close-up window, the breaking marker and the caption to this frame.'''

    _moveSurfaceAxes(artists['overview'], data)
    _moveSurfaceAxes(artists['closeUp'], data)

    closeAxes = artists['closeAxes']
    closeAxes.set_xlim(*data.windowX)
    closeAxes.set_ylim(*data.windowZ)
    width, height = data.windowX[1] - data.windowX[0], data.windowZ[1] - data.windowZ[0]
    artists['window'].set_xy((data.windowX[0], data.windowZ[0]))
    artists['window'].set_width(width)
    artists['window'].set_height(height)
    if data.breakingX is not None:
        artists['marker'].set_data([data.breakingX], [artists['zTop'] - 0.05 * data.depth])
    else:
        artists['marker'].set_data([], [])
    artists['caption'].set_text(_waveCaption(data))

def drawSurfaceWaveFrame(wave, frameIndex: int = -1):

    '''

    One recorded frame of a free-surface wave: the whole beach above, stretched vertically, and a
    true-scale close-up of the crest below.

    Returns:
    --------
    matplotlib.figure.Figure or None
        None when the wave has no frames.

    '''

    data = surfaceWaveFigure(wave, frameIndex)
    if data is None:
        return None

    applyMplStyle()
    figure, (overview, closeUp) = plt.subplots(2, 1, figsize = (11, 6.6), constrained_layout = True,
                                               gridspec_kw = {'height_ratios': [1.0, 1.6]})
    artists = _layoutSurfaceFrame(figure, overview, closeUp, data)
    _updateSurfaceFrame(artists, data)

    return figure

def writeSurfaceWaveAnimation(wave, path: str, fps: int = 12, dpi: int = 90, frameStep: int = 1,
                              colors: int | None = 64) -> str:

    '''

    Write the recorded transient of a free-surface wave as an animated GIF.

    Laid out once, moved frame by frame, and mapped onto one shared palette, as
    `writeBreakingWaveAnimation` does for particles.

    Parameters:
    -----------
    wave : pysurf.SurfaceBreakingWave
        A run surface breaking wave
    path : str
        Output path, ending in .gif
    fps : int
        Frames per second of the GIF
    dpi : int
        Resolution of each frame
    frameStep : int
        Use every n-th recorded frame
    colors : int | None
        Size of the shared palette; None keeps the writer's per-frame palettes

    Returns:
    --------
    str : The path written

    '''

    from matplotlib.animation import PillowWriter

    indices = list(range(0, len(wave.frames), max(1, int(frameStep))))
    if not indices:
        raise ValueError('The wave has no recorded frames to animate: run it first.')

    first = surfaceWaveFigure(wave, indices[0])
    applyMplStyle()
    figure, (overview, closeUp) = plt.subplots(2, 1, figsize = (11, 6.6), constrained_layout = True,
                                               gridspec_kw = {'height_ratios': [1.0, 1.6]})
    artists = _layoutSurfaceFrame(figure, overview, closeUp, first)

    writer = PillowWriter(fps = fps)
    with writer.saving(figure, path, dpi):
        for index in indices:
            _updateSurfaceFrame(artists, surfaceWaveFigure(wave, index))
            writer.grab_frame()
    plt.close(figure)

    if colors:
        _sharePalette(path, colors)

    return path

#----------------------------------------------------------------------#
# -- Overturning Wave Figures -- #
#----------------------------------------------------------------------#

@dataclass
class OverturningFigure:

    '''

    One instant of a wave from the potential flow model: the water as a polygon bounded by the free
    surface, which may overhang, and the bed, with the surface colored by the water's speed.

    The close-up and the overview are framed as in `BreakingWaveFigure`.

    '''

    time: float                                             # [s]
    surface: np.ndarray                                     # [m], free-surface nodes, shape (N, 2)
    speed: np.ndarray                                       # [m/s], at each node
    bedX: np.ndarray                                        # [m], bed vertices
    bedZ: np.ndarray                                        # [m]
    stillWaterLevel: float                                  # [m]
    depth: float                                            # [m], offshore still water depth
    waveHeight: float                                       # [m], incident wave height
    beachSlope: float                                       # [-]
    slopeParameter: float                                   # [-], Grilli's S0
    predictedBreakerType: str
    crestX: float                                           # [m]
    crestHeight: float                                      # [m], above still water
    windowX: tuple = (0.0, 1.0)                             # [m], close-up span
    windowZ: tuple = (0.0, 1.0)                             # [m], close-up height range
    speedScale: float = 1.0                                 # [m/s], top of the color scale
    viewX: tuple = (0.0, 1.0)                               # [m], overview span
    breakingX: float | None = None                          # [m], breaking point, once reached
    breakingIndex: float | None = None                      # [-], Hb/hb
    breakingDepthRatio: float | None = None                 # [-], hb/d
    phase: str = 'shoaling'
    title: str = 'Solitary wave overturning, potential flow'

def overturningFigure(wave, frameIndex: int = -1, windowDepths: float = 5.0, closestZoom: float = 0.35):

    '''

    Description of one recorded frame of an overturning wave, or None when it has no frames yet.

    Parameters:
    -----------
    wave : pysurf.OverturningWave
        A built (and usually run) overturning wave
    frameIndex : int
        Which recorded frame
    windowDepths : float
        Width of the close-up window offshore, in depths
    closestZoom : float
        Smallest fraction of its offshore size the close-up window closes in to

    '''

    if not getattr(wave, 'frames', None):
        return None

    frame = wave.frames[frameIndex]
    view = _waveView(wave, frame.time, windowDepths, closestZoom, 'overturning')

    return OverturningFigure(time = frame.time, surface = frame.surface, speed = frame.speed, **view)

def _waterPolygon(data) -> np.ndarray:

    '''The water region as one closed polygon: the surface left to right, then the bed back.'''

    surface = data.surface
    inside = (data.bedX > surface[0, 0]) & (data.bedX < surface[-1, 0])
    bedX = np.concatenate([[surface[-1, 0]], data.bedX[inside][::-1], [surface[0, 0]]])
    bed = np.column_stack([bedX, np.interp(bedX, data.bedX, data.bedZ)])

    return np.vstack([surface, bed])

def _surfaceSegments(data) -> tuple[np.ndarray, np.ndarray]:

    '''The surface as segments for a speed-colored line, with the mean speed of each.'''

    surface = data.surface
    segments = np.stack([surface[:-1], surface[1:]], axis = 1)

    return segments, 0.5 * (data.speed[:-1] + data.speed[1:])

def _drawOverturningAxes(axes, data: OverturningFigure, xRange: tuple, zRange: tuple, lineWidth: float) -> dict:

    '''The sand, the water polygon and the speed-colored surface on one set of axes, with handles.'''

    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.patches import Polygon

    _drawBeach(axes, data, zRange[0] - data.depth, levelOnTop = True)
    water = Polygon(_waterPolygon(data), closed = True, facecolor = _waterColormap()(0.15), edgecolor = 'none', zorder = 2)
    axes.add_patch(water)
    segments, speed = _surfaceSegments(data)
    line = LineCollection(segments, cmap = _waterColormap(), norm = Normalize(0.0, data.speedScale), linewidths = lineWidth,
                          zorder = 5)
    line.set_array(speed)
    axes.add_collection(line)
    axes.set_xlim(*xRange)
    axes.set_ylim(*zRange)
    axes.grid(False)

    return {'water': water, 'line': line}

def _moveOverturningAxes(handles: dict, data: OverturningFigure) -> None:

    '''Move the water polygon and the surface line to this frame.'''

    handles['water'].set_xy(_waterPolygon(data))
    segments, speed = _surfaceSegments(data)
    handles['line'].set_segments(segments)
    handles['line'].set_array(speed)

def _layoutOverturningFrame(figure, overview, closeUp, data: OverturningFigure) -> dict:

    '''The parts of an overturning frame that stay put, and handles on the parts that move.'''

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    d = data.depth
    zTop = data.stillWaterLevel + 1.8 * data.waveHeight + 0.15 * d
    zRange = (-0.15 * d, zTop)
    overviewHandles = _drawOverturningAxes(overview, data, data.viewX, zRange, 1.2)
    overview.set_ylabel('Height [m]')
    overview.set_title('Whole tank (vertical scale stretched)')

    closeHandles = _drawOverturningAxes(closeUp, data, data.windowX, data.windowZ, 2.0)
    closeUp.set_aspect('equal', adjustable = 'box')
    closeUp.set_xlabel('Distance [m]')
    closeUp.set_ylabel('Height [m]')
    closeUp.set_title('Crest, true scale')

    window = plt.Rectangle((data.windowX[0], data.windowZ[0]), data.windowX[1] - data.windowX[0],
                           data.windowZ[1] - data.windowZ[0], fill = False, edgecolor = _color('accentText'),
                           lw = 1.0, zorder = 6)
    overview.add_patch(window)
    marker, = overview.plot([], [], marker = 'v', color = _color('red'), markersize = 8, ls = 'none', zorder = 7)

    colorbar = figure.colorbar(ScalarMappable(norm = Normalize(0.0, data.speedScale), cmap = _waterColormap()),
                               ax = [overview, closeUp], location = 'right', shrink = 0.8, pad = 0.01)
    colorbar.set_label('Surface water speed [m/s]')
    figure.suptitle(_waveTitle(data), color = _color('text'), fontsize = 11)
    caption = closeUp.text(0.01, 0.97, '', transform = closeUp.transAxes, va = 'top', ha = 'left', fontsize = 9,
                           color = _color('text'), zorder = 8)

    return {'overview': overviewHandles, 'closeUp': closeHandles, 'closeAxes': closeUp, 'window': window,
            'marker': marker, 'caption': caption, 'zTop': zTop}

def _updateOverturningFrame(artists: dict, data: OverturningFigure) -> None:

    '''Move the water, the close-up window, the breaking marker and the caption to this frame.'''

    _moveOverturningAxes(artists['overview'], data)
    _moveOverturningAxes(artists['closeUp'], data)
    closeAxes = artists['closeAxes']
    closeAxes.set_xlim(*data.windowX)
    closeAxes.set_ylim(*data.windowZ)
    artists['window'].set_xy((data.windowX[0], data.windowZ[0]))
    artists['window'].set_width(data.windowX[1] - data.windowX[0])
    artists['window'].set_height(data.windowZ[1] - data.windowZ[0])
    if data.breakingX is not None:
        artists['marker'].set_data([data.breakingX], [artists['zTop'] - 0.05 * data.depth])
    else:
        artists['marker'].set_data([], [])
    artists['caption'].set_text(_waveCaption(data))

def drawOverturningFrame(wave, frameIndex: int = -1):

    '''One recorded frame of an overturning wave: the whole tank above and a true-scale close-up below.'''

    data = overturningFigure(wave, frameIndex)
    if data is None:
        return None

    applyMplStyle()
    figure, (overview, closeUp) = plt.subplots(2, 1, figsize = (11, 6.6), constrained_layout = True,
                                               gridspec_kw = {'height_ratios': [1.0, 1.6]})
    artists = _layoutOverturningFrame(figure, overview, closeUp, data)
    _updateOverturningFrame(artists, data)

    return figure

def drawOverturningProfiles(wave, count: int = 8):

    '''

    The free surface at evenly spaced instants from shortly before breaking to the last state, at
    true scale over the bed: the face steepening, turning vertical and curling into a jet.

    Returns:
    --------
    matplotlib.figure.Figure or None
        None when the wave has no frames.

    '''

    frames = getattr(wave, 'frames', None)
    if not frames:
        return None

    d = wave.depth
    point = wave.breakingPoint
    endTime = frames[-1].time
    startTime = (point.time if point is not None else endTime) - 1.2 * float(np.sqrt(d / constants.gravity))
    chosen = [frame for frame in frames if frame.time >= startTime]
    indices = np.unique(np.linspace(0, len(chosen) - 1, min(count, len(chosen))).round().astype(int))
    chosen = [chosen[i] for i in indices]

    applyMplStyle()
    figure, axes = plt.subplots(figsize = (11, 4.6), constrained_layout = True)
    bathymetry = wave.bathymetry
    last = chosen[-1].surface
    crestX = float(last[crestIndex(last, getattr(wave, 'crestRange', None)), 0])
    xRange = (crestX - 2.0 * d, crestX + 1.2 * d)
    inWindow = (last[:, 0] > xRange[0]) & (last[:, 0] < xRange[1])
    zLow = float(np.min(bathymetry.elevation(np.linspace(*xRange, 50)))) - 0.05 * d
    zHigh = float(np.max(last[inWindow, 1])) + 0.15 * d

    floor = zLow - d
    sandX = np.concatenate([bathymetry.x, [bathymetry.x[-1], bathymetry.x[0]]])
    sandZ = np.concatenate([bathymetry.z, [floor, floor]])
    axes.fill(sandX, sandZ, color = _color('yellow'), zorder = 1)
    axes.plot(bathymetry.x, bathymetry.z, color = _color('orange'), lw = 1.0, zorder = 2)
    axes.axhline(bathymetry.stillWaterLevel, color = _color('textMuted'), lw = 0.8, ls = '--', zorder = 2)
    colors = figurePalette.colormap('ocean')(np.linspace(0.35, 1.0, len(chosen)))
    for frame, color in zip(chosen, colors):
        axes.plot(frame.surface[:, 0], frame.surface[:, 1], color = color, lw = 1.4, zorder = 3,
                  label = f't = {frame.time:.2f} s')
    if point is not None:
        axes.plot([point.crestX], [bathymetry.stillWaterLevel + point.crestHeight], marker = 'v', color = _color('red'),
                  ls = 'none', markersize = 7, zorder = 4, label = 'face vertical')

    axes.set_xlim(*xRange)
    axes.set_ylim(zLow, zHigh)
    axes.set_aspect('equal', adjustable = 'box')
    axes.set_xlabel('Distance [m]')
    axes.set_ylabel('Height [m]')
    axes.legend(loc = 'upper left', fontsize = 8, ncol = 2, framealpha = 0.9)
    title = (f'Overturning profiles   H/d = {wave.waveHeight / d:.2f}, slope 1:{1.0 / wave.beachSlope:.0f}')
    if point is not None:
        title += f'   breaks at hb/d = {point.depth / d:.2f}, Hb/hb = {point.breakingIndex:.2f}'
    axes.set_title(title, color = _color('text'))

    return figure

def drawOverturnShape(wave):

    '''

    The overturn at the last state, measured as Feddersen et al. (2023, 2024) measure it at jet
    impact: the surface at true scale, the overturn boundary enclosing the tube, the Longuet-Higgins
    (1982) curve fitted to it with its length L and width W, the impact height H_I, and the measures
    over H_I^2.

    Returns:
    --------
    matplotlib.figure.Figure or None
        None when the surface does not overhang.

    '''

    solver = getattr(wave, 'solver', None)
    jet = wave.jet if solver is not None and solver.surface is not None else None
    if jet is None:
        return None

    d = wave.depth
    bathymetry = wave.bathymetry
    level = bathymetry.stillWaterLevel
    surface = solver.surface
    highest = crestIndex(surface, getattr(wave, 'crestRange', None))
    impactHeight = float(surface[highest, 1]) - level
    curve = jet.fittedCurve()

    span = jet.overturnLength
    xRange = (min(float(jet.boundary[:, 0].min()), float(surface[highest, 0])) - 0.6 * span,
              float(jet.boundary[:, 0].max()) + 0.6 * span)
    # Down to the bed or two impact heights below still water, whichever is higher, so a deep bed
    # under a small breaker does not shrink it to a speck
    zLow = max(min(level - 0.1 * impactHeight, float(np.min(bathymetry.elevation(np.linspace(*xRange, 50))))),
               level - 2.0 * impactHeight)
    zHigh = float(surface[highest, 1]) + 0.25 * impactHeight

    applyMplStyle()
    figure, axes = plt.subplots(figsize = (10, 5.6), constrained_layout = True)
    floor = zLow - d
    sandX = np.concatenate([bathymetry.x, [bathymetry.x[-1], bathymetry.x[0]]])
    sandZ = np.concatenate([bathymetry.z, [floor, floor]])
    axes.fill(sandX, sandZ, color = _color('yellow'), zorder = 1)
    inside = (bathymetry.x > surface[0, 0]) & (bathymetry.x < surface[-1, 0])
    bedX = np.concatenate([[surface[-1, 0]], bathymetry.x[inside][::-1], [surface[0, 0]]])
    water = np.vstack([surface, np.column_stack([bedX, bathymetry.elevation(bedX)])])
    axes.fill(water[:, 0], water[:, 1], color = _waterColormap()(0.15), zorder = 2)
    axes.axhline(level, color = _color('textMuted'), lw = 0.8, ls = '--', zorder = 3)
    axes.plot(surface[:, 0], surface[:, 1], color = _color('text'), lw = 0.8, zorder = 4)
    axes.plot(jet.boundary[:, 0], jet.boundary[:, 1], color = _color('blue'), lw = 2.2, zorder = 5,
              label = 'overturn boundary')
    axes.plot(curve[:, 0], curve[:, 1], color = _color('red'), lw = 1.4, ls = '--', zorder = 6,
              label = 'fitted Longuet-Higgins curve')

    # The curve's long axis from its back, and its width a third of the way along, where it is widest
    angle = np.radians(jet.overturnAngle)
    along = np.array([np.cos(angle), -np.sin(angle)])
    across = np.array([np.sin(angle), np.cos(angle)])
    back = jet.overturnBack
    axis = np.vstack([back, back + jet.overturnLength * along])
    middle = back + jet.overturnLength / 3.0 * along
    width = np.vstack([middle - 0.5 * jet.overturnWidth * across, middle + 0.5 * jet.overturnWidth * across])
    axes.plot(axis[:, 0], axis[:, 1], color = _color('red'), lw = 0.8, ls = ':', zorder = 6)
    axes.plot(width[:, 0], width[:, 1], color = _color('red'), lw = 0.8, ls = ':', zorder = 6)
    axes.plot([surface[highest, 0]], [surface[highest, 1]], marker = 'D', color = _color('orange'), ls = 'none',
              markersize = 7, zorder = 7, label = 'impact height $H_I$')
    axes.plot([jet.impactPoint[0]], [jet.impactPoint[1]], marker = 'o', color = _color('purple'), ls = 'none',
              markersize = 6, zorder = 7, label = 'impact point')

    text = (f'$H_I/d$ = {impactHeight / d:.3f}\n'
            f'$A_O/H_I^2$ = {jet.cavityArea / impactHeight**2:.3f}\n'
            f'$A_J/H_I^2$ = {jet.jetArea / impactHeight**2:.3f}\n'
            f'$W/L$ = {jet.aspectRatio:.3f}\n'
            f'$\\theta_O$ = {jet.overturnAngle:.1f} deg')
    axes.text(0.99, 0.97, text, transform = axes.transAxes, ha = 'right', va = 'top', fontsize = 9,
              color = _color('text'), bbox = {'facecolor': _color('surface'), 'edgecolor': _color('border'),
                                              'boxstyle': 'round,pad=0.4'}, zorder = 8)
    axes.set_xlim(*xRange)
    axes.set_ylim(zLow, zHigh)
    axes.set_aspect('equal', adjustable = 'box')
    axes.set_xlabel('Distance [m]')
    axes.set_ylabel('Height [m]')
    axes.legend(loc = 'lower left', fontsize = 8, framealpha = 0.9)
    axes.set_title(f'Overturn at jet impact   H/d = {wave.waveHeight / d:.2f}   t = {solver.time:.2f} s',
                   color = _color('text'))

    return figure

def writeOverturningAnimation(wave, path: str, fps: int = 12, dpi: int = 90, frameStep: int = 1,
                              colors: int | None = 64) -> str:

    '''

    Write the recorded transient of an overturning wave as an animated GIF, laid out once, moved frame
    by frame and mapped onto one shared palette.

    Returns:
    --------
    str : The path written

    '''

    from matplotlib.animation import PillowWriter

    indices = list(range(0, len(wave.frames), max(1, int(frameStep))))
    if indices and indices[-1] != len(wave.frames) - 1:
        # The last frame is the jet at touchdown: always include it
        indices.append(len(wave.frames) - 1)
    if not indices:
        raise ValueError('The wave has no recorded frames to animate: run it first.')

    first = overturningFigure(wave, indices[0])
    applyMplStyle()
    figure, (overview, closeUp) = plt.subplots(2, 1, figsize = (11, 6.6), constrained_layout = True,
                                               gridspec_kw = {'height_ratios': [1.0, 1.6]})
    artists = _layoutOverturningFrame(figure, overview, closeUp, first)

    writer = PillowWriter(fps = fps)
    with writer.saving(figure, path, dpi):
        for index in indices:
            _updateOverturningFrame(artists, overturningFigure(wave, index))
            writer.grab_frame()
        # Hold the final frame for a second
        for _ in range(fps):
            writer.grab_frame()
    plt.close(figure)

    if colors:
        _sharePalette(path, colors)

    return path

#----------------------------------------------------------------------#
# -- Reef Break Figures -- #
#----------------------------------------------------------------------#

@dataclass
class ReefFigure:

    '''

    One instant of regular swell over a reef from the SGN model: the water column colored by
    depth-averaged speed and whitened where a front is breaking, the whole profile above and a
    true-scale close-up of the reef edge below.

    '''

    time: float                                             # [s]
    x: np.ndarray                                           # [m], cell centers
    cellSize: float                                         # [m]
    surface: np.ndarray                                     # [m]
    speeds: np.ndarray                                      # [m/s]
    breakingZone: np.ndarray
    bedX: np.ndarray                                        # [m]
    bedZ: np.ndarray                                        # [m]
    stillWaterLevel: float                                  # [m]
    depth: float                                            # [m], offshore still water depth
    waveHeight: float                                       # [m]
    period: float                                           # [s]
    windowX: tuple                                          # [m]
    windowZ: tuple                                          # [m]
    viewX: tuple                                            # [m]
    speedScale: float                                       # [m/s]
    breakingStarts: np.ndarray                              # [m], where fronts started breaking so far
    breakingFronts: int
    title: str = 'Swell breaking on a reef, Serre-Green-Naghdi'

def reefFigure(reef, frameIndex: int = -1):

    '''

    Description of one recorded frame of a reef break, or None when it has no frames yet.

    The close-up spans three offshore depths seaward of the top of the reef face to four depths
    over the reef flat, at true scale.

    '''

    if not getattr(reef, 'frames', None):
        return None

    frame = reef.frames[frameIndex]
    bathymetry, d = reef.bathymetry, reef.depth
    edge = float(bathymetry.x[2])
    windowX = (edge - 3.0 * d, edge + 4.0 * d)
    zLow = float(np.min(bathymetry.elevation(np.linspace(*windowX, 100)))) - 0.05 * d
    zHigh = bathymetry.stillWaterLevel + 1.5 * reef.waveHeight + 0.1 * d
    starts = np.array([event.x for event in reef.events if event.kind == 'start' and event.time <= frame.time])

    return ReefFigure(time = frame.time, x = reef.solver.x, cellSize = reef.solver.cellSize, surface = frame.surface,
                      speeds = np.abs(frame.velocity), breakingZone = frame.breakingZone, bedX = bathymetry.x,
                      bedZ = bathymetry.z, stillWaterLevel = bathymetry.stillWaterLevel, depth = d,
                      waveHeight = reef.waveHeight, period = reef.period, windowX = windowX, windowZ = (zLow, zHigh),
                      viewX = (reef.spongeWidth, bathymetry.endX),
                      speedScale = 0.6 * float(np.sqrt(constants.gravity * d)), breakingStarts = starts,
                      breakingFronts = int(np.count_nonzero(np.diff(frame.breakingZone.astype(int)) == 1)))

def _reefCaption(data: ReefFigure) -> str:

    '''The status line: time in seconds and periods, and the fronts breaking.'''

    return (f't = {data.time:6.1f} s   ({data.time / data.period:5.1f} periods)   breaking fronts: {data.breakingFronts}'
            f'   breaking starts so far: {len(data.breakingStarts)}')

def _layoutReefFrame(figure, overview, closeUp, data: ReefFigure) -> dict:

    '''The parts of a reef frame that stay put, and handles on the parts that move.'''

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch

    d = data.depth
    zRange = (-0.1 * d, data.stillWaterLevel + 2.0 * data.waveHeight + 0.1 * d)
    overviewHandles = _drawSurfaceAxes(overview, data, data.viewX, zRange, max(1, int(len(data.x) / 600)))
    overview.set_ylabel('Height [m]')
    overview.set_title('Reef profile (vertical scale stretched)')
    starts, = overview.plot([], [], marker = '|', color = _color('red'), markersize = 9, ls = 'none', zorder = 7,
                            alpha = 0.6)

    closeHandles = _drawSurfaceAxes(closeUp, data, data.windowX, data.windowZ, 1)
    closeUp.set_aspect('equal', adjustable = 'box')
    closeUp.set_xlabel('Distance [m]')
    closeUp.set_ylabel('Height [m]')
    closeUp.set_title('Reef edge, true scale')
    foam = _waterColors(np.zeros(1), np.ones(1, dtype = bool), data.speedScale)[0]
    closeUp.legend(handles = [Patch(facecolor = foam, edgecolor = _color('border'), label = 'Breaking front (bore)')],
                   loc = 'upper right', fontsize = 8, framealpha = 0.9)

    window = plt.Rectangle((data.windowX[0], data.windowZ[0]), data.windowX[1] - data.windowX[0],
                           data.windowZ[1] - data.windowZ[0], fill = False, edgecolor = _color('accentText'),
                           lw = 1.0, zorder = 6)
    overview.add_patch(window)
    colorbar = figure.colorbar(ScalarMappable(norm = Normalize(0.0, data.speedScale), cmap = _waterColormap()),
                               ax = [overview, closeUp], location = 'right', shrink = 0.8, pad = 0.01)
    colorbar.set_label('Depth-averaged speed [m/s]')
    figure.suptitle(f'{data.title}   H = {data.waveHeight:.1f} m, T = {data.period:.0f} s, offshore depth {d:.0f} m',
                    color = _color('text'), fontsize = 11)
    caption = overview.text(0.01, 0.95, '', transform = overview.transAxes, va = 'top', ha = 'left', fontsize = 9,
                            color = _color('text'), zorder = 8)

    return {'overview': overviewHandles, 'closeUp': closeHandles, 'starts': starts, 'caption': caption,
            'startsHeight': zRange[1] - 0.05 * d}

def _updateReefFrame(artists: dict, data: ReefFigure) -> None:

    '''Move the water, the breaking-start ticks and the caption to this frame.'''

    _moveSurfaceAxes(artists['overview'], data)
    _moveSurfaceAxes(artists['closeUp'], data)
    artists['starts'].set_data(data.breakingStarts, np.full(len(data.breakingStarts), artists['startsHeight']))
    artists['caption'].set_text(_reefCaption(data))

def drawReefFrame(reef, frameIndex: int = -1):

    '''One recorded frame of a reef break: the whole profile above and a true-scale close-up of the reef edge below.'''

    data = reefFigure(reef, frameIndex)
    if data is None:
        return None

    applyMplStyle()
    figure, (overview, closeUp) = plt.subplots(2, 1, figsize = (11, 6.0), constrained_layout = True,
                                               gridspec_kw = {'height_ratios': [1.0, 1.3]})
    artists = _layoutReefFrame(figure, overview, closeUp, data)
    _updateReefFrame(artists, data)

    return figure

def writeReefAnimation(reef, path: str, fps: int = 12, dpi: int = 90, frameStep: int = 1, lastPeriods: float | None = None,
                       colors: int | None = 64) -> str:

    '''

    Write a reef break as an animated GIF, laid out once, moved frame by frame and mapped onto one
    shared palette.

    Parameters:
    -----------
    lastPeriods : float | None
        Animate only the last this many wave periods; all frames by default

    Returns:
    --------
    str : The path written

    '''

    from matplotlib.animation import PillowWriter

    frames = reef.frames
    startTime = frames[-1].time - lastPeriods * reef.period if lastPeriods else -np.inf
    indices = [i for i in range(0, len(frames), max(1, int(frameStep))) if frames[i].time >= startTime]
    if not indices:
        raise ValueError('The reef has no recorded frames to animate: run it first.')

    first = reefFigure(reef, indices[0])
    applyMplStyle()
    figure, (overview, closeUp) = plt.subplots(2, 1, figsize = (11, 6.0), constrained_layout = True,
                                               gridspec_kw = {'height_ratios': [1.0, 1.3]})
    artists = _layoutReefFrame(figure, overview, closeUp, first)

    writer = PillowWriter(fps = fps)
    with writer.saving(figure, path, dpi):
        for index in indices:
            _updateReefFrame(artists, reefFigure(reef, index))
            writer.grab_frame()
    plt.close(figure)

    if colors:
        _sharePalette(path, colors)

    return path

#----------------------------------------------------------------------#
# -- Splash-Up Figures -- #
#----------------------------------------------------------------------#

@dataclass
class SplashFigure:

    '''

    One instant of the SPH splash-up: the water as particles colored by speed over the window's
    bed, the piston and the end wall, and a true-scale close-up that travels with the breaker.

    '''

    time: float                                             # [s]
    positions: np.ndarray                                   # [m], fluid particles, shape (N, 2)
    speeds: np.ndarray                                      # [m/s]
    bedX: np.ndarray                                        # [m], window bed vertices
    bedZ: np.ndarray                                        # [m]
    stillWaterLevel: float                                  # [m]
    depth: float                                            # [m], offshore still water depth
    waveHeight: float                                       # [m], incident wave height
    spacing: float                                          # [m], particle spacing
    handoffTime: float                                      # [s]
    pistonX: float                                          # [m]
    frontX: float                                           # [m], the shoreward wall
    windowX: tuple = (0.0, 1.0)                             # [m], close-up span
    windowZ: tuple = (0.0, 1.0)                             # [m], close-up height range
    viewX: tuple = (0.0, 1.0)                               # [m], overview span
    viewZ: tuple = (0.0, 1.0)                               # [m], overview height range
    speedScale: float = 1.0                                 # [m/s], top of the color scale
    title: str = 'Splash-up: the potential flow overturn handed to SPH at the jet\'s impact'

def splashFigure(splash, frameIndex: int = -1, windowDepths: float = 3.5):

    '''

    Description of one recorded frame of a splash-up, or None when it has no frames yet. The
    close-up starts a quarter depth behind where the jet lands and travels with the wave at its
    crest speed.

    '''

    if not getattr(splash, 'frames', None):
        return None

    frame = splash.frames[frameIndex]
    d, bed = splash.depth, splash.bathymetry
    level = bed.stillWaterLevel
    height = splash.wave.waveHeight
    elapsed = frame.time - splash.handoffTime
    width = windowDepths * d
    start = float(np.clip(splash.impactX - 1.5 * d + splash.crestSpeed * elapsed, bed.startX, bed.endX - width))
    floor = float(np.min(bed.elevation(np.linspace(start, start + width, 30)))) - 0.05 * d
    zSpan = width * 0.42
    viewFloor = float(np.min(bed.z)) - 0.05 * d

    title = ('Splash-up: the potential flow overturn handed to SPH at the jet\'s impact' if splash.frameIndex is None
             else 'Splash-up: the potential flow overturn handed to SPH before the jet lands')

    return SplashFigure(time = frame.time, positions = frame.positions, speeds = frame.speeds, bedX = bed.x,
                        bedZ = bed.z, stillWaterLevel = level, depth = d, waveHeight = height, spacing = splash.spacing,
                        handoffTime = splash.handoffTime, pistonX = frame.pistonX, frontX = bed.endX,
                        windowX = (start, start + width), windowZ = (floor, floor + zSpan),
                        viewX = (bed.startX - 0.1 * d, bed.endX + 0.1 * d),
                        viewZ = (viewFloor, level + 2.0 * height), speedScale = 1.6 * math.sqrt(constants.gravity * d),
                        title = title)

def _splashCaption(data: SplashFigure) -> str:

    '''The status line: time on the wave's clock, the same over sqrt(d / g), and time since the hand-off.'''

    scale = math.sqrt(data.depth / constants.gravity)

    return (f't = {data.time:6.3f} s   (t sqrt(g/d) = {data.time / scale:5.2f})   '
            f'{data.time - data.handoffTime:5.3f} s after the hand-off to SPH')

def _layoutSplashFrame(figure, overview, closeUp, data: SplashFigure) -> dict:

    '''The parts of a splash-up frame that stay put, and handles on the parts that move.'''

    overviewScatter = _drawWaveAxes(overview, data, data.viewX, data.viewZ, 1.0)
    overview.set_ylabel('Height [m]')
    overview.set_title('SPH window (vertical scale stretched)')
    piston = overview.axvline(data.pistonX, color = _color('purple'), lw = 1.6, zorder = 5, label = 'piston')
    overview.axvline(data.frontX, color = _color('textMuted'), lw = 1.6, zorder = 5, label = 'wall')
    overview.legend(loc = 'upper right', fontsize = 8, framealpha = 0.9)

    closeScatter = _drawWaveAxes(closeUp, data, data.windowX, data.windowZ, 6.0)
    closeUp.set_aspect('equal', adjustable = 'box')
    closeUp.set_xlabel('Distance [m]')
    closeUp.set_ylabel('Height [m]')
    closeUp.set_title('Breaker, true scale')

    window = plt.Rectangle((data.windowX[0], data.windowZ[0]), data.windowX[1] - data.windowX[0],
                           data.windowZ[1] - data.windowZ[0], fill = False, edgecolor = _color('accentText'),
                           lw = 1.0, zorder = 4)
    overview.add_patch(window)
    colorbar = figure.colorbar(closeScatter, ax = [overview, closeUp], location = 'right', shrink = 0.8, pad = 0.01)
    colorbar.set_label('Particle speed [m/s]')
    figure.suptitle(data.title, color = _color('text'), fontsize = 11)
    caption = closeUp.text(0.01, 0.97, '', transform = closeUp.transAxes, va = 'top', ha = 'left', fontsize = 9,
                           color = _color('text'))

    # Marker sizes depend on the settled axes boxes, which exist only after a layout pass
    figure.canvas.draw()
    overviewScatter.set_sizes([_particleMarkerArea(overview, data.spacing, data.viewX[1] - data.viewX[0],
                                                   data.viewZ[1] - data.viewZ[0])])

    return {'overview': overviewScatter, 'closeUp': closeScatter, 'closeAxes': closeUp, 'window': window,
            'piston': piston, 'caption': caption}

def _updateSplashFrame(artists: dict, data: SplashFigure) -> None:

    '''Move the particles, the piston, the close-up window and the caption to this frame.'''

    for key in ('overview', 'closeUp'):
        artists[key].set_offsets(data.positions)
        artists[key].set_array(data.speeds)

    closeAxes = artists['closeAxes']
    closeAxes.set_xlim(*data.windowX)
    closeAxes.set_ylim(*data.windowZ)
    width, height = data.windowX[1] - data.windowX[0], data.windowZ[1] - data.windowZ[0]
    artists['closeUp'].set_sizes([_particleMarkerArea(closeAxes, data.spacing, width, height)])
    artists['window'].set_xy((data.windowX[0], data.windowZ[0]))
    artists['window'].set_width(width)
    artists['window'].set_height(height)
    artists['piston'].set_xdata([data.pistonX, data.pistonX])
    artists['caption'].set_text(_splashCaption(data))

def drawSplashFrame(splash, frameIndex: int = -1):

    '''

    One recorded frame of a splash-up: the SPH window above, stretched vertically, and a true-scale
    close-up of the breaker below.

    Returns:
    --------
    matplotlib.figure.Figure or None
        None when the splash-up has no frames.

    '''

    data = splashFigure(splash, frameIndex)
    if data is None:
        return None

    applyMplStyle()
    figure, (overview, closeUp) = plt.subplots(2, 1, figsize = (11, 6.6), constrained_layout = True,
                                               gridspec_kw = {'height_ratios': [1.0, 1.6]})
    artists = _layoutSplashFrame(figure, overview, closeUp, data)
    _updateSplashFrame(artists, data)

    return figure

def writeSplashAnimation(splash, path: str, fps: int = 12, dpi: int = 90, frameStep: int = 1,
                         colors: int | None = 64) -> str:

    '''

    Write the recorded splash-up as an animated GIF, laid out once and moved frame by frame, with
    one shared palette.

    Returns:
    --------
    str : The path written

    '''

    from matplotlib.animation import PillowWriter

    indices = list(range(0, len(splash.frames), max(1, int(frameStep))))
    if not indices:
        raise ValueError('The splash-up has no recorded frames to animate: run it first.')

    applyMplStyle()
    figure, (overview, closeUp) = plt.subplots(2, 1, figsize = (11, 6.6), constrained_layout = True,
                                               gridspec_kw = {'height_ratios': [1.0, 1.6]})
    artists = _layoutSplashFrame(figure, overview, closeUp, splashFigure(splash, indices[0]))

    writer = PillowWriter(fps = fps)
    with writer.saving(figure, path, dpi):
        for index in indices:
            _updateSplashFrame(artists, splashFigure(splash, index))
            writer.grab_frame()
        # Hold the last frame for a second, so the loop does not jump straight back
        for _ in range(fps):
            writer.grab_frame()
    plt.close(figure)

    if colors:
        _sharePalette(path, colors)

    return path

#----------------------------------------------------------------------#
# -- Export -- #
#----------------------------------------------------------------------#

def writeInteractiveFigure(figure, path: str) -> str:

    '''

    Write a plotly figure to a standalone HTML file without opening a browser tab.

    Returns the path written, or None when there was no figure to write.

    '''

    if figure is None:
        return None
    # 'directory' drops one shared plotly.min.js beside the HTML instead of inlining a copy into
    # every file. The files are only portable as a folder, which is how an export is written.
    _writePlotlyFigure(figure, filename = path, auto_open = False, include_plotlyjs = 'directory')
    return path

# Output file name -> builder taking a Surfboard and returning a plotly figure or None
interactiveFigureBuilders = {
    'boardProfilesInteractive.html': lambda board: plotlyBoardProfiles(boardProfileFigure(board)),
    'crossSectionsInteractive.html': lambda board: plotlyCrossSections(crossSectionFigure(board)),
    'boardSurfaceView.html':         lambda board: boardSurfaceFigure(board),
}

def exportInteractiveFigures(board, folder: str) -> list:

    '''

    Write the interactive HTML companion for every board figure that has data.

    A builder that raises is skipped with a note rather than sinking the export.

    Parameters:
    -----------
    board : pysurf.Surfboard
        A board with at least its profiles generated
    folder : str
        Destination directory, created if absent

    Returns:
    --------
    list : paths of the files written

    '''

    os.makedirs(folder, exist_ok = True)
    written = []
    for name, builder in interactiveFigureBuilders.items():
        try:
            figure = builder(board)
        except Exception as error:                  # noqa: BLE001 -- one bad figure must not sink the export
            print(f'Skipping interactive view {name}: {type(error).__name__}: {error}')
            continue
        path = writeInteractiveFigure(figure, os.path.join(folder, name))
        if path is not None:
            written.append(path)
    if written:
        print(f'Saving {len(written)} interactive figure(s) to .html')
    return written
