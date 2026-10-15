
# -- pySurf: Figure Palette -- #

'''

The colors every pySurf figure is drawn in: the Surfy Pastels palette, in a light and a dark mode.

Light is the default and the palette's signature: sun-bleached pastels on warm sand, with seafoam
turquoise as the accent. Dark is the same ocean at night, a deep teal-navy ground with the same
coral, sand and seafoam family brightened to hold up against it.

The palette names nineteen colors per mode. Four are backgrounds (`bg`, `surface`, `surface2`,
`border`), three are text weights, three carry the accent, two carry accent text (`accentText` for
labels and thin strokes in the accent hue, `onAccent` for text on an accent fill), and seven are
status colors whose meaning is fixed in both modes: `green` is pass and complete only, `red` error,
`yellow` warning, `blue` information, `orange` attention, `purple` analysis outputs, `cyan` code
and paths. Plots draw categories from `categories`, which starts on green by the palette's own
rule.

The pastels are fill colors. Light seafoam on sand is 2.0:1, under the 3:1 a graphical stroke
needs, so a curve that has to be read is drawn in `accentText` (deep seafoam in light mode, the
accent itself in dark) and filled in `accent`. The status colors mark points, fills and bands;
the meaning a label carries goes in `text`.

Three continuous colormaps go with each mode, built only from the palette's own colors:

    ocean       seafoam, light to deep: one cool hue whose lightness runs monotonically, for a
                magnitude such as speed or depth, and for stepping through stations along a board
    sunset      sand-gold through sunset to coral, for height and thickness
    deviation   sky through sand to coral, diverging about zero, for a signed difference

Figures are drawn in one mode at a time. It is light unless `setFigureMode('dark')` is called or
the environment variable `PYSURF_FIGURE_MODE` is `dark`. Matplotlib is imported only by the
helpers that need it.

Sean Bowman [10/10/2026]

'''

# Standard library imports
import os

modes = ('light', 'dark')

palettes = {
    'light': {
        'bg':          '#F7F1E6',   # warm sand
        'surface':     '#FFFCF6',
        'surface2':    '#EFE4CE',
        'border':      '#E1D3B8',
        'text':        '#3A3F4B',
        'textMuted':   '#5F6673',
        'textDim':     '#9CA3AF',   # decorative only: about 2.3:1 on sand
        'accent':      '#4FBDB3',   # seafoam turquoise
        'accentDim':   '#3D9C93',
        'accentMuted': '#D8EFEC',
        'accentText':  '#237068',   # seafoam deepened for labels and strokes
        'onAccent':    '#16232A',
        'green':       '#8FBF7F',   # palm
        'red':         '#E8776B',   # coral
        'yellow':      '#F0C36D',   # sand-gold
        'blue':        '#7FB8D9',   # sky
        'orange':      '#F2A65A',   # sunset
        'purple':      '#B79FD1',   # orchid
        'cyan':        '#6FD1C7',   # aqua
    },
    'dark': {
        'bg':          '#16232A',   # night ocean
        'surface':     '#1D2E36',
        'surface2':    '#263C46',
        'border':      '#35505C',
        'text':        '#E8F0EC',
        'textMuted':   '#9FB3AE',
        'textDim':     '#6B7F7A',
        'accent':      '#5FD4C7',   # brightened seafoam
        'accentDim':   '#47AFA3',
        'accentMuted': '#1E3B39',
        'accentText':  '#5FD4C7',   # the dark accent already clears 9:1
        'onAccent':    '#16232A',
        'green':       '#9BCB86',
        'red':         '#F08A7C',
        'yellow':      '#F2CE82',
        'blue':        '#8FC4E3',
        'orange':      '#F5B072',
        'purple':      '#C6ACDE',
        'cyan':        '#7EDDD0',
    },
}

# Color stops for each colormap, low value first. Every stop is a palette color of its own mode.
colormapStops = {
    'light': {
        'ocean':     ('#D8EFEC', '#6FD1C7', '#4FBDB3', '#3D9C93', '#237068'),
        'sunset':    ('#F0C36D', '#F2A65A', '#E8776B'),
        'deviation': ('#7FB8D9', '#EFE4CE', '#E8776B'),
    },
    'dark': {
        'ocean':     ('#1E3B39', '#47AFA3', '#5FD4C7', '#7EDDD0', '#E8F0EC'),
        'sunset':    ('#F08A7C', '#F5B072', '#F2CE82'),
        'deviation': ('#8FC4E3', '#263C46', '#F08A7C'),
    },
}

_figureMode = os.environ.get('PYSURF_FIGURE_MODE', 'light').strip().lower()
if _figureMode not in modes:
    _figureMode = 'light'

def setFigureMode(mode: str) -> None:

    '''

    Choose the mode every figure drawn after this call uses.

    Parameters:
    -----------
    mode : str
        'light' or 'dark'

    Raises:
    -------
    ValueError : If mode is not one of the two

    '''

    global _figureMode

    if mode not in modes:
        raise ValueError(f'Figure mode must be one of {", ".join(modes)}, not {mode!r}.')
    _figureMode = mode

def figureMode() -> str:

    '''The mode figures are drawn in.'''

    return _figureMode

def colors(mode: str = None) -> dict:

    '''

    The nineteen named colors of a mode, the active figure mode by default.

    '''

    return palettes[mode or _figureMode]

def categories(mode: str = None) -> list:

    '''The categorical color sequence plots draw series from, in the palette's order.'''

    named = colors(mode)

    return [named[key] for key in ('green', 'accent', 'red', 'yellow', 'blue', 'orange', 'purple', 'cyan')]

def colormap(name: str, mode: str = None):

    '''

    A colormap as a Matplotlib colormap.

    Parameters:
    -----------
    name : str
        'ocean', 'sunset' or 'deviation'
    mode : str
        'light' or 'dark'; the active figure mode by default

    Returns:
    --------
    matplotlib.colors.LinearSegmentedColormap

    '''

    from matplotlib.colors import LinearSegmentedColormap

    mode = mode or _figureMode

    return LinearSegmentedColormap.from_list(f'surf{name.capitalize()}{mode.capitalize()}',
                                             list(colormapStops[mode][name]))

def plotlyColorscale(name: str, mode: str = None) -> list:

    '''

    A colormap as a plotly colorscale, `[[position, color], ...]`.

    A name that is not one of the palette's colormaps is handed back unchanged, so a plotly
    colorscale name still works where one is passed.

    '''

    mode = mode or _figureMode
    if name not in colormapStops[mode]:
        return name
    stops = colormapStops[mode][name]

    return [[index / (len(stops) - 1), color] for index, color in enumerate(stops)]

def matplotlibStyle(mode: str = None) -> dict:

    '''

    Matplotlib rcParams for a mode: page and panel backgrounds, text, edges, grid and the
    categorical cycle.

    '''

    from cycler import cycler

    named = colors(mode)

    return {
        'figure.facecolor':  named['bg'],
        'figure.edgecolor':  named['bg'],
        'savefig.facecolor': named['bg'],
        'axes.facecolor':    named['surface'],
        'axes.edgecolor':    named['border'],
        'axes.labelcolor':   named['text'],
        'axes.titlecolor':   named['text'],
        'text.color':        named['text'],
        'xtick.color':       named['textMuted'],
        'ytick.color':       named['textMuted'],
        'grid.color':        named['border'],
        'lines.color':       named['accentText'],
        'patch.edgecolor':   named['border'],
        'legend.facecolor':  named['surface'],
        'legend.edgecolor':  named['border'],
        'axes.prop_cycle':   cycler(color = categories(mode)),
    }

def plotlyLayout(mode: str = None) -> dict:

    '''

    Keyword arguments for a plotly `update_layout` call in a mode: backgrounds, font, grid and
    the categorical colorway.

    '''

    named = colors(mode)
    axis = {'gridcolor': named['border'], 'linecolor': named['border'], 'zerolinecolor': named['border']}

    return {
        'paper_bgcolor': named['bg'],
        'plot_bgcolor':  named['surface'],
        'font':          {'color': named['text']},
        'colorway':      categories(mode),
        'xaxis':         axis,
        'yaxis':         dict(axis),
    }
