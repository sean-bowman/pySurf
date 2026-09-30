
# -- Reference Comparison Viewer -- #

'''
Interactive Plotly Dash viewer for comparing generated surfboard geometry
against a reference STL. Supports real-time Python parametric mesh updates
via sliders.

Features:
  - Reference board as grey semi-transparent 3D overlay
  - Python parametric board with per-vertex deviation coloring
    (red = outside reference, green = inside reference)
  - Toggle visibility of each board independently
  - Sliders for all SurfboardParameters values
  - RMS deviation display

Usage:
  from pysurf.Surfboard.SurfPhysics.visualization.referenceComparisonViewer import launchViewer
  launchViewer(referencePath, pythonParams)

Sean Bowman [02/26/2026]
'''

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    import dash
    from dash import dcc, html, Input, Output
    import plotly.graph_objects as go
    _dashAvailable = True
except ImportError:
    _dashAvailable = False

from pysurf.Surfboard.SurfPhysics.geometry.parameters import SurfboardParameters
from pysurf.Surfboard.SurfPhysics.geometry.surfaceMeshGenerator import SurfaceMeshGenerator
from pysurf.Surfboard.SurfPhysics.optimization.coordinateTransformer import CoordinateTransformer


#--------------------------------------------------------------------#
# -- Module-Level State (shared across Dash callbacks) -- #
#--------------------------------------------------------------------#

# Cached reference mesh and proximity query (loaded once at startup)
_referenceMesh: Optional['trimesh.Trimesh'] = None
_referenceQuery = None   # trimesh.proximity.ProximityQuery


#--------------------------------------------------------------------#
# -- Mesh Conversion Helpers -- #
#--------------------------------------------------------------------#

def _meshToTrace(
    mesh: 'trimesh.Trimesh',
    name: str,
    color: str = 'lightblue',
    opacity: float = 0.8,
    intensities: Optional[np.ndarray] = None,
    colorscale: Optional[list] = None,
    showscale: bool = False,
    cmin: float = -20.0,
    cmax: float = 20.0,
) -> go.Mesh3d:
    '''
    Convert a trimesh.Trimesh to a Plotly Mesh3d trace.

    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Input mesh (vertices in mm)
    name : str
        Legend label for the trace
    color : str
        Flat color (used if intensities is None)
    opacity : float
        Mesh transparency (0 = invisible, 1 = opaque)
    intensities : np.ndarray | None
        Per-vertex scalar values for deviation coloring
    colorscale : list | None
        Plotly colorscale for deviation coloring
    showscale : bool
        Whether to display the colorbar
    cmin : float
        Minimum intensity value for colorscale mapping (mm)
    cmax : float
        Maximum intensity value for colorscale mapping (mm)

    Returns:
    --------
    go.Mesh3d : Plotly 3D mesh trace
    '''
    verts = mesh.vertices
    faces = mesh.faces

    if intensities is not None:
        return go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=intensities,
            colorscale=colorscale or _deviationColorscale(),
            cmin=cmin,
            cmax=cmax,
            showscale=showscale,
            colorbar=dict(title='Deviation (mm)', thickness=15) if showscale else None,
            opacity=opacity,
            name=name,
            hovertemplate='%{name}<extra></extra>',
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2),
        )
    else:
        return go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color,
            opacity=opacity,
            name=name,
            hovertemplate='%{name}<extra></extra>',
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2),
        )


def _deviationColorscale() -> list:
    '''
    Returns a diverging green-white-red colorscale for deviation display.

    Green = inside reference (generated too small, negative)
    White = on reference surface (zero deviation)
    Red   = outside reference (generated too large, positive)

    Returns:
    --------
    list : Plotly colorscale definition
    '''
    return [
        [0.0, '#1a7a1a'],    # dark green (inside ref, negative)
        [0.35, '#66cc66'],   # light green
        [0.5, '#f5f5f5'],    # near-white (on surface)
        [0.65, '#cc6666'],   # light red
        [1.0, '#7a1a1a'],    # dark red (outside ref, positive)
    ]


def _computeDeviations(
    generatedMesh: 'trimesh.Trimesh',
    referenceQuery: 'trimesh.proximity.ProximityQuery',
    refMesh: 'trimesh.Trimesh',
) -> tuple[np.ndarray, float]:
    '''
    Compute per-vertex signed distances from generated mesh to reference.

    Positive = vertex is outside reference (generated too large, red)
    Negative = vertex is inside reference (generated too small, green)

    Parameters:
    -----------
    generatedMesh : trimesh.Trimesh
        The generated board mesh
    referenceQuery : trimesh.proximity.ProximityQuery
        Pre-built proximity query on reference mesh
    refMesh : trimesh.Trimesh
        Reference mesh (needed for sign determination)

    Returns:
    --------
    tuple : (per-vertex signed distances in mm, RMS deviation in mm)
    '''
    vertices = generatedMesh.vertices  # (N, 3) in mm

    # Nearest point on reference surface to each generated vertex
    closestPoints, distances, _ = referenceQuery.on_surface(vertices)

    # Determine sign: is the vertex inside or outside the reference?
    # Use trimesh.ray for inside/outside test on a subsample for speed
    try:
        containment = refMesh.contains(vertices)
        signedDistances = np.where(containment, -distances, distances)
    except Exception:
        # Fallback: use unsigned distances if containment test fails
        signedDistances = distances

    rmsMm = float(np.sqrt(np.mean(signedDistances ** 2)))
    return signedDistances, rmsMm


def _loadAndAlignReference(stlPath: str) -> 'trimesh.Trimesh':
    '''
    Load the reference STL and align it to the project coordinate system.

    Applies auto-detected scale and axis remapping via CoordinateTransformer,
    then re-centers so the nose is at X=0.

    Parameters:
    -----------
    stlPath : str
        Path to reference STL file

    Returns:
    --------
    trimesh.Trimesh : Aligned reference mesh in mm
    '''
    rawMesh = trimesh.load(str(stlPath))

    scaleFactor, axisMapping = CoordinateTransformer.autoDetect(
        rawMesh, expectedLengthMm=1828.0
    )

    # Apply axis remapping then scale
    verts = rawMesh.vertices.copy()
    axisMap = {'x': 0, 'y': 1, 'z': 2}
    if len(axisMapping) == 3:
        newVerts = np.zeros_like(verts)
        for newAxis, oldAxisChar in enumerate(axisMapping):
            newVerts[:, newAxis] = verts[:, axisMap[oldAxisChar]]
        verts = newVerts
    verts *= scaleFactor

    aligned = rawMesh.copy()
    aligned.vertices = verts

    # Repair if needed
    if not aligned.is_watertight:
        trimesh.repair.fill_holes(aligned)
        trimesh.repair.fix_normals(aligned)

    return aligned


def _paramsFromSliders(sliderValues: dict) -> SurfboardParameters:
    '''
    Build a SurfboardParameters from slider value dictionary.

    Parameters:
    -----------
    sliderValues : dict
        Mapping of parameter name to value

    Returns:
    --------
    SurfboardParameters : Board parameters
    '''
    return SurfboardParameters(
        length=sliderValues['length'],
        maxWidth=sliderValues['maxWidth'],
        maxThickness=sliderValues['maxThickness'],
        noseWidth=sliderValues['noseWidth'],
        tailWidth=sliderValues['tailWidth'],
        widePointOffset=sliderValues['widePointOffset'],
        tailTipHalfWidth=sliderValues.get('tailTipHalfWidth', 15.0),
        noseRocker=sliderValues['noseRocker'],
        tailRocker=sliderValues['tailRocker'],
        deckCrown=sliderValues['deckCrown'],
        bottomConcave=sliderValues['bottomConcave'],
    )


def _generatePythonMeshTrace(
    params: SurfboardParameters,
    visible: bool,
) -> tuple[go.Mesh3d, float]:
    '''
    Generate a Python parametric mesh and compute its deviation trace.

    Parameters:
    -----------
    params : SurfboardParameters
        Board parameters
    visible : bool
        Whether to show this trace

    Returns:
    --------
    tuple : (Mesh3d trace, RMS deviation in mm)
    '''
    global _referenceMesh, _referenceQuery

    gen = SurfaceMeshGenerator.fromPreset(params, 'draft')
    mesh = gen.generate()

    if _referenceQuery is not None and _referenceMesh is not None:
        signedDistances, rmsMm = _computeDeviations(mesh, _referenceQuery, _referenceMesh)
        devRange = max(20.0, float(np.percentile(np.abs(signedDistances), 95)))
        trace = _meshToTrace(
            mesh,
            name='Python Parametric',
            intensities=signedDistances,
            colorscale=_deviationColorscale(),
            showscale=True,
            cmin=-devRange,
            cmax=devRange,
            opacity=0.85,
        )
    else:
        rmsMm = 0.0
        trace = _meshToTrace(mesh, name='Python Parametric', color='#4488cc', opacity=0.85)

    if not visible:
        trace.visible = False

    return trace, rmsMm


def _referenceTrace(visible: bool = True) -> go.Mesh3d:
    '''
    Build the reference mesh Plotly trace (grey, semi-transparent).

    Parameters:
    -----------
    visible : bool
        Whether to show this trace

    Returns:
    --------
    go.Mesh3d : Reference mesh trace
    '''
    global _referenceMesh
    trace = _meshToTrace(
        _referenceMesh,
        name='Reference STL',
        color='#aaaaaa',
        opacity=0.35,
    )
    if not visible:
        trace.visible = False
    return trace


def _buildFigure(
    traces: list,
    boardLengthMm: float = 1828.0,
) -> go.Figure:
    '''
    Assemble a Plotly figure from mesh traces with a fixed 3D camera.

    Parameters:
    -----------
    traces : list
        List of go.Mesh3d traces
    boardLengthMm : float
        Board length in mm (used to set axis ranges)

    Returns:
    --------
    go.Figure : Complete Plotly figure
    '''
    fig = go.Figure(data=traces)

    halfLen = boardLengthMm / 2.0
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Length (mm)', range=[0, boardLengthMm]),
            yaxis=dict(title='Width (mm)', range=[-350, 350]),
            zaxis=dict(title='Thickness (mm)', range=[-20, 100]),
            aspectmode='data',
            bgcolor='#1a1a2e',
            camera=dict(
                eye=dict(x=-0.5, y=-2.0, z=0.8),
                center=dict(x=0, y=0, z=0),
            ),
        ),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        font=dict(color='#e0e0e0'),
        legend=dict(
            bgcolor='rgba(30,30,50,0.8)',
            bordercolor='#444',
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True,
        height=600,
    )
    return fig


#--------------------------------------------------------------------#
# -- Slider Configuration -- #
#--------------------------------------------------------------------#

_SLIDER_CONFIG = [
    dict(id='length',          label='Length (mm)',          min=1500, max=3000, step=10,  markStep=250),
    dict(id='maxWidth',        label='Max Width (mm)',        min=400,  max=650,  step=5,   markStep=50),
    dict(id='maxThickness',    label='Max Thickness (mm)',    min=45,   max=100,  step=1,   markStep=10),
    dict(id='noseWidth',       label='Nose Width (mm)',       min=200,  max=500,  step=5,   markStep=50),
    dict(id='tailWidth',       label='Tail Width (mm)',       min=200,  max=550,  step=5,   markStep=50),
    dict(id='widePointOffset', label='Wide Point Offset (mm)',min=-200, max=200,  step=5,   markStep=50),
    dict(id='noseRocker',      label='Nose Rocker (mm)',      min=40,   max=250,  step=5,   markStep=50),
    dict(id='tailRocker',      label='Tail Rocker (mm)',      min=10,   max=100,  step=2,   markStep=20),
    dict(id='deckCrown',       label='Deck Crown (mm)',       min=0,    max=30,   step=0.5, markStep=5),
    dict(id='bottomConcave',   label='Bottom Concave (mm)',   min=0,    max=15,   step=0.5, markStep=3),
]


def _makeSlider(cfg: dict, value: float) -> html.Div:
    '''
    Build a labelled Dash slider component.

    Parameters:
    -----------
    cfg : dict
        Slider configuration dict (id, label, min, max, step, markStep)
    value : float
        Initial slider value

    Returns:
    --------
    html.Div : Slider with label and value display
    '''
    marks = {
        int(v) if v == int(v) else v: str(int(v) if v == int(v) else v)
        for v in np.arange(cfg['min'], cfg['max'] + cfg['markStep'], cfg['markStep'])
    }

    return html.Div([
        html.Div([
            html.Span(cfg['label'], style={'fontSize': '12px', 'color': '#b0b0c0'}),
            html.Span(
                f'{value:.1f}',
                id=f'val-{cfg["id"]}',
                style={'fontSize': '12px', 'color': '#e0e0ff', 'float': 'right'},
            ),
        ], style={'display': 'block', 'marginBottom': '2px'}),
        dcc.Slider(
            id=f'slider-{cfg["id"]}',
            min=cfg['min'],
            max=cfg['max'],
            step=cfg['step'],
            value=value,
            marks=marks,
            tooltip={'placement': 'bottom', 'always_visible': False},
            updatemode='mouseup',  # Only update on release for performance
        ),
    ], style={'marginBottom': '12px'})


#--------------------------------------------------------------------#
# -- Dash App Builder -- #
#--------------------------------------------------------------------#

def _buildApp(
    referencePath: str,
    pythonParams: SurfboardParameters,
) -> 'dash.Dash':
    '''
    Build and configure the Dash application.

    Parameters:
    -----------
    referencePath : str
        Path to the reference STL file
    pythonParams : SurfboardParameters
        Initial Python parametric parameters (fitted to reference)

    Returns:
    --------
    dash.Dash : Configured Dash app (not yet running)
    '''
    global _referenceMesh, _referenceQuery

    # -- Load reference mesh --
    print('Loading reference mesh for viewer...')
    _referenceMesh = _loadAndAlignReference(referencePath)
    _referenceQuery = trimesh.proximity.ProximityQuery(_referenceMesh)
    print(f'  Reference: {len(_referenceMesh.vertices)} vertices')

    # -- Initial Python parametric trace --
    pythonTrace, pythonRms = _generatePythonMeshTrace(pythonParams, visible=True)

    # -- Initial figure --
    fig = _buildFigure(
        [_referenceTrace(visible=True), pythonTrace],
        boardLengthMm=pythonParams.length,
    )

    # -- Slider initial values --
    sliderVals = {
        'length':          pythonParams.length,
        'maxWidth':        pythonParams.maxWidth,
        'maxThickness':    pythonParams.maxThickness,
        'noseWidth':       pythonParams.noseWidth,
        'tailWidth':       pythonParams.tailWidth,
        'widePointOffset': pythonParams.widePointOffset,
        'noseRocker':      pythonParams.noseRocker,
        'tailRocker':      pythonParams.tailRocker,
        'deckCrown':       pythonParams.deckCrown,
        'bottomConcave':   pythonParams.bottomConcave,
    }

    # -- Build layout --
    sliderElements = [_makeSlider(cfg, sliderVals[cfg['id']]) for cfg in _SLIDER_CONFIG]

    panelStyle = {
        'width': '300px',
        'minWidth': '300px',
        'backgroundColor': '#111122',
        'padding': '16px',
        'overflowY': 'auto',
        'height': '100vh',
        'boxSizing': 'border-box',
        'borderLeft': '1px solid #334',
    }

    layout = html.Div([
        # Header bar
        html.Div(
            html.H2('Surfboard Reference Comparison', style={
                'margin': '0', 'padding': '10px 20px',
                'color': '#d0d8ff', 'fontSize': '18px', 'fontWeight': '500',
            }),
            style={'backgroundColor': '#0d0d1a', 'borderBottom': '1px solid #334'},
        ),

        # Main content row
        html.Div([
            # 3D viewport
            html.Div([
                dcc.Graph(
                    id='board-graph',
                    figure=fig,
                    config={'displayModeBar': True, 'scrollZoom': True},
                    style={'height': '100%'},
                ),
            ], style={'flex': '1', 'overflow': 'hidden', 'backgroundColor': '#1a1a2e'}),

            # Side panel
            html.Div([
                # Visibility toggles
                html.Div([
                    html.Label('Visibility', style={'color': '#b0b0c0', 'fontSize': '13px', 'fontWeight': '600'}),
                    dcc.Checklist(
                        id='visibility-toggles',
                        options=[
                            {'label': ' Reference STL', 'value': 'ref'},
                            {'label': ' Python Parametric', 'value': 'py'},
                        ],
                        value=['ref', 'py'],
                        style={'color': '#d0d0e0', 'fontSize': '13px'},
                        labelStyle={'display': 'block', 'marginBottom': '4px'},
                    ),
                ], style={'marginBottom': '16px', 'padding': '10px', 'backgroundColor': '#1a1a2e',
                          'borderRadius': '4px', 'border': '1px solid #334'}),

                # RMS display
                html.Div(id='rms-display', children=[
                    html.Div(f'Python RMS: {pythonRms:.2f} mm',
                             style={'color': '#88dd88', 'fontSize': '13px'}),
                ], style={'marginBottom': '16px', 'padding': '10px', 'backgroundColor': '#1a1a2e',
                          'borderRadius': '4px', 'border': '1px solid #334'}),

                # Parameter sliders
                html.Div([
                    html.Label('Python Parametric Parameters',
                               style={'color': '#b0b0c0', 'fontSize': '13px', 'fontWeight': '600',
                                      'marginBottom': '10px', 'display': 'block'}),
                    *sliderElements,
                ], style={'padding': '10px', 'backgroundColor': '#1a1a2e',
                          'borderRadius': '4px', 'border': '1px solid #334',
                          'marginBottom': '12px'}),

            ], style=panelStyle),
        ], style={'display': 'flex', 'flex': '1', 'overflow': 'hidden'}),

    ], style={
        'display': 'flex',
        'flexDirection': 'column',
        'height': '100vh',
        'backgroundColor': '#0d0d1a',
        'fontFamily': "'Segoe UI', Arial, sans-serif",
        'color': '#e0e0e0',
    })

    # -- Create Dash app --
    app = dash.Dash(__name__, title='Surfboard Comparison Viewer')
    app.layout = layout

    # -- Register callbacks --
    _registerCallbacks(app, pythonParams)

    return app


def _registerCallbacks(app: 'dash.Dash', defaultParams: SurfboardParameters) -> None:
    '''
    Register all Dash callbacks for the comparison viewer.

    Parameters:
    -----------
    app : dash.Dash
        The Dash application instance
    defaultParams : SurfboardParameters
        Default board parameters (used as baseline)
    '''

    sliderIds = [f'slider-{cfg["id"]}' for cfg in _SLIDER_CONFIG]
    paramIds = [cfg['id'] for cfg in _SLIDER_CONFIG]

    # -- Main graph update callback --
    @app.callback(
        Output('board-graph', 'figure'),
        Output('rms-display', 'children'),
        [Input(sid, 'value') for sid in sliderIds] + [Input('visibility-toggles', 'value')],
    )
    def updateGraph(*args):
        sliderValues = dict(zip(paramIds, args[:len(paramIds)]))
        visibleSets = args[len(paramIds)]  # list of visible trace keys

        showRef = 'ref' in visibleSets
        showPy = 'py' in visibleSets

        params = _paramsFromSliders(sliderValues)

        traces = [_referenceTrace(visible=showRef)]

        # Python parametric board
        pyTrace, pyRms = _generatePythonMeshTrace(params, visible=showPy)
        traces.append(pyTrace)

        fig = _buildFigure(traces, boardLengthMm=params.length)

        # RMS display
        rmsChildren = [
            html.Div(
                f'Python RMS: {pyRms:.2f} mm',
                style={'color': '#88dd88', 'fontSize': '13px'},
            ),
        ]

        return fig, rmsChildren

    # -- Slider value display callbacks --
    for cfg in _SLIDER_CONFIG:
        sid = cfg['id']

        @app.callback(
            Output(f'val-{sid}', 'children'),
            Input(f'slider-{sid}', 'value'),
        )
        def updateVal(value):
            return f'{value:.1f}'


#--------------------------------------------------------------------#
# -- Public Entry Point -- #
#--------------------------------------------------------------------#

def launchViewer(
    referencePath: str,
    pythonParams: SurfboardParameters,
    port: int = 8050,
    debug: bool = False,
) -> None:
    '''
    Launch the interactive Plotly Dash comparison viewer.

    Displays the reference STL as a grey semi-transparent overlay and
    the Python parametric board with per-vertex deviation coloring.

    Parameters:
    -----------
    referencePath : str
        Path to the reference STL file
    pythonParams : SurfboardParameters
        Initial Python parametric parameters (e.g. from ReverseEngineer)
    port : int
        Local port to serve the Dash app on (default 8050)
    debug : bool
        Enable Dash debug mode

    Examples:
    ---------
    >>> from pysurf.Surfboard.SurfPhysics.visualization.referenceComparisonViewer import launchViewer
    >>> launchViewer('referenceThruster.stl', pythonParams)
    '''
    if not _dashAvailable:
        raise ImportError(
            'dash is required for the comparison viewer. '
            'Install with: pip install dash'
        )
    if trimesh is None:
        raise ImportError(
            'trimesh is required for the comparison viewer. '
            'Install with: pip install trimesh'
        )

    app = _buildApp(
        referencePath=referencePath,
        pythonParams=pythonParams,
    )

    print(f'\nLaunching Surfboard Comparison Viewer at http://127.0.0.1:{port}')
    print('Open this URL in your browser to interact with the 3D comparison.')
    print('Press Ctrl+C to stop the server.\n')

    app.run(debug=debug, port=port)
