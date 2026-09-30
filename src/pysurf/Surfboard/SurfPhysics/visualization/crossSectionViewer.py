
# -- Cross-Section Comparison Viewer -- #

'''
Interactive Plotly Dash viewer for comparing parametric surfboard geometry
against a reference STL using 2D cross-section slices.

At startup the reference STL is sliced at N axial stations and the raw
(Y, Z) contour points at each station are stored. The four synchronized
panels update in real time as parameter sliders are adjusted:

  Panel 1 (top-left)  : Cross-section overlay at selected station
                         Reference raw contour + parametric closed curve
  Panel 2 (top-right) : Outline comparison (planform / top view)
                         Reference half-width vs. parametric, board length axis
  Panel 3 (bottom-left): Rocker comparison (side view)
                         Reference rocker curve vs. parametric
  Panel 4 (bottom-right): Thickness distribution
                         Reference thickness vs. parametric

Usage:
  from pysurf.Surfboard.SurfPhysics.visualization.crossSectionViewer import launchCrossSectionViewer
  launchCrossSectionViewer(referencePath, pythonParams)

Sean Bowman [02/28/2026]
'''

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import dash
    from dash import dcc, html, Input, Output
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _dashAvailable = True
except ImportError:
    _dashAvailable = False

from pysurf.Surfboard.SurfPhysics.geometry.parameters import SurfboardParameters
from pysurf.Surfboard.SurfPhysics.geometry.outline import Outline
from pysurf.Surfboard.SurfPhysics.geometry.rocker import RockerProfile
from pysurf.Surfboard.SurfPhysics.geometry.crossSection import CrossSection
from pysurf.Surfboard.SurfPhysics.validation.reverseEngineer import ReverseEngineer, ExtractedProfiles


#--------------------------------------------------------------------#
# -- Module-Level State (shared across Dash callbacks) -- #
#--------------------------------------------------------------------#

# Cached reference data -- loaded once at startup
_refProfiles: Optional[ExtractedProfiles] = None
_refContours: Optional[list] = None          # list of (N, 2) np.ndarray or None
_boardLengthMm: float = 1828.0
_nStations: int = 50


#--------------------------------------------------------------------#
# -- Colors / Theme -- #
#--------------------------------------------------------------------#

_BG_DARK    = '#0d0d1a'
_BG_PANEL   = '#111122'
_BG_PLOT    = '#1a1a2e'
_BORDER     = '#334466'
_TEXT       = '#d0d8ff'
_TEXT_DIM   = '#8090b0'
_REF_COLOR  = '#4488ff'      # blue: reference
_PARAM_COLOR = '#ff6644'     # orange-red: parametric
_STATION_COLOR = '#ffdd44'   # yellow: station indicator line


#--------------------------------------------------------------------#
# -- Figure Builders -- #
#--------------------------------------------------------------------#

def _figLayout(title: str = '') -> dict:
    '''Common Plotly layout settings for all panels (no xaxis/yaxis: each panel provides its own).'''
    return dict(
        title=dict(text=title, font=dict(color=_TEXT, size=12), x=0.5),
        paper_bgcolor=_BG_PLOT,
        plot_bgcolor=_BG_PLOT,
        font=dict(color=_TEXT, size=10),
        margin=dict(l=45, r=10, t=30, b=40),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=9),
            x=0, y=1,
        ),
    )


def _stationLine(xMm: float, yMin: float, yMax: float) -> go.Scatter:
    '''Vertical dashed line marking the selected station on a board-length axis.'''
    return go.Scatter(
        x=[xMm, xMm], y=[yMin, yMax],
        mode='lines',
        line=dict(color=_STATION_COLOR, width=1.5, dash='dash'),
        name='Station',
        showlegend=False,
        hoverinfo='skip',
    )


def _buildCrossSectionFig(
    stationIdx: int,
    outline: Outline,
    rocker: RockerProfile,
    crossSection: CrossSection,
) -> go.Figure:
    '''
    Build the cross-section overlay panel for a given station index.

    Shows the raw reference contour (scatter dots) and the parametric
    closed cross-section curve at the same axial position.

    Parameters:
    -----------
    stationIdx : int
        Index into _refProfiles and _refContours arrays
    outline : Outline
        Parametric outline model
    rocker : RockerProfile
        Parametric rocker model
    crossSection : CrossSection
        Parametric cross-section model

    Returns:
    --------
    go.Figure : Plotly figure for the cross-section panel
    '''
    t = float(_refProfiles.tValues[stationIdx])
    xMm = t * _boardLengthMm
    rockerRef = float(_refProfiles.rockerHeights[stationIdx])

    traces = []

    # -- Reference: raw contour scatter --
    contour = _refContours[stationIdx] if _refContours else None
    if contour is not None and len(contour) >= 3:
        # Normalize Z to be relative to the reference rocker at this station
        yPoints = contour[:, 0]     # width
        zPoints = contour[:, 1] - rockerRef   # height relative to center plane
        # Close the loop
        yClose = np.append(yPoints, yPoints[0])
        zClose = np.append(zPoints, zPoints[0])
        traces.append(go.Scatter(
            x=yClose, y=zClose,
            mode='lines+markers',
            marker=dict(size=2, color=_REF_COLOR, opacity=0.6),
            line=dict(color=_REF_COLOR, width=1.5),
            name='Reference (raw)',
        ))

    # -- Reference: smooth envelope from extracted profiles --
    lf = _refProfiles.lateralFractions
    hw_ref = float(_refProfiles.halfWidths[stationIdx])
    y_lf = lf * hw_ref

    deckRef    = _refProfiles.deckProfiles[stationIdx]     # already relative to rocker
    bottomRef  = _refProfiles.bottomProfiles[stationIdx]

    # Build closed envelope: right bottom → right deck → left deck → left bottom → close
    yEnvelope = np.concatenate([y_lf, y_lf[::-1], [-y_lf[0]]])
    zEnvelope = np.concatenate([bottomRef, deckRef[::-1], [bottomRef[0]]])
    traces.append(go.Scatter(
        x=yEnvelope, y=zEnvelope,
        mode='lines',
        line=dict(color=_REF_COLOR, width=2, dash='dot'),
        name='Reference (profile)',
        opacity=0.7,
    ))

    # -- Parametric: closed cross-section curve --
    hw_param = outline.getHalfWidth(t)
    nLateral = 40
    lateralFractions = np.linspace(0.0, 1.0, nLateral)

    deckParam   = np.array([crossSection.getDeckHeight(t, float(lf)) for lf in lateralFractions])
    bottomParam = np.array([crossSection.getBottomHeight(t, float(lf)) for lf in lateralFractions])

    y_param = lateralFractions * hw_param
    # Closed loop: right bottom → right deck → left deck → left bottom → close
    yParam = np.concatenate([y_param, y_param[::-1], [y_param[0]]])
    zParam = np.concatenate([bottomParam, deckParam[::-1], [bottomParam[0]]])

    # Mirror left side
    yParamFull = np.concatenate([-y_param[::-1], y_param])
    zParamFull_deck   = np.concatenate([deckParam[::-1], deckParam])
    zParamFull_bottom = np.concatenate([bottomParam[::-1], bottomParam])

    # Full cross-section: left rail bottom → center bottom → right rail bottom
    #                   → right rail deck  → center deck  → left rail deck → close
    yFull = np.concatenate([-y_param[::-1], y_param, y_param[::-1], -y_param])
    zFull = np.concatenate([bottomParam[::-1], bottomParam, deckParam, deckParam[::-1]])

    traces.append(go.Scatter(
        x=yFull, y=zFull,
        mode='lines',
        line=dict(color=_PARAM_COLOR, width=2),
        name='Parametric',
        fill='toself',
        fillcolor='rgba(255,100,68,0.1)',
    ))

    # -- Zero line (center plane) --
    traces.append(go.Scatter(
        x=[-hw_ref * 1.05, hw_ref * 1.05], y=[0, 0],
        mode='lines',
        line=dict(color='#444466', width=1, dash='dot'),
        showlegend=False,
        hoverinfo='skip',
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        **_figLayout(f'Cross-Section  t={t:.2f}  ({xMm:.0f} mm)'),
        xaxis=dict(
            title='Width (mm)',
            gridcolor='#2a2a40', zerolinecolor='#3a3a60', linecolor='#3a3a60',
            scaleanchor='y', scaleratio=1,
        ),
        yaxis=dict(
            title='Height rel. center (mm)',
            gridcolor='#2a2a40', zerolinecolor='#3a3a60', linecolor='#3a3a60',
        ),
    )
    return fig


def _buildOutlineFig(
    stationIdx: int,
    outline: Outline,
) -> go.Figure:
    '''
    Build the outline (planform top-view) comparison panel.

    Parameters:
    -----------
    stationIdx : int
        Index of the selected station (for the vertical indicator line)
    outline : Outline
        Parametric outline model

    Returns:
    --------
    go.Figure : Plotly figure for the outline panel
    '''
    xRef = _refProfiles.tValues * _boardLengthMm
    hwRef = _refProfiles.halfWidths

    # Parametric at 200 points
    tParam = np.linspace(0.0, 1.0, 200)
    hwParam = np.array([outline.getHalfWidth(float(t)) for t in tParam])
    xParam = tParam * _boardLengthMm

    stationX = float(_refProfiles.tValues[stationIdx]) * _boardLengthMm
    yMin, yMax = -max(hwRef.max(), hwParam.max()) * 1.05, max(hwRef.max(), hwParam.max()) * 1.05

    traces = [
        # Reference both sides
        go.Scatter(x=xRef, y=hwRef,  mode='lines', line=dict(color=_REF_COLOR, width=2),
                   name='Reference'),
        go.Scatter(x=xRef, y=-hwRef, mode='lines', line=dict(color=_REF_COLOR, width=2),
                   showlegend=False),
        # Parametric both sides
        go.Scatter(x=xParam, y=hwParam,  mode='lines', line=dict(color=_PARAM_COLOR, width=2),
                   name='Parametric'),
        go.Scatter(x=xParam, y=-hwParam, mode='lines', line=dict(color=_PARAM_COLOR, width=2),
                   showlegend=False),
        # Station indicator
        _stationLine(stationX, yMin, yMax),
    ]

    outlineRms = float(np.sqrt(np.mean(
        (hwRef - np.interp(_refProfiles.tValues, tParam, hwParam)) ** 2
    )))

    fig = go.Figure(data=traces)
    fig.update_layout(
        **_figLayout(f'Outline  RMS={outlineRms:.1f} mm'),
        xaxis=dict(title='Board position (mm)', gridcolor='#2a2a40',
                   zerolinecolor='#3a3a60', linecolor='#3a3a60', range=[0, _boardLengthMm]),
        yaxis=dict(title='Half-width (mm)', gridcolor='#2a2a40',
                   zerolinecolor='#3a3a60', linecolor='#3a3a60',
                   range=[yMin, yMax]),
    )
    return fig


def _buildRockerFig(
    stationIdx: int,
    rocker: RockerProfile,
) -> go.Figure:
    '''
    Build the rocker (side-view) comparison panel.

    Parameters:
    -----------
    stationIdx : int
        Index of the selected station
    rocker : RockerProfile
        Parametric rocker model

    Returns:
    --------
    go.Figure : Plotly figure for the rocker panel
    '''
    xRef = _refProfiles.tValues * _boardLengthMm
    rkRef = _refProfiles.rockerHeights

    tParam = np.linspace(0.0, 1.0, 200)
    rkParam = np.array([rocker.getRockerHeight(float(t)) for t in tParam])
    xParam = tParam * _boardLengthMm

    stationX = float(_refProfiles.tValues[stationIdx]) * _boardLengthMm
    yMin = -5.0
    yMax = max(rkRef.max(), rkParam.max()) * 1.1 + 5.0

    rockerRms = float(np.sqrt(np.mean(
        (rkRef - np.interp(_refProfiles.tValues, tParam, rkParam)) ** 2
    )))

    traces = [
        go.Scatter(x=xRef, y=rkRef, mode='lines', line=dict(color=_REF_COLOR, width=2),
                   name='Reference'),
        go.Scatter(x=xParam, y=rkParam, mode='lines', line=dict(color=_PARAM_COLOR, width=2),
                   name='Parametric'),
        _stationLine(stationX, yMin, yMax),
    ]

    fig = go.Figure(data=traces)
    fig.update_layout(
        **_figLayout(f'Rocker  RMS={rockerRms:.1f} mm'),
        xaxis=dict(title='Board position (mm)', gridcolor='#2a2a40',
                   zerolinecolor='#3a3a60', linecolor='#3a3a60', range=[0, _boardLengthMm]),
        yaxis=dict(title='Rocker height (mm)', gridcolor='#2a2a40',
                   zerolinecolor='#3a3a60', linecolor='#3a3a60',
                   range=[yMin, yMax]),
    )
    return fig


def _buildThicknessFig(
    stationIdx: int,
    crossSection: CrossSection,
) -> go.Figure:
    '''
    Build the thickness distribution comparison panel.

    Parameters:
    -----------
    stationIdx : int
        Index of the selected station
    crossSection : CrossSection
        Parametric cross-section model

    Returns:
    --------
    go.Figure : Plotly figure for the thickness panel
    '''
    xRef = _refProfiles.tValues * _boardLengthMm
    thkRef = _refProfiles.thicknesses

    tParam = np.linspace(0.0, 1.0, 200)
    thkParam = np.array([crossSection.getLocalThickness(float(t)) for t in tParam])
    xParam = tParam * _boardLengthMm

    stationX = float(_refProfiles.tValues[stationIdx]) * _boardLengthMm
    yMin = 0.0
    yMax = max(thkRef.max(), thkParam.max()) * 1.15 + 5.0

    thkRms = float(np.sqrt(np.mean(
        (thkRef - np.interp(_refProfiles.tValues, tParam, thkParam)) ** 2
    )))

    traces = [
        go.Scatter(x=xRef, y=thkRef, mode='lines', line=dict(color=_REF_COLOR, width=2),
                   name='Reference'),
        go.Scatter(x=xParam, y=thkParam, mode='lines', line=dict(color=_PARAM_COLOR, width=2),
                   name='Parametric'),
        _stationLine(stationX, yMin, yMax),
    ]

    fig = go.Figure(data=traces)
    fig.update_layout(
        **_figLayout(f'Thickness  RMS={thkRms:.1f} mm'),
        xaxis=dict(title='Board position (mm)', gridcolor='#2a2a40',
                   zerolinecolor='#3a3a60', linecolor='#3a3a60', range=[0, _boardLengthMm]),
        yaxis=dict(title='Thickness (mm)', gridcolor='#2a2a40',
                   zerolinecolor='#3a3a60', linecolor='#3a3a60',
                   range=[yMin, yMax]),
    )
    return fig


#--------------------------------------------------------------------#
# -- Slider Configuration -- #
#--------------------------------------------------------------------#

_SLIDER_CONFIG = [
    dict(id='length',          label='Length (mm)',           min=1500, max=3000, step=10),
    dict(id='maxWidth',        label='Max Width (mm)',         min=200,  max=800,  step=5),
    dict(id='maxThickness',    label='Max Thickness (mm)',     min=20,   max=150,  step=1),
    dict(id='noseWidth',       label='Nose Width (mm)',        min=50,   max=600,  step=5),
    dict(id='tailWidth',       label='Tail Width (mm)',        min=50,   max=600,  step=5),
    dict(id='widePointOffset', label='Wide Point Offset (mm)', min=-300, max=300,  step=5),
    dict(id='noseRocker',      label='Nose Rocker (mm)',       min=10,   max=300,  step=5),
    dict(id='tailRocker',      label='Tail Rocker (mm)',       min=5,    max=150,  step=2),
    dict(id='deckCrown',       label='Deck Crown (mm)',        min=0,    max=40,   step=0.5),
    dict(id='bottomConcave',   label='Bottom Concave (mm)',    min=0,    max=20,   step=0.5),
]


def _makeSlider(cfg: dict, value: float) -> html.Div:
    '''
    Build a labelled Dash slider with current value display.

    Parameters:
    -----------
    cfg : dict
        Slider config: id, label, min, max, step
    value : float
        Initial value

    Returns:
    --------
    html.Div : Slider component
    '''
    return html.Div([
        html.Div([
            html.Span(cfg['label'], style={'fontSize': '11px', 'color': _TEXT_DIM}),
            html.Span(
                f'{value:.1f}',
                id=f'csv-val-{cfg["id"]}',
                style={'fontSize': '11px', 'color': _TEXT, 'float': 'right'},
            ),
        ], style={'display': 'block', 'marginBottom': '2px'}),
        dcc.Slider(
            id=f'csv-slider-{cfg["id"]}',
            min=cfg['min'],
            max=cfg['max'],
            step=cfg['step'],
            value=value,
            marks=None,
            tooltip={'placement': 'bottom', 'always_visible': False},
            updatemode='drag',
        ),
    ], style={'marginBottom': '10px'})


#--------------------------------------------------------------------#
# -- App Builder -- #
#--------------------------------------------------------------------#

def _buildApp(
    pythonParams: SurfboardParameters,
) -> 'dash.Dash':
    '''
    Build and configure the cross-section comparison Dash app.

    Parameters:
    -----------
    pythonParams : SurfboardParameters
        Initial board parameters (from reverse engineering)

    Returns:
    --------
    dash.Dash : Configured app (not yet running)
    '''
    # Initial slider values from pythonParams
    initVals = {
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

    # Build initial figures
    outline     = Outline(pythonParams)
    rocker      = RockerProfile(pythonParams)
    cs          = CrossSection(pythonParams)
    initStation = _nStations // 2

    figCrossSection = _buildCrossSectionFig(initStation, outline, rocker, cs)
    figOutline      = _buildOutlineFig(initStation, outline)
    figRocker       = _buildRockerFig(initStation, rocker)
    figThickness    = _buildThicknessFig(initStation, cs)

    # Station slider marks at key positions
    stationMarks = {}
    for pct in [0, 25, 50, 75, 100]:
        idx = max(0, min(_nStations - 1, int(pct / 100 * (_nStations - 1))))
        xMm = int(_refProfiles.tValues[idx] * _boardLengthMm)
        stationMarks[idx] = f'{xMm}mm'

    panelStyle = {
        'width': '280px',
        'minWidth': '280px',
        'backgroundColor': _BG_PANEL,
        'padding': '12px',
        'overflowY': 'auto',
        'height': '100vh',
        'boxSizing': 'border-box',
        'borderLeft': f'1px solid {_BORDER}',
    }

    graphStyle = {
        'flex': '1',
        'minWidth': '0',
        'height': '100%',
    }

    gridStyle = {
        'display': 'grid',
        'gridTemplateColumns': '1fr 1fr',
        'gridTemplateRows': '1fr 1fr',
        'flex': '1',
        'gap': '2px',
        'overflow': 'hidden',
    }

    layout = html.Div([
        # Header
        html.Div(
            html.H2('Surfboard Cross-Section Comparison', style={
                'margin': '0', 'padding': '10px 20px',
                'color': _TEXT, 'fontSize': '16px', 'fontWeight': '500',
            }),
            style={'backgroundColor': '#0a0a14', 'borderBottom': f'1px solid {_BORDER}'},
        ),

        # Main row
        html.Div([
            # 2×2 plot grid
            html.Div([
                dcc.Graph(id='csv-cross-section', figure=figCrossSection, style=graphStyle,
                          config={'displayModeBar': False}),
                dcc.Graph(id='csv-outline', figure=figOutline, style=graphStyle,
                          config={'displayModeBar': False}),
                dcc.Graph(id='csv-rocker', figure=figRocker, style=graphStyle,
                          config={'displayModeBar': False}),
                dcc.Graph(id='csv-thickness', figure=figThickness, style=graphStyle,
                          config={'displayModeBar': False}),
            ], style=gridStyle),

            # Side panel
            html.Div([
                # Station selector
                html.Div([
                    html.Label('Station', style={
                        'color': _TEXT_DIM, 'fontSize': '11px', 'fontWeight': '600',
                        'display': 'block', 'marginBottom': '4px',
                    }),
                    dcc.Slider(
                        id='csv-station',
                        min=0,
                        max=_nStations - 1,
                        step=1,
                        value=initStation,
                        marks=stationMarks,
                        tooltip={'placement': 'bottom', 'always_visible': True},
                        updatemode='drag',
                    ),
                ], style={
                    'marginBottom': '16px', 'padding': '10px',
                    'backgroundColor': _BG_PLOT, 'borderRadius': '4px',
                    'border': f'1px solid {_BORDER}',
                }),

                # RMS display
                html.Div(id='csv-rms-display', style={
                    'marginBottom': '16px', 'padding': '10px',
                    'backgroundColor': _BG_PLOT, 'borderRadius': '4px',
                    'border': f'1px solid {_BORDER}',
                    'fontSize': '11px',
                }),

                # Parameter sliders
                html.Div([
                    html.Label('Board Parameters',
                               style={'color': _TEXT_DIM, 'fontSize': '11px', 'fontWeight': '600',
                                      'display': 'block', 'marginBottom': '8px'}),
                    *[_makeSlider(cfg, initVals[cfg['id']]) for cfg in _SLIDER_CONFIG],
                ], style={
                    'padding': '10px', 'backgroundColor': _BG_PLOT,
                    'borderRadius': '4px', 'border': f'1px solid {_BORDER}',
                }),
            ], style=panelStyle),
        ], style={'display': 'flex', 'flex': '1', 'overflow': 'hidden', 'height': '0'}),

    ], style={
        'display': 'flex',
        'flexDirection': 'column',
        'height': '100vh',
        'backgroundColor': _BG_DARK,
        'fontFamily': "'Segoe UI', Arial, sans-serif",
        'color': _TEXT,
    })

    app = dash.Dash(__name__, title='Cross-Section Comparison')
    app.layout = layout

    _registerCallbacks(app)
    return app


#--------------------------------------------------------------------#
# -- Callbacks -- #
#--------------------------------------------------------------------#

def _registerCallbacks(app: 'dash.Dash') -> None:
    '''
    Register the main update callback and slider value display callbacks.

    Parameters:
    -----------
    app : dash.Dash
        The Dash application instance
    '''
    sliderIds = [cfg['id'] for cfg in _SLIDER_CONFIG]
    inputList = (
        [Input('csv-station', 'value')] +
        [Input(f'csv-slider-{sid}', 'value') for sid in sliderIds]
    )

    @app.callback(
        Output('csv-cross-section', 'figure'),
        Output('csv-outline', 'figure'),
        Output('csv-rocker', 'figure'),
        Output('csv-thickness', 'figure'),
        Output('csv-rms-display', 'children'),
        inputList,
    )
    def updateAll(stationIdx, *sliderVals):
        valMap = dict(zip(sliderIds, sliderVals))

        params = SurfboardParameters(
            length          = valMap['length'],
            maxWidth        = valMap['maxWidth'],
            maxThickness    = valMap['maxThickness'],
            noseWidth       = valMap['noseWidth'],
            tailWidth       = valMap['tailWidth'],
            widePointOffset = valMap['widePointOffset'],
            noseRocker      = valMap['noseRocker'],
            tailRocker      = valMap['tailRocker'],
            deckCrown       = valMap['deckCrown'],
            bottomConcave   = valMap['bottomConcave'],
        )

        outline     = Outline(params)
        rocker      = RockerProfile(params)
        cs          = CrossSection(params)

        idx = max(0, min(int(stationIdx), _nStations - 1))

        figCS  = _buildCrossSectionFig(idx, outline, rocker, cs)
        figOut = _buildOutlineFig(idx, outline)
        figRck = _buildRockerFig(idx, rocker)
        figThk = _buildThicknessFig(idx, cs)

        # Compute overall RMS values for RMS display
        tVals  = _refProfiles.tValues
        hwRef  = _refProfiles.halfWidths
        rkRef  = _refProfiles.rockerHeights
        thkRef = _refProfiles.thicknesses

        hwParam  = np.array([outline.getHalfWidth(float(t)) for t in tVals])
        rkParam  = np.array([rocker.getRockerHeight(float(t)) for t in tVals])
        thkParam = np.array([cs.getLocalThickness(float(t)) for t in tVals])

        outlineRms   = float(np.sqrt(np.mean((hwRef - hwParam) ** 2)))
        rockerRms    = float(np.sqrt(np.mean((rkRef - rkParam) ** 2)))
        thkRms       = float(np.sqrt(np.mean((thkRef - thkParam) ** 2)))

        rmsChildren = [
            html.Div('RMS Errors (mm)', style={'color': _TEXT_DIM, 'marginBottom': '4px',
                                                'fontWeight': '600'}),
            html.Div(f'Outline:    {outlineRms:.2f} mm',
                     style={'color': '#88ccff', 'fontFamily': 'monospace'}),
            html.Div(f'Rocker:     {rockerRms:.2f} mm',
                     style={'color': '#88ccff', 'fontFamily': 'monospace'}),
            html.Div(f'Thickness:  {thkRms:.2f} mm',
                     style={'color': '#88ccff', 'fontFamily': 'monospace'}),
        ]

        return figCS, figOut, figRck, figThk, rmsChildren

    # Per-slider value label callbacks
    for cfg in _SLIDER_CONFIG:
        sid = cfg['id']

        @app.callback(
            Output(f'csv-val-{sid}', 'children'),
            Input(f'csv-slider-{sid}', 'value'),
        )
        def _updateVal(value):
            return f'{value:.1f}'


#--------------------------------------------------------------------#
# -- Public Entry Point -- #
#--------------------------------------------------------------------#

def launchCrossSectionViewer(
    referencePath: str,
    pythonParams: SurfboardParameters,
    nStations: int = 50,
    port: int = 8051,
    debug: bool = False,
) -> None:
    '''
    Pre-slice the reference STL and launch the 4-panel cross-section viewer.

    At startup, the reference STL is loaded, fin-removed, and sliced at
    `nStations` axial positions. Raw (Y, Z) contour points and extracted
    profiles are cached in module-level state for fast callback access.

    Parameters:
    -----------
    referencePath : str
        Path to the reference STL file
    pythonParams : SurfboardParameters
        Initial board parameters (typically from ReverseEngineer.fitParameters)
    nStations : int
        Number of axial stations to pre-slice (default 50)
    port : int
        Local port for the Dash server (default 8051)
    debug : bool
        Enable Dash debug mode

    Examples:
    ---------
    >>> launchCrossSectionViewer('referenceThruster.stl', params)
    '''
    if not _dashAvailable:
        raise ImportError(
            'dash is required for the cross-section viewer. '
            'Install with: pip install dash'
        )

    global _refProfiles, _refContours, _boardLengthMm, _nStations
    _nStations = nStations

    print('\nLoading and slicing reference STL...')
    reEng = ReverseEngineer(
        stlPath          = referencePath,
        expectedLengthMm = 1828.0,
        removeFins       = True,
        autoTransform    = True,
    )

    print(f'Extracting profiles at {nStations} stations...')
    _refProfiles  = reEng.extractProfiles(nStations=nStations)
    _boardLengthMm = reEng.boardLengthMm

    print(f'Computing raw contours at {nStations} stations...')
    _refContours = []
    for i, t in enumerate(_refProfiles.tValues):
        contour = reEng.getContourAt(float(t))
        _refContours.append(contour)
        if (i + 1) % 10 == 0:
            print(f'  {i + 1}/{nStations} done')

    print('Building Dash app...')
    app = _buildApp(pythonParams)

    print(f'\nLaunching Cross-Section Comparison Viewer at http://127.0.0.1:{port}')
    print('Adjust sliders to match the reference: all 4 panels update live.')
    print('Press Ctrl+C to stop the server.\n')

    app.run(debug=debug, port=port)
