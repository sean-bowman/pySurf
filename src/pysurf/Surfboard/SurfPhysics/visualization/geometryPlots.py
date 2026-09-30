# -- Board Geometry Visualizations -- #

'''

Plotly-based interactive plots for surfboard geometry.

Sean Bowman [02/03/2026]

'''

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pysurf.Surfboard.SurfPhysics.geometry.board import BoardGeometry
from pysurf.Surfboard.SurfPhysics.geometry.parameters import SurfboardParameters
from pysurf.Surfboard.SurfPhysics.units import mToMm
from pysurf.Surfboard.SurfPhysics.visualization import theme

def plotOutline(board: BoardGeometry, params: SurfboardParameters) -> go.Figure:

    '''

    Top-down planform outline showing both rails.

    Parameters:
    -----------
    board : BoardGeometry
        Board geometry
    params : SurfboardParameters
        Board parameters

    Returns:
    --------
    go.Figure : Plotly figure

    '''
    nPoints = 200
    tValues = np.linspace(0, 1, nPoints)

    # Half-widths in mm for display
    halfWidths = np.array([board.getHalfWidthM(t) * 1000 for t in tValues])
    xPositions = tValues * params.length  # mm from nose

    fig = go.Figure()

    # Right rail
    fig.add_trace(go.Scatter(
        x=xPositions, y=halfWidths, mode='lines',
        name='Rail', line=dict(color=theme.BLUE, width=2),
        showlegend=False,
    ))
    # Left rail (mirrored)
    fig.add_trace(go.Scatter(
        x=xPositions, y=-halfWidths, mode='lines',
        name='Rail', line=dict(color=theme.BLUE, width=2),
        showlegend=False,
    ))
    # Centerline
    fig.add_trace(go.Scatter(
        x=xPositions, y=np.zeros_like(xPositions), mode='lines',
        line=dict(color=theme.REFERENCE_LINE, dash='dot', width=0.5),
        showlegend=False,
    ))

    # Annotations: wide point, nose width, tail width
    widePointX = params.widePointX
    widePointHalf = board.getHalfWidthM(widePointX / params.length) * 1000
    fig.add_annotation(x=widePointX, y=widePointHalf + 20,
                       text=f'Wide: {params.maxWidth:.0f}mm',
                       showarrow=False, font=dict(size=10))

    fig.update_layout(
        title='Board Outline (Planform)',
        xaxis_title='Length from Nose (mm)',
        yaxis_title='Half-Width (mm)',
        yaxis=dict(scaleanchor='x', scaleratio=1),
        template=theme.TEMPLATE,
        height=350,
    )

    return fig

def plotRockerProfile(board: BoardGeometry, params: SurfboardParameters) -> go.Figure:

    '''

    Side-view rocker curve.

    Parameters:
    -----------
    board : BoardGeometry
        Board geometry
    params : SurfboardParameters
        Board parameters

    Returns:
    --------
    go.Figure : Plotly figure

    '''
    nPoints = 200
    tValues = np.linspace(0, 1, nPoints)

    rockerMm = np.array([board.getRockerHeightM(t) * 1000 for t in tValues])
    xPositions = tValues * params.length  # mm from nose

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=xPositions, y=-rockerMm, mode='lines',
        name='Rocker', line=dict(color=theme.RED, width=2),
    ))

    # Flat reference line
    fig.add_hline(y=0, line=dict(color=theme.REFERENCE_LINE, dash='dash', width=0.5))

    # Annotations
    fig.add_annotation(x=0, y=-params.noseRocker,
                       text=f'Nose: {params.noseRocker:.0f}mm',
                       showarrow=True, arrowhead=2)
    fig.add_annotation(x=params.length, y=-params.tailRocker,
                       text=f'Tail: {params.tailRocker:.0f}mm',
                       showarrow=True, arrowhead=2)

    fig.update_layout(
        title='Rocker Profile (Side View)',
        xaxis_title='Length from Nose (mm)',
        yaxis_title='Rocker Height (mm)',
        template=theme.TEMPLATE,
        height=300,
    )

    return fig

def plotCrossSections(
    board: BoardGeometry,
    params: SurfboardParameters,
    stations: list[float] | None = None,
) -> go.Figure:

    '''

    Cross-sectional profiles at multiple longitudinal stations.

    Parameters:
    -----------
    board : BoardGeometry
        Board geometry
    params : SurfboardParameters
        Board parameters
    stations : list[float]
        Normalized positions to show (default: several key stations)

    Returns:
    --------
    go.Figure : Plotly figure

    '''
    if stations is None:
        stations = [0.1, 0.25, 0.4, 0.6, 0.8, 0.95]

    nLateral = 50
    colors = theme.STATION_COLORS

    fig = go.Figure()

    for idx, t in enumerate(stations):
        halfWidthMm = board.getHalfWidthM(t) * 1000
        lateralFractions = np.linspace(0, 1, nLateral)

        yValues = lateralFractions * halfWidthMm
        deckMm = np.array([board.getDeckHeightM(t, lf) * 1000 for lf in lateralFractions])
        bottomMm = np.array([board.getBottomHeightM(t, lf) * 1000 for lf in lateralFractions])

        # Full cross-section: deck left-to-right then bottom right-to-left
        yFull = np.concatenate([-yValues[::-1], yValues])
        zFull = np.concatenate([deckMm[::-1], deckMm])
        yBottom = np.concatenate([-yValues[::-1], yValues])
        zBottom = np.concatenate([bottomMm[::-1], bottomMm])

        color = colors[idx % len(colors)]
        stationPct = int(t * 100)

        fig.add_trace(go.Scatter(
            x=yFull, y=zFull, mode='lines',
            name=f'{stationPct}% deck',
            line=dict(color=color, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=yBottom, y=zBottom, mode='lines',
            name=f'{stationPct}% bottom',
            line=dict(color=color, width=2, dash='dash'),
            showlegend=False,
        ))

    fig.update_layout(
        title='Cross-Sections at Multiple Stations',
        xaxis_title='Width (mm)',
        yaxis_title='Height (mm)',
        yaxis=dict(scaleanchor='x', scaleratio=1),
        template=theme.TEMPLATE,
        height=400,
    )

    return fig

def plotBoardOverview(board: BoardGeometry, params: SurfboardParameters) -> go.Figure:

    '''

    Combined 4-subplot board geometry overview.

    Parameters:
    -----------
    board : BoardGeometry
        Board geometry
    params : SurfboardParameters
        Board parameters

    Returns:
    --------
    go.Figure : Plotly figure

    '''
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Outline (Top View)', 'Rocker (Side View)',
                        'Cross-Sections', 'Thickness Distribution'),
    )

    nPoints = 200
    tValues = np.linspace(0, 1, nPoints)
    xPositions = tValues * params.length

    # 1. Outline
    halfWidths = np.array([board.getHalfWidthM(t) * 1000 for t in tValues])
    fig.add_trace(go.Scatter(x=xPositions, y=halfWidths, mode='lines',
                             line=dict(color=theme.BLUE, width=2), showlegend=False),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=xPositions, y=-halfWidths, mode='lines',
                             line=dict(color=theme.BLUE, width=2), showlegend=False),
                  row=1, col=1)

    # 2. Rocker
    rockerMm = np.array([board.getRockerHeightM(t) * 1000 for t in tValues])
    fig.add_trace(go.Scatter(x=xPositions, y=-rockerMm, mode='lines',
                             line=dict(color=theme.RED, width=2), showlegend=False),
                  row=1, col=2)

    # 3. Cross-sections at 3 stations
    stations = [0.25, 0.4, 0.75]
    colors = [theme.GREEN, theme.ORANGE, theme.PURPLE]
    for idx, t in enumerate(stations):
        halfWidthMm = board.getHalfWidthM(t) * 1000
        latFrac = np.linspace(0, 1, 30)
        yV = latFrac * halfWidthMm
        dMm = np.array([board.getDeckHeightM(t, lf) * 1000 for lf in latFrac])
        bMm = np.array([board.getBottomHeightM(t, lf) * 1000 for lf in latFrac])
        yFull = np.concatenate([-yV[::-1], yV])
        zDeck = np.concatenate([dMm[::-1], dMm])
        zBot = np.concatenate([bMm[::-1], bMm])
        fig.add_trace(go.Scatter(x=yFull, y=zDeck, mode='lines',
                                 line=dict(color=colors[idx], width=1.5), showlegend=False),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=yFull, y=zBot, mode='lines',
                                 line=dict(color=colors[idx], width=1.5, dash='dash'),
                                 showlegend=False),
                      row=2, col=1)

    # 4. Thickness distribution
    thickness = np.array([board.getThicknessM(t) * 1000 for t in tValues])
    fig.add_trace(go.Scatter(x=xPositions, y=thickness, mode='lines',
                             line=dict(color=theme.BROWN, width=2), showlegend=False),
                  row=2, col=2)

    fig.update_layout(
        title=f'Board Geometry Overview ({params.length:.0f}mm x {params.maxWidth:.0f}mm x {params.maxThickness:.0f}mm)',
        template=theme.TEMPLATE,
        height=700,
    )

    return fig
