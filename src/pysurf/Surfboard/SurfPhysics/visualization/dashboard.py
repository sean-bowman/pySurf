# -- Combined Analysis Dashboard -- #

'''

Multi-panel Plotly dashboard combining board geometry, wave physics,
and hydrodynamic force analysis into a single interactive layout.

Sean Bowman [02/03/2026]

'''

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pysurf.Surfboard.SurfPhysics import constants as const
from pysurf.Surfboard.SurfPhysics.geometry.board import BoardGeometry
from pysurf.Surfboard.SurfPhysics.geometry.parameters import SurfboardParameters
from pysurf.Surfboard.SurfPhysics.waves.waveConditions import WaveConditions
from pysurf.Surfboard.SurfPhysics.waves.linearWaveTheory import LinearWaveTheory
from pysurf.Surfboard.SurfPhysics.hydrodynamics.forceBalance import ForceBalance
from pysurf.Surfboard.SurfPhysics.visualization import theme

def createAnalysisDashboard(
    params: SurfboardParameters,
    waveConditions: WaveConditions,
    riderMassKg: float = 75.0,
) -> go.Figure:

    '''

    Create a 6-panel analysis dashboard.

    Layout:
        Row 1: Board Outline  |  Rocker Profile
        Row 2: Wave Profile   |  Drag Breakdown
        Row 3: L/D vs Speed   |  Planing Equilibrium

    Parameters:
    -----------
    params : SurfboardParameters
        Board parameters
    waveConditions : WaveConditions
        Wave conditions for analysis
    riderMassKg : float
        Rider mass in kg

    Returns:
    --------
    go.Figure : Plotly figure with 6 subplots

    '''
    board = BoardGeometry(params)
    waveModel = LinearWaveTheory()
    forces = ForceBalance(board, params, riderMassKg)

    nPoints = 150
    tValues = np.linspace(0, 1, nPoints)
    xPositions = tValues * params.length
    speedRange = np.linspace(0.5, 10.0, 40)

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Board Outline', 'Rocker Profile',
            'Wave Profile', 'Drag Breakdown',
            'Lift-to-Drag Ratio', 'Planing Equilibrium',
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    #--------------------------------------------------------------------#
    # Row 1, Col 1: Board Outline
    #--------------------------------------------------------------------#
    halfWidths = np.array([board.getHalfWidthM(t) * 1000 for t in tValues])
    fig.add_trace(go.Scatter(x=xPositions, y=halfWidths, mode='lines',
                             line=dict(color=theme.BLUE, width=2), showlegend=False),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=xPositions, y=-halfWidths, mode='lines',
                             line=dict(color=theme.BLUE, width=2), showlegend=False),
                  row=1, col=1)
    fig.update_xaxes(title_text='mm from nose', row=1, col=1)
    fig.update_yaxes(title_text='mm', row=1, col=1)

    #--------------------------------------------------------------------#
    # Row 1, Col 2: Rocker Profile
    #--------------------------------------------------------------------#
    rockerMm = np.array([board.getRockerHeightM(t) * 1000 for t in tValues])
    fig.add_trace(go.Scatter(x=xPositions, y=-rockerMm, mode='lines',
                             line=dict(color=theme.RED, width=2), showlegend=False),
                  row=1, col=2)
    fig.update_xaxes(title_text='mm from nose', row=1, col=2)
    fig.update_yaxes(title_text='mm', row=1, col=2)

    #--------------------------------------------------------------------#
    # Row 2, Col 1: Wave Profile
    #--------------------------------------------------------------------#
    wavelength = waveModel.waveLength(waveConditions)
    xWave = np.linspace(0, 3 * wavelength, 200)
    eta = np.array([waveModel.surfaceElevation(xi, 0.0, waveConditions) for xi in xWave])
    fig.add_trace(go.Scatter(x=xWave, y=eta, mode='lines',
                             line=dict(color=theme.BLUE, width=2), showlegend=False),
                  row=2, col=1)
    fig.update_xaxes(title_text='x (m)', row=2, col=1)
    fig.update_yaxes(title_text='eta (m)', row=2, col=1)

    #--------------------------------------------------------------------#
    # Row 2, Col 2: Drag Breakdown
    #--------------------------------------------------------------------#
    breakdown = forces.dragBreakdown(speedRange, trimAngleDeg=5.0)
    fig.add_trace(go.Scatter(x=speedRange, y=breakdown['planing_drag'], mode='lines',
                             name='Planing', fill='tozeroy',
                             line=dict(color=theme.RED), showlegend=False),
                  row=2, col=2)
    fig.add_trace(go.Scatter(x=speedRange, y=breakdown['total'], mode='lines',
                             name='Total', line=dict(color=theme.WHITE, width=2),
                             showlegend=False),
                  row=2, col=2)
    fig.update_xaxes(title_text='Speed (m/s)', row=2, col=2)
    fig.update_yaxes(title_text='Drag (N)', row=2, col=2)

    #--------------------------------------------------------------------#
    # Row 3, Col 1: L/D vs Speed
    #--------------------------------------------------------------------#
    curves = forces.performanceCurves(minSpeed=0.5, maxSpeed=10.0, nPoints=40)
    fig.add_trace(go.Scatter(x=curves['speed'], y=curves['liftToDrag'], mode='lines',
                             line=dict(color=theme.GREEN, width=2), showlegend=False),
                  row=3, col=1)
    fig.update_xaxes(title_text='Speed (m/s)', row=3, col=1)
    fig.update_yaxes(title_text='L/D', row=3, col=1)

    #--------------------------------------------------------------------#
    # Row 3, Col 2: Trim Angle vs Speed
    #--------------------------------------------------------------------#
    fig.add_trace(go.Scatter(x=curves['speed'], y=curves['trimAngleDeg'], mode='lines',
                             line=dict(color=theme.PURPLE, width=2), showlegend=False),
                  row=3, col=2)
    fig.update_xaxes(title_text='Speed (m/s)', row=3, col=2)
    fig.update_yaxes(title_text='Trim (deg)', row=3, col=2)

    #--------------------------------------------------------------------#
    # Layout
    #--------------------------------------------------------------------#
    boardMassKg = forces.boardMassKg
    volumeL = board.computeVolume() * 1e3  # m^3 to liters
    waveSpeed = waveModel.waveSpeed(waveConditions)

    fig.update_layout(
        title=(
            f'SurfPhysics Analysis Dashboard -- '
            f'{params.length:.0f}mm {params.defaultFinConfiguration} '
            f'| {volumeL:.1f}L | {boardMassKg:.2f}kg | '
            f'Wave: H={waveConditions.height:.1f}m T={waveConditions.period:.0f}s '
            f'c={waveSpeed:.1f}m/s'
        ),
        template=theme.TEMPLATE,
        height=1000,
        showlegend=False,
    )

    return fig
