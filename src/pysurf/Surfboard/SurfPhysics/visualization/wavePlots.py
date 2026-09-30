# -- Wave Physics Visualizations -- #

'''

Plotly-based interactive plots for wave physics analysis.

Sean Bowman [02/03/2026]

'''

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pysurf.Surfboard.SurfPhysics import constants as const
from pysurf.Surfboard.SurfPhysics.visualization import theme
from pysurf.Surfboard.SurfPhysics.waves.waveConditions import WaveConditions
from pysurf.Surfboard.SurfPhysics.waves.linearWaveTheory import LinearWaveTheory

def plotWaveProfile(
    waveConditions: WaveConditions,
    waveModel: LinearWaveTheory,
    xRange: tuple[float, float] = (0, 50),
    timeS: float = 0.0,
) -> go.Figure:

    '''

    Plot surface elevation eta(x) at a given time.

    Parameters:
    -----------
    waveConditions : WaveConditions
        Wave state
    waveModel : LinearWaveTheory
        Wave theory implementation
    xRange : tuple[float, float]
        Horizontal range in meters
    timeS : float
        Time snapshot in seconds

    Returns:
    --------
    go.Figure : Plotly figure

    '''
    x = np.linspace(xRange[0], xRange[1], 500)
    eta = np.array([waveModel.surfaceElevation(xi, timeS, waveConditions) for xi in x])

    wavelength = waveModel.waveLength(waveConditions)
    phaseSpeed = waveModel.waveSpeed(waveConditions)

    fig = go.Figure()

    # Water surface
    fig.add_trace(go.Scatter(
        x=x, y=eta, mode='lines', name='Surface elevation',
        line=dict(color=theme.BLUE, width=2),
    ))

    # Still water level
    fig.add_hline(y=0, line=dict(color=theme.REFERENCE_LINE, dash='dash', width=1))

    # Seabed
    fig.add_hline(
        y=-waveConditions.depth,
        line=dict(color=theme.BROWN, width=2),
        annotation_text=f'd = {waveConditions.depth:.1f} m',
    )

    fig.update_layout(
        title=f'Wave Profile (H={waveConditions.height:.1f}m, T={waveConditions.period:.0f}s, '
              f'L={wavelength:.1f}m, c={phaseSpeed:.2f}m/s)',
        xaxis_title='Distance (m)',
        yaxis_title='Elevation (m)',
        template=theme.TEMPLATE,
        height=400,
    )

    return fig

def plotWaveKinematics(
    waveConditions: WaveConditions,
    waveModel: LinearWaveTheory,
    xPositionM: float = 0.0,
) -> go.Figure:

    '''

    Velocity and pressure profiles under the wave at a fixed x position.

    Parameters:
    -----------
    waveConditions : WaveConditions
        Wave state
    waveModel : LinearWaveTheory
        Wave theory implementation
    xPositionM : float
        Horizontal position to evaluate at

    Returns:
    --------
    go.Figure : Plotly figure with 3 subplots (u, w, p vs z)

    '''
    d = waveConditions.depth
    zValues = np.linspace(-d, 0, 100)

    # Under the crest (t=0) for maximum values
    u = np.array([waveModel.velocityField(xPositionM, z, 0.0, waveConditions)[0] for z in zValues])
    w = np.array([waveModel.velocityField(xPositionM, z, 0.0, waveConditions)[1] for z in zValues])
    p = np.array([waveModel.pressure(xPositionM, z, 0.0, waveConditions) for z in zValues])

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Horizontal Velocity', 'Vertical Velocity', 'Pressure'),
    )

    fig.add_trace(
        go.Scatter(x=u, y=zValues, mode='lines', name='u(z)',
                   line=dict(color=theme.RED, width=2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=w, y=zValues, mode='lines', name='w(z)',
                   line=dict(color=theme.GREEN, width=2)),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=p / 1000, y=zValues, mode='lines', name='p(z)',
                   line=dict(color=theme.PURPLE, width=2)),
        row=1, col=3,
    )

    fig.update_xaxes(title_text='u (m/s)', row=1, col=1)
    fig.update_xaxes(title_text='w (m/s)', row=1, col=2)
    fig.update_xaxes(title_text='p (kPa)', row=1, col=3)

    for col in range(1, 4):
        fig.update_yaxes(title_text='z (m)', row=1, col=col)

    fig.update_layout(
        title='Wave Kinematics (under crest)',
        template=theme.TEMPLATE,
        height=400,
        showlegend=False,
    )

    return fig

def plotBreakingCriteria(
    depthRange: np.ndarray | None = None,
    periodRange: np.ndarray | None = None,
) -> go.Figure:

    '''

    Plot maximum wave height vs depth showing breaking envelopes.

    Parameters:
    -----------
    depthRange : np.ndarray
        Water depths in meters
    periodRange : np.ndarray
        Wave periods for steepness-limited curves

    Returns:
    --------
    go.Figure : Plotly figure

    '''
    if depthRange is None:
        depthRange = np.linspace(0.5, 10.0, 100)
    if periodRange is None:
        periodRange = np.array([6.0, 8.0, 10.0, 12.0, 14.0])

    fig = go.Figure()

    # Depth-limited breaking: H = 0.78 * d
    hDepthLimited = const.breakingDepthRatio * depthRange
    fig.add_trace(go.Scatter(
        x=depthRange, y=hDepthLimited, mode='lines',
        name='Depth-limited (H/d = 0.78)',
        line=dict(color=theme.RED, width=2),
    ))

    # Steepness-limited breaking: H = L/7 for various periods
    waveModel = LinearWaveTheory()
    colors = theme.PALETTE

    for idx, period in enumerate(periodRange):
        hSteepness = []
        for d in depthRange:
            wc = WaveConditions(height=0.1, period=period, depth=d)
            wavelength = waveModel.waveLength(wc)
            hSteepness.append(wavelength * const.breakingSteepnessRatio)
        fig.add_trace(go.Scatter(
            x=depthRange, y=hSteepness, mode='lines',
            name=f'Steepness (T={period:.0f}s)',
            line=dict(color=colors[idx % len(colors)], width=1, dash='dash'),
        ))

    fig.update_layout(
        title='Wave Breaking Criteria',
        xaxis_title='Water Depth (m)',
        yaxis_title='Maximum Wave Height (m)',
        template=theme.TEMPLATE,
        height=450,
    )

    return fig
