# -- Hydrodynamic Force Visualizations -- #

'''

Plotly-based interactive plots for hydrodynamic force analysis.

Sean Bowman [02/03/2026]

'''

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pysurf.Surfboard.SurfPhysics.hydrodynamics.forceBalance import ForceBalance
from pysurf.Surfboard.SurfPhysics.visualization import theme

def plotDragBreakdown(
    forceBalance: ForceBalance,
    speedRange: np.ndarray | None = None,
    trimAngleDeg: float = 5.0,
) -> go.Figure:

    '''

    Stacked area chart of drag components vs speed.

    Parameters:
    -----------
    forceBalance : ForceBalance
        Force balance model
    speedRange : np.ndarray
        Speeds in m/s (default: 0.5 to 10)
    trimAngleDeg : float
        Fixed trim angle for the breakdown

    Returns:
    --------
    go.Figure : Plotly figure

    '''
    if speedRange is None:
        speedRange = np.linspace(0.5, 10.0, 50)

    breakdown = forceBalance.dragBreakdown(speedRange, trimAngleDeg)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=speedRange, y=breakdown['planing_drag'],
        mode='lines', name='Planing (friction+pressure+spray)',
        fill='tozeroy', line=dict(color=theme.RED),
    ))
    fig.add_trace(go.Scatter(
        x=speedRange, y=breakdown['planing_drag'] + breakdown['form_drag'],
        mode='lines', name='Form drag',
        fill='tonexty', line=dict(color=theme.ORANGE),
    ))
    fig.add_trace(go.Scatter(
        x=speedRange, y=breakdown['total'],
        mode='lines', name='Total drag',
        line=dict(color=theme.WHITE, width=2),
    ))

    fig.update_layout(
        title=f'Drag Breakdown (trim = {trimAngleDeg:.0f} deg)',
        xaxis_title='Speed (m/s)',
        yaxis_title='Drag Force (N)',
        template=theme.TEMPLATE,
        height=400,
    )

    return fig

def plotLiftVsSpeed(
    forceBalance: ForceBalance,
    speedRange: np.ndarray | None = None,
) -> go.Figure:

    '''

    Buoyancy + planing lift vs speed with weight reference line.

    Parameters:
    -----------
    forceBalance : ForceBalance
        Force balance model
    speedRange : np.ndarray
        Speeds in m/s

    Returns:
    --------
    go.Figure : Plotly figure

    '''
    if speedRange is None:
        speedRange = np.linspace(0.5, 10.0, 50)

    curves = forceBalance.performanceCurves(
        minSpeed=speedRange[0], maxSpeed=speedRange[-1], nPoints=len(speedRange)
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=curves['speed'], y=curves['liftN'],
        mode='lines', name='Planing lift',
        line=dict(color=theme.BLUE, width=2),
    ))

    # Weight reference
    fig.add_hline(
        y=forceBalance.totalWeightN,
        line=dict(color=theme.RED, dash='dash', width=1),
        annotation_text=f'Weight: {forceBalance.totalWeightN:.0f} N',
    )

    fig.update_layout(
        title='Lift vs Speed',
        xaxis_title='Speed (m/s)',
        yaxis_title='Lift Force (N)',
        template=theme.TEMPLATE,
        height=400,
    )

    return fig

def plotLiftToDrag(
    forceBalance: ForceBalance,
    speedRange: np.ndarray | None = None,
) -> go.Figure:

    '''

    Lift-to-drag ratio vs speed. Peak L/D indicates optimal surfing speed.

    Parameters:
    -----------
    forceBalance : ForceBalance
        Force balance model
    speedRange : np.ndarray
        Speeds in m/s

    Returns:
    --------
    go.Figure : Plotly figure

    '''
    if speedRange is None:
        speedRange = np.linspace(0.5, 10.0, 50)

    curves = forceBalance.performanceCurves(
        minSpeed=speedRange[0], maxSpeed=speedRange[-1], nPoints=len(speedRange)
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=curves['speed'], y=curves['liftToDrag'],
        mode='lines', name='L/D',
        line=dict(color=theme.GREEN, width=2),
    ))

    # Annotate peak
    peakIdx = int(np.argmax(curves['liftToDrag']))
    peakSpeed = curves['speed'][peakIdx]
    peakLD = curves['liftToDrag'][peakIdx]
    fig.add_annotation(
        x=peakSpeed, y=peakLD,
        text=f'Peak L/D = {peakLD:.1f} at {peakSpeed:.1f} m/s',
        showarrow=True, arrowhead=2,
    )

    fig.update_layout(
        title='Lift-to-Drag Ratio vs Speed',
        xaxis_title='Speed (m/s)',
        yaxis_title='L/D Ratio',
        template=theme.TEMPLATE,
        height=400,
    )

    return fig

def plotPlaningEquilibrium(
    forceBalance: ForceBalance,
    speedRange: np.ndarray | None = None,
) -> go.Figure:

    '''

    Trim angle, wetted length, and drag vs speed.

    Parameters:
    -----------
    forceBalance : ForceBalance
        Force balance model
    speedRange : np.ndarray
        Speeds in m/s

    Returns:
    --------
    go.Figure : Plotly figure with 3 subplots

    '''
    if speedRange is None:
        speedRange = np.linspace(0.5, 10.0, 50)

    curves = forceBalance.performanceCurves(
        minSpeed=speedRange[0], maxSpeed=speedRange[-1], nPoints=len(speedRange)
    )

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=('Trim Angle', 'Wetted Length', 'Total Drag'),
    )

    fig.add_trace(go.Scatter(
        x=curves['speed'], y=curves['trimAngleDeg'],
        mode='lines', name='Trim', line=dict(color=theme.RED, width=2)),
        row=1, col=1,
    )

    fig.add_trace(go.Scatter(
        x=curves['speed'], y=curves['wettedLengthM'] * 100,
        mode='lines', name='Wetted length', line=dict(color=theme.BLUE, width=2)),
        row=2, col=1,
    )

    fig.add_trace(go.Scatter(
        x=curves['speed'], y=curves['dragN'],
        mode='lines', name='Drag', line=dict(color=theme.ORANGE, width=2)),
        row=3, col=1,
    )

    fig.update_yaxes(title_text='Trim (deg)', row=1, col=1)
    fig.update_yaxes(title_text='Wetted Length (cm)', row=2, col=1)
    fig.update_yaxes(title_text='Drag (N)', row=3, col=1)
    fig.update_xaxes(title_text='Speed (m/s)', row=3, col=1)

    fig.update_layout(
        title='Planing Equilibrium vs Speed',
        template=theme.TEMPLATE,
        height=700,
        showlegend=False,
    )

    return fig
