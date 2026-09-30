# -- Deviation Visualizer Module -- #

'''
Plotly-based visualizations for mesh comparison results.

Creates heatmaps, overlays, and statistical charts for analyzing
geometry deviations between reference and generated meshes.

Sean Bowman [02/05/2026]
'''

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import trimesh
except ImportError:
    trimesh = None

from pysurf.Surfboard.SurfPhysics.visualization import theme
from pysurf.Surfboard.SurfPhysics.validation.distanceMetrics import DistanceStatistics, RegionalDeviation


#--------------------------------------------------------------------#
# -- Deviation Visualizer -- #
#--------------------------------------------------------------------#

class DeviationVisualizer:
    '''
    Creates visualizations for mesh comparison analysis.

    All plots use consistent dark theme from pysurf.Surfboard.SurfPhysics.visualization.theme.
    '''

    # Diverging colorscale: blue (inside/negative) -> white (zero) -> red (outside/positive)
    DEVIATION_COLORSCALE = [
        [0.0, '#2166AC'],   # Dark blue (most negative)
        [0.25, '#67A9CF'],  # Light blue
        [0.5, '#F7F7F7'],   # White (zero deviation)
        [0.75, '#EF8A62'],  # Light red
        [1.0, '#B2182B'],   # Dark red (most positive)
    ]

    def __init__(self) -> None:
        '''Initialize the visualizer.'''
        if trimesh is None:
            raise ImportError(
                'trimesh is required for deviation visualization. '
                'Install with: pip install trimesh'
            )

    def createDeviationHeatmap(
        self,
        mesh: 'trimesh.Trimesh',
        vertexDistances: np.ndarray,
        title: str = 'Surface Deviation Heatmap',
        clampMm: float | None = None,
    ) -> go.Figure:
        '''
        3D mesh colored by signed distance at each vertex.

        Parameters:
        -----------
        mesh : trimesh.Trimesh
            Mesh to visualize (typically the generated mesh)
        vertexDistances : np.ndarray
            Signed distance for each vertex (same length as mesh.vertices)
        title : str
            Plot title
        clampMm : float | None
            Clamp color scale to +/- this value (auto-detect if None)

        Returns:
        --------
        go.Figure : Interactive 3D Plotly figure
        '''
        vertices = mesh.vertices
        faces = mesh.faces

        # Determine color scale range
        if clampMm is None:
            maxAbs = np.percentile(np.abs(vertexDistances), 99)
            clampMm = max(1.0, maxAbs)  # At least 1mm range

        # Clamp distances for coloring
        clampedDistances = np.clip(vertexDistances, -clampMm, clampMm)

        fig = go.Figure()

        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=clampedDistances,
            colorscale=self.DEVIATION_COLORSCALE,
            cmin=-clampMm,
            cmax=clampMm,
            colorbar=dict(
                title='Deviation (mm)',
                ticksuffix=' mm',
                len=0.75,
            ),
            hovertemplate=(
                'X: %{x:.1f}mm<br>'
                'Y: %{y:.1f}mm<br>'
                'Z: %{z:.1f}mm<br>'
                'Deviation: %{intensity:.2f}mm<extra></extra>'
            ),
        ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='data',
            ),
            template=theme.TEMPLATE,
            height=600,
        )

        return fig

    def createOutlineOverlay(
        self,
        referenceMesh: 'trimesh.Trimesh',
        generatedMesh: 'trimesh.Trimesh',
        nStations: int = 100,
    ) -> go.Figure:
        '''
        Top-view planform comparison (outline overlay).

        Extracts outline by finding maximum Y at each X slice.

        Parameters:
        -----------
        referenceMesh : trimesh.Trimesh
            Reference mesh
        generatedMesh : trimesh.Trimesh
            Generated mesh
        nStations : int
            Number of X stations to sample

        Returns:
        --------
        go.Figure : 2D outline comparison
        '''
        fig = go.Figure()

        for mesh, name, color, dash in [
            (referenceMesh, 'Reference', theme.BLUE, 'solid'),
            (generatedMesh, 'Generated', theme.RED, 'dash'),
        ]:
            vertices = mesh.vertices
            xMin, xMax = vertices[:, 0].min(), vertices[:, 0].max()
            xStations = np.linspace(xMin, xMax, nStations)

            # Extract half-widths at each station
            halfWidths = []
            for x in xStations:
                # Find vertices near this X
                tolerance = (xMax - xMin) / nStations * 0.6
                nearMask = np.abs(vertices[:, 0] - x) < tolerance
                if np.sum(nearMask) > 0:
                    maxY = np.max(np.abs(vertices[nearMask, 1]))
                    halfWidths.append(maxY)
                else:
                    halfWidths.append(0)

            halfWidths = np.array(halfWidths)

            # Plot both rails
            fig.add_trace(go.Scatter(
                x=xStations, y=halfWidths, mode='lines',
                name=f'{name} (right rail)',
                line=dict(color=color, width=2, dash=dash),
            ))
            fig.add_trace(go.Scatter(
                x=xStations, y=-halfWidths, mode='lines',
                name=f'{name} (left rail)',
                line=dict(color=color, width=2, dash=dash),
                showlegend=False,
            ))

        # Centerline
        fig.add_hline(y=0, line=dict(color=theme.REFERENCE_LINE, dash='dot', width=0.5))

        fig.update_layout(
            title='Outline Comparison (Top View)',
            xaxis_title='X Position (mm)',
            yaxis_title='Half-Width (mm)',
            yaxis=dict(scaleanchor='x', scaleratio=1),
            template=theme.TEMPLATE,
            height=400,
        )

        return fig

    def createRockerOverlay(
        self,
        referenceMesh: 'trimesh.Trimesh',
        generatedMesh: 'trimesh.Trimesh',
        nStations: int = 100,
    ) -> go.Figure:
        '''
        Side-view rocker profile comparison.

        Extracts bottom profile by finding minimum Z at each X slice.

        Parameters:
        -----------
        referenceMesh : trimesh.Trimesh
            Reference mesh
        generatedMesh : trimesh.Trimesh
            Generated mesh
        nStations : int
            Number of X stations to sample

        Returns:
        --------
        go.Figure : 2D rocker comparison
        '''
        fig = go.Figure()

        for mesh, name, color, dash in [
            (referenceMesh, 'Reference', theme.BLUE, 'solid'),
            (generatedMesh, 'Generated', theme.RED, 'dash'),
        ]:
            vertices = mesh.vertices
            xMin, xMax = vertices[:, 0].min(), vertices[:, 0].max()
            xStations = np.linspace(xMin, xMax, nStations)

            # Extract minimum Z (bottom) at each station near centerline
            rockerZ = []
            for x in xStations:
                tolerance = (xMax - xMin) / nStations * 0.6
                # Near centerline (|Y| < 20mm)
                nearMask = (np.abs(vertices[:, 0] - x) < tolerance) & (np.abs(vertices[:, 1]) < 20)
                if np.sum(nearMask) > 0:
                    minZ = np.min(vertices[nearMask, 2])
                    rockerZ.append(minZ)
                else:
                    rockerZ.append(np.nan)

            rockerZ = np.array(rockerZ)

            fig.add_trace(go.Scatter(
                x=xStations, y=rockerZ, mode='lines',
                name=name,
                line=dict(color=color, width=2, dash=dash),
            ))

        fig.update_layout(
            title='Rocker Profile Comparison (Side View)',
            xaxis_title='X Position (mm)',
            yaxis_title='Z Position (mm)',
            template=theme.TEMPLATE,
            height=350,
        )

        return fig

    def createCrossSectionOverlay(
        self,
        referenceMesh: 'trimesh.Trimesh',
        generatedMesh: 'trimesh.Trimesh',
        stations: list[float] | None = None,
    ) -> go.Figure:
        '''
        Cross-section comparison at multiple longitudinal stations.

        Parameters:
        -----------
        referenceMesh : trimesh.Trimesh
            Reference mesh
        generatedMesh : trimesh.Trimesh
            Generated mesh
        stations : list[float] | None
            Normalized positions (0-1) along board length (default: 6 stations)

        Returns:
        --------
        go.Figure : Multi-panel cross-section comparison
        '''
        if stations is None:
            stations = [0.1, 0.25, 0.4, 0.6, 0.8, 0.95]

        nRows = 2
        nCols = 3
        fig = make_subplots(
            rows=nRows, cols=nCols,
            subplot_titles=[f't = {t:.0%}' for t in stations],
        )

        colors = theme.STATION_COLORS

        for idx, t in enumerate(stations):
            row = idx // nCols + 1
            col = idx % nCols + 1
            color = colors[idx % len(colors)]

            for mesh, name, dash in [
                (referenceMesh, 'Reference', 'solid'),
                (generatedMesh, 'Generated', 'dash'),
            ]:
                vertices = mesh.vertices
                xMin, xMax = vertices[:, 0].min(), vertices[:, 0].max()
                xTarget = xMin + t * (xMax - xMin)

                # Get cross-section points
                yCoords, zCoords = self._extractCrossSection(mesh, xTarget)

                if len(yCoords) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=yCoords, y=zCoords, mode='lines',
                            name=f'{name} @ {t:.0%}',
                            line=dict(color=color if dash == 'solid' else theme.RED, width=2, dash=dash),
                            showlegend=(idx == 0),  # Only show legend for first station
                        ),
                        row=row, col=col,
                    )

        fig.update_layout(
            title='Cross-Section Comparison',
            template=theme.TEMPLATE,
            height=500,
        )

        # Equal aspect ratio for all subplots
        for i in range(1, nRows * nCols + 1):
            fig.update_xaxes(scaleanchor=f'y{i}' if i > 1 else 'y', scaleratio=1, row=(i-1)//nCols+1, col=(i-1)%nCols+1)

        return fig

    def createDeviationHistogram(
        self,
        distances: np.ndarray,
        stats: DistanceStatistics,
        title: str = 'Signed Distance Distribution',
    ) -> go.Figure:
        '''
        Histogram of signed distances with percentile markers.

        Parameters:
        -----------
        distances : np.ndarray
            Array of signed distances
        stats : DistanceStatistics
            Precomputed statistics
        title : str
            Plot title

        Returns:
        --------
        go.Figure : Histogram with annotations
        '''
        fig = go.Figure()

        # Histogram
        fig.add_trace(go.Histogram(
            x=distances,
            nbinsx=100,
            name='Distance Distribution',
            marker_color=theme.BLUE,
            opacity=0.7,
        ))

        # Percentile lines
        fig.add_vline(x=stats.meanMm, line=dict(color=theme.GREEN, width=2),
                      annotation_text=f'Mean: {stats.meanMm:.2f}mm')
        fig.add_vline(x=-stats.percentile95Mm, line=dict(color=theme.ORANGE, dash='dash', width=1),
                      annotation_text=f'-P95: {-stats.percentile95Mm:.2f}mm')
        fig.add_vline(x=stats.percentile95Mm, line=dict(color=theme.ORANGE, dash='dash', width=1),
                      annotation_text=f'+P95: {stats.percentile95Mm:.2f}mm')

        # Zero reference
        fig.add_vline(x=0, line=dict(color=theme.REFERENCE_LINE, dash='dot', width=1))

        fig.update_layout(
            title=title,
            xaxis_title='Signed Distance (mm)',
            yaxis_title='Count',
            template=theme.TEMPLATE,
            height=350,
            annotations=[
                dict(
                    x=0.02, y=0.98, xref='paper', yref='paper',
                    text=(
                        f'RMS: {stats.rmsMm:.2f}mm<br>'
                        f'Std: {stats.stdMm:.2f}mm<br>'
                        f'Range: [{stats.minMm:.1f}, {stats.maxMm:.1f}]mm'
                    ),
                    showarrow=False, font=dict(size=10),
                    align='left', bgcolor='rgba(0,0,0,0.5)',
                ),
            ],
        )

        return fig

    def createRegionalBarChart(
        self,
        regionalDeviations: list[RegionalDeviation],
        title: str = 'RMS Deviation by Region',
    ) -> go.Figure:
        '''
        Bar chart showing RMS deviation for each board region.

        Parameters:
        -----------
        regionalDeviations : list[RegionalDeviation]
            Per-region deviation statistics
        title : str
            Plot title

        Returns:
        --------
        go.Figure : Grouped bar chart
        '''
        regions = [rd.region for rd in regionalDeviations]
        rmsValues = [rd.stats.rmsMm for rd in regionalDeviations]
        meanValues = [rd.stats.meanMm for rd in regionalDeviations]
        maxAbsValues = [max(abs(rd.stats.minMm), abs(rd.stats.maxMm)) for rd in regionalDeviations]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=regions, y=rmsValues,
            name='RMS',
            marker_color=theme.BLUE,
        ))
        fig.add_trace(go.Bar(
            x=regions, y=meanValues,
            name='Mean',
            marker_color=theme.GREEN,
        ))
        fig.add_trace(go.Bar(
            x=regions, y=maxAbsValues,
            name='Max |Dev|',
            marker_color=theme.RED,
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Board Region',
            yaxis_title='Deviation (mm)',
            barmode='group',
            template=theme.TEMPLATE,
            height=400,
        )

        return fig

    def createComparisonDashboard(
        self,
        referenceMesh: 'trimesh.Trimesh',
        generatedMesh: 'trimesh.Trimesh',
        samplePoints: np.ndarray,
        distances: np.ndarray,
        stats: DistanceStatistics,
        regionalDeviations: list[RegionalDeviation],
    ) -> go.Figure:
        '''
        Combined dashboard with multiple comparison views.

        Parameters:
        -----------
        referenceMesh : trimesh.Trimesh
            Reference mesh
        generatedMesh : trimesh.Trimesh
            Generated mesh
        samplePoints : np.ndarray
            Sample point locations
        distances : np.ndarray
            Signed distances at sample points
        stats : DistanceStatistics
            Overall statistics
        regionalDeviations : list[RegionalDeviation]
            Per-region statistics

        Returns:
        --------
        go.Figure : Multi-panel dashboard figure
        '''
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Outline Comparison',
                'Rocker Profile',
                'Distance Distribution',
                'Regional Deviation',
            ),
            specs=[
                [{}, {}],
                [{}, {}],
            ],
        )

        # 1. Outline comparison (top-left)
        outlineFig = self.createOutlineOverlay(referenceMesh, generatedMesh, nStations=80)
        for trace in outlineFig.data:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=1)

        # 2. Rocker comparison (top-right)
        rockerFig = self.createRockerOverlay(referenceMesh, generatedMesh, nStations=80)
        for trace in rockerFig.data:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=2)

        # 3. Histogram (bottom-left)
        fig.add_trace(go.Histogram(
            x=distances, nbinsx=50,
            marker_color=theme.BLUE, opacity=0.7,
            showlegend=False,
        ), row=2, col=1)

        # 4. Regional bar chart (bottom-right)
        regions = [rd.region for rd in regionalDeviations]
        rmsValues = [rd.stats.rmsMm for rd in regionalDeviations]
        fig.add_trace(go.Bar(
            x=regions, y=rmsValues,
            marker_color=theme.BLUE,
            showlegend=False,
        ), row=2, col=2)

        fig.update_layout(
            title=f'Mesh Comparison Dashboard (RMS: {stats.rmsMm:.2f}mm, Max: {stats.maxMm:.2f}mm)',
            template=theme.TEMPLATE,
            height=800,
        )

        return fig

    def _extractCrossSection(
        self,
        mesh: 'trimesh.Trimesh',
        xPosition: float,
        tolerance: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        '''
        Extract Y-Z profile at a given X position.

        Parameters:
        -----------
        mesh : trimesh.Trimesh
            Source mesh
        xPosition : float
            X coordinate to slice at
        tolerance : float | None
            Width of slice (auto-computed if None)

        Returns:
        --------
        tuple : (yCoords, zCoords) arrays
        '''
        vertices = mesh.vertices

        if tolerance is None:
            xRange = vertices[:, 0].max() - vertices[:, 0].min()
            tolerance = xRange / 200  # 0.5% of length

        # Find vertices near the X position
        nearMask = np.abs(vertices[:, 0] - xPosition) < tolerance
        nearVertices = vertices[nearMask]

        if len(nearVertices) < 3:
            return np.array([]), np.array([])

        # Sort by Y coordinate for a clean profile
        sortIdx = np.argsort(nearVertices[:, 1])
        yCoords = nearVertices[sortIdx, 1]
        zCoords = nearVertices[sortIdx, 2]

        return yCoords, zCoords
