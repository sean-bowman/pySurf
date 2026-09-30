# -- Visualizer -- #

'''

Interactive visualization wrapper for surfboard analysis results.

Provides a Visualizer class that creates Plotly dashboards from
the structured results dictionary produced by PhysicsAnalyzer.

Sean Bowman [02/04/2026]

'''

from __future__ import annotations

import plotly.graph_objects as go

from pysurf.Surfboard.SurfPhysics.visualization.dashboard import createAnalysisDashboard

class Visualizer:

    '''

    Interactive visualization of surfboard analysis results.

    Wraps the createAnalysisDashboard() function, extracting the
    required parameters from the PhysicsAnalyzer results dict.

    '''

    def __init__(self) -> None:
        self._figure: go.Figure | None = None

    @property
    def figure(self) -> go.Figure | None:
        '''Last generated Plotly figure.'''
        return self._figure

    def createDashboard(self, results: dict) -> go.Figure:

        '''

        Create and display the 6-panel analysis dashboard.

        Extracts params, waveConditions, and riderMass from the
        results dictionary and delegates to createAnalysisDashboard().

        Parameters:
        -----------
        results : dict
            Structured results from PhysicsAnalyzer.results.
            Must contain keys: 'params', 'waveConditions', 'riderMass'

        Returns:
        --------
        go.Figure : Plotly figure with 6 subplots

        '''
        params = results['params']
        waveConditions = results['waveConditions']
        riderMass = results.get('riderMass', 75.0)

        print()
        print('-' * 62)
        print('  Generating interactive dashboard...')

        self._figure = createAnalysisDashboard(params, waveConditions, riderMass)
        self._figure.show()

        print('  Dashboard opened in browser.')
        print()

        return self._figure
