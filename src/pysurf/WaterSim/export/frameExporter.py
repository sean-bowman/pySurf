# -- Simulation Frame Exporter -- #

'''
Exports SPH simulation frames as JSON for visualization.

Collects particle state snapshots during simulation and writes
them to a JSON file that can be loaded by Manim animations
or the Three.js viewer.

The output format stores particle positions, velocity magnitudes,
and energy history for each frame, along with simulation metadata.

Sean Bowman [02/05/2026]
'''

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np

from pysurf.WaterSim.sph.protocols import SimulationConfig, SimulationState
from pysurf.WaterSim.sph.particles import ParticleSystem


class FrameExporter:
    '''
    Collects and exports simulation frame data as JSON.

    Usage:
        exporter = FrameExporter()
        # During simulation loop:
        exporter.addFrame(state, particles)
        # After simulation:
        exporter.export(config, outputDir='output')

    Output JSON format:
    {
        "meta": { "type": "waterSim", "dimensions": 2, "created": "...", ... },
        "config": { "tankWidth": 0.5, ... },
        "frames": [
            {
                "time": 0.0,
                "positions": [[x0, y0], [x1, y1], ...],
                "velocityMagnitudes": [v0, v1, ...],
                "densities": [rho0, rho1, ...]
            },
            ...
        ],
        "energy": {
            "times": [...],
            "kinetic": [...],
            "potential": [...],
            "total": [...]
        }
    }
    '''

    def __init__(self) -> None:
        self._frames: list[dict] = []
        self._energyHistory: dict[str, list[float]] = {
            'times': [],
            'kinetic': [],
            'potential': [],
            'total': [],
        }

    @property
    def nFrames(self) -> int:
        '''Number of collected frames.'''
        return len(self._frames)

    def addFrame(self, state: SimulationState, particles: ParticleSystem) -> None:
        '''
        Record a simulation frame.

        Only fluid particle data is stored (boundary particles
        are static and can be reconstructed from config).

        Parameters:
        -----------
        state : SimulationState
            Current simulation state diagnostics
        particles : ParticleSystem
            Current particle system
        '''
        fluidMask = particles.isFluid
        fluidPositions = particles.positions[fluidMask]
        fluidVelocities = particles.velocities[fluidMask]
        fluidDensities = particles.densities[fluidMask]

        # Velocity magnitudes for color mapping
        velMagnitudes = np.linalg.norm(fluidVelocities, axis=1)

        frame = {
            'time': round(state.time, 6),
            'positions': fluidPositions.tolist(),
            'velocityMagnitudes': np.round(velMagnitudes, 6).tolist(),
            'densities': np.round(fluidDensities, 2).tolist(),
        }
        self._frames.append(frame)

        # Track energy history
        self._energyHistory['times'].append(round(state.time, 6))
        self._energyHistory['kinetic'].append(round(state.kineticEnergy, 6))
        self._energyHistory['potential'].append(round(state.potentialEnergy, 6))
        self._energyHistory['total'].append(round(state.totalEnergy, 6))

    def export(
        self,
        config: SimulationConfig,
        outputDir: str = 'src/pysurf/WaterSim/output',
        scenarioName: str = 'sloshing',
    ) -> str:
        '''
        Write all collected frames to a JSON file.

        Parameters:
        -----------
        config : SimulationConfig
            Simulation configuration for metadata
        outputDir : str
            Output directory path
        scenarioName : str
            Scenario name for the filename

        Returns:
        --------
        str : Path to the exported JSON file
        '''
        os.makedirs(outputDir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'waterSim_{scenarioName}_{timestamp}.json'
        filepath = os.path.join(outputDir, filename)

        output = {
            'meta': {
                'type': 'waterSim',
                'dimensions': config.dimensions,
                'nFrames': len(self._frames),
                'nFluidParticles': len(self._frames[0]['positions']) if self._frames else 0,
                'particleSpacing': config.particleSpacing,
                'created': datetime.now().isoformat(),
            },
            'config': {
                'domainMin': config.domainMin.tolist(),
                'domainMax': config.domainMax.tolist(),
                'particleSpacing': config.particleSpacing,
                'referenceDensity': config.referenceDensity,
                'endTime': config.endTime,
            },
            'frames': self._frames,
            'energy': self._energyHistory,
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=None, separators=(',', ':'))

        return filepath
