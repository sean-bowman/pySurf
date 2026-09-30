# -- SurfPhysics Package -- #

'''

Surfboard physics simulation module.

Wave theory, hydrodynamic force analysis, and interactive visualization
for parametric surfboard designs.

Sean Bowman [02/03/2026]

'''

__version__ = '0.1.0'

from pysurf.Surfboard.SurfPhysics.analyzer import PhysicsAnalyzer
from pysurf.Surfboard.SurfPhysics.geometry.parameters import SurfboardParameters
from pysurf.Surfboard.SurfPhysics.waves.waveConditions import WaveConditions
from pysurf.Surfboard.SurfPhysics.visualization.visualizer import Visualizer
from pysurf.Surfboard.SurfPhysics.export.viewerExporter import ViewerExporter

# Validation tools
from pysurf.Surfboard.SurfPhysics.validation import (
    MeshComparisonAnalyzer,
    ComparisonResult,
    DistanceStatistics,
)
