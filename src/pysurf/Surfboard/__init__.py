# -- Surfboard Package -- #

'''

Surfboard design, physics, and visualization toolkit.

Parametric surfboard geometry generation, hydrodynamic analysis,
interactive 3D visualization, and physics animations.

Wildcard import exposes key classes:
    from pysurf.Surfboard import *
    params = SurfboardParameters.shortboard()
    board = BoardGeometry(params)

Sub-modules:
    - SurfPhysics: Parametric geometry, wave theory, hydrodynamics
    - SurfAnimations: Manim animation scenes
    - SurfViewer: Three.js interactive web viewer

Sean Bowman [02/08/2026]

'''

# Parametric geometry
from pysurf.Surfboard.SurfPhysics.geometry.parameters import SurfboardParameters
from pysurf.Surfboard.SurfPhysics.geometry.board import BoardGeometry
from pysurf.Surfboard.SurfPhysics.geometry.surfaceMeshGenerator import (
    SurfaceMeshGenerator,
    RESOLUTION_PRESETS,
)
from pysurf.Surfboard.SurfPhysics.geometry.outline import Outline
from pysurf.Surfboard.SurfPhysics.geometry.rocker import RockerProfile
from pysurf.Surfboard.SurfPhysics.geometry.crossSection import CrossSection

# Physics analysis
from pysurf.Surfboard.SurfPhysics.analyzer import PhysicsAnalyzer
from pysurf.Surfboard.SurfPhysics.visualization.visualizer import Visualizer
from pysurf.Surfboard.SurfPhysics.export.viewerExporter import ViewerExporter

# Validation
from pysurf.Surfboard.SurfPhysics.validation.reverseEngineer import ReverseEngineer

# Animations
from pysurf.Surfboard.SurfAnimations.render import renderScene, renderAll, SCENES

__all__ = [
    # Geometry
    'SurfboardParameters',
    'BoardGeometry',
    'SurfaceMeshGenerator',
    'RESOLUTION_PRESETS',
    'Outline',
    'RockerProfile',
    'CrossSection',
    # Physics
    'PhysicsAnalyzer',
    'Visualizer',
    'ViewerExporter',
    # Validation
    'ReverseEngineer',
    # Animations
    'renderScene',
    'renderAll',
    'SCENES',
]
