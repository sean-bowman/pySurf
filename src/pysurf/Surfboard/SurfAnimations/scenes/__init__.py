# -- Animation Scenes Subpackage -- #

'''

Manim Scene classes for physics animations.

Each scene is a self-contained Manim animation that can be
rendered via the renderScene() API or the CLI.

'''

from pysurf.Surfboard.SurfAnimations.scenes.waveIntro import WavePhysicsIntro
from pysurf.Surfboard.SurfAnimations.scenes.boardOnWave import BoardOnWaveSideView, BoardOnWavePerspective
from pysurf.Surfboard.SurfAnimations.scenes.performanceComparison import PerformanceComparison
from pysurf.Surfboard.SurfAnimations.scenes.sloshingTank import SloshingTankAnimation
