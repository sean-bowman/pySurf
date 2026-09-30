# -- SurfAnimations Package -- #

'''

Manim-based physics animation module.

Provides animated visualizations of wave theory, surfboard
hydrodynamics, performance comparison, and SPH water simulation.

API usage:
    from pysurf.Surfboard.SurfAnimations import renderScene, SCENES
    renderScene('wave_intro', quality='high')

Sean Bowman [02/05/2026]

'''

from pysurf.Surfboard.SurfAnimations.render import renderScene, renderAll, SCENES
from pysurf.Surfboard.SurfAnimations.scenes.waveIntro import WavePhysicsIntro
from pysurf.Surfboard.SurfAnimations.scenes.boardOnWave import BoardOnWaveSideView, BoardOnWavePerspective
from pysurf.Surfboard.SurfAnimations.scenes.performanceComparison import PerformanceComparison
from pysurf.Surfboard.SurfAnimations.scenes.sloshingTank import SloshingTankAnimation
