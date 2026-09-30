# -- SPH Engine Package -- #

'''

Core Smoothed Particle Hydrodynamics (SPH) engine.

Provides kernel functions, particle systems, neighbor search,
boundary handling, time integration, and the WCSPH solver.

Sean Bowman [02/05/2026]

'''

from pysurf.WaterSim.sph.protocols import SimulationConfig, SimulationState
from pysurf.WaterSim.sph.kernels import CubicSplineKernel, WendlandC2Kernel, createKernel
from pysurf.WaterSim.sph.waveMaker import WaveMakerConfig, PistonWaveMaker
from pysurf.WaterSim.sph.breakingDetection import BreakingEvent, BreakingDetector
