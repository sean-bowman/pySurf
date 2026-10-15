# -- WaterSim Package -- #

'''

Ready-to-run SPH water simulations: scenarios, the command-line runner, and frame export.

The physics lives in pySurf's flat modules, which any simulation can use on its own:

    waveTheory      closed-form wave mechanics and breaking criteria
    bathymetry      bed profiles
    sphKernels, sphNeighbors, sphParticles, sphBoundaries, sphSolver    the WCSPH engine
    waveMakers      piston wave maker and solitary wave initial condition
    waveBreaking    free surface, front face lean and runup measures

This package assembles them into named problems:

    scenarios/  sloshing tank and numerical wave tank
    runner.py   `python -m pysurf.WaterSim`
    export/     frame data as JSON

The breaking solitary wave on a beach is `pysurf.BreakingWave`.

Sean Bowman [02/05/2026]

'''

__version__ = '0.1.0'

from pysurf.WaterSim.runner import WaterSimRunner
from pysurf.WaterSim.scenarios.sloshingTank import SloshingTankConfig
from pysurf.WaterSim.export.frameExporter import FrameExporter
