# -- Simulation Scenarios Package -- #

'''

Pre-configured simulation scenarios for SPH water simulation.

Each scenario provides initial conditions (particle layout,
boundary geometry) and configuration for a specific problem.

Sean Bowman [02/05/2026]

'''

from pysurf.WaterSim.scenarios.sloshingTank import SloshingTankConfig, createSloshingTank
from pysurf.WaterSim.scenarios.numericalWaveTank import (
    WaveTankConfig,
    WaveTankResult,
    WaveTankBoundaryHandler,
    createNumericalWaveTank,
)
