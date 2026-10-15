
# -- pySurf: WaterSim Tests -- #

'''

Checks that WaterSim stands apart from the board model and that its scenarios assemble and step.

WaterSim builds named problems from the flat SPH and wave modules; it may import any shared pySurf
module except the board. The sloshing and wave tank presets are stepped a few times as smoke
tests. They are not validated here: the sloshing period is checked in
tests/testValidationSloshing.py.

Sean Bowman [10/11/2026]

'''

import ast
import os

import numpy as np

import pysurf.WaterSim
from pysurf.sphSolver import WcsphSolver
from pysurf.WaterSim.scenarios.numericalWaveTank import WaveTankConfig, createNumericalWaveTank
from pysurf.WaterSim.scenarios.sloshingTank import SloshingTankConfig, createSloshingTank

waterSimDirectory = os.path.dirname(pysurf.WaterSim.__file__)

def waterSimImports():

    '''Every module name imported anywhere in the WaterSim package, with the file importing it.'''

    for folder, _, fileNames in os.walk(waterSimDirectory):
        for fileName in fileNames:
            if fileName.endswith('.py'):
                path = os.path.join(folder, fileName)
                tree = ast.parse(open(path, encoding = 'utf-8').read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        yield path, node.module
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            yield path, alias.name

def testWaterSimImportsNothingFromTheBoardDomain():

    '''The board can change without the water simulation breaking.'''

    offenders = [(path, module) for path, module in waterSimImports() if module.startswith('pysurf.Surfboard')]

    assert offenders == []

def testWaterSimExposesItsRunner():

    assert hasattr(pysurf.WaterSim, 'WaterSimRunner')
    assert hasattr(pysurf.WaterSim, 'SloshingTankConfig')

def testTheSloshingSolverSteps():

    simConfig, particles, boundary = createSloshingTank(SloshingTankConfig.small2D())
    solver = WcsphSolver(config = simConfig, boundary = boundary)
    solver.initialize(particles)

    for _ in range(5):
        state = solver.step()

    assert state.step == 5
    assert np.all(np.isfinite(particles.positions))
    assert np.all(np.isfinite(particles.velocities))

def testTheSloshingTankStartsWithFluidInsideTheContainer():

    tankConfig = SloshingTankConfig.small2D()
    _, particles, boundary = createSloshingTank(tankConfig)
    fluid = particles.positions[particles.isFluid]

    assert particles.nFluid > 0 and particles.nBoundary > 0
    assert np.all(fluid > boundary.containerMin) and np.all(fluid < boundary.containerMax)

def testTheWaveTankAssembles():

    result = createNumericalWaveTank(WaveTankConfig.small2D())

    assert result.particles.nFluid > 0
    assert np.all(np.isfinite(result.particles.positions))
