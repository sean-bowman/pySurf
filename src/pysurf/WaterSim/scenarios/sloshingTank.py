# -- Sloshing Tank Scenario -- #

'''
Rectangular tank water sloshing scenario.

Initializes a partially filled rectangular container with a tilted
initial water surface to induce sloshing motion. The tilt creates
an asymmetric water column that sloshes back and forth under gravity.

The scenario creates:
1. Fluid particles filling the tank below a tilted free surface
2. Boundary particles lining the left, bottom, and right walls
3. A SimulationConfig with appropriate domain and parameters

Sean Bowman [02/05/2026]
'''

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pysurf import constants
from pysurf.sphBoundaries import ContainerBoundary
from pysurf.sphParticles import ParticleSystem, fillRegion
from pysurf.sphSolver import SimulationConfig

# Boundary particle layers behind each wall
boundaryLayers: int = 3

#--------------------------------------------------------------------#
# -- Sloshing Tank Configuration -- #
#--------------------------------------------------------------------#

@dataclass
class SloshingTankConfig:
    '''
    Configuration for a sloshing tank scenario.

    Parameters:
    -----------
    tankWidth : float
        Tank width [m]
    tankHeight : float
        Tank height [m]
    fillRatio : float
        Water fill ratio (0-1, fraction of tank height)
    initialTiltDeg : float
        Initial water surface tilt angle [degrees].
        Positive tilt means water is higher on the right side.
    particleSpacing : float
        Inter-particle spacing [m]
    smoothingLengthRatio : float
        Ratio h / particleSpacing
    endTime : float
        Simulation end time [s]
    outputInterval : float
        Time between output frames [s]
    '''

    tankWidth: float = 0.5
    tankHeight: float = 0.3
    fillRatio: float = 0.6
    initialTiltDeg: float = 5.0
    particleSpacing: float = 0.005
    smoothingLengthRatio: float = 1.3
    endTime: float = 3.0
    outputInterval: float = 0.02

    @classmethod
    def small2D(cls) -> SloshingTankConfig:
        '''
        Small 2D tank for quick testing.

        ~800 fluid particles, runs in seconds.
        '''
        return cls(
            tankWidth=0.3,
            tankHeight=0.2,
            fillRatio=0.5,
            initialTiltDeg=8.0,
            particleSpacing=0.005,
            endTime=2.0,
        )

    @classmethod
    def standard2D(cls) -> SloshingTankConfig:
        '''
        Standard 2D sloshing tank.

        ~2000 fluid particles, good quality results.
        '''
        return cls(
            tankWidth=0.5,
            tankHeight=0.3,
            fillRatio=0.6,
            initialTiltDeg=5.0,
            particleSpacing=0.004,
            endTime=3.0,
        )

#--------------------------------------------------------------------#
# -- Scenario Creation -- #
#--------------------------------------------------------------------#

def createSloshingTank(tankConfig: SloshingTankConfig) -> tuple[SimulationConfig, ParticleSystem, ContainerBoundary]:

    '''

    Build a sloshing tank: fluid below a tilted free surface, boundary layers on the floor and
    side walls, and the simulation configuration.

    The free surface starts as the line y = fillHeight + tan(tilt) (x - W/2), which projects mostly
    onto the first sloshing mode, so the tank sloshes at close to the first-mode period.

    Parameters:
    -----------
    tankConfig : SloshingTankConfig
        Scenario configuration

    Returns:
    --------
    tuple : (simulation configuration, particle system, container boundary)

    '''

    s = tankConfig.particleSpacing
    width, height = tankConfig.tankWidth, tankConfig.tankHeight
    fillHeight = tankConfig.fillRatio * height
    tilt = math.tan(math.radians(tankConfig.initialTiltDeg))

    containerMin, containerMax = np.array([0.0, 0.0]), np.array([width, height])
    boundary = ContainerBoundary(containerMin, containerMax, s, nLayers = boundaryLayers, openTop = True)

    def belowSurface(points: np.ndarray) -> np.ndarray:
        return points[:, 1] < fillHeight + tilt * (points[:, 0] - width / 2.0)

    fluidPositions = fillRegion(belowSurface, containerMin, containerMax, s)
    particles = ParticleSystem.assemble(fluidPositions, boundary.positions(), s, constants.freshwaterDensity)

    simConfig = SimulationConfig(
        domainMin            = containerMin,
        domainMax            = containerMax,
        particleSpacing      = s,
        smoothingLengthRatio = tankConfig.smoothingLengthRatio,
        referenceDensity     = constants.freshwaterDensity,
        gravity              = np.array([0.0, -constants.gravity]),
        endTime              = tankConfig.endTime,
        outputInterval       = tankConfig.outputInterval,
        dimensions           = 2,
        kernelType           = 'cubicSpline',
        initialSurface       = lambda x: fillHeight + tilt * (x - width / 2.0),
    )

    return (simConfig, particles, boundary)
