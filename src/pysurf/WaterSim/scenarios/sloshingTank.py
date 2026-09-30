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

from pysurf.WaterSim import constants as const
from pysurf.WaterSim.sph.protocols import SimulationConfig
from pysurf.WaterSim.sph.particles import ParticleSystem
from pysurf.WaterSim.sph.boundaryHandling import BoundaryHandler


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

def createSloshingTank(
    tankConfig: SloshingTankConfig,
) -> tuple[SimulationConfig, ParticleSystem, BoundaryHandler]:
    '''
    Create a sloshing tank simulation from configuration.

    Generates:
    1. Fluid particles below a tilted free surface line
    2. Boundary particles along container walls (open top)
    3. SimulationConfig with domain and numerical parameters

    The tilted free surface is defined as:
        y_surface(x) = fillHeight + tan(tiltAngle) * (x - tankWidth/2)

    Particles below this line are created as fluid; those above
    are empty space.

    Parameters:
    -----------
    tankConfig : SloshingTankConfig
        Scenario configuration

    Returns:
    --------
    tuple[SimulationConfig, ParticleSystem, BoundaryHandler] :
        Ready-to-run configuration, initialized particle system,
        and boundary handler
    '''
    s = tankConfig.particleSpacing
    w = tankConfig.tankWidth
    h = tankConfig.tankHeight
    fillHeight = tankConfig.fillRatio * h
    tiltRad = math.radians(tankConfig.initialTiltDeg)

    # Domain bounds (container interior)
    containerMin = np.array([0.0, 0.0])
    containerMax = np.array([w, h])

    #--------------------------------------------------------------------#
    # Create boundary handler and generate boundary particles
    #--------------------------------------------------------------------#
    boundaryHandler = BoundaryHandler(
        containerMin=containerMin,
        containerMax=containerMax,
        spacing=s,
        nLayers=const.defaultBoundaryLayers,
        dimensions=2,
        openTop=True,
    )

    boundaryPositions, boundaryMasses = boundaryHandler.generateBoundaryParticles(
        referenceDensity=const.referenceDensity,
    )
    nBoundary = len(boundaryPositions)

    #--------------------------------------------------------------------#
    # Create fluid particles below tilted free surface
    #--------------------------------------------------------------------#

    # Generate candidate grid positions inside the container
    xCoords = np.arange(s / 2.0, w, s)
    yCoords = np.arange(s / 2.0, h, s)
    xx, yy = np.meshgrid(xCoords, yCoords, indexing='xy')
    candidatePositions = np.column_stack([xx.ravel(), yy.ravel()])

    # Tilted free surface: y_surface(x) = fillHeight + tan(tilt) * (x - w/2)
    # Particles below this line are fluid
    xCentered = candidatePositions[:, 0] - w / 2.0
    surfaceY = fillHeight + math.tan(tiltRad) * xCentered
    fluidMask = candidatePositions[:, 1] < surfaceY

    fluidPositions = candidatePositions[fluidMask]
    nFluid = len(fluidPositions)

    # Particle mass from spacing and density
    particleVolume = s ** 2
    fluidMass = const.referenceDensity * particleVolume

    #--------------------------------------------------------------------#
    # Combine fluid and boundary particles
    #--------------------------------------------------------------------#

    nTotal = nFluid + nBoundary
    positions = np.vstack([fluidPositions, boundaryPositions])
    velocities = np.zeros((nTotal, 2))
    accelerations = np.zeros((nTotal, 2))
    densities = np.full(nTotal, const.referenceDensity)
    pressures = np.zeros(nTotal)
    masses = np.concatenate([
        np.full(nFluid, fluidMass),
        boundaryMasses,
    ])
    isFluid = np.concatenate([
        np.ones(nFluid, dtype=bool),
        np.zeros(nBoundary, dtype=bool),
    ])

    particles = ParticleSystem(
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        densities=densities,
        pressures=pressures,
        masses=masses,
        isFluid=isFluid,
    )

    #--------------------------------------------------------------------#
    # Build simulation config
    #--------------------------------------------------------------------#

    simConfig = SimulationConfig(
        domainMin=containerMin,
        domainMax=containerMax,
        particleSpacing=s,
        smoothingLengthRatio=tankConfig.smoothingLengthRatio,
        referenceDensity=const.referenceDensity,
        gravity=np.array([0.0, -const.gravity]),
        endTime=tankConfig.endTime,
        outputInterval=tankConfig.outputInterval,
        dimensions=2,
        kernelType='cubicSpline',
    )

    return (simConfig, particles, boundaryHandler)
