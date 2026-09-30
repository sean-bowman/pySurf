# -- Numerical Wave Tank Scenario -- #

'''
3D numerical wave tank with piston wave-maker and beach.

Creates a laboratory-style wave flume for studying wave propagation
and breaking. Features:
- Piston-type wave-maker at one end (x=0)
- Flat bottom section for wave propagation
- Sloping beach for wave breaking
- Open top free surface

Tank geometry (x-axis is wave propagation):
    [PISTON] |-------- FLAT --------|----- BEACH -----|
      x=0                          x_slope           x=L

Coordinate convention:
    x: Length (wave propagation direction)
    y: Width (lateral)
    z: Height (vertical, gravity in -z)

References:
-----------
DualSPHysics - Wave generation in numerical tanks
Biesel (1951) - Wave maker theory

Sean Bowman [02/06/2026]
'''

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from pysurf.WaterSim import constants as const
from pysurf.WaterSim.sph.protocols import SimulationConfig
from pysurf.WaterSim.sph.particles import ParticleSystem
from pysurf.WaterSim.sph.boundaryHandling import BoundaryHandler
from pysurf.WaterSim.sph.waveMaker import WaveMakerConfig, PistonWaveMaker
from pysurf.WaterSim.sph.breakingDetection import BreakingDetector


#--------------------------------------------------------------------#
# -- Wave Tank Configuration -- #
#--------------------------------------------------------------------#

@dataclass
class WaveTankConfig:
    '''
    Configuration for a 3D numerical wave tank.

    Parameters:
    -----------
    tankLength : float
        Total tank length in wave direction [m]
    tankWidth : float
        Tank width (lateral) [m]
    stillWaterDepth : float
        Still water depth at flat section [m]
    beachSlope : float
        Beach slope (rise/run), e.g., 0.1 = 1:10 slope
    beachStartRatio : float
        Where beach starts as fraction of tank length (0-1)
    waveHeight : float
        Target wave height [m]
    wavePeriod : float
        Wave period [s]
    particleSpacing : float
        Inter-particle spacing [m]
    smoothingLengthRatio : float
        Ratio h / particleSpacing
    endTime : float
        Simulation end time [s]
    outputInterval : float
        Time between output frames [s]
    dimensions : int
        2 for 2D tank, 3 for full 3D
    '''

    tankLength: float = 5.0
    tankWidth: float = 0.5
    stillWaterDepth: float = 0.5
    beachSlope: float = 0.1  # 1:10 slope
    beachStartRatio: float = 0.6
    waveHeight: float = 0.1
    wavePeriod: float = 1.5
    particleSpacing: float = 0.02
    smoothingLengthRatio: float = 1.3
    endTime: float = 10.0
    outputInterval: float = 0.05
    dimensions: int = 3
    rampCycles: int = 3
    useSecondOrderWaves: bool = True

    @property
    def beachStartX(self) -> float:
        '''X-coordinate where beach slope begins [m].'''
        return self.beachStartRatio * self.tankLength

    @property
    def beachLength(self) -> float:
        '''Length of the beach slope section [m].'''
        return self.tankLength - self.beachStartX

    @property
    def beachRise(self) -> float:
        '''Total height rise of beach [m].'''
        return self.beachLength * self.beachSlope

    def floorHeight(self, x: float) -> float:
        '''
        Get floor height at position x (above z=0 baseline).

        Flat at z=0 for x < beachStartX, then rises linearly.

        Parameters:
        -----------
        x : float
            X-coordinate [m]

        Returns:
        --------
        float : Floor height [m]
        '''
        if x < self.beachStartX:
            return 0.0
        else:
            return (x - self.beachStartX) * self.beachSlope

    def waterDepth(self, x: float) -> float:
        '''
        Get local water depth at position x.

        Parameters:
        -----------
        x : float
            X-coordinate [m]

        Returns:
        --------
        float : Water depth [m]
        '''
        return max(0.0, self.stillWaterDepth - self.floorHeight(x))

    @classmethod
    def small2D(cls) -> WaveTankConfig:
        '''
        Small 2D wave tank for quick testing.

        ~5000 particles, suitable for development.
        '''
        return cls(
            tankLength=2.0,
            tankWidth=0.1,  # Ignored in 2D
            stillWaterDepth=0.3,
            beachSlope=0.15,
            beachStartRatio=0.5,
            waveHeight=0.05,
            wavePeriod=1.0,
            particleSpacing=0.01,
            endTime=5.0,
            dimensions=2,
        )

    @classmethod
    def standard2D(cls) -> WaveTankConfig:
        '''
        Standard 2D wave tank.

        ~20000 particles, good for validation.
        '''
        return cls(
            tankLength=4.0,
            tankWidth=0.1,
            stillWaterDepth=0.4,
            beachSlope=0.1,
            beachStartRatio=0.6,
            waveHeight=0.08,
            wavePeriod=1.2,
            particleSpacing=0.008,
            endTime=8.0,
            dimensions=2,
        )

    @classmethod
    def small3D(cls) -> WaveTankConfig:
        '''
        Small 3D wave tank for development.

        ~50000 particles, runs in reasonable time.
        '''
        return cls(
            tankLength=2.0,
            tankWidth=0.3,
            stillWaterDepth=0.3,
            beachSlope=0.15,
            beachStartRatio=0.5,
            waveHeight=0.05,
            wavePeriod=1.0,
            particleSpacing=0.015,
            endTime=5.0,
            dimensions=3,
        )

    @classmethod
    def standard3D(cls) -> WaveTankConfig:
        '''
        Standard 3D wave tank.

        ~500000 particles, requires significant compute time.
        '''
        return cls(
            tankLength=5.0,
            tankWidth=0.5,
            stillWaterDepth=0.5,
            beachSlope=0.1,
            beachStartRatio=0.6,
            waveHeight=0.1,
            wavePeriod=1.5,
            particleSpacing=0.02,
            endTime=10.0,
            dimensions=3,
        )


#--------------------------------------------------------------------#
# -- Extended Boundary Handler for Beach -- #
#--------------------------------------------------------------------#

class WaveTankBoundaryHandler(BoundaryHandler):
    '''
    Boundary handler with sloping beach floor.

    Extends the standard boundary handler to create a floor
    that is flat up to beachStartX, then rises with constant slope.

    Parameters:
    -----------
    containerMin : np.ndarray
        Lower corner of the container
    containerMax : np.ndarray
        Upper corner of the container
    spacing : float
        Inter-particle spacing
    nLayers : int
        Number of boundary particle layers
    dimensions : int
        Number of spatial dimensions (2 or 3)
    beachStartX : float
        X-coordinate where beach slope begins
    beachSlope : float
        Beach slope (rise/run)
    '''

    def __init__(
        self,
        containerMin: np.ndarray,
        containerMax: np.ndarray,
        spacing: float,
        nLayers: int,
        dimensions: int,
        beachStartX: float,
        beachSlope: float,
    ) -> None:
        super().__init__(
            containerMin=containerMin,
            containerMax=containerMax,
            spacing=spacing,
            nLayers=nLayers,
            dimensions=dimensions,
            openTop=True,
        )
        self._beachStartX = beachStartX
        self._beachSlope = beachSlope

    def _generate2D(
        self, referenceDensity: float
    ) -> tuple[np.ndarray, np.ndarray]:
        '''
        Generate 2D boundary particles with sloping beach floor.
        '''
        s = self._spacing
        xMin, yMin = self._containerMin
        xMax, yMax = self._containerMax

        allPositions: list[np.ndarray] = []

        for layer in range(self._nLayers):
            offset = (layer + 1) * s

            # Floor with slope: generates particles at varying y
            xCoords = np.arange(
                xMin - self._nLayers * s + s / 2.0,
                xMax + self._nLayers * s,
                s,
            )

            # Floor height based on beach slope
            floorY = np.where(
                xCoords < self._beachStartX,
                yMin - offset + s / 2.0,
                yMin - offset + s / 2.0 + (xCoords - self._beachStartX) * self._beachSlope,
            )
            floorPositions = np.column_stack([xCoords, floorY])
            allPositions.append(floorPositions)

            # Left wall: x = xMin - offset, y from yMin to yMax
            yCoords = np.arange(yMin + s / 2.0, yMax, s)
            leftX = np.full_like(yCoords, xMin - offset + s / 2.0)
            leftPositions = np.column_stack([leftX, yCoords])
            allPositions.append(leftPositions)

            # Right wall: x = xMax + offset
            # Wall starts at beach floor height
            beachFloorRight = (xMax - self._beachStartX) * self._beachSlope
            yWallCoords = np.arange(yMin + beachFloorRight + s / 2.0, yMax, s)
            rightX = np.full_like(yWallCoords, xMax + offset - s / 2.0)
            rightPositions = np.column_stack([rightX, yWallCoords])
            allPositions.append(rightPositions)

        positions = np.vstack(allPositions)
        particleVolume = s ** self._dimensions
        mass = referenceDensity * particleVolume
        masses = np.full(len(positions), mass)

        return (positions, masses)

    def _generate3D(
        self, referenceDensity: float
    ) -> tuple[np.ndarray, np.ndarray]:
        '''
        Generate 3D boundary particles with sloping beach floor.
        '''
        s = self._spacing
        xMin, yMin, zMin = self._containerMin
        xMax, yMax, zMax = self._containerMax

        allPositions: list[np.ndarray] = []

        # Extended domain for corners
        xMinExt = xMin - self._nLayers * s
        xMaxExt = xMax + self._nLayers * s
        yMinExt = yMin - self._nLayers * s
        yMaxExt = yMax + self._nLayers * s

        for layer in range(self._nLayers):
            offset = (layer + 1) * s

            # ----------------------------------------------------------------
            # Floor with beach slope
            # ----------------------------------------------------------------
            xCoords = np.arange(xMinExt + s / 2.0, xMaxExt, s)
            yCoords = np.arange(yMinExt + s / 2.0, yMaxExt, s)
            xx, yy = np.meshgrid(xCoords, yCoords, indexing='xy')

            # Floor height varies with x
            floorZ = np.where(
                xx < self._beachStartX,
                zMin - offset + s / 2.0,
                zMin - offset + s / 2.0 + (xx - self._beachStartX) * self._beachSlope,
            )
            floorPositions = np.column_stack([
                xx.ravel(), yy.ravel(), floorZ.ravel()
            ])
            allPositions.append(floorPositions)

            # ----------------------------------------------------------------
            # Left wall (x = xMin - offset), piston side
            # ----------------------------------------------------------------
            yWall = np.arange(yMinExt + s / 2.0, yMaxExt, s)
            zWall = np.arange(zMin + s / 2.0, zMax, s)
            yyWall, zzWall = np.meshgrid(yWall, zWall, indexing='xy')
            leftX = np.full_like(yyWall, xMin - offset + s / 2.0)
            leftPositions = np.column_stack([
                leftX.ravel(), yyWall.ravel(), zzWall.ravel()
            ])
            allPositions.append(leftPositions)

            # ----------------------------------------------------------------
            # Right wall (x = xMax + offset), beach end
            # Beach floor height at right wall
            # ----------------------------------------------------------------
            beachFloorRight = max(0.0, (xMax - self._beachStartX) * self._beachSlope)
            zWallRight = np.arange(zMin + beachFloorRight + s / 2.0, zMax, s)
            yyWallR, zzWallR = np.meshgrid(yWall, zWallRight, indexing='xy')
            rightX = np.full_like(yyWallR, xMax + offset - s / 2.0)
            rightPositions = np.column_stack([
                rightX.ravel(), yyWallR.ravel(), zzWallR.ravel()
            ])
            allPositions.append(rightPositions)

            # ----------------------------------------------------------------
            # Front wall (y = yMin - offset)
            # ----------------------------------------------------------------
            xWall = np.arange(xMin + s / 2.0, xMax, s)
            zWallFront = np.arange(zMin + s / 2.0, zMax, s)
            xxWall, zzFront = np.meshgrid(xWall, zWallFront, indexing='xy')
            frontY = np.full_like(xxWall, yMin - offset + s / 2.0)
            frontPositions = np.column_stack([
                xxWall.ravel(), frontY.ravel(), zzFront.ravel()
            ])
            allPositions.append(frontPositions)

            # ----------------------------------------------------------------
            # Back wall (y = yMax + offset)
            # ----------------------------------------------------------------
            backY = np.full_like(xxWall, yMax + offset - s / 2.0)
            backPositions = np.column_stack([
                xxWall.ravel(), backY.ravel(), zzFront.ravel()
            ])
            allPositions.append(backPositions)

        positions = np.vstack(allPositions)
        particleVolume = s ** self._dimensions
        mass = referenceDensity * particleVolume
        masses = np.full(len(positions), mass)

        return (positions, masses)


#--------------------------------------------------------------------#
# -- Scenario Creation -- #
#--------------------------------------------------------------------#

@dataclass
class WaveTankResult:
    '''
    Result of wave tank creation.

    Contains all objects needed to run a wave tank simulation.

    Parameters:
    -----------
    config : SimulationConfig
        SPH simulation configuration
    particles : ParticleSystem
        Initialized particle system
    boundaryHandler : WaveTankBoundaryHandler
        Boundary handler for the tank
    waveMaker : PistonWaveMaker
        Wave-maker boundary condition
    breakingDetector : BreakingDetector
        Breaking event detector
    tankConfig : WaveTankConfig
        Original tank configuration
    '''
    config: SimulationConfig
    particles: ParticleSystem
    boundaryHandler: WaveTankBoundaryHandler
    waveMaker: PistonWaveMaker
    breakingDetector: BreakingDetector
    tankConfig: WaveTankConfig


def createNumericalWaveTank(
    tankConfig: WaveTankConfig,
) -> WaveTankResult:
    '''
    Create a numerical wave tank simulation from configuration.

    Generates:
    1. Fluid particles filling the tank below still water level
    2. Boundary particles with sloping beach floor
    3. Piston wave-maker on the left wall
    4. Breaking detector configured for wave parameters
    5. SimulationConfig with domain and numerical parameters

    Parameters:
    -----------
    tankConfig : WaveTankConfig
        Wave tank configuration

    Returns:
    --------
    WaveTankResult : Complete simulation setup
    '''
    s = tankConfig.particleSpacing
    L = tankConfig.tankLength
    W = tankConfig.tankWidth
    d = tankConfig.stillWaterDepth
    dim = tankConfig.dimensions

    # Tank height (allow for wave amplitude + freeboard)
    tankHeight = d + tankConfig.waveHeight * 3.0

    # Domain bounds
    if dim == 2:
        containerMin = np.array([0.0, 0.0])
        containerMax = np.array([L, tankHeight])
    else:
        containerMin = np.array([0.0, 0.0, 0.0])
        containerMax = np.array([L, W, tankHeight])

    #--------------------------------------------------------------------#
    # Create boundary handler with beach slope
    #--------------------------------------------------------------------#
    boundaryHandler = WaveTankBoundaryHandler(
        containerMin=containerMin,
        containerMax=containerMax,
        spacing=s,
        nLayers=const.defaultBoundaryLayers,
        dimensions=dim,
        beachStartX=tankConfig.beachStartX,
        beachSlope=tankConfig.beachSlope,
    )

    boundaryPositions, boundaryMasses = boundaryHandler.generateBoundaryParticles(
        referenceDensity=const.referenceDensity,
    )
    nBoundary = len(boundaryPositions)

    #--------------------------------------------------------------------#
    # Create fluid particles below still water level (accounting for beach)
    #--------------------------------------------------------------------#
    fluidPositions = _generateFluidParticles(tankConfig, s)
    nFluid = len(fluidPositions)

    # Particle mass from spacing and density
    particleVolume = s ** dim
    fluidMass = const.referenceDensity * particleVolume

    #--------------------------------------------------------------------#
    # Combine fluid and boundary particles
    #--------------------------------------------------------------------#
    nTotal = nFluid + nBoundary
    positions = np.vstack([fluidPositions, boundaryPositions])
    velocities = np.zeros((nTotal, dim))
    accelerations = np.zeros((nTotal, dim))
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
    # Create wave-maker
    #--------------------------------------------------------------------#
    waveMakerConfig = WaveMakerConfig(
        waveHeight=tankConfig.waveHeight,
        wavePeriod=tankConfig.wavePeriod,
        waterDepth=d,
        rampCycles=tankConfig.rampCycles,
        useSecondOrder=tankConfig.useSecondOrderWaves,
    )
    waveMaker = PistonWaveMaker(waveMakerConfig)

    # Initialize wave-maker with boundary particles at x=0
    waveMaker.initializeBoundaryParticles(
        particles,
        pistonX=0.0,
        tolerance=s * const.defaultBoundaryLayers,
    )

    #--------------------------------------------------------------------#
    # Create breaking detector
    #--------------------------------------------------------------------#
    breakingDetector = BreakingDetector(
        waveSpeed=waveMaker.phaseSpeed,
        velocityThreshold=0.85,
        beachSlope=tankConfig.beachSlope,
        clusterRadius=s * 5.0,
    )

    #--------------------------------------------------------------------#
    # Build simulation config
    #--------------------------------------------------------------------#
    if dim == 2:
        gravityVec = np.array([0.0, -const.gravity])
    else:
        gravityVec = np.array([0.0, 0.0, -const.gravity])

    simConfig = SimulationConfig(
        domainMin=containerMin,
        domainMax=containerMax,
        particleSpacing=s,
        smoothingLengthRatio=tankConfig.smoothingLengthRatio,
        referenceDensity=const.referenceDensity,
        gravity=gravityVec,
        endTime=tankConfig.endTime,
        outputInterval=tankConfig.outputInterval,
        dimensions=dim,
        kernelType='cubicSpline',
    )

    return WaveTankResult(
        config=simConfig,
        particles=particles,
        boundaryHandler=boundaryHandler,
        waveMaker=waveMaker,
        breakingDetector=breakingDetector,
        tankConfig=tankConfig,
    )


def _generateFluidParticles(
    tankConfig: WaveTankConfig,
    spacing: float,
) -> np.ndarray:
    '''
    Generate fluid particles below still water level.

    Accounts for beach slope - no particles where floor
    is above water level.

    Parameters:
    -----------
    tankConfig : WaveTankConfig
        Tank configuration
    spacing : float
        Particle spacing [m]

    Returns:
    --------
    np.ndarray : Fluid particle positions, shape (N, dim)
    '''
    s = spacing
    L = tankConfig.tankLength
    W = tankConfig.tankWidth
    d = tankConfig.stillWaterDepth
    dim = tankConfig.dimensions

    if dim == 2:
        # 2D: particles in (x, y) plane, y is vertical
        xCoords = np.arange(s / 2.0, L, s)
        yCoords = np.arange(s / 2.0, d + s, s)
        xx, yy = np.meshgrid(xCoords, yCoords, indexing='xy')
        candidatePositions = np.column_stack([xx.ravel(), yy.ravel()])

        # Keep particles below still water level and above beach floor
        floorHeight = np.where(
            candidatePositions[:, 0] < tankConfig.beachStartX,
            0.0,
            (candidatePositions[:, 0] - tankConfig.beachStartX) * tankConfig.beachSlope,
        )
        waterSurface = d  # Still water level in 2D y-coordinate
        fluidMask = (candidatePositions[:, 1] < waterSurface) & \
                    (candidatePositions[:, 1] > floorHeight)

        return candidatePositions[fluidMask]

    else:
        # 3D: particles in (x, y, z) space, z is vertical
        xCoords = np.arange(s / 2.0, L, s)
        yCoords = np.arange(s / 2.0, W, s)
        zCoords = np.arange(s / 2.0, d + s, s)
        xx, yy, zz = np.meshgrid(xCoords, yCoords, zCoords, indexing='xy')
        candidatePositions = np.column_stack([
            xx.ravel(), yy.ravel(), zz.ravel()
        ])

        # Floor height varies with x
        floorHeight = np.where(
            candidatePositions[:, 0] < tankConfig.beachStartX,
            0.0,
            (candidatePositions[:, 0] - tankConfig.beachStartX) * tankConfig.beachSlope,
        )
        waterSurface = d  # Still water level in z-coordinate

        fluidMask = (candidatePositions[:, 2] < waterSurface) & \
                    (candidatePositions[:, 2] > floorHeight)

        return candidatePositions[fluidMask]
