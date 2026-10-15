
# -- Numerical Wave Tank Scenario -- #

'''

A laboratory-style wave flume: a piston wave maker at x = 0, a flat bottom, and a plane beach the
waves shoal and break on. 2D, or 3D across a channel of the tank's width.

    [PISTON] |-------- FLAT --------|----- BEACH -----|
      x = 0                     beachStartX          x = L

Coordinates: x along the tank (wave travel), y across it in 3D, and the vertical last.

The floor is a `pysurf.bathymetry.Bathymetry`, the walls a `pysurf.sphBoundaries.BedBoundary`, the
paddle a `pysurf.waveMakers.PistonWaveMaker`, and breaking is watched by a
`pysurf.waveBreaking.BreakingDetector` whose breaker type comes from the offshore Iribarren number.

References:
-----------
DualSPHysics -- Wave generation in numerical tanks
Biesel (1951) -- Wave maker theory
Battjes (1974) -- Surf similarity

Sean Bowman [02/06/2026]

'''

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pysurf import constants, waveTheory
from pysurf.bathymetry import Bathymetry
from pysurf.sphBoundaries import BedBoundary
from pysurf.sphParticles import ParticleSystem, fillRegion
from pysurf.sphSolver import SimulationConfig
from pysurf.waveBreaking import BreakingDetector
from pysurf.waveMakers import PistonWaveMaker, WaveMakerConfig

# Boundary particle layers behind each wall
boundaryLayers: int = 3

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

    @property
    def bathymetry(self) -> Bathymetry:
        '''The tank floor: flat to beachStartX, then the beach slope to the end of the tank.'''
        return Bathymetry.planeBeach(depth = self.stillWaterDepth, toeX = self.beachStartX,
                                     slope = self.beachSlope, endX = self.tankLength)

    def floorHeight(self, x: float) -> float:
        '''Floor elevation above the flat bottom at x [m].'''
        return float(self.bathymetry.elevation(x))

    def waterDepth(self, x: float) -> float:
        '''Still water depth at x [m], zero on the dry beach.'''
        return float(self.bathymetry.depth(x))

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
# -- Scenario Creation -- #
#--------------------------------------------------------------------#

@dataclass
class WaveTankResult:

    '''

    Everything a wave tank run needs.

    Parameters:
    -----------
    config : SimulationConfig
        SPH simulation configuration
    particles : ParticleSystem
        Initialized particle system
    boundary : BedBoundary
        Floor, beach and walls
    waveMaker : PistonWaveMaker
        The paddle
    breakingDetector : BreakingDetector
        Kinematic breaking detector
    tankConfig : WaveTankConfig
        The configuration the tank was built from

    '''

    config: SimulationConfig
    particles: ParticleSystem
    boundary: BedBoundary
    waveMaker: PistonWaveMaker
    breakingDetector: BreakingDetector
    tankConfig: WaveTankConfig

def createNumericalWaveTank(tankConfig: WaveTankConfig) -> WaveTankResult:

    '''

    Build a numerical wave tank: fluid below still water over the floor and beach, the boundary,
    the paddle, the breaking detector, and the simulation configuration.

    Parameters:
    -----------
    tankConfig : WaveTankConfig
        Tank configuration

    Returns:
    --------
    WaveTankResult : Ready to run

    '''

    s, dim = tankConfig.particleSpacing, tankConfig.dimensions
    d, length, width = tankConfig.stillWaterDepth, tankConfig.tankLength, tankConfig.tankWidth
    bathymetry = tankConfig.bathymetry

    # Room above still water for the wave crests
    tankHeight = d + 3.0 * tankConfig.waveHeight

    boundary = BedBoundary(bathymetry, s, nLayers = boundaryLayers, wallTop = tankHeight,
                           width = width if dim == 3 else None)

    def underStillWater(points: np.ndarray) -> np.ndarray:
        bed = bathymetry.elevation(points[:, 0])
        return (points[:, -1] >= bed + (0.5 - 1e-9) * s) & (points[:, -1] < d)

    if dim == 2:
        lower, upper = [0.0, 0.0], [length, d]
    else:
        lower, upper = [0.0, 0.0, 0.0], [length, width, d]
    fluidPositions = fillRegion(underStillWater, lower, upper, s)
    particles = ParticleSystem.assemble(fluidPositions, boundary.positions(), s, constants.freshwaterDensity)

    waveMaker = PistonWaveMaker(WaveMakerConfig(
        waveHeight     = tankConfig.waveHeight,
        wavePeriod     = tankConfig.wavePeriod,
        waterDepth     = d,
        rampCycles     = tankConfig.rampCycles,
        useSecondOrder = tankConfig.useSecondOrderWaves,
    ))
    waveMaker.initializeBoundaryParticles(particles, pistonX = 0.0, tolerance = s * boundaryLayers)

    # The breaker type belongs to the incident wave and the slope: Battjes' offshore Iribarren number
    deepWavelength = float(waveTheory.deepWaterWavelength(tankConfig.wavePeriod))
    xi = waveTheory.iribarrenNumber(tankConfig.beachSlope, tankConfig.waveHeight, deepWavelength)
    breakingDetector = BreakingDetector(
        waveSpeed         = waveMaker.phaseSpeed,
        velocityThreshold = 0.85,
        breakerType       = waveTheory.iribarrenBreakerType(xi, basis = 'offshore'),
        clusterRadius     = 5.0 * s,
    )

    gravity = np.zeros(dim)
    gravity[-1] = -constants.gravity
    containerMax = np.array([length, tankHeight]) if dim == 2 else np.array([length, width, tankHeight])

    simConfig = SimulationConfig(
        domainMin            = np.zeros(dim),
        domainMax            = containerMax,
        particleSpacing      = s,
        smoothingLengthRatio = tankConfig.smoothingLengthRatio,
        referenceDensity     = constants.freshwaterDensity,
        gravity              = gravity,
        endTime              = tankConfig.endTime,
        outputInterval       = tankConfig.outputInterval,
        dimensions           = dim,
        kernelType           = 'cubicSpline',
        initialSurface       = d,
    )

    return WaveTankResult(config = simConfig, particles = particles, boundary = boundary, waveMaker = waveMaker,
                          breakingDetector = breakingDetector, tankConfig = tankConfig)
