# -- SPH Simulation Protocols -- #

'''
Abstract protocols and result dataclasses for SPH simulations.

Defines the core data structures (SimulationConfig, SimulationState)
and solver protocol that all SPH implementations must satisfy.

Sean Bowman [02/05/2026]
'''

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pysurf.WaterSim.sph.particles import ParticleSystem


#--------------------------------------------------------------------#
# -- Simulation Configuration -- #
#--------------------------------------------------------------------#

@dataclass
class SimulationConfig:
    '''
    Configuration for an SPH simulation.

    Defines the domain geometry, particle resolution, numerical
    parameters, and output settings. All values in SI units.

    Parameters:
    -----------
    domainMin : np.ndarray
        Lower corner of simulation domain [m]
    domainMax : np.ndarray
        Upper corner of simulation domain [m]
    particleSpacing : float
        Initial inter-particle spacing [m]
    smoothingLengthRatio : float
        Ratio h / particleSpacing (typically 1.2 - 1.5)
    referenceDensity : float
        Reference fluid density rho_0 [kg/m^3]
    gravity : np.ndarray
        Gravity vector [m/s^2]
    endTime : float
        Simulation end time [s]
    maxTimeStep : float
        Maximum allowed time step [s]
    outputInterval : float
        Time between output frames [s]
    dimensions : int
        Number of spatial dimensions (2 or 3)
    kernelType : str
        Kernel type: 'cubicSpline' or 'wendlandC2'
    '''

    domainMin: np.ndarray
    domainMax: np.ndarray
    particleSpacing: float = 0.005
    smoothingLengthRatio: float = 1.3
    referenceDensity: float = 1000.0
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, -9.81]))
    endTime: float = 3.0
    maxTimeStep: float = 1e-4
    outputInterval: float = 0.02
    dimensions: int = 2
    kernelType: str = 'cubicSpline'

    @property
    def smoothingLength(self) -> float:
        '''Smoothing length h = ratio * spacing [m].'''
        return self.smoothingLengthRatio * self.particleSpacing

    @property
    def supportRadius(self) -> float:
        '''
        Kernel support radius [m].

        For cubic spline and Wendland C2 kernels, the support
        radius is 2h (kernel evaluates to zero beyond this).
        '''
        return 2.0 * self.smoothingLength

    @property
    def domainSize(self) -> np.ndarray:
        '''Domain extent in each dimension [m].'''
        return self.domainMax - self.domainMin

    @classmethod
    def fromJson(cls, configPath: str) -> SimulationConfig:
        '''
        Load configuration from a JSON file.

        Reads the 'simulation', 'sph', and 'fluid' sections
        and constructs a SimulationConfig.

        Parameters:
        -----------
        configPath : str
            Path to the JSON configuration file

        Returns:
        --------
        SimulationConfig : Loaded configuration
        '''
        with open(configPath, 'r') as f:
            data = json.load(f)

        simSection = data.get('simulation', {})
        sphSection = data.get('sph', {})
        fluidSection = data.get('fluid', {})
        tankSection = data.get('tank', {})

        dimensions = simSection.get('dimensions', 2)

        # Build gravity vector based on dimensions
        gravityMag = fluidSection.get('gravity', 9.81)
        if dimensions == 2:
            gravityVec = np.array([0.0, -gravityMag])
        else:
            gravityVec = np.array([0.0, 0.0, -gravityMag])

        # Domain bounds from tank dimensions
        tankWidth = tankSection.get('width', 0.5)
        tankHeight = tankSection.get('height', 0.3)

        if dimensions == 2:
            domainMin = np.array([0.0, 0.0])
            domainMax = np.array([tankWidth, tankHeight])
        else:
            tankDepth = tankSection.get('depth', 0.2)
            domainMin = np.array([0.0, 0.0, 0.0])
            domainMax = np.array([tankWidth, tankDepth, tankHeight])

        return cls(
            domainMin=domainMin,
            domainMax=domainMax,
            particleSpacing=sphSection.get('particleSpacing', 0.005),
            smoothingLengthRatio=sphSection.get('smoothingLengthRatio', 1.3),
            referenceDensity=fluidSection.get('density', 1000.0),
            gravity=gravityVec,
            endTime=simSection.get('endTime', 3.0),
            maxTimeStep=sphSection.get('maxTimeStep', 1e-4),
            outputInterval=simSection.get('outputInterval', 0.02),
            dimensions=dimensions,
            kernelType=sphSection.get('kernelType', 'cubicSpline'),
        )


#--------------------------------------------------------------------#
# -- Simulation State -- #
#--------------------------------------------------------------------#

@dataclass
class SimulationState:
    '''
    Snapshot of the simulation at a given time.

    Captures scalar diagnostics for monitoring simulation
    health (energy conservation, density accuracy, etc.).

    Parameters:
    -----------
    time : float
        Current simulation time [s]
    step : int
        Time step number
    dt : float
        Current time step size [s]
    kineticEnergy : float
        Total kinetic energy of fluid particles [J]
    potentialEnergy : float
        Total gravitational potential energy of fluid particles [J]
    maxVelocity : float
        Maximum particle velocity magnitude [m/s]
    maxDensityError : float
        Maximum relative density error |rho - rho_0| / rho_0
    '''

    time: float
    step: int
    dt: float
    kineticEnergy: float
    potentialEnergy: float
    maxVelocity: float
    maxDensityError: float

    @property
    def totalEnergy(self) -> float:
        '''Total mechanical energy (KE + PE) [J].'''
        return self.kineticEnergy + self.potentialEnergy


#--------------------------------------------------------------------#
# -- Solver Protocol -- #
#--------------------------------------------------------------------#

class SphSolver(Protocol):
    '''Protocol for SPH solvers (WCSPH, ISPH, etc.).'''

    def initialize(self, particles: ParticleSystem) -> None:
        '''Set up initial conditions and data structures.'''
        ...

    def step(self) -> SimulationState:
        '''Advance one time step and return the new state.'''
        ...

    @property
    def currentState(self) -> SimulationState:
        '''Current simulation state snapshot.'''
        ...

    @property
    def particles(self) -> ParticleSystem:
        '''Access the particle system.'''
        ...
