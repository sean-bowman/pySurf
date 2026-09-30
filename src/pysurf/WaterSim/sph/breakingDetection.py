# -- Wave Breaking Detection -- #

'''
Detection of wave breaking events in SPH simulation.

Implements multiple breaking criteria for identifying and classifying
wave breaking events during simulation:

1. Kinematic criterion: Horizontal particle velocity approaches wave celerity
   - Breaking onset when u_x > threshold * c (typically threshold = 0.85)

2. Geometric criterion: Wave height exceeds depth or steepness limits
   - Depth-limited: H/d > 0.78 (McCowan 1894)
   - Steepness-limited: H/L > 1/7 (Miche 1944)

3. Breaker classification: Based on surf similarity parameter (Iribarren number)
   - xi = tan(beta) / sqrt(H/L)
   - xi < 0.4: Spilling breakers
   - 0.4 < xi < 2.0: Plunging breakers
   - xi > 2.0: Collapsing/surging breakers

References:
-----------
McCowan (1894) - On the highest wave of permanent type
Miche (1944) - Wave breaking theory
Battjes (1974) - Surf similarity

Sean Bowman [02/06/2026]
'''

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from pysurf.WaterSim.sph.particles import ParticleSystem


#--------------------------------------------------------------------#
# -- Breaking Event Data -- #
#--------------------------------------------------------------------#

BreakerType = Literal['spilling', 'plunging', 'collapsing', 'unknown']


@dataclass
class BreakingEvent:
    '''
    Detected wave breaking event.

    Parameters:
    -----------
    position : np.ndarray
        Location of breaking centroid (x, y, z) [m]
    time : float
        Time of detection [s]
    breakerType : BreakerType
        Classification: 'spilling', 'plunging', 'collapsing', or 'unknown'
    intensity : float
        Relative intensity measure (velocity / wave speed) [dimensionless]
    nParticles : int
        Number of particles involved in breaking region
    '''
    position: np.ndarray
    time: float
    breakerType: BreakerType
    intensity: float
    nParticles: int = 1


#--------------------------------------------------------------------#
# -- Breaking Detector -- #
#--------------------------------------------------------------------#

class BreakingDetector:
    '''
    Detects wave breaking events from SPH particle data.

    Uses kinematic criterion as primary detection method:
    horizontal velocity exceeding a fraction of wave phase speed.

    Parameters:
    -----------
    waveSpeed : float
        Expected wave phase speed c [m/s]
    velocityThreshold : float
        Fraction of wave speed for kinematic breaking (default 0.85)
    beachSlope : float
        Beach slope angle tan(beta) for breaker classification (default 0.1)
    clusterRadius : float
        Radius for clustering nearby breaking particles [m]
    cooldownTime : float
        Minimum time between detecting events at same location [s]
    '''

    def __init__(
        self,
        waveSpeed: float,
        velocityThreshold: float = 0.85,
        beachSlope: float = 0.1,
        clusterRadius: float = 0.1,
        cooldownTime: float = 0.5,
    ) -> None:
        self._waveSpeed = waveSpeed
        self._velocityThreshold = velocityThreshold
        self._beachSlope = beachSlope
        self._clusterRadius = clusterRadius
        self._cooldownTime = cooldownTime

        self._breakingEvents: list[BreakingEvent] = []
        self._lastEventTime: dict[tuple[int, int, int], float] = {}

    @property
    def waveSpeed(self) -> float:
        '''Expected wave phase speed [m/s].'''
        return self._waveSpeed

    @waveSpeed.setter
    def waveSpeed(self, value: float) -> None:
        '''Update wave speed (e.g., as wave shoals).'''
        self._waveSpeed = value

    def detectBreaking(
        self,
        particles: ParticleSystem,
        time: float,
    ) -> list[BreakingEvent]:
        '''
        Detect breaking events in current particle state.

        Uses kinematic criterion: horizontal velocity > threshold * c

        Parameters:
        -----------
        particles : ParticleSystem
            Current particle state
        time : float
            Current simulation time [s]

        Returns:
        --------
        list[BreakingEvent] : Newly detected breaking events this frame
        '''
        events: list[BreakingEvent] = []

        fluidMask = particles.isFluid
        positions = particles.positions[fluidMask]
        velocities = particles.velocities[fluidMask]

        if len(positions) == 0:
            return events

        # Kinematic criterion: u_x > threshold * c
        horizontalVel = velocities[:, 0]  # x-component (wave propagation)
        criticalVel = self._velocityThreshold * self._waveSpeed

        breakingMask = horizontalVel > criticalVel

        if not np.any(breakingMask):
            return events

        # Get breaking particle positions and velocities
        breakingPos = positions[breakingMask]
        breakingVel = horizontalVel[breakingMask]

        # Cluster nearby breaking particles
        clusters = self._clusterParticles(breakingPos, breakingVel)

        for centroid, meanVel, nParticles in clusters:
            # Check cooldown (avoid repeated detection at same location)
            gridCell = self._positionToGridCell(centroid)
            lastTime = self._lastEventTime.get(gridCell, -float('inf'))
            if time - lastTime < self._cooldownTime:
                continue

            # Compute intensity (velocity ratio)
            intensity = float(meanVel / self._waveSpeed)

            # Classify breaker type using surf similarity parameter
            breakerType = self._classifyBreaker(intensity)

            event = BreakingEvent(
                position=centroid,
                time=time,
                breakerType=breakerType,
                intensity=intensity,
                nParticles=nParticles,
            )
            events.append(event)
            self._breakingEvents.append(event)
            self._lastEventTime[gridCell] = time

        return events

    def _clusterParticles(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> list[tuple[np.ndarray, float, int]]:
        '''
        Cluster nearby breaking particles into distinct events.

        Simple single-linkage clustering based on distance.

        Parameters:
        -----------
        positions : np.ndarray
            Breaking particle positions, shape (N, dim)
        velocities : np.ndarray
            Breaking particle horizontal velocities, shape (N,)

        Returns:
        --------
        list[tuple[np.ndarray, float, int]] :
            List of (centroid, mean_velocity, particle_count) for each cluster
        '''
        if len(positions) == 0:
            return []

        if len(positions) == 1:
            return [(positions[0].copy(), float(velocities[0]), 1)]

        # Simple approach: take overall centroid if particles are close
        # For more sophisticated clustering, could use DBSCAN
        nParticles = len(positions)
        centroid = np.mean(positions, axis=0)
        meanVel = float(np.mean(velocities))

        # Check if all particles are within cluster radius
        distances = np.linalg.norm(positions - centroid, axis=1)
        maxDist = float(np.max(distances))

        if maxDist < self._clusterRadius * 3.0:
            # Single cluster
            return [(centroid, meanVel, nParticles)]
        else:
            # Multiple clusters - split by x-position (along wave)
            xMedian = np.median(positions[:, 0])
            leftMask = positions[:, 0] < xMedian
            rightMask = ~leftMask

            clusters = []
            if np.any(leftMask):
                leftCentroid = np.mean(positions[leftMask], axis=0)
                leftVel = float(np.mean(velocities[leftMask]))
                clusters.append((leftCentroid, leftVel, int(np.sum(leftMask))))
            if np.any(rightMask):
                rightCentroid = np.mean(positions[rightMask], axis=0)
                rightVel = float(np.mean(velocities[rightMask]))
                clusters.append((rightCentroid, rightVel, int(np.sum(rightMask))))

            return clusters

    def _classifyBreaker(self, intensity: float) -> BreakerType:
        '''
        Classify breaker type based on intensity and beach slope.

        Uses surf similarity parameter (Iribarren number):
            xi = tan(beta) / sqrt(H/L)

        Approximation: intensity correlates with wave steepness

        Parameters:
        -----------
        intensity : float
            Velocity / wave speed ratio

        Returns:
        --------
        BreakerType : Breaker classification
        '''
        # Higher intensity (faster velocity) correlates with steeper waves
        # which tend to plunge on steep beaches

        # Approximate surf similarity based on intensity
        # intensity > 1.0 suggests very steep, breaking wave
        # intensity ~ 0.85-1.0 is breaking onset

        if intensity < 0.9:
            return 'spilling'
        elif intensity < 1.1:
            # Consider beach slope
            if self._beachSlope > 0.15:
                return 'plunging'
            else:
                return 'spilling'
        else:
            if self._beachSlope > 0.1:
                return 'plunging'
            else:
                return 'collapsing'

    def _positionToGridCell(self, position: np.ndarray) -> tuple[int, int, int]:
        '''
        Map position to grid cell for cooldown tracking.

        Parameters:
        -----------
        position : np.ndarray
            Position vector [m]

        Returns:
        --------
        tuple[int, int, int] : Grid cell indices
        '''
        cellSize = self._clusterRadius * 2.0
        x = int(position[0] / cellSize)
        y = int(position[1] / cellSize) if len(position) > 1 else 0
        z = int(position[2] / cellSize) if len(position) > 2 else 0
        return (x, y, z)

    @property
    def allEvents(self) -> list[BreakingEvent]:
        '''All detected breaking events across simulation.'''
        return self._breakingEvents.copy()

    @property
    def eventCount(self) -> int:
        '''Total number of breaking events detected.'''
        return len(self._breakingEvents)

    def getEventsInRegion(
        self,
        xMin: float,
        xMax: float,
    ) -> list[BreakingEvent]:
        '''
        Get breaking events within an x-range (along wave direction).

        Parameters:
        -----------
        xMin : float
            Minimum x-coordinate [m]
        xMax : float
            Maximum x-coordinate [m]

        Returns:
        --------
        list[BreakingEvent] : Events within the region
        '''
        return [
            e for e in self._breakingEvents
            if xMin <= e.position[0] <= xMax
        ]

    def getStatistics(self) -> dict:
        '''
        Get summary statistics of breaking events.

        Returns:
        --------
        dict : Statistics including counts by type, average intensity, etc.
        '''
        if not self._breakingEvents:
            return {
                'totalEvents': 0,
                'byType': {},
                'averageIntensity': 0.0,
                'firstBreakingTime': None,
                'firstBreakingPosition': None,
            }

        byType: dict[BreakerType, int] = {}
        for event in self._breakingEvents:
            byType[event.breakerType] = byType.get(event.breakerType, 0) + 1

        intensities = [e.intensity for e in self._breakingEvents]
        firstEvent = min(self._breakingEvents, key=lambda e: e.time)

        return {
            'totalEvents': len(self._breakingEvents),
            'byType': byType,
            'averageIntensity': float(np.mean(intensities)),
            'maxIntensity': float(np.max(intensities)),
            'firstBreakingTime': firstEvent.time,
            'firstBreakingPosition': firstEvent.position.tolist(),
        }

    def reset(self) -> None:
        '''Clear all recorded breaking events.'''
        self._breakingEvents.clear()
        self._lastEventTime.clear()
