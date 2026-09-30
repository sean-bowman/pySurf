# -- Wave Model Protocol -- #

'''

Abstract protocol defining the interface that any wave model must satisfy.

Allows swapping LinearWaveTheory for higher-order models (Stokes, Cnoidal)
without changing consumer code.

Sean Bowman [02/03/2026]

'''

from __future__ import annotations

from typing import Protocol

from pysurf.Surfboard.SurfPhysics.waves.waveConditions import WaveConditions


class WaveModel(Protocol):

    '''

    Protocol for wave physics models.

    Any concrete wave model (linear, Stokes, cnoidal) must implement
    these methods to be used by the analysis pipeline.

    '''

    def waveSpeed(self, waveConditions: WaveConditions) -> float:
        '''Phase speed c in m/s.'''
        ...

    def groupSpeed(self, waveConditions: WaveConditions) -> float:
        '''Group velocity cg in m/s.'''
        ...

    def waveLength(self, waveConditions: WaveConditions) -> float:
        '''Wavelength L in meters.'''
        ...

    def surfaceElevation(
        self, x: float, t: float, waveConditions: WaveConditions
    ) -> float:
        '''Surface elevation eta(x, t) in meters.'''
        ...

    def velocityField(
        self, x: float, z: float, t: float, waveConditions: WaveConditions
    ) -> tuple[float, float]:
        '''(u, w) velocity components in m/s at point (x, z) at time t.'''
        ...

    def pressure(
        self, x: float, z: float, t: float, waveConditions: WaveConditions
    ) -> float:
        '''Dynamic pressure in Pa at point (x, z) at time t.'''
        ...

    def energyDensity(self, waveConditions: WaveConditions) -> float:
        '''Energy per unit surface area E in J/m^2.'''
        ...

    def energyFlux(self, waveConditions: WaveConditions) -> float:
        '''Energy flux (power per unit crest width) P in W/m.'''
        ...

    def isBroken(self, waveConditions: WaveConditions) -> bool:
        '''Whether the wave has broken under the given conditions.'''
        ...
