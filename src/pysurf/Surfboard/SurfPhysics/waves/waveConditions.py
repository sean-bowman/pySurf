# -- Wave Conditions Dataclass -- #

'''

Defines a wave state for analysis.

Encapsulates wave height, period, depth, and direction with
convenience presets for common surfing conditions.

Sean Bowman [02/03/2026]

'''

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class WaveConditions:

    '''

    Defines a wave state for physics analysis.

    Parameters:
    -----------
    height : float
        Wave height H in meters (trough to crest)
    period : float
        Wave period T in seconds
    depth : float
        Water depth d in meters
    direction : float
        Wave propagation direction in degrees (0 = shore-normal)

    '''

    height: float       # m
    period: float        # s
    depth: float         # m
    direction: float = 0.0  # degrees

    @property
    def angularFrequency(self) -> float:
        '''Angular frequency omega = 2*pi/T [rad/s].'''
        return 2.0 * math.pi / self.period

    @property
    def amplitude(self) -> float:
        '''Wave amplitude a = H/2 [m].'''
        return self.height / 2.0

    @classmethod
    def typicalBeachBreak(cls) -> WaveConditions:

        '''

        Typical surfable beach break conditions.
        H=1.5m, T=10s, d=2.5m: chest-to-shoulder-high surf.

        '''

        return cls(height=1.5, period=10.0, depth=2.5)

    @classmethod
    def smallDay(cls) -> WaveConditions:

        '''

        Mellow small surf conditions.
        H=0.6m, T=8s, d=1.5m: knee-to-waist-high.

        '''

        return cls(height=0.6, period=8.0, depth=1.5)

    @classmethod
    def overhead(cls) -> WaveConditions:

        '''

        Overhead surf conditions.
        H=2.0m, T=12s, d=3.0m: powerful, well-organized swell.

        '''

        return cls(height=2.0, period=12.0, depth=3.0)
