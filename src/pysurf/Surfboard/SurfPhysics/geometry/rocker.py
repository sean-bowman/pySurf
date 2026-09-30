# -- Surfboard Rocker Profile -- #

'''

Computes the rocker profile (Z-offset of the bottom surface along the board length).

Direct port of SurfboardGeometry/Surfboard/RockerProfile.cs.
Uses composite power curves with a flat spot at 60% from the nose.

Sean Bowman [02/03/2026]

'''

from __future__ import annotations

import math

from pysurf.Surfboard.SurfPhysics.geometry.parameters import SurfboardParameters


class RockerProfile:

    '''

    Surfboard rocker profile: vertical offset of the bottom surface.

    The lowest point (flat spot) is at Z = 0, located at 60% from the nose.
    The nose and tail curve upward from there using power functions.

    Nose region (t < 0.60): z = noseRocker * pow(1 - t/0.60, 2.5)
    Tail region (t >= 0.60): z = tailRocker * pow((t - 0.60)/0.40, 2.0)

    '''

    def __init__(self, params: SurfboardParameters) -> None:

        '''

        Initialize rocker profile from surfboard parameters.

        Parameters:
        -----------
        params : SurfboardParameters
            Board dimensions including rocker heights

        '''

        self._params = params

        # The flat spot is typically at ~60% from the nose
        self._flatSpotT: float = 0.60

    def getRockerHeight(self, t: float) -> float:

        '''

        Get the Z-offset (rocker height) at a normalized longitudinal position.

        Parameters:
        -----------
        t : float
            Normalized position: 0.0 = nose tip, 1.0 = tail tip

        Returns:
        --------
        float : Z-offset in mm (0 = flat spot, positive = upward)

        '''

        t = max(0.0, min(1.0, t))

        if t <= self._flatSpotT:
            # Nose region: rises from flat spot toward nose tip
            # noseT = 0 at flat spot, 1 at nose tip
            noseT = 1.0 - (t / self._flatSpotT)

            # Power curve: steep at nose tip, flat near center
            noseExponent = 2.5
            return self._params.noseRocker * math.pow(noseT, noseExponent)
        else:
            # Tail region: rises from flat spot toward tail tip
            # tailT = 0 at flat spot, 1 at tail tip
            tailT = (t - self._flatSpotT) / (1.0 - self._flatSpotT)

            # Power curve: smoother than nose, gradual kick at tail
            tailExponent = 2.0
            return self._params.tailRocker * math.pow(tailT, tailExponent)
