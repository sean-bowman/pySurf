
# -- Validation: Energy Lost in Active Breaking -- #

'''

The SPH continuation of a plunging breaker, `pysurf.SplashUp`, against the energy dissipation that
Mostert and Deike (2020) measured in two-phase DNS of solitary waves breaking on plane slopes.

They define the active breaking time as running from the moment the face turns vertical, t1, to
the first moment after it when the kinetic energy has fallen to the potential, t2, and the
dissipation rate per unit crest width as epsilon = dE / (t2 - t1), dE the fall in mechanical energy
over it. Across their slopes of 2 to 5 degrees the rate collapses onto their inertial model,

    (epsilon / epsilon0) (db / d0) = 0.0281 (hb / d0)^3.152,    epsilon0 = rho g^(3/2) d0^(5/2),

hb the wave height at breaking and db the still water depth under the vertical face. For their
plunging case, H0/h0 = 0.3 on a 3 degree beach, their energy budget (figure 8b) falls from 0.96 E0
at t1 to 0.66 E0 at t2, a share of 0.31 of the energy at breaking.

The same case runs here: the boundary element overturn to the jet's impact, then SPH at 48
particles per depth in a window from eight depths behind the crest, behind a wall that stands still
so the window's energy changes only by dissipation, up a dry beach eight depths past the shoreline.
The criteria, set before the comparison was run, allow 30 percent: their data collapse onto the
fit within about 15 percent, and the SPH model's dissipation comes from its artificial viscosity,
not from resolved turbulence. The result: the scaled rate is 7 percent below the fit and the share
lost 11 percent below theirs; at 64 particles per depth both move by under 1 percent.

Run with `pytest -m slow`; the case takes about half an hour.

Sean Bowman [10/14/2026]

'''

import functools
import math

import pytest

from pysurf.bathymetry import Bathymetry
from pysurf.OverturningWave import OverturningWave
from pysurf.SplashUp import SplashUp

mostertShareLost = 0.31         # [-] (0.96 - 0.66) / 0.96, figure 8b, H0/h0 = 0.3 on 3 degrees

@functools.lru_cache(maxsize = None)
def mostertBreaking(particlesPerDepth: int = 48) -> dict:

    '''Run Mostert and Deike's plunging case through active breaking and return its dissipation.'''

    slope = math.tan(math.radians(3.0))
    wave = OverturningWave(depth = 1.0, waveHeight = 0.3, beachSlope = slope, wallDepthRatio = 0.05)
    wave.run(progress = False)
    beach = Bathymetry.planeBeach(depth = 1.0, toeX = wave.toeX, slope = slope, endX = wave.shorelineX + 8.0)
    scale = math.sqrt(1.0 / 9.81)
    splash = SplashUp(wave, particlesPerDepth = particlesPerDepth, behindDepths = 8.0, aheadDepths = 50.0,
                      piston = False, bed = beach, duration = 11.0 * scale, frameInterval = 0.1 * scale)
    splash.run(progress = False)

    return splash.breakingDissipation()

@pytest.mark.slow
def testTheDissipationRateFollowsTheInertialModel():

    result = mostertBreaking()

    assert result['activeEndReached'], result
    assert abs(result['scaledDissipation'] / result['scaledDissipationFit'] - 1.0) < 0.30, result

@pytest.mark.slow
def testTheShareOfEnergyLostMatchesTheDns():

    result = mostertBreaking()

    assert abs(result['energyShareLost'] / mostertShareLost - 1.0) < 0.30, result
