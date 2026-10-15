
# -- Validation: Solitary Wave Breaking on a 1:35 Slope -- #

'''

The SPH breaking point against Grilli et al. (1997).

Grilli, Svendsen and Subramanya computed solitary waves shoaling on plane slopes with a fully
nonlinear potential flow model and took the breaking point where the front face first turns
vertical. For H0/h0 = 0.3 on a 1:35 slope they give a breaking index Hb/hb = 1.380 and a breaking
height Hb/h0 = 0.476 (Grilli et al. 1997, tabulated in Table 8 of Senthilkumar 2016,
arXiv:1601.06822). Together these place the breaking depth at hb/h0 = 0.476 / 1.380 = 0.345.

`pysurf.BreakingWave` runs that case with its defaults: H/d = 0.3, slope 1:35, 16 particles per
depth. `waveBreaking.BreakingTracker` records breaking once the front face leans forward of vertical
by one particle spacing over three consecutive checks, so the reported point follows the vertical-
face instant by the time the lip takes to grow one spacing, which moves it slightly shoreward and
into shallower water than Grilli's definition.

The model fails this comparison at 16 particles per depth:

    quantity    Grilli      SPH, free-slip walls    SPH, viscous walls
    Hb/d        0.476       0.325 (-32 %)           0.262 (-45 %)
    hb/d        0.345       0.097 (-72 %)           0.119 (-66 %)
    Hb/hb       1.380       3.34                    2.21

Numerical damping drains the wave faster than shoaling builds it over the 30 depths of slope it
crosses (tests/testValidationSolitary.py measures the damping on a flat bottom), so it reaches the
shallows lower than it should and breaks late. The test is marked as an expected failure, strict,
so a solver change that brings it within the criteria shows up as an unexpected pass.

Run with `pytest -m slow`; the case takes about 35 minutes.

Sean Bowman [10/11/2026]

'''

import pytest

from pysurf.BreakingWave import BreakingWave

grilliBreakingIndex = 1.380         # [-] Hb/hb, Grilli et al. (1997), H0/h0 = 0.3, slope 1:35
grilliBreakingHeight = 0.476        # [-] Hb/h0
grilliBreakingDepth = grilliBreakingHeight / grilliBreakingIndex     # [-] hb/h0

def measureBreaking(particlesPerDepth: int = 16) -> dict:

    '''Run the Grilli case and return the breaking summary.'''

    wave = BreakingWave(depth = 1.0, waveHeight = 0.3, beachSlope = 1.0 / 35.0, particlesPerDepth = particlesPerDepth)
    wave.run(progress = False)

    return wave.summary()

@pytest.mark.slow
@pytest.mark.xfail(strict = True, reason = 'numerical damping at 16 particles per depth: breaks at hb/d = 0.10, '
                                           'Hb/d = 0.32 against Grilli 0.345 and 0.476')
def testBreakingMatchesGrilli():

    summary = measureBreaking(16)

    assert summary['broke'], summary
    assert summary['predictedBreakerType'] == 'plunging'
    assert abs(summary['breakingHeightOverDepth'] / grilliBreakingHeight - 1.0) < 0.10, summary
    assert abs(summary['breakingDepthOverDepth'] / grilliBreakingDepth - 1.0) < 0.15, summary
