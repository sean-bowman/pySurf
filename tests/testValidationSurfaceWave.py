
# -- Validation: Serre-Green-Naghdi Breaking and Runup -- #

'''

The free-surface model of a solitary wave on a beach, `pysurf.SurfaceBreakingWave`, against two
published references.

Breaking: Grilli, Svendsen and Subramanya (1997) computed solitary waves shoaling on plane slopes
with a fully nonlinear potential flow model and took the breaking point where the front face first
turns vertical. For H0/h0 = 0.3 on a 1:35 slope they give Hb/hb = 1.380 and Hb/h0 = 0.476
(tabulated in Table 8 of Senthilkumar 2016, arXiv:1601.06822), so hb/h0 = 0.345. The SGN surface
cannot turn vertical; the model starts breaking when its front passes 30 degrees, the value Tissier
et al. (2012) use, taken from Cienfuegos et al.'s (2010) fit to laboratory breaking. The acceptance
criteria are those applied to the SPH model: Hb/d within 10 percent and hb/d within 15 percent.

Runup: Synolakis (1987) derived the maximum runup of a solitary wave that climbs a plane beach
without breaking, R/d = 2.831 sqrt(cot beta) (H/d)^(5/4), and confirmed it with laboratory
measurements on a 1:19.85 slope. The case here is H/d = 0.019 on that slope, S0 = 0.56, well inside
Grilli's no-breaking range. The law comes from linear shallow-water theory and agrees with the
measurements to a few percent at this height; the acceptance criterion is 10 percent.

Run with `pytest -m slow`; each case takes about a minute.

Sean Bowman [10/11/2026]

'''

import math

import pytest

from pysurf.SurfaceBreakingWave import SurfaceBreakingWave

grilliBreakingIndex = 1.380         # [-] Hb/hb, Grilli et al. (1997), H0/h0 = 0.3, slope 1:35
grilliBreakingHeight = 0.476        # [-] Hb/h0
grilliBreakingDepth = grilliBreakingHeight / grilliBreakingIndex     # [-] hb/h0

def measureBreaking(cellsPerDepth: int = 50, breakingAngle: float = 30.0) -> dict:

    '''Run the Grilli case and return the breaking summary.'''

    wave = SurfaceBreakingWave(depth = 1.0, waveHeight = 0.3, beachSlope = 1.0 / 35.0, cellsPerDepth = cellsPerDepth,
                               breakingAngle = breakingAngle)
    wave.run(progress = False)

    return wave.summary()

def synolakisRunup(heightRatio: float, slope: float) -> float:

    '''Synolakis (1987) maximum runup of a non-breaking solitary wave, R/d [-].'''

    return 2.831 * math.sqrt(1.0 / slope) * heightRatio**1.25

def measureRunup(heightRatio: float = 0.019, slope: float = 1.0 / 19.85, cellsPerDepth: int = 50) -> dict:

    '''Run a non-breaking solitary wave up a beach and return its summary with the reference runup.'''

    wave = SurfaceBreakingWave(depth = 1.0, waveHeight = heightRatio, beachSlope = slope, cellsPerDepth = cellsPerDepth,
                               runupAllowance = 0.3)
    wave.run(endTime = 1.5 * wave.estimatedArrivalTime(), progress = False)
    summary = wave.summary()
    summary['referenceRunupOverDepth'] = synolakisRunup(heightRatio, slope)

    return summary

@pytest.mark.slow
def testBreakingMatchesGrilli():

    summary = measureBreaking()

    assert summary['broke'], summary
    assert summary['predictedBreakerType'] == 'plunging'
    assert abs(summary['breakingHeightOverDepth'] / grilliBreakingHeight - 1.0) < 0.10, summary
    assert abs(summary['breakingDepthOverDepth'] / grilliBreakingDepth - 1.0) < 0.15, summary

@pytest.mark.slow
def testNonBreakingRunupMatchesSynolakis():

    summary = measureRunup()

    assert not summary['broke'], summary
    assert abs(summary['maxRunupOverDepth'] / summary['referenceRunupOverDepth'] - 1.0) < 0.10, summary
