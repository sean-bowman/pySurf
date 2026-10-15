
# -- Validation: Potential Flow Breaking and Overturning -- #

'''

The boundary element model of a solitary wave overturning, `pysurf.OverturningWave`, against three
references.

Breaking point: Grilli, Svendsen and Subramanya (1997). They computed solitary waves shoaling on
plane slopes with a fully nonlinear potential flow model, the same class of model as
`pysurf.bemSolver`, and took the breaking point where the front face first turns vertical, the
definition the model here applies directly. For H0/h0 = 0.3 on a 1:35 slope they give Hb/hb = 1.380
and Hb/h0 = 0.476 (tabulated in Table 8 of Senthilkumar 2016, arXiv:1601.06822), so hb/h0 = 0.345.
The acceptance criteria are those applied to the SGN and SPH models: Hb/d within 10 percent and
hb/d within 15 percent. The starting wave differs from Grilli's: theirs is the exact potential flow
solitary wave (Tanaka 1986), this one the exact SGN wave with the vertical velocity structure of
potential flow added, which adjusts by about 1 percent in height over the first depths of travel.

Overturn at jet impact: Feddersen, Hanson, Mostert and Fincham (2024), a two-phase direct numerical
simulation of the soliton of the Surf Ranch experiments of Feddersen et al. (2023): a Green-Naghdi
solitary wave of a0/h0 = 0.6 starting at x0/h0 = 15, a flat bed to x/h0 = 30, a 0.0693 slope up to a
flat at hs/h0 = 0.371, and walls at x/h0 = 0 and 60. At zero wind they measure, at jet impact, the
impact location x/h0 = 42.05, the highest elevation H/h0 = 0.64, the overturn area A_O/H^2 = 0.352
and the aspect ratio W/L = 0.300 of a Longuet-Higgins (1982) curve fitted to the overturn boundary;
Pick and Feddersen (2026) quote A_J/H^2 = 0.22 and theta_O = 33 degrees for the same case. The
acceptance criteria, set before the comparison was run: impact location within 0.5 h0, H within 5
percent, A_O/H^2 and W/L within 15 percent, A_J/H^2 within 20 percent, and theta_O within 5 degrees.
W/L fails: the model gives 0.385. The field-scale measurement at zero wind (Feddersen et al. 2023)
is 0.38 and Pick and Feddersen's potential flow model 0.41, so the DNS sits apart from both.

Overturn scaling across slopes: Pick and Feddersen (2026), 30 potential flow cases of solitary waves
on plane slopes, fitted against psi0 = s / (H0/h0)^(1/4). Five cases within the range of their data
are run on plane beaches ending at a wall in 0.05 h0 of water, the depth of their shallow flat, and
each measure is required to lie within three times the root mean square error of their fit. A_O, W/L,
theta_O and Hb/hb pass in all five. The jet area A_J is 14 to 19 percent above the fit in three of
them, more than three times the fit's 0.005 RMSE, and passes only at H0/h0 = 0.6 and on 1:30.

The SPH continuation (`pysurf.SplashUp`) is checked against the potential flow it continues, on the
same Surf Ranch case: handed to SPH at 48 particles per depth a tenth of a second before the jet
lands, the SPH water must arrive at the impact with the potential flow's surface, no part of either
surface more than one particle spacing from the other. That checks the hand-off and the SPH flow
over the overturn. The flow after the impact is checked in tests/testValidationSplashUp.py.

Run with `pytest -m slow`; the cases take about fifty minutes together.

Sean Bowman [10/12/2026]

'''

import functools
import math

import pytest

import numpy as np

from pysurf.bathymetry import Bathymetry
from pysurf.bemSolver import jetGeometry
from pysurf.OverturningWave import OverturningWave
from pysurf.SplashUp import SplashUp

grilliBreakingIndex = 1.380         # [-] Hb/hb, Grilli et al. (1997), H0/h0 = 0.3, slope 1:35
grilliBreakingHeight = 0.476        # [-] Hb/h0
grilliBreakingDepth = grilliBreakingHeight / grilliBreakingIndex     # [-] hb/h0

# Feddersen et al. (2024), zero wind, at jet impact
dnsImpactX = 42.05                  # [-] x/h0 where the jet lands
dnsImpactHeight = 0.64              # [-] highest elevation over h0
dnsOverturnArea = 0.352             # [-] A_O / H^2
dnsJetArea = 0.22                   # [-] A_J / H^2, as quoted by Pick and Feddersen (2026)
dnsAspectRatio = 0.300              # [-] W / L
dnsOverturnAngle = 33.0             # [deg] theta_O, as quoted by Pick and Feddersen (2026)

def measureBreaking(crestSpacingRatio: float = 0.02) -> dict:

    '''Run the Grilli case and return the breaking summary.'''

    wave = OverturningWave(depth = 1.0, waveHeight = 0.3, beachSlope = 1.0 / 35.0, crestSpacingRatio = crestSpacingRatio)
    wave.run(progress = False)

    return wave.summary()

@functools.lru_cache(maxsize = None)
def surfRanchWave(crestSpacingRatio: float = 0.02) -> OverturningWave:

    '''The soliton of Feddersen et al. (2024), run to jet impact.'''

    slope, shallowDepth = 0.0693, 0.371
    reefLength = 60.0 - (30.0 + (1.0 - shallowDepth) / slope)
    bed = Bathymetry.reefBreak(depth = 1.0, toeX = 30.0, faceSlope = slope, reefDepth = shallowDepth,
                               reefLength = reefLength)
    wave = OverturningWave(depth = 1.0, waveHeight = 0.6, bathymetry = bed, beachSlope = slope, startX = 15.0,
                           crestSpacingRatio = crestSpacingRatio)
    wave.run(progress = False)

    return wave

def measureSurfRanchOverturn(crestSpacingRatio: float = 0.02) -> dict:

    '''The summary of the Surf Ranch soliton at jet impact.'''

    return surfRanchWave(crestSpacingRatio).summary()

@functools.lru_cache(maxsize = None)
def measurePlaneSlopeOverturn(inverseSlope: int, heightRatio: float) -> dict:

    '''Run a solitary wave on a plane slope ending in 0.05 h0 of water and return its summary.'''

    wave = OverturningWave(depth = 1.0, waveHeight = heightRatio, beachSlope = 1.0 / inverseSlope, wallDepthRatio = 0.05)
    wave.run(progress = False)

    return wave.summary()

def pickFeddersenScaling(slope: float, heightRatio: float) -> dict:

    '''

    The fits of Pick and Feddersen (2026), eqs. 3.6 to 3.10, and their root mean square errors, as
    (value, RMSE) for each summary key.

    '''

    psi = slope / heightRatio**0.25

    return {
        'overturnAreaOverImpactHeightSquared': (5.319 * psi - 0.043, 0.013),
        'jetAreaOverImpactHeightSquared': (37.072 * psi**2 - 0.587 * psi + 0.020, 0.005),
        'overturnAspectRatio': (1.661 * psi + 0.298, 0.010),
        'overturnAngle': (-5746.4 * psi**2 + 225.2 * psi + 48.4, 1.2),
        'breakingIndex': (0.871 * math.exp(11.874 * psi), 0.078),
    }

planeSlopeCases = [(30, 0.3), (20, 0.3), (15, 0.3), (20, 0.2), (20, 0.6)]

@pytest.mark.slow
def testBreakingMatchesGrilli():

    summary = measureBreaking()

    assert summary['broke'], summary
    assert abs(summary['breakingHeightOverDepth'] / grilliBreakingHeight - 1.0) < 0.10, summary
    assert abs(summary['breakingDepthOverDepth'] / grilliBreakingDepth - 1.0) < 0.15, summary
    assert abs(summary['breakingIndex'] / grilliBreakingIndex - 1.0) < 0.10, summary

@pytest.mark.slow
def testOverturnMatchesTheDns():

    summary = measureSurfRanchOverturn()

    assert summary['touchedDown'], summary
    assert abs(summary['impactX'] - dnsImpactX) < 0.5, summary
    assert abs(summary['impactHeightOverDepth'] / dnsImpactHeight - 1.0) < 0.05, summary
    assert abs(summary['overturnAreaOverImpactHeightSquared'] / dnsOverturnArea - 1.0) < 0.15, summary
    assert abs(summary['jetAreaOverImpactHeightSquared'] / dnsJetArea - 1.0) < 0.20, summary
    assert abs(summary['overturnAngle'] - dnsOverturnAngle) < 5.0, summary

@pytest.mark.slow
@pytest.mark.xfail(strict = True, reason = 'W/L is 0.385 against the DNS 0.300; the field-scale value is 0.38')
def testOverturnAspectRatioMatchesTheDns():

    summary = measureSurfRanchOverturn()

    assert abs(summary['overturnAspectRatio'] / dnsAspectRatio - 1.0) < 0.15, summary

@pytest.mark.slow
@pytest.mark.parametrize('inverseSlope, heightRatio', planeSlopeCases)
def testOverturnFollowsThePotentialFlowScaling(inverseSlope, heightRatio):

    summary = measurePlaneSlopeOverturn(inverseSlope, heightRatio)
    scaling = pickFeddersenScaling(1.0 / inverseSlope, heightRatio)

    assert summary['touchedDown'], summary
    for key in ('overturnAreaOverImpactHeightSquared', 'overturnAspectRatio', 'overturnAngle', 'breakingIndex'):
        value, rmse = scaling[key]
        assert abs(summary[key] - value) < 3.0 * rmse, (key, summary[key], value)

jetAreaMiss = pytest.mark.xfail(strict = True, reason = 'A_J is 14 to 19 percent above the fit, beyond three RMSE')

@pytest.mark.slow
@pytest.mark.parametrize('inverseSlope, heightRatio', [
    (30, 0.3),
    pytest.param(20, 0.3, marks = jetAreaMiss),
    pytest.param(15, 0.3, marks = jetAreaMiss),
    pytest.param(20, 0.2, marks = jetAreaMiss),
    (20, 0.6),
])
def testJetAreaFollowsThePotentialFlowScaling(inverseSlope, heightRatio):

    summary = measurePlaneSlopeOverturn(inverseSlope, heightRatio)
    value, rmse = pickFeddersenScaling(1.0 / inverseSlope, heightRatio)['jetAreaOverImpactHeightSquared']

    assert abs(summary['jetAreaOverImpactHeightSquared'] - value) < 3.0 * rmse, (summary, value)

@pytest.mark.slow
def testSplashUpCarriesTheOverturnToImpact():

    wave = surfRanchWave()
    impactTime = wave.solver.time
    index = int(np.argmin([abs(frame.time - (impactTime - 0.1)) for frame in wave.frames]))
    lead = impactTime - wave.frames[index].time
    splash = SplashUp(wave, particlesPerDepth = 48, frameIndex = index, duration = lead, frameInterval = lead / 5.0)
    splash.run(progress = False)
    final = wave.solver.surface
    region = (final[np.argmax(final[:, 1]), 0] - wave.depth, jetGeometry(final).impactPoint[0] + wave.depth)
    mismatch = splash.surfaceMismatch(final, region)

    assert mismatch['missingMax'] < 1.0, mismatch
    assert mismatch['strayMax'] < 1.0, mismatch
