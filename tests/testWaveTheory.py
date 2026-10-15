
# -- pySurf: Wave Theory Validation -- #

'''

Validation of pysurf.waveTheory against closed-form results.

Each check has an exact reference. The dispersion relation is checked against itself, as a
residual. The wavelength is checked in the deep-water limit, where tanh(kd) -> 1 and
L = g T^2 / (2 pi) exactly, and the phase speed in the shallow-water limit, where
tanh(kd) -> kd and c = sqrt(g d). The group-to-phase speed ratio is checked in both limits
(0.5 and 1.0), the energy density against rho g H^2 / 8, and the kinematics against the
orbital velocity amplitude a omega at the surface in deep water. The solitary wave is checked
against its own closed forms: its volume 2 H / kappa, its celerity, and a velocity field that
satisfies continuity and the kinematic surface condition. The Biesel transfer function is checked
against its shallow and deep limits, and the sloshing frequency against the shallow-water standing
wave.

The breaking criteria are checked for internal consistency: Grilli's no-breaking bound written as
a height and as a slope parameter must agree.

These validate the implementation of the theory. They do not validate linear theory against
measured waves, which it departs from as steepness H/L and relative height H/d grow.

Sean Bowman [10/11/2026]

'''

import math

import numpy as np
import pytest
from scipy.integrate import quad

from pysurf import constants, waveTheory
from pysurf.waveTheory import LinearWaveTheory, WaveConditions

theory = LinearWaveTheory()
g = constants.gravity

#----------------------------------------------------------------------#
# -- Linear Waves -- #
#----------------------------------------------------------------------#

@pytest.mark.parametrize('period', [6.0, 8.0, 10.0, 12.0, 14.0])
@pytest.mark.parametrize('depth', [0.1, 2.5, 1000.0])
def testTheDispersionResidualIsAtMachinePrecision(period, depth):

    omega = 2.0 * math.pi / period
    k = waveTheory.dispersionWavenumber(omega, depth)

    assert abs(omega**2 - g * k * math.tanh(k * depth)) / omega**2 < 1e-12

def testDispersionTakesArrays():

    periods = np.array([4.0, 8.0, 16.0])
    k = waveTheory.dispersionWavenumber(2.0 * np.pi / periods, 3.0)

    assert k.shape == (3,)
    assert k == pytest.approx([waveTheory.dispersionWavenumber(2.0 * np.pi / T, 3.0) for T in periods])

@pytest.mark.parametrize('period', [6.0, 10.0, 14.0])
def testTheDeepWaterWavelengthIsTheClosedForm(period):

    '''At d = 1000 m, kd > 30 for every period here, so tanh(kd) = 1 to double precision.'''

    assert waveTheory.linearWavelength(period, 1000.0) == pytest.approx(waveTheory.deepWaterWavelength(period), rel = 1e-12)
    assert theory.waveLength(WaveConditions(1.0, period, 1000.0)) == pytest.approx(g * period**2 / (2 * math.pi), rel = 1e-12)

@pytest.mark.parametrize('depth', [0.05, 0.1, 0.2])
def testTheShallowWaterPhaseSpeedIsTheClosedForm(depth):

    '''At T = 60 s and d <= 0.2 m, kd < 0.024, so c / sqrt(g d) differs from 1 by under 1e-4.'''

    assert waveTheory.linearPhaseSpeed(60.0, depth) == pytest.approx(waveTheory.shallowWaterSpeed(depth), rel = 1e-4)

def testGroupSpeedReachesItsDeepAndShallowLimits():

    assert waveTheory.linearGroupSpeed(8.0, 1000.0) / waveTheory.linearPhaseSpeed(8.0, 1000.0) == pytest.approx(0.5, rel = 1e-9)
    assert waveTheory.linearGroupSpeed(60.0, 0.1) / waveTheory.linearPhaseSpeed(60.0, 0.1) == pytest.approx(1.0, rel = 1e-4)

def testEnergyDensityIsAnEighthRhoGHSquared():

    waves = WaveConditions.typicalBeachBreak()

    assert theory.energyDensity(waves) == pytest.approx(constants.seawaterDensity * g * waves.height**2 / 8.0)
    assert theory.energyFlux(waves) == pytest.approx(theory.energyDensity(waves) * theory.groupSpeed(waves))

def testDensityIsAnInput():

    waves = WaveConditions.smallDay()

    assert LinearWaveTheory(density = 1000.0).energyDensity(waves) / theory.energyDensity(waves) \
        == pytest.approx(1000.0 / constants.seawaterDensity)

def testDeepWaterSurfaceOrbitalSpeedIsAOmega():

    waves = WaveConditions(height = 1.0, period = 8.0, depth = 1000.0)
    u, w = theory.velocityField(0.0, 0.0, 0.0, waves)

    assert u == pytest.approx(waves.amplitude * waves.angularFrequency, rel = 1e-9)
    assert w == pytest.approx(0.0, abs = 1e-12)

def testPressureIsHydrostaticAtANodeAndAddsTheWaveAtACrest():

    waves = WaveConditions(height = 1.0, period = 8.0, depth = 1000.0)
    k = waveTheory.dispersionWavenumber(waves.angularFrequency, waves.depth)
    node = 0.25 * 2.0 * math.pi / k

    assert theory.pressure(node, -2.0, 0.0, waves) == pytest.approx(constants.seawaterDensity * g * 2.0, rel = 1e-9)
    assert theory.pressure(0.0, 0.0, 0.0, waves) == pytest.approx(constants.seawaterDensity * g * waves.amplitude, rel = 1e-9)

#----------------------------------------------------------------------#
# -- Solitary Waves -- #
#----------------------------------------------------------------------#

def testSolitaryCelerityAndDecayAreTheirClosedForms():

    assert waveTheory.solitaryWaveSpeed(0.3, 1.0) == pytest.approx(math.sqrt(g * 1.3))
    assert waveTheory.solitaryWaveDecay(0.3, 1.0) == pytest.approx(math.sqrt(0.225))

def testTheSolitaryProfilePeaksAtTheCrestWithTheWaveHeight():

    x = np.linspace(-20.0, 20.0, 4001)
    eta = waveTheory.solitaryWaveProfile(x, 0.3, 1.0, crestX = 2.0)

    assert x[np.argmax(eta)] == pytest.approx(2.0, abs = 0.01)
    assert eta.max() == pytest.approx(0.3, rel = 1e-6)
    assert eta == pytest.approx(waveTheory.solitaryWaveProfile(4.0 - x, 0.3, 1.0, crestX = 2.0), rel = 1e-9)

def testTheSolitaryVolumeMatchesItsClosedForm():

    '''The integral of H sech^2(kappa x) over the line is 2 H / kappa.'''

    volume, _ = quad(lambda x: waveTheory.solitaryWaveProfile(x, 0.3, 1.0), -60.0, 60.0, limit = 200)

    assert volume == pytest.approx(2.0 * 0.3 / waveTheory.solitaryWaveDecay(0.3, 1.0), rel = 1e-6)

def testTheSolitaryVelocityCarriesTheWaveFlux():

    '''(d + eta) u = c eta: the column moves the wave's volume at the celerity.'''

    x = np.linspace(-5.0, 5.0, 101)
    eta = waveTheory.solitaryWaveProfile(x, 0.3, 1.0)
    u = waveTheory.solitaryWaveVelocity(x, 0.3, 1.0)

    assert (1.0 + eta) * u == pytest.approx(waveTheory.solitaryWaveSpeed(0.3, 1.0) * eta, rel = 1e-12)

def testTheHalfLengthIsWhereTheProfileFallsToTheFraction():

    halfLength = waveTheory.solitaryWaveHalfLength(0.3, 1.0, 0.01)

    assert waveTheory.solitaryWaveProfile(halfLength, 0.3, 1.0) == pytest.approx(0.003, rel = 1e-9)

#----------------------------------------------------------------------#
# -- Wave Makers and Sloshing -- #
#----------------------------------------------------------------------#

def testBieselReachesKdInShallowWaterAndTwoInDeepWater():

    '''H/S = 2 - 2 (2 kd + 1) / sinh(2 kd) for large kd: 1.7e-7 short of 2 at kd = 10, 1e-15 at kd = 20.'''

    assert waveTheory.pistonTransferFunction(1e-3) == pytest.approx(1e-3, rel = 1e-3)
    assert waveTheory.pistonTransferFunction(10.0) == pytest.approx(2.0 - 2.0 * 21.0 / math.sinh(20.0), rel = 1e-12)
    assert waveTheory.pistonTransferFunction(20.0) == pytest.approx(2.0, rel = 1e-12)
    assert waveTheory.pistonTransferFunction(100.0) == pytest.approx(2.0)

def testBieselIsContinuousAtItsDeepWaterSwitch():

    '''The overflow guard switches to the limit at 2 kd = 50; the two sides must meet.'''

    assert waveTheory.pistonTransferFunction(24.999) == pytest.approx(waveTheory.pistonTransferFunction(25.001), rel = 1e-12)

def testSloshingReachesTheShallowStandingWave():

    '''For a shallow tank, omega_1 = (pi / W) sqrt(g d) to first order in (k d)^2.'''

    width, depth = 10.0, 0.05
    expected = math.pi / width * math.sqrt(g * depth)

    assert waveTheory.sloshingFrequency(width, depth) == pytest.approx(expected, rel = 1e-4)

def testSloshingFrequencyFollowsTheDispersionRelation():

    width, depth = 0.3, 0.1
    k = math.pi / width

    assert waveTheory.sloshingFrequency(width, depth) == pytest.approx(math.sqrt(g * k * math.tanh(k * depth)))
    assert waveTheory.sloshingFrequency(width, depth, mode = 2) > waveTheory.sloshingFrequency(width, depth)

#----------------------------------------------------------------------#
# -- Breaking Criteria -- #
#----------------------------------------------------------------------#

def testMcCowanAndMiche():

    assert waveTheory.isDepthLimitedBreaking(2.0, 2.0)
    assert not waveTheory.isDepthLimitedBreaking(2.0, 10.0)
    assert waveTheory.isSteepnessLimitedBreaking(2.0, 10.0)
    assert not waveTheory.isSteepnessLimitedBreaking(2.0, 100.0)
    assert not theory.isBroken(WaveConditions(height = 2.0, period = 10.0, depth = 10.0))

def testTheDefaultCasePlunges():

    '''H/d = 0.3 on a 1:35 slope: S0 = 1.521 / 35 / sqrt(0.3) = 0.0793, inside the plunging band.'''

    slopeParameter = waveTheory.solitarySlopeParameter(1.0 / 35.0, 0.3, 1.0)

    assert slopeParameter == pytest.approx(1.521 / 35.0 / math.sqrt(0.3))
    assert waveTheory.solitaryBreakerType(slopeParameter) == 'plunging'

@pytest.mark.parametrize('slopeParameter, expected', [(0.01, 'spilling'), (0.1, 'plunging'), (0.33, 'surging'), (0.5, 'none')])
def testGrilliBands(slopeParameter, expected):

    assert waveTheory.solitaryBreakerType(slopeParameter) == expected

@pytest.mark.parametrize('slope', [1.0 / 8.0, 1.0 / 19.85, 1.0 / 35.0])
def testTheNoBreakingHeightIsTheSlopeParameterBound(slope):

    '''H/d = 16.9 tan^2(beta) is exactly where S0 reaches 0.37.'''

    heightRatio = waveTheory.solitaryNonBreakingHeight(slope)

    assert waveTheory.solitarySlopeParameter(slope, heightRatio, 1.0) == pytest.approx(0.37, rel = 1e-12)

def testIribarrenBands():

    xi = waveTheory.iribarrenNumber(0.1, 1.0, 100.0)

    assert xi == pytest.approx(1.0)
    assert waveTheory.iribarrenBreakerType(0.3) == 'spilling'
    assert waveTheory.iribarrenBreakerType(1.0) == 'plunging'
    assert waveTheory.iribarrenBreakerType(4.0) == 'surging'
    assert waveTheory.iribarrenBreakerType(0.45, basis = 'breaking') == 'plunging'
    with pytest.raises(ValueError):
        waveTheory.iribarrenBreakerType(1.0, basis = 'midway')

#----------------------------------------------------------------------#
# -- Serre-Green-Naghdi -- #
#----------------------------------------------------------------------#

def testTheSerreSolitaryWaveIsWiderThanTheFirstOrderWave():

    '''Same height and celerity; decay smaller by sqrt(d / (d + H)).'''

    ratio = waveTheory.serreSolitaryWaveDecay(0.3, 1.0) / waveTheory.solitaryWaveDecay(0.3, 1.0)

    assert ratio == pytest.approx(math.sqrt(1.0 / 1.3))
    assert waveTheory.serreSolitaryWaveProfile(2.0, 0.3, 1.0, crestX = 2.0) == pytest.approx(0.3)
    assert waveTheory.serreSolitaryWaveHalfLength(0.3, 1.0) == pytest.approx(waveTheory.solitaryWaveHalfLength(0.3, 1.0) / ratio)

def testTheSerreSolitaryVelocityCarriesAConstantFluxInTheMovingFrame():

    '''h (u - c) = -c d in every column of a wave of permanent form.'''

    x = np.linspace(-8.0, 8.0, 81)
    eta = waveTheory.serreSolitaryWaveProfile(x, 0.4, 1.0)
    u = waveTheory.serreSolitaryWaveVelocity(x, 0.4, 1.0)
    c = waveTheory.solitaryWaveSpeed(0.4, 1.0)

    assert (1.0 + eta) * (u - c) == pytest.approx(np.full_like(x, -c), rel = 1e-12)

def testSerreDispersionMatchesAiryInLongWavesAndFallsBelowInShortOnes():

    k = np.array([0.01, 1.0, 3.0])
    sgn = waveTheory.serreAngularFrequency(k, 1.0)
    airy = np.sqrt(g * k * np.tanh(k))

    assert sgn[0] == pytest.approx(airy[0], rel = 1e-5)
    assert sgn[1] / airy[1] == pytest.approx(0.9924, abs = 1e-4)
    assert sgn[2] / airy[2] == pytest.approx(0.8682, abs = 1e-4)
