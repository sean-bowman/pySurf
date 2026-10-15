
# -- Validation: Waves Overturning on a Submerged Reef -- #

'''

`pysurf.ReefOverturn` against the flume of Blenkinsopp and Chaplin (2008): regular waves of 0.6 Hz,
H0 = 75.5 mm in deep water (H0/L0 = 0.0174), breaking over a 1:10 reef face 0.618 m high in 0.68
to 0.72 m of water. The SGN model carries the train over the reef; the boundary element model takes
the last wave to break from a third of a period before SGN breaks it to the jet's impact.

The reference values are their points for H0/L0 = 0.0174, digitized from their figures by marker
color: the incident height Hi from hc/Hi in figure 10 (68.4 to 69.6 mm over the nine water levels),
the transmission coefficient Kt (figure 10), the reflection coefficient Kr (figure 11), the cavity's
length over width lc/wc (figure 6) and its area over the breaking height squared Ac/Hb^2 (figure 7).
They found the breaker depth index Hb/hb near 0.85 at every submergence and steepness, and the
breaker height index Hb/H0 to follow 0.5 (H0/L0)^-0.2, 1.124 at this steepness.

The criteria follow the errors they state: 20 percent on the cavity area, 10 percent on each length
and so 15 percent on their ratio; 20 percent on the breaker indices and on Kr and Kt, about twice
the scatter of their points about their trends; 5 percent on the incident height. Three water
levels span their range: hc = 62, 77 and 92 mm.

    quantity        hc = 62 mm              hc = 77 mm              hc = 92 mm
                    model   flume   error   model   flume   error   model   flume   error
    Hi [mm]         66.6    69.0    -3%     68.1    68.7    -1%     68.4    69.0    -1%
    Kr              0.256   0.122   +111%   0.329   0.161   +104%   0.287   0.192   +50%
    Kt              0.773   0.537   +44%    0.811   0.653   +24%    0.885   0.796   +11%
    Hb / hb         0.99    0.85    +16%    1.20    0.85    +41%    1.16    0.85    +37%
    Hb / H0         1.23    1.12    +9%     1.54    1.12    +37%    1.63    1.12    +45%
    lc / wc         2.29    2.10    +9%     2.62    1.90    +38%    4.00    1.56    +157%
    Ac / Hb^2       0.42    0.22    +88%    0.45    0.20    +125%   0.024   0.116   -80%

The incident wave matches the flume. Everything after it does not: the SGN train reflects 1.5 to
2.1 times what the flume reflected and passes more over the reef, so by the linear balance
1 - Kr^2 - Kt^2 its breaking dissipates 0.41 to 0.48 of the energy the flume's did, and the wave
the boundary element model receives breaks 16 to 41 percent higher against the depth than the
flume's. Those failures are marked as expected so a change that closes one shows up.

Run with `pytest -m slow`; the three cases take about seven minutes.

Sean Bowman [10/15/2026]

'''

import functools

import pytest

from pysurf.ReefOverturn import ReefOverturn

# hc [mm]: (Hi [m], Kr, Kt, lc / wc, Ac / Hb^2), H0/L0 = 0.0174, figures 10, 11, 6 and 7
flume = {
    62: (0.0690, 0.122, 0.537, 2.10, 0.221),
    77: (0.0687, 0.161, 0.653, 1.90, 0.202),
    92: (0.0690, 0.192, 0.796, 1.56, 0.116),
}
breakerDepthIndex = 0.85                    # [-] section 3.1.1
breakerHeightIndex = 0.5 * 0.0174**-0.2     # [-] equation 3

@functools.lru_cache(maxsize = None)
def reefCase(crestMillimeters: int) -> dict:

    '''Run one of the flume's water levels to the jet's impact and return its summary.'''

    reef = ReefOverturn(crestDepth = crestMillimeters / 1000.0, deepWaterHeight = 0.0755, frequency = 0.6)
    reef.run(progress = False)

    return reef.summary()

def errors(index: int, key: str) -> dict:

    '''Relative error of a summary value against the flume's, at each water level.'''

    return {hc: reefCase(hc)[key] / values[index] - 1.0 for hc, values in flume.items()}

@pytest.mark.slow
def testTheIncidentWaveMatchesTheFlume():

    result = errors(0, 'incidentHeight')

    assert all(abs(error) < 0.05 for error in result.values()), result

@pytest.mark.slow
@pytest.mark.xfail(strict = True, reason = 'The SGN train reflects 50 to 110 percent more than the flume.')
def testTheReefReflectsAsTheFlumeDid():

    result = errors(1, 'reflectionCoefficient')

    assert all(abs(error) < 0.20 for error in result.values()), result

@pytest.mark.slow
@pytest.mark.xfail(strict = True, reason = 'The SGN train passes up to 44 percent more over the reef than the flume.')
def testTheReefTransmitsAsTheFlumeDid():

    result = errors(2, 'transmissionCoefficient')

    assert all(abs(error) < 0.20 for error in result.values()), result

@pytest.mark.slow
@pytest.mark.xfail(strict = True, reason = 'Waves break 16 to 41 percent higher against the depth than in the flume.')
def testTheBreakerDepthIndexMatchesTheFlume():

    result = {hc: reefCase(hc)['breakerDepthIndex'] / breakerDepthIndex - 1.0 for hc in flume}

    assert all(abs(error) < 0.20 for error in result.values()), result

@pytest.mark.slow
@pytest.mark.xfail(strict = True, reason = 'Waves break up to 45 percent higher against H0 than in the flume.')
def testTheBreakerHeightIndexMatchesTheFlume():

    result = {hc: reefCase(hc)['breakerHeightIndex'] / breakerHeightIndex - 1.0 for hc in flume}

    assert all(abs(error) < 0.20 for error in result.values()), result

@pytest.mark.slow
@pytest.mark.xfail(strict = True, reason = 'Above hc/H0 = 1 the cavity is too long for its width.')
def testTheCavityShapeMatchesTheFlume():

    result = errors(3, 'cavityRatio')

    assert all(abs(error) < 0.15 for error in result.values()), result

@pytest.mark.slow
@pytest.mark.xfail(strict = True, reason = 'The cavity is twice the flume\'s at low submergence, a fifth at high.')
def testTheCavityAreaMatchesTheFlume():

    result = errors(4, 'cavityAreaOverBreakingHeightSquared')

    assert all(abs(error) < 0.20 for error in result.values()), result
