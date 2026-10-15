
# -- pySurf: SGN to Boundary Element Hand-off Tests -- #

'''

Tests of pysurf.sgnHandoff: walls are placed where the discharge crosses zero, and a linear
progressive wave handed off from its discharge and surface has the surface potential of the exact
wave, (g a / omega) sin(k x).

Sean Bowman [10/14/2026]

'''

import math

import numpy as np
import pytest

from pysurf.bathymetry import Bathymetry
from pysurf.sgnHandoff import dischargeZeroCrossing, sgnSurfaceState

def testAWallStandsAtTheNearestZeroCrossingOfTheDischarge():

    x = np.linspace(0.0, 10.0, 1001)
    discharge = np.cos(x)

    assert dischargeZeroCrossing(x, discharge, 4.0) == pytest.approx(1.5 * math.pi, abs = 1e-4)
    with pytest.raises(ValueError):
        dischargeZeroCrossing(x, np.ones_like(x), 4.0)

def testALinearProgressiveWaveHandsOffWithItsSurfacePotential():

    # eta = a cos(k x) travelling right carries q = (omega a / k) cos(k x); the walls stand where q,
    # and with it u at every depth, is zero
    gravity, depth, length, amplitude = 9.81, 1.0, 4.0, 1e-5
    k = 2.0 * math.pi / length
    omega = math.sqrt(gravity * k * math.tanh(k * depth))
    x = np.linspace(0.0, 3.0 * length, 1201)
    surface = depth + amplitude * np.cos(k * x)
    discharge = omega * amplitude / k * np.cos(k * x)
    low = dischargeZeroCrossing(x, discharge, 0.3 * length)
    high = dischargeZeroCrossing(x, discharge, 2.2 * length)
    tank = Bathymetry([low, high], [0.0, 0.0], depth)
    nodes, potential = sgnSurfaceState(x, surface, discharge, tank, 0.05)
    exact = gravity * amplitude / omega * np.sin(k * nodes[:, 0])

    assert low == pytest.approx(0.25 * length, abs = 1e-3) and high == pytest.approx(2.25 * length, abs = 1e-3)
    assert np.max(np.abs((potential - potential.mean()) - (exact - exact.mean()))) < 2e-3 * gravity * amplitude / omega

def testAWallAvoidsABoreWhereTheDischargeAlsoCrossesZero():

    # Two zero crossings within reach: a gentle one at pi, and a bore's front near x = 6 where the
    # discharge drops by 2 over a tenth and the surface would rise fast against a wall
    x = np.linspace(0.0, 10.0, 2001)
    discharge = 0.2 * (x - np.pi) - (1.0 + np.tanh((x - 6.0) / 0.05))

    assert dischargeZeroCrossing(x, discharge, 5.0) == pytest.approx(6.0, abs = 0.05)
    assert dischargeZeroCrossing(x, discharge, 5.0, 3.5) == pytest.approx(np.pi, abs = 1e-3)
