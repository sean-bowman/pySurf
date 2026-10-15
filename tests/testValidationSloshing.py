
# -- Validation: First-Mode Sloshing Period -- #

'''

The SPH sloshing tank against the linear first-mode period of a rectangular tank.

Linear potential flow gives the natural frequencies of a rectangular tank of width W filled to depth
d as omega_n^2 = g k_n tanh(k_n d) with k_n = n pi / W (Dean and Dalrymple 1991, ch. 4; Ibrahim
2005, ch. 2). A free surface tilted across the tank projects mostly onto the first mode (81 percent
of the wall amplitude; the higher odd modes carry the rest in proportion 1 / n^2), so the water
sloshes at close to the first-mode period T1 = 2 pi / omega_1.

The tank is 0.3 m wide and filled to 0.1 m, so k1 d = 1.047. That sits next to k d = 1.058, where
the second-order frequency correction of a standing wave changes sign, so the amplitude used here
(15 mm at the walls, 15 percent of the depth) shifts the period by well under 1 percent. The
amplitude must span several particle spacings: a wave one spacing high lives inside the surface
layer of particles, and at 5 mm spacing a 5 mm wave sloshes 5 to 6 percent fast and loses 35 to 50
percent of its amplitude each period, whatever the viscosity.

The period is fitted to the horizontal center of mass of the water, which the first mode dominates,
as a decaying cosine about a mean that relaxes toward the level-surface value.

Run with `pytest -m slow`.

Sean Bowman [10/11/2026]

'''

import math

import numpy as np
import pytest
from scipy.optimize import curve_fit

from pysurf import waveTheory
from pysurf.sphSolver import WcsphSolver
from pysurf.WaterSim.scenarios.sloshingTank import SloshingTankConfig, createSloshingTank

def dampedSloshing(t, amplitude, decayRate, omega, phase, mean, drift, driftRate):

    '''A decaying cosine about a mean that relaxes exponentially.'''

    return amplitude * np.exp(-decayRate * t) * np.cos(omega * t + phase) + mean + drift * np.exp(-driftRate * t)

def measureSloshingPeriod(particleSpacing: float = 0.005, tiltDeg: float = 5.71, periods: float = 4.0) -> dict:

    '''

    Run a tilted-surface tank and fit its sloshing period.

    Returns:
    --------
    dict : fitted and linear first-mode period [s], their ratio, and the amplitude lost per period

    '''

    tank = SloshingTankConfig(tankWidth = 0.3, tankHeight = 0.2, fillRatio = 0.5, initialTiltDeg = tiltDeg,
                              particleSpacing = particleSpacing)
    config, particles, boundary = createSloshingTank(tank)
    config.maxTimeStep = 1e-3
    depth = tank.fillRatio * tank.tankHeight
    reference = 2.0 * math.pi / waveTheory.sloshingFrequency(tank.tankWidth, depth)

    solver = WcsphSolver(config, boundary = boundary)
    solver.initialize(particles)

    times, centers = [], []
    nextSample = 0.0
    while solver.time < periods * reference:
        solver.step()
        if solver.time >= nextSample:
            times.append(solver.time)
            centers.append(float(np.mean(particles.positions[particles.isFluid, 0])))
            nextSample += reference / 100.0

    times, centers = np.asarray(times), np.asarray(centers) - centers[0]
    guess = [np.min(centers), 1.0, 2.0 * math.pi / reference, 0.0, centers[-1], 0.0, 1.0]
    fit, _ = curve_fit(dampedSloshing, times, centers, p0 = guess, maxfev = 20000)
    period = 2.0 * math.pi / fit[2]

    return {'period': period, 'reference': reference, 'ratio': period / reference,
            'decayPerPeriod': 1.0 - math.exp(-fit[1] * period)}

@pytest.mark.slow
def testTheTankSloshesAtTheFirstModePeriod():

    '''The fitted period is within 3 percent of the linear first-mode period.'''

    result = measureSloshingPeriod()

    assert abs(result['ratio'] - 1.0) < 0.03, result
