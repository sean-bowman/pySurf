# -- Gate 1: the energy lost in active breaking on Mostert and Deike's beach -- #

'''

Mostert and Deike's (2020) plunging case: a solitary wave of H0/h0 = 0.3 on a 3 degree plane beach,
`BreakingWave` on the 'deltaSph' scheme from bed-fitted columns, from start to end. The same tank
from still water runs first, and its energy drift comes out of the wave's
(`BreakingWave.breakingDissipation(reference = still)`). Active breaking runs from the breaking point
to the kinetic energy falling to the potential, as Mostert and Deike define it.

Reference: their two-phase DNS loses 0.31 of the wave's energy at breaking over active breaking;
across slopes of 2 to 5 degrees their dissipation rate collapses onto the inertial fit
(epsilon / epsilon0)(db / d0) = 0.0281 (hb / d0)^3.152.

Criterion, set before the runs: the share lost within 30 percent of 0.31. The scaled rate against
the fit is reported without a rule.

With --dualSphysicsSettings the run takes the settings DualSPHysics runs this case with: c0 = 20
sqrt(g h0), negative pressure allowed, no viscosity limiter, the artificial viscosity applied
against the bed at rest, h = 1.5 spacings. A diagnostic of where the deficit lies; the same
criterion applies.

Recorded: with pySurf's settings (2026-11-12) the share lost is 0.20 and 0.17 at 32 and 64
particles per depth, 35 and 47 percent short: fails, further off at the finer spacing. With
DualSPHysics's settings (2026-11-19) it is 0.607 and 0.545 at 32 and 64, 96 and 76 percent over:
fails the other way, and the wave loses about 1 percent of its energy per h0 travelled before it
breaks. The model has no turbulence; its artificial viscosity and density diffusion stand in for
the loss.

    python mostertEnergy.py <ppd> [--dualSphysicsSettings]

Writes results/mostertEnergy_<ppd>[_dualSphysics].json.

Sean Bowman [11/19/2026]

'''

import argparse
import math
import time

import numpy as np

import common
from common import percent, verdict, writeResult

from pysurf import constants, waveTheory
from pysurf.BreakingWave import BreakingWave
from pysurf.errors import NumericalInstabilityError

mostertShareLost = 0.31
waveHeight = 0.3
dualSphysics = {'soundSpeed': 20.0, 'smoothingLengthRatio': 1.5,
                'solverOptions': {'negativePressure': True, 'viscosityLimiter': False, 'boundaryViscosityFactor': 1.0}}

def makeWave(ppd: int, startingWave: str, settings: dict | None) -> BreakingWave:

    '''The beach run or its still-water reference, with pySurf's settings or `settings`.'''

    wave = BreakingWave(depth = 1.0, waveHeight = waveHeight, beachSlope = math.tan(math.radians(3.0)),
                        particlesPerDepth = ppd, scheme = 'deltaSph', backend = 'gpu', startingWave = startingWave,
                        solverOptions = settings['solverOptions'] if settings else None,
                        smoothingLengthRatio = settings['smoothingLengthRatio'] if settings else None,
                        frameInterval = 0.1 * common.timeScale)
    if settings:
        # The scheme sets c0 as a multiple of the wave's speed; this sets it in sqrt(g h0)
        wave.soundSpeedFactor = (settings['soundSpeed'] * math.sqrt(constants.gravity)
                                 / waveTheory.solitaryWaveSpeed(waveHeight, 1.0, constants.gravity))
    return wave

def main(ppd: int, useDualSphysics: bool) -> dict:

    '''Run the case at `ppd` particles per depth and report it against the criterion.'''

    settings = dualSphysics if useDualSphysics else None
    started = time.perf_counter()
    still = makeWave(ppd, 'still', settings)
    still.run(progress = False)
    stillPotential = np.array([frame.potentialEnergy for frame in still.frames])
    print(f'{ppd} ppd still water: potential energy change {stillPotential[-1] - stillPotential[0]:+.1f} J/m over the run, '
          f'{time.perf_counter() - started:.0f} s', flush = True)
    wave = makeWave(ppd, 'exact', settings)
    failure = None
    try:
        wave.run(progress = True)
    except NumericalInstabilityError as error:
        # Keep the frames up to the failure and measure what they hold
        failure = repr(error)
        print(f'{ppd} ppd stopped: {failure}', flush = True)
    summary = wave.summary()
    summary.update(failure = failure, wall = time.perf_counter() - started)
    record = {'case': 'mostertEnergy', 'particlesPerDepth': ppd, 'settings': 'dualSphysics' if settings else 'pySurf',
              'summary': summary, 'reference': {'shareLost': mostertShareLost}}
    try:
        dissipation = wave.breakingDissipation(reference = still)
    except ValueError as error:
        dissipation = {'error': repr(error)}
    record['dissipation'] = dissipation
    if 'energyShareLost' in dissipation:
        shareError = percent(dissipation['energyShareLost'], mostertShareLost)
        rateError = percent(dissipation['scaledDissipation'], dissipation['scaledDissipationFit'])
        passes = abs(shareError) <= 30.0
        record['verdict'] = {'shareError': shareError, 'rateError': rateError, 'passes': passes}
        print(f'{ppd} ppd, {record["settings"]} settings: share lost {dissipation["energyShareLost"]:.4f} '
              f'({shareError:+.1f} percent against 0.31, limit 30): {verdict(passes)}; scaled rate {rateError:+.1f} percent '
              f'against the fit; Hb/h0 {dissipation["breakingHeightOverDepth"]:.3f}, energy at breaking '
              f'{dissipation["breakingEnergy"] / dissipation["startEnergy"]:.3f} of the start; {summary["wall"]:.0f} s',
              flush = True)
    else:
        record['verdict'] = {'passes': False}
        print(f'{ppd} ppd: active breaking not measured ({dissipation.get("error")}); fails', flush = True)
    writeResult(f'mostertEnergy_{ppd}' + ('_dualSphysics' if settings else ''), record)
    return record

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Gate 1: the energy lost in active breaking on Mostert and Deike's beach.")
    parser.add_argument('ppd', type = int, help = 'particles per depth')
    parser.add_argument('--dualSphysicsSettings', action = 'store_true', help = "run with DualSPHysics's settings")
    arguments = parser.parse_args()
    main(arguments.ppd, arguments.dualSphysicsSettings)
