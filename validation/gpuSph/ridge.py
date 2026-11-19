# -- Gate 2: the breaking point and the flow under it on Guyenne and Grilli's 3D ridge -- #

'''

`BreakingWave3D` on the sloping ridge of Guyenne and Grilli (2006): z_b/h0 = (x - 5.225) sech²(k y)
/ 15 for x > 5.225, flat offshore, a wall at x = 19 h0, the full tank |y| <= 4 h0 between free-slip
walls, the exact solitary wave of H0/h0 = 0.6 started from bed-fitted columns with its crest at
x = 5.785 h0, as theirs started. Times count from that start. The run stops once the centre line's
breaking point is final, or at 10 sqrt(h0/g).

With --zone the crest's water splits once with the smoothing length kept (`refinement`): from 1 h0
behind each strip's crest to 1.5 ahead, from 0.4 h0 under still water up, from the crest's arrival
at x = 12 h0, the daughters merging 1.5 h0 behind.

Reference: Guyenne and Grilli (2006), 3D fully nonlinear potential flow; their table 1 gives the
breaking point for k = 0.10 to 0.50, their figure 16 the velocity at k = 0.25.

Criteria (tests/testValidationRidgeOverturn.py's, set before the comparison):

    breaking point   on the centre line, from the two particle rows nearest y = 0: time and crest
                     height within 5 percent, crest position within 0.5 h0
    velocity         at k = 0.25, the horizontal velocity at z = -0.2 h0 at t = 7.973 sqrt(h0/g)
                     on their figure 16 grid: the largest speed within 10 percent of 0.651
                     sqrt(g h0), the largest spanwise velocity within 30 percent of 0.085 of the
                     largest streamwise one

Recorded at 32 particles per depth: without the zone (2026-11-13) the breaking point passes at four
of the five k (time +1.2 to +4.7 percent, crest position +0.07 to +0.41 h0, crest height -4.5 to
-0.5 percent) and fails at k = 0.25, where the crest stands 0.53 h0 past theirs; the velocity
passes (speed -0.3 percent, spanwise ratio +0.1). With the zone (2026-11-16) k = 0.25 passes all
five: time +2.9 percent, crest position +0.39 h0, crest height -0.2 percent, speed -0.6, ratio +1.0.
A ridge takes 5 to 7 GPU hours at 32 particles per depth without the zone and 27.9 with it.

    python ridge.py <k> <ppd> [--zone] [--end <sqrt(h0/g)>]

Writes results/ridge<k * 100>_<ppd>[_zone].json.

Sean Bowman [11/19/2026]

'''

import argparse
import math
import time

import numpy as np

import common
from common import percent, verdict, writeResult

from pysurf import constants
from pysurf.bathymetry import HeightField
from pysurf.BreakingWave3D import BreakingWave3D

# Guyenne and Grilli's table 1: breaking time [sqrt(h0/g)], crest position [h0], crest height [h0]
reference = {0.10: (7.661, 15.802, 0.685), 0.17: (7.757, 15.904, 0.693), 0.25: (7.890, 16.093, 0.706),
             0.34: (8.147, 16.470, 0.719), 0.50: (8.639, 17.100, 0.740)}
velocityTime, velocityDepth = 7.973, -0.2             # [sqrt(h0/g)], [h0] from still water
ridgeMaxSpeed, ridgeSpanwiseRatio = 0.651, 0.085
zone = {'levels': 1, 'behind': 1.0, 'ahead': 1.5, 'above': -0.4, 'fromX': 12.0, 'mergeBehind': 1.5}

def velocityGrid(field: HeightField) -> np.ndarray:

    '''Guyenne and Grilli's figure 16 grid at z = -0.2 h0, where the bed lies under it.'''

    gridX, gridY = np.meshgrid(np.arange(14.0, 19.0 + 1e-9, 0.35), np.arange(-1.90, 1.90 + 1e-9, 0.38))
    z = 1.0 + velocityDepth
    wet = field.elevation(gridX, gridY) < z
    return np.column_stack([gridX[wet], gridY[wet], np.full(int(wet.sum()), z)])

def main(k: float, ppd: int, useZone: bool, endTime: float) -> dict:

    '''Run the ridge at `ppd` particles per depth and report it against the criteria.'''

    if k not in reference:
        raise SystemExit(f'k = {k} is not one of {sorted(reference)}')
    field = HeightField.ridge(k)
    timeScale = common.timeScale
    recordTimes = (velocityTime * timeScale,) if k == 0.25 else ()
    wave = BreakingWave3D(field, 0.6, 5.785, depth = 1.0, particlesPerDepth = ppd, searchFromX = 5.225,
                          recordTimes = recordTimes, frameInterval = 0.1 * timeScale, endTime = endTime * timeScale,
                          refinement = zone if useZone else None)
    started = time.perf_counter()
    wave.build()
    print(f'ridge k = {k}, {ppd} ppd{" with the crest zone" if useZone else ""}: {wave.particles.nFluid} fluid particles, '
          f'built in {time.perf_counter() - started:.0f} s', flush = True)
    wave.run(stopAt = 'breaking')
    summary = wave.summary()
    refTime, refX, refHeight = reference[k]
    record = {'case': 'ridge', 'k': k, 'particlesPerDepth': ppd, 'zone': useZone, 'summary': summary,
              'reference': {'time': refTime, 'crestX': refX, 'crestHeight': refHeight}}

    point = wave.breakingPoint
    if point is not None:
        t, x, h = point.time / timeScale, point.crestX, point.crestHeight
        timeError, positionError, heightError = percent(t, refTime), x - refX, percent(h, refHeight)
        passes = abs(timeError) <= 5.0 and abs(positionError) <= 0.5 and abs(heightError) <= 5.0
        record['breaking'] = {'time': t, 'crestX': x, 'crestHeight': h, 'timeError': timeError,
                              'crestXError': positionError, 'heightError': heightError, 'passes': passes,
                              'final': bool(wave.tracker.breakingFinal)}
        print(f'breaking: t {t:.3f} sqrt(h0/g) ({timeError:+.1f} percent, limit 5), x {x:.3f} h0 ({positionError:+.3f}, '
              f'limit 0.5), crest {h:.3f} h0 ({heightError:+.1f} percent, limit 5): {verdict(passes)}', flush = True)
    else:
        record['breaking'] = {'passes': False}
        print('breaking: none read; fails', flush = True)

    if recordTimes and wave.snapshots:
        grid = velocityGrid(field)
        velocity = wave.velocity(grid, recordTimes[0]) / math.sqrt(constants.gravity)
        finite = np.all(np.isfinite(velocity), axis = 1)
        speed = float(np.hypot(velocity[finite, 0], velocity[finite, 1]).max())
        ratio = float(np.abs(velocity[finite, 1]).max() / np.abs(velocity[finite, 0]).max())
        speedError, ratioError = percent(speed, ridgeMaxSpeed), percent(ratio, ridgeSpanwiseRatio)
        passes = abs(speedError) <= 10.0 and abs(ratioError) <= 30.0
        record['velocity'] = {'points': int(len(grid)), 'read': int(finite.sum()), 'maxSpeed': speed,
                              'speedError': speedError, 'spanwiseRatio': ratio, 'ratioError': ratioError, 'passes': passes}
        print(f'velocity at t = {velocityTime} sqrt(h0/g): largest speed {speed:.3f} sqrt(g h0) ({speedError:+.1f} percent, '
              f'limit 10), spanwise ratio {ratio:.3f} ({ratioError:+.1f} percent, limit 30): {verdict(passes)}', flush = True)
    print(f'{summary["fluidParticles"]} fluid particles at the end, {summary["wallClockSeconds"]:.0f} s to '
          f'{summary["time"] / timeScale:.3f} sqrt(h0/g)', flush = True)
    writeResult(f'ridge{round(100 * k)}_{ppd}' + ('_zone' if useZone else ''), record)
    return record

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Gate 2: Guyenne and Grilli's 3D ridge.")
    parser.add_argument('k', type = float, help = 'the ridge parameter: 0.10, 0.17, 0.25, 0.34 or 0.50')
    parser.add_argument('ppd', type = int, help = 'particles per depth')
    parser.add_argument('--zone', action = 'store_true', help = "split the crest's water once")
    parser.add_argument('--end', type = float, default = 10.0, help = 'end time [sqrt(h0/g)]')
    arguments = parser.parse_args()
    main(arguments.k, arguments.ppd, arguments.zone, arguments.end)
