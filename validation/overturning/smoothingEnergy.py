# -- The energy the five-point smoothing removes in the overturn, and the jet area it may move -- #

'''

The boundary element model (`OverturningWave`) smooths its free surface with Longuet-Higgins and
Cokelet's (1976) five-point formula every ten steps, and every three once the front face overhangs.
Longuet-Higgins and Cokelet smoothed every 5 to 20 steps and found each 5-point pass removed 0.06
percent of a steep wave's energy. The jet area at impact is 14 to 19 percent above Pick and
Feddersen's (2026) fit in three of five plane-slope cases, and it grows fastest in the last instants
before impact, when the smoothing runs most often.

Cases: the five plane-slope cases of tests/testValidationOverturning.py, as the test runs them (the
default SGN start, a wall at 0.05 h0 of water): 1:30, 1:20 and 1:15 at H0/h0 = 0.3, 1:20 at 0.2 and
1:20 at 0.6; and the Surf Ranch soliton of Feddersen et al. (2024). Each runs with the overhang
interval (`steepSmoothingInterval`) at 3, the default, 6 and 10, the shoaling interval staying at 10,
and records every smoothing pass's energy and volume before and after it (`trackSmoothing`).

Reading, set before the runs (2026-11-23):

- Jet area. The failing cases are 1:20 and 1:15 at 0.3 and 1:20 at 0.2. Each one's gap g is its
  A_J/H_I^2 at interval 3 less the fit. A case moves if A_J/H_I^2 at interval 10 lies g/3 or more
  closer to the fit than at interval 3. The smoothing is part of the failure if two or three cases
  move, not part of it if none does, and unresolved if one does. A move of g/3 or more away from the
  fit is reported as a sensitivity that does not explain the failure.
- A run that stops before the jet lands is reported and not counted; the reading then takes the
  longest interval that lands, if that is 6; otherwise the case is unreadable.
- A_O/H_I^2, W/L, theta_O and Hb/hb at every interval against the fit's three-RMSE band; a pass at
  interval 3 that fails at 6 or 10 is reported. The interval-3 runs must reproduce the recorded
  results (the jet area 14 to 19 percent over the fit in the failing cases), or the comparison stops.
- Energy, from the first pass with the face overhanging to the landing: the share the passes
  remove, the share everything else removes (time stepping, regridding, trimmed loops), and the
  mean share of one pass, against Longuet-Higgins and Cokelet's 0.06 percent. Reported only.

Recorded (2026-11-23): all 18 runs land, none trims a loop, and interval 3 reproduces the recorded
results. The jet area falls as the interval grows in all five plane cases, 3 to 9 percent from 3 to
10, most of it by 6:

    case            A_J/H_I^2 at 3, 6, 10       fit      over the fit at 3 / 10
    1:30, 0.3       0.0821  0.0776  0.0751     0.0688   +19.4 / +9.2 percent
    1:20, 0.3       0.1732  0.1663  0.1648     0.1496   +15.8 / +10.2
    1:15, 0.3       0.3200  0.3128  0.3091     0.2679   +19.4 / +15.4
    1:20, 0.2       0.2092  0.2026  0.2001     0.1834   +14.1 / +9.2
    1:20, 0.6       0.1079  0.1035  0.1016     0.1063   +1.5 / -4.4

By the rule the smoothing is part of the failure, at the line: 1:20 at 0.3 moves 0.0084 against
g/3 = 0.0079, 1:20 at 0.2 moves 0.0091 against 0.0086, and 1:15 at 0.3 moves 0.0110 against 0.0174.
It is a small part: at interval 10 the three failing cases still stand 9 to 15 percent over the fit,
outside three RMSE. A_O, W/L, theta_O and Hb/hb pass at every interval. On the Surf Ranch soliton
A_J falls from 0.195 to 0.187, away from the DNS's 0.22 and inside its 20 percent; the impact
moves by under 0.05 h0 and theta_O by under 0.3 degrees. The smoothing removes -0.05 to +0.07
percent of the wave's energy over the overturn and under 0.0013 percent a pass; everything else
adds 0.11 to 0.35 percent over the same span, more the less often the surface is smoothed. Each
run takes 17 to 40 minutes on one thread.

    python smoothingEnergy.py run [--workers 9]     every case at every interval, in parallel
    python smoothingEnergy.py report                the reading, from the records

Writes results/smoothing_<case>_i<interval>.json.

Sean Bowman [11/23/2026]

'''

import os

# One BLAS thread a run: the runs go in parallel, one process each
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[1] / 'src'))

from pysurf.bathymetry import Bathymetry
from pysurf.OverturningWave import OverturningWave

resultsDir = here / 'results'
intervals = (3, 6, 10)
planeCases = {'slope30_h0.3': (30, 0.3), 'slope20_h0.3': (20, 0.3), 'slope15_h0.3': (15, 0.3),
              'slope20_h0.2': (20, 0.2), 'slope20_h0.6': (20, 0.6)}
failingCases = ('slope20_h0.3', 'slope15_h0.3', 'slope20_h0.2')
caseNames = tuple(planeCases) + ('surfRanch',)
measures = ('overturnAreaOverImpactHeightSquared', 'jetAreaOverImpactHeightSquared', 'overturnAspectRatio',
            'overturnAngle', 'breakingIndex')

def pickFeddersenScaling(slope: float, heightRatio: float) -> dict:

    '''Pick and Feddersen's (2026) fits, eqs. 3.6 to 3.10, as (value, RMSE), as tests/testValidationOverturning.py has them.'''

    psi = slope / heightRatio**0.25

    return {
        'overturnAreaOverImpactHeightSquared': (5.319 * psi - 0.043, 0.013),
        'jetAreaOverImpactHeightSquared': (37.072 * psi**2 - 0.587 * psi + 0.020, 0.005),
        'overturnAspectRatio': (1.661 * psi + 0.298, 0.010),
        'overturnAngle': (-5746.4 * psi**2 + 225.2 * psi + 48.4, 1.2),
        'breakingIndex': (0.871 * math.exp(11.874 * psi), 0.078),
    }

def makeWave(case: str, interval: int) -> OverturningWave:

    '''The case's wave with the overhang smoothing interval given, tracking its smoothing.'''

    if case == 'surfRanch':
        slope, shallowDepth = 0.0693, 0.371
        reefLength = 60.0 - (30.0 + (1.0 - shallowDepth) / slope)
        bed = Bathymetry.reefBreak(depth = 1.0, toeX = 30.0, faceSlope = slope, reefDepth = shallowDepth,
                                   reefLength = reefLength)
        return OverturningWave(depth = 1.0, waveHeight = 0.6, bathymetry = bed, beachSlope = slope, startX = 15.0,
                               steepSmoothingInterval = interval, trackSmoothing = True)

    inverseSlope, heightRatio = planeCases[case]
    return OverturningWave(depth = 1.0, waveHeight = heightRatio, beachSlope = 1.0 / inverseSlope,
                           wallDepthRatio = 0.05, steepSmoothingInterval = interval, trackSmoothing = True)

def runOne(case: str, interval: int) -> str:

    '''Run one case at one interval and write its record.'''

    started = time.perf_counter()
    wave = makeWave(case, interval)
    wave.run(progress = False)
    passes = [{'time': item.time, 'step': item.step, 'overhanging': item.overhanging,
               'energyBefore': item.energyBefore, 'energyRemoved': item.energyRemoved,
               'volumeBefore': item.volumeBefore, 'volumeAfter': item.volumeAfter}
              for item in wave.solver.smoothingPasses]
    record = {'case': case, 'steepSmoothingInterval': interval, 'summary': wave.summary(),
              'budget': wave.smoothingBudget(), 'passes': passes, 'steps': wave.solver.steps,
              'wallSeconds': time.perf_counter() - started}
    resultsDir.mkdir(exist_ok = True)
    path = resultsDir / f'smoothing_{case}_i{interval}.json'
    path.write_text(json.dumps(record, indent = 1, default = float), encoding = 'utf-8', newline = '\n')

    return f'{case} at interval {interval}: {wave.stopReason}, {wave.solver.steps} steps, {record["wallSeconds"]:.0f} s'

def runAll(workers: int) -> None:

    '''Every case at every interval, in parallel.'''

    jobs = [(case, interval) for case in caseNames for interval in intervals]
    with ProcessPoolExecutor(max_workers = workers) as pool:
        futures = {pool.submit(runOne, case, interval): (case, interval) for case, interval in jobs}
        for future in as_completed(futures):
            case, interval = futures[future]
            try:
                print(future.result(), flush = True)
            except Exception as error:      # one failed run is reported and the others go on
                print(f'{case} at interval {interval}: failed: {error!r}', flush = True)

def readRecord(case: str, interval: int) -> dict | None:

    path = resultsDir / f'smoothing_{case}_i{interval}.json'

    return json.loads(path.read_text(encoding = 'utf-8')) if path.exists() else None

def report() -> None:

    '''Print each case's measures and budget at each interval, then the reading.'''

    records = {(case, interval): readRecord(case, interval) for case in caseNames for interval in intervals}

    print('case            interval  landed  loops   A_O     A_J     W/L     theta   Hb/hb   | smooth  other   mean/pass  passes')
    for case in caseNames:
        fit = pickFeddersenScaling(1.0 / planeCases[case][0], planeCases[case][1]) if case in planeCases else None
        for interval in intervals:
            record = records[(case, interval)]
            if record is None:
                print(f'{case:15s} {interval:5d}     no record')
                continue
            summary, budget = record['summary'], record['budget']
            values = [summary.get(key, float('nan')) for key in measures]
            flags = ''
            if fit is not None and summary['touchedDown']:
                flags = ''.join('.' if abs(summary[key] - fit[key][0]) < 3.0 * fit[key][1] else 'x' for key in measures)
            def share(key: str) -> str:
                return f'{100.0 * budget[key]:7.3f}%' if budget[key] is not None else '    -   '
            print(f'{case:15s} {interval:5d}     {str(summary["touchedDown"]):6s} {summary["trimmedLoops"]:4d}   '
                  + '  '.join(f'{value:6.3f}' for value in values) + f'  {flags:5s} | {share("smoothingShareOverOverturn")} '
                  f'{share("otherShareOverOverturn")} {share("meanShareEachPass")}  {budget["passesDuringOverturn"]:4d}')
        if fit is not None:
            print(f'{"  fit":15s} {"":5s}     {"":6s} {"":4s}   ' + '  '.join(f'{fit[key][0]:6.3f}' for key in measures))

    # The reading
    print()
    key = 'jetAreaOverImpactHeightSquared'
    reproduced, moving, away, readable = True, [], [], []
    for case in failingCases:
        fit = pickFeddersenScaling(1.0 / planeCases[case][0], planeCases[case][1])[key][0]
        base = records[(case, 3)]
        if base is None or not base['summary']['touchedDown']:
            print(f'{case}: interval 3 did not land; the comparison stops')
            reproduced = False
            continue
        excess = base['summary'][key] / fit - 1.0
        if not 0.135 <= excess <= 0.195:
            reproduced = False
        print(f'{case}: interval 3 A_J {base["summary"][key]:.4f}, fit {fit:.4f}, {100.0 * excess:+.1f} percent')
        longest = next((interval for interval in (10, 6) if records[(case, interval)] is not None
                        and records[(case, interval)]['summary']['touchedDown']), None)
        if longest is None:
            print(f'  unreadable: neither 6 nor 10 landed')
            continue
        readable.append(case)
        gap = base['summary'][key] - fit
        value = records[(case, longest)]['summary'][key]
        toward = (base['summary'][key] - value) * math.copysign(1.0, gap)
        print(f'  interval {longest}: A_J {value:.4f}; moved {toward:+.4f} toward the fit against g/3 = {abs(gap) / 3.0:.4f}')
        if toward >= abs(gap) / 3.0:
            moving.append(case)
        elif -toward >= abs(gap) / 3.0:
            away.append(case)
    if not reproduced:
        print('Interval 3 does not reproduce the recorded jet areas: the comparison stops.')
        return
    verdict = ('part of the failure' if len(moving) >= 2 else 'not part of the failure' if not moving and len(readable) == 3
               else 'unresolved')
    print(f'Cases that move toward the fit: {moving or "none"}; away from it: {away or "none"}; readable: {len(readable)} of 3')
    print(f'Reading: the smoothing is {verdict}.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'The energy the smoothing removes in the overturn.')
    parser.add_argument('mode', choices = ('run', 'report'))
    parser.add_argument('--workers', type = int, default = 9)
    arguments = parser.parse_args()
    if arguments.mode == 'run':
        runAll(arguments.workers)
    else:
        report()
