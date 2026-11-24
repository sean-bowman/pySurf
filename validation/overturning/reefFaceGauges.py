# -- The largest elevation along the steep reef face against Quiroga and Cheung's gauges, boundary elements -- #

'''

The waves of reefFaceBreaking.py on the hf = 0.1 m flat, read at the flume's gauges up the 1:12
face. Quiroga and Cheung's (2013) figure 7 gives the control tests' largest elevation over H0 at
WG2 to WG9 (validation/overturning/reference/quiroga2013Fig7.json); the model's, from frames every
0.01 s, is the highest surface over each gauge until the jet lands. The gauges stand at their
distance from the toe, the flume's toe at 54.8 m less the face's 28.32 m.

Reading, set before the runs (2026-11-23): for the four breaking cases (nominal H0/h0 = 0.2 to
0.5, run at the measured 0.2108, 0.3206, 0.4105 and 0.5102), the excess e = model less flume at
each gauge. The waves part early if e at WG3 exceeds 0.03 in three or four of them, and the start is
then the leading suspect; they part late if e stays within 0.03 through WG6 in three or four and
exceeds it by WG8, and the gap then lies near breaking; anything else is gradual. The nominal 0.1
case (run at 0.106, the paper's 6 percent over target), which shoals the whole face, is reported
beside them.

Recorded (2026-11-23): by the rule the waves part gradually (one case early, the largest; none
late). The model's wave is not the larger one up the face. Its largest elevation at WG2 to WG4
lies within -0.031 to +0.013 H0 of the flume's for nominal 0.1 to 0.4, and 0.030 to 0.040 above
it at 0.5; at WG5 to WG8 it lies 0.085 below to 0.005 above the flume's for 0.1 to 0.4, below at
all but one gauge, and 0.007 to 0.043 above at 0.5. Its crest at breaking lies within -4.1 to +2.0 percent of the largest
elevation the flume's gauges recorded before breaking, while the video's breaking heights sit
below those same gauge readings, 7 to 14 percent in figure 12 and 3 to 6 percent in figure 7. The
model breaks 0.1 to 0.7 m seaward of the video's breaking points. Each run takes 29 to 43 minutes
on one thread.

    python reefFaceGauges.py run [--workers 5]
    python reefFaceGauges.py report

Writes results/reefFaceGauges_H<height>.json.

Sean Bowman [11/23/2026]

'''

import os

# One BLAS thread a run: the runs go in parallel, one process each
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from reefFaceBreaking import makeWave, reefHeight, resultsDir

here = Path(__file__).resolve().parent
reference = json.loads((here / 'reference' / 'quiroga2013Fig7.json').read_text(encoding = 'utf-8'))
flat, h0 = '0.1', reefHeight + 0.1
flumeToeX = 54.8 - reefHeight * 12.0                # [m] from the wavemaker
heights = {'0.1': 0.106, '0.2': 0.2108, '0.3': 0.3206, '0.4': 0.4105, '0.5': 0.5102}
tolerance = 0.03

def surfaceAt(surface: np.ndarray, x: float) -> float:

    '''The highest elevation of a surface over x, wherever it crosses x [m].'''

    xs, zs = surface[:, 0], surface[:, 1]
    crossing = np.flatnonzero((xs[:-1] - x) * (xs[1:] - x) <= 0.0)
    if crossing.size == 0:
        return -np.inf
    i = crossing
    share = np.where(xs[i + 1] != xs[i], (x - xs[i]) / np.where(xs[i + 1] != xs[i], xs[i + 1] - xs[i], 1.0), 0.0)

    return float(np.max(zs[i] + share * (zs[i + 1] - zs[i])))

def runOne(nominal: str) -> str:

    started = time.perf_counter()
    height = heights[nominal]
    wave, edgeX = makeWave(flat, height)
    wave.frameInterval = 0.01
    wave.run(progress = False)
    toeX = wave.toeX
    still = wave.bathymetry.stillWaterLevel
    gauges = {}
    for name, flumeX in reference['gaugesX_m'].items():
        modelX = toeX + (flumeX - flumeToeX) / h0
        peak = max(surfaceAt(frame.surface, modelX) for frame in wave.frames)
        gauges[name] = {'x_h0': modelX, 'etaMaxOverH0': (peak - still) / height}
    point = wave.breakingPoint
    record = {'nominal': nominal, 'incidentHeightOverDepth': height, 'gauges': gauges, 'summary': wave.summary(),
              'breakingCrestX_m': (flumeToeX + (point.crestX - toeX) * h0) if point is not None else None,
              'frames': len(wave.frames), 'wallSeconds': time.perf_counter() - started}
    resultsDir.mkdir(exist_ok = True)
    path = resultsDir / f'reefFaceGauges_H{height:.4f}.json'
    path.write_text(json.dumps(record, indent = 1, default = float), encoding = 'utf-8', newline = '\n')

    return f'nominal {nominal}: {wave.stopReason}, {len(wave.frames)} frames, {record["wallSeconds"]:.0f} s'

def runAll(workers: int) -> None:

    with ProcessPoolExecutor(max_workers = workers) as pool:
        futures = {pool.submit(runOne, nominal): nominal for nominal in heights}
        for future in as_completed(futures):
            try:
                print(future.result(), flush = True)
            except Exception as error:      # one failed run is reported and the others go on
                print(f'nominal {futures[future]}: failed: {error!r}', flush = True)

def report() -> None:

    names = list(reference['gaugesX_m'])
    print('nominal  ' + '  '.join(f'{name:>13s}' for name in names) + '   (flume / model / excess)')
    early, late = 0, 0
    for nominal, height in heights.items():
        path = resultsDir / f'reefFaceGauges_H{height:.4f}.json'
        if not path.exists():
            print(f'{nominal}: no record')
            continue
        record = json.loads(path.read_text(encoding = 'utf-8'))
        flume = reference['panels'][nominal]['gauges']
        excess = {name: record['gauges'][name]['etaMaxOverH0'] - flume[name]['etaMaxOverH0'] for name in names}
        print(f'{nominal:7s}  ' + '  '.join(f'{flume[name]["etaMaxOverH0"]:.3f}/{record["gauges"][name]["etaMaxOverH0"]:.3f}'
                                           f'/{excess[name]:+.3f}'.rjust(13) for name in names))
        breaking = record['summary']
        if record['breakingCrestX_m'] is not None:
            print(f'         model breaks at {record["breakingCrestX_m"]:.2f} m, crest {breaking["breakingHeightOverDepth"] / height:.3f} H0;'
                  f' flume video {reference["panels"][nominal]["videoBreaking"]}')
        if nominal == '0.1':
            continue
        if excess['WG3'] > tolerance:
            early += 1
        elif all(abs(excess[name]) <= tolerance for name in ('WG2', 'WG3', 'WG4', 'WG5', 'WG6')) and (
                max(excess['WG7'], excess['WG8']) > tolerance):
            late += 1
    verdict = 'early' if early >= 3 else 'late' if late >= 3 else 'gradual'
    print(f'Cases parting early: {early}; late: {late}. Reading: the waves part {verdict}.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'The largest elevation along the steep reef face, boundary elements.')
    parser.add_argument('mode', choices = ('run', 'report'))
    parser.add_argument('--workers', type = int, default = 5)
    arguments = parser.parse_args()
    if arguments.mode == 'run':
        runAll(arguments.workers)
    else:
        report()
