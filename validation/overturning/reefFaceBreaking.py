# -- The breaking point on a steep reef face against Quiroga and Cheung's flume, boundary elements -- #

'''

Solitary waves breaking on the 1:12 face of the fringing reef Quiroga and Cheung (2013) built in
Oregon State University's Large Wave Flume: an offshore depth h0 = 2.36 m plus the flat depth hf, a
1:12 face rising 2.36 m to a flat under hf of water, and a wall closing the flat 28.9 m past the reef
edge. Their control tests (smooth concrete) give the breaking height and depth from video, with
breaking taken where the front turns near vertical, as Grilli et al. (1997) take it.

Reference: validation/overturning/reference/quiroga2013Fig11Fig12.json, their figures 11 and 12
digitized. H_b is the crest above still water at breaking; h_b the still water depth under the
breaking front (their figure 4). The model's readings are `breakingHeightOverDepth` and
`breakingFaceDepthOverDepth` from `OverturningWave`, the front first vertical; the depth under the
crest is reported beside them.

Cases: the flats hf = 0.1, 0.2 and 0.3 m, each control point's H0/h0, the exact potential flow
solitary wave started a half length from the seaward wall with the toe a half length plus WG1's
8.3 m ahead of it, lengths in h0, default spacings. The dry flat, hf = 0, is not run: it needs a
moving shoreline.

Criteria, set before the runs (2026-11-23): H_b/h0 within 10 percent of the flume's in every case;
h_b/h0 within 15 percent where the flume's wave breaks on the face, its h_b more than 0.01 h0 deeper
than the flat (hf 0.1 m from H0/h0 = 0.26, hf 0.2 m from 0.40, hf 0.3 m at 0.45); elsewhere h_b is
reported, not judged. A flat depth is validated if every case on it passes; a run that does not
reach breaking is reported, not counted.

Recorded (2026-11-23): fails. No flat depth validates; 2 of the 16 cases pass both criteria (hf
0.3 m at H0/h0 = 0.40 and 0.45). Every breaking height is high, +2.5 to +23.7 percent (mean +12.7),
11 of 16 beyond 10 percent; where h_b is judged the model breaks deeper, +1.4 to +28.8 percent, 5
of 9 beyond 15. From WG1 to breaking the video's heights grow by 0.96 to 1.14 times H0 (mostly 1.01
to 1.08), the model's by 1.13 to 1.20. No breaking angle closes both gaps: reading breaking earlier
than the face first vertical, as "near vertical" might, lowers the model's H_b toward the flume's
but moves its h_b further offshore, away from it. The flume's own gauges up the face do not support
a larger model wave (reefFaceGauges.py): on the 0.1 m flat the model's crest at breaking lies within
-4.1 to +2.0 percent of the largest elevation those gauges recorded before breaking, and the video's
breaking heights sit 7 to 14 percent below the same gauge readings. Every run lands its jet; each
takes 20 to 80 minutes on one thread.

    python reefFaceBreaking.py run [--workers 9]
    python reefFaceBreaking.py report

Writes results/reefFace_hf<flat>_H<height>.json.

Sean Bowman [11/23/2026]

'''

import os

# One BLAS thread a run: the runs go in parallel, one process each
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[1] / 'src'))

from pysurf import waveTheory
from pysurf.bathymetry import Bathymetry
from pysurf.OverturningWave import OverturningWave

resultsDir = here / 'results'
reference = json.loads((here / 'reference' / 'quiroga2013Fig11Fig12.json').read_text(encoding = 'utf-8'))
flatsRun = ('0.1', '0.2', '0.3')
reefHeight, faceSlope, gaugeDistance, flatLength = 2.36, 1.0 / 12.0, 8.3, 28.9      # [m], [-], [m], [m]
heightTolerance, depthTolerance, onFaceMargin = 0.10, 0.15, 0.01

def makeWave(flat: str, heightRatio: float) -> tuple[OverturningWave, float]:

    '''The case's wave over the flume's profile in units of h0, and the reef edge's x [h0].'''

    hf = float(flat)
    h0 = reefHeight + hf
    halfLength = waveTheory.serreSolitaryWaveHalfLength(heightRatio, 1.0, 0.01)
    startX = halfLength
    toeX = startX + halfLength + gaugeDistance / h0
    bed = Bathymetry.reefBreak(depth = 1.0, toeX = toeX, faceSlope = faceSlope, reefDepth = hf / h0,
                               reefLength = flatLength / h0)
    edgeX = toeX + (1.0 - hf / h0) / faceSlope
    wave = OverturningWave(depth = 1.0, waveHeight = heightRatio, bathymetry = bed, beachSlope = faceSlope,
                           startX = startX, startingWave = 'exact')

    return wave, edgeX

def runOne(flat: str, heightRatio: float) -> str:

    started = time.perf_counter()
    wave, edgeX = makeWave(flat, heightRatio)
    wave.run(progress = False)
    summary = wave.summary()
    point = wave.breakingPoint
    record = {'flat': flat, 'incidentHeightOverDepth': heightRatio, 'summary': summary,
              'crestPastEdge': (point.crestX - edgeX) if point is not None else None,
              'facePastEdge': (point.faceX - edgeX) if point is not None else None,
              'wallSeconds': time.perf_counter() - started}
    resultsDir.mkdir(exist_ok = True)
    path = resultsDir / f'reefFace_hf{flat}_H{heightRatio:.4f}.json'
    path.write_text(json.dumps(record, indent = 1, default = float), encoding = 'utf-8', newline = '\n')

    return f'hf {flat} H0/h0 {heightRatio:.4f}: {wave.stopReason}, {wave.solver.steps} steps, {record["wallSeconds"]:.0f} s'

def cases() -> list[tuple[str, float]]:

    return [(flat, row['incidentHeightOverDepth']) for flat in flatsRun for row in reference['series'][flat]['points']]

def runAll(workers: int) -> None:

    with ProcessPoolExecutor(max_workers = workers) as pool:
        futures = {pool.submit(runOne, flat, height): (flat, height) for flat, height in cases()}
        for future in as_completed(futures):
            flat, height = futures[future]
            try:
                print(future.result(), flush = True)
            except Exception as error:      # one failed run is reported and the others go on
                print(f'hf {flat} H0/h0 {height:.4f}: failed: {error!r}', flush = True)

def report() -> None:

    '''Each case against the flume, and each flat depth's verdict.'''

    print('hf   H0/h0   | flume Hb/h0  hb/h0 | model Hb/h0  error  | face hb/h0  error  judged | crest hb/h0 | past edge')
    for flat in flatsRun:
        series = reference['series'][flat]
        flatDepth = series['flatDepthOverDepth']
        passes, counted = True, 0
        for row in series['points']:
            height = row['incidentHeightOverDepth']
            path = resultsDir / f'reefFace_hf{flat}_H{height:.4f}.json'
            if not path.exists():
                print(f'{flat}  {height:.4f}  no record')
                passes = False
                continue
            record = json.loads(path.read_text(encoding = 'utf-8'))
            summary = record['summary']
            if not summary['broke']:
                print(f'{flat}  {height:.4f}  did not break ({summary["stopReason"]}): not counted')
                continue
            counted += 1
            heightError = summary['breakingHeightOverDepth'] / row['breakingHeightOverDepth'] - 1.0
            depthError = summary['breakingFaceDepthOverDepth'] / row['breakingDepthOverDepth'] - 1.0
            onFace = row['breakingDepthOverDepth'] > flatDepth + onFaceMargin
            casePasses = abs(heightError) <= heightTolerance and (not onFace or abs(depthError) <= depthTolerance)
            passes &= casePasses
            print(f'{flat}  {height:.4f}  |   {row["breakingHeightOverDepth"]:.4f}   {row["breakingDepthOverDepth"]:.4f} |'
                  f'   {summary["breakingHeightOverDepth"]:.4f}   {100.0 * heightError:+6.1f}% |'
                  f'   {summary["breakingFaceDepthOverDepth"]:.4f}  {100.0 * depthError:+6.1f}%  {"yes" if onFace else "no ":3s}  |'
                  f'   {summary["breakingDepthOverDepth"]:.4f}    | {record["facePastEdge"]:+.3f} h0'
                  f'  {"pass" if casePasses else "FAIL"}')
        print(f'hf {flat} m: {"validated" if passes and counted else "not validated"} ({counted} cases counted)')
        print()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'The breaking point on a steep reef face, boundary elements.')
    parser.add_argument('mode', choices = ('run', 'report'))
    parser.add_argument('--workers', type = int, default = 9)
    arguments = parser.parse_args()
    if arguments.mode == 'run':
        runAll(arguments.workers)
    else:
        report()
