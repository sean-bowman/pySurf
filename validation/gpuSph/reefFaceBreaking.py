# -- The breaking point on a steep reef face against Quiroga and Cheung's flume, particles -- #

'''

Solitary waves breaking on the 1:12 face of Quiroga and Cheung's (2013) fringing reef in Oregon
State University's Large Wave Flume, run with `BreakingWave` on the 'deltaSph' scheme and the GPU
backend at 64 particles per depth: offshore depth h0 = 2.36 m plus the flat depth hf, the face
rising 2.36 m to the flat, a wall 28.9 m past the reef edge. The tank is the boundary element grid's
(validation/overturning/reefFaceBreaking.py): lengths in h0, the exact solitary wave started a half
length from the seaward wall with the toe a half length plus WG1's 8.3 m ahead of it. The dry flat,
hf = 0, is a bed rising to still water level at the edge, which the particles carry.

Reference: validation/overturning/reference/quiroga2013Fig11Fig12.json, the control tests' breaking
height H_b (crest above still water) and depth h_b (still water depth under the breaking front)
read from video, where the front turns near vertical. The model's readings are
`breakingHeightOverDepth` and `breakingFaceDepthOverDepth`, its breaking point the start of the
face's last vertical run before the jet lands; the depth under the crest is reported beside them.

Cases: hf 0.3 m at H0/h0 = 0.3205, 0.4004 and 0.4503, the reef-flat validation's tank; hf 0 m at
0.3102, 0.4101 and 0.5198.

Criteria, set before the runs (2026-11-23): H_b/h0 within 10 percent of the flume's in every case;
h_b/h0 within 15 percent where the flume's wave breaks on the face, its h_b more than 0.01 h0 deeper
than the flat (here hf 0.3 m at 0.45 and every hf 0 m case); elsewhere h_b is reported, not judged.
A flat depth is validated if every case on it passes; a run that does not reach breaking is
reported, not counted. The boundary element model fails the same reference high, by 2.5 to 23.7
percent in H_b, while agreeing with the flume's gauges up the face; the gauges for these flat depths
are in the thesis (Quiroga 2012, figures 13 and 16) and are not read here.

Recorded (2026-11-24, 64 ppd): fails; neither flat depth validates, 3 of 6 cases pass. hf 0.3 m:
H_b +20.6, +0.4 and +4.9 percent at 0.32, 0.40 and 0.45, h_b +11.2 percent at 0.45 (judged); the
smallest wave breaks 0.11 h0 past the edge and fails on H_b. hf 0 m: H_b +11.7, +9.0 and +15.0
percent at 0.31, 0.41 and 0.52, h_b +0.9, +7.4 and +21.4 percent; the first and last fail. Every
error has the boundary element model's sign, high and deeper. On the 0.3 m flat the two models
agree: the particles' H_b lies 1.9 to 2.5 percent under the boundary elements' and their h_b within
0.7 percent, a cross-check of two methods that both fail the video. Each run takes 4.5 to 6 minutes
on an RTX 2070 Super (86,000 to 98,000 fluid particles).

    python reefFaceBreaking.py [--ppd 64]

Writes results/reefFace_<ppd>_hf<flat>_H<height>.json.

Sean Bowman [11/23/2026]

'''

import argparse
import json
import time

import common
from common import writeResult

from pysurf import waveTheory
from pysurf.bathymetry import Bathymetry
from pysurf.BreakingWave import BreakingWave

reference = json.loads((common.here.parent / 'overturning' / 'reference' / 'quiroga2013Fig11Fig12.json')
                       .read_text(encoding = 'utf-8'))
cases = [('0.3', 0.3205), ('0.3', 0.4004), ('0.3', 0.4503), ('0.0', 0.3102), ('0.0', 0.4101), ('0.0', 0.5198)]
reefHeight, faceSlope, gaugeDistance, flatLength = 2.36, 1.0 / 12.0, 8.3, 28.9      # [m], [-], [m], [m]
heightTolerance, depthTolerance, onFaceMargin = 0.10, 0.15, 0.01

def bedFor(flat: str, heightRatio: float) -> tuple[Bathymetry, float]:

    '''The flume's profile in units of h0 for the exact wave of this height, and the reef edge's x.'''

    hf = float(flat)
    h0 = reefHeight + hf
    halfLength = waveTheory.solitaryWaveHalfLength(heightRatio, 1.0, 0.01)
    toeX = 2.0 * halfLength + gaugeDistance / h0
    edgeX = toeX + (1.0 - hf / h0) / faceSlope
    if hf > 0.0:
        bed = Bathymetry.reefBreak(depth = 1.0, toeX = toeX, faceSlope = faceSlope, reefDepth = hf / h0,
                                   reefLength = flatLength / h0)
    else:
        # A dry flat at still water level from the edge to the wall
        bed = Bathymetry([0.0, toeX, edgeX, edgeX + flatLength / h0], [0.0, 0.0, 1.0, 1.0], stillWaterLevel = 1.0)

    return bed, edgeX

def runCase(flat: str, heightRatio: float, ppd: int) -> dict:

    started = time.perf_counter()
    bed, edgeX = bedFor(flat, heightRatio)
    wave = BreakingWave(depth = 1.0, waveHeight = heightRatio, beachSlope = faceSlope, bathymetry = bed,
                        particlesPerDepth = ppd, scheme = 'deltaSph', startingWave = 'exact', backend = 'gpu',
                        frameInterval = 0.05 * max(1.0, ppd / 32.0))
    wave.build()
    print(f'hf {flat} m, H0/h0 {heightRatio}: {wave.particles.nFluid} fluid particles, toe {wave.toeX:.3f}, '
          f'edge {edgeX:.3f} h0', flush = True)
    # The breaking point is final at the jet's landing, where the run stops
    wave.run(progress = False, stopAt = 'breaking')
    summary = wave.summary()
    series = reference['series'][flat]
    row = next(point for point in series['points'] if abs(point['incidentHeightOverDepth'] - heightRatio) < 1e-4)
    record = {'case': 'reefFaceBreaking', 'particlesPerDepth': ppd, 'flat': flat, 'incidentHeightOverDepth': heightRatio,
              'flume': row, 'summary': summary, 'wallSeconds': time.perf_counter() - started}
    if summary.get('broke'):
        heightError = summary['breakingHeightOverDepth'] / row['breakingHeightOverDepth'] - 1.0
        depthError = summary['breakingFaceDepthOverDepth'] / row['breakingDepthOverDepth'] - 1.0
        onFace = row['breakingDepthOverDepth'] > series['flatDepthOverDepth'] + onFaceMargin
        passes = abs(heightError) <= heightTolerance and (not onFace or abs(depthError) <= depthTolerance)
        crestPastEdge = summary['breakingDistanceFromToe'] - (edgeX - wave.toeX)
        record.update(heightError = heightError, depthError = depthError, depthJudged = onFace, passes = passes,
                      crestPastEdge = crestPastEdge)
        print(f'  Hb/h0 {summary["breakingHeightOverDepth"]:.4f} against {row["breakingHeightOverDepth"]:.4f} '
              f'({100.0 * heightError:+.1f} percent); hb/h0 under the front {summary["breakingFaceDepthOverDepth"]:.4f} '
              f'against {row["breakingDepthOverDepth"]:.4f} ({100.0 * depthError:+.1f} percent, '
              f'{"judged" if onFace else "not judged"}), under the crest {summary["breakingDepthOverDepth"]:.4f}; '
              f'crest {crestPastEdge:+.3f} h0 past the edge: {"passes" if passes else "FAILS"}; '
              f'{record["wallSeconds"]:.0f} s', flush = True)
    else:
        print(f'  no breaking point: not counted; {record["wallSeconds"]:.0f} s', flush = True)
    writeResult(f'reefFace_{ppd}_hf{flat}_H{heightRatio:.4f}', record)

    return record

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'The breaking point on a steep reef face, particles.')
    parser.add_argument('--ppd', type = int, default = 64, help = 'particles per depth')
    arguments = parser.parse_args()
    records = [runCase(flat, height, arguments.ppd) for flat, height in cases]
    for flat in ('0.3', '0.0'):
        mine = [record for record in records if record['flat'] == flat]
        counted = [record for record in mine if record['summary'].get('broke')]
        validated = bool(counted) and all(record['passes'] for record in counted)
        print(f'hf {flat} m: {"validated" if validated else "not validated"} ({len(counted)} of {len(mine)} cases counted)',
              flush = True)
