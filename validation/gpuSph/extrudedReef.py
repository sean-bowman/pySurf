# -- The Surf Ranch reef in 2D and extruded across a 3D tank: a consistency check of the 3D model -- #

'''

Gate 1's Surf Ranch reef (a 0.0693 face from x = 30 h0 to a shelf 0.371 h0 deep, an end wall at
60 h0), the exact solitary wave of H/h0 = 0.6 started at x = 15 h0 from bed-fitted columns, the
'deltaSph' scheme, run to the jet's landing. '2d' runs `BreakingWave`; '3d' runs `BreakingWave3D`
over the same bed extruded across a tank `width` h0 wide (1 by default), read on its middle section,
where the side walls do not reach (they disturb three particle rows beside them). With --split the
crest's water splits once in the zone that follows the crest from the toe: 1 h0 behind, 1.5 ahead,
from 0.15 h0 over still water up (in 3D each strip 0.25 h0 across reads its own crest), the
daughters merging --mergeBehind h0 behind the crest when that is given.

No measurement of a 3D overturn's lip exists, so each 3D run is compared with the 2D run of the same
zone: a cross-check between two forms of one model. Criteria, set before the runs (2026-11-14), half
the gates' tolerances, so the 3D form adds at most half the error they allow: breaking time and
touchdown time within 2.5 percent, Hb within 5 percent, impact x within 0.25 h0, H_I within 2.5
percent, A_O/H² within 7.5 percent, A_J/H² within 10 percent, theta_O within 2.5 degrees.

Recorded at 32 particles per depth (2026-11-14 and 15): unsplit, both forms stop on a void that
closes inside the crest, so they compare on breaking alone, which passes (time -0.25 percent, Hb
+1.3 percent; the 3D run took 4.7 GPU hours, 1.51 million fluid particles). Split, all eight
readings pass (breaking time -0.35 percent, touchdown time -0.46, Hb +1.2, impact x -0.14 h0, H_I
-0.5 percent, A_O/H² +1.8, A_J/H² -0.4, theta_O -0.6 degrees; 8.5 GPU hours). Merging 1.5 h0 behind
leaves all eight within 0.005 percent of the unmerged run's and the 3D run takes 6.8 GPU hours.

    python extrudedReef.py <2d|3d> <ppd> [--split] [--mergeBehind <h0>] [--width <h0>]
    python extrudedReef.py compare <ppd> [--split] [--mergeBehind <h0>]

Writes results/extrudedReef_<2d|3d>_<ppd>[_split][_merge].json; 'compare' reads the pair.

Sean Bowman [11/19/2026]

'''

import argparse
import json
import time

import common
from common import percent, verdict, writeResult

from pysurf.bathymetry import Bathymetry, HeightField
from pysurf.BreakingWave import BreakingWave
from pysurf.BreakingWave3D import BreakingWave3D

slope, shelf, toeX, endX = 0.0693, 0.371, 30.0, 60.0
# Reading, how it is compared, limit
limits = {'breakingTime': ('percent', 2.5), 'touchdownTime': ('percent', 2.5), 'breakingCrestHeight': ('percent', 5.0),
          'impactX': ('difference', 0.25), 'impactHeightOverDepth': ('percent', 2.5),
          'overturnAreaOverImpactHeightSquared': ('percent', 7.5), 'jetAreaOverImpactHeightSquared': ('percent', 10.0),
          'overturnAngle': ('difference', 2.5)}

def resultName(mode: str, ppd: int, split: bool, mergeBehind: float | None) -> str:

    '''The results file's name for one form of the run.'''

    return f'extrudedReef_{mode}_{ppd}' + ('_split' if split else '') + ('_merge' if mergeBehind is not None else '')

def run(mode: str, ppd: int, split: bool, mergeBehind: float | None, width: float) -> dict:

    '''Run one form of the case to the jet's landing.'''

    bed = Bathymetry.reefBreak(depth = 1.0, toeX = toeX, faceSlope = slope, reefDepth = shelf,
                               reefLength = endX - (toeX + (1.0 - shelf) / slope))
    zone = {'levels': 1, 'behind': 1.0, 'ahead': 1.5, 'above': 0.15, 'fromX': toeX, 'mergeBehind': mergeBehind}
    started = time.perf_counter()
    if mode == '2d':
        wave = BreakingWave(depth = 1.0, waveHeight = 0.6, beachSlope = slope, bathymetry = bed, startX = 15.0,
                            particlesPerDepth = ppd, scheme = 'deltaSph', startingWave = 'exact', backend = 'gpu',
                            refinement = zone if split else None)
        wave.run(progress = True, stopAt = 'touchdown')
        summary = wave.summary()
        readings = {key: summary.get(key) for key in limits}
        readings['breakingCrestHeight'] = summary.get('breakingHeightOverDepth')
    else:
        field = HeightField.fromFunction(lambda x, y: bed.elevation(x) + 0.0 * y, (0.0, endX, 0.0, width), 1.0 / 64.0,
                                         stillWaterLevel = bed.stillWaterLevel)
        wave = BreakingWave3D(field, 0.6, 15.0, depth = 1.0, particlesPerDepth = ppd, searchFromX = toeX,
                              refinement = {**zone, 'strip': 0.25} if split else None)
        wave.run(progress = True, stopAt = 'breaking')
        full = wave.summary()
        summary = {**{key: value for key, value in full.items() if key != 'sections'}, **full['sections'][wave.sections[0]]}
        readings = {key: summary.get(key) for key in limits}
    record = {'case': 'extrudedReef', 'mode': mode, 'particlesPerDepth': ppd, 'split': split, 'mergeBehind': mergeBehind,
              'width': width if mode == '3d' else None, 'fluidAtEnd': int(wave.particles.nFluid),
              'wall': time.perf_counter() - started, 'readings': readings, 'summary': summary}
    print(f'{mode} {ppd} ppd{" split" if split else ""}: ' + ', '.join(f'{key} {value}' for key, value in readings.items())
          + f'; {record["wall"]:.0f} s', flush = True)
    writeResult(resultName(mode, ppd, split, mergeBehind), record)
    return record

def compare(ppd: int, split: bool, mergeBehind: float | None) -> dict:

    '''The 3D form's eight readings against the 2D form's, at half the gates' tolerances.'''

    pair = {}
    for mode in ('2d', '3d'):
        path = common.resultsDir / f'{resultName(mode, ppd, split, mergeBehind)}.json'
        pair[mode] = json.loads(path.read_text(encoding = 'utf-8'))['readings']
    comparison = {}
    for key, (kind, limit) in limits.items():
        flat, extruded = pair['2d'].get(key), pair['3d'].get(key)
        if flat is None or extruded is None:
            print(f'  {key}: not read in both forms', flush = True)
            continue
        difference = percent(extruded, flat) if kind == 'percent' else extruded - flat
        comparison[key] = {'2d': flat, '3d': extruded, 'difference': difference, 'limit': limit,
                           'passes': abs(difference) <= limit}
        print(f'  {key}: 2D {flat:.4f}, 3D {extruded:.4f}, {difference:+.2f} {"percent" if kind == "percent" else ""} '
              f'(limit {limit}): {verdict(abs(difference) <= limit)}', flush = True)
    writeResult(resultName('compare', ppd, split, mergeBehind), comparison)
    return comparison

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'The Surf Ranch reef in 2D and extruded across a 3D tank.')
    parser.add_argument('mode', choices = ('2d', '3d', 'compare'))
    parser.add_argument('ppd', type = int, help = 'particles per depth')
    parser.add_argument('--split', action = 'store_true', help = "split the crest's water once")
    parser.add_argument('--mergeBehind', type = float, default = None, help = 'merge the daughters this far behind the crest [h0]')
    parser.add_argument('--width', type = float, default = 1.0, help = "the 3D tank's width [h0]")
    arguments = parser.parse_args()
    if arguments.mode == 'compare':
        compare(arguments.ppd, arguments.split, arguments.mergeBehind)
    else:
        run(arguments.mode, arguments.ppd, arguments.split, arguments.mergeBehind, arguments.width)
