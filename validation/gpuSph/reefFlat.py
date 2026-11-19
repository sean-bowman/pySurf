# -- Validation after breaking: a solitary wave over the Oregon State fringing reef, 2D -- #

'''

The smooth-bed control tests (BF0) of Quiroga and Cheung (2013) in Oregon State University's Large
Wave Flume: offshore depth h0 = 2.66 m, a 1:12 reef face rising 2.36 m to a flat reef under
hf = 0.30 m of water (hf/h0 = 0.1128), a rigid wall closing the flat 83.7 m from the wavemaker. The
reef edge sits at 54.8 m, scaled from the thesis's figure 3; the thesis bounds it to 54.4 to 54.9 m.
Lengths here are in h0, and the gauges stand at their distance from the edge.

`BreakingWave` on the 'deltaSph' scheme (bed-fitted columns, free-slip bed, single phase), the exact
solitary wave started with its crest a half length from the seaward wall and the toe 7 h0 past it.
Every 0.02 sqrt(h0/g) each gauge reads the column volume within two spacings (depth = n s² / 4s, eta
= depth - hf), which spray does not reach, and the top of the water (the highest particle within a
spacing, plus half a spacing), each from its own still-water value. Each gauge's eta_max is the
incident bore's: the largest column reading from the front's arrival (column eta above 0.1 H0) for 3
sqrt(h0/g), or until the bore reflected from the end wall could return, less 0.2, whichever is
sooner.

Reference: the BF0 records' eta_max/H0, digitized from Quiroga (2012) figure 25 (reference/
quiroga2012Fig25Bf0.json, about ±0.02); for H0/h0 = 0.4, WG12 0.608, WG13 0.558, WG14 0.530 at 10.6,
17.9 and 25.2 m past the edge.

Criteria, set before the runs (2026-11-18): at 128 particles per depth and H0/h0 = 0.4, the column
eta_max/H0 at WG12, WG13 and WG14 each within 10 percent of the flume's; 64 particles per depth
beside it, a gauge that moves more than 5 percent between the two reported as unconverged. A scatter
diagnostic, set before its runs (2026-11-19): two runs whose changes leave the physics alone, the
time step 10 percent shorter (--cflNumber 0.225) and the crest started half a spacing shoreward
(--shift 0.5); a failure clear of the scatter if all three 128 runs lie more than 10 percent above
the flume, and a change with resolution resolved if the three 64 and three 128 values do not overlap.

Expected biases, both raising the bore: the lab's concrete bed has friction (about a tenth of the
bore height over the flat, by a Manning estimate) and pySurf's bed is free-slip; the lab's bore
carries air and pySurf's water does not.

Recorded (2026-11-18 and 19): fails. At 128 particles per depth WG12 reads 0.757, 0.786 and 0.767
over the three runs, 24.5 to 29.3 percent above the flume, clear of their scatter, and above every
64 run (0.605 to 0.732): the failure and its growth with resolution are both resolved. At WG13 and
WG14 the three 128 runs straddle the criterion (+0.6 to +28.6 and -53 to +22.5 percent). 128 runs
in about 2 GPU hours (346,831 particles), 64 in about 17 minutes.

    python reefFlat.py <ppd> [--ratio 0.4] [--cflNumber 0.225] [--shift 0.5]

Writes results/reefFlat_<ppd>_r<ratio * 100>[_cfl][_shift].json.

Sean Bowman [11/19/2026]

'''

import argparse
import math
import time

import numpy as np

import common
from common import percent, readReference, writeResult

from pysurf import waveTheory
from pysurf.bathymetry import Bathymetry
from pysurf.BreakingWave import BreakingWave

labDepth, labFlat, labEdge, labWall = 2.66, 0.30, 54.8, 83.7      # [m]

def readGauges(positions: np.ndarray, gauges: dict, bed: Bathymetry, spacing: float) -> tuple:

    '''Column and top-of-water eta at each gauge [h0] from one frame's fluid positions.'''

    s = spacing
    column, top = [], []
    for x in gauges.values():
        near = np.abs(positions[:, 0] - x) <= 2.0 * s
        bedHeight = 1.0 - float(bed.depth(np.array([x]))[0])
        column.append(near.sum() * s * s / (4.0 * s) + bedHeight - 1.0)
        close = np.abs(positions[:, 0] - x) <= s
        top.append(float(positions[close, 1].max()) + 0.5 * s - 1.0 if close.any() else math.nan)
    return np.array(column), np.array(top)

def evaluate(times: np.ndarray, columns: np.ndarray, tops: np.ndarray, gauges: dict, wallX: float, ratio: float) -> tuple:

    '''Arrivals, the WG12 to WG14 front speed [sqrt(g h0)] and each gauge's incident eta_max / H0.'''

    names = list(gauges)
    arrivals = {}
    for k, name in enumerate(names):
        above = np.flatnonzero(columns[:, k] > 0.1 * ratio)
        arrivals[name] = float(times[above[0]]) if above.size else math.nan
    speed = ((gauges['WG14'] - gauges['WG12']) / (arrivals['WG14'] - arrivals['WG12'])
             if np.isfinite(arrivals['WG14']) and np.isfinite(arrivals['WG12']) else math.nan)
    results = {}
    for k, name in enumerate(names):
        start = arrivals[name]
        if not np.isfinite(start):
            results[name] = None
            continue
        back = 2.0 * (wallX - gauges[name]) / speed if np.isfinite(speed) and speed > 0 else math.inf
        stop = start + min(3.0, back - 0.2)
        window = (times >= start) & (times <= stop)
        results[name] = {'x': gauges[name], 'arrival': start, 'windowEnd': stop,
                         'columnMax': float(columns[window, k].max()) / ratio,
                         'topMax': float(np.nanmax(tops[window, k])) / ratio}
    return arrivals, speed, results

def main(ppd: int, ratio: float, cflNumber: float | None, shift: float, endTime: float) -> dict:

    '''Run the reef flat and read the gauges against the flume.'''

    flume = readReference('quiroga2012Fig25Bf0.json')
    labGauges = flume['gaugesX_m']
    measured = flume['points'].get(f'{ratio:g}', {})
    reefDepth, slope = labFlat / labDepth, 1.0 / 12.0
    toeX = waveTheory.solitaryWaveHalfLength(ratio, 1.0, 0.01) + 7.0
    edgeX = toeX + (1.0 - reefDepth) / slope
    wallX = edgeX + (labWall - labEdge) / labDepth
    gauges = {name: edgeX + (x - labEdge) / labDepth for name, x in labGauges.items()}
    bed = Bathymetry.reefBreak(depth = 1.0, toeX = toeX, faceSlope = slope, reefDepth = reefDepth,
                               reefLength = wallX - edgeX)
    wave = BreakingWave(depth = 1.0, waveHeight = ratio, beachSlope = slope, bathymetry = bed, particlesPerDepth = ppd,
                        scheme = 'deltaSph', startingWave = 'exact', backend = 'gpu', frameInterval = 0.02 * common.timeScale,
                        solverOptions = {'cflNumber': cflNumber} if cflNumber is not None else None)
    s = wave.spacing
    if shift:
        wave.startX = wave.halfLength + shift * s
    started = time.perf_counter()
    wave.build()
    print(f'H0/h0 {ratio}, {ppd} ppd: {wave.particles.nFluid} fluid particles; toe {toeX:.3f}, edge {edgeX:.3f}, '
          f'wall {wallX:.3f} h0', flush = True)

    # Read every frame's gauges as the run goes, then drop the frame's arrays
    times, columns, tops = [], [], []
    done = 0
    while wave.solver.time < endTime - 1e-9:
        wave.run(endTime = min(endTime, wave.solver.time + 15 * wave.frameInterval), progress = False, stopAt = 'end')
        for frame in wave.frames[done:]:
            column, top = readGauges(frame.positions, gauges, bed, s)
            times.append(frame.time)
            columns.append(column)
            tops.append(top)
            frame.positions, frame.speeds = None, None
        done = len(wave.frames)
    times = np.array(times) / common.timeScale
    # Each column holds a whole number of spacings, so still water stands up to half a spacing off the
    # true level; every gauge reads from its own still-water value
    columns, tops = np.array(columns), np.array(tops)
    columns, tops = columns - columns[0], tops - tops[0]
    arrivals, speed, results = evaluate(times, columns, tops, gauges, wallX, ratio)
    print(f'front speed WG12 to WG14: {speed:.3f} sqrt(g h0); {time.perf_counter() - started:.0f} s', flush = True)
    for name, result in results.items():
        if result is None:
            print(f'  {name}: not reached', flush = True)
            continue
        line = f'  {name}: arrival t\' {result["arrival"]:.2f}, column eta_max/H0 {result["columnMax"]:.3f}, top {result["topMax"]:.3f}'
        if name in measured:
            error = percent(result['columnMax'], measured[name])
            result.update(flume = measured[name], error = error, withinTenPercent = abs(error) <= 10.0)
            line += f'; flume {measured[name]:.3f}, {error:+.1f} percent (limit 10 at 128 ppd)'
        print(line, flush = True)
    record = {'case': 'reefFlat', 'particlesPerDepth': ppd, 'ratio': ratio, 'cflNumber': cflNumber, 'shift': shift,
              'geometry': {'toeX': toeX, 'edgeX': edgeX, 'wallX': wallX, 'gauges': gauges, 'labEdge_m': labEdge},
              'summary': wave.summary(), 'arrivals': arrivals, 'frontSpeed': speed, 'results': results,
              'times': times, 'columns': columns, 'tops': tops}
    name = f'reefFlat_{ppd}_r{round(100 * ratio)}' + ('_cfl' if cflNumber is not None else '') + ('_shift' if shift else '')
    writeResult(name, record)
    return record

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'A solitary wave over the Oregon State fringing reef, 2D.')
    parser.add_argument('ppd', type = int, help = 'particles per depth')
    parser.add_argument('--ratio', type = float, default = 0.4, help = 'H0 / h0: 0.3, 0.4 or 0.45')
    parser.add_argument('--cflNumber', type = float, default = None, help = 'the time step\'s CFL number (the scheme\'s is 0.25)')
    parser.add_argument('--shift', type = float, default = 0.0, help = 'crest start shoreward [spacings]')
    parser.add_argument('--end', type = float, default = 11.0, help = 'end time [s]')
    arguments = parser.parse_args()
    main(arguments.ppd, arguments.ratio, arguments.cflNumber, arguments.shift, arguments.end)
