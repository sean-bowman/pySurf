# -- The Surf Ranch reef turned 25 degrees, against DualSPHysics's run of the same tank: a cross-check -- #

'''

The 2D Surf Ranch profile with its toe along x = 30 + y tan 25 h0, the 0.0693 face up to a shelf
0.371 h0 deep and the shelf to an end wall at x = 52 h0, across a span of 4 h0 between free-slip
walls (the geometry of examples/openFoam/cases/breakingWave3D). The exact solitary wave of H/h0 = 0.6 starts
with its crest at x = 15 h0. `BreakingWave3D` reads nine sections, y = 0 to 4 h0 every 0.5, each two
particle rows wide. Times count from the crest's passage of the gauge at x = 20 h0, as
DualSPHysics's do. The run stops 0.1 s after the seven inner sections have each touched down, or at
an end bound 7.0 s after the gauge.

With --zone the crest's water splits once with the smoothing length kept, from where still water
first falls to 0.55 h0: 1 h0 behind each strip's crest to 1.5 ahead, from 0.4 h0 under still water
up, the daughters merging 1.5 h0 behind.

Reference: DualSPHysics's 32 ppd run (reference/dualSphysics/shoalReef32.json). No measurement exists for
this tank, so this is a cross-check between two unvalidated codes: nothing passes or fails. Per
section: breaking time, the crest's distance past its own shelf edge (x = 30 + y tan 25 + 9.08 h0),
Hb/h0, touchdown time, and at touchdown H/h0, A_O/H², A_J/H² and theta_O. A line through the seven
inner sections' breaking times gives the speed the break crosses the span, against DualSPHysics's
9.90 m/s. Set before the runs: a difference of the size and sign the two schemes show in 2D (there
DualSPHysics touched down 0.55 s after pySurf and broke lower) is the schemes'; a much larger one,
or one of the other sign, points first at the 3D setup.

Recorded at 32 particles per depth with the zone (2026-11-19; 32.1 GPU hours, 6.6 million particles
by the end): the inner sections break within 0.04 s of DualSPHysics's, 0.15 to 0.42 h0 farther past
the edge and 8 to 11 percent higher; they touch down 0.03 to 0.13 s later, the other sign from 2D;
the break crosses the span at 10.31 m/s, +4.1 percent. The face traces show the bed, walls and start
carrying both waves onto the face on one schedule, so the departure from 2D lies in how one code or
both break in this tank.

    python shoalReef.py <ppd> [--zone] [--end <s after the gauge>]

Writes results/shoalReef_<ppd>[_zone].json.

Sean Bowman [11/19/2026]

'''

import argparse
import math
import time

import numpy as np

import common
from common import readReference, writeResult

from pysurf import waveTheory
from pysurf.bathymetry import HeightField
from pysurf.BreakingWave3D import BreakingWave3D

height, slope, shelfDepth, toeX, endX, span, angle = 0.6, 0.0693, 0.371, 30.0, 52.0, 4.0, 25.0
crestX, gaugeX = 15.0, 20.0
sections = tuple(0.5 * k for k in range(9))
inner = tuple(y for y in sections if 0.0 < y < span)

def reefBed(x: np.ndarray, y: np.ndarray) -> np.ndarray:

    '''The toe runs along x = 30 + y tan 25, so the chevron's point stands at y = 0 and its bay at y = 4.'''

    return np.clip((x - toeX - y * math.tan(math.radians(angle))) * slope, 0.0, 1.0 - shelfDepth)

def passage(times: np.ndarray, crestXs: np.ndarray, target: float) -> float:

    '''The time the crest passes x = target, from a line through its first pass of three readings or more within 0.5 h0.'''

    indices = np.flatnonzero(np.isfinite(crestXs) & (np.abs(crestXs - target) < 0.5))
    run = next(run for run in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if run.size >= 3)
    gradient, intercept = np.polyfit(times[run], crestXs[run], 1)
    return (target - intercept) / gradient

def main(ppd: int, useZone: bool, afterGauge: float, densityDiffusion: str = 'hydrostatic') -> dict:

    '''Run the turned reef and set its sections beside DualSPHysics's.'''

    wallX = crestX - waveTheory.solitaryWaveHalfLength(height, 1.0, 0.01)
    field = HeightField.fromFunction(reefBed, (wallX, endX, 0.0, span), 1.0 / 64.0, stillWaterLevel = 1.0)
    fromX = field.shoalingStart(depth = 1.0, share = 0.45)
    refinement = ({'levels': 1, 'behind': 1.0, 'ahead': 1.5, 'above': -0.4, 'fromX': fromX, 'mergeBehind': 1.5}
                  if useZone else None)
    endTime = (gaugeX - crestX) / waveTheory.exactSolitaryWaveSpeed(height, 1.0) + afterGauge
    wave = BreakingWave3D(field, height, crestX, depth = 1.0, particlesPerDepth = ppd, sections = sections,
                          searchFromX = toeX, frameInterval = 0.1 * common.timeScale, endTime = endTime,
                          refinement = refinement, solverOptions = {'densityDiffusion': densityDiffusion})
    started = time.perf_counter()
    wave.build()
    print(f'shoal reef, {ppd} ppd{" with the crest zone" if useZone else ""}: {wave.particles.nFluid} fluid particles, '
          f'built in {time.perf_counter() - started:.0f} s; end bound {endTime:.3f} s', flush = True)

    # Run in pieces of 0.1 s. The section frames serve only figures, and at 32 particles per depth
    # they would hold about 8 GB of host memory, so each piece drops them; the trackers read the
    # water on their own clock and never read the frames
    finishedAt = None
    while wave.solver.time < endTime - 1e-9:
        wave.run(endTime = min(endTime, wave.solver.time + 0.1), progress = False)
        for y in wave.sectionFrames:
            wave.sectionFrames[y] = []
        t = wave.solver.time
        final = [y for y in inner if wave.trackers[y].breakingFinal]
        print(f'  t = {t:6.3f} s (t\' {t / common.timeScale:6.2f})  wall {wave.wallClockSeconds:9.1f} s  '
              f'fluid {wave.particles.nFluid}  final {len(final)} of {len(inner)}', flush = True)
        if finishedAt is None and len(final) == len(inner):
            finishedAt = t
        if finishedAt is not None and t >= finishedAt + 0.1 - 1e-9:
            break

    summary = wave.summary()
    dualSph = readReference('dualSphysics/shoalReef32.json')
    tanAngle = math.tan(math.radians(angle))
    passages = [passage(np.array(wave.trackers[y].times, float), np.array(wave.trackers[y].crestX, float), gaugeX)
                for y in sections]
    gauge = float(np.mean(passages))
    record = {'case': 'shoalReef', 'particlesPerDepth': ppd, 'zone': useZone, 'gaugePassage': gauge,
              'summary': summary, 'sections': {}}
    print(f'crest passes x = {gaugeX} h0 at {gauge:.4f} s; times below are after it, pySurf / DualSPHysics', flush = True)
    rows = []
    for y in sections:
        section, reference = summary['sections'][y], dualSph['sections'][f'{y:.1f}']
        if not section['broke']:
            print(f'  y {y:.1f}: no breaking point read', flush = True)
            continue
        edge = toeX + y * tanAngle + (1.0 - shelfDepth) / slope
        touchdown = section['touchdownTime']
        row = {'breakingTime': section['breakingTime'] - gauge, 'crestPastEdge': section['breakingCrestX'] - edge,
               'breakingHeight': section['breakingCrestHeight'],
               'touchdownTime': touchdown - gauge if touchdown is not None else None,
               'impactHeight': section.get('impactHeightOverDepth'),
               'overturnArea': section.get('overturnAreaOverImpactHeightSquared'),
               'jetArea': section.get('jetAreaOverImpactHeightSquared'), 'overturnAngle': section.get('overturnAngle'),
               'dualSphysics': dict(zip(('breakingTime', 'crestPastEdge', 'breakingHeight', 'touchdownTime', 'impactHeight',
                                         'overturnArea', 'jetArea', 'overturnAngle'), reference))}
        record['sections'][f'{y:.1f}'] = row
        landing = f'{row["touchdownTime"]:.3f} / {reference[3]:.3f}' if row['touchdownTime'] is not None else 'none read'
        print(f'  y {y:.1f}: breaks {row["breakingTime"]:.3f} / {reference[0]:.3f} s, past the edge '
              f'{row["crestPastEdge"]:.2f} / {reference[1]:.2f} h0, Hb/h0 {row["breakingHeight"]:.3f} / {reference[2]:.3f}, '
              f'lands {landing} s', flush = True)
        if y in inner:
            rows.append((y, row['breakingTime']))
    if len(rows) >= 3:
        ys, times = np.array([row[0] for row in rows]), np.array([row[1] for row in rows])
        speed = float(np.polyfit(times, ys, 1)[0])
        record['breakCrossingSpeed'] = speed
        print(f'the break crosses the span at {speed:.2f} m/s through {len(rows)} inner sections: '
              f'{100.0 * (speed / dualSph["breakCrossingSpeed_m_s"] - 1.0):+.1f} percent against DualSPHysics\'s '
              f'{dualSph["breakCrossingSpeed_m_s"]}', flush = True)
    print(f'{summary["fluidParticles"]} fluid particles at the end, {summary["wallClockSeconds"]:.0f} s', flush = True)
    record['densityDiffusion'] = densityDiffusion
    writeResult(f'shoalReef_{ppd}' + ('_zone' if useZone else '') + common.diffusionSuffix(densityDiffusion), record)
    return record

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'The Surf Ranch reef turned 25 degrees, against DualSPHysics.')
    parser.add_argument('ppd', type = int, help = 'particles per depth')
    parser.add_argument('--zone', action = 'store_true', help = "split the crest's water once")
    parser.add_argument('--end', type = float, default = 7.0, help = 'end bound [s after the gauge]')
    common.addDiffusionArgument(parser)
    arguments = parser.parse_args()
    main(arguments.ppd, arguments.zone, arguments.end, arguments.densityDiffusion)
