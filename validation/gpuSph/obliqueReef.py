# -- A wave crossing a uniform oblique reef: the crest's speed across the span against the closed form -- #

'''

The Surf Ranch reef profile laid with straight contours (a 0.0693 face from the toe to a shelf
0.371 h0 deep), met at 30 degrees by the exact solitary wave of H/h0 = 0.6. `BreakingWave3D` turns
the bed into the wave's frame, places the crest and the offshore wall by `offshoreReach`, and reads
a section every h0 across a tank between free-slip walls.

Closed form: over an infinite reef the flow is a pattern travelling along the contours, so in the
wave's frame the crest crosses y at c0 / tan 30, with c0 the offshore wave's speed (1.250 sqrt(g h0),
giving 2.165 sqrt(g h0)), at every distance past the toe. For each section and each distance d past
its toe along x (-2, 0, 4, 8 and 11 h0), the crest's passage of x = y tan 30 + d comes from a line
through its readings within 0.5 h0; a line through those times against y gives the crossing speed.
The same line through the breaking times follows. No numeric limit was set before the runs; the
check is read against the closed form.

Recorded at 16 particles per depth (2026-11-17), the tank 16 h0 across (1.74 million particles,
3.8 GPU hours): through 14 sections the crest crosses within 1.7 percent of c0 / tan 30 at every
distance (8 h0 across: within 3.5 percent). The section at y = -3 was left out of the fits after the
run: its face turned short of vertical and its recorded breaking point lies on a surface 0.4 h0
behind the crest (--skip -3 repeats that). The break crosses 9 to 11 percent slower than the closed
form (8 h0 across: 15 to 21 percent): the free-slip walls mirror the bed into a chevron reef, and the
breaking point creeps shoreward between them, a limit of a walled tank; the closed form covers the
crest only.

    python obliqueReef.py <ppd> [--span 16] [--zone] [--skip <y> ...]

Writes results/obliqueReef_<ppd>_span<span>[_zone].json.

Sean Bowman [11/19/2026]

'''

import argparse
import math
import time

import numpy as np

import common
from common import writeResult

from pysurf import waveTheory
from pysurf.bathymetry import HeightField
from pysurf.BreakingWave3D import BreakingWave3D, offshoreReach

angle, height, slope, shelfDepth = 30.0, 0.6, 0.0693, 0.371
distances = (-2.0, 0.0, 4.0, 8.0, 11.0)

def reefBed(x: np.ndarray, y: np.ndarray) -> np.ndarray:

    '''The reef in its own frame: contours along y, the toe at x = 0.'''

    return np.clip(x * slope, 0.0, 1.0 - shelfDepth) + 0.0 * y

def passage(times: np.ndarray, crestX: np.ndarray, target: float) -> float:

    '''The time the crest passes x = target, from a line through its first pass of three readings or more within 0.5 h0.'''

    indices = np.flatnonzero(np.isfinite(crestX) & (np.abs(crestX - target) < 0.5))
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    run = next((run for run in runs if run.size >= 3), None)
    if run is None:
        return math.nan
    gradient, intercept = np.polyfit(times[run], crestX[run], 1)
    return (target - intercept) / gradient

def crossing(ys: np.ndarray, times: np.ndarray, expected: float) -> dict:

    '''The speed a line through (time, y) gives, against the closed form; none from fewer than three sections.'''

    keep = np.isfinite(times)
    if keep.sum() < 3:
        return {'speed': math.nan, 'error': math.nan, 'sections': int(keep.sum()), 'rms': math.nan}
    fit = np.polyfit(times[keep], ys[keep], 1)
    residual = ys[keep] - np.polyval(fit, times[keep])
    return {'speed': float(fit[0]), 'error': 100.0 * (fit[0] / expected - 1.0), 'sections': int(keep.sum()),
            'rms': float(np.sqrt(np.mean(residual**2)))}

def main(ppd: int, span: float, useZone: bool, skipped: list) -> dict:

    '''Run the oblique reef and read the crest's and the break's speed across the span.'''

    halfSpan = 0.5 * span
    endX = halfSpan * math.tan(math.radians(angle)) + 18.7
    sections = tuple(float(y) for y in range(-int(halfSpan) + 1, int(halfSpan)))
    source = HeightField.fromFunction(reefBed, (-30.0, 40.0, -30.0, 30.0), 1.0 / 32.0, stillWaterLevel = 1.0)
    turned = source.rotated(angle, (-25.0, endX, -halfSpan, halfSpan), pivot = (0.0, 0.0))
    wallX, crestX = offshoreReach(turned, height, depth = 1.0)
    region = (wallX, endX, -halfSpan, halfSpan)
    # The zone acts where still water first falls to 0.55 h0
    fromX = turned.cropped(region).shoalingStart(depth = 1.0, share = 0.45)
    refinement = ({'levels': 1, 'behind': 1.0, 'ahead': 1.5, 'above': -0.4, 'fromX': fromX, 'mergeBehind': 1.5}
                  if useZone else None)
    # Run until the last section's jet should have landed: the 2D impact lies about 12.6 h0 past the toe
    # along the reef's normal; the crest averages about 1.15 sqrt(g h0)
    lastImpact = sections[-1] * math.tan(math.radians(angle)) + 12.6 / math.cos(math.radians(angle))
    endTime = math.ceil((lastImpact - crestX) / 1.15 + 1.0)
    wave = BreakingWave3D(source, height, crestX, depth = 1.0, particlesPerDepth = ppd, sections = sections,
                          frameInterval = 0.1 * common.timeScale, endTime = endTime * common.timeScale,
                          refinement = refinement, direction = angle, region = region, pivot = (0.0, 0.0))
    started = time.perf_counter()
    wave.build()
    print(f'oblique reef, {ppd} ppd, {span:.0f} h0 across: {wave.particles.nFluid} fluid particles, built in '
          f'{time.perf_counter() - started:.0f} s; crest from x = {crestX:.3f}, end t\' {endTime}', flush = True)
    wave.run()
    summary = wave.summary()

    expected = waveTheory.exactSolitaryWaveSpeed(height, 1.0) / math.tan(math.radians(angle))
    used = [y for y in sections if not any(abs(y - s) < 1e-9 for s in skipped)]
    ys = np.array(used)
    tanAngle = math.tan(math.radians(angle))
    record = {'case': 'obliqueReef', 'particlesPerDepth': ppd, 'span': span, 'zone': useZone, 'skipped': skipped,
              'closedForm': expected, 'summary': summary, 'crest': {}}
    for distance in distances:
        times = []
        for y in used:
            tracker = wave.trackers[y]
            times.append(passage(np.array(tracker.times, float), np.array(tracker.crestX, float), y * tanAngle + distance))
        result = crossing(ys, np.array(times), expected)
        record['crest'][f'{distance:+.0f}'] = result
        print(f'crest {distance:+5.1f} h0 past the toe: crosses at {result["speed"]:.3f} m/s ({result["error"]:+.1f} percent '
              f'against {expected:.3f}) through {result["sections"]} sections, rms {result["rms"]:.3f} h0', flush = True)
    sectionSummaries = summary['sections']
    breaking = np.array([sectionSummaries[y]['breakingTime'] if sectionSummaries[y]['broke'] else math.nan for y in used], float)
    record['breaking'] = crossing(ys, breaking, expected)
    record['breakingPastToe'] = {f'{y:+.0f}': (sectionSummaries[y]['breakingCrestX'] - y * tanAngle
                                               if sectionSummaries[y]['broke'] else None) for y in used}
    print(f'the break crosses at {record["breaking"]["speed"]:.3f} m/s ({record["breaking"]["error"]:+.1f} percent); '
          f'{summary["fluidParticles"]} fluid particles, {summary["wallClockSeconds"]:.0f} s', flush = True)
    writeResult(f'obliqueReef_{ppd}_span{span:.0f}' + ('_zone' if useZone else ''), record)
    return record

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'A wave crossing a uniform oblique reef at 30 degrees.')
    parser.add_argument('ppd', type = int, help = 'particles per depth')
    parser.add_argument('--span', type = float, default = 16.0, help = 'the tank\'s width [h0]')
    parser.add_argument('--zone', action = 'store_true', help = "split the crest's water once")
    parser.add_argument('--skip', type = float, nargs = '*', default = [], help = 'sections to leave out of the fits')
    arguments = parser.parse_args()
    main(arguments.ppd, arguments.span, arguments.zone, arguments.skip)
