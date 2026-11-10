
# -- DualSPHysics on the Surf Ranch Reef Turned Across the Span (3D) -- #

'''

Runs the Surf Ranch reef turned 25 degrees across a 4 h0 span (dsphReef.py; the geometry of
openFoam/cases/breakingWave3D) in DualSPHysics and reports how the H0/h0 = 0.6 soliton breaks
across it: phase 3's end-to-end case of planning/gpuSphRoadmap.md, run in another code. No
measurement exists for this case, so nothing here is a pass or a fail: the results are reported, and
later serve as a cross-check for pySurf's 3D model and the OpenFOAM case.

    sections    on each saved slice (two rows centred on y = 0, 0.5, ..., 4 h0), the section's
                bed is the 2D Surf Ranch profile with its toe at x = 30 + y tan(25 deg).
                `BreakingTracker` (set up as `BreakingWave.build` sets it) reads the breaking
                point and the overturn at touchdown, and `BreakingWave.summary` reports them as the
                2D gate does (impact x, impact height, A_O/H^2, A_J/H^2, theta_O); each particle
                stands for s^3 / (2 s) of the section's plane
    peel        the breaking point's travel across the span: a line fitted to the breaking times
                against y of the sections clear of the side walls (0 < y < 4 h0), its speed set
                against c / tan(25 deg), with c the mean crest speed those sections' trackers
                record over the second before breaking

The side walls are free-slip, as OpenFOAM's are, so the tank is one bay of a reef mirrored at each
wall: a chevron, not an endless oblique reef, and c / tan(25 deg) (the endless reef's peel speed)
gives the scale only. The two slices against the walls are reported and left out of the fit: at
16 ppd a skin of water lifting off the wall at y = 4 h0 read as a breaking face 4.5 h0 from the toe.
    plan        the water standing above still water at six times through the break, in plan,
                coloured by height (temp/phase1_5/reef<ppd>_plan.png)

The paddle is tuned in a slab of the tank 0.5 h0 wide, from the ridge's tuned height, until the gauge
at x = 20 h0 reads 0.6 h0 within 2 percent (`gateRuns.tune`); the full span's own gauge is then held
to the same 2 percent. Times count from the crest's passage of x = 20 h0, where OpenFOAM's case
generates its wave.

    python reefRuns.py <ppd> [paddle height to start from] [--measure]

Results in pySurf/temp/phase1_5/reef<ppd>.json. These are cross-checks: none validates pySurf.

Sean Bowman [11/09/2026]

'''

# Standard library imports
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Third-party imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

tools = Path(__file__).resolve().parent
sys.path.insert(0, str(tools.parents[1] / 'src'))
sys.path.insert(0, str(tools))

# Local imports
import gateRuns
from dsphOutput import partTimes, readPartArrays
from dsphReef import ReefCase, reefSlope, reefToeX, shelfDepth
from pysurf import constants
from pysurf.BreakingWave import BreakingWave
from pysurf.bathymetry import Bathymetry
from pysurf.waveBreaking import BreakingTracker

ridgeTunedPaddle = 0.6121               # [m], the ridge's slab at 32 ppd, slip mode 3 (ridge25_32.json)
trackWindow = (3.0, 8.0)                # [s] after the crest's passage of the gauge

def reefCase(ppd: int, paddleHeight: float, endTime: float, window: tuple | None, width: float = 4.0) -> ReefCase:

    '''The case for one paddle height: the piston two half lengths and a stroke offshore of the gauge, as in 2D.'''

    wave = gateRuns.setup('surfRanch', ppd)['wave']
    stroke = math.sqrt(16.0 * paddleHeight / 3.0)
    gaugeX = 20.0
    return ReefCase(name = 'reef', k = 0.0, paddleHeight = paddleHeight, spacing = 1.0 / ppd,
                    paddleX = gaugeX - stroke - 2.0 * wave.halfLength, endTime = endTime, width = width,
                    gaugeX = gaugeX, gaugeY = 0.5 * width, trackWindow = window, coarseStep = 0.1, slipMode = 3,
                    sliceCentres = tuple(0.5 * i for i in range(9)) if width >= 4.0 else (0.5 * width,))

def sectionWave(case: ReefCase, ppd: int, y: float) -> BreakingWave:

    '''pySurf's 2D Surf Ranch wave on the section at y, its toe moved to x = 30 + y tan(angle), with its tracker set up.'''

    toeX = reefToeX + y * case.shift
    bed = Bathymetry.reefBreak(depth = 1.0, toeX = toeX, faceSlope = reefSlope, reefDepth = shelfDepth,
                               reefLength = case.endX - (toeX + (1.0 - shelfDepth) / reefSlope))
    wave = BreakingWave(depth = 1.0, waveHeight = 0.6, beachSlope = reefSlope, bathymetry = bed, startX = case.gaugeX,
                        particlesPerDepth = ppd, scheme = 'deltaSph', startingWave = 'serre')
    wave.bathymetry = bed
    wave.tracker = BreakingTracker(bed, wave.spacing, faceReach = 1.0, searchFromX = toeX, faceMeasure = 'contour')
    return wave

def measure(ppd: int, runName: str, passage: float, case: ReefCase) -> dict:

    '''Each section's breaking point and overturn at touchdown, and the peel across the span.'''

    s = case.spacing
    outDir = f'{gateRuns.windowsRuns}/{runName}/out'
    times = partTimes(outDir)
    frames = sorted(n for n, t in times.items() if trackWindow[0] - 1e-6 <= t - passage <= trackWindow[1] + 1e-6)
    sections = [(centre, bounds, sectionWave(case, ppd, centre))
                for centre, bounds in zip(case.sliceCentres, case.sliceBounds())]
    for n in frames:
        if all(wave.tracker.touchdownSurface is not None for _, _, wave in sections):
            break
        position = readPartArrays(f'{outDir}/data/Part_{n:04d}.bi4')['Pos'].astype(float)
        now = times[n] - passage
        for centre, (low, high), wave in sections:
            if wave.tracker.touchdownSurface is not None:
                continue
            inside = position[(position[:, 1] >= low) & (position[:, 1] <= high)]
            xz = inside[:, [0, 2]]
            wave.tracker.update(now, xz, np.full(len(xz), s**3 / (high - low)))
    keys = ['broke', 'breakingTime', 'breakingHeightOverDepth', 'breakingDepthOverDepth', 'breakingDistanceFromToe',
            'touchedDown', 'touchdownTime', 'impactX', 'impactHeightOverDepth', 'overturnAreaOverImpactHeightSquared',
            'jetAreaOverImpactHeightSquared', 'overturnAngle']
    result = {'sections': [], 'framesRead': len(frames)}
    for centre, (low, high), wave in sections:
        summary = wave.summary()
        point = wave.tracker.breakingPoint
        entry = {'y': centre, 'slice': (low, high), 'againstWall': low <= 0.0 or high >= case.width,
                 **{key: summary.get(key) for key in keys}}
        if point is not None:
            entry['breakingCrestX'] = point.crestX
            # The crest speed into breaking: the tracker's crest positions over the last second
            # before its breaking point
            t0, x0 = np.array(wave.tracker.times), np.array(wave.tracker.crestX)
            recent = (t0 >= point.time - 1.0) & (t0 <= point.time)
            if recent.sum() > 2:
                entry['crestSpeed'] = float(np.polyfit(t0[recent], x0[recent], 1)[0])
        result['sections'].append(entry)
    inner = [entry for entry in result['sections'] if not entry['againstWall'] and entry.get('breakingTime') is not None]
    if len(inner) >= 2:
        y = np.array([entry['y'] for entry in inner])
        t = np.array([entry['breakingTime'] for entry in inner])
        slope = float(np.polyfit(y, t, 1)[0])
        result['peelSpeed'] = 1.0 / slope if abs(slope) > 1e-12 else None
        result['peelSections'] = len(inner)
        speeds = [entry['crestSpeed'] for entry in inner if 'crestSpeed' in entry]
        if speeds:
            result['crestSpeed'] = float(np.mean(speeds))
            result['peelSpeedFromCrest'] = result['crestSpeed'] / case.shift
    return result

def planFigure(ppd: int, runName: str, passage: float, case: ReefCase, sections: list) -> Path:

    '''The water above still water in plan at six times from the first breaking to the last touchdown.'''

    outDir = f'{gateRuns.windowsRuns}/{runName}/out'
    times = partTimes(outDir)
    marks = [entry['breakingTime'] for entry in sections if entry.get('breakingTime') is not None]
    marks += [entry['touchdownTime'] for entry in sections if entry.get('touchdownTime') is not None]
    start, end = (min(marks) - 0.3, max(marks) + 0.3) if marks else trackWindow
    fig, axes = plt.subplots(6, 1, figsize = (10, 13), sharex = True)
    y = np.linspace(0.0, case.width, 2)
    for ax, target in zip(axes, np.linspace(start, end, 6)):
        n = min(times, key = lambda m: abs(times[m] - passage - target))
        position = readPartArrays(f'{outDir}/data/Part_{n:04d}.bi4')['Pos'].astype(float)
        crest = position[position[:, 2] > case.depth + 2.0 * case.spacing]
        order = np.argsort(crest[:, 2])
        art = ax.scatter(crest[order, 0], crest[order, 1], c = crest[order, 2] - case.depth, s = 0.2, cmap = 'viridis',
                         vmin = 0.0, vmax = 0.8, rasterized = True)
        ax.plot(reefToeX + y * case.shift, y, 'k--', lw = 0.8)
        ax.plot(case.edgeX + y * case.shift, y, 'k-', lw = 0.8)
        for entry in sections:
            if entry.get('breakingTime') is not None and entry['breakingTime'] <= times[n] - passage:
                ax.plot(entry['breakingCrestX'], entry['y'], 'r+', ms = 8)
        ax.set_ylim(0.0, case.width)
        ax.set_aspect('equal')
        ax.set_ylabel('y / h0')
        ax.set_title(f't = {times[n] - passage:.2f} s after the crest passes x = 20 h0', fontsize = 9)
    axes[-1].set_xlabel('x / h0 (dashed: toe; solid: shelf edge; +: breaking points read so far)')
    axes[-1].set_xlim(case.trackFromX, case.endX)
    fig.colorbar(art, ax = axes, label = 'height above still water / h0', shrink = 0.5)
    path = gateRuns.results / f'reef{ppd}_plan.png'
    fig.savefig(path, dpi = 110)
    plt.close(fig)
    return path

def main(ppd: int, start: float | None, measureOnly: bool = False) -> None:

    '''Tune the paddle in a slab, run the full span, and report the break across it.'''

    gateRuns.results.mkdir(parents = True, exist_ok = True)
    target = gateRuns.setup('surfRanch', ppd)['wave'].waveHeight
    runName = f'reef{ppd}'
    resultPath = gateRuns.results / f'{runName}.json'
    if measureOnly:
        saved = json.loads(resultPath.read_text(encoding = 'utf-8'))
        tuned, wall = saved['tuning'], saved['runWall']
    else:
        print(f'reef, {ppd} ppd: tuning the paddle in a slab', flush = True)
        tuned = gateRuns.tune('surfRanch', ppd, start if start is not None else ridgeTunedPaddle, prefix = f'reefSlab{ppd}',
                              build = lambda height: reefCase(ppd, height, 1.0, None, width = 0.5))
        final = reefCase(ppd, tuned['paddleHeight'], tuned['passage'] + trackWindow[1] + 0.05,
                         (tuned['passage'] + trackWindow[0], tuned['passage'] + trackWindow[1]))
        wall = gateRuns.runCase(runName, final)
    passage, height = gateRuns.gaugeCrest(runName, 1.0, target)
    print(f'{runName}: run {wall:.0f} s; gauge {height:.4f} m ({100 * (height / target - 1):+.2f} percent) at {passage:.3f} s',
          flush = True)
    case = reefCase(ppd, tuned['paddleHeight'], 1.0, None)
    summary = {'particlesPerDepth': ppd, 'angle': case.angle, 'sliceCentres': case.sliceCentres, 'tuning': tuned,
               'runWall': wall, 'gaugeHeight': height,
               'passage': passage, 'gaugeWithinTwoPercent': abs(height / target - 1.0) <= 0.02}
    summary.update(measure(ppd, runName, passage, case))
    value = lambda entry, key, form: 'none' if entry.get(key) is None else format(entry[key], form)
    for entry in summary['sections']:
        print(f'{runName} y = {entry["y"]:.1f}{" (wall)" if entry["againstWall"] else ""}: breaks at {value(entry, "breakingTime", ".3f")} s, Hb/h0 '
              f'{value(entry, "breakingHeightOverDepth", ".3f")}, hb/h0 {value(entry, "breakingDepthOverDepth", ".3f")}, '
              f'crest x {value(entry, "breakingCrestX", ".3f")}; touches down at {value(entry, "touchdownTime", ".3f")} s, '
              f'impact x {value(entry, "impactX", ".3f")}, H/h0 {value(entry, "impactHeightOverDepth", ".3f")}, '
              f'A_O/H2 {value(entry, "overturnAreaOverImpactHeightSquared", ".3f")}, '
              f'A_J/H2 {value(entry, "jetAreaOverImpactHeightSquared", ".3f")}, '
              f'theta_O {value(entry, "overturnAngle", ".1f")}', flush = True)
    if summary.get('peelSpeed') is not None:
        print(f'{runName}: the breaking point crosses the span at {summary["peelSpeed"]:.2f} m/s '
              f'({summary["peelSections"]} sections clear of the walls)'
              + (f'; c / tan(angle) from their mean crest speed {summary["crestSpeed"]:.2f} m/s gives '
                 f'{summary["peelSpeedFromCrest"]:.2f}' if 'crestSpeed' in summary else ''), flush = True)
    summary['planFigure'] = str(planFigure(ppd, runName, passage, case, summary['sections']))
    resultPath.write_text(json.dumps(summary, indent = 1, default = lambda v: v.tolist() if hasattr(v, 'tolist') else float(v)),
                          encoding = 'utf-8')
    print('RUN-END', flush = True)

if __name__ == '__main__':
    arguments = sys.argv[1:]
    measureOnly = '--measure' in arguments
    arguments = [a for a in arguments if a != '--measure']
    if not arguments:
        sys.exit('python reefRuns.py <ppd> [paddle height to start from] [--measure]')
    main(int(arguments[0]), float(arguments[1]) if len(arguments) > 1 else None, measureOnly)
