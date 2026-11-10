
# -- DualSPHysics on pySurf's 2D Gate Cases -- #

'''

Runs pySurf's phase 1 gate cases in DualSPHysics and measures them with pySurf's own tools
(phase 1.5 of planning/gpuSphRoadmap.md):

    grilli      H0/h0 = 0.3 on a 1:35 plane beach; the breaking point against Grilli et al. (1997):
                Hb/h0 within 10 percent of 0.476, hb/h0 within 15 percent of 0.345
    surfRanch   H0/h0 = 0.6 over the Surf Ranch reef (face slope 0.0693, shelf 0.371 h0 deep, toe
                at x = 30 h0); the overturn at touchdown against Feddersen et al.'s (2024) DNS
                (impact x within 0.5 h0, H within 5 percent, A_O/H^2 within 15, A_J/H^2 within 20,
                theta_O within 5 degrees), and the surface 0.1 sqrt(h0/g) before the boundary
                element model's impact within two spacings of that model's
    mostert     H0/h0 = 0.3 on a 3 degree plane beach, Mostert and Deike's (2020) plunging case;
                set up here, run and measured by mostertRuns.py

Each case keeps pySurf's coordinates and bed (`dsphCase.SolitaryCase`), extended flat offshore to a
piston that generates the solitary wave. Three stages:

    tune        runs to just past the crest's passage of pySurf's starting crest position, reads
                the wave's height there from the free-surface gauge (its still-water reading at
                t = 0 subtracted), and scales the paddle's target height until the gauge reads
                the case's height within 2 percent: this matches the input, not the result
    run         runs with the tuned paddle, writing Part files at the breaking tracker's interval,
                0.01 sqrt(h0/g), over the window where pySurf's runs break or touch down
    measure     feeds the fluid of each Part file, its time shifted so the crest passes the gauge at
                t = 0 as pySurf's starting crest does, through `waveBreaking.BreakingTracker` set up
                as `BreakingWave.build` sets it, and reports `BreakingWave.summary`'s measures

    python gateRuns.py <grilli|surfRanch> <ppd> [paddle height to start from] [--reuse]
                       [--window start end] [--slip mode] [--measure]

--reuse takes the paddle already tuned for the case; --window replaces the tracking window [s,
pySurf time]; --measure reads the finished run again with its saved tuning and window (a window
not saved is recovered from the run's Part file times); --slip sets mDBC's slip mode and adds _slip<mode> to the run's name. Free slip (3)
fails in 2D with GenCase's normals from the particles: without mDBC's no-penetration option the water
leaves the domain within 1.5 s; with it (set for slip modes 2 and 3) the still water level falls 0.09
h0 in 4 s.
Runs land in ~/dsphRuns/<case><ppd> in WSL; results in pySurf/temp/phase1_5/<case><ppd>.json. These
are cross-checks: none validates pySurf.

Sean Bowman [11/08/2026]

'''

# Standard library imports
from __future__ import annotations

import json
import math
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

# Third-party imports
import numpy as np
from scipy.spatial import cKDTree

tools = Path(__file__).resolve().parent
sys.path.insert(0, str(tools.parents[1] / 'src'))
sys.path.insert(0, str(tools))

# Local imports
from dsphCase import SolitaryCase
from dsphOutput import partTimes, readPartArrays
from pysurf import constants
from pysurf.bathymetry import Bathymetry
from pysurf.BreakingWave import BreakingWave
from pysurf.waveBreaking import BreakingTracker, particleSurface

# The WSL distribution and user the runs live under: every tool here takes its run folders from these
wslDistribution, wslUser = 'Ubuntu-24.04', 'seanb'
wslRuns = f'/home/{wslUser}/dsphRuns'
windowsRuns = f'//wsl.localhost/{wslDistribution}{wslRuns}'
results = tools.parents[1] / 'temp' / 'phase1_5'
# This folder as WSL mounts it (C:\a\b becomes /mnt/c/a/b), for the run script
runScript = f'/mnt/{tools.drive[0].lower()}{tools.as_posix()[len(tools.drive):]}/runCase.sh'
grilliHeight, grilliDepth = 0.476, 0.476 / 1.380
dns = {'impactX': 42.05, 'impactHeightOverDepth': 0.64, 'overturnAreaOverImpactHeightSquared': 0.352,
       'jetAreaOverImpactHeightSquared': 0.22, 'overturnAngle': 33.0}
# A tracking window [s, pySurf time] given on the command line, in place of the case's
windowOverride = None
# mDBC's slip mode given on the command line (--slip), in place of the default 1
slipOverride = None

def setup(case: str, ppd: int) -> dict:

    '''pySurf's wave object (for its geometry and measures), the bed, and the case's windows in pySurf time [s].'''

    if case == 'grilli':
        wave = BreakingWave(depth = 1.0, waveHeight = 0.3, beachSlope = 1.0 / 35.0, particlesPerDepth = ppd, scheme = 'deltaSph')
        bed = Bathymetry.planeBeach(depth = 1.0, toeX = wave.toeX, slope = 1.0 / 35.0,
                                    endX = wave.toeX + (1.0 + wave.runupAllowance) * 35.0)
        # pySurf's runs break at 8.37 to 8.48 s; the tracker reads from 6.5 s to 9.5 s
        return {'wave': wave, 'bed': bed, 'window': (6.5, 9.5), 'stop': 'breaking'}
    if case == 'surfRanch':
        slope, shelf = 0.0693, 0.371
        bed = Bathymetry.reefBreak(depth = 1.0, toeX = 30.0, faceSlope = slope, reefDepth = shelf,
                                   reefLength = 60.0 - (30.0 + (1.0 - shelf) / slope))
        wave = BreakingWave(depth = 1.0, waveHeight = 0.6, beachSlope = slope, bathymetry = bed, startX = 15.0,
                            particlesPerDepth = ppd, scheme = 'deltaSph', startingWave = 'serre')
        # pySurf's runs touch down at 6.32 to 6.56 s; the boundary element model's comparison time is 6.28 s
        return {'wave': wave, 'bed': bed, 'window': (5.0, 7.5), 'stop': 'touchdown'}
    if case == 'mostert':
        # Mostert and Deike's plunging case, measured by mostertRuns.py
        slope = math.tan(math.radians(3.0))
        wave = BreakingWave(depth = 1.0, waveHeight = 0.3, beachSlope = slope, particlesPerDepth = ppd, scheme = 'deltaSph')
        bed = Bathymetry.planeBeach(depth = 1.0, toeX = wave.toeX, slope = slope,
                                    endX = wave.toeX + (1.0 + wave.runupAllowance) / slope)
        # pySurf's 64 ppd run breaks at 6.12 s and ends active breaking at 9.29 s
        return {'wave': wave, 'bed': bed, 'window': (5.0, 10.0), 'stop': 'breaking'}
    if case == 'ridge':
        # Guyenne and Grilli's ridge, run and measured by ridgeRuns.py: the centre line's section
        bed = Bathymetry.planeBeach(depth = 1.0, toeX = 5.225, slope = 1.0 / 15.0, endX = 19.0)
        wave = BreakingWave(depth = 1.0, waveHeight = 0.6, beachSlope = 1.0 / 15.0, bathymetry = bed, startX = 5.785,
                            particlesPerDepth = ppd, scheme = 'deltaSph')
        # Guyenne and Grilli's 3D runs break at 7.66 to 8.64 sqrt(h0/g), 2.45 to 2.76 s
        return {'wave': wave, 'bed': bed, 'window': (1.8, 4.5), 'stop': 'breaking'}
    raise ValueError(f"Case '{case}' is not 'grilli', 'surfRanch', 'mostert' or 'ridge'.")

def solitaryCase(case: str, ppd: int, paddleHeight: float, endTime: float, window: tuple | None, coarseStep: float) -> SolitaryCase:

    '''The DualSPHysics case for one paddle height: the piston two half lengths and a stroke offshore of the gauge.'''

    spec = setup(case, ppd)
    wave, d = spec['wave'], 1.0
    stroke = math.sqrt(16.0 * paddleHeight * d / 3.0)
    return SolitaryCase(name = case, bathymetry = spec['bed'], depth = d, paddleHeight = paddleHeight,
                        gaugeX = wave.crestX, spacing = d / ppd, paddleX = wave.crestX - stroke - 2.0 * wave.halfLength,
                        endTime = endTime, trackWindow = window, coarseStep = coarseStep,
                        slipMode = slipOverride if slipOverride is not None else 1,
                        noPenetration = slipOverride is not None and slipOverride >= 2)

def runCase(runName: str, case: SolitaryCase) -> float:

    '''Write the case, run GenCase and the GPU solver in WSL, return the wall time [s].'''

    case.write(f'{windowsRuns}/{runName}')
    started = time.perf_counter()
    done = subprocess.run(['wsl.exe', '--exec', 'bash', runScript, f'{wslRuns}/{runName}', case.name],
                          capture_output = True, text = True)
    if done.returncode != 0:
        raise RuntimeError(f'{runName} failed:\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}')
    return time.perf_counter() - started

def gaugeCrest(runName: str, depth: float, target: float) -> tuple[float, float]:

    '''The crest's passage time [s] and height [m] at the gauge: the first peak, the still-water reading at t = 0 subtracted.'''

    data = np.genfromtxt(f'{windowsRuns}/{runName}/out/GaugesSWL_Crest.csv', delimiter = ';', skip_header = 1)
    t, eta = data[:, 0], data[:, 3] - data[0, 3]
    rising = np.flatnonzero(eta > 0.5 * target)
    if rising.size == 0:
        raise RuntimeError(f'{runName}: the wave never reached half its height at the gauge.')
    # The first peak: within the time a wave length takes to pass after the gauge first sees half the height
    first = rising[0]
    span = (t >= t[first]) & (t <= t[first] + 3.0 * math.sqrt(depth / constants.gravity) * 4.0)
    index = np.flatnonzero(span)[int(np.argmax(eta[span]))]
    # A parabola through the three samples about the peak
    i0, i1, i2 = index - 1, index, index + 1
    a, b, c = np.polyfit(t[[i0, i1, i2]] - t[i1], eta[[i0, i1, i2]], 2)
    offset = -b / (2.0 * a) if a < 0 else 0.0
    return float(t[i1] + offset), float(c - b * b / (4.0 * a) if a < 0 else eta[i1])

def boundaryCount(runName: str) -> int:

    '''Boundary particles of the run (fixed and moving), from its GenCase log: fluid particles have Idp at or above it.'''

    log = Path(f'{windowsRuns}/{runName}/gencase.log').read_text(encoding = 'utf-8', errors = 'replace')
    return int(re.search(r'bound=(\d+)', log).group(1))

def tune(case: str, ppd: int, start: float | None, build = None, prefix: str | None = None) -> dict:

    '''

    Scale the paddle's target height until the gauge reads the case's height within 2 percent.
    `build(paddleHeight)` returns the trial case (by default `solitaryCase` with the case's own
    scheme); the trials run as <prefix>_tune<n>, by default <case><ppd>_tune<n>.

    '''

    spec = setup(case, ppd)
    target = spec['wave'].waveHeight
    paddle = start if start is not None else target
    build = build or (lambda height: solitaryCase(case, ppd, height, 1.0, None, 1.0))
    prefix = prefix or f'{case}{ppd}'
    history = []
    for attempt in range(5):
        runName = f'{prefix}_tune{attempt}'
        # Travel from the paddle to the gauge at the crest speed, plus the generation and the crest's
        # passage: 11 sqrt(h0/g) past the travel time (the first 32 ppd run passed 2.6 s after it)
        trial = build(paddle)
        travel = (trial.gaugeX - trial.paddleX) / math.sqrt(constants.gravity * (1.0 + target))
        trial.endTime = travel + 11.0 * math.sqrt(1.0 / constants.gravity)
        wall = runCase(runName, trial)
        passage, height = gaugeCrest(runName, 1.0, target)
        history.append({'paddleHeight': paddle, 'gaugeHeight': height, 'passage': passage, 'wall': wall})
        print(f'  {runName}: paddle {paddle:.4f} m, gauge {height:.4f} m ({100 * (height / target - 1):+.2f} percent) '
              f'at {passage:.3f} s; {wall:.0f} s', flush = True)
        if abs(height / target - 1.0) <= 0.01:
            break
        # The generated height grows nearly in proportion to the paddle's: step by the secant, or scale
        # when the secant's slope is below half the proportional one (the gauge's peak carries about 1
        # percent of noise, so close trials can give a flat or negative slope)
        slope = 0.0
        if len(history) >= 2 and abs(history[-1]['paddleHeight'] - history[-2]['paddleHeight']) > 1e-9:
            slope = ((history[-1]['gaugeHeight'] - history[-2]['gaugeHeight'])
                     / (history[-1]['paddleHeight'] - history[-2]['paddleHeight']))
        if slope > 0.5 * height / paddle:
            paddle = paddle + (target - height) / slope
        else:
            paddle = paddle * target / height
    best = min(history, key = lambda entry: abs(entry['gaugeHeight'] / target - 1.0))
    return {'paddleHeight': best['paddleHeight'], 'gaugeHeight': best['gaugeHeight'],
            'gaugeError': best['gaugeHeight'] / target - 1.0, 'passage': best['passage'], 'history': history,
            'withinTwoPercent': abs(best['gaugeHeight'] / target - 1.0) <= 0.02}

def surfaceGap(sph: np.ndarray, reference: np.ndarray, crestX: float, spacing: float) -> tuple:

    '''Largest and rms distance from each surface to the other over the overturn, in spacings (as the pySurf gate script measures it).'''

    def resample(line: np.ndarray, step: float) -> np.ndarray:
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(line, axis = 0).T))])
        samples = np.arange(0.0, arc[-1], step)
        return np.column_stack([np.interp(samples, arc, line[:, 0]), np.interp(samples, arc, line[:, 1])])

    box = lambda line, margin: line[(line[:, 0] > crestX - 2.0 - margin) & (line[:, 0] < crestX + 2.0 + margin)
                                    & (line[:, 1] > 1.0 - 0.05 - margin)]
    a, b = resample(sph, 0.1 * spacing), resample(reference, 0.1 * spacing)
    # A surface that never enters the overturn's box (a wave lost before the comparison time) has no gap
    if min(len(box(a, 0.0)), len(box(b, 0.0))) == 0:
        return (float('nan'),) * 4
    toB = cKDTree(box(b, 0.5)).query(box(a, 0.0))[0]
    toA = cKDTree(box(a, 0.5)).query(box(b, 0.0))[0]
    both = np.concatenate([toB, toA]) / spacing
    return float(both.max()), float(np.sqrt(np.mean(both**2))), float(toB.max() / spacing), float(toA.max() / spacing)

def measure(case: str, ppd: int, runName: str, passage: float) -> dict:

    '''Feed the run's Part files through pySurf's breaking tracker; return BreakingWave.summary's measures.'''

    spec = setup(case, ppd)
    if windowOverride is not None:
        spec['window'] = windowOverride
    wave, bed = spec['wave'], spec['bed']
    s, d = wave.spacing, wave.depth
    # As BreakingWave.build sets them up
    wave.bathymetry = bed
    wave.tracker = BreakingTracker(bed, s, faceReach = d, searchFromX = wave.toeX, faceMeasure = 'contour')
    tracker = wave.tracker
    nBound = boundaryCount(runName)
    times = partTimes(f'{windowsRuns}/{runName}/out')
    window = spec['window']
    frames = sorted(n for n, t in times.items() if window[0] - 1e-6 <= t - passage <= window[1] + 1e-6)
    extra = {}
    if case == 'surfRanch':
        bem = pickle.load(open(tools.parents[1] / 'temp' / 'phase1' / 'surfRanchBem.pkl', 'rb'))
        impactTime = bem['summary']['endTime']
        compareTarget = impactTime - 0.1 * math.sqrt(1.0 / constants.gravity)
        compareTime, bemSurface = min(bem['frames'], key = lambda frame: abs(frame[0] - compareTarget))
    for n in frames:
        arrays = readPartArrays(f'{windowsRuns}/{runName}/out/data/Part_{n:04d}.bi4')
        position = arrays['Pos'].astype(float)
        fluid = position[(arrays['Idp'] >= nBound), :][:, [0, 2]]
        fluid = fluid[fluid[:, 0] >= bed.startX]
        now = times[n] - passage
        tracker.update(now, fluid)
        if case == 'surfRanch' and 'surfaceGapLargest' not in extra and now >= compareTime - 0.5 * wave.trackInterval * math.sqrt(d / constants.gravity):
            crestX = float(bemSurface[np.argmax(bemSurface[:, 1]), 0])
            window2 = particleSurface(fluid, s, np.array([crestX - 3.0, 1.0 - 0.2]), np.array([crestX + 3.0, 2.0]))
            largest, rms, toBem, toSph = surfaceGap(window2.surface, bemSurface, crestX, s)
            extra.update({'surfaceGapLargest': largest, 'surfaceGapRms': rms, 'surfaceGapSphToBem': toBem,
                          'surfaceGapBemToSph': toSph, 'surfaceGapTime': now, 'compareTime': compareTime})
        if (spec['stop'] == 'breaking' and tracker.breakingFinal) or \
           (spec['stop'] == 'touchdown' and tracker.touchdownSurface is not None):
            break
    summary = wave.summary()
    summary.update(extra)
    summary['framesRead'] = len(frames)
    summary['lastTime'] = times[frames[-1]] - passage if frames else None
    return summary

def fineWindow(runName: str, passage: float, coarseStep: float = 0.5) -> tuple:

    '''The span of a finished run's finely spaced Part files [s after the gauge passage].'''

    times = partTimes(f'{windowsRuns}/{runName}/out')
    t = np.array(sorted(times.values())) - passage
    fine = np.flatnonzero(np.diff(t) < 0.5 * coarseStep)
    return float(t[fine[0]]), float(t[fine[-1] + 1])

def main(case: str, ppd: int, start: float | None, window: tuple | None = None, reuse: bool = False,
         measureOnly: bool = False) -> None:

    '''

    Tune, run and measure one case at one resolution; print and save the comparison. With
    `measureOnly`, measure the finished run again over its own window, keeping its saved tuning,
    gauge reading and run time.

    '''

    global windowOverride
    results.mkdir(parents = True, exist_ok = True)
    spec = setup(case, ppd)
    baseName = f'{case}{ppd}'
    runName = baseName + (f'_slip{slipOverride}' if slipOverride is not None else '')
    if measureOnly:
        saved = json.loads((results / f'{runName}.json').read_text(encoding = 'utf-8'))
        tuned, passage, height, wall = saved['tuning'], saved['passage'], saved['gaugeHeight'], saved['runWall']
        if window is None:
            window = tuple(saved['window']) if 'window' in saved else fineWindow(runName, passage)
        spec['window'] = windowOverride = window
        print(f'{runName}: measuring the finished run from {window[0]:.3f} to {window[1]:.3f} s', flush = True)
    else:
        if reuse:
            tuned = json.loads((results / f'{baseName}.json').read_text(encoding = 'utf-8'))['tuning']
            print(f'{case} {ppd} ppd: the tuned paddle, {tuned["paddleHeight"]:.4f} m', flush = True)
        else:
            print(f'{case} {ppd} ppd: tuning the paddle', flush = True)
            tuned = tune(case, ppd, start)
        if window is not None:
            spec['window'] = windowOverride = window
        window = spec['window']
        final = solitaryCase(case, ppd, tuned['paddleHeight'], tuned['passage'] + window[1] + 0.05,
                             (tuned['passage'] + window[0], tuned['passage'] + window[1]), 0.5)
        wall = runCase(runName, final)
        passage, height = gaugeCrest(runName, 1.0, spec['wave'].waveHeight)
        print(f'{runName}: run {wall:.0f} s; gauge {height:.4f} m at {passage:.3f} s', flush = True)
    summary = measure(case, ppd, runName, passage)
    summary.update({'tuning': tuned, 'gaugeHeight': height, 'passage': passage, 'runWall': wall, 'window': tuple(window)})
    if case == 'grilli' and summary.get('broke'):
        hb, db = summary['breakingHeightOverDepth'], summary['breakingDepthOverDepth']
        summary['passesGrilli'] = abs(hb / grilliHeight - 1) <= 0.10 and abs(db / grilliDepth - 1) <= 0.15
        print(f'{runName}: Hb/d {hb:.3f} ({100 * (hb / grilliHeight - 1):+.1f} percent), hb/d {db:.3f} '
              f'({100 * (db / grilliDepth - 1):+.1f} percent), t {summary["breakingTime"]:.3f} s; '
              f'{"passes" if summary["passesGrilli"] else "fails"}', flush = True)
    if case == 'surfRanch':
        if summary.get('touchedDown'):
            limits = {'impactX': ('abs', 0.5), 'impactHeightOverDepth': ('rel', 0.05),
                      'overturnAreaOverImpactHeightSquared': ('rel', 0.15), 'jetAreaOverImpactHeightSquared': ('rel', 0.20),
                      'overturnAngle': ('abs', 5.0)}
            # A measure the tracker could not read (None) fails its criterion
            passed = {key: summary.get(key) is not None
                      and (abs(summary[key] - dns[key]) <= tol if kind == 'abs' else abs(summary[key] / dns[key] - 1) <= tol)
                      for key, (kind, tol) in limits.items()}
            summary['criteriaPassed'] = passed
            value = lambda key: 'none' if summary.get(key) is None else f'{summary[key]:.3f}'
            line = ', '.join(f'{key} {value(key)} (DNS {dns[key]}){"" if passed[key] else " FAILS"}' for key in dns)
            print(f'{runName}: touchdown at {summary["touchdownTime"]:.3f} s: {line}; {sum(passed.values())} of 5 pass; '
                  f'Hb/d {summary.get("breakingHeightOverDepth") or float("nan"):.3f}', flush = True)
        else:
            print(f'{runName}: no touchdown read by {summary["lastTime"]:.3f} s', flush = True)
        if 'surfaceGapLargest' in summary:
            print(f'{runName}: surface gap at {summary["surfaceGapTime"]:.4f} s: largest {summary["surfaceGapLargest"]:.2f} '
                  f'spacings (SPH to BEM {summary["surfaceGapSphToBem"]:.2f}, BEM to SPH {summary["surfaceGapBemToSph"]:.2f}), '
                  f'rms {summary["surfaceGapRms"]:.2f}', flush = True)
    (results / f'{runName}.json').write_text(json.dumps(summary, indent = 1, default = lambda v: v.tolist() if hasattr(v, 'tolist') else float(v)),
                                             encoding = 'utf-8')
    print('RUN-END', flush = True)

if __name__ == '__main__':
    arguments = sys.argv[1:]
    reuse = '--reuse' in arguments
    if '--slip' in arguments:
        at = arguments.index('--slip')
        slipOverride = int(arguments[at + 1])
        del arguments[at:at + 2]
    window = None
    if '--window' in arguments:
        at = arguments.index('--window')
        window = (float(arguments[at + 1]), float(arguments[at + 2]))
        del arguments[at:at + 3]
    measureOnly = '--measure' in arguments
    arguments = [a for a in arguments if a not in ('--reuse', '--measure')]
    if len(arguments) < 2:
        sys.exit('python gateRuns.py <grilli|surfRanch> <ppd> [paddle height to start from] [--reuse] [--window start end] '
                 '[--slip mode] [--measure]')
    main(arguments[0], int(arguments[1]), float(arguments[2]) if len(arguments) > 2 else None, window, reuse, measureOnly)
