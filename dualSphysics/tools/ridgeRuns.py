
# -- DualSPHysics on Guyenne and Grilli's Ridge (3D) -- #

'''

Runs the sloping ridge of Guyenne and Grilli (2006) in DualSPHysics, half the tank, and measures it
against their fully nonlinear potential flow with gate 2's criteria (phase 1.5 of
planning/gpuSphRoadmap.md; the criteria of tests/testValidationRidgeOverturn.py):

    breaking point   on the centre line, the start of the front face's last vertical run before
                     the jet's touchdown, read by `BreakingTracker` (set up as `BreakingWave.build`
                     sets it, on the centre line's section) from the fluid within two spacings of
                     y = 0, each particle's area s^3 / (2 s): time and crest height within 5
                     percent, crest position within 0.5 h0 (their table 1: k = 0.25 at 7.890,
                     16.093, 0.706; k = 0.50 at 8.639, 17.100, 0.740). Gate 2's one spacing of
                     y = 0 holds two rows in a full tank and one in the half tank; the two rows
                     nearest the plane keep the full tank's particles per area. One row's surface
                     is ragged enough to read a sloping face as vertical: at 32 ppd, k = 0.25, on
                     the first bed band, it fires at t = 5.94 sqrt(h0/g) and x = 13.3 h0, where
                     the two rows read 8.015. Two rows can still read a short vertical run that
                     slopes back, as on the corrected band from 7.514 to 7.534; the last run is
                     the face that overturns (planning/gpuSphRoadmap.md, decision 14)
    velocity         for k = 0.25, the horizontal velocity at z = -0.2 h0 at t = 7.973 sqrt(h0/g)
                     on their figure 16 grid (x from 14 to 19 h0 every 0.35, |y| up to 1.9 every
                     0.38, where the bed lies under that level): the largest speed within 10
                     percent of 0.651 sqrt(g h0), the largest spanwise velocity within 30 percent
                     of 0.085 of the largest streamwise one. The velocity at a grid point is the
                     Shepard-normalized Wendland average of the particles in the saved layer, the
                     layer mirrored across y = 0 (v changes sign) to complete the kernel there

Two stages:

    tune     a slab of the tank 0.5 h0 wide (`RidgeCase.width`), run to just past the crest's
             passage of x = 5.785 h0: the paddle's target height scaled until the gauge reads 0.6
             h0 within 2 percent (`gateRuns.tune`)
    run      the half tank with the tuned paddle, Part files every 0.01 sqrt(h0/g) over the
             tracking window, holding only the centre-line slice and the velocity layer

Times count from the crest's passage of x = 5.785 h0, as Guyenne and Grilli's start from the crest
there.

    python ridgeRuns.py <k> <ppd> [paddle height to start from] [--slip mode] [--measure]

Results in pySurf/temp/phase1_5/ridge<k * 100><ppd>.json. These are cross-checks: none validates
pySurf.

Sean Bowman [11/09/2026]

'''

# Standard library imports
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Third-party imports
import numpy as np
from scipy.spatial import cKDTree

tools = Path(__file__).resolve().parent
sys.path.insert(0, str(tools.parents[1] / 'src'))
sys.path.insert(0, str(tools))

# Local imports
import gateRuns
from dsphOutput import partTimes, readPartArrays
from dsphRidge import RidgeCase
from pysurf import constants
from pysurf.waveBreaking import BreakingTracker

timeScale = math.sqrt(1.0 / constants.gravity)
reference = {0.25: (7.890, 16.093, 0.706), 0.50: (8.639, 17.100, 0.740)}
velocityTime, velocityLevel = 7.973, 0.8
ridgeMaxSpeed, ridgeSpanwiseRatio = 0.651, 0.085

def ridgeCase(k: float, ppd: int, paddleHeight: float, endTime: float, window: tuple | None, slipMode: int,
              width: float = 4.0) -> RidgeCase:

    '''The case for one paddle height: the piston two half lengths and a stroke offshore of the gauge, as in 2D.'''

    wave = gateRuns.setup('ridge', ppd)['wave']
    stroke = math.sqrt(16.0 * paddleHeight / 3.0)
    return RidgeCase(name = 'ridge', k = k, paddleHeight = paddleHeight, spacing = 1.0 / ppd,
                     paddleX = wave.crestX - stroke - 2.0 * wave.halfLength, endTime = endTime, width = width,
                     gaugeY = 0.5 * width if width < 1.0 else 1.0, trackWindow = window, coarseStep = 0.1,
                     slipMode = slipMode)

def wendland(r: np.ndarray, h: float) -> np.ndarray:

    '''The 3D Wendland C2 kernel's shape, (1 - q/2)^4 (2q + 1) for q = r/h up to 2; the normalization cancels in a Shepard average.'''

    q = np.clip(r / h, 0.0, 2.0)
    return (1.0 - 0.5 * q)**4 * (2.0 * q + 1.0)

def gridVelocity(position: np.ndarray, velocity: np.ndarray, spacing: float, k: float) -> dict:

    '''The horizontal velocity on Guyenne and Grilli's figure 16 grid at z = -0.2 h0, in sqrt(g h0).'''

    layer = np.abs(position[:, 2] - velocityLevel) <= 1.5 * spacing
    p, v = position[layer], velocity[layer]
    # Mirror across the symmetry plane: y and v change sign
    p = np.vstack([p, p * np.array([1.0, -1.0, 1.0])])
    v = np.vstack([v, v * np.array([1.0, -1.0, 1.0])])
    x = np.arange(14.0, 19.0 + 1e-9, 0.35)
    y = np.arange(0.0, 1.90 + 1e-9, 0.38)
    gridX, gridY = np.meshgrid(x, y)
    bed = np.clip(gridX - 5.225, 0.0, None) / 15.0 / np.cosh(k * gridY)**2
    wet = bed < velocityLevel
    points = np.column_stack([gridX[wet], gridY[wet], np.full(int(wet.sum()), velocityLevel)])
    h = 1.5 * spacing
    tree = cKDTree(p)
    u = np.full((len(points), 2), np.nan)
    for i, neighbours in enumerate(tree.query_ball_point(points, 2.0 * h)):
        if len(neighbours) >= 8:
            weight = wendland(np.linalg.norm(p[neighbours] - points[i], axis = 1), h)
            u[i] = weight @ v[neighbours][:, :2] / weight.sum()
    found = np.all(np.isfinite(u), axis = 1)
    u = u / math.sqrt(constants.gravity)
    speed = np.hypot(u[found, 0], u[found, 1])
    return {'gridPoints': int(len(points)), 'gridPointsFound': int(found.sum()),
            'maxSpeed': float(speed.max()), 'spanwiseRatio': float(np.abs(u[found, 1]).max() / np.abs(u[found, 0]).max()),
            'points': points, 'velocity': u}

def measure(k: float, ppd: int, runName: str, passage: float) -> dict:

    '''The centre line's breaking point and, for k = 0.25, the velocity on the figure 16 grid.'''

    spec = gateRuns.setup('ridge', ppd)
    wave, bed, window = spec['wave'], spec['bed'], spec['window']
    s, d = wave.spacing, wave.depth
    tracker = BreakingTracker(bed, s, faceReach = d, searchFromX = wave.toeX, faceMeasure = 'contour')
    outDir = f'{gateRuns.windowsRuns}/{runName}/out'
    times = partTimes(outDir)
    frames = sorted(n for n, t in times.items() if window[0] - 1e-6 <= t - passage <= window[1] + 1e-6)
    # The saved slice holds two rows (y up to 2 dp); each particle stands for s^3 / width of the plane
    sliceWidth = 2.0 * s
    result = {}
    target = velocityTime * timeScale
    velocityFrame = min(frames, key = lambda n: abs(times[n] - passage - target)) if frames else None
    for n in frames:
        now = times[n] - passage
        if tracker.breakingFinal and (velocityFrame is None or n > velocityFrame):
            break
        arrays = readPartArrays(f'{outDir}/data/Part_{n:04d}.bi4')
        position = arrays['Pos'].astype(float)
        if not tracker.breakingFinal:
            centre = position[position[:, 1] <= sliceWidth]
            tracker.update(now, centre[:, [0, 2]], np.full(len(centre), s**3 / sliceWidth))
        if n == velocityFrame:
            result['velocityGrid'] = gridVelocity(position, arrays['Vel'].astype(float), s, k)
            result['velocityGrid']['time'] = now / timeScale
    point = tracker.breakingPoint
    if point is not None:
        result.update({'breakingTime': point.time / timeScale, 'breakingCrestX': point.crestX,
                       'breakingCrestHeight': point.crestHeight, 'breakingFaceDepth': point.faceDepth})
    result['framesRead'] = len(frames)
    return result

def main(k: float, ppd: int, start: float | None, slipMode: int = 3, measureOnly: bool = False) -> None:

    '''Tune the paddle in a slab, run the half tank, and measure it against Guyenne and Grilli.'''

    gateRuns.results.mkdir(parents = True, exist_ok = True)
    spec = gateRuns.setup('ridge', ppd)
    window, target = spec['window'], spec['wave'].waveHeight
    runName = f'ridge{round(100 * k)}_{ppd}' + ('' if slipMode == 3 else f'_slip{slipMode}')
    resultPath = gateRuns.results / f'{runName}.json'
    if measureOnly:
        saved = json.loads(resultPath.read_text(encoding = 'utf-8'))
        tuned, wall = saved['tuning'], saved['runWall']
    else:
        print(f'ridge k = {k}, {ppd} ppd, slip mode {slipMode}: tuning the paddle in a slab', flush = True)
        tuned = gateRuns.tune('ridge', ppd, start, prefix = f'ridgeSlab{round(100 * k)}_{ppd}_slip{slipMode}',
                              build = lambda height: ridgeCase(k, ppd, height, 1.0, None, slipMode, width = 0.5))
        final = ridgeCase(k, ppd, tuned['paddleHeight'], tuned['passage'] + window[1] + 0.05,
                          (tuned['passage'] + window[0], tuned['passage'] + window[1]), slipMode)
        wall = gateRuns.runCase(runName, final)
    passage, height = gateRuns.gaugeCrest(runName, 1.0, target)
    print(f'{runName}: run {wall:.0f} s; gauge {height:.4f} m ({100 * (height / target - 1):+.2f} percent) at {passage:.3f} s',
          flush = True)
    summary = {'k': k, 'particlesPerDepth': ppd, 'slipMode': slipMode, 'tuning': tuned, 'runWall': wall,
               'gaugeHeight': height, 'passage': passage}
    summary.update(measure(k, ppd, runName, passage))
    if 'breakingTime' in summary and k in reference:
        t, x, hb = reference[k]
        passed = {'time': abs(summary['breakingTime'] / t - 1.0) <= 0.05, 'crestX': abs(summary['breakingCrestX'] - x) <= 0.5,
                  'crestHeight': abs(summary['breakingCrestHeight'] / hb - 1.0) <= 0.05}
        summary['criteriaPassed'] = passed
        print(f'{runName}: breaks at t {summary["breakingTime"]:.3f} sqrt(h0/g) (GG {t}, {100 * (summary["breakingTime"] / t - 1):+.1f} '
              f'percent), x {summary["breakingCrestX"]:.3f} h0 (GG {x}, {summary["breakingCrestX"] - x:+.3f}), crest '
              f'{summary["breakingCrestHeight"]:.3f} h0 (GG {hb}, {100 * (summary["breakingCrestHeight"] / hb - 1):+.1f} percent); '
              f'{sum(passed.values())} of 3 pass', flush = True)
    elif 'breakingTime' not in summary:
        print(f'{runName}: no breaking point read in the window {window}', flush = True)
    if 'velocityGrid' in summary and abs(k - 0.25) < 1e-9:
        grid = summary['velocityGrid']
        grid['passesSpeed'] = abs(grid['maxSpeed'] / ridgeMaxSpeed - 1.0) <= 0.10
        grid['passesRatio'] = abs(grid['spanwiseRatio'] / ridgeSpanwiseRatio - 1.0) <= 0.30
        print(f'{runName}: at t {grid["time"]:.3f} sqrt(h0/g), z = -0.2 h0: largest speed {grid["maxSpeed"]:.3f} sqrt(g h0) '
              f'(GG {ridgeMaxSpeed}, {100 * (grid["maxSpeed"] / ridgeMaxSpeed - 1):+.1f} percent), spanwise ratio '
              f'{grid["spanwiseRatio"]:.3f} (GG {ridgeSpanwiseRatio}, {100 * (grid["spanwiseRatio"] / ridgeSpanwiseRatio - 1):+.0f} '
              f'percent); {grid["gridPointsFound"]} of {grid["gridPoints"]} grid points', flush = True)
    resultPath.write_text(json.dumps(summary, indent = 1, default = lambda v: v.tolist() if hasattr(v, 'tolist') else float(v)),
                          encoding = 'utf-8')
    print('RUN-END', flush = True)

if __name__ == '__main__':
    arguments = sys.argv[1:]
    slipMode = 3
    if '--slip' in arguments:
        at = arguments.index('--slip')
        slipMode = int(arguments[at + 1])
        del arguments[at:at + 2]
    measureOnly = '--measure' in arguments
    arguments = [a for a in arguments if a != '--measure']
    if len(arguments) < 2:
        sys.exit('python ridgeRuns.py <k> <ppd> [paddle height to start from] [--slip mode] [--measure]')
    main(float(arguments[0]), int(arguments[1]), float(arguments[2]) if len(arguments) > 2 else None, slipMode, measureOnly)
