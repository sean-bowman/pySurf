# -- Gate 1: the Surf Ranch overturn at the jet's landing, and the surface before it -- #

'''

The Surf Ranch soliton of Feddersen et al. (2024): H/h0 = 0.6 started at x = 15 h0 as the Serre
wave, over a 0.0693 reef face from the toe at x = 30 h0 to a shelf 0.371 h0 deep and an end wall
at 60 h0. `BreakingWave` on the 'deltaSph' scheme from bed-fitted columns, run to the jet's landing.

Two references:

    overturn    Feddersen et al.'s two-phase DNS at the jet's landing: impact x = 42.05 h0,
                H_I/h0 = 0.64, A_O/H_I² = 0.352, A_J/H_I² = 0.22, theta_O = 33 degrees
    surface     pySurf's boundary element model (`OverturningWave`, validated against the same
                DNS) 0.1 sqrt(h0/g) before its own impact; it is run once and cached in
                results/surfRanchBem.pkl (about 20 minutes)

Criteria, set before the runs: impact x within 0.5 h0, H_I within 5 percent, A_O/H_I² within 15,
A_J/H_I² within 20, theta_O within 5 degrees; the two surfaces within 2 spacings of each other over
the overturn (crest x ± 2 h0, above 0.95 h0).

Recorded (2026-11-12): at 128 particles per depth all five overturn criteria pass (impact 0.04 h0
short, H_I -0.9 percent, A_O/H_I² -3, A_J/H_I² -11.5, theta_O -3.6 degrees); at 64 four pass
(theta_O -5.2 degrees fails); at 32 only A_J/H_I² passes. The surfaces lie up to 11, 15 and 14
spacings apart at 32, 64 and 128: fails. The overturn area does not converge monotonically
(0.495, 0.308, 0.341 H_I² against 0.352), so the pass at 128 is not a converged result.

    python surfRanchOverturn.py <ppd>

Writes results/surfRanchOverturn_<ppd>.json.

Sean Bowman [11/19/2026]

'''

import argparse
import math
import pickle
import time

import numpy as np
from scipy.spatial import cKDTree

import common
from common import percent, verdict, writeResult

from pysurf.bathymetry import Bathymetry
from pysurf.BreakingWave import BreakingWave
from pysurf.OverturningWave import OverturningWave
from pysurf.waveBreaking import particleSurface

dns = {'impactX': 42.05, 'impactHeightOverDepth': 0.64, 'overturnAreaOverImpactHeightSquared': 0.352,
       'jetAreaOverImpactHeightSquared': 0.22, 'overturnAngle': 33.0}
slope, shelf, toeX, endX = 0.0693, 0.371, 30.0, 60.0
bemCache = common.resultsDir / 'surfRanchBem.pkl'

def reef() -> Bathymetry:

    '''The Surf Ranch reef: the face from the toe to the shelf, the shelf to the end wall.'''

    return Bathymetry.reefBreak(depth = 1.0, toeX = toeX, faceSlope = slope, reefDepth = shelf,
                                reefLength = endX - (toeX + (1.0 - shelf) / slope))

def boundaryElementReference() -> dict:

    '''The boundary element run to its jet's impact: its summary and surfaces, from the cache when present.'''

    if bemCache.exists():
        with open(bemCache, 'rb') as handle:
            return pickle.load(handle)
    started = time.perf_counter()
    wave = OverturningWave(depth = 1.0, waveHeight = 0.6, bathymetry = reef(), beachSlope = slope, startX = 15.0,
                           crestSpacingRatio = 0.02)
    wave.run(progress = False)
    reference = {'summary': wave.summary(), 'frames': [(frame.time, frame.surface) for frame in wave.frames]}
    common.resultsDir.mkdir(parents = True, exist_ok = True)
    with open(bemCache, 'wb') as handle:
        pickle.dump(reference, handle, protocol = 5)
    print(f'boundary element reference: impact at {reference["summary"]["endTime"]:.4f} s, '
          f'{time.perf_counter() - started:.0f} s', flush = True)
    return reference

def resample(line: np.ndarray, step: float) -> np.ndarray:

    '''A polyline at points `step` apart along its arc.'''

    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(line, axis = 0).T))])
    samples = np.arange(0.0, arc[-1], step)
    return np.column_stack([np.interp(samples, arc, line[:, 0]), np.interp(samples, arc, line[:, 1])])

def surfaceGap(sph: np.ndarray, reference: np.ndarray, crestX: float, spacing: float) -> tuple:

    '''Largest and rms distance from each surface to the other over the overturn, in spacings.'''

    box = lambda line, margin: line[(line[:, 0] > crestX - 2.0 - margin) & (line[:, 0] < crestX + 2.0 + margin)
                                    & (line[:, 1] > 1.0 - 0.05 - margin)]
    a, b = resample(sph, 0.1 * spacing), resample(reference, 0.1 * spacing)
    toB = cKDTree(box(b, 0.5)).query(box(a, 0.0))[0]
    toA = cKDTree(box(a, 0.5)).query(box(b, 0.0))[0]
    both = np.concatenate([toB, toA]) / spacing
    return float(both.max()), float(np.sqrt(np.mean(both**2)))

def main(ppd: int, densityDiffusion: str = 'hydrostatic') -> dict:

    '''Run the case at `ppd` particles per depth and report it against the criteria.'''

    reference = boundaryElementReference()
    impactTime = reference['summary']['endTime']
    compareTime, bemSurface = min(reference['frames'], key = lambda frame: abs(frame[0] - (impactTime - 0.1 * common.timeScale)))
    started = time.perf_counter()
    wave = BreakingWave(depth = 1.0, waveHeight = 0.6, beachSlope = slope, bathymetry = reef(), startX = 15.0,
                        particlesPerDepth = ppd, scheme = 'deltaSph', startingWave = 'serre', backend = 'gpu',
                        frameInterval = 0.05, solverOptions = {'densityDiffusion': densityDiffusion})
    wave.run(endTime = compareTime, progress = True)

    # The surface before impact, read on the water within 3 h0 of the boundary element crest
    fluid = wave.particles.isFluid
    positions = wave.particles.positions[fluid].copy()
    crestX = float(bemSurface[np.argmax(bemSurface[:, 1]), 0])
    volumes = (wave.particles.masses[fluid] / wave.solver.config.referenceDensity
               if wave.particles.lengths is not None else None)
    window = particleSurface(positions, wave.spacing, np.array([crestX - 3.0, 0.8]), np.array([crestX + 3.0, 2.0]),
                             volumes = volumes)
    largest, rms = surfaceGap(window.surface, bemSurface, crestX, wave.spacing)
    surfacePasses = largest <= 2.0
    print(f'{ppd} ppd at {compareTime:.4f} s, 0.1 sqrt(h0/g) before the boundary element impact: surfaces up to '
          f'{largest:.2f} spacings apart (rms {rms:.2f}, limit 2): {verdict(surfacePasses)}', flush = True)

    wave.run(progress = True, stopAt = 'touchdown')
    summary = wave.summary()
    summary['wall'] = time.perf_counter() - started
    record = {'case': 'surfRanchOverturn', 'particlesPerDepth': ppd, 'densityDiffusion': densityDiffusion,
              'summary': summary, 'dns': dns,
              'surface': {'compareTime': compareTime, 'largestGap': largest, 'rmsGap': rms, 'passes': surfacePasses}}
    if summary.get('touchedDown'):
        checks = {'impactX': (summary['impactX'] - dns['impactX'], 0.5, 'h0'),
                  'impactHeightOverDepth': (percent(summary['impactHeightOverDepth'], dns['impactHeightOverDepth']), 5.0,
                                            'percent'),
                  'overturnAreaOverImpactHeightSquared': (percent(summary['overturnAreaOverImpactHeightSquared'],
                                                                  dns['overturnAreaOverImpactHeightSquared']), 15.0, 'percent'),
                  'jetAreaOverImpactHeightSquared': (percent(summary['jetAreaOverImpactHeightSquared'],
                                                             dns['jetAreaOverImpactHeightSquared']), 20.0, 'percent'),
                  'overturnAngle': (summary['overturnAngle'] - dns['overturnAngle'], 5.0, 'degrees')}
        record['overturn'] = {key: {'error': error, 'limit': limit, 'passes': abs(error) <= limit}
                              for key, (error, limit, unit) in checks.items()}
        for key, (error, limit, unit) in checks.items():
            print(f'  {key}: {summary[key]:.3f} (DNS {dns[key]}), {error:+.2f} {unit}, limit {limit}: '
                  f'{verdict(abs(error) <= limit)}', flush = True)
        passed = sum(item['passes'] for item in record['overturn'].values())
        print(f'{ppd} ppd: {passed} of 5 overturn criteria pass; touchdown at {summary["touchdownTime"]:.3f} s, '
              f'{summary["wall"]:.0f} s', flush = True)
    else:
        print(f'{ppd} ppd: no touchdown read; the overturn criteria fail', flush = True)
    writeResult(f'surfRanchOverturn_{ppd}' + common.diffusionSuffix(densityDiffusion), record)
    return record

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Gate 1: the Surf Ranch overturn at the jet\'s landing.')
    parser.add_argument('ppd', type = int, help = 'particles per depth')
    common.addDiffusionArgument(parser)
    arguments = parser.parse_args()
    main(arguments.ppd, arguments.densityDiffusion)
