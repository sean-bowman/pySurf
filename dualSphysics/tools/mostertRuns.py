
# -- DualSPHysics on Mostert and Deike's Energy Loss (2D) -- #

'''

Runs Mostert and Deike's (2020) plunging case, a solitary wave of H0/h0 = 0.3 on a 3 degree plane
beach, in DualSPHysics, and measures the energy lost in active breaking as pySurf measures it
(`BreakingWave.breakingDissipation`; phase 1.5 of planning/gpuSphRoadmap.md). The criterion: the
share of the wave's energy lost between the face turning vertical and the kinetic energy falling to
the potential, within 30 percent of their 0.31.

Three runs per resolution and viscosity, in WSL under ~/dsphRuns/mostert<ppd><viscosity>:

    tune        as `gateRuns.tune`: the paddle's target height scaled until the gauge at pySurf's
                starting crest position reads 0.3 h0 within 2 percent
    wave        the tuned paddle; Part files every 0.1 sqrt(h0/g), pySurf's frame interval for
                this case, and every 0.01 sqrt(h0/g) over the tracking window
    still       the same tank with the paddle held still (`SolitaryCase.paddleMoves` False), Part
                files every 0.1 sqrt(h0/g)

The energies come from the fluid particles: kinetic sum(m |v|^2) / 2 and potential sum(m g z), with
m = rho0 dp^2. The still run settles as the wave run does (the lattice over the slope, the water
compressing under its own weight); its kinetic energy and the change in its potential energy are
taken out, as `BreakingWave.energyHistory(reference = ...)` takes out pySurf's. The remaining
potential energy differs from the wave's, rho g int(eta^2 / 2) dx, by the water the piston
displaced, raised to still water level: with the piston's face moved S, the excess volume is S h0,
and the difference rho g S h0^2 / 2 is taken out with S read from the piston's particles. The face
turning vertical comes from `BreakingTracker`, set up as `BreakingWave.build` sets it and fed the
fluid of each Part file in the window. Times count from the crest's passage of the gauge.

    python mostertRuns.py <ppd> <artificial|sps> [paddle height to start from] [--measure]

--measure reads the runs already made and measures them again. Results in
pySurf/temp/phase1_5/mostert<ppd><viscosity>.json. These are cross-checks: none validates pySurf.

Sean Bowman [11/09/2026]

'''

# Standard library imports
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

# Third-party imports
import numpy as np
from scipy.integrate import trapezoid

tools = Path(__file__).resolve().parent
sys.path.insert(0, str(tools.parents[1] / 'src'))
sys.path.insert(0, str(tools))

# Local imports
import gateRuns
from dsphCase import SolitaryCase
from dsphOutput import partTimes, readPartArrays
from pysurf import constants, waveTheory
from pysurf.waveBreaking import BreakingTracker

mostertShareLost = 0.31
frameStep = 0.1 * math.sqrt(1.0 / constants.gravity)

def mostertCase(ppd: int, viscosity: str, paddleHeight: float, endTime: float, window: tuple | None,
                coarseStep: float, paddleMoves: bool = True) -> SolitaryCase:

    '''The DualSPHysics case for one paddle height, viscosity and paddle state.'''

    case = gateRuns.solitaryCase('mostert', ppd, paddleHeight, endTime, window, coarseStep)
    case.viscosity = viscosity
    case.paddleMoves = paddleMoves
    return case

def movingIds(runName: str) -> tuple[int, int]:

    '''The piston's particle ids (first, last), from the run's GenCase log.'''

    log = Path(f'{gateRuns.windowsRuns}/{runName}/gencase.log').read_text(encoding = 'utf-8', errors = 'replace')
    found = re.search(r'Moving\.+:\s*[\d,]+\s+id:\((\d+)-(\d+)\)', log)
    return int(found.group(1)), int(found.group(2))

def energies(runName: str, spacing: float, passage: float, tracker: BreakingTracker | None = None,
             window: tuple | None = None, startX: float = 0.0) -> dict:

    '''

    Kinetic and potential energy of the fluid [J/m], the fluid particle count, and the piston's
    displacement [m] at every Part file of a run, times shifted by `passage`. Each fluid particle
    carries rho0 dp^2 (the case's rhop0, 1000 kg/m^3). With a tracker, the fluid of each frame
    inside `window` (shifted time) is fed to it until it reads breaking.

    '''

    nBound = gateRuns.boundaryCount(runName)
    first, last = movingIds(runName)
    outDir = f'{gateRuns.windowsRuns}/{runName}/out'
    times = partTimes(outDir)
    g, mass = constants.gravity, 1000.0 * spacing**2
    rows, pistonStart = [], None
    for n in sorted(times):
        arrays = readPartArrays(f'{outDir}/data/Part_{n:04d}.bi4')
        idp, position, velocity = arrays['Idp'], arrays['Pos'].astype(float), arrays['Vel'].astype(float)
        fluid = idp >= nBound
        piston = (idp >= first) & (idp <= last)
        pistonX = float(position[piston, 0].mean())
        pistonStart = pistonX if pistonStart is None else pistonStart
        now = times[n] - passage
        rows.append((now, 0.5 * mass * float(np.sum(velocity[fluid] ** 2)), mass * g * float(np.sum(position[fluid, 2])),
                     int(fluid.sum()), pistonX - pistonStart))
        if tracker is not None and not tracker.breakingFinal and window[0] - 1e-6 <= now <= window[1] + 1e-6:
            xz = position[fluid][:, [0, 2]]
            tracker.update(now, xz[xz[:, 0] >= startX])
    t, kinetic, potential, count, stroke = (np.array(column) for column in zip(*rows))
    return {'time': t, 'kinetic': kinetic, 'potential': potential, 'count': count, 'stroke': stroke, 'mass': mass}

def waveEnergy(wave: dict, still: dict, depth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    '''The wave's kinetic energy and potential energy over still water [J/m]: the still run's taken out, and the piston's displaced water.'''

    t = wave['time']
    kinetic = wave['kinetic'] - np.interp(t, still['time'], still['kinetic'])
    rho, g = constants.freshwaterDensity, constants.gravity
    potential = (wave['potential'] - np.interp(t, still['time'], still['potential'])
                 - rho * g * wave['stroke'] * depth**2 / 2.0)
    return t, kinetic, potential

def dissipation(t: np.ndarray, kinetic: np.ndarray, potential: np.ndarray, point, depth: float) -> dict:

    '''The energy lost in active breaking, as `BreakingWave.breakingDissipation` measures it.'''

    energy = kinetic + potential
    d, g, rho = depth, constants.gravity, constants.freshwaterDensity
    breakingEnergy = float(np.interp(point.time, t, energy))
    surplus = kinetic - potential
    after = np.flatnonzero((t > point.time) & (surplus <= 0.0))
    reached = after.size > 0
    if reached:
        k = int(after[0])
        share = surplus[k - 1] / (surplus[k - 1] - surplus[k])
        endTime = float(t[k - 1] + share * (t[k] - t[k - 1]))
        endEnergy = float(energy[k - 1] + share * (energy[k] - energy[k - 1]))
    else:
        endTime, endEnergy = float(t[-1]), float(energy[-1])
    rate = (breakingEnergy - endEnergy) / (endTime - point.time)
    return {
        'breakingTime': point.time,
        'activeEndTime': endTime,
        'activeDuration': endTime - point.time,
        'activeDurationOverScale': (endTime - point.time) / math.sqrt(d / g),
        'breakingEnergy': breakingEnergy,
        'activeEndEnergy': endEnergy,
        'energyShareLost': (breakingEnergy - endEnergy) / breakingEnergy,
        'dissipationRate': rate,
        'scaledDissipation': rate / (rho * g**1.5 * d**2.5) * (point.faceDepth / d),
        'scaledDissipationFit': 0.0281 * (point.crestHeight / d)**3.152,
        'breakingHeightOverDepth': point.crestHeight / d,
        'breakingFaceDepthOverDepth': point.faceDepth / d,
        'activeEndReached': reached,
    }

def main(ppd: int, viscosity: str, start: float | None, measureOnly: bool = False) -> None:

    '''Tune, run the wave and the still tank, and measure the energy lost in active breaking.'''

    gateRuns.results.mkdir(parents = True, exist_ok = True)
    spec = gateRuns.setup('mostert', ppd)
    wave, bed, window = spec['wave'], spec['bed'], spec['window']
    d, target = wave.depth, wave.waveHeight
    runName = f'mostert{ppd}{viscosity}'
    resultPath = gateRuns.results / f'{runName}.json'
    if measureOnly:
        saved = json.loads(resultPath.read_text(encoding = 'utf-8'))
        tuned, walls = saved['tuning'], saved['walls']
    else:
        print(f'mostert {ppd} ppd, {viscosity} viscosity: tuning the paddle', flush = True)
        tuned = gateRuns.tune('mostert', ppd, start, prefix = runName,
                              build = lambda height: mostertCase(ppd, viscosity, height, 1.0, None, 1.0))
        endTime = tuned['passage'] + 16.0
        fine = (tuned['passage'] + window[0], tuned['passage'] + window[1])
        walls = {'wave': gateRuns.runCase(runName, mostertCase(ppd, viscosity, tuned['paddleHeight'], endTime, fine, frameStep)),
                 'still': gateRuns.runCase(f'{runName}_still', mostertCase(ppd, viscosity, tuned['paddleHeight'], endTime, None,
                                                                            frameStep, paddleMoves = False))}
        print(f'{runName}: wave run {walls["wave"]:.0f} s, still run {walls["still"]:.0f} s', flush = True)
    passage, height = gateRuns.gaugeCrest(runName, d, target)
    tracker = BreakingTracker(bed, wave.spacing, faceReach = d, searchFromX = wave.toeX, faceMeasure = 'contour')
    waveRun = energies(runName, wave.spacing, passage, tracker, window, bed.startX)
    stillRun = energies(f'{runName}_still', wave.spacing, passage)
    t, kinetic, potential = waveEnergy(waveRun, stillRun, d)

    # The accounting at the gauge: the potential energy against the exact wave's of the gauge's height
    x = np.linspace(-30.0, 30.0, 12001)
    exactPotential = constants.freshwaterDensity * constants.gravity * trapezoid(0.5 * waveTheory.exactSolitarySurface(x, height, d)[0]**2, x)
    summary = {'particlesPerDepth': ppd, 'viscosity': viscosity, 'tuning': tuned, 'walls': walls,
               'gaugeHeight': height, 'passage': passage,
               'atPassage': {'kinetic': float(np.interp(0.0, t, kinetic)), 'potential': float(np.interp(0.0, t, potential)),
                             'exactPotential': float(exactPotential)},
               'fluidParticles': int(waveRun['count'][0]), 'fluidParticlesLeast': int(waveRun['count'].min()),
               'stillFluidParticlesLeast': int(stillRun['count'].min()), 'stroke': float(waveRun['stroke'][-1]),
               'stillStroke': float(np.abs(stillRun['stroke']).max()), 'particleMass': waveRun['mass']}
    point = tracker.breakingPoint
    if point is None or not np.isfinite(point.faceDepth):
        summary['error'] = f'no breaking point read from the smoothed surface between {window[0]} and {window[1]} s'
        print(f'{runName}: {summary["error"]}', flush = True)
    else:
        summary.update(dissipation(t, kinetic, potential, point, d))
        summary['passes'] = abs(summary['energyShareLost'] / mostertShareLost - 1.0) <= 0.30
        print(f'{runName}: share lost {summary["energyShareLost"]:.4f} (Mostert {mostertShareLost}, '
              f'{100 * (summary["energyShareLost"] / mostertShareLost - 1):+.1f} percent), scaled rate '
              f'{summary["scaledDissipation"]:.5f} against the fit {summary["scaledDissipationFit"]:.5f} '
              f'({100 * (summary["scaledDissipation"] / summary["scaledDissipationFit"] - 1):+.1f} percent), '
              f'active {summary["activeDurationOverScale"]:.2f} sqrt(d/g), end reached {summary["activeEndReached"]}, '
              f'Hb/d {summary["breakingHeightOverDepth"]:.3f}, db/d {summary["breakingFaceDepthOverDepth"]:.3f}, '
              f'breaks at {point.time:.3f} s; {"passes" if summary["passes"] else "fails"}', flush = True)
    at = summary['atPassage']
    print(f'{runName}: at the gauge, H {height:.4f} m, kinetic {at["kinetic"]:.0f} J/m, potential {at["potential"]:.0f} '
          f'(exact wave {at["exactPotential"]:.0f}); stroke {summary["stroke"]:.4f} m; fluid {summary["fluidParticles"]}, '
          f'least {summary["fluidParticlesLeast"]}', flush = True)
    summary['history'] = {'time': t, 'kinetic': kinetic, 'potential': potential}
    resultPath.write_text(json.dumps(summary, indent = 1, default = lambda v: v.tolist() if hasattr(v, 'tolist') else float(v)),
                          encoding = 'utf-8')
    print('RUN-END', flush = True)

if __name__ == '__main__':
    arguments = [a for a in sys.argv[1:] if a != '--measure']
    if len(arguments) < 2 or arguments[1] not in ('artificial', 'sps'):
        sys.exit('python mostertRuns.py <ppd> <artificial|sps> [paddle height to start from] [--measure]')
    main(int(arguments[0]), arguments[1], float(arguments[2]) if len(arguments) > 2 else None, '--measure' in sys.argv)
