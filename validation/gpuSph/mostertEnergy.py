# -- Gate 1: the energy lost in active breaking on Mostert and Deike's beach -- #

'''

Mostert and Deike's (2020) plunging case: a solitary wave of H0/h0 = 0.3 on a 3 degree plane beach,
`BreakingWave` on the 'deltaSph' scheme from bed-fitted columns, from start to end. The same tank
from still water runs first, and its energy drift comes out of the wave's
(`BreakingWave.breakingDissipation(reference = still)`). Active breaking runs from the breaking point
to the kinetic energy falling to the potential, as Mostert and Deike define it.

Reference: their two-phase DNS loses 0.31 of the wave's energy at breaking over active breaking;
across slopes of 2 to 5 degrees their dissipation rate collapses onto the inertial fit
(epsilon / epsilon0)(db / d0) = 0.0281 (hb / d0)^3.152.

Criterion, set before the runs: the share lost within 30 percent of 0.31. The scaled rate against
the fit is reported without a rule.

With --dualSphysicsSettings the run takes the settings DualSPHysics runs this case with: c0 = 20
sqrt(g h0), negative pressure allowed, no viscosity limiter, the artificial viscosity applied
against the bed at rest, h = 1.5 spacings. A diagnostic of where the deficit lies; the same
criterion applies.

Recorded: with pySurf's settings (2026-11-12) the share lost is 0.20 and 0.17 at 32 and 64
particles per depth, 35 and 47 percent short: fails, further off at the finer spacing. With
DualSPHysics's settings (2026-11-19) it is 0.607 and 0.545 at 32 and 64, 96 and 76 percent over:
fails the other way, and the wave loses about 1 percent of its energy per h0 travelled before it
breaks. The model has no turbulence; its artificial viscosity and density diffusion stand in for
the loss.

With --energyBudget both runs keep the solver's energy budget (`pysurf.sphEnergy`), and the run
reports over active breaking what each part of the scheme took from the fluid's energy, the still
reference's taken out (`BreakingWave.breakingBudget`). Reading, set before the runs (2026-11-22): if
the walls or the time stepping take more than a tenth of the mechanical energy lost, they are part
of the deficit, and that is looked at before a turbulence closure or an air phase.

Recorded with the budget (2026-11-23), the share lost unchanged at 0.202 and 0.165: of the energy
lost in active breaking the density diffusion takes 0.60 and 0.74 at 32 and 64 particles per depth,
the limited viscosity 0.17 and 0.26, the walls 0.38 and 0.14 (the clamp at the bed 0.08 and 0.07 of
it), and the continuity correction gives back 0.17; time stepping 0.02 to 0.03, the closure under
1e-5 J/m. By the reading the walls are part of the deficit at both. A 64 run whose device kernels
differed only in rounding read a share of 0.156 and the walls at 0.085: at 64 the walls' share sits
near the line.

With --viscosity the run takes another viscosity (`common.viscosityForms`): none, the delta-LES-SPH
closure alone, or the closure with the scheme's limited viscosity; the still reference takes the same.
Recorded at 64 particles per depth with Antuono's term and the budget (2026-11-27): the closure alone
loses 0.156 (the scheme with the same term 0.155), the closure with the scheme's viscosity 0.182;
both fail. The closure moves the loss from the diffusion to the viscosity (0.66 and 0.37 of it,
against the scheme's 0.27 and 0.66) and leaves the energy lost nearly where it was, 427 against
429 J/m.

    python mostertEnergy.py <ppd> [--dualSphysicsSettings] [--energyBudget] [--densityDiffusion antuono]
                                  [--viscosity closure]

Writes results/mostertEnergy_<ppd>[_dualSphysics][_budget].json, with endings for a diffusive term or
a viscosity other than the scheme's.

Sean Bowman [11/19/2026]

'''

import argparse
import math
import time

import numpy as np

import common
from common import percent, verdict, writeResult

from pysurf import constants, waveTheory
from pysurf.BreakingWave import BreakingWave
from pysurf.errors import NumericalInstabilityError

mostertShareLost = 0.31
waveHeight = 0.3
budgetShareLimit = 0.1          # [-] of the mechanical energy lost, above which the walls or the time stepping read as part of the deficit
dualSphysics = {'soundSpeed': 20.0, 'smoothingLengthRatio': 1.5,
                'solverOptions': {'negativePressure': True, 'viscosityLimiter': False, 'boundaryViscosityFactor': 1.0}}

def makeWave(ppd: int, startingWave: str, settings: dict | None, energyBudget: bool = False,
             densityDiffusion: str = 'hydrostatic', viscosity: str = 'scheme') -> BreakingWave:

    '''The beach run or its still-water reference, with pySurf's settings or `settings`, keeping the energy budget if asked.'''

    options, alpha = common.viscositySettings(viscosity)
    wave = BreakingWave(depth = 1.0, waveHeight = waveHeight, beachSlope = math.tan(math.radians(3.0)),
                        particlesPerDepth = ppd, scheme = 'deltaSph', backend = 'gpu', startingWave = startingWave,
                        solverOptions = dict(settings['solverOptions'] if settings else {}, **options,
                                             densityDiffusion = densityDiffusion),
                        alphaViscosity = alpha,
                        smoothingLengthRatio = settings['smoothingLengthRatio'] if settings else None,
                        frameInterval = 0.1 * common.timeScale, energyBudget = energyBudget)
    if settings:
        # The scheme sets c0 as a multiple of the wave's speed; this sets it in sqrt(g h0)
        wave.soundSpeedFactor = (settings['soundSpeed'] * math.sqrt(constants.gravity)
                                 / waveTheory.solitaryWaveSpeed(waveHeight, 1.0, constants.gravity))
    return wave

def reportBudget(wave: BreakingWave, still: BreakingWave) -> dict:

    '''The energy budget over active breaking, printed against the reading.'''

    budget = wave.breakingBudget(reference = still)
    shares = budget['shareOfLoss']
    partOfDeficit = [name for name in ('walls', 'timeStepping') if abs(shares[name]) > budgetShareLimit]
    budget['reading'] = {'limit': budgetShareLimit, 'partOfDeficit': partOfDeficit}
    lost = -budget['mechanicalChange']
    print(f'  energy budget over active breaking, {lost:.2f} J/m of mechanical energy lost '
          f'(closure {budget["closure"]:+.2e} J/m):', flush = True)
    for name, share in shares.items():
        print(f'    {name:13s} {share:+.4f} of the loss', flush = True)
    print('  reading: ' + (', '.join(partOfDeficit) + ' part of the deficit' if partOfDeficit
                           else 'neither the walls nor the time stepping take a tenth of the loss'), flush = True)

    return budget

def main(ppd: int, useDualSphysics: bool, energyBudget: bool = False, densityDiffusion: str = 'hydrostatic',
         viscosity: str = 'scheme') -> dict:

    '''Run the case at `ppd` particles per depth and report it against the criterion.'''

    settings = dualSphysics if useDualSphysics else None
    started = time.perf_counter()
    still = makeWave(ppd, 'still', settings, energyBudget, densityDiffusion, viscosity)
    still.run(progress = False)
    stillPotential = np.array([frame.potentialEnergy for frame in still.frames])
    print(f'{ppd} ppd still water: potential energy change {stillPotential[-1] - stillPotential[0]:+.1f} J/m over the run, '
          f'{time.perf_counter() - started:.0f} s', flush = True)
    wave = makeWave(ppd, 'exact', settings, energyBudget, densityDiffusion, viscosity)
    failure = None
    try:
        wave.run(progress = True)
    except NumericalInstabilityError as error:
        # Keep the frames up to the failure and measure what they hold
        failure = repr(error)
        print(f'{ppd} ppd stopped: {failure}', flush = True)
    summary = wave.summary()
    summary.update(failure = failure, wall = time.perf_counter() - started)
    record = {'case': 'mostertEnergy', 'particlesPerDepth': ppd, 'settings': 'dualSphysics' if settings else 'pySurf',
              'densityDiffusion': densityDiffusion, 'viscosity': viscosity,
              'summary': summary, 'reference': {'shareLost': mostertShareLost}}
    try:
        dissipation = wave.breakingDissipation(reference = still)
    except ValueError as error:
        dissipation = {'error': repr(error)}
    record['dissipation'] = dissipation
    if 'energyShareLost' in dissipation:
        shareError = percent(dissipation['energyShareLost'], mostertShareLost)
        rateError = percent(dissipation['scaledDissipation'], dissipation['scaledDissipationFit'])
        passes = abs(shareError) <= 30.0
        record['verdict'] = {'shareError': shareError, 'rateError': rateError, 'passes': passes}
        print(f'{ppd} ppd, {record["settings"]} settings: share lost {dissipation["energyShareLost"]:.4f} '
              f'({shareError:+.1f} percent against 0.31, limit 30): {verdict(passes)}; scaled rate {rateError:+.1f} percent '
              f'against the fit; Hb/h0 {dissipation["breakingHeightOverDepth"]:.3f}, energy at breaking '
              f'{dissipation["breakingEnergy"] / dissipation["startEnergy"]:.3f} of the start; {summary["wall"]:.0f} s',
              flush = True)
    else:
        record['verdict'] = {'passes': False}
        print(f'{ppd} ppd: active breaking not measured ({dissipation.get("error")}); fails', flush = True)
    if energyBudget and 'energyShareLost' in dissipation:
        record['budget'] = reportBudget(wave, still)
    writeResult(f'mostertEnergy_{ppd}' + ('_dualSphysics' if settings else '') + ('_budget' if energyBudget else '')
                + common.diffusionSuffix(densityDiffusion) + common.viscositySuffix(viscosity), record)
    return record

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Gate 1: the energy lost in active breaking on Mostert and Deike's beach.")
    parser.add_argument('ppd', type = int, help = 'particles per depth')
    parser.add_argument('--dualSphysicsSettings', action = 'store_true', help = "run with DualSPHysics's settings")
    parser.add_argument('--energyBudget', action = 'store_true', help = 'keep and report the energy budget')
    common.addDiffusionArgument(parser)
    common.addViscosityArgument(parser)
    arguments = parser.parse_args()
    main(arguments.ppd, arguments.dualSphysicsSettings, arguments.energyBudget, arguments.densityDiffusion,
         arguments.viscosity)
