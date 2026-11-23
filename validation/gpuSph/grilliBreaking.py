# -- Gate 1: the breaking point on Grilli's 1:35 slope -- #

'''

A solitary wave of H/h0 = 0.3 on a 1:35 plane beach, `BreakingWave` on the 'deltaSph' scheme from
bed-fitted columns, run until its breaking point is final at the jet's landing.

Reference: Grilli, Svendsen and Subramanya (1997), fully nonlinear potential flow, Hb/h0 = 0.476
and Hb/hb = 1.380 (hb/h0 = 0.345), as tabulated in Table 8 of Senthilkumar (2016,
arXiv:1601.06822).

Criteria, set before the runs: Hb/h0 within 10 percent, hb/h0 within 15 percent. The breaking point
is the start of the front face's last vertical run before the jet lands (decision 14 of the GPU SPH
roadmap).

Recorded (2026-11-12): Hb/h0 0.486, 0.496 and 0.484 (+2.1, +4.3, +1.6 percent) and hb/h0 0.278,
0.316 and 0.335 (-19.5, -8.4, -2.8 percent) at 32, 64 and 128 particles per depth: passes at 64 and
128, fails at 32 on the depth. 64 particles per depth runs in about 11 minutes on an RTX 2070 Super
(128,947 particles).

With --sortInterval n the run reorders the fluid's state in the GPU hash grid's cell order after every
n-th breaking check (`BreakingWave.sortInterval`); at 64 particles per depth a check is about ten steps.
Criteria for a sorted run, set before it: the breaking time, Hb/h0 and hb/h0 within 1 percent of the
unsorted run's, and Grilli's criteria. Recorded (2026-11-23) at 64 particles per depth with
--sortInterval 10: all three within 2e-6 of the unsorted run (27.540 sqrt(h0/g), 0.4963, 0.3159):
passes. After breaking the two runs part as rounding changes make them: the jet lands 0.07 percent
apart.

    python grilliBreaking.py <ppd> [--sortInterval <n>]

Writes results/grilliBreaking_<ppd>[_sorted].json.

Sean Bowman [11/19/2026]

'''

import argparse
import time

import common
from common import percent, verdict, writeResult

from pysurf.BreakingWave import BreakingWave

grilliHeight, grilliIndex = 0.476, 1.380
grilliDepth = grilliHeight / grilliIndex

def main(ppd: int, sortInterval: int = 0) -> dict:

    '''Run the case at `ppd` particles per depth and report it against the criteria.'''

    started = time.perf_counter()
    wave = BreakingWave(depth = 1.0, waveHeight = 0.3, beachSlope = 1.0 / 35.0, particlesPerDepth = ppd,
                        scheme = 'deltaSph', backend = 'gpu', frameInterval = 0.05 * max(1.0, ppd / 32.0),
                        sortInterval = sortInterval)
    # The breaking point is final at the jet's landing, where the run stops
    wave.run(progress = True, stopAt = 'breaking')
    summary = wave.summary()
    summary['wall'] = time.perf_counter() - started
    record = {'case': 'grilliBreaking', 'particlesPerDepth': ppd, 'sortInterval': sortInterval, 'summary': summary,
              'reference': {'breakingHeightOverDepth': grilliHeight, 'breakingDepthOverDepth': grilliDepth}}
    if summary.get('broke'):
        heightError = percent(summary['breakingHeightOverDepth'], grilliHeight)
        depthError = percent(summary['breakingDepthOverDepth'], grilliDepth)
        passes = abs(heightError) <= 10.0 and abs(depthError) <= 15.0
        record['verdict'] = {'heightError': heightError, 'depthError': depthError, 'passes': passes}
        print(f'{ppd} ppd: Hb/h0 {summary["breakingHeightOverDepth"]:.3f} ({heightError:+.1f} percent, limit 10), '
              f'hb/h0 {summary["breakingDepthOverDepth"]:.3f} ({depthError:+.1f} percent, limit 15) at '
              f'{summary["breakingTime"] / common.timeScale:.3f} sqrt(h0/g): {verdict(passes)}; '
              f'{summary["fluidParticles"]} particles, {summary["wall"]:.0f} s', flush = True)
    else:
        record['verdict'] = {'passes': False}
        print(f'{ppd} ppd: no breaking point read; fails', flush = True)
    writeResult(f'grilliBreaking_{ppd}' + ('_sorted' if sortInterval else ''), record)
    return record

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Gate 1: the breaking point on Grilli's 1:35 slope.")
    parser.add_argument('ppd', type = int, help = 'particles per depth')
    parser.add_argument('--sortInterval', type = int, default = 0, help = 'breaking checks between sorts; 0 for none')
    arguments = parser.parse_args()
    main(arguments.ppd, arguments.sortInterval)
