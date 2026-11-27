# -- Where and when whitewater starts on the Surf Ranch run, against the breaking tracker -- #

'''

Ihmsen et al.'s (2012) diffuse particles (tools/rendering/whitewater.py) run over the saved frames
of the Surf Ranch shoal run (the reef turned 25 degrees, 32 particles per depth with the crest zone,
an exact solitary wave of H/h0 = 0.6), with the constants set before any run: the trapped-air and
wave-crest thresholds as Ihmsen et al. publish them, the kinetic threshold per kilogram, neighbor
sums over 2 spacings weighted by volume. Each frame interval spawns from that frame's fluid inside
the box x 38 to 47, y 0 to 4, z 0.4 to 2.6 h0, and the spawns are counted in sections of the crest
0.5 h0 wide centered on the breaking tracker's sections y = 0.5 to 3.5 h0.

Reading, set 2026-11-26 before any run (roadmap item 5c): in each of the seven sections, (1)
whitewater starts, the first frame whose spawn count reaches 1 percent of the section's largest
per-frame count, no earlier than 0.1 t' before the tracker's breaking time and no later than its
touchdown time; (2) under 5 percent of the section's spawns over the window fall before its
breaking time. The wall sections at y = 0 and 4 are left out; the tracker itself triggers there
falsely at t' 12. A failure is reported as read; the thresholds are not moved to meet it. The
spawn rates, lifetimes and radii are a look; this reading tests only where and when spawning
starts.

    python whitewaterOnset.py [<run folder>]

Writes results/whitewaterOnset.json.

Sean Bowman [11/26/2026]

'''

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

repoRoot = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repoRoot / 'tools' / 'rendering'))

from whitewater import Whitewater, WhitewaterSettings, bubble, foam, spray

timeScale = math.sqrt(1.0 / 9.80665)                 # [s], sqrt(h0 / g) with h0 = 1 m
spacing = 1.0 / 32
box = (38.0, 47.0, 0.0, 4.0, 0.4, 2.6)
sectionCentres = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
first = 19.5                                         # [t']

def main(folder: Path) -> dict:
    summary = json.loads((folder / 'summary.json').read_text())
    sections = {centre: summary['sections'][f'{centre}'] for centre in sectionCentres}
    paths = sorted(folder.glob('frame_*.npz'))
    times = np.array([float(np.load(path)['time']) for path in paths]) / timeScale
    chosen = [i for i in range(len(paths)) if times[i] >= first - 1e-6]
    whitewater = Whitewater(WhitewaterSettings(spacing = spacing), box)
    frameTimes, counts, census, drivers = [], [], [], []
    began = time.perf_counter()
    for k, index in enumerate(chosen[:-1]):
        data = np.load(paths[index])
        positions = data['positions'].astype(np.float32)
        velocities = data['velocities'].astype(np.float32)
        volumes = np.where(data['split'].astype(bool), spacing**3 / 8.0, spacing**3)
        dt = (times[chosen[k + 1]] - times[index]) * timeScale                          # [s]
        result = whitewater.step(positions, velocities, volumes, dt)
        y = result['spawned'][:, 1]
        counts.append([int(np.sum(np.abs(y - centre) < 0.25)) for centre in sectionCentres])
        frameTimes.append(float(times[index]))
        census.append({name: int(np.sum(whitewater.kinds == kind)) for name, kind in
                       (('spray', spray), ('foam', foam), ('bubbles', bubble))})
        measures = result['measures']
        inside = measures['inside']
        drivers.append({'spawning': int(np.sum(measures['spawnCount'][inside] > 0)),
                        'trappedAir': int(np.sum(measures['airPotential'][inside] > 0)),
                        'waveCrest': int(np.sum(measures['crestPotential'][inside] > 0)),
                        'energy': int(np.sum(measures['energyPotential'][inside] > 0))})
        if k % 10 == 0:
            print(f"t' {times[index]:.3f}: spawned {len(y)}, diffuse {len(whitewater.kinds)} {census[-1]}, "
                  f'{drivers[-1]}, {time.perf_counter() - began:.0f} s', flush = True)
    counts = np.array(counts)
    frameTimes = np.array(frameTimes)
    readings = []
    for column, centre in enumerate(sectionCentres):
        breaking = sections[centre]['breakingTime'] / timeScale
        touchdown = sections[centre]['touchdownTime'] / timeScale
        series = counts[:, column]
        peak = series.max()
        onset = float(frameTimes[np.argmax(series >= 0.01 * peak)]) if peak > 0 else None
        early = float(series[frameTimes < breaking].sum() / max(series.sum(), 1))
        startsInWindow = onset is not None and breaking - 0.1 <= onset <= touchdown
        readings.append({'y': centre, 'breaking': breaking, 'touchdown': touchdown, 'onset': onset,
                         'peakPerFrame': int(peak), 'spawns': int(series.sum()), 'shareBeforeBreaking': early,
                         'startPasses': bool(startsInWindow), 'sharePasses': bool(early < 0.05)})
        print(f"y {centre:.1f}: breaking t' {breaking:.3f}, touchdown {touchdown:.3f}, onset "
              f"{onset if onset is None else round(onset, 3)}, before breaking {100 * early:.2f} %, "
              f"start {'passes' if startsInWindow else 'FAILS'}, share {'passes' if early < 0.05 else 'FAILS'}",
              flush = True)
    record = {'run': str(folder.name), 'box': box, 'spacing': spacing, 'settings': vars(WhitewaterSettings(spacing = spacing)),
              'firstFrame': first, 'sections': readings,
              'passes': all(r['startPasses'] and r['sharePasses'] for r in readings),
              'frames': {'time': frameTimes.tolist(), 'spawnsBySection': counts.tolist(), 'census': census,
                         'drivers': drivers}}
    out = Path(__file__).resolve().parent / 'results' / 'whitewaterOnset.json'
    out.parent.mkdir(exist_ok = True)
    out.write_text(json.dumps(record, indent = 1), newline = '\n')
    print(f"written {out}: {'PASSES' if record['passes'] else 'FAILS'}; RUN-END", flush = True)

    return record

if __name__ == '__main__':
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r'D:\pySurfRuns\surfRanch32'))
