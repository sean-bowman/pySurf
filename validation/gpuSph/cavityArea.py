# -- The tube at touchdown on the turned Surf Ranch reef, against measured overturns: a cross-check -- #

'''

The cavity a plunging jet encloses at touchdown, over the square of the crest-to-trough breaking
height, A_c / H_b², read from a 3D run that saved its whole fluid (temp/surfRanchFull: the Surf
Ranch reef turned 25 degrees, 32 particles per depth with the crest zone, an exact solitary wave of
H/h0 = 0.6, the shelf 0.371 h0 deep). Each section's breaking point, crest height and cavity come
from the run's summary.json; the trough ahead of the crest is read from the saved frame nearest the
breaking time, from the same column profile the tracker reads (`waveBreaking.troughAhead`), so
H_b = crest less trough. No rerun is needed: the frames from t' 19.5 on hold every inner section's
breaking (t' 20.3 to 21.2).

Reference: Blenkinsopp and Chaplin (2008), figure 7 (reference/blenkinsopp2008Fig7.json), cavities
at the plunge point of regular waves over a 1:10 reef, A_c from the cavity's measured length and
width (errors near 20 percent), against h_c / H_o, the depth over the reef crest over the deepwater
wave height, 0.72 to 1.72; Kiger and Duncan's (2012) pooled range, 0.029 to 0.41, beside it.

Reading, set 2026-11-23 before any value here was read: the shelf's h_c / H_o = 0.371 / 0.6 = 0.62
lies below their range; their cavity grows as h_c / H_o falls and as the steepness falls, and a
solitary wave is the long-wave end. A section's A_c / H_b² within their nine points at h_c / H_o of
0.9 or less (0.154 to 0.408) is consistent; above it reads as lying the way their trend runs, below
it as against it. Their waves are periodic and they credit part of the trend to the return flow
from the previous wave, which a solitary wave lacks; a cross-check, not validation. The inner
sections only, y = 0.5 to 3.5 h0: the wall sections' breaking points are false triggers.

Recorded (2026-11-23): all seven inner sections consistent, A_c / H_b² from 0.233 to 0.258 (A_c 0.111
to 0.125 h0², H_b 0.673 to 0.699 h0, the trough ahead at still water within 0.002 h0), inside the band
and the pooled range. H_b / h_c is 1.81 to 1.88, against 0.8 to 1.35 in their figure 4(a): the
solitary wave breaks far higher over the shelf than their regular waves do, another way the
conditions differ. The trough's reading was settled on this run: a plain column minimum caught
columns that hold part of a particle column, where the bed-fitted columns straddle the profile's
bins and against the end wall 10 h0 ahead, at 0.05 to 0.12 h0 below still water, and gave A_c / H_b²
of 0.172 to 0.225, consistent too; the reading now takes a running median over three columns within
twice the face reach (`waveBreaking.troughAhead`).

    python cavityArea.py [<run folder>]

Writes results/cavityArea.json.

Sean Bowman [11/23/2026]

'''

import argparse
import json
from pathlib import Path

import numpy as np

from common import readReference, writeResult

from pysurf.waveBreaking import freeSurfaceProfile, troughAhead

shelfDepth, incidentHeight = 0.371, 0.6          # [h0], the run's shelf depth and wave height
innerSections = tuple(0.5 * k for k in range(1, 8))
sectionRows = 2                                   # rows a section takes, as `BreakingWave3D` reads it
faceReach = 1.0                                   # [h0], the section trackers' face reach; the trough is read within twice it
bandRelativeSubmergence = 0.9                     # [-], h_c / H_o of the reference points the band takes
pooledRange = (0.029, 0.41)                       # Kiger and Duncan (2012)

def frameTimes(folder: Path) -> list:

    '''Every saved frame's time [s] and path, in order.'''

    frames = []
    for path in sorted(folder.glob('frame_*.npz')):
        with np.load(path) as data:
            frames.append((float(data['time']), path))
    return frames

def trough(frame: Path, y: float, crestX: float, region: list, spacing: float) -> tuple:

    '''The trough ahead of a section's crest, and the profile's crest near it, from one saved frame [h0].'''

    with np.load(frame) as data:
        positions = data['positions'].astype(float)
    inSection = np.abs(positions[:, 1] - y) < 0.5 * sectionRows * spacing
    plane = positions[inSection][:, [0, 2]]
    centers, elevation = freeSurfaceProfile(plane, region[0], region[1], spacing)
    near = np.abs(centers - crestX) < 0.5
    return troughAhead(centers, elevation, crestX, 2.0 * faceReach) - 1.0, float(np.nanmax(elevation[near])) - 1.0

def main(folder: Path) -> dict:

    '''Read each inner section's A_c / H_b² and set it against the band.'''

    summary = json.loads((folder / 'summary.json').read_text(encoding = 'utf-8'))
    spacing = 1.0 / summary['particlesPerDepth']
    frames = frameTimes(folder)
    times = np.array([time for time, _ in frames])
    reference = np.array(readReference('blenkinsopp2008Fig7.json')['points'])
    band = reference[reference[:, 0] <= bandRelativeSubmergence, 1]
    low, high = float(band.min()), float(band.max())
    print(f'band: {len(band)} points at h_c/H_o <= {bandRelativeSubmergence}, A_c/H_b^2 {low:.3f} to {high:.3f}; '
          f'the shelf at h_c/H_o = {shelfDepth / incidentHeight:.3f}', flush = True)

    sections = {}
    for y in innerSections:
        section = summary['sections'][f'{y:.1f}']
        nearest = int(np.argmin(np.abs(times - section['breakingTime'])))
        troughHeight, profileCrest = trough(frames[nearest][1], y, section['breakingCrestX'], summary['region'], spacing)
        breakingHeight = section['breakingCrestHeight'] - troughHeight
        ratio = section['cavityAreaOverDepthSquared'] / breakingHeight**2
        reading = 'consistent' if low <= ratio <= high else ('above the band' if ratio > high else 'below the band')
        sections[y] = {'breakingTime': section['breakingTime'], 'frameTime': times[nearest],
                       'crestX': section['breakingCrestX'], 'crestHeight': section['breakingCrestHeight'],
                       'profileCrestHeight': profileCrest, 'troughHeight': troughHeight, 'breakingHeight': breakingHeight,
                       'breakerIndex': breakingHeight / shelfDepth, 'cavityArea': section['cavityAreaOverDepthSquared'],
                       'cavityAreaOverBreakingHeightSquared': ratio, 'reading': reading,
                       'withinPooledRange': pooledRange[0] <= ratio <= pooledRange[1]}
        print(f'  y {y:.1f}: frame {times[nearest] - section["breakingTime"]:+.4f} s from breaking; crest '
              f'{section["breakingCrestHeight"]:.3f} (profile {profileCrest:.3f}), trough {troughHeight:+.4f}, '
              f'H_b {breakingHeight:.3f} h0 (H_b/h_c {breakingHeight / shelfDepth:.2f}); A_c {section["cavityAreaOverDepthSquared"]:.4f} h0^2, '
              f'A_c/H_b^2 {ratio:.3f}: {reading}', flush = True)

    ratios = np.array([entry['cavityAreaOverBreakingHeightSquared'] for entry in sections.values()])
    print(f'A_c/H_b^2 {ratios.min():.3f} to {ratios.max():.3f} over the seven inner sections, against the band '
          f'{low:.3f} to {high:.3f} and the pooled {pooledRange[0]} to {pooledRange[1]}', flush = True)
    record = {'case': 'cavityArea', 'run': str(folder), 'particlesPerDepth': summary['particlesPerDepth'],
              'relativeSubmergence': shelfDepth / incidentHeight, 'band': {'relativeSubmergenceAtMost': bandRelativeSubmergence,
              'points': len(band), 'low': low, 'high': high}, 'pooledRange': pooledRange, 'sections': sections}
    writeResult('cavityArea', record)
    return record

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'The tube at touchdown on the turned Surf Ranch reef, against measured overturns.')
    parser.add_argument('folder', nargs = '?', default = r'D:\pySurfRuns\surfRanch32', help = 'the run folder')
    arguments = parser.parse_args()
    main(Path(arguments.folder))
