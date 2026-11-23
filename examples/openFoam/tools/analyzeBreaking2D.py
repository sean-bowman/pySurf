
# -- Breaking Point of the 2D OpenFOAM Solitary Wave -- #

'''

The breaking point of the solitaryBreaking2D case against Grilli, Svendsen and Subramanya (1997).

Grilli et al. computed solitary waves shoaling on plane slopes with fully nonlinear potential flow
and took the breaking point where the front face first turns vertical. For H0/h0 = 0.3 on a 1:35
slope they give Hb/hb = 1.380 and Hb/h0 = 0.476 (tabulated in Table 8 of Senthilkumar 2016,
arXiv:1601.06822), so hb/h0 = 0.345: Hb is the crest height above still water and hb the still
water depth under the crest.

The free surface of each written frame is the alpha.water = 0.5 contour through the cell centres.
Along its front face, from the crest down to still water, the lean dx/ds is positive while the face
slopes forward, zero where it stands vertical and negative under an overhang. The breaking point is
the first instant the smallest lean reaches zero, interpolated between the frames either side; the
crest's height and position are interpolated to the same instant.

The acceptance criteria are those pySurf applies to its own breaking models, set before any
comparison: Hb/h0 within 10 percent and hb/h0 within 15 percent of Grilli's. Grilli's starting wave
is the exact potential flow solitary wave (Tanaka 1986); this case generates Grimshaw's third-order
wave at the inlet, so a few percent of difference in the breaking height belongs to the starting
wave rather than to the solver.

Run from a reconstructed case with cell centres written (Allrun does both):

    python3 analyzeBreaking2D.py path/to/solitaryBreaking2D

It prints the comparison, writes breaking2D.json and breaking2D.png in the case, and needs numpy and
matplotlib.

Sean Bowman [10/24/2026]

'''

# Standard library imports
import argparse
import json
import math
import re
from pathlib import Path

# Third-party imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.tri import Triangulation

#----------------------------------------------------------------------#
# -- References and geometry -- #
#----------------------------------------------------------------------#

grilliBreakingHeight = 0.476                                # [-] Hb/h0, Grilli et al. (1997)
grilliBreakingIndex = 1.380                                 # [-] Hb/hb
grilliBreakingDepth = grilliBreakingHeight / grilliBreakingIndex    # [-] hb/h0
bemBreakingHeight = 0.504                                   # [-] Hb/h0, pysurf.OverturningWave
bemBreakingDepth = 0.359                                    # [-] hb/h0, pysurf.OverturningWave
heightTolerance, depthTolerance = 0.10, 0.15                # [-] acceptance criteria

stillWaterLevel = 1.0                                       # [m], h0 = 1 m
waveHeight = 0.3                                            # [m]
toeX, slope, endDepth = 10.0, 1.0 / 35.0, 0.27              # [m], [-], [m], as makeBlockMeshDict.py

def bedElevation(x: np.ndarray) -> np.ndarray:

    '''The flume's bed elevation [m]: flat to the toe, 1:35 to a depth of 0.27 m, then flat.'''

    return np.clip((np.asarray(x) - toeX) * slope, 0.0, stillWaterLevel - endDepth)

#----------------------------------------------------------------------#
# -- Reading OpenFOAM fields -- #
#----------------------------------------------------------------------#

def readInternalField(path: Path, count: int | None = None) -> np.ndarray:

    '''

    The internalField of an ASCII OpenFOAM field file, scalar (N,) or vector (N, 3).

    Raises:
    -------
    ValueError : If the field is binary or its length does not match `count`.

    '''

    text = path.read_text(errors = 'replace')
    uniform = re.search(r'internalField\s+uniform\s+([^;]+);', text)
    if uniform:
        if count is None:
            raise ValueError(f'{path} is uniform; give the cell count.')
        value = uniform.group(1).strip().strip('()').split()
        return np.tile(np.array(value, dtype = float), (count, 1)).squeeze()

    header = re.search(r'internalField\s+nonuniform\s+List<(\w+)>\s*(\d+)\s*\(', text)
    if header is None:
        raise ValueError(f'{path}: no ASCII nonuniform internalField; write the case with writeFormat ascii.')
    kind, length = header.group(1), int(header.group(2))
    block = text[header.end():text.index('\n)', header.end())]
    values = np.array(block.replace('(', ' ').replace(')', ' ').split(), dtype = float)
    values = values.reshape(length, 3) if kind == 'vector' else values
    if len(values) != length or (count is not None and length != count):
        raise ValueError(f'{path}: read {len(values)} values for {length} cells.')

    return values

def timeDirectories(case: Path) -> list[tuple[float, Path]]:

    '''The written times holding alpha.water, in order, the initial state left out.'''

    times = []
    for folder in case.iterdir():
        try:
            time = float(folder.name)
        except ValueError:
            continue
        if time > 0.0 and (folder / 'alpha.water').exists():
            times.append((time, folder))

    return sorted(times)

#----------------------------------------------------------------------#
# -- The free surface and its front face -- #
#----------------------------------------------------------------------#

def contourSegments(triangulation: Triangulation, values: np.ndarray, level: float = 0.5) -> list[np.ndarray]:

    '''The polylines of a level of a field given at scattered points, each shape (M, 2).'''

    figure = plt.figure()
    try:
        contours = plt.tricontour(triangulation, values, levels = [level])
        if hasattr(contours, 'get_paths'):
            # Matplotlib 3.8 and later: one path per level, its pieces starting at each MOVETO.
            # Path.to_polygons would simplify the path and cut the surface into fragments
            segments = []
            for path in contours.get_paths():
                if path.codes is None:
                    segments.append(path.vertices)
                    continue
                starts = list(np.flatnonzero(path.codes == MplPath.MOVETO)) + [len(path.codes)]
                segments.extend(path.vertices[start:stop] for start, stop in zip(starts[:-1], starts[1:]))
        else:
            segments = list(contours.allsegs[0])
    finally:
        plt.close(figure)

    return [np.asarray(segment) for segment in segments if len(segment) > 2]

def smoothCurve(curve: np.ndarray, spacing: float, passes: int = 3) -> np.ndarray:

    '''

    A polyline resampled at a uniform arc-length `spacing` and smoothed by `passes` of a three-point
    average, its ends held. A contour through cell centres carries a kink at every cell it crosses;
    smoothed over a few cells, its direction is the surface's.

    '''

    lengths = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(curve, axis = 0).T))])
    stations = np.linspace(0.0, lengths[-1], max(int(lengths[-1] / spacing), 2) + 1)
    resampled = np.column_stack([np.interp(stations, lengths, curve[:, 0]), np.interp(stations, lengths, curve[:, 1])])
    for _ in range(passes):
        resampled[1:-1] = 0.25 * resampled[:-2] + 0.5 * resampled[1:-1] + 0.25 * resampled[2:]

    return resampled

def surfaceMeasures(surface: np.ndarray, cellSize: float) -> dict:

    '''

    The crest of the free surface and the smallest lean of its front face. The surface is the
    longest contour, run from offshore to shoreward and smoothed over the cell size; the front face
    runs from the crest to where the surface first comes back within 5 percent of the wave height of
    still water.

    '''

    if surface[0, 0] > surface[-1, 0]:
        surface = surface[::-1]
    surface = smoothCurve(surface, 0.5 * cellSize)
    crest = int(np.argmax(surface[:, 1]))
    settled = np.flatnonzero(surface[crest:, 1] < stillWaterLevel + 0.05 * waveHeight)
    stop = crest + (int(settled[0]) if settled.size else len(surface) - crest)
    front = surface[crest:stop + 1]
    if len(front) < 4:
        return {'crestX': float(surface[crest, 0]), 'crestZ': float(surface[crest, 1]), 'minimumLean': 1.0}
    step = np.gradient(front, axis = 0)
    lean = step[:, 0] / np.maximum(np.hypot(step[:, 0], step[:, 1]), 1e-12)

    return {'crestX': float(surface[crest, 0]), 'crestZ': float(surface[crest, 1]),
            'minimumLean': float(np.min(lean))}

def breakingPoint(frames: list[dict]) -> dict | None:

    '''The first instant the front face stands vertical, interpolated between frames; None if it never does.'''

    for before, after in zip(frames[:-1], frames[1:]):
        if before['minimumLean'] > 0.0 >= after['minimumLean']:
            share = before['minimumLean'] / (before['minimumLean'] - after['minimumLean'])
            interpolate = lambda key: before[key] + share * (after[key] - before[key])
            crestX = interpolate('crestX')
            depth = stillWaterLevel - float(bedElevation(crestX))
            height = interpolate('crestZ') - stillWaterLevel
            return {'time': interpolate('time'), 'crestX': crestX, 'breakingHeight': height,
                    'breakingDepth': depth, 'breakingIndex': height / depth,
                    'framesApart': after['time'] - before['time']}

    return None

def waterVolume(case: Path) -> dict | None:

    '''The water volume the waterVolume function object recorded, first and last [m^3].'''

    rows = []
    for path in sorted((case / 'postProcessing' / 'waterVolume').glob('*/volFieldValue.dat')):
        for line in path.read_text().splitlines():
            if line.strip() and not line.startswith('#'):
                rows.append([float(value) for value in line.split()[:2]])
    if not rows:
        return None
    rows = np.array(sorted(rows))

    return {'first': float(rows[0, 1]), 'last': float(rows[-1, 1]), 'times': (float(rows[0, 0]), float(rows[-1, 0]))}

#----------------------------------------------------------------------#
# -- Report -- #
#----------------------------------------------------------------------#

def report(point: dict, volume: dict | None, width: float) -> dict:

    '''Print the comparison and return it.'''

    rows = [('Hb/h0', point['breakingHeight'], grilliBreakingHeight, bemBreakingHeight, heightTolerance),
            ('hb/h0', point['breakingDepth'], grilliBreakingDepth, bemBreakingDepth, depthTolerance),
            ('Hb/hb', point['breakingIndex'], grilliBreakingIndex, bemBreakingHeight / bemBreakingDepth, None)]
    print(f'Breaking at t = {point["time"]:.3f} s, crest at x = {point["crestX"]:.3f} m '
          f'({point["crestX"] - toeX:.2f} h0 past the toe); frames {point["framesApart"]:.3f} s apart')
    print(f'{"quantity":<8}{"OpenFOAM":>10}{"Grilli 1997":>13}{"error":>9}{"criterion":>11}{"verdict":>9}'
          f'{"pySurf BEM":>12}')
    results = {}
    for name, value, reference, bem, tolerance in rows:
        error = value / reference - 1.0
        verdict = '' if tolerance is None else ('pass' if abs(error) < tolerance else 'fail')
        criterion = '' if tolerance is None else f'{100 * tolerance:.0f} %'
        print(f'{name:<8}{value:>10.3f}{reference:>13.3f}{100 * error:>8.1f}%{criterion:>11}{verdict:>9}{bem:>12.3f}')
        results[name] = {'openfoam': value, 'grilli': reference, 'error': error, 'verdict': verdict, 'pysurfBem': bem}

    if volume is not None:
        # A solitary wave carries 2 H / k of water per unit width, k = sqrt(3 H / (4 h^3)): the
        # volume the wave maker should have added once the wave is fully in
        added = (volume['last'] - volume['first']) / width
        expected = 2.0 * waveHeight / math.sqrt(3.0 * waveHeight / (4.0 * stillWaterLevel**3))
        print(f'Water added by the wave maker: {added:.3f} m^3/m; a first-order solitary wave of this height '
              f'carries {expected:.3f} m^3/m ({100 * (added / expected - 1.0):+.1f} %)')
        results['waterAdded'] = {'openfoam': added, 'solitaryWave': expected}

    return results

def drawFrames(frames: list[dict], point: dict, path: Path) -> None:

    '''The free surface in the frames around the breaking point, over the bed.'''

    near = [frame for frame in frames if abs(frame['time'] - point['time']) <= 0.4]
    figure, axis = plt.subplots(figsize = (11, 4.5), constrained_layout = True)
    xs = np.linspace(point['crestX'] - 4.0, point['crestX'] + 2.0, 200)
    axis.fill_between(xs, bedElevation(xs) - 0.1, bedElevation(xs), color = '#c8b48c')
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, max(len(near), 1)))
    for color, frame in zip(colors, near):
        surface = frame['surface']
        inside = (surface[:, 0] > xs[0]) & (surface[:, 0] < xs[-1])
        axis.plot(surface[inside, 0], surface[inside, 1], color = color, lw = 1.2, label = f't = {frame["time"]:.2f} s')
    axis.plot(point['crestX'], stillWaterLevel + point['breakingHeight'], 'o', color = 'crimson',
              label = 'breaking point')
    axis.axhline(stillWaterLevel, color = 'grey', lw = 0.6, ls = ':')
    axis.set_aspect('equal')
    axis.set_xlabel('x [m]')
    axis.set_ylabel('z [m]')
    axis.set_title('Solitary wave H0/h0 = 0.3 on a 1:35 slope: free surface around the breaking point')
    axis.legend(fontsize = 7, ncol = 2, loc = 'upper left')
    figure.savefig(path, dpi = 120)
    plt.close(figure)

def main() -> None:

    parser = argparse.ArgumentParser(description = 'Breaking point of solitaryBreaking2D against Grilli et al. (1997).')
    parser.add_argument('case', type = Path, nargs = '?', default = Path('.'))
    args = parser.parse_args()
    case = args.case.resolve()

    centres = readInternalField(case / '0' / 'C')
    triangulation = Triangulation(centres[:, 0], centres[:, 2])
    # The finest cell height near still water: the distance to the nearest centre above
    band = np.abs(centres[:, 2] - stillWaterLevel) < 0.1
    cellSize = float(np.median(np.diff(np.unique(np.round(centres[band, 2], 6)))))
    frames = []
    for time, folder in timeDirectories(case):
        alpha = readInternalField(folder / 'alpha.water', len(centres))
        segments = contourSegments(triangulation, alpha)
        if not segments:
            continue
        surface = max(segments, key = len)
        frames.append({'time': time, 'surface': surface, **surfaceMeasures(surface, cellSize)})
        print(f'  t = {time:7.3f} s  crest x = {frames[-1]["crestX"]:7.3f} m  '
              f'H = {frames[-1]["crestZ"] - stillWaterLevel:6.3f} m  smallest lean {frames[-1]["minimumLean"]:+.3f}')

    point = breakingPoint(frames)
    if point is None:
        print('The front face never turned vertical in the written frames; run further or write more often.')
        return
    volume = waterVolume(case)
    # The 2D mesh is one cell across; the volume function object integrates over that width
    spanWidth = 2.0 * float(np.max(centres[:, 1]))
    results = report(point, volume, spanWidth)
    results['breakingPoint'] = point
    (case / 'breaking2D.json').write_text(json.dumps(results, indent = 2))
    drawFrames(frames, point, case / 'breaking2D.png')
    print(f'Wrote {case / "breaking2D.json"} and {case / "breaking2D.png"}')

if __name__ == '__main__':
    main()
