# -- pySurf: Water Surface Frames -- #

'''

A run's saved frames drawn as a water surface (`waterSurface`), with spray, foam and bubbles
(`whitewater`) if asked, to an mp4 or to stills. Each frame file (`frame_*.npz`) holds the time in
seconds, the particles' positions and velocities, and which are split daughters, whose volume is
an eighth of a spacing cubed.

The surface is the 0.5 level of the particles' volumes spread by the Wendland C2 kernel at
R = 3 spacings, its normals from the same volumes at R_n = 6 spacings, coloured by speed (white at
`--white`). Whitewater, when drawn, is stepped once per saved frame from the first frame drawn (or
from `--whitewaterFrom`), so a still with it carries every frame before it. Every drawing carries
the run's "Not validated" label, and with whitewater a note that it is a look.

    python renderSurfaceFrames.py <frames folder> <output.mp4 | output.png>
        --box x0 x1 y0 y1 z0 z1 --target x y z --distance d --elevation deg --azimuth deg
        [--fov deg] [--size W H] [--from t'] [--to t'] [--fps n] [--ppd n] [--depth h0]
        [--white m/s] [--whitewater] [--whitewaterFrom t'] [--flip] [--still t' ...]
        [--caption text]

The Surf Ranch shoal run, looking back along the crest from shoreward of the reef:

    python tools/rendering/renderSurfaceFrames.py D:/pySurfRuns/surfRanch32 surfRanchSurface.mp4
        --box 38 47 0 4 0.4 2.6 --target 41.9 1.4 1.2 --distance 5.5 --elevation 10 --azimuth 62
        --fov 34 --from 19.5 --white 7.92 --whitewater
        --caption "The Surf Ranch reef, its contours at 25 deg to the crest, 3D SPH at 32 particles
        per depth with the crest's water split once"

Sean Bowman [11/26/2026]

'''

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

from waterSurface import Camera, Look, WaterSurfaceRenderer
from whitewater import Whitewater, WhitewaterSettings, composite

gravity = 9.80665          # [m/s^2]
ink = (213, 222, 233)

def frameTimes(folder: Path) -> list:
    '''(time [s], path) of every frame in the folder, in time order.'''
    paths = sorted(folder.glob('frame_*.npz'))
    return sorted((float(np.load(path)['time']), path) for path in paths)

def loadFrame(path: Path, spacing: float) -> dict:
    '''Positions, velocities and volumes [m, m/s, m^3] of one frame.'''
    data = np.load(path)
    split = data['split'].astype(bool) if 'split' in data else np.zeros(len(data['positions']), bool)
    return {'time': float(data['time']), 'positions': data['positions'].astype(np.float32),
            'velocities': data['velocities'].astype(np.float32),
            'volumes': np.where(split, spacing**3 / 8.0, spacing**3)}

def caption(image: np.ndarray, lines: list) -> np.ndarray:

    '''The image (H, W, 3) in 0 to 1 as 8-bit, with `lines` written along its lower left.'''

    picture = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(picture)
    size = max(12, picture.height // 42)
    font = ImageFont.truetype(font_manager.findfont('DejaVu Sans'), size)
    y = picture.height - int(1.45 * size) * len(lines) - size // 2
    for line in lines:
        draw.text((size, y), line, font = font, fill = ink)
        y += int(1.45 * size)

    return np.asarray(picture)

def main(arguments: list | None = None) -> None:
    parser = argparse.ArgumentParser(description = 'Draw saved SPH frames as a water surface.')
    parser.add_argument('frames', type = Path)
    parser.add_argument('output', type = Path)
    parser.add_argument('--box', type = float, nargs = 6, required = True)
    parser.add_argument('--target', type = float, nargs = 3, required = True)
    parser.add_argument('--distance', type = float, required = True)
    parser.add_argument('--elevation', type = float, required = True)
    parser.add_argument('--azimuth', type = float, required = True)
    parser.add_argument('--fov', type = float, default = 30.0)
    parser.add_argument('--size', type = int, nargs = 2, default = (1280, 640))
    parser.add_argument('--from', dest = 'first', type = float, default = 0.0)
    parser.add_argument('--to', dest = 'last', type = float, default = math.inf)
    parser.add_argument('--fps', type = int, default = 30)
    parser.add_argument('--ppd', type = int, default = 32)
    parser.add_argument('--depth', type = float, default = 1.0)
    parser.add_argument('--white', type = float, default = None)
    parser.add_argument('--whitewater', action = 'store_true')
    parser.add_argument('--whitewaterFrom', type = float, default = None)
    parser.add_argument('--flip', action = 'store_true')
    parser.add_argument('--still', type = float, nargs = '+', default = None)
    parser.add_argument('--caption', default = None)
    options = parser.parse_args(arguments)

    depth = options.depth
    spacing = depth / options.ppd                                                     # [m]
    timeScale = math.sqrt(depth / gravity)                                            # [s]
    width, height = (value + value % 2 for value in options.size)                     # even, for yuv420p
    camera = Camera.orbit(options.target, options.distance, options.elevation, options.azimuth, width, height,
                          options.fov)
    white = options.white if options.white is not None else 2.0 * math.sqrt(gravity * depth)
    look = Look(supportRadius = 3.0 * spacing, cellSize = 0.5 * spacing, normalRadius = 6.0 * spacing, speedScale = white)
    renderer = WaterSurfaceRenderer(camera, look, tuple(options.box))
    whitewater = Whitewater(WhitewaterSettings(spacing = spacing), tuple(options.box)) if options.whitewater else None
    description = options.caption or (f'3D SPH at {options.ppd} particles per depth; the surface reconstructed from '
                                      'the particles, coloured by speed')
    lines = [description + '.  Not validated.']
    if whitewater is not None:
        lines.append('Spray, foam and bubbles after Ihmsen et al. (2012): a look, not physics.')

    frames = frameTimes(options.frames)
    stills = sorted(options.still) if options.still else []
    # A still draws the frame nearest its time
    stillFrames = {min(frames, key = lambda item: abs(item[0] / timeScale - target))[1] for target in stills}
    stillTimes = [t / timeScale for t, path in frames if path in stillFrames]
    first = min(stillTimes) if stills else options.first
    last = max(stillTimes) if stills else options.last
    start = options.whitewaterFrom if (whitewater is not None and options.whitewaterFrom is not None) else first
    chosen = [(t, path) for t, path in frames if min(start, first) - 1e-6 <= t / timeScale <= last + 1e-6]

    encoder = None
    if not stills:
        encoder = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo',
                                    '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(options.fps), '-i', '-',
                                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'slow',
                                    str(options.output)], stdin = subprocess.PIPE)
    previous, began, written = None, time.perf_counter(), 0
    for t, path in chosen:
        frame = loadFrame(path, spacing)
        if whitewater is not None and previous is not None:
            # Carry the diffuse particles to this frame's time through the previous frame's fluid
            whitewater.step(previous['positions'], previous['velocities'], previous['volumes'], frame['time'] - previous['time'])
        previous = frame
        drawn = (path in stillFrames) if stills else (t / timeScale >= first - 1e-6)
        if not drawn:
            continue
        surface = renderer.render(frame['positions'], frame['volumes'], np.linalg.norm(frame['velocities'], axis = 1))
        image = surface['image']
        if whitewater is not None:
            front, behind = whitewater.splat(camera, renderer.surfaceDistance(surface), spacing,
                                             float(np.mean(look.extinction)))
            image = composite(image, front, behind)
        if options.flip:
            image = image[:, ::-1]
        picture = caption(image, lines + [f"t' = {t / timeScale:.3f}  ({t:.3f} s)"])
        if stills:
            name = options.output.with_name(f'{options.output.stem}_t{t / timeScale:06.3f}.png')
            Image.fromarray(picture).save(name)
            print(f'written {name}', flush = True)
        else:
            encoder.stdin.write(picture.tobytes())
        written += 1
        if written % 10 == 1:
            count = 0 if whitewater is None else len(whitewater.kinds)
            print(f"frame {written}, t' {t / timeScale:.3f}, diffuse {count}, {time.perf_counter() - began:.0f} s",
                  flush = True)
    if encoder is not None:
        encoder.stdin.close()
        encoder.wait()
        print(f'written {options.output}: {written} frames, {time.perf_counter() - began:.0f} s', flush = True)

if __name__ == '__main__':
    main()
