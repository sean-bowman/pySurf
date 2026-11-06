
# -- pySurf: SPH Frame Files -- #

'''

The frames of an SPH run, streamed to disk one file each, for runs whose frames do not fit in
memory together.

    FrameWriter     one .npz file per frame in a run directory, and frames.json indexing them
    readIndex       a run's index: each frame's number, time, step, particle count and file
    readFrame       one frame's arrays

A frame holds the fluid particles' positions and speeds in float32, 16 bytes a particle in 3D: one
frame of 14 million particles is 224 MB. A model may add per-particle fields of its own, and
scalars, which go into the index. The index is rewritten after every frame, so a run that stops
early can still be read up to its last frame.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

Bookkeeping only. tests/testSphFrames.py writes frames and reads them back: positions and speeds
to float32, extra fields and scalars, the index of a run cut short, and compressed files.

Sean Bowman [11/05/2026]

'''

# Standard library imports
from __future__ import annotations

import json
from pathlib import Path

# Third-party imports
import numpy as np

indexName = 'frames.json'

class FrameWriter:

    '''

    Writes an SPH run's frames into a directory.

    Parameters:
    -----------
    directory : str | Path
        Run directory, created if missing; frames already indexed there are replaced
    compress : bool
        Write compressed .npz files: smaller, and slower to write and read

    '''

    def __init__(self, directory: str | Path, compress: bool = False) -> None:

        self.directory = Path(directory)
        self.directory.mkdir(parents = True, exist_ok = True)
        self.compress = compress
        self.frames: list[dict] = []

    def write(self, time: float, particles, step: int | None = None, fields: dict | None = None,
              scalars: dict | None = None) -> Path:

        '''

        Write the fluid's state at one instant.

        Parameters:
        -----------
        time : float
            Simulated time [s]
        particles : ParticleSystem
            The particle state; its fluid particles are written
        step : int | None
            Solver steps taken
        fields : dict | None
            Further per-particle arrays, one row per fluid particle, stored as given
        scalars : dict | None
            Numbers recorded in the index with the frame

        Returns:
        --------
        Path : The frame's file

        '''

        fluid = particles.isFluid
        arrays = {'positions': particles.positions[fluid].astype(np.float32),
                  'speeds': np.linalg.norm(particles.velocities[fluid], axis = 1).astype(np.float32)}
        count = len(arrays['speeds'])
        for name, values in (fields or {}).items():
            values = np.asarray(values)
            if len(values) != count:
                raise ValueError(f"Field '{name}' has {len(values)} rows for {count} fluid particles.")
            arrays[name] = values
        number = len(self.frames)
        path = self.directory / f'frame{number:05d}.npz'
        (np.savez_compressed if self.compress else np.savez)(path, **arrays)
        entry = {'frame': number, 'time': float(time), 'step': step, 'count': count, 'file': path.name}
        entry.update({name: float(value) for name, value in (scalars or {}).items()})
        self.frames.append(entry)
        with open(self.directory / indexName, 'w', encoding = 'utf-8') as handle:
            json.dump({'frames': self.frames}, handle, indent = 1)

        return path

def readIndex(directory: str | Path) -> list[dict]:

    '''A run's index: one entry per frame, in order.'''

    with open(Path(directory) / indexName, 'r', encoding = 'utf-8') as handle:
        return json.load(handle)['frames']

def readFrame(directory: str | Path, frame: int) -> dict[str, np.ndarray]:

    '''One frame's arrays: positions, speeds, and any fields written with it.'''

    entry = readIndex(directory)[frame]
    with np.load(Path(directory) / entry['file'], allow_pickle = False) as data:
        return {name: data[name] for name in data.files}
