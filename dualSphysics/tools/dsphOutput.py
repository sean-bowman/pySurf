
# -- Reading DualSPHysics Particle Output -- #

'''

Frames of a DualSPHysics run, read two ways.

    binary      `readPartArrays` reads the solver's own Part files (out/data/Part_NNNN.bi4) directly:
                positions, ids, velocities and densities, with no conversion step. `partTimes`
                gives each file's time from out/RunPARTs.csv. The gate tools read runs this way
    CSV         `readFrame` and `readFrames` read the files PartVTK writes with

                    PartVTK_linux64 -dirdata out/data -savecsv csv/PartFluid -onlytype:-all,fluid -vars:-all

                each holding one output time: a header line, a line of values whose first entry
                is the time, a blank line, then a line of column names and one row per particle,
                fields separated by semicolons. Extra variables written with -vars (velocity,
                density) come back as further columns when asked for by name. The dam break check
                reads its runs this way

Sean Bowman [11/08/2026]

'''

# Standard library imports
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Third-party imports
import numpy as np

@dataclass
class PartFrame:

    '''One output time of a DualSPHysics run: positions [m] and any further columns by name.'''

    time: float
    positions: np.ndarray
    columns: dict = field(default_factory = dict)

def readFrame(path: str | Path, extra: tuple[str, ...] = ()) -> PartFrame:

    '''
    Read one PartVTK CSV file.

    Parameters
    ----------
    path : str | Path
        The CSV file
    extra : tuple[str, ...]
        Column names to return besides the position, as PartVTK writes them (e.g. 'Vel.x [m/s]')

    Returns
    -------
    PartFrame
        The output time and the particles' positions, shape (n, 3)
    '''

    lines = Path(path).read_text(encoding = 'utf-8').splitlines()
    time = float(lines[1].split(';')[0])
    # The particle table starts at the line of column names after the blank line
    start = next(i for i, line in enumerate(lines) if line.startswith('Pos.x'))
    names = [name.strip() for name in lines[start].split(';') if name.strip()]
    rows = [line for line in lines[start + 1:] if line.strip()]
    data = np.array([[float(value) for value in row.split(';') if value.strip()] for row in rows]).reshape(-1, len(names))
    positions = data[:, :3]
    columns = {name: data[:, names.index(name)] for name in extra}
    return PartFrame(time = time, positions = positions, columns = columns)

# Byte sizes of the JBinaryData types that particle arrays use (JBinaryDataDef::TpData)
arrayTypes = {7: ('<i4', 1), 8: ('<u4', 1), 9: ('<i8', 1), 10: ('<u8', 1), 11: ('<f4', 1), 12: ('<f8', 1),
              20: ('<i4', 3), 21: ('<u4', 3), 22: ('<f4', 3), 23: ('<f8', 3)}
arrayCode = b'\x06\x00\x00\x00\nARRAY'

def readPartArrays(path: str | Path) -> dict[str, np.ndarray]:

    '''

    Every typed array in one of DualSPHysics's binary Part files (data/Part_NNNN.bi4), by name: 'Pos'
    (float3, or 'Posd' as double3 with SavePosDouble), 'Idp', 'Vel', 'Rhop' and any others it saved.

    The file is a JBinaryData tree (src/source/JBinaryData.cpp). Each array is written as the
    string "\nARRAY", its name, a 4-byte hide flag, a 4-byte type code, then its element count
    and its data size in bytes (4-byte integers, or 8-byte in files over 4 GB), then the raw
    little-endian data. Strings are a 4-byte length followed by their bytes. The reader finds each
    array by its code and reads it in place, so it needs no other part of the tree.

    '''

    raw = Path(path).read_bytes()
    arrays = {}
    start = raw.find(arrayCode)
    while start >= 0:
        cursor = start + len(arrayCode)
        nameLength = int.from_bytes(raw[cursor:cursor + 4], 'little')
        name = raw[cursor + 4:cursor + 4 + nameLength].decode('ascii')
        cursor += 4 + nameLength + 4                    # name, then the hide flag (a bool written as 4 bytes)
        typeCode = int.from_bytes(raw[cursor:cursor + 4], 'little', signed = True)
        cursor += 4
        if typeCode in arrayTypes:
            dtype, width = arrayTypes[typeCode]
            itemSize = np.dtype(dtype).itemsize * width
            count = int.from_bytes(raw[cursor:cursor + 4], 'little')
            size = int.from_bytes(raw[cursor + 4:cursor + 8], 'little')
            header = 8
            if size != count * itemSize:
                # Files over 4 GB store the count and the size as 8-byte integers
                count = int.from_bytes(raw[cursor:cursor + 8], 'little')
                size = int.from_bytes(raw[cursor + 8:cursor + 16], 'little')
                header = 16
            data = np.frombuffer(raw, dtype = dtype, count = count * width, offset = cursor + header)
            arrays[name] = data.reshape(count, width) if width > 1 else data.copy()
        start = raw.find(arrayCode, cursor)
    return arrays

def partTimes(outDir: str | Path) -> dict[int, float]:

    '''The time of each Part file [s], by its number, from the run's out/RunPARTs.csv.'''

    times = {}
    for line in Path(outDir, 'RunPARTs.csv').read_text(encoding = 'utf-8').splitlines()[1:]:
        fields = line.split(';')
        if len(fields) > 1 and fields[0].strip().isdigit():
            times[int(fields[0])] = float(fields[1])
    return times

def readFrames(csvDir: str | Path, prefix: str = 'PartFluid', extra: tuple[str, ...] = ()) -> list[PartFrame]:

    '''Every frame of a run, in output order, from the files <prefix>_NNNN.csv in csvDir.'''

    # PartVTK also writes <prefix>_stats.csv, a per-frame summary; only the numbered files are frames
    paths = sorted(Path(csvDir).glob(f'{prefix}_[0-9]*.csv'))
    if not paths:
        raise FileNotFoundError(f'No {prefix}_*.csv files in {csvDir}.')
    return [readFrame(path, extra) for path in paths]
