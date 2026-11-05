
# -- OpenFOAM Wave Flume blockMeshDict Generator -- #

'''

Writes the blockMeshDict of a wave flume whose bed is piecewise linear in x and may run oblique to
the incoming crest, for the two cases in openFoam/cases:

    solitaryBreaking2D    Grilli, Svendsen and Subramanya (1997): h0 = 1 m, a 1:35 plane slope from
                          x = 10 m, one cell across, front and back empty
    breakingWave3D        the Surf Ranch reef of Feddersen et al. (2024): a 0.0693 face from x = 30 m
                          up to a flat 0.371 m deep, the face and flat edge turned `angle` degrees
                          from the crest so the wave breaks progressively across the span

The mesh is two layers of blocks. The lower layer runs from the bed to `zSplit`, above the highest
bed and below the lowest trough, and holds a fixed number of cells, so it thins where the bed
rises. The upper layer runs from `zSplit` to the top with uniform cells everywhere, so the free
surface, the jet and the splash are resolved at the same size over deep water and shallow.

Each bed station is a line across the span, at x = xStart + y tan(angle), with a bed elevation;
between stations the bed is a plane, so an oblique face is represented exactly.

    python makeBlockMeshDict.py solitaryBreaking2D --cell 0.02
    python makeBlockMeshDict.py breakingWave3D --cell 0.05

Sean Bowman [10/24/2026]

'''

# Standard library imports
import argparse
import math
from dataclasses import dataclass
from pathlib import Path

caseFolder = Path(__file__).resolve().parent.parent / 'cases'

#----------------------------------------------------------------------#
# -- Geometry -- #
#----------------------------------------------------------------------#

@dataclass
class Station:

    '''

    A line across the flume where the bed changes slope.

    Parameters:
    -----------
    x : float
        Position of the line at y = 0 [m]
    bedZ : float
        Bed elevation along the line [m]
    oblique : bool
        Whether the line turns with the flume's angle; the inlet and outlet stay square

    '''

    x: float
    bedZ: float
    oblique: bool = True

@dataclass
class Flume:

    '''

    A wave flume: bed stations, the layer heights, the span and the cell sizes.

    Parameters:
    -----------
    stations : list[Station]
        Bed stations from the inlet to the outlet
    cellX : list[float]
        Cell length in x within each interval between stations [m]
    zSplit : float
        Top of the lower block layer [m]
    zTop : float
        Top of the domain [m]
    width : float
        Span in y [m]; one cell across for a 2D flume
    angle : float
        Turn of the oblique stations from the y axis [deg]
    cellZ : float
        Cell height in the upper layer [m]
    lowerCells : int
        Cells across the lower layer
    cellY : float | None
        Cell width in y [m]; None for one cell across (2D)

    '''

    stations: list
    cellX: list
    zSplit: float
    zTop: float
    width: float
    angle: float
    cellZ: float
    lowerCells: int
    cellY: float | None = None

    def stationX(self, station: Station, y: float) -> float:

        '''Where a station crosses the span position y [m].'''

        return station.x + (y * math.tan(math.radians(self.angle)) if station.oblique else 0.0)

    def cellCounts(self) -> tuple[list[int], int, list[int]]:

        '''Cells in x per interval, across the span, and in z per layer.'''

        counts = []
        for (first, second), size in zip(zip(self.stations[:-1], self.stations[1:]), self.cellX):
            length = 0.5 * ((self.stationX(second, 0.0) - self.stationX(first, 0.0))
                            + (self.stationX(second, self.width) - self.stationX(first, self.width)))
            counts.append(max(1, round(length / size)))
        spanCells = 1 if self.cellY is None else max(1, round(self.width / self.cellY))
        upperCells = max(1, round((self.zTop - self.zSplit) / self.cellZ))

        return counts, spanCells, [self.lowerCells, upperCells]

    def validate(self) -> None:

        '''Check that the layers stack above the bed and the stations run shoreward.'''

        highestBed = max(station.bedZ for station in self.stations)
        if not highestBed < self.zSplit < self.zTop:
            raise ValueError(f'Need highest bed {highestBed} < zSplit {self.zSplit} < zTop {self.zTop}.')
        for y in (0.0, self.width):
            xs = [self.stationX(station, y) for station in self.stations]
            if any(b <= a for a, b in zip(xs[:-1], xs[1:])):
                raise ValueError(f'The stations cross at y = {y}: {xs}.')
        if len(self.cellX) != len(self.stations) - 1:
            raise ValueError('Give one x cell size per interval between stations.')

#----------------------------------------------------------------------#
# -- The two cases -- #
#----------------------------------------------------------------------#

def grilliFlume(cell: float) -> Flume:

    '''

    Grilli et al. (1997), H0/h0 = 0.3 on 1:35, at h0 = 1 m: flat to x = 10 m, the slope to a depth
    of 0.27 m (past the breaking depth of 0.345 m), then 3 m flat to the outlet. Cells of `cell`
    in the breaking region (x > 27 m) and 2.5 times that offshore.

    '''

    slope, toe, endDepth = 1.0 / 35.0, 10.0, 0.27
    fineStart = 27.0
    slopeEnd = toe + (1.0 - endDepth) / slope
    stations = [Station(0.0, 0.0, False), Station(toe, 0.0, False),
                Station(fineStart, (fineStart - toe) * slope, False), Station(slopeEnd, 1.0 - endDepth, False),
                Station(slopeEnd + 3.0, 1.0 - endDepth, False)]

    return Flume(stations = stations, cellX = [2.5 * cell, 2.5 * cell, cell, cell], zSplit = 0.8, zTop = 1.8,
                 width = cell, angle = 0.0, cellZ = cell, lowerCells = 12)

def surfRanchFlume(cell: float, angle: float, width: float) -> Flume:

    '''

    The Surf Ranch reef of Feddersen et al. (2024), h0 = 1 m: from an inlet at x = 20 m, flat to the
    toe at x = 30 m, the 0.0693 face up to the flat 0.371 m deep, and the flat to x = 52 m. The toe
    and the flat's edge turn `angle` degrees across a span of `width`; the inlet and outlet stay
    square. Cells of `cell` over the face and flat, twice that offshore.

    '''

    slope, shelfDepth, toe = 0.0693, 0.371, 30.0
    edge = toe + (1.0 - shelfDepth) / slope
    stations = [Station(20.0, 0.0, False), Station(toe, 0.0), Station(edge, 1.0 - shelfDepth),
                Station(52.0, 1.0 - shelfDepth, False)]

    return Flume(stations = stations, cellX = [2.0 * cell, cell, cell], zSplit = 0.8, zTop = 2.0, width = width,
                 angle = angle, cellZ = cell, lowerCells = 10, cellY = cell)

#----------------------------------------------------------------------#
# -- Writing -- #
#----------------------------------------------------------------------#

header = '''/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     |
    \\\\  /    A nd           | www.openfoam.com
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
'''

def blockMeshDict(flume: Flume, note: str) -> str:

    '''The blockMeshDict text for a flume.'''

    flume.validate()
    nx, ny, nz = flume.cellCounts()
    levels = lambda station: (station.bedZ, flume.zSplit, flume.zTop)

    # Vertex index: station k, side j (0 at y = 0, 1 at y = width), level l (bed, split, top)
    def vertex(k: int, j: int, l: int) -> int:
        return (2 * k + j) * 3 + l

    vertexLines = []
    for k, station in enumerate(flume.stations):
        for j, y in enumerate((0.0, flume.width)):
            for l, z in enumerate(levels(station)):
                vertexLines.append(f'    ({flume.stationX(station, y):.6f} {y:.6f} {z:.6f})'
                                   f'    // {vertex(k, j, l)}')

    blockLines, faces = [], {'inlet': [], 'outlet': [], 'bottom': [], 'atmosphere': [], 'sides': []}
    last = len(flume.stations) - 2
    for k in range(len(flume.stations) - 1):
        for l in range(2):
            v = [vertex(k, 0, l), vertex(k + 1, 0, l), vertex(k + 1, 1, l), vertex(k, 1, l),
                 vertex(k, 0, l + 1), vertex(k + 1, 0, l + 1), vertex(k + 1, 1, l + 1), vertex(k, 1, l + 1)]
            blockLines.append(f'    hex ({" ".join(map(str, v))}) ({nx[k]} {ny} {nz[l]}) simpleGrading (1 1 1)')
            if k == 0:
                faces['inlet'].append((v[0], v[4], v[7], v[3]))
            if k == last:
                faces['outlet'].append((v[1], v[5], v[6], v[2]))
            if l == 0:
                faces['bottom'].append((v[0], v[1], v[2], v[3]))
            if l == 1:
                faces['atmosphere'].append((v[4], v[5], v[6], v[7]))
            faces['sides'].append((v[0], v[1], v[5], v[4]))
            faces['sides'].append((v[3], v[2], v[6], v[7]))

    types = {'inlet': 'patch', 'outlet': 'patch', 'bottom': 'wall', 'atmosphere': 'patch',
             'sides': 'empty' if flume.cellY is None else 'patch'}
    patchText = []
    for name, faceList in faces.items():
        listed = '\n'.join(f'            ({" ".join(map(str, face))})' for face in faceList)
        patchText.append(f'    {name}\n    {{\n        type {types[name]};\n        faces\n        (\n{listed}\n        );\n    }}')

    cells = sum(nx) * ny * sum(nz)

    return (header + f'\n// {note}\n// {cells} cells: x {nx}, y {ny}, z {nz}\n\nscale   1;\n\nvertices\n(\n'
            + '\n'.join(vertexLines) + '\n);\n\nblocks\n(\n' + '\n'.join(blockLines)
            + '\n);\n\nedges\n(\n);\n\nboundary\n(\n' + '\n'.join(patchText)
            + '\n);\n\nmergePatchPairs\n(\n);\n\n// ************************************************************************* //\n')

def main() -> None:

    parser = argparse.ArgumentParser(description = 'Write a wave flume blockMeshDict.')
    parser.add_argument('case', choices = ['solitaryBreaking2D', 'breakingWave3D'])
    parser.add_argument('--cell', type = float, default = None, help = 'cell size in the breaking region [m]')
    parser.add_argument('--angle', type = float, default = 25.0, help = '3D: turn of the reef from the crest [deg]')
    parser.add_argument('--width', type = float, default = 4.0, help = '3D: span [m]')
    parser.add_argument('--output', type = Path, default = None, help = 'file to write; the case by default')
    args = parser.parse_args()

    if args.case == 'solitaryBreaking2D':
        cell = args.cell or 0.02
        flume = grilliFlume(cell)
        note = f'Grilli et al. (1997) flume, 1:35 slope, cell {cell} m'
    else:
        cell = args.cell or 0.05
        flume = surfRanchFlume(cell, args.angle, args.width)
        note = f'Surf Ranch reef turned {args.angle} deg across {args.width} m, cell {cell} m'

    output = args.output or caseFolder / args.case / 'system' / 'blockMeshDict'
    output.parent.mkdir(parents = True, exist_ok = True)
    output.write_text(blockMeshDict(flume, note), encoding = 'utf-8', newline = '\n')
    nx, ny, nz = flume.cellCounts()
    print(f'{output}: {sum(nx) * ny * sum(nz)} cells (x {nx}, y {ny}, z {nz})')

if __name__ == '__main__':
    main()
