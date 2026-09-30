
# -- Parametric Surface Sampler -- #

'''

Samples the parametric surfboard surface into a structured vertex grid
suitable for Three.js BufferGeometry construction.

Queries the BoardGeometry facade at a regular grid of longitudinal and
lateral stations, producing 2D arrays of deck and bottom surface heights.
Only the positive-Y (right rail) half is stored; the JavaScript viewer
mirrors it for the full board.

All output values are in millimeters to match the project coordinate system.

Sean Bowman [02/04/2026]

'''

from __future__ import annotations

import numpy as np

from pysurf.Surfboard.SurfPhysics.geometry.board import BoardGeometry
from pysurf.Surfboard.SurfPhysics.geometry.parameters import SurfboardParameters
from pysurf.Surfboard.SurfPhysics.units import mToMm

def sampleParametricSurface(
    board: BoardGeometry,
    params: SurfboardParameters,
    nLongitudinal: int = 100,
    nLateral: int = 25,
) -> dict:

    '''

    Sample the parametric board surface into a vertex grid.

    Produces structured arrays of outline, rocker, deck, and bottom
    surface data that can be serialized to JSON and reconstructed
    as a Three.js BufferGeometry.

    The coordinate system matches the C# SurfboardBody convention:
        X = along board length (0 = nose, length = tail) [mm]
        Y = across width (0 = centerline, +halfWidth = right rail) [mm]
        Z = vertical (0 = rocker flat spot, positive = up) [mm]

    Parameters:
    -----------
    board : BoardGeometry
        Parametric board geometry (returns SI meters internally)
    params : SurfboardParameters
        Board parameters (dimensions in mm)
    nLongitudinal : int
        Number of stations along the board length
    nLateral : int
        Number of stations across the half-width

    Returns:
    --------
    dict : Structured surface data with keys:
        - nLongitudinal, nLateral
        - tValues: list of normalized longitudinal positions
        - lateralFractions: list of normalized lateral positions
        - outline.halfWidthsMm: half-width at each t [mm]
        - rocker.heightsMm: rocker Z-offset at each t [mm]
        - deckSurface.heightsMm: 2D array [nLong x nLat] of deck Z [mm]
        - bottomSurface.heightsMm: 2D array [nLong x nLat] of bottom Z [mm]

    '''
    tValues = np.linspace(0.0, 1.0, nLongitudinal).tolist()
    lateralFractions = np.linspace(0.0, 1.0, nLateral).tolist()

    # Sample outline half-widths (mm)
    halfWidthsMm = [
        mToMm(board.getHalfWidthM(t)) for t in tValues
    ]

    # Sample rocker heights (mm)
    rockerHeightsMm = [
        mToMm(board.getRockerHeightM(t)) for t in tValues
    ]

    # Sample deck and bottom surface heights (mm)
    # These are relative to the center plane (before rocker offset)
    # The full Z position is: rocker(t) + surfaceHeight(t, lf)
    deckSurface = []
    bottomSurface = []

    for t in tValues:
        deckRow = []
        bottomRow = []

        for lf in lateralFractions:
            deckZ = mToMm(board.getDeckHeightM(t, lf))
            bottomZ = mToMm(board.getBottomHeightM(t, lf))
            deckRow.append(round(deckZ, 4))
            bottomRow.append(round(bottomZ, 4))

        deckSurface.append(deckRow)
        bottomSurface.append(bottomRow)

    # Round values to reduce JSON size
    halfWidthsMm = [round(v, 4) for v in halfWidthsMm]
    rockerHeightsMm = [round(v, 4) for v in rockerHeightsMm]
    tValues = [round(v, 6) for v in tValues]
    lateralFractions = [round(v, 6) for v in lateralFractions]

    return {
        'nLongitudinal': nLongitudinal,
        'nLateral': nLateral,
        'tValues': tValues,
        'lateralFractions': lateralFractions,
        'outline': {
            'halfWidthsMm': halfWidthsMm,
        },
        'rocker': {
            'heightsMm': rockerHeightsMm,
        },
        'deckSurface': {
            'heightsMm': deckSurface,
        },
        'bottomSurface': {
            'heightsMm': bottomSurface,
        },
    }
