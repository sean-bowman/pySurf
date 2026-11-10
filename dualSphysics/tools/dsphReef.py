
# -- DualSPHysics Case Writer: the Surf Ranch Reef Turned Across the Span (3D) -- #

'''

Writes the GenCase XML and bed geometry for a 3D DualSPHysics run of the Surf Ranch reef of
Feddersen et al. (2024) turned across the span, the geometry of openFoam/cases/breakingWave3D
(`makeBlockMeshDict.surfRanchFlume`) and the end-to-end case of phase 3 in
planning/gpuSphRoadmap.md. In pySurf's coordinates (still water at z = h0 = 1 m):

    z_b = 0.0693 (x - 30 - y tan(angle)), between 0 and 1 - 0.371 h0

flat offshore, the 0.0693 face from the toe at x = 30 + y tan(angle) up to the shelf 0.371 h0 deep,
and the shelf to the end wall at x = 52 h0; the span runs 0 <= y <= 4 h0 and the toe and the
shelf's edge turn `angle` (25 degrees) across it, so the wave breaks first at y = 0 and the break
travels across. The shelf past the face is the shoal the wave breaks onto.

The tank is `RidgeCase`'s (dsphRidge.py) with this bed: the bed band drawn from the bed surface,
mDBC's normals from that surface and from the walls' planes, free-slip walls (`slipMode` 3; v5.4's
slip mode applies to every wall), the piston square across the tank, and the scheme of phase 1.5d.
The bed is drawn on a grid sheared along the reef, x = x' + y tan(angle) with lines of constant x'
at the toe and the shelf's edge, so each grid cell lies on one plane and the triangles carry the
bed exactly.

    gauge       on the line x = 20 h0, y = 2 h0: where OpenFOAM's case generates its wave, read
                for the wave's height and for the time offset
    output      Part files hold the fluid of two-row slices across the span (each `sliceWidth`
                wide, centred on `sliceCentres` and clipped to the tank) shoreward of
                `trackFromX`, and all the water standing more than two spacings above still water
                there, for the break line in plan

Sean Bowman [11/09/2026]

'''

# Standard library imports
from __future__ import annotations

import math
from dataclasses import dataclass

# Third-party imports
import numpy as np

# Local imports
from dsphRidge import RidgeCase

reefToeX = 30.0
reefSlope = 0.0693
shelfDepth = 0.371
reefEndX = 52.0

@dataclass
class ReefCase(RidgeCase):

    '''The Surf Ranch reef turned across the span, for DualSPHysics; `k` is unused (pass 0).'''

    gaugeX: float = 20.0                 # [m], OpenFOAM's inlet
    gaugeY: float = 2.0                  # [m]
    endX: float = reefEndX               # [m]
    angle: float = 25.0                  # [deg], the toe's and the edge's turn from the y axis
    sliceCentres: tuple = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)  # [m]
    trackFromX: float = 28.0             # [m]
    triangleRise: float = 0.1            # [dp], the drawing triangles' largest rise in z

    @property
    def shift(self) -> float:
        '''tan(angle): the toe's advance in x per unit y.'''
        return math.tan(math.radians(self.angle))

    @property
    def edgeX(self) -> float:
        '''The shelf's edge at y = 0 [m].'''
        return reefToeX + (self.depth - shelfDepth) / reefSlope

    def bedElevation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        '''The reef's bed elevation over the flat bed [m].'''

        return np.clip((x - reefToeX - y * self.shift) * reefSlope, 0.0, self.depth - shelfDepth)

    def _grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        '''

        The bed on a grid sheared along the reef: lines of constant x' = x - y tan(angle) at the
        toe, the edge and every 0.25 m, wide enough that every y of the band covers the tank's x
        extent; y every 0.25 m (the bed is planar inside each cell, so y needs no finer lines).

        '''

        s, band = self.spacing, self.layers * self.spacing
        x0, x1 = self.paddleX - band - 2.0 * s, self.endX + band + 2.0 * s
        y0, y1 = -band - 2.0 * s, self.width + band + 2.0 * s
        u0, u1 = x0 - max(y1, 0.0) * self.shift, x1 - min(y0, 0.0) * self.shift
        # On the face each triangle rises at most `triangleRise` spacings: GenCase keeps the lattice
        # points near a triangle by its extent in z, so a tall triangle keeps points above the bed
        faceStep = min(0.25, self.triangleRise * s / reefSlope)
        faceLines = np.linspace(reefToeX, self.edgeX, int(np.ceil((self.edgeX - reefToeX) / faceStep)) + 1)
        flat = np.arange(u0, u1, 0.25)
        flat = flat[(flat < reefToeX - 0.5 * faceStep) | (flat > self.edgeX + 0.5 * faceStep)]
        us = np.unique(np.concatenate([flat, faceLines, [u1]]))
        ys = np.unique(np.concatenate([np.arange(y0, y1, 0.25), [y1]]))
        U, Y = np.meshgrid(us, ys, indexing = 'ij')
        X = U + Y * self.shift
        return X, Y, self.bedElevation(X, Y)

    def sliceBounds(self) -> list[tuple[float, float]]:

        '''The y range of each saved slice: `sliceWidth` (two spacings by default) centred on each centre, clipped to the tank.'''

        width = self.sliceWidth if self.sliceWidth is not None else 2.0 * self.spacing
        bounds = []
        for centre in self.sliceCentres:
            low = min(max(centre - 0.5 * width, 0.0), self.width - width)
            bounds.append((low, low + width))
        return bounds

    def outputFilters(self, xLow: float, xHigh: float, top: float) -> str:

        '''The `filterpos` entries: the slices across the span and the water above still water, shoreward of `trackFromX`.'''

        s, d = self.spacing, self.depth
        entries = [f'''                <filterpos operation="add">
                    <posmin x="{self.trackFromX:.6f}" y="{low:.8f}" z="-1" />
                    <posmax x="{xHigh + 1.0:.6f}" y="{high:.8f}" z="{top + 1.0:.6f}" />
                </filterpos>''' for low, high in self.sliceBounds()]
        entries.append(f'''                <filterpos operation="add">
                    <posmin x="{self.trackFromX:.6f}" y="-1" z="{d + 2.0 * s:.8f}" />
                    <posmax x="{xHigh + 1.0:.6f}" y="{self.width + 1.0:.6f}" z="{top + 1.0:.6f}" />
                </filterpos>''')
        return '\n'.join(entries)
