
# -- DualSPHysics Case Writer: the Guyenne and Grilli Ridge (3D) -- #

'''

Writes the GenCase XML and bed geometry for a 3D DualSPHysics run of the sloping ridge of Guyenne
and Grilli (2006), the case of phase 2 in planning/gpuSphRoadmap.md:

    z_b / h0 = (x - 5.225) sech^2(k y) / 15 for x > 5.225, flat offshore

in pySurf's coordinates (still water at z = h0), with a wall at x = 19 h0. Half the tank, 0 <= y <= 4
h0, as the roadmap sets it:

    bed         a band of boundary particles under the ridge (mk 0): for each layer m the bed
                surface moved (m + 0.6) spacings down its normal (band<m>.stl), drawn in triangles
                that rise at most a tenth of a spacing, so the lattice sits half a spacing off the
                flat bed as in the 2D cases and no boundary particle lies above the sloping bed
    walls       the symmetry plane y = 0 (mk 1), the end wall at x = 19 h0 (mk 2) and the tank's
                wall at y = 4 h0 (mk 3); `layers` spacings thick, outside the water
    normals     mDBC's normals for the bed from the bed surface itself (bed.vtk, triangles), which
                puts the wall half a spacing above the band's top row; for the walls and the paddle
                from their planes (GenCase computes normals from the particles only in 2D)
    paddle      a piston (mk 10) across the tank with its face at `paddleX`, moved to the nearest
                lattice plane, driven by `piston_solitary` (Goring 1978), or held still with
                `paddleMoves` False
    water       filled from a seed point up to still water; the fill box reaches a quarter spacing
                past the walls' planes, since GenCase drops the lattice points half a spacing inside
                a box's lower faces
    gauge       a free-surface gauge ('Crest') on the line x = `gaugeX`, y = `gaugeY`, every 0.002 s:
                the starting crest position, where the wave's height is read and its passage sets
                the time offset
    output      Part files every `coarseStep`, and every `trackStep` inside `trackWindow`, holding
                only the fluid within `sliceWidth` of y = 0 (two rows by default: the centre line,
                for the breaking point) and a layer `layerHalfThickness` either side of z = 0.8 h0 (z = -0.2 h0 below
                still water) over 13.5 <= x <= 19.5 h0 and y <= 2.2 h0 (Guyenne and Grilli's
                velocity grid)

Version 5.4 has no symmetry plane, and its mDBC slip mode applies to every wall: a free-slip plane at
y = 0 (`slipMode` 3) makes the bed and the other walls free-slip too. With the normals from the
geometry, free slip holds the still water level within 0.003 h0 at 16 ppd. In 2D, with GenCase's
normals from the particles, it fails: the Surf Ranch tank lost 62 percent of its water by 10 s, and
with mDBC's no-penetration option its still water level fell 0.09 h0 in 4 s. The 2D runs therefore
keep slip mode 1, and the slip mode's effect on them is not measured. The scheme otherwise follows
phase 1.5d: Wendland kernel at h = 1.5 dp, symplectic time stepping, Fourtakas density diffusion at
0.1, artificial viscosity at alpha 0.01, and mDBC walls.

Sean Bowman [11/09/2026]

'''

# Standard library imports
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

# Third-party imports
import numpy as np

ridgeToeX = 5.225
ridgeSlope = 1.0 / 15.0
tankEndX = 19.0

@dataclass
class RidgeCase:

    '''One half-tank ridge case for DualSPHysics, in pySurf's coordinates (bed at z = 0 offshore).'''

    name: str
    k: float                             # [1/h0], the ridge's transverse modulation
    paddleHeight: float                  # [m], the generator's target wave height
    spacing: float                       # [m], dp
    paddleX: float                       # [m], the paddle face at rest
    endTime: float                       # [s]
    depth: float = 1.0                   # [m]
    gaugeX: float = 5.785                # [m], the starting crest position
    gaugeY: float = 1.0                  # [m]
    width: float = 4.0                   # [m], the half tank; a narrow slab for tuning the paddle
    trackWindow: tuple[float, float] | None = None
    trackStep: float = 0.01 * math.sqrt(1.0 / 9.80665)
    coarseStep: float = 0.1
    alpha: float = 0.01
    slipMode: int = 3
    smoothingRatio: float = 1.5
    layers: int = 4
    paddleMoves: bool = True
    sliceWidth: float | None = None      # [m], default 2 dp
    layerHalfThickness: float | None = None  # [m], default 1.5 dp
    endX: float = tankEndX               # [m], the end wall's plane
    triangleRise: float = 0.1            # [dp], the drawing triangles' largest rise in z

    def __post_init__(self) -> None:
        # The paddle's face on a lattice plane, midway between the piston's rows and the water's
        self.paddleX = round(self.paddleX / self.spacing) * self.spacing

    def bedElevation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        '''The ridge's bed elevation over the flat bed [m].'''

        return np.maximum(x - ridgeToeX, 0.0) * ridgeSlope / np.cosh(self.k * y)**2

    def _grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        '''

        The bed sampled for drawing, as the (x, y, z) of each vertex of a structured grid over the
        band's extent. GenCase keeps the lattice points near a drawn triangle by the triangle's
        extent in z, so a triangle that rises much more than a tenth of a spacing keeps points above
        the bed: at 32 ppd, triangles 0.25 m long on the 1/15 slope (0.53 spacings of rise) left
        1.3 percent of the band's top row up to 0.27 spacings above it. Each triangle here rises
        at most `triangleRise` spacings, half of it along x and half along y: x lines every 0.25 m
        offshore of the toe and at a fixed step up the slope; y lines at a step set by the largest
        spanwise slope of the bed at that y, up to 0.05 m.

        '''

        s, band = self.spacing, self.layers * self.spacing
        x0, x1 = self.paddleX - band - 2.0 * s, self.endX + band + 2.0 * s
        y0, y1 = -band - 2.0 * s, self.width + band + 2.0 * s
        rise = 0.5 * self.triangleRise * s
        slopeStep = rise / ridgeSlope
        offshore = np.arange(x0, ridgeToeX, 0.25)
        slope = np.linspace(ridgeToeX, x1, int(np.ceil((x1 - ridgeToeX) / slopeStep)) + 1)
        xs = np.unique(np.concatenate([offshore, slope]))
        # The spanwise slope is largest at the far end: (x1 - toe) / 15 * k * 2 sech^2(k y) |tanh(k y)|
        spanwise = lambda y: (x1 - ridgeToeX) * ridgeSlope * self.k * 2.0 * np.abs(np.tanh(self.k * y)) / np.cosh(self.k * y)**2
        ys, y = [y0], y0
        while y < y1:
            step = min(0.05, rise / max(spanwise(y), 1e-12))
            step = min(step, rise / max(spanwise(y + step), 1e-12), 0.05)
            y = min(y + step, y1)
            ys.append(y)
        X, Y = np.meshgrid(xs, np.array(ys), indexing = 'ij')
        return X, Y, self.bedElevation(X, Y)

    def _triangles(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        '''The grid's vertices (N, 3), and its triangles as vertex indices (M, 3), two per cell, counterclockwise from above.'''

        X, Y, z = self._grid()
        nx, ny = X.shape
        vertices = np.stack([X, Y, z], axis = -1).reshape(-1, 3)
        index = np.arange(nx * ny).reshape(nx, ny)
        a, b, c, e = index[:-1, :-1].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel(), index[:-1, 1:].ravel()
        return vertices, np.concatenate([np.column_stack([a, b, c]), np.column_stack([a, c, e])]), index

    def bedSurface(self) -> str:

        '''The bed surface itself as legacy VTK polydata (triangles), for mDBC's normals.'''

        vertices, triangles, _ = self._triangles()
        points = '\n'.join(f'{x:.6f} {y:.6f} {z:.6f}' for x, y, z in vertices)
        polygons = '\n'.join(f'3 {a} {b} {c}' for a, b, c in triangles)
        return '\n'.join(['# vtk DataFile Version 3.0', f'{self.name} bed', 'ASCII', 'DATASET POLYDATA',
                          f'POINTS {len(vertices)} float', points,
                          f'POLYGONS {len(triangles)} {4 * len(triangles)}', polygons]) + '\n'

    def bandStl(self, layer: int) -> bytes:

        '''

        One layer of the bed band as a binary STL (triangles, normals up): the bed surface lowered
        (layer + 0.6 + (|dz_b/dx| + |dz_b/dy|) / 2) spacings. GenCase keeps a lattice point when the
        surface crosses its cell, a cube one spacing wide, which reaches (1 + |dz_b/dx| +
        |dz_b/dy|) / 2 spacings above or below the point. On the flat bed the layers keep the rows
        at -0.5 to -(layers - 1/2) spacings, as in the 2D cases, and on a slope the top layer keeps
        no point above the bed. Lowered (layer + 0.6) spacings alone, the top layer kept points up
        to 0.07 spacings above the bed where it tilts across the span (at k = 0.5 its spanwise
        slope reaches 0.35); a layer may keep two points in a column, the lower inside the band.

        '''

        vertices, triangles, _ = self._triangles()
        x, y, eps = vertices[:, 0], vertices[:, 1], 1e-4
        gradX = (self.bedElevation(x + eps, y) - self.bedElevation(x - eps, y)) / (2.0 * eps)
        gradY = (self.bedElevation(x, y + eps) - self.bedElevation(x, y - eps)) / (2.0 * eps)
        lowered = vertices.copy()
        lowered[:, 2] -= (layer + 0.6 + 0.5 * (np.abs(gradX) + np.abs(gradY))) * self.spacing
        corners = lowered[triangles]
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        normals /= np.linalg.norm(normals, axis = 1)[:, None]
        records = np.zeros(len(triangles), dtype = [('normal', '<f4', 3), ('corners', '<f4', (3, 3)), ('attribute', '<u2')])
        records['normal'], records['corners'] = normals, corners
        return f'{self.name} bed'.encode().ljust(80, b' ') + np.uint32(len(triangles)).tobytes() + records.tobytes()

    def outputFilters(self, xLow: float, xHigh: float, top: float) -> str:

        '''The `filterpos` entries of the Part files: the centre-line slice and the velocity layer.'''

        s, d = self.spacing, self.depth
        sliceWidth = self.sliceWidth if self.sliceWidth is not None else 2.0 * s
        layerHalf = self.layerHalfThickness if self.layerHalfThickness is not None else 1.5 * s
        return f'''                <filterpos operation="add">
                    <posmin x="{xLow - 1.0:.6f}" y="-1" z="-1" />
                    <posmax x="{xHigh + 1.0:.6f}" y="{sliceWidth:.8f}" z="{top + 1.0:.6f}" />
                </filterpos>
                <filterpos operation="add">
                    <posmin x="13.5" y="-1" z="{0.8 * d - layerHalf:.8f}" />
                    <posmax x="19.5" y="2.2" z="{0.8 * d + layerHalf:.8f}" />
                </filterpos>'''

    def xml(self) -> str:

        '''The GenCase case definition; the bed band is read from band<m>.stl beside it.'''

        s, d, n, w = self.spacing, self.depth, self.layers, self.width
        band = n * s
        top = d + 1.5 * self.paddleHeight
        lowest = -0.5 * s - band - 1.5 * s
        endX = self.endX
        timeout = [f'<tout time="0" timeout="{self.coarseStep}" />']
        if self.trackWindow is not None:
            timeout += [f'<tout time="{self.trackWindow[0]:.6f}" timeout="{self.trackStep:.8f}" />',
                        f'<tout time="{self.trackWindow[1]:.6f}" timeout="{self.coarseStep}" />']
        timeoutXml = '\n'.join('                ' + entry for entry in timeout)
        paddleXml = f'''            <wavepaddles>
                <piston_solitary>
                    <mkbound value="10" />
                    <theory value="2" comment="Boussinesq (Goring 1978)" />
                    <start value="0" />
                    <depth value="{d:.6f}" />
                    <pistondir x="1" y="0" z="0" />
                    <waveheight value="{self.paddleHeight:.6f}" />
                </piston_solitary>
            </wavepaddles>
''' if self.paddleMoves else ''
        xLow, xHigh = self.paddleX - band, endX + band
        # The bed band: one surface per layer, the bed moved (m + 0.6) spacings down its normal
        # (`bandStl`). Moved only half a spacing, the top layer kept points up to 0.09 spacings
        # above the sloping bed, on the water's side of the wall
        bandXml = '\n'.join(f'                    <drawfilestl file="band{m}.stl" />' for m in range(n))
        # Each box's water-side face runs through its inner row: in full draw mode GenCase keeps the
        # lattice points within half a spacing of a face, and a face on the wall's plane would take
        # the water's first row as well. Each plane's normal points from its particles into the water
        planes = [(1, (0.0, 0.0, 0.0), (0, 1, 0)), (2, (endX, 0.0, 0.0), (-1, 0, 0)),
                  (3, (0.0, w, 0.0), (0, -1, 0)), (10, (self.paddleX, 0.0, 0.0), (1, 0, 0))]
        planesXml = '\n'.join(f'''            <norplane mkbound="{mk}">
                <point x="{p[0]:.6f}" y="{p[1]:.6f}" z="{p[2]:.6f}" />
                <normal x="{v[0]}" y="{v[1]}" z="{v[2]}" />
                <maxdisth v="2" />
                <clear v="true" />
            </norplane>''' for mk, p, v in planes)
        return f'''<?xml version="1.0" encoding="UTF-8" ?>
<!-- {self.name}: written by pySurf/dualSphysics/tools/{type(self).__module__}.py -->
<case>
    <casedef>
        <constantsdef>
            <gravity x="0" y="0" z="-9.80665" />
            <rhop0 value="1000" />
            <rhopgradient value="2" />
            <hswl value="{d:.6f}" auto="false" />
            <gamma value="7" />
            <speedsystem value="0" auto="true" />
            <coefsound value="20" />
            <speedsound value="0" auto="true" />
            <hdp value="{self.smoothingRatio}" />
            <cflnumber value="0.2" />
        </constantsdef>
        <mkconfig boundcount="240" fluidcount="9" />
        <geometry>
            <definition dp="{s:.8f}">
                <pointref x="{0.5 * s:.8f}" y="{0.5 * s:.8f}" z="{0.5 * s:.8f}" />
                <pointmin x="{xLow - 2 * s:.6f}" y="{-band - 2 * s:.6f}" z="{lowest:.6f}" />
                <pointmax x="{xHigh + 2 * s:.6f}" y="{w + band + 2 * s:.6f}" z="{top + 2 * s:.6f}" />
            </definition>
            <commands>
                <mainlist>
                    <setshapemode>actual | bound</setshapemode>
                    <setdrawmode mode="full" />
                    <setmkbound mk="0" />
                    <setdrawmode mode="face" />
{bandXml}
                    <setdrawmode mode="full" />
                    <setmkbound mk="1" />
                    <drawbox>
                        <boxfill>solid</boxfill>
                        <point x="{xLow:.6f}" y="{-band:.6f}" z="{lowest:.6f}" />
                        <size x="{xHigh - xLow:.6f}" y="{band - 0.5 * s:.6f}" z="{top - lowest:.6f}" />
                    </drawbox>
                    <setmkbound mk="3" />
                    <drawbox>
                        <boxfill>solid</boxfill>
                        <point x="{xLow:.6f}" y="{w + 0.5 * s:.6f}" z="{lowest:.6f}" />
                        <size x="{xHigh - xLow:.6f}" y="{band - 0.5 * s:.6f}" z="{top - lowest:.6f}" />
                    </drawbox>
                    <setmkbound mk="2" />
                    <drawbox>
                        <boxfill>solid</boxfill>
                        <point x="{endX + 0.5 * s:.6f}" y="{0.5 * s:.6f}" z="{lowest:.6f}" />
                        <size x="{band - 0.5 * s:.6f}" y="{w - s:.6f}" z="{top - lowest:.6f}" />
                    </drawbox>
                    <setmkbound mk="10" />
                    <drawbox>
                        <boxfill>solid</boxfill>
                        <point x="{xLow:.6f}" y="{0.5 * s:.6f}" z="{lowest:.6f}" />
                        <size x="{band - 0.5 * s:.6f}" y="{w - s:.6f}" z="{top - lowest:.6f}" />
                    </drawbox>
                    <setmkfluid mk="0" />
                    <fillbox x="{self.gaugeX:.6f}" y="{0.5 * w:.6f}" z="{0.5 * d:.6f}">
                        <modefill>void</modefill>
                        <point x="{self.paddleX - 0.25 * s:.6f}" y="{-0.25 * s:.6f}" z="{lowest:.6f}" />
                        <size x="{endX - self.paddleX + 0.5 * s:.6f}" y="{w + 0.5 * s:.6f}" z="{d - lowest:.6f}" />
                    </fillbox>
                </mainlist>
            </commands>
        </geometry>
        <motion>
            <objreal ref="10">
                <begin mov="1" start="0" />
                <mvnull id="1" />
            </objreal>
        </motion>
        <normals>
            <norgeometry>
                <geometryfile file="../bed.vtk" comment="relative to the output directory" />
                <distanceh v="2" />
            </norgeometry>
{planesXml}
        </normals>
    </casedef>
    <execution>
        <special>
{paddleXml}            <gauges>
                <default>
                    <output value="true" />
                    <computedt value="0.002" />
                    <outputdt value="0.002" />
                </default>
                <swl name="Crest">
                    <pointdp coefdp="0.5" />
                    <point0 x="{self.gaugeX:.6f}" y="{self.gaugeY:.6f}" z="{0.5 * d:.6f}" />
                    <point2 x="{self.gaugeX:.6f}" y="{self.gaugeY:.6f}" z="{top:.6f}" />
                </swl>
            </gauges>
            <outputparts preselection="none">
{self.outputFilters(xLow, xHigh, top)}
                <filtertype operation="confirm">
                    <type v="fluid" />
                </filtertype>
            </outputparts>
            <timeout>
{timeoutXml}
            </timeout>
        </special>
        <parameters>
            <parameter key="SavePosDouble" value="0" />
            <parameter key="Boundary" value="2" comment="mDBC" />
            <parameter key="SlipMode" value="{self.slipMode}" comment="mDBC: 1 DBC velocity, 3 free slip" />
            <parameter key="NoPenetration" value="0" />
            <parameter key="StepAlgorithm" value="2" comment="Symplectic" />
            <parameter key="Kernel" value="2" comment="Wendland" />
            <parameter key="ViscoTreatment" value="1" comment="Artificial" />
            <parameter key="Visco" value="{self.alpha}" />
            <parameter key="ViscoBoundFactor" value="1" />
            <parameter key="DensityDT" value="2" comment="Fourtakas" />
            <parameter key="DensityDTvalue" value="0.1" />
            <parameter key="Shifting" value="0" />
            <parameter key="CoefDtMin" value="0.05" />
            <parameter key="DtIni" value="0" />
            <parameter key="DtMin" value="0" />
            <parameter key="DtAllParticles" value="0" />
            <parameter key="TimeMax" value="{self.endTime:.6f}" />
            <parameter key="TimeOut" value="{self.coarseStep}" />
            <parameter key="PartsOutMax" value="1" />
            <parameter key="RhopOutMin" value="700" />
            <parameter key="RhopOutMax" value="1300" />
            <simulationdomain>
                <posmin x="default" y="default" z="default" />
                <posmax x="default" y="default" z="default + 50%" />
            </simulationdomain>
        </parameters>
    </execution>
</case>
'''

    def write(self, caseDir: str | Path) -> Path:

        '''Write <name>_Def.xml, band<m>.stl for each layer and bed.vtk into caseDir; returns the XML's path.'''

        caseDir = Path(caseDir)
        caseDir.mkdir(parents = True, exist_ok = True)
        for layer in range(self.layers):
            (caseDir / f'band{layer}.stl').write_bytes(self.bandStl(layer))
        (caseDir / 'bed.vtk').write_text(self.bedSurface(), encoding = 'utf-8', newline = '\n')
        path = caseDir / f'{self.name}_Def.xml'
        path.write_text(self.xml(), encoding = 'utf-8', newline = '\n')
        return path
