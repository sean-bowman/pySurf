
# -- DualSPHysics Case Writer: a Solitary Wave over a pySurf Bed (2D) -- #

'''

Writes the GenCase XML for a 2D DualSPHysics run of one of pySurf's solitary-wave cases, built from
pySurf's own `Bathymetry` so both codes see the same bed.

    bed         the pySurf bed line, extended flat offshore to the paddle, drawn as a band of
                boundary particles `layers` spacings thick under the line (mk 0)
    paddle      a piston `layers` spacings thick (mk 10) driven by DualSPHysics's solitary-wave
                generator (`piston_solitary`, Goring's 1978 theory), its face at `paddleX`; with
                `paddleMoves` False it stays put, and the tank starts exactly as the moving case
                does (the still-water reference of an energy budget)
    end wall    where the bed ends under still water, a vertical wall of the same thickness
    water       filled from a seed point up to still water, bounded by the bed and the walls
    gauge       a free-surface gauge ('Crest') on the vertical line x = `gaugeX`, every 0.002 s:
                pySurf's starting crest position, where the wave's height is read and its passage
                sets the time offset between the two codes
    output      Part files every `coarseStep`, and every `trackStep` inside `trackWindow` (the
                interval pySurf's breaking tracker reads at)

The particle lattice sits half a spacing off the bed line, as pySurf's does: the boundary particles
lie at z = (k + 1/2) dp, so the bed band's top layer is half a spacing under a flat bed and the water's
bottom layer half a spacing over it. The scheme is DualSPHysics's own, fixed in phase 1.5d of
planning/gpuSphRoadmap.md: Wendland kernel at h = 1.5 dp, symplectic time stepping, Fourtakas density
diffusion at 0.1, mDBC walls with normals from the boundary particles (`slipMode` 1, the boundary
velocity of plain DBC; 3 makes every wall free-slip), and either artificial viscosity (alpha) or
laminar plus sub-particle-scale viscosity (water's 1e-6 m^2/s).

Sean Bowman [11/08/2026]

'''

# Standard library imports
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Third-party imports
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

# Local imports
from pysurf.bathymetry import Bathymetry

@dataclass
class SolitaryCase:

    '''One 2D solitary-wave case for DualSPHysics, in pySurf's coordinates (x along the tank, z up, bed at z = 0 offshore).'''

    name: str
    bathymetry: Bathymetry
    depth: float                         # [m]
    paddleHeight: float                  # [m], the generator's target wave height
    gaugeX: float                        # [m], pySurf's starting crest position
    spacing: float                       # [m], dp
    paddleX: float                       # [m], the paddle face at rest
    endTime: float                       # [s]
    trackWindow: tuple[float, float] | None = None
    trackStep: float = 0.01 * np.sqrt(1.0 / 9.80665)
    coarseStep: float = 0.1
    viscosity: str = 'artificial'
    alpha: float = 0.01
    smoothingRatio: float = 1.5
    layers: int = 4
    paddleMoves: bool = True
    slipMode: int = 1                    # mDBC: 1 the boundary velocity of plain DBC, 3 free slip
    noPenetration: bool = False          # mDBC's no-penetration option, for slip modes 2 and 3

    def _bedLine(self) -> np.ndarray:

        '''The bed line from behind the paddle to the bed's end: pySurf's vertices, flat offshore of its start.'''

        bed, s = self.bathymetry, self.spacing
        left = self.paddleX - (self.layers + 2) * s
        inside = bed.x[(bed.x > left) & (bed.x <= bed.endX)]
        x = np.concatenate([[left], inside])
        if x[-1] < bed.endX:
            x = np.append(x, bed.endX)
        z = np.where(x < bed.startX, float(bed.elevation(np.array([bed.startX]))[0]), bed.elevation(np.clip(x, bed.startX, bed.endX)))
        return np.column_stack([x, z])

    @property
    def endWall(self) -> bool:
        '''Whether the bed ends under still water, so the tank needs a wall there.'''
        return float(self.bathymetry.elevation(np.array([self.bathymetry.endX]))[0]) < self.depth

    def xml(self) -> str:

        '''The GenCase case definition.'''

        s, d, n = self.spacing, self.depth, self.layers
        line = self._bedLine()
        top = d + 3.0 * self.paddleHeight
        band = n * s
        # Bed band: the bed line lowered half a spacing, then the same line `band` lower, back to the
        # start. GenCase keeps lattice points within half a spacing of a polygon's edge, so on the bed
        # line itself the top row would sit half a spacing above the bed; lowered, the top row lies
        # half a spacing under it and mDBC's wall, half a spacing above the outer row, falls on it
        upper = line - np.array([0.0, 0.5 * s])
        polygon = np.vstack([upper, (upper - np.array([0.0, band]))[::-1]])
        polygonXml = '\n'.join(f'                        <point x="{x:.6f}" y="0" z="{z:.6f}" />' for x, z in polygon)
        endX = float(line[-1, 0])
        lowest = float(line[:, 1].min()) - band - 1.5 * s
        endWall = (f'''                    <drawbox>
                        <boxfill>solid</boxfill>
                        <point x="{endX:.6f}" y="-1" z="{float(line[-1, 1]) - band:.6f}" />
                        <size x="{band:.6f}" y="2" z="{top - float(line[-1, 1]) + band:.6f}" />
                    </drawbox>''' if self.endWall else '')
        # Water: seeded a spacing offshore of the gauge, half a spacing over the bed
        seedX = self.gaugeX
        seedZ = float(self.bathymetry.elevation(np.array([max(seedX, self.bathymetry.startX)]))[0]) + 0.5 * d
        if self.viscosity == 'artificial':
            viscosity = (f'<parameter key="ViscoTreatment" value="1" comment="Artificial" />\n'
                         f'            <parameter key="Visco" value="{self.alpha}" />')
        elif self.viscosity == 'sps':
            viscosity = ('<parameter key="ViscoTreatment" value="2" comment="Laminar+SPS" />\n'
                         '            <parameter key="Visco" value="1e-6" units_comment="m2/s" />')
        else:
            raise ValueError(f"Viscosity '{self.viscosity}' is not 'artificial' or 'sps'.")
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
        return f'''<?xml version="1.0" encoding="UTF-8" ?>
<!-- {self.name}: written by pySurf/dualSphysics/tools/dsphCase.py -->
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
                <pointref x="{0.5 * s:.8f}" y="0" z="{0.5 * s:.8f}" />
                <pointmin x="{float(line[0, 0]) - 2 * s:.6f}" y="0" z="{lowest:.6f}" />
                <pointmax x="{endX + band + 2 * s:.6f}" y="0" z="{top + 2 * s:.6f}" />
            </definition>
            <commands>
                <mainlist>
                    <setshapemode>actual | bound</setshapemode>
                    <setdrawmode mode="full" />
                    <setmkbound mk="0" />
                    <drawpolygon>
{polygonXml}
                    </drawpolygon>
{endWall}
                    <setmkbound mk="10" />
                    <drawbox>
                        <boxfill>solid</boxfill>
                        <point x="{self.paddleX - band:.6f}" y="-1" z="{lowest:.6f}" />
                        <size x="{band:.6f}" y="2" z="{top - lowest:.6f}" />
                    </drawbox>
                    <setmkfluid mk="0" />
                    <fillbox x="{seedX:.6f}" y="0" z="{seedZ:.6f}">
                        <modefill>void</modefill>
                        <point x="{self.paddleX:.6f}" y="-1" z="{lowest:.6f}" />
                        <size x="{endX - self.paddleX:.6f}" y="2" z="{d - lowest:.6f}" />
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
            <norparts mkbound="0,10">
                <maxdisth v="2.0" />
            </norparts>
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
                    <point0 x="{self.gaugeX:.6f}" y="0" z="{0.5 * d:.6f}" />
                    <point2 x="{self.gaugeX:.6f}" y="0" z="{top:.6f}" />
                </swl>
            </gauges>
            <timeout>
{timeoutXml}
            </timeout>
        </special>
        <parameters>
            <parameter key="SavePosDouble" value="0" />
            <parameter key="Boundary" value="2" comment="mDBC" />
            <parameter key="SlipMode" value="{self.slipMode}" comment="mDBC: 1 DBC velocity, 3 free slip" />
            <parameter key="NoPenetration" value="{int(self.noPenetration)}" />
            <parameter key="StepAlgorithm" value="2" comment="Symplectic" />
            <parameter key="Kernel" value="2" comment="Wendland" />
            {viscosity}
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

        '''Write <name>_Def.xml into caseDir; returns its path.'''

        caseDir = Path(caseDir)
        caseDir.mkdir(parents = True, exist_ok = True)
        path = caseDir / f'{self.name}_Def.xml'
        path.write_text(self.xml(), encoding = 'utf-8', newline = '\n')
        return path
