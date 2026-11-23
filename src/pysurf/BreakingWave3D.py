
# -- pySurf: A Solitary Wave Breaking over a 3D Bed -- #

'''

A solitary wave breaking over a bed of any shape, run in SPH on its own: `BreakingWave`'s
low-dissipation scheme in a rectangular tank over a height field.

    tank        the bed (`bathymetry.HeightField`) with free-slip walls on its bounds
                (`sphBoundaries.HeightFieldBoundary`); for a wave arriving at an angle, the bed
                turned so the wave travels along +x (`direction`, `HeightField.rotated`), its side
                walls then running along the wave
    start       the exact potential flow solitary wave of height H in water `depth` deep, its crest
                at `crestX` and straight along y (`waveMakers.exactSolitaryWaveFluid3D`), with its
                pressure, in columns half a spacing clear of the bed (`bedFitted`) or on the
                square lattice, each particle moved off its place at random (`startJitter`) or
                not; each boundary particle starts with the pressure of the water just inside it
    scheme      `BreakingWave`'s 'deltaSph': Wendland C2 at h = 3 spacings, hydrostatic delta-SPH
                density diffusion, the kernel gradient correction in the continuity equation, wall
                pressure extrapolated from the water, alpha 0.01 with the limited artificial
                viscosity, fluid density held at 0.9 rho0 or above; every wall free-slip
    sections    vertical planes y = constant on which a `waveBreaking.BreakingTracker` reads the
                breaking point from the smoothed surface ('contour'): the fluid within
                `sectionRows` / 2 spacings of the plane, each particle standing for its volume over
                the rows' width, s^3 / (rows s) of the plane's area unsplit, so the plane holds as
                many particles per area as a 2D run
    refinement  optionally, the fluid split (`WcsphSolver.refine`, keeping h) in a zone that
                follows the crest: the tank is cut across y into strips, each strip's crest is
                its highest fluid particle, and the zone reaches behind and ahead of that crest
                along x, from a floor over still water up. Daughters stay split when the zone
                moves on
    reach       `offshoreReach` places the crest a half length offshore of where the bed starts to
                shoal and the tank's offshore wall a half length behind it, so the wave crosses
                as little water as it can before it breaks
    snapshots   the whole fluid's positions, velocities and masses at `recordTimes`, and each
                section's fluid every `frameInterval`
    checkpoint  the run's whole state between calls to `run` (`saveCheckpoint`), which `restore`
                takes up in a new process, so a long run can stop and continue as if it had not

`velocity` gives the flow at any points at a snapshot's time: the Shepard-normalized Wendland
average of the fluid around each point within the kernel's support, each particle weighted by its
volume.

A side wall disturbs the flow beside it. Over a flat bed at 32 particles per depth, 0.3 s after
the start of a solitary wave of H/d = 0.3 between walls 1 d apart, the top of the particle row next
to each wall stands 13 percent lower at the crest than mid-tank, the second row 7 percent lower,
the third 1 percent, and the rows beyond within 0.3 percent. Still water between the same walls
settles alike beside them and mid-tank (at 16 particles per depth). The likely cause, not yet
confirmed, is that the wall's particles carry no velocity of their own, so the continuity sum of a
particle sliding along the wall misses part of the divergence the wall's side of its kernel would
carry. A plane of symmetry modeled as a wall would carry the same error onto a section measured on
it, so the model has none: a symmetric case runs in the full tank.

Run on the GPU backend: with the height-field bed the device enforces the boundary, so the state
stays on the device between the trackers' readings. Gravity is -z, x is the direction of travel.

References:
-----------
Guyenne, P. & Grilli, S.T. (2006) -- Numerical study of three-dimensional overturning waves in
    shallow water, J. Fluid Mech. 547, 361-388.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

The model's claim is the breaking point, at 32 particles per depth with the crest's water split
(below). The overturn's shape in 3D is not validated. After breaking the scheme dissipates too
little: Mostert and Deike's energy loss and the Oregon State reef flat both fail in 2D
(`pysurf.BreakingWave`). Bores and splashes are therefore unvalidated, and probably high.

The scheme is `BreakingWave`'s, whose 2D validation status that module states. The 3D runs below
are in validation/gpuSph (ridge.py, obliqueReef.py, extrudedReef.py, shoalReef.py). In 3D the
model is compared with the sloping ridge of Guyenne and Grilli (2006), with the criteria of
tests/testValidationRidgeOverturn.py, run at 32 particles per depth in the full tank outside the
test suite (5 to 7 GPU hours a ridge).

From bed-fitted columns, the default, the breaking point passes for four of the five ridges and
fails at k = 0.25 on position: time +1.2 to +4.7 percent (limit 5), crest position +0.07 to +0.53
h0 (limit 0.5, which k = 0.25 exceeds), crest height -4.5 to -0.5 percent (limit 5). The velocity
at k = 0.25 passes: the largest horizontal speed at z = -0.2 h0 within 0.3 percent of 0.651
sqrt(g h0) (limit 10) and the largest spanwise velocity 0.085 of the largest streamwise one
against 0.085 (+0.1 percent, limit 30). The columns' rows stay ordered ahead of the wave and
buckle under its front, and the face carries their ends. At k = 0.50 the face slope jumps from
+0.14 to -0.62 within 0.04 sqrt(h0/g) where the top row ends, so that ridge's breaking point, 0.31
sqrt(h0/g) earlier than the lattice start's, is set by the start's arrangement.

From the square lattice the breaking point passes for all five ridges, k = 0.10 to 0.50: time +3.2
to +4.8 percent, crest position +0.35 to +0.47 h0, crest height -0.6 to +1.5 percent. The velocity
at k = 0.25 fails: the speed +4.1 percent and the spanwise ratio 0.125 (+47 percent). That start
leaves the lowest particles 0.5 to 1.5 spacings above a slope; they fall into the steps, and the
disordered layer they leave on the bed carries the spanwise excess. Still water from it never
settles; from bed-fitted columns it does.

From either start the crest arrives on time, within 0.15 h0 of the reference position at its
breaking time and within 2 percent in height (except k = 0.50 from bed-fitted columns, -4.4), and
the face turns vertical late: 0.10 to 0.37 sqrt(h0/g) from bed-fitted columns, 0.25 to 0.41 from
the lattice. Time and position therefore sit near the edges of their bands. The lag is the 2D
scheme's at this resolution: the ridge's centre-line section run in 2D from bed-fitted columns
lags Guyenne and Grilli's 2D result by 0.372 and 0.101 sqrt(h0/g) at 32 and 64 particles per depth
(from the lattice 0.362, 0.132 and 0.041 at 32, 64 and 128). Less that 2D lag, the breaking
point's shift from 2D to 3D matches theirs for every k from the lattice, within 0.12 sqrt(h0/g)
in time and 0.1 h0 in position. From bed-fitted columns it matches within 0.02 sqrt(h0/g) and 0.15
h0 at k = 0.25 and 0.34, and comes 0.11 to 0.27 sqrt(h0/g) early at k = 0.10, 0.17 and 0.50. No
run reads the jet's touchdown. The bed-fitted runs at k = 0.25 and 0.34 stop on a pocket closing
inside the curling lip, about 0.6 h0 above the water ahead, which the touchdown test takes for the
tube; each holds one vertical run, so the breaking point is the same either way. The others run on
until the face runs into water too shallow to read.

With the crest's water split, the breaking point at k = 0.25 passes (phase 4's exit gate, run
outside the test suite, 27.9 GPU hours). From bed-fitted columns at 32 particles per depth, split
once with h kept from 1 h0 behind each strip's crest to 1.5 ahead and from 0.4 h0 under still
water up, from x = 12 h0, merging 1.5 h0 behind: breaking time +2.9 percent, crest position +0.39
h0 and crest height -0.2 percent (unsplit +4.7, +0.53 and -1.9); speed -0.6 percent and spanwise
ratio +1.0 percent. That run reads no touchdown: the jet lands against the tank's end wall after
the face reader has turned from the overhang. In 2D the same zone fails the Surf Ranch overturn
(`pysurf.BreakingWave`).
tests/testBreakingWave3D.py checks the build: over a flat bed the start is the 2D exact wave in
every lattice row, with the 2D rule's boundary pressures; a section reads its own two rows; and
after a short run on the GPU the flow mid-depth under the crest lies within 5 percent of the exact
wave's (2 percent at 8 and 16 particles per depth), with or without the crest's water split; that
a wave arriving at an angle meets the bed turned so it travels along +x; that `offshoreReach`
leaves the first-order wave's tails at 1 percent of its height at the shoaling bed and the wall;
and that a run restored from a checkpoint on the GPU, its crest's water split and partly merged,
ends bit for bit where the run that never stopped ends.

A wave arriving at an angle is checked against the closed form for an endless oblique reef, run
outside the test suite. There the flow is a pattern travelling along the contours, so the crest
crosses the span at c0 / tan of the angle. The test bed is the Surf Ranch profile turned 30 degrees,
at 16 particles per depth. In a tank 16 h0 across, the crest's passage at each distance from 2 h0
before the toe to 11 h0 past it crosses 14 sections within 1.7 percent of c0 / tan 30; in a tank
8 h0 across, within 3.5 percent. The breaking point does not hold its distance past the toe: the
break crosses 9 to 11 percent slower than c0 / tan 30 in the wide tank and 15 to 21 percent slower
in the narrow one. The free-slip walls mirror the bed into a chevron reef, and the wall at its point
moves the breaking point over the first 3 h0. In the interior the breaking point creeps shoreward
by an amount the runs do not split between the incident wave's numerical decay and the far wall.
That is a limit of a tank between free-slip walls; the closed form covers the crest only.

The Surf Ranch reef turned 25 degrees, between free-slip walls 4 h0 apart, is cross-checked against
DualSPHysics's run of the same tank, not validated: no measurement exists for it. At 32 particles
per depth with the crest's water split (32 GPU hours, 6.6 million particles by the end), the seven
inner sections break within 0.04 s of DualSPHysics's, 0.15 to 0.42 h0 farther past the shelf edge
and 8 to 11 percent higher. The break crosses the span at 10.31 m/s against its 9.90. The jets land
0.03 to 0.13 s after DualSPHysics's; in 2D pySurf's landed 0.55 s before. The runs do not say which
code departs from 2D in this tank.

The refinement zone in 3D is cross-checked against the 2D model, not validated: no measurement of
a 3D overturn's lip exists. On gate 1's Surf Ranch reef with the bed uniform across y (the exact
solitary wave of H/h0 = 0.6, bed-fitted columns, 32 particles per depth, the crest's water split
once from the toe), a tank 1 h0 wide read on its middle section matches the 2D run split the same
way within half the gates' tolerances, run outside the test suite (8.5 GPU hours): breaking time
-0.35 percent, touchdown time -0.46 percent, Hb +1.2 percent, impact x -0.14 h0, H_I -0.5 percent,
A_O/H^2 +1.8 percent, A_J/H^2 -0.4 percent, theta_O -0.6 degrees. Unsplit, the pair matches on
breaking (time -0.25 percent, Hb +1.3 percent); both unsplit runs stop on a void that closes
inside the crest, so neither reads a jet. The tests check that the zone splits the water it
should, keeps the mass and the sections' water, merges the daughters behind each strip's crest,
and carries the exact wave. Kept split, the water the zone sweeps raises a step's cost to 3.2 times
the unsplit step by touchdown on that reef. Merging the daughters 1.5 h0 behind the crest holds it
near twice the unsplit step, leaves all eight readings of the split run within 0.005 percent and
removes 1.1×10⁻⁴ of the wave's energy (6.8 GPU hours against 8.5).

Sean Bowman [11/10/2026]

'''

# Standard library imports
from __future__ import annotations

import hashlib
import math
import os
import pickle
import time as timeModule
from dataclasses import dataclass, fields
from pathlib import Path

# Third-party imports
import numpy as np
from scipy.spatial import cKDTree

# Local imports
from pysurf import constants, waveTheory
from pysurf.bathymetry import HeightField
from pysurf.bemSolver import JetGeometry
from pysurf.BreakingWave import (checkMergeBehind, defaultJitter, defaultPacking, jitteredFaceArc, latticeFaceArc,
                                 overturnMeasures, refinementDefaults, refinementRules, touchdownJet)
from pysurf.errors import InvalidInputError, NumericalInstabilityError
from pysurf.sphBoundaries import HeightFieldBoundary
from pysurf.sphPacking import PackingResult, packPoints
from pysurf.sphParticles import ParticleSystem
from pysurf.sphRefinement import refinementLevels
from pysurf.sphSolver import SimulationConfig, WcsphSolver, taitDensity, taitEquationOfState
from pysurf.validation import applyRules, numericRule
from pysurf.waveBreaking import BreakingTracker
from pysurf.waveMakers import exactSolitaryBoundaryPressure3D, exactSolitaryWaveFluid3D

# `BreakingWave`'s crest-following zone, with the width in depths of the strips across y that each
# read their own crest
refinementDefaults3D = {**refinementDefaults, 'strip': 0.25}
refinementRules3D = refinementRules + (
    numericRule('strip', 'Refinement strip width across the tank', minimum = 0.0),
)

# Written into every checkpoint; `restore` refuses a file without it
checkpointFormat = 'pysurf.BreakingWave3D checkpoint 1'

def offshoreReach(field: HeightField, waveHeight: float, depth: float | None = None, share: float = 0.05,
                  fraction: float = 0.01) -> tuple[float, float]:

    '''

    Where to start the solitary wave and where the tank's offshore wall can stand, so that the
    wave crosses as little water as it can before it breaks: the crest a half length offshore of
    where the bed starts to shoal (`HeightField.shoalingStart`), so the wave's front meets the
    shoaling bed at `fraction` of its height, and the wall a further half length offshore, so its
    back leaves the wall at that fraction. The half length is the first-order wave's
    (`waveTheory.solitaryWaveHalfLength`); the exact wave decays faster, so its tails stand lower.

    Parameters:
    -----------
    field : HeightField
        The bed, turned so the wave travels along +x
    waveHeight : float
        Solitary wave height H [m]
    depth : float | None
        The offshore still water depth [m]; None for the deepest along the field's offshore edge
    share, fraction : float
        The rise that marks shoaling, as a share of the depth, and the tail height left at the
        shoaling bed and at the wall, as a fraction of H

    Returns:
    --------
    tuple : (wallX, crestX), the offshore wall's position and the crest's [m]

    Raises:
    -------
    InvalidInputError : If the bed does not shoal, or the wall would stand offshore of the field.

    '''

    depth = field.offshoreDepth() if depth is None else float(depth)
    shoal = field.shoalingStart(depth, share)
    if shoal is None:
        raise InvalidInputError('The bed never rises far enough to shoal the wave.', parameterName = 'field')
    halfLength = waveTheory.solitaryWaveHalfLength(float(waveHeight), depth, fraction)     # [m]
    crestX, wallX = shoal - halfLength, shoal - 2.0 * halfLength
    if wallX < field.startX - 1e-9:
        raise InvalidInputError(f'The tank needs its offshore wall at x = {wallX:.3f}, offshore of the bed, which starts '
                                f'at {field.startX:.3f}.', parameterName = 'field', value = field.bounds)

    return wallX, crestX

@dataclass
class SectionFrame:

    '''One section's fluid at an instant: positions [m] and velocities [m/s], shape (N, 3), and masses [kg].'''

    time: float
    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray | None = None

class BreakingWave3D:

    '''

    A solitary wave breaking over a height-field bed in 3D SPH.

    Parameters:
    -----------
    field : HeightField
        The bed, its still water level and its bounds, which are the tank's walls
    waveHeight : float
        Solitary wave height H [m]
    crestX : float
        Starting crest position [m]
    depth : float | None
        Still water depth of the starting wave [m]; None for the depth at (startX, startY)
    particlesPerDepth : int
        depth / particle spacing
    sections : tuple | None
        y of each plane a breaking tracker reads [m]; the first is the one `breakingPoint` and
        `stopAt` follow. None for the tank's middle
    sectionRows : int
        Particle rows a section takes: those within rows / 2 spacings of its plane
    searchFromX : float | None
        A crest offshore of this position cannot register as breaking [m]; None for `crestX`
    recordTimes : tuple
        Times at which the whole fluid's state is kept [s]
    frameInterval : float
        Simulated time between section frames [s]
    trackInterval : float
        Simulated time between the trackers' readings [sqrt(depth / g)]
    endTime : float | None
        Simulated time to run to [s]; None for the time the wave takes to reach the end wall
    wallTop : float | None
        Top of the tank's walls [m]; None for still water plus twice the wave height
    smoothingLengthRatio : float
        h / spacing
    alphaViscosity : float
        Artificial viscosity coefficient
    soundSpeedFactor : float
        c0 over the solitary wave's speed
    backend : str
        The solver's backend: 'gpu', or 'numpy' for small tests
    solverOptions : dict | None
        `SimulationConfig` fields that replace the scheme's
    bedFitted : bool
        Lay the starting wave in columns half a spacing clear of the bed
        (`sphParticles.bedFittedPoints`); False for the square lattice, whose lowest particles sit
        half a spacing to one and a half above a slope and fall into those steps
    startJitter : float | None
        Move each starting particle off its place by a seeded uniform random offset of up to this
        many spacings along x, y and z, reflected back into the water, with the exact flow at its
        final place (`waveMakers.exactSolitaryWaveFluid3D`); 0 for none. None takes
        `BreakingWave`'s for its 'deltaSph' scheme
    startSeed : int
        Seed of the start's random offsets
    startPacking : bool | None
        Relax the laid start before its flow and pressure are set, between the boundary's fixed
        particles and below the wave's surface (`sphPacking.packPoints`, as `BreakingWave`'s
        `startPacking`). None takes `BreakingWave`'s for its 'deltaSph' scheme
    faceArc : float | None
        The arc the sections' face slope is read over, in spacings (`waveBreaking.contourFace`);
        None for 2 when the start lies on its lattice and 6 when it is jittered
    refinement : dict | None
        Split the fluid in a zone that follows the crest (`WcsphSolver.refine`, keeping h), with
        `BreakingWave`'s settings: 'levels' (1 to 3), 'behind', 'ahead' and 'above' in depths,
        'fromX', the crest position the zone starts at [m], `searchFromX` by default, and
        'mergeBehind', the distance behind the crest past which daughters merge back, in depths
        (None keeps them split); and 'strip', the width across y of the strips that each read
        their own crest, in depths. Applied at every reading of the sections. A daughter keeps its
        mother's h and so meets about eight times her neighbors: on the ridge at 16 particles per
        depth, splitting once the starting crest's water above 0.15 depths over still water
        doubled the cost of a step, and the full depth over 4.5 depths of the tank's length
        quadrupled it. Daughters that stay split make the cost grow as the zone sweeps new water;
        merging them behind the crest bounds it. None for a uniform spacing
    direction : float
        The wave's direction of travel over `field`, counterclockwise from +x [deg]. The bed turns
        (`HeightField.rotated`) so the wave travels along +x, and `crestX`, `sections`,
        `searchFromX` and the refinement's 'fromX' are positions in the turned frame. The side
        walls then run along the wave, so their free slip holds exactly for it and only water the
        bed turns aside meets them
    region : tuple | None
        (startX, endX, startY, endY), the tank in the turned frame [m], needed when `direction` is
        not 0; with `direction` 0 it crops `field` (`HeightField.cropped`). None keeps `field`'s
        bounds
    pivot : tuple | None
        (x, y), the point of `field` the bed turns about [m]; None for the centre of its bounds
    sortInterval : int
        Reorder the fluid's state in the GPU hash grid's cell order (`WcsphSolver.sortParticles`)
        after every `sortInterval`-th breaking check, counted on the trackers' record so a run
        taken up from a checkpoint sorts at the same checks; 0 for never

    '''

    def __init__(self, field: HeightField, waveHeight: float, crestX: float, depth: float | None = None,
                 particlesPerDepth: int = 32, sections: tuple | None = None,
                 sectionRows: int = 2, searchFromX: float | None = None, recordTimes: tuple = (),
                 frameInterval: float = 0.1, trackInterval: float = 0.01, endTime: float | None = None,
                 wallTop: float | None = None, smoothingLengthRatio: float = 3.0, alphaViscosity: float = 0.01,
                 soundSpeedFactor: float = 10.0, backend: str = 'gpu', solverOptions: dict | None = None,
                 bedFitted: bool = True, startJitter: float | None = None, startSeed: int = 1,
                 faceArc: float | None = None, startPacking: bool | None = None,
                 refinement: dict | None = None, direction: float = 0.0, region: tuple | None = None,
                 pivot: tuple | None = None, sortInterval: int = 0) -> None:

        self.direction = float(direction)                   # [deg]
        if self.direction != 0.0:
            if region is None:
                raise InvalidInputError('A wave arriving at an angle needs the tank region in the turned frame.',
                                        parameterName = 'region', value = region)
            field = field.rotated(self.direction, region, pivot = pivot)
        elif region is not None:
            field = field.cropped(region)
        self.field = field
        self.waveHeight = float(waveHeight)                 # [m]
        self.crestX = float(crestX)                         # [m]
        self.depth = float(depth if depth is not None
                           else field.stillWaterLevel - field.elevation(field.startX, field.startY))  # [m]
        self.particlesPerDepth = int(particlesPerDepth)
        middle = 0.5 * (field.startY + field.endY)
        self.sections = tuple(float(y) for y in (sections if sections is not None else (middle,)))
        self.sectionRows = int(sectionRows)
        self.searchFromX = float(searchFromX if searchFromX is not None else crestX)    # [m]
        self.recordTimes = tuple(sorted(float(t) for t in recordTimes))                # [s]
        self.frameInterval = float(frameInterval)           # [s]
        self.trackInterval = float(trackInterval)           # [sqrt(d / g)]
        self.endTime = endTime                              # [s]
        self.wallTop = float(wallTop if wallTop is not None else field.stillWaterLevel + 2.0 * waveHeight)  # [m]
        self.smoothingLengthRatio = float(smoothingLengthRatio)
        self.alphaViscosity = float(alphaViscosity)
        self.soundSpeedFactor = float(soundSpeedFactor)
        self.backend = backend
        self.solverOptions = dict(solverOptions or {})
        self.bedFitted = bool(bedFitted)
        self.startJitter = float(startJitter if startJitter is not None else defaultJitter['deltaSph'])
        self.startSeed = int(startSeed)
        self.startPacking = bool(startPacking if startPacking is not None else defaultPacking['deltaSph'])
        self.faceArc = float(faceArc if faceArc is not None
                             else jitteredFaceArc if self.startJitter > 0.0 else latticeFaceArc)
        self.refinement = ({**refinementDefaults3D, **refinement} if refinement is not None else None)
        self.sortInterval = int(sortInterval)
        self._validateInputs()

        self.packing: PackingResult | None = None
        self.boundary: HeightFieldBoundary | None = None
        self.particles: ParticleSystem | None = None
        self.solver: WcsphSolver | None = None
        self.trackers: dict[float, BreakingTracker] = {}
        self.sectionFrames: dict[float, list[SectionFrame]] = {}
        self.snapshots: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        self.snapshotMasses: dict[float, np.ndarray] = {}
        self.wallClockSeconds = 0.0
        self.mergedGroups = 0
        self.mergedEnergy = 0.0                     # [J], kinetic energy the merges removed
        self._fluidMass = 0.0                       # [kg], of a particle that has not split

    # -- Private Helper Methods -- #
    def _validateInputs(self) -> None:

        '''

        Raises:
        -------
        InvalidInputError : If the wave, the resolution or a section does not fit the tank.

        '''

        field = self.field
        if not (0.0 < self.waveHeight < self.depth):
            raise InvalidInputError('The wave height must be positive and below the depth.', parameterName = 'waveHeight',
                                    value = self.waveHeight)
        if not (field.startX < self.crestX < field.endX):
            raise InvalidInputError('The crest must start inside the tank.', parameterName = 'crestX', value = self.crestX)
        if self.particlesPerDepth < 2 or self.sectionRows < 1:
            raise InvalidInputError('Particles per depth must be at least 2 and section rows at least 1.',
                                    parameterName = 'particlesPerDepth', value = self.particlesPerDepth)
        for y in self.sections:
            if not (field.startY - 1e-9 <= y <= field.endY + 1e-9):
                raise InvalidInputError('Every section must lie inside the tank.', parameterName = 'sections', value = y)
        if not (0.0 <= self.startJitter < 0.5):
            raise InvalidInputError('The start jitter must lie from 0 to below half a spacing.', parameterName = 'startJitter',
                                    value = self.startJitter)
        if not (1.0 <= self.faceArc <= 12.0):
            raise InvalidInputError('The face slope arc must lie from 1 to 12 spacings.', parameterName = 'faceArc',
                                    value = self.faceArc)
        if self.sortInterval < 0:
            raise InvalidInputError('The sort interval is a count of breaking checks, 0 or more.',
                                    parameterName = 'sortInterval', value = self.sortInterval)
        if self.refinement is not None:
            unknown = set(self.refinement) - set(refinementDefaults3D)
            if unknown:
                raise InvalidInputError(f'Unknown refinement settings: {sorted(unknown)}.', parameterName = 'refinement',
                                        value = self.refinement)
            applyRules(self.refinement, refinementRules3D)
            checkMergeBehind(self.refinement)
            if self.backend not in ('numpy', 'gpu'):
                raise InvalidInputError('Only the numpy and gpu backends split particles.', parameterName = 'refinement',
                                        value = self.refinement)

    def _sectionMask(self, y: float) -> np.ndarray:

        '''The fluid particles a section takes, as a mask over the fluid.'''

        p = self.particles
        across = p.positions[p.isFluid, 1]

        return np.abs(across - y) < 0.5 * self.sectionRows * self.spacing

    def _readSections(self) -> None:

        '''Feed each section's fluid, in its (x, z) plane, to its tracker.'''

        p, s = self.particles, self.spacing
        fluid = p.positions[p.isFluid]
        # Each particle stands for its volume over the section's width; a daughter for an eighth
        areas = p.masses[p.isFluid] / (self.solver.config.referenceDensity * self.sectionRows * s)
        for y, tracker in self.trackers.items():
            mask = self._sectionMask(y)
            tracker.update(self.solver.time, fluid[mask][:, [0, 2]], areas[mask])

    def _recordSections(self) -> None:

        '''Keep each section's fluid at this instant.'''

        p = self.particles
        positions, velocities, masses = p.positions[p.isFluid], p.velocities[p.isFluid], p.masses[p.isFluid]
        for y in self.sections:
            mask = self._sectionMask(y)
            self.sectionFrames[y].append(SectionFrame(self.solver.time, positions[mask].copy(), velocities[mask].copy(),
                                                      masses[mask].copy()))

    def _keepSnapshot(self, time: float) -> None:

        '''Keep the whole fluid's positions, velocities and masses at `time` [s].'''

        p = self.particles
        self.snapshots[time] = (p.positions[p.isFluid].copy(), p.velocities[p.isFluid].copy())
        self.snapshotMasses[time] = p.masses[p.isFluid].copy()

    def _stripCrests(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        '''

        Cut the tank across y into strips `refinement['strip']` depths wide and read each strip's
        crest as the x of its highest fluid particle. Returns each strip's crest [m], -inf where a
        strip holds no fluid, and the strip each given particle lies in.

        '''

        strip, count = self._strips(y)
        # Sorted by strip, then height: the last particle of each strip is its highest
        order = np.lexsort((z, strip))
        last = np.flatnonzero(np.diff(strip[order], append = count))
        crests = np.full(count, -np.inf)
        crests[strip[order][last]] = x[order][last]

        return crests, strip

    def _strips(self, y: np.ndarray) -> tuple[np.ndarray, int]:

        '''The strip across y each position lies in, and how many strips cut the tank.'''

        field = self.field
        width = self.refinement['strip'] * self.depth
        count = max(1, int(math.ceil((field.endY - field.startY) / width - 1e-9)))

        return np.clip(np.floor((y - field.startY) / width).astype(int), 0, count - 1), count

    def _refineCrest(self) -> int:

        '''

        Split the fluid in the zone that follows each strip's crest, a level at a time, until every
        particle in it has split `refinement['levels']` times, and merge the daughters more than
        `refinement['mergeBehind']` behind their strip's crest or in a strip with no water above
        the zone's floor. Returns how many particles split.

        '''

        settings, d = self.refinement, self.depth
        fromX = settings['fromX'] if settings['fromX'] is not None else self.searchFromX
        floor = self.field.stillWaterLevel + settings['above'] * d
        total = 0
        for _ in range(int(settings['levels'])):
            # Only water above the floor can lie in the zone, and a strip with any holds its crest there
            p = self.particles
            above = np.flatnonzero(p.isFluid & (p.positions[:, 2] > floor))
            x, y, z = p.positions[above].T
            crests, strip = self._stripCrests(x, y, z)
            crest = crests[strip]
            zone = (crest >= fromX) & (x > crest - settings['behind'] * d) & (x < crest + settings['ahead'] * d)
            if not zone.any():
                break
            mask = np.zeros(p.nParticles, dtype = bool)
            mask[above[zone]] = True
            split = self.solver.refine(mask & (refinementLevels(p, self._fluidMass) < settings['levels']))
            total += split
            if split == 0:
                break
        if settings['mergeBehind'] is not None:
            p = self.particles
            daughters = np.flatnonzero(p.isFluid & (refinementLevels(p, self._fluidMass) > 0))
            if daughters.size:
                above = p.isFluid & (p.positions[:, 2] > floor)
                crests, _ = self._stripCrests(*p.positions[above].T)
                strip, _ = self._strips(p.positions[daughters, 1])
                crest = crests[strip]
                behind = (p.positions[daughters, 0] < crest - settings['mergeBehind'] * d) | np.isneginf(crest)
                mask = np.zeros(p.nParticles, dtype = bool)
                mask[daughters[behind]] = True
                merged = self.solver.merge(mask, self._fluidMass)
                self.mergedGroups += merged.groups
                self.mergedEnergy += merged.removedEnergy

        return total

    def _checkpointSettings(self) -> dict:

        '''

        The settings a checkpoint must match: everything that shapes the tank, the bed (by a hash of
        its heights), the start and the scheme.

        '''

        field = self.field
        return {'bounds': tuple(float(v) for v in field.bounds), 'stillWaterLevel': float(field.stillWaterLevel),
                'bed': hashlib.sha256(np.ascontiguousarray(field.z, dtype = float).tobytes()).hexdigest(),
                'waveHeight': self.waveHeight, 'crestX': self.crestX, 'depth': self.depth,
                'particlesPerDepth': self.particlesPerDepth, 'sections': self.sections, 'sectionRows': self.sectionRows,
                'searchFromX': self.searchFromX, 'trackInterval': self.trackInterval, 'wallTop': self.wallTop,
                'smoothingLengthRatio': self.smoothingLengthRatio, 'alphaViscosity': self.alphaViscosity,
                'soundSpeedFactor': self.soundSpeedFactor, 'solverOptions': self.solverOptions,
                'bedFitted': self.bedFitted, 'startJitter': self.startJitter, 'startSeed': self.startSeed,
                'startPacking': self.startPacking, 'faceArc': self.faceArc, 'refinement': self.refinement,
                'direction': self.direction}

    # -- Public Methods -- #
    @property
    def spacing(self) -> float:
        '''Particle spacing [m].'''
        return self.depth / self.particlesPerDepth

    @property
    def boundaryLayers(self) -> int:
        '''Boundary particle layers: enough to fill the kernel's support, 2h, and never fewer than 3.'''
        return max(3, math.ceil(2.0 * self.smoothingLengthRatio - 1e-9))

    @property
    def smoothingLength(self) -> float:
        '''h [m].'''
        return self.smoothingLengthRatio * self.spacing

    @property
    def timeScale(self) -> float:
        '''sqrt(depth / g) [s].'''
        return math.sqrt(self.depth / constants.gravity)

    @property
    def tracker(self) -> BreakingTracker | None:
        '''The first section's tracker.'''
        return self.trackers.get(self.sections[0])

    @property
    def breakingPoint(self):
        '''Where the wave broke on the first section, or None.'''
        return self.tracker.breakingPoint if self.tracker is not None else None

    def build(self) -> None:

        '''Lay out the tank, the boundary, the starting wave, the solver and the trackers.'''

        field, d, s, H = self.field, self.depth, self.spacing, self.waveHeight
        rho, level, layers = constants.freshwaterDensity, field.stillWaterLevel, self.boundaryLayers

        self.boundary = HeightFieldBoundary(field, s, nLayers = layers, wallTop = self.wallTop)
        floor = self.boundary.latticeFloor
        pack = None
        if self.startPacking:
            fixed = self.boundary.positions()
            kernelType = self.solverOptions.get('kernelType', 'wendlandC2')

            def pack(points: np.ndarray, surface) -> np.ndarray:
                self.packing = packPoints(points, fixed, s, surface, lambda x, y: np.maximum(field.elevation(x, y), floor),
                                          smoothingLengthRatio = self.smoothingLengthRatio, kernelType = kernelType,
                                          backend = 'gpu' if self.backend == 'gpu' else 'numpy')
                return self.packing.positions
        fluidPositions, fluidVelocities, fluidPressures = exactSolitaryWaveFluid3D(field, H, d, self.crestX, s, floor,
                                                                                   bedFitted = self.bedFitted,
                                                                                   jitter = self.startJitter,
                                                                                   seed = self.startSeed, pack = pack)
        self.particles = ParticleSystem.assemble(fluidPositions, self.boundary.positions(), s, rho, fluidVelocities)
        self._fluidMass = float(self.particles.masses[0])
        self.mergedGroups, self.mergedEnergy = 0, 0.0

        # `BreakingWave`'s deltaSph scheme (see its build for the reasons behind each setting)
        schemeOptions = dict(kernelType = 'wendlandC2', densityFilterInterval = 0, densityDiffusion = 'hydrostatic',
                             xsphEpsilon = 0.0, wallPressure = 'extrapolated', continuityCorrection = True,
                             densityFloor = 0.9, viscosityLimiter = True)
        schemeOptions.update(self.solverOptions)
        soundSpeed = self.soundSpeedFactor * waveTheory.solitaryWaveSpeed(H, d)
        margin = (layers + 2.0) * s
        travel = (field.endX - self.crestX) / waveTheory.solitaryWaveSpeed(H, d)
        config = SimulationConfig(
            domainMin       = np.array([field.startX - margin, field.startY - margin, floor - margin]),
            domainMax       = np.array([field.endX + margin, field.endY + margin, self.wallTop]),
            particleSpacing = s,
            smoothingLengthRatio = self.smoothingLengthRatio,
            referenceDensity = rho,
            gravity         = np.array([0.0, 0.0, -constants.gravity]),
            endTime         = self.endTime if self.endTime is not None else travel,
            maxTimeStep     = 1.0,
            outputInterval  = self.frameInterval,
            dimensions      = 3,
            speedOfSound    = soundSpeed,
            alphaViscosity  = self.alphaViscosity,
            initialSurface  = lambda x: level + waveTheory.solitaryWaveProfile(x, H, d, self.crestX),
            backend         = self.backend,
            **schemeOptions,
        )
        self.solver = WcsphSolver(config, boundary = self.boundary)
        self.solver.initialize(self.particles)

        # The exact wave's pressure, carried by each particle's density
        p = self.particles
        wallPressures = exactSolitaryBoundaryPressure3D(field, p.positions[~p.isFluid], H, d, self.crestX, s,
                                                        constants.gravity)
        p.densities[p.isFluid] = taitDensity(rho * fluidPressures, rho, soundSpeed)
        p.densities[~p.isFluid] = taitDensity(rho * wallPressures, rho, soundSpeed)
        p.pressures = taitEquationOfState(p.densities, rho, soundSpeed)

        self.trackers = {y: BreakingTracker(field.section(y), s, faceReach = d, searchFromX = self.searchFromX,
                                            faceMeasure = 'contour', faceArc = self.faceArc) for y in self.sections}
        self.sectionFrames = {y: [] for y in self.sections}
        self.snapshots, self.snapshotMasses = {}, {}
        self._recordSections()
        if any(abs(t) < 1e-12 for t in self.recordTimes):
            self._keepSnapshot(0.0)

    def run(self, endTime: float | None = None, progress: bool = True, stopAt: str = 'end') -> None:

        '''

        Step the simulation to `endTime`, reading the sections and keeping the snapshots.

        The trackers read the sections every `trackInterval` sqrt(depth / g); the solver advances
        between those instants, and the record times, in one call, so on the GPU backend the state
        stays on the device in between.

        Parameters:
        -----------
        endTime : float | None
            Simulated time to stop at [s]; the configured end time by default
        progress : bool
            Print a line roughly every tenth of the run
        stopAt : str
            'end', or 'breaking' to stop once the first section's breaking point is final
            (`BreakingTracker.breakingFinal`: the jet has touched down)

        Raises:
        -------
        NumericalInstabilityError : If the time step collapses or a position stops being finite.

        '''

        if stopAt not in ('end', 'breaking'):
            raise ValueError(f"stopAt '{stopAt}' is not 'end' or 'breaking'.")
        if self.solver is None:
            self.build()

        solver, tracker = self.solver, self.tracker
        stopTime = endTime if endTime is not None else solver.config.endTime
        trackStep = self.trackInterval * self.timeScale
        nextTrack = solver.time + trackStep
        nextFrame = solver.time + self.frameInterval
        pending = [t for t in self.recordTimes if t > solver.time + 1e-12]
        nextReport = solver.time + stopTime / 10.0
        started = timeModule.perf_counter()

        if progress:
            print(f'BreakingWave3D: {self.particles.nFluid} fluid and {self.particles.nBoundary} boundary particles, '
                  f'running to {stopTime:.3f} s')

        # A hundred times the steps an interval takes at the acoustic time step: more means the step
        # has collapsed, which a diverging run does before its positions stop being finite
        config = solver.config
        stepLimit = max(1000, int(100.0 * min(trackStep, self.frameInterval) * solver.speedOfSound
                                  / (config.cflNumber * config.smoothingLength)))

        while solver.time < stopTime and not (stopAt == 'breaking' and tracker.breakingFinal):
            target = min([nextTrack, nextFrame, stopTime] + pending[:1])
            state = solver.advance(target, maxSteps = stepLimit)
            if solver.time < target:
                raise NumericalInstabilityError('The time step collapsed.', variableName = 'dt', operation = 'WCSPH step',
                                                context = {'time': solver.time, 'step': state.step, 'dt': state.dt})

            p = self.particles
            if solver.time >= nextTrack:
                self._readSections()
                if self.refinement is not None:
                    self._refineCrest()
                if self.sortInterval and len(tracker.times) % self.sortInterval == 0:
                    solver.sortParticles()
                nextTrack += trackStep
            if solver.time >= nextFrame:
                if not np.all(np.isfinite(p.positions)):
                    raise NumericalInstabilityError('Particle positions stopped being finite.', variableName = 'positions',
                                                    operation = 'WCSPH step', context = {'time': solver.time, 'step': state.step})
                self._recordSections()
                nextFrame += self.frameInterval
            while pending and solver.time >= pending[0] - 1e-12:
                self._keepSnapshot(pending.pop(0))
            if progress and solver.time >= nextReport:
                elapsed = timeModule.perf_counter() - started
                broken = ('breaking recorded' if tracker.breakingFinal
                          else 'breaking provisional' if tracker.breakingPoint else 'not broken')
                crest = tracker.crestHeight[-1] if tracker.crestHeight else float('nan')
                print(f'  t = {solver.time:6.3f} s  step {state.step:7d}  dt {state.dt:.2e} s  wall {elapsed:8.1f} s  '
                      f'crest {crest:.3f} m  {broken}', flush = True)
                nextReport += stopTime / 10.0

        self.wallClockSeconds += timeModule.perf_counter() - started

    def saveCheckpoint(self, path: str | Path) -> None:

        '''

        Write the run's state between calls to `run`: the particles (densities, pressures and the
        zone's split masses and lengths included), the solver's clock, the trackers' records, the
        merge tally and the wall clock, with the settings it was run with. `restore` from it
        continues the run as if it had not stopped. The file is a pickle written to a partial file
        first, which then replaces any earlier checkpoint at `path`, so a crash while writing leaves
        the earlier one whole. The section frames and snapshots stay out: a run that keeps them for
        long holds gigabytes, and a script that needs them writes them as it goes.

        Parameters:
        -----------
        path : str | Path
            The checkpoint file

        Raises:
        -------
        InvalidInputError : Before `build`, with no state to save.

        '''

        if self.solver is None:
            raise InvalidInputError('There is no state to save before build.', parameterName = 'path', value = str(path))
        p, path = self.particles, Path(path)
        state = {'format': checkpointFormat, 'settings': self._checkpointSettings(),
                 'particles': {field.name: getattr(p, field.name) for field in fields(p)},
                 'clock': self.solver.clock, 'trackers': self.trackers, 'fluidMass': self._fluidMass,
                 'mergedGroups': self.mergedGroups, 'mergedEnergy': self.mergedEnergy,
                 'wallClockSeconds': self.wallClockSeconds}
        partial = path.with_name(path.name + '.partial')
        with open(partial, 'wb') as handle:
            pickle.dump(state, handle, protocol = 5)
        os.replace(partial, path)

    def restore(self, path: str | Path) -> None:

        '''

        Take up a checkpoint `saveCheckpoint` wrote for a run with these settings: build the tank,
        its boundary and the solver, then put the saved particles, clock, trackers and tallies in
        place of the start. The section frames and snapshots start empty. Load only checkpoints
        this code wrote: the file is a pickle, and unpickling runs code.

        Parameters:
        -----------
        path : str | Path
            The checkpoint file

        Raises:
        -------
        InvalidInputError : If the file is not a checkpoint of this format, or a run with other
            settings wrote it.

        '''

        with open(path, 'rb') as handle:
            state = pickle.load(handle)
        if not isinstance(state, dict) or state.get('format') != checkpointFormat:
            raise InvalidInputError('The file is not a BreakingWave3D checkpoint of this format.', parameterName = 'path',
                                    value = str(path))
        if state['settings'] != self._checkpointSettings():
            raise InvalidInputError('A run with other settings wrote the checkpoint.', parameterName = 'path',
                                    value = str(path))
        self.build()
        self.particles = ParticleSystem(**state['particles'])
        self.solver.restore(self.particles, **state['clock'])
        self.trackers = state['trackers']
        self.sectionFrames = {y: [] for y in self.sections}
        self.snapshots, self.snapshotMasses = {}, {}
        self._fluidMass = state['fluidMass']
        self.mergedGroups, self.mergedEnergy = state['mergedGroups'], state['mergedEnergy']
        self.wallClockSeconds = state['wallClockSeconds']

    def velocity(self, points: np.ndarray, time: float) -> np.ndarray:

        '''

        The flow at points at a snapshot's time [m/s], shape (N, 3): the Shepard-normalized Wendland
        average, at the smoothing length, of the fluid within the kernel's support of each point,
        each particle weighted by its mass, which is its volume. NaN where no fluid lies within
        reach.

        Raises:
        -------
        KeyError : If no snapshot was kept at `time`.

        '''

        key = min(self.snapshots, key = lambda t: abs(t - time)) if self.snapshots else None
        if key is None or abs(key - time) > 1e-9:
            raise KeyError(f'No snapshot at t = {time} s; recordTimes holds {sorted(self.snapshots)}.')
        positions, velocities = self.snapshots[key]
        masses = self.snapshotMasses.get(key, np.ones(len(positions)))
        h = self.smoothingLength

        points = np.atleast_2d(np.asarray(points, dtype = float))
        tree = cKDTree(positions)
        result = np.full((len(points), 3), np.nan)
        for index, neighbors in enumerate(tree.query_ball_point(points, 2.0 * h)):
            if not neighbors:
                continue
            q = np.linalg.norm(positions[neighbors] - points[index], axis = 1) / h
            weights = np.maximum(1.0 - 0.5 * q, 0.0)**4 * (2.0 * q + 1.0) * masses[neighbors]
            if weights.sum() > 0.0:
                result[index] = weights @ velocities[neighbors] / weights.sum()

        return result

    def overturn(self, y: float | None = None) -> JetGeometry | None:

        '''

        A section's jet and the tube under it on its surface just before touchdown, read as
        `BreakingWave.overturn` reads them (`BreakingWave.touchdownJet`); None before touchdown.
        y [m] picks the section, None the first.

        '''

        tracker = self.trackers.get(self.sections[0] if y is None else float(y))
        if tracker is None or tracker.touchdownSurface is None:
            return None

        return touchdownJet(tracker.touchdownSurface, self.spacing, self.depth)

    def summary(self) -> dict:

        '''

        Each section's breaking point and, once its jet has landed, the touchdown's measures as
        `BreakingWave.summary` reports them (`BreakingWave.overturnMeasures`), and the run's size
        and cost.

        '''

        sections = {}
        for y, tracker in self.trackers.items():
            point = tracker.breakingPoint
            sections[y] = {
                'broke': point is not None,
                'breakingFinal': tracker.breakingFinal,
                'breakingTime': point.time if point is not None else None,
                'breakingCrestX': point.crestX if point is not None else None,
                'breakingCrestHeight': point.crestHeight if point is not None else None,
                'breakingTroughHeight': point.troughHeight if point is not None else None,
                'breakingWaveHeight': point.waveHeight if point is not None else None,
                'touchedDown': tracker.touchdownSurface is not None,
                'touchdownTime': tracker.touchdownTime if tracker.touchdownSurface is not None else None,
            }
            jet = self.overturn(y)
            if jet is not None:
                sections[y].update(overturnMeasures(tracker, jet, self.field.stillWaterLevel, self.depth))

        result = {'particlesPerDepth': self.particlesPerDepth, 'fluidParticles': self.particles.nFluid,
                  'boundaryParticles': self.particles.nBoundary, 'time': self.solver.time,
                  'wallClockSeconds': self.wallClockSeconds, 'refinement': self.refinement, 'direction': self.direction,
                  'region': self.field.bounds, 'sections': sections}
        if self.refinement is not None and self.refinement['mergeBehind'] is not None:
            result.update({'mergedGroups': self.mergedGroups, 'mergedEnergy': self.mergedEnergy})

        return result
