
# -- pySurf: Potential Flow to SPH Hand-off -- #

'''

The potential flow of a boundary element run handed to SPH particles, so the flow can go on where
potential flow cannot: past the moment a plunging jet lands, when the free surface stops being one
curve and the water stops being irrotational.

    window      a stretch of the tank around the jet, from behind the crest to ahead of where the
                jet lands, with the bed under it and a wall at each end
    particles   a lattice filling the water in the window, half a spacing clear of the bed and the
                walls and a quarter spacing clear of the free surface; the tube under the jet lies
                outside the water and starts empty
    velocity    grad(phi) at each particle, from the boundary integral representation
    pressure    the unsteady Bernoulli pressure (`bemSolver.FlowField`), set as each particle's
                density through the inverse equation of state, so the water starts in the balance
                the potential flow had instead of settling under gravity
    boundary    each bed particle takes the pressure of the water half a spacing above the bed,
                plus the hydrostatic increment down to it; each wall particle that of the water
                beside it, and none above the surface
    piston      the offshore wall moves with the depth-averaged velocity of the potential flow at
                its position, with the wave behind it carried forward unchanged at its crest
                speed, so cutting the tank sends no reflection toward the jet

The shoreward wall stands in water still at rest ahead of the wave, far enough ahead that the
splash-up does not reach it in the time simulated.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

The hand-off carries the potential flow's velocity and pressure, which tests/testBemSolver.py
checks against exact harmonic fields (velocity to 2e-4 k and pressure to 3e-4 rho u^2 at 80 nodes,
second order). tests/testSphHandoff.py checks that still water hands off at rest with hydrostatic
pressure, that the particles fill the water and leave the tube empty, and that the piston follows
the flow. Whether SPH carries the flow on correctly from there is checked in `pysurf.SplashUp`.

Sean Bowman [10/12/2026]

'''

# Standard library imports
from __future__ import annotations

# Third-party imports
import numpy as np
from scipy.spatial import cKDTree

# Local imports
from pysurf import constants
from pysurf.bathymetry import Bathymetry
from pysurf.bemSolver import FlowField
from pysurf.sphParticles import latticePoints

def windowBathymetry(bathymetry: Bathymetry, backX: float, frontX: float) -> Bathymetry:

    '''The bed between x = backX and frontX, its vertices kept and its ends cut where the window ends.'''

    inner = bathymetry.x[(bathymetry.x > backX) & (bathymetry.x < frontX)]
    x = np.concatenate([[backX], inner, [frontX]])

    return Bathymetry(x, bathymetry.elevation(x), bathymetry.stillWaterLevel)

def nearestSurfacePoint(surface: np.ndarray, points: np.ndarray, candidates: int = 4) -> tuple:

    '''

    The nearest point of a free surface to each of `points`, its distance, and the surface normal
    there pointing into the water, below a surface running left to right. The search covers the
    elements on either side of the few nearest nodes.

    Returns:
    --------
    tuple : (distance [m], nearest point, shape (M, 2) [m], inward normal, shape (M, 2))

    '''

    count = min(candidates, len(surface))
    _, nodes = cKDTree(surface).query(points, k = count)
    nodes = nodes.reshape(len(points), count)
    starts = np.concatenate([nodes - 1, nodes], axis = 1).clip(0, len(surface) - 2)
    a, b = surface[starts], surface[starts + 1]
    ab = b - a
    share = np.clip(np.sum((points[:, None, :] - a) * ab, axis = 2) / np.maximum(np.sum(ab**2, axis = 2), 1e-300),
                    0.0, 1.0)
    closest = a + share[..., None] * ab
    distance = np.hypot(*(points[:, None, :] - closest).transpose(2, 0, 1))
    best = np.argmin(distance, axis = 1)
    rows = np.arange(len(points))
    tangent = ab[rows, best] / np.hypot(*ab[rows, best].T)[:, None]

    return distance[rows, best], closest[rows, best], np.column_stack([tangent[:, 1], -tangent[:, 0]])

def seedWater(field: FlowField, bed: Bathymetry, spacing: float, surfaceClearance: float = 0.25) -> np.ndarray:

    '''

    Lattice points filling the water over a window's bed, half a spacing clear of the bed and of
    the walls at its two ends, shape (N, 2) [m]. A point is in the water when the boundary of the
    potential flow winds once around it, which leaves the tube under a jet, and the air between
    the jet and the face, empty.

    A lattice point that falls closer to the free surface than `surfaceClearance` spacings moves
    in along the surface normal to that depth. The velocity there is a difference of the potential
    across a step a millionth of the tank long; a point on the surface itself would straddle the
    jump in the representation across the boundary, and read it as a velocity.

    '''

    top = float(np.max(field.surface[:, 1])) + spacing
    points = latticePoints([bed.startX, float(np.min(bed.z))], [bed.endX, top], spacing)
    # The tolerance keeps the lattice rows and columns exactly half a spacing from a flat bed and a wall
    clear = ((points[:, 0] <= bed.endX - (0.5 - 1e-9) * spacing)
             & (points[:, 1] >= bed.elevation(points[:, 0]) + (0.5 - 1e-9) * spacing))
    points = points[clear]
    points = points[field.inside(points)]

    distance, closest, inward = nearestSurfacePoint(field.surface, points)
    near = distance < surfaceClearance * spacing
    points[near] = closest[near] + surfaceClearance * spacing * inward[near]

    return points

def seedStillWater(bed: Bathymetry, spacing: float, fromX: float) -> np.ndarray:

    '''

    Lattice points filling still water over a window's bed beyond x = fromX, where the potential
    flow's tank has ended, shape (N, 2) [m]: half a spacing clear of the bed and of the wall at the
    window's end, up to still water level. The lattice is the one `seedWater` lays, so the two meet
    without a seam.

    '''

    level = bed.stillWaterLevel
    points = latticePoints([bed.startX, float(np.min(bed.z))], [bed.endX, level], spacing)
    keep = ((points[:, 0] > fromX) & (points[:, 0] <= bed.endX - (0.5 - 1e-9) * spacing)
            & (points[:, 1] >= bed.elevation(points[:, 0]) + (0.5 - 1e-9) * spacing))

    return points[keep]

def boundaryPressure(field: FlowField, bed: Bathymetry, positions: np.ndarray, spacing: float,
                     stillWaterFromX: float | None = None) -> np.ndarray:

    '''

    The pressure each boundary particle starts with [Pa]: that of the water at the nearest point
    half a spacing inside it (above the bed, or in from a wall), plus the hydrostatic increment
    down to the particle; zero where that point is above the surface. Beyond `stillWaterFromX`,
    where the water is still, the pressure is hydrostatic beneath still water level.

    '''

    reference = np.column_stack([np.clip(positions[:, 0], bed.startX + 0.5 * spacing, bed.endX - 0.5 * spacing),
                                 positions[:, 1]])
    reference[:, 1] = np.maximum(reference[:, 1], bed.elevation(reference[:, 0]) + 0.5 * spacing)
    still = (np.zeros(len(positions), dtype = bool) if stillWaterFromX is None
             else (reference[:, 0] > stillWaterFromX))
    wet = ~still & field.inside(reference)
    pressure = np.zeros(len(positions))
    if wet.any():
        hydrostatic = field.density * field.gravity * (reference[wet, 1] - positions[wet, 1])
        pressure[wet] = np.maximum(field.pressure(reference[wet]), 0.0) + hydrostatic
    pressure[still] = field.density * field.gravity * np.maximum(bed.stillWaterLevel - positions[still, 1], 0.0)

    return pressure

def columnVelocity(field: FlowField, bathymetry: Bathymetry, x: np.ndarray, points: int = 16) -> np.ndarray:

    '''

    The horizontal velocity averaged over the water column at stations x [m/s], by Gauss-Legendre
    quadrature from the bed to the surface. The stations must lie where the surface is one curve,
    behind the crest.

    '''

    surface = field.surface
    crest = int(np.argmax(surface[:, 1]))
    top = np.interp(x, surface[:crest + 1, 0], surface[:crest + 1, 1])
    bottom = bathymetry.elevation(x)
    nodes, weights = np.polynomial.legendre.leggauss(points)
    z = bottom[:, None] + 0.5 * (nodes[None, :] + 1.0) * (top - bottom)[:, None]
    velocity = field.velocity(np.column_stack([np.repeat(x, points), z.ravel()]))[:, 0].reshape(len(x), points)

    return 0.5 * velocity @ weights

class PistonDrive:

    '''

    The velocity of a piston standing in for the water behind it: the depth-averaged velocity the
    potential flow had, at the hand-off, a crest-speed's travel behind the piston's position, which
    treats the wave behind the window as translating unchanged.

    Parameters:
    -----------
    field : FlowField
        The potential flow at the hand-off
    bathymetry : Bathymetry
        The bed of the whole tank
    pistonX : float
        Starting position of the piston [m]
    crestSpeed : float
        Speed the wave travels at [m/s]
    duration : float
        Time the piston will run [s]; the stations reach that far behind it
    stations : int
        Columns the velocity is averaged over

    '''

    def __init__(self, field: FlowField, bathymetry: Bathymetry, pistonX: float, crestSpeed: float,
                 duration: float, stations: int = 40) -> None:

        self.crestSpeed = crestSpeed                # [m/s]
        reach = max(pistonX - crestSpeed * duration, bathymetry.startX + 1e-6 * (pistonX - bathymetry.startX))
        self.stationX = np.linspace(reach, pistonX, stations)                       # [m]
        self.stationVelocity = columnVelocity(field, bathymetry, self.stationX)     # [m/s]

    def velocity(self, pistonX: float, time: float) -> float:

        '''Piston velocity at position `pistonX` [m], `time` after the hand-off [s] [m/s].'''

        return float(np.interp(pistonX - self.crestSpeed * time, self.stationX, self.stationVelocity))

def handoffSoundSpeed(depth: float, waveHeight: float, factor: float = 10.0, gravity: float = constants.gravity) -> float:

    '''The artificial speed of sound, `factor` times the solitary wave speed sqrt(g (d + H)) [m/s].'''

    return factor * float(np.sqrt(gravity * (depth + waveHeight)))
