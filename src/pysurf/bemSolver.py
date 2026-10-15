
# -- pySurf: Boundary Element Potential Flow Solver -- #

'''

Fully nonlinear potential flow in a 2D vertical slice, solved with the boundary element method and
mixed Eulerian-Lagrangian time stepping: a free surface that can fold over into a plunging jet.

The water is inviscid and irrotational, so its velocity is the gradient of a potential phi that
satisfies Laplace's equation inside the domain. Green's second identity turns that into an equation
on the boundary alone,

    c(x_l) phi(x_l) + int_Gamma phi dG/dn dGamma = int_Gamma dphi/dn G dGamma,    G = -ln(r) / (2 pi)

so only the boundary is discretized: the free surface, the bed, and a wall at each end. On the free
surface phi is known and the equation gives dphi/dn; on the bed and walls dphi/dn is known (zero for
a fixed boundary) and it gives phi. The boundary is cut into straight elements with phi and dphi/dn
varying linearly along each; the integrals over a straight element are evaluated in closed form, so
there is no quadrature error and no special treatment for a node close to an element, which matters
when the jet nears the water below it. The free term c is found from the rigid-mode condition (a
constant phi has no flux), which handles corners of any angle.

The free-surface nodes move with the water (Lagrangian), and the potential at each follows from the
dynamic condition, zero pressure at the surface:

    D x / D t = grad(phi),      D phi / D t = -g (z - SWL) + |grad(phi)|^2 / 2

The surface velocity combines the tangential derivative of phi along the surface (from a cubic
spline in arc length) with the normal derivative from the boundary solution. Time stepping is
classical fourth-order Runge-Kutta, with a boundary solution at every stage. The step is half of
sqrt(spacing / g) on the shortest element, and no node moves more than one spacing in it; once the
face overhangs, no more than half of one, since a jet tip moving faster folds the intermediate
surfaces the later stages are evaluated on, and the tip tangles.

Because the nodes move with the water they crowd where it converges, so the surface is regridded
at intervals: new nodes spaced along its arc length, finer around the crest, no further apart than
0.3 of the local radius of curvature or half the distance to any part of the surface folded over
them (the face under a landing jet), and down to half the crest spacing, with positions and
potential interpolated by cubic splines. At the tip of a forming jet the surface turns on the scale
of the jet's thickness, a few hundredths of the depth, and with fewer nodes than that the splines
overshoot into a loop. A Lagrangian surface also grows a two-node sawtooth, worst on the jet's
underside, which the five-point smoothing of Longuet-Higgins and Cokelet (1976) removes every ten
steps, and every three once the face overhangs. Should the very tip of the jet still wind into a
loop a few nodes long, those nodes are trimmed and the run goes on. A surface node meeting a wall
slides along it.

The jet lands when it comes within one element of the face below it (`jetGap`). Closer than that
the boundary solution cannot resolve the water between the two and throws the face up into a spike
within a few steps; a moment later the free surface would cross itself and the fluid domain stop
being simply connected, which a single-boundary potential flow cannot represent.
`pysurf.OverturningWave` ends its run there, and records the breaking point as Grilli et al. (1997)
define it, the instant the front face first turns vertical.

References:
-----------
Longuet-Higgins, M.S. & Cokelet, E.D. (1976) -- The deformation of steep surface waves on water I.
    A numerical method of computation, Proc. R. Soc. Lond. A 350, 1-26.
Grilli, S.T., Skourup, J. & Svendsen, I.A. (1989) -- An efficient boundary element method for
    nonlinear water waves, Engineering Analysis with Boundary Elements 6(2), 97-107.
Grilli, S.T., Svendsen, I.A. & Subramanya, R. (1997) -- Breaking criterion and characteristics for
    solitary waves on slopes, J. Waterway, Port, Coastal, Ocean Eng. 123(3), 102-112.
Guyenne, P. & Grilli, S.T. (2006) -- Numerical study of three-dimensional overturning waves in
    shallow water, J. Fluid Mech. 547, 361-388.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

tests/testBemSolver.py checks the element integrals against numerical quadrature (to 1e-9), the
boundary solution against the exact harmonic function cosh(k (z + d)) cos(k x) in a rectangle
(surface flux error 4.8e-5 at 160 nodes, falling by four per halving away from the corners, where
linear elements are first order), and the time stepping against the Airy period of a small standing
wave (-0.03 percent at k d = 1, -0.11 percent at k d = 3) with volume conserved to 1e-8 and energy
to 1e-5 over two periods. The flow inside the water (`FlowField`) is checked against the same
harmonic field, the potential to 1.4e-4 and the velocity to 2e-4 k at 80 nodes, both second order
from an element below the surface, and its pressure against the exact unsteady field with zero
pressure on the flat surface, to 1.5e-4 of rho u^2. tests/testValidationOverturning.py runs a
solitary wave on a 1:35 slope against Grilli et al. (1997); the results are in
`pysurf.OverturningWave`.

Sean Bowman [10/11/2026]

'''

# Standard library imports
from __future__ import annotations

import math
from dataclasses import dataclass

# Third-party imports
import numpy as np
from scipy.interpolate import CubicSpline

# Local imports
from pysurf import constants
from pysurf.bathymetry import Bathymetry

#----------------------------------------------------------------------#
# -- Boundary Integrals -- #
#----------------------------------------------------------------------#

def elementIntegrals(points: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> tuple:

    '''

    Integrals of the Green's function and its normal derivative over straight elements, weighted by
    the linear shape functions of each element's two end nodes, for every collocation point.

    For a collocation point x_l and an element from a to b of length L, with N_a = 1 - u / L and
    N_b = u / L along it:

        G_k = int N_k G dGamma,     H_k = int N_k dG/dn dGamma,     G = -ln(r) / (2 pi)

    with n the element's outward normal for a counterclockwise boundary (the tangent turned 90
    degrees clockwise). In the element's frame the point sits at (u0, v0) and r^2 = (u - u0)^2 +
    v0^2; every integral has a closed form in logarithms and the angle the element subtends at the
    point, which stays finite as the point approaches the element and is exact when it lies on it.

    Parameters:
    -----------
    points : np.ndarray
        Collocation points, shape (M, 2)
    starts, ends : np.ndarray
        Element end points, shape (E, 2)

    Returns:
    --------
    tuple : (Ga, Gb, Ha, Hb), each shaped (M, E)

    '''

    segment = ends - starts
    length = np.hypot(segment[:, 0], segment[:, 1])
    tangent = segment / length[:, None]
    normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])

    dx = points[:, None, 0] - starts[None, :, 0]
    dz = points[:, None, 1] - starts[None, :, 1]
    u0 = dx * tangent[None, :, 0] + dz * tangent[None, :, 1]
    v0 = dx * normal[None, :, 0] + dz * normal[None, :, 1]
    L = length[None, :]
    w0, w1 = -u0, L - u0
    r0, r1 = w0**2 + v0**2, w1**2 + v0**2
    tiny = 1e-300
    log0, log1 = np.log(np.maximum(r0, tiny)), np.log(np.maximum(r1, tiny))

    # Angle the element subtends at the point: v0 times int du / r^2. A point on the element's own
    # line (a node at the element's end, to rounding) sees no angle; arctan2 of two rounding-level
    # arguments would return +-pi there. Collocation points are nodes, so none lies inside an element.
    onLine = np.abs(v0) <= 1e-12 * L
    angle = np.where(onLine, 0.0, np.arctan2(v0 * (w1 - w0), v0**2 + w0 * w1))
    # v0 times int u du / r^2
    vFirst = np.where(onLine, 0.0, 0.5 * v0 * (log1 - log0)) + u0 * angle

    Ha = (angle - vFirst / L) / (2.0 * math.pi)
    Hb = (vFirst / L) / (2.0 * math.pi)

    # int ln r du and int u ln r du
    logZeroth = 0.5 * (w1 * log1 - w0 * log0 - 2.0 * (w1 - w0) + 2.0 * v0 * angle)
    logFirst = 0.25 * ((r1 * log1 - w1**2) - (r0 * log0 - w0**2)) + u0 * logZeroth

    Ga = -(logZeroth - logFirst / L) / (2.0 * math.pi)
    Gb = -(logFirst / L) / (2.0 * math.pi)

    return Ga, Gb, Ha, Hb

#----------------------------------------------------------------------#
# -- Boundary Solution -- #
#----------------------------------------------------------------------#

@dataclass
class BoundarySolution:

    '''

    The boundary solution for one free-surface shape.

    Parameters:
    -----------
    normalDerivative : np.ndarray
        dphi/dn at the free-surface nodes, outward normal (up for a flat surface) [m/s]
    bedPotential : np.ndarray
        phi at the bed nodes [m^2/s]
    nodes : np.ndarray
        Every boundary node counterclockwise, corners once on each side, shape (N, 2) [m]
    elementA, elementB : np.ndarray
        Start and end node of each element, indices into `nodes`
    potential, flux : np.ndarray
        phi [m^2/s] and dphi/dn [m/s] at every boundary node
    surfacePotential : np.ndarray
        phi at the free-surface nodes, left to right [m^2/s]

    '''

    normalDerivative: np.ndarray
    bedPotential: np.ndarray
    nodes: np.ndarray | None = None
    elementA: np.ndarray | None = None
    elementB: np.ndarray | None = None
    potential: np.ndarray | None = None
    flux: np.ndarray | None = None
    surfacePotential: np.ndarray | None = None

def solveBoundary(surface: np.ndarray, potential: np.ndarray, bed: np.ndarray, wallSpacing: float) -> BoundarySolution:

    '''

    Solve Laplace's equation in the region between a free surface and a bed closed by vertical walls
    at both ends: phi given on the surface, zero normal flux on the bed and walls.

    The boundary runs counterclockwise: along the bed from left to right, up the right wall, back
    along the surface from right to left, and down the left wall. Each corner appears once in each
    of the two boundaries it joins. Where the bed meets a wall the two entries share one unknown phi.
    Where the surface meets a wall, phi is the surface's on both sides and the wall side's flux is
    zero, so the only unknown is the surface side's flux, solved for at that point like any other
    surface node.

    Parameters:
    -----------
    surface : np.ndarray
        Free-surface nodes from left wall to right wall, shape (Nf, 2) [m]
    potential : np.ndarray
        phi at the surface nodes [m^2/s]
    bed : np.ndarray
        Bed nodes from left wall to right wall, shape (Nb, 2) [m]; the walls stand at the bed's
        first and last x
    wallSpacing : float
        Target node spacing up the walls [m]

    Returns:
    --------
    BoundarySolution : dphi/dn on the surface and phi on the bed

    '''

    nSurface, nBed = len(surface), len(bed)

    def wall(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
        height = abs(top[1] - bottom[1])
        count = max(3, int(math.ceil(height / max(min(wallSpacing, 0.5 * height), 1e-12))) + 1)
        return np.column_stack([np.full(count, bottom[0]), np.linspace(bottom[1], top[1], count)])

    rightWall = wall(bed[-1], surface[-1])
    leftWall = wall(bed[0], surface[0])[::-1]
    nRight, nLeft = len(rightWall), len(leftWall)
    polylines = [bed, rightWall, surface[::-1], leftWall]
    nodes = np.vstack(polylines)

    # Element connectivity within each polyline
    offsets = np.cumsum([0] + [len(line) for line in polylines])
    elementA = np.concatenate([np.arange(offsets[i], offsets[i + 1] - 1) for i in range(4)])
    elementB = elementA + 1

    # Entry index ranges
    bedEntries = np.arange(0, nBed)
    rightEntries = np.arange(offsets[1], offsets[2])
    surfaceEntries = np.arange(offsets[2], offsets[3])              # right to left
    leftEntries = np.arange(offsets[3], offsets[4])
    nEntries = len(nodes)

    # Unknown phi: one per bed node and interior wall node; wall ends shared with the bed reuse the
    # bed's column, wall tops take the surface's known phi
    phiColumn = np.full(nEntries, -1)
    phiColumn[bedEntries] = np.arange(nBed)
    rightInterior, leftInterior = rightEntries[1:-1], leftEntries[1:-1]
    phiColumn[rightInterior] = nBed + np.arange(len(rightInterior))
    phiColumn[leftInterior] = nBed + len(rightInterior) + np.arange(len(leftInterior))
    phiColumn[rightEntries[0]] = phiColumn[bedEntries[-1]]
    phiColumn[leftEntries[-1]] = phiColumn[bedEntries[0]]
    nPhi = nBed + len(rightInterior) + len(leftInterior)

    knownPhi = np.zeros(nEntries)
    knownPhi[surfaceEntries] = potential[::-1]
    knownPhi[rightEntries[-1]] = potential[-1]
    knownPhi[leftEntries[0]] = potential[0]
    phiIsKnown = phiColumn < 0

    # Collocation at every bed node, interior wall node and surface node
    collocationEntries = np.concatenate([bedEntries, rightInterior, leftInterior, surfaceEntries])
    points = nodes[collocationEntries]

    Ga, Gb, Ha, Hb = elementIntegrals(points, nodes[elementA], nodes[elementB])
    G = np.zeros((len(points), nEntries))
    H = np.zeros((len(points), nEntries))
    G[:, elementA] += Ga
    G[:, elementB] += Gb
    H[:, elementA] += Ha
    H[:, elementB] += Hb
    # Rigid mode: a constant phi carries no flux, which fixes the free term at every point
    freeTerm = -H.sum(axis = 1)

    nUnknown = nPhi + nSurface
    A = np.zeros((len(points), nUnknown))
    rightHandSide = -H[:, phiIsKnown] @ knownPhi[phiIsKnown]
    unknownEntries = np.flatnonzero(~phiIsKnown)
    np.add.at(A.T, phiColumn[unknownEntries], H.T[unknownEntries])
    A[:, nPhi:] = -G[:, surfaceEntries]

    rows = np.arange(len(points))
    ownColumn = phiColumn[collocationEntries]
    onUnknown = ownColumn >= 0
    A[rows[onUnknown], ownColumn[onUnknown]] += freeTerm[onUnknown]
    rightHandSide[~onUnknown] -= freeTerm[~onUnknown] * knownPhi[collocationEntries[~onUnknown]]

    # NumPy's LAPACK, not SciPy's: the SciPy build here factors this size twenty times slower
    solution = np.linalg.solve(A, rightHandSide)

    potentialAll = knownPhi.copy()
    potentialAll[unknownEntries] = solution[phiColumn[unknownEntries]]
    fluxAll = np.zeros(nEntries)
    fluxAll[surfaceEntries] = solution[nPhi:]

    return BoundarySolution(normalDerivative = solution[nPhi:][::-1], bedPotential = solution[:nBed], nodes = nodes,
                            elementA = elementA, elementB = elementB, potential = potentialAll, flux = fluxAll,
                            surfacePotential = np.asarray(potential, dtype = float).copy())

def solveNeumann(surface: np.ndarray, surfaceFlux: np.ndarray, bed: np.ndarray, wallSpacing: float) -> BoundarySolution:

    '''

    Solve Laplace's equation with the normal derivative given on the whole boundary: dphi/dn on the
    free surface, zero on the bed and the walls. It gives the surface potential of a flow known by
    how its surface moves: in water whose surface rises at eta_t, the kinematic condition makes
    dphi/dn = eta_t / sqrt(1 + eta_x^2) there.

    The boundary is the one `solveBoundary` uses, with phi unknown at every point of it, shared at
    the corners. phi is fixed only up to a constant, which is set by phi = 0 at the free surface's
    first node; the fluxes must sum to zero over the boundary, as they do when no water crosses the
    walls, and the system is solved by least squares so a discrete imbalance does not spoil it.

    Parameters:
    -----------
    surface : np.ndarray
        Free-surface nodes from left wall to right wall, shape (Nf, 2) [m]
    surfaceFlux : np.ndarray
        dphi/dn at the surface nodes, outward normal (up for a flat surface) [m/s]
    bed : np.ndarray
        Bed nodes from left wall to right wall, shape (Nb, 2) [m]
    wallSpacing : float
        Target node spacing up the walls [m]

    Returns:
    --------
    BoundarySolution : with `surfacePotential`, `potential` at every boundary node, `bedPotential`,
        and the given surface flux as `normalDerivative`

    '''

    nSurface, nBed = len(surface), len(bed)

    def wall(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
        height = abs(top[1] - bottom[1])
        count = max(3, int(math.ceil(height / max(min(wallSpacing, 0.5 * height), 1e-12))) + 1)
        return np.column_stack([np.full(count, bottom[0]), np.linspace(bottom[1], top[1], count)])

    rightWall = wall(bed[-1], surface[-1])
    leftWall = wall(bed[0], surface[0])[::-1]
    polylines = [bed, rightWall, surface[::-1], leftWall]
    nodes = np.vstack(polylines)
    offsets = np.cumsum([0] + [len(line) for line in polylines])
    elementA = np.concatenate([np.arange(offsets[i], offsets[i + 1] - 1) for i in range(4)])
    elementB = elementA + 1

    bedEntries = np.arange(0, nBed)
    rightEntries = np.arange(offsets[1], offsets[2])
    surfaceEntries = np.arange(offsets[2], offsets[3])              # right to left
    leftEntries = np.arange(offsets[3], offsets[4])
    nEntries = len(nodes)

    # One unknown phi per distinct point: the bed, the walls' interiors and the surface; each wall's
    # foot shares the bed's end and its top the surface's end
    column = np.full(nEntries, -1)
    column[bedEntries] = np.arange(nBed)
    rightInterior, leftInterior = rightEntries[1:-1], leftEntries[1:-1]
    column[rightInterior] = nBed + np.arange(len(rightInterior))
    column[surfaceEntries] = nBed + len(rightInterior) + np.arange(nSurface)
    column[leftInterior] = nBed + len(rightInterior) + nSurface + np.arange(len(leftInterior))
    column[rightEntries[0]], column[rightEntries[-1]] = column[bedEntries[-1]], column[surfaceEntries[0]]
    column[leftEntries[0]], column[leftEntries[-1]] = column[surfaceEntries[-1]], column[bedEntries[0]]
    nUnknown = nBed + len(rightInterior) + nSurface + len(leftInterior)

    flux = np.zeros(nEntries)
    flux[surfaceEntries] = np.asarray(surfaceFlux, dtype = float)[::-1]

    # Collocation once at every distinct point
    collocationEntries = np.concatenate([bedEntries, rightInterior, surfaceEntries, leftInterior])
    points = nodes[collocationEntries]
    Ga, Gb, Ha, Hb = elementIntegrals(points, nodes[elementA], nodes[elementB])
    G = np.zeros((len(points), nEntries))
    H = np.zeros((len(points), nEntries))
    G[:, elementA] += Ga
    G[:, elementB] += Gb
    H[:, elementA] += Ha
    H[:, elementB] += Hb
    freeTerm = -H.sum(axis = 1)

    A = np.zeros((len(points), nUnknown))
    np.add.at(A.T, column, H.T)
    A[np.arange(len(points)), column[collocationEntries]] += freeTerm
    rightHandSide = G @ flux

    # phi = 0 at the surface's first node (its last entry) removes the free constant
    reference = int(column[surfaceEntries[-1]])
    keep = np.arange(nUnknown) != reference
    solution = np.zeros(nUnknown)
    solution[keep] = np.linalg.lstsq(A[:, keep], rightHandSide, rcond = None)[0]

    return BoundarySolution(normalDerivative = np.asarray(surfaceFlux, dtype = float).copy(),
                            bedPotential = solution[:nBed], nodes = nodes, elementA = elementA, elementB = elementB,
                            potential = solution[column], flux = flux,
                            surfacePotential = solution[column[surfaceEntries]][::-1])

def interiorPotential(solution: BoundarySolution, points: np.ndarray, chunk: int = 1000) -> np.ndarray:

    '''

    phi at points inside the water, from the boundary integral representation with the free term one,

        phi(x) = int dphi/dn G dGamma - int phi dG/dn dGamma,

    over the boundary data of a solution [m^2/s]. The element integrals are exact, so the result is
    the potential of the discrete boundary data however near the boundary the point lies; within an
    element length of it, that data's linear interpolation between nodes is what limits accuracy.

    Parameters:
    -----------
    solution : BoundarySolution
        A boundary solution from `solveBoundary`
    points : np.ndarray
        Points inside the water, shape (M, 2) [m]
    chunk : int
        Points evaluated together, which bounds the memory the element integrals take

    '''

    starts, ends = solution.nodes[solution.elementA], solution.nodes[solution.elementB]
    fluxA, fluxB = solution.flux[solution.elementA], solution.flux[solution.elementB]
    potentialA, potentialB = solution.potential[solution.elementA], solution.potential[solution.elementB]
    values = np.empty(len(points))
    for first in range(0, len(points), chunk):
        Ga, Gb, Ha, Hb = elementIntegrals(points[first:first + chunk], starts, ends)
        values[first:first + chunk] = Ga @ fluxA + Gb @ fluxB - Ha @ potentialA - Hb @ potentialB

    return values

def windingShare(solution: BoundarySolution, points: np.ndarray, chunk: int = 1000) -> np.ndarray:

    '''

    The angle the whole boundary subtends at each point over 2 pi: one inside the water, zero outside
    it, exactly, since the double-layer integral of a constant sums the angles each element subtends.

    '''

    starts, ends = solution.nodes[solution.elementA], solution.nodes[solution.elementB]
    share = np.empty(len(points))
    for first in range(0, len(points), chunk):
        _, _, Ha, Hb = elementIntegrals(points[first:first + chunk], starts, ends)
        share[first:first + chunk] = -(Ha.sum(axis = 1) + Hb.sum(axis = 1))

    return share

def interiorVelocity(solution: BoundarySolution, points: np.ndarray, step: float | None = None) -> np.ndarray:

    '''

    grad(phi) at points inside the water, from central differences of `interiorPotential` [m/s].

    The potential is evaluated exactly, so the differences carry only their truncation error, of
    order step^2 times the third derivative, and rounding, of order 1e-16 |phi| / step; the default
    step, a millionth of the boundary's extent, keeps both far below the discretization error.

    '''

    if step is None:
        step = 1e-6 * float(np.max(np.ptp(solution.nodes, axis = 0)))
    shifts = np.array([[step, 0.0], [-step, 0.0], [0.0, step], [0.0, -step]])
    shifted = (points[None, :, :] + shifts[:, None, :]).reshape(-1, 2)
    phi = interiorPotential(solution, shifted).reshape(4, -1)

    return np.column_stack([(phi[0] - phi[1]) / (2.0 * step), (phi[2] - phi[3]) / (2.0 * step)])

#----------------------------------------------------------------------#
# -- Free Surface Geometry -- #
#----------------------------------------------------------------------#

def arcLength(points: np.ndarray) -> np.ndarray:

    '''Cumulative arc length along a polyline, starting at zero [m].'''

    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis = 0).T))])

def crestSpan(surface: np.ndarray, crestRange: tuple[float, float] | None = None) -> slice:

    '''

    The run of surface nodes over which a crest is sought: from the first node at or past x =
    crestRange[0] to the last at or before crestRange[1], or the whole surface when `crestRange` is
    None or holds fewer than four nodes. A tank may hold other waves, such as the bores behind a
    reef, higher than the one to follow.

    '''

    if crestRange is None:
        return slice(0, len(surface))
    inside = np.flatnonzero((surface[:, 0] >= crestRange[0]) & (surface[:, 0] <= crestRange[1]))
    if inside.size < 4:
        return slice(0, len(surface))

    return slice(int(inside[0]), int(inside[-1]) + 1)

def crestIndex(surface: np.ndarray, crestRange: tuple[float, float] | None = None) -> int:

    '''Index of the highest surface node within `crestSpan`.'''

    span = crestSpan(surface, crestRange)

    return span.start + int(np.argmax(surface[span, 1]))

def surfaceVelocity(surface: np.ndarray, potential: np.ndarray, normalDerivative: np.ndarray) -> np.ndarray:

    '''

    Velocity of the water at the free-surface nodes, grad(phi) = phi_s t + phi_n n [m/s].

    The tangent t and the tangential derivative phi_s come from cubic splines of position and
    potential in arc length; n is the tangent turned up. The two end nodes slide on the walls, so
    their horizontal velocity is zero.

    '''

    s = arcLength(surface)
    xSpline, zSpline = CubicSpline(s, surface[:, 0]), CubicSpline(s, surface[:, 1])
    phiSpline = CubicSpline(s, potential)
    tangent = np.column_stack([xSpline(s, 1), zSpline(s, 1)])
    tangent /= np.hypot(tangent[:, 0], tangent[:, 1])[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])

    velocity = phiSpline(s, 1)[:, None] * tangent + normalDerivative[:, None] * normal
    velocity[[0, -1], 0] = 0.0

    return velocity

def frontSlopeMinimum(surface: np.ndarray) -> float:

    '''

    The smallest dx/ds along the surface, from a cubic spline in arc length: one for a flat surface,
    zero where the face is vertical, negative where it overhangs.

    '''

    s = arcLength(surface)

    return float(np.min(CubicSpline(s, surface[:, 0])(s, 1)))

def regridSurface(surface: np.ndarray, potential: np.ndarray, spacing) -> tuple[np.ndarray, np.ndarray]:

    '''

    New surface nodes at a target spacing along the arc length, with positions and potential
    interpolated by cubic splines; the end nodes stay at the walls.

    Parameters:
    -----------
    surface : np.ndarray
        Free-surface nodes, shape (N, 2) [m]
    potential : np.ndarray
        phi at the nodes [m^2/s]
    spacing : Callable
        Target spacing as a function of arc length, spacing(s) [m]

    Returns:
    --------
    tuple : (surface, potential) on the new nodes

    Raises:
    -------
    ValueError : If the new spacing calls for more than three times the nodes, the sign of a surface
        that has tangled, whose splines then measure an arc length many times its true one.

    '''

    # Nodes on top of their neighbor make the arc-length splines oscillate; keep one of each pair
    s = arcLength(surface)
    keep = np.concatenate([[True], np.diff(s) > 1e-6 * s[-1]])
    keep[-1] = True
    surface, potential = surface[keep], potential[keep]
    s = arcLength(surface)
    fine = np.linspace(0.0, s[-1], 20 * len(s))
    density = np.concatenate([[0.0], np.cumsum(0.5 * (1.0 / spacing(fine[1:]) + 1.0 / spacing(fine[:-1])) * np.diff(fine))])
    count = max(4, int(round(density[-1])))
    if count > 3 * len(s) + 100:
        raise ValueError(f'Regridding called for {count} nodes from {len(s)}: the surface has tangled.')
    targets = np.interp(np.linspace(0.0, density[-1], count + 1), density, fine)
    targets[0], targets[-1] = 0.0, s[-1]

    newSurface = np.column_stack([CubicSpline(s, surface[:, 0])(targets), CubicSpline(s, surface[:, 1])(targets)])
    newSurface[0, 0], newSurface[-1, 0] = surface[0, 0], surface[-1, 0]

    return newSurface, CubicSpline(s, potential)(targets)

def smoothSurface(surface: np.ndarray, potential: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    '''

    The five-point smoothing of Longuet-Higgins and Cokelet (1976) applied to node positions and
    potential: f_j <- (-f_{j-2} + 4 f_{j-1} + 10 f_j + 4 f_{j+1} - f_{j+2}) / 16.

    It removes the two-node sawtooth that grows on a Lagrangian free surface, worst on the underside
    of a forming jet, and leaves a smooth profile unchanged to fourth order in node spacing. The
    two nodes at each wall are left alone, and the wall nodes keep their x.

    '''

    def smooth(values: np.ndarray) -> np.ndarray:
        result = values.copy()
        result[2:-2] = (-values[:-4] + 4.0 * values[1:-3] + 10.0 * values[2:-2] + 4.0 * values[3:-1] - values[4:]) / 16.0
        return result

    smoothed = np.column_stack([smooth(surface[:, 0]), smooth(surface[:, 1])])

    return smoothed, smooth(potential)

def selfCrossings(surface: np.ndarray, start: int, stop: int, gap: int = 3) -> list[tuple[int, int]]:

    '''

    Pairs of surface segments between nodes `start` and `stop`, at least `gap` segments apart, that
    intersect, as (i, j) with segment i running from node i to node i + 1 and i < j.

    '''

    a, b = surface[start:stop - 1], surface[start + 1:stop]
    count = len(a)
    i, j = np.triu_indices(count, k = gap)
    if i.size == 0:
        return []

    def orientation(p, q, r):
        return (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0])

    d1 = orientation(a[i], b[i], a[j])
    d2 = orientation(a[i], b[i], b[j])
    d3 = orientation(a[j], b[j], a[i])
    d4 = orientation(a[j], b[j], b[i])
    crossing = (d1 * d2 < 0.0) & (d3 * d4 < 0.0)

    return [(int(start + p), int(start + q)) for p, q in zip(i[crossing], j[crossing])]

def trimLoop(surface: np.ndarray, potential: np.ndarray, first: int, second: int) -> tuple[np.ndarray, np.ndarray]:

    '''

    Remove the nodes of a small loop the surface has wound into, between crossing segments `first`
    and `second`: nodes first + 1 to second go, and the surface joins straight across.

    '''

    keep = np.ones(len(surface), dtype = bool)
    keep[first + 1:second + 1] = False

    return surface[keep], potential[keep]

def surfaceCrossesItself(surface: np.ndarray, start: int, stop: int, gap: int = 3) -> bool:

    '''

    Whether any two segments of the surface between nodes `start` and `stop`, at least `gap` nodes
    apart, intersect: the jet has reached the water below it.

    '''

    a, b = surface[start:stop - 1], surface[start + 1:stop]
    count = len(a)
    i, j = np.triu_indices(count, k = gap)
    if i.size == 0:
        return False

    def orientation(p, q, r):
        return (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0])

    d1 = orientation(a[i], b[i], a[j])
    d2 = orientation(a[i], b[i], b[j])
    d3 = orientation(a[j], b[j], a[i])
    d4 = orientation(a[j], b[j], b[i])

    return bool(np.any((d1 * d2 < 0.0) & (d3 * d4 < 0.0)))

def foldDistance(surface: np.ndarray, ratio: float = 3.0) -> np.ndarray:

    '''

    Distance from each node to the nearest node of another part of the surface, one at least `ratio`
    times as far away along the surface as in a straight line [m]. A smooth surface comes that close
    to itself only where it folds back, as the jet does over the face below it and the jet's top
    over its underside; elsewhere the distance is infinite.

    '''

    s = arcLength(surface)
    distance = np.hypot(surface[:, None, 0] - surface[None, :, 0], surface[:, None, 1] - surface[None, :, 1])
    folded = np.abs(s[:, None] - s[None, :]) > ratio * distance

    return np.where(folded, distance, np.inf).min(axis = 1)

@dataclass
class JetGeometry:

    '''

    The plunging jet and the tube of air under it, from a surface that overhangs, in the measures of
    Feddersen et al. (2023, 2024) and Pick and Feddersen (2026), which they take at jet impact.

    The overturn boundary runs along the surface from the jet node nearest the face, under the jet
    and round the tube, to the face node nearest that node; the straight line between the two closes
    it. At impact the line is shorter than an element, and the boundary encloses the air in the tube.

    Parameters:
    -----------
    tip : np.ndarray
        Jet node nearest the face [m]
    impactPoint : np.ndarray
        Face node nearest the tip, where the jet lands [m]
    cavityArea : float
        Area enclosed by the overturn boundary, A_O [m^2]
    tubeLength, tubeHeight : float
        Horizontal and vertical extent of the overturn boundary [m]
    jetArea : float
        Area of the upper region of water where the surface is multi-valued in x, A_J [m^2]
    overturnLength, overturnWidth : float
        Length L and width W of the Longuet-Higgins (1982) curve fitted to the overturn boundary [m]
    overturnAngle : float
        Clockwise rotation of the fitted curve's long axis from horizontal, theta_O [deg]
    overturnBack : np.ndarray
        Back of the fitted curve, where its long axis starts [m]
    fitError : float
        Root mean square distance from the overturn boundary to the fitted curve [m]
    boundary : np.ndarray
        The overturn boundary, surface nodes from the tip to the impact point, shape (M, 2) [m]

    '''

    tip: np.ndarray
    impactPoint: np.ndarray
    cavityArea: float
    tubeLength: float
    tubeHeight: float
    jetArea: float
    overturnLength: float
    overturnWidth: float
    overturnAngle: float
    overturnBack: np.ndarray
    fitError: float
    boundary: np.ndarray

    @property
    def tubeRatio(self) -> float:
        '''Tube length over height [-].'''
        return self.tubeLength / self.tubeHeight if self.tubeHeight > 0.0 else float('nan')

    @property
    def aspectRatio(self) -> float:
        '''Overturn width over length, W / L [-].'''
        return self.overturnWidth / self.overturnLength if self.overturnLength > 0.0 else float('nan')

    def fittedCurve(self, count: int = 200) -> np.ndarray:
        '''Points on the fitted Longuet-Higgins (1982) curve, shape (2 count, 2) [m].'''
        return _longuetHigginsCurve(np.array([self.overturnLength, self.overturnWidth, math.radians(self.overturnAngle),
                                              self.overturnBack[0], self.overturnBack[1]]), count)

def _overhang(surface: np.ndarray) -> tuple[int, int] | None:

    '''

    The most shoreward node of the overhang and the back of the tube as node indices, or None when
    the surface does not overhang.

    Along the surface from left to right, x increases up the back of the wave and out along the top
    of the jet, falls back along the jet's underside to the back of the tube, and increases again
    down the face. The overhang is the run of nodes behind the furthest x reached so far, which a
    one-node wiggle in the surface cannot split; the jet is the run that falls furthest behind. Its
    most shoreward node is the one before the run starts, the back of the tube its node of least x.

    '''

    x = surface[:, 0]
    deficit = np.maximum.accumulate(x) - x
    behind = deficit > 1e-9 * max(float(np.ptp(x)), 1e-300)
    if not behind.any():
        return None

    # Runs of nodes behind the furthest x, and the one that falls furthest behind
    edges = np.diff(np.concatenate([[0], behind.astype(int), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    deepest = int(np.argmax([deficit[a:b].max() for a, b in zip(starts, ends)]))
    runStart, runEnd = int(starts[deepest]), int(ends[deepest])

    return runStart - 1, runStart + int(np.argmin(x[runStart:runEnd]))

def _multiValuedArea(surface: np.ndarray, low: float, high: float, samples: int = 400) -> float:

    '''

    Area between the two highest crossings of the surface with vertical lines from x = `low` to
    `high`, counting only lines that cross it three times or more: the jet, above the tube [m^2].

    '''

    x = np.linspace(low, high, samples)
    a, b = surface[:-1], surface[1:]
    thickness = np.zeros(samples)
    for k, xk in enumerate(x):
        crossing = (a[:, 0] - xk) * (b[:, 0] - xk) < 0.0
        if np.count_nonzero(crossing) < 3:
            continue
        share = (xk - a[crossing, 0]) / (b[crossing, 0] - a[crossing, 0])
        z = np.sort(a[crossing, 1] + share * (b[crossing, 1] - a[crossing, 1]))[::-1]
        thickness[k] = z[0] - z[1]

    return float(np.sum(0.5 * (thickness[1:] + thickness[:-1]) * np.diff(x)))

def _longuetHigginsCurve(parameters: np.ndarray, count: int = 200) -> np.ndarray:

    '''

    Points on the overturn curve of Longuet-Higgins (1982), z' / W = +-(3 sqrt(3) / 4) sqrt(x' / L)
    (x' / L - 1) for 0 <= x' <= L, rotated clockwise by theta about its back at (x0, z0).

    Parameters:
    -----------
    parameters : np.ndarray
        (L [m], W [m], theta [rad], x0 [m], z0 [m])

    '''

    length, width, angle, x0, z0 = parameters
    # Squared spacing puts more points at the back, where the curve turns
    share = np.linspace(0.0, 1.0, count) ** 2
    half = 0.75 * math.sqrt(3.0) * width * np.sqrt(share) * (share - 1.0)
    along = np.concatenate([length * share, length * share[::-1]])
    across = np.concatenate([half, -half[::-1]])

    return np.column_stack([x0 + along * math.cos(angle) + across * math.sin(angle),
                            z0 - along * math.sin(angle) + across * math.cos(angle)])

def _fitOverturn(boundary: np.ndarray, area: float) -> tuple[float, float, float, float, np.ndarray]:

    '''

    Fit the Longuet-Higgins (1982) curve to an overturn boundary as Feddersen et al. (2023) fit it to
    lidar returns: five parameters (L, W, theta, x0, z0) chosen by a Nelder-Mead search to minimize
    the mean square distance from the boundary to the curve.

    The boundary is resampled at even arc length, so the node clustering at the jet tip does not
    weight the fit. The search starts from the boundary point furthest from the impact as the back,
    the line from it to the impact as the axis, and the width that gives the curve the boundary's
    area, A = 2 sqrt(3) L W / 5.

    Returns:
    --------
    tuple : (L [m], W [m], theta [deg], rms distance [m], back of the curve (x0, z0) [m])

    '''

    from scipy.optimize import minimize

    s = arcLength(boundary)
    even = np.linspace(0.0, s[-1], 150)
    points = np.column_stack([np.interp(even, s, boundary[:, 0]), np.interp(even, s, boundary[:, 1])])

    impact = 0.5 * (boundary[0] + boundary[-1])
    back = points[int(np.argmax(np.hypot(*(points - impact).T)))]
    length = float(np.hypot(*(impact - back)))
    angle = math.atan2(back[1] - impact[1], impact[0] - back[0])
    width = 5.0 * area / (2.0 * math.sqrt(3.0) * max(length, 1e-12))

    def cost(parameters: np.ndarray) -> float:
        if parameters[0] <= 0.0 or parameters[1] <= 0.0:
            return 1e6
        curve = _longuetHigginsCurve(parameters)
        distance = np.hypot(points[:, None, 0] - curve[None, :, 0], points[:, None, 1] - curve[None, :, 1])
        return float(np.mean(np.min(distance, axis = 1) ** 2))

    start = np.array([length, width, angle, back[0], back[1]])
    result = minimize(cost, start, method = 'Nelder-Mead',
                      options = {'xatol': 1e-6 * length, 'fatol': 1e-12 * length**2, 'maxiter': 4000})
    fitLength, fitWidth, fitAngle = result.x[:3]

    return float(fitLength), float(fitWidth), math.degrees(fitAngle), math.sqrt(result.fun), result.x[3:].copy()

def _closestPair(surface: np.ndarray, back: int, stop: int, gap: int) -> tuple[int, int] | None:

    '''The jet node and face node nearest each other, at least `gap` nodes apart, as indices.'''

    crest = int(np.argmax(surface[:, 1]))
    if not crest < back < stop - 1:
        return None
    jet, face = np.arange(crest, back + 1), np.arange(back, stop)
    distance = np.hypot(surface[jet, None, 0] - surface[None, face, 0], surface[jet, None, 1] - surface[None, face, 1])
    distance = np.where(face[None, :] - jet[:, None] >= gap, distance, np.inf)
    i, j = np.unravel_index(int(np.argmin(distance)), distance.shape)

    return int(jet[i]), int(face[j])

def jetGeometry(surface: np.ndarray, gap: int = 15) -> JetGeometry | None:

    '''

    The jet and the tube under it, or None when the surface does not overhang or no part of the face
    lies under the jet.

    Parameters:
    -----------
    surface : np.ndarray
        Free-surface nodes, shape (N, 2) [m]
    gap : int
        Fewest nodes along the surface between the tip and the impact point, so the turn at the back
        of the tube does not count as the jet meeting the face

    '''

    overhang = _overhang(surface)
    if overhang is None:
        return None
    outermost, back = overhang
    pair = _closestPair(surface, back, len(surface), gap)
    if pair is None:
        return None
    tip, impact = pair

    boundary = surface[tip:impact + 1]
    area = 0.5 * abs(np.dot(boundary[:, 0], np.roll(boundary[:, 1], -1))
                     - np.dot(boundary[:, 1], np.roll(boundary[:, 0], -1)))
    jetArea = _multiValuedArea(surface, float(surface[back, 0]), float(surface[outermost, 0]))
    length, width, angle, error, fitBack = _fitOverturn(boundary, float(area))

    return JetGeometry(tip = surface[tip].copy(), impactPoint = surface[impact].copy(), cavityArea = float(area),
                       tubeLength = float(np.ptp(boundary[:, 0])), tubeHeight = float(np.ptp(boundary[:, 1])),
                       jetArea = jetArea, overturnLength = length, overturnWidth = width, overturnAngle = angle,
                       overturnBack = fitBack, fitError = error, boundary = boundary.copy())

def jetGap(surface: np.ndarray, stop: int, gap: int = 15) -> tuple[float, float] | None:

    '''

    Closest approach between the jet and the face below it, and the length of the element at that
    approach, or None when the surface does not overhang.

    The jet runs from the crest to the back of the tube and the face from there to node `stop`. Every
    node on either side is measured against every element on the other, skipping pairs fewer than
    `gap` nodes apart along the surface, which sit on either side of the turn at the back of the
    tube. Once the gap is shorter than the element, the boundary solution can no longer resolve the
    water between the jet and the face, and the next steps throw the face up into a spike.

    Returns:
    --------
    tuple : (gap [m], element length [m])

    '''

    overhang = _overhang(surface)
    if overhang is None:
        return None
    _, back = overhang
    crest = int(np.argmax(surface[:, 1]))
    if not crest < back < stop - 1:
        return None

    def closest(nodes: np.ndarray, elements: np.ndarray) -> tuple[float, float]:
        a, b = surface[elements], surface[elements + 1]
        p = surface[nodes]
        ab = b - a
        lengthSquared = np.maximum(np.sum(ab**2, axis = 1), 1e-300)
        share = np.clip(np.einsum('ijk,jk->ij', p[:, None, :] - a[None, :, :], ab) / lengthSquared, 0.0, 1.0)
        distance = np.hypot(*(p[:, None, :] - a[None, :, :] - share[..., None] * ab[None, :, :]).transpose(2, 0, 1))
        apart = np.abs(nodes[:, None] - elements[None, :]) >= gap
        distance = np.where(apart, distance, np.inf)
        i, j = np.unravel_index(int(np.argmin(distance)), distance.shape)
        return float(distance[i, j]), float(np.sqrt(lengthSquared[j]))

    jet, face = np.arange(crest, back + 1), np.arange(back, stop)

    return min(closest(jet, face[:-1]), closest(face, jet[:-1]))

#----------------------------------------------------------------------#
# -- Solver -- #
#----------------------------------------------------------------------#

@dataclass
class BemConfig:

    '''

    Discretization and time stepping of a boundary element run. SI units throughout.

    Parameters:
    -----------
    surfaceSpacing : float
        Free-surface node spacing near the wave, outside the crest region [m]
    crestSpacing : float
        Free-surface node spacing around the crest once the face is steeper than `steepAngle`;
        twice this while it is gentler [m]
    crestBehind, crestAhead : float
        Arc length behind and ahead of the crest kept at the crest spacing [m]
    transition : float
        Arc length over which the spacing changes from crest to surface spacing [m]
    farSpacing : float
        Free-surface node spacing far from the wave, where the water is nearly still [m]
    nearDistance : float
        Arc length beyond the crest region kept at the surface spacing before it grows toward the
        far spacing [m]
    steepAngle : float
        Steepest surface slope, from horizontal, beyond which the crest takes its fine spacing [deg]
    curvatureShare : float
        Largest spacing as a share of the local radius of curvature, so a jet tip a few hundredths of
        the depth across still has nodes enough to stay smooth
    minimumSpacing : float
        Smallest spacing curvature or a fold can call for [m]; 0 takes half the crest spacing, finer
        nodes at a jet tip being the first to go unstable
    foldShare : float
        Largest spacing as a share of the distance to another part of the surface folded over it,
        so the boundary solution resolves the water between a landing jet and the face below
    steepRegridInterval : int
        Steps between regriddings once the face is vertical
    smoothingInterval : int
        Steps between applications of the five-point smoothing; 0 for none
    steepSmoothingInterval : int
        Steps between smoothings once the face is vertical
    bedSpacing : float
        Largest bed node spacing [m]; in shallow water the spacing falls to `bedDepthShare` of the
        still water depth, so the bed keeps pace with the surface above it
    bedDepthShare : float
        Bed spacing as a share of the local still water depth
    gravity : float
        Gravitational acceleration [m/s^2]
    courantNumber : float
        Time step as a share of sqrt(spacing / g), the period scale of the shortest wave the surface
        nodes resolve
    motionLimit : float
        Largest node displacement per step, in shortest surface spacings
    steepMotionLimit : float
        The same once the face overhangs, when the jet tip crosses the finest nodes fastest
    regridInterval : int
        Steps between regriddings
    crestRange : tuple | None
        Stretch of x, as (low, high) [m], the crest is sought in; None for the whole surface

    '''

    surfaceSpacing: float
    crestSpacing: float
    crestBehind: float
    crestAhead: float
    transition: float
    bedSpacing: float
    farSpacing: float = 0.0
    nearDistance: float = 0.0
    steepAngle: float = 20.0
    curvatureShare: float = 0.3
    minimumSpacing: float = 0.0
    foldShare: float = 0.5
    steepRegridInterval: int = 2
    smoothingInterval: int = 10
    steepSmoothingInterval: int = 3
    bedDepthShare: float = 0.6
    gravity: float = constants.gravity
    courantNumber: float = 0.5
    motionLimit: float = 1.0
    steepMotionLimit: float = 0.5
    regridInterval: int = 5
    crestRange: tuple[float, float] | None = None

class BemSolver:

    '''

    Time stepping of a free surface over a bed between two vertical walls.

    Parameters:
    -----------
    bathymetry : Bathymetry
        The bed and its still water level; the walls stand at its two ends
    config : BemConfig
        Node spacing and time stepping

    '''

    def __init__(self, bathymetry: Bathymetry, config: BemConfig) -> None:

        self.bathymetry = bathymetry
        self.config = config
        self.stillWaterLevel = bathymetry.stillWaterLevel      # [m]
        self.bed = self._bedNodes()
        self.surface: np.ndarray | None = None
        self.potential: np.ndarray | None = None
        self.time = 0.0                                         # [s]
        self.steps = 0
        self.timeStep = 0.0                                     # [s]
        self.velocity: np.ndarray | None = None

    # -- Private Helper Methods -- #
    def _bedNodes(self) -> np.ndarray:

        '''Bed nodes from wall to wall, spaced by the smaller of the bed spacing and a share of the depth.'''

        config, bathymetry = self.config, self.bathymetry
        x = np.linspace(bathymetry.startX, bathymetry.endX, 20001)
        depth = np.maximum(bathymetry.depth(x), 1e-6)
        spacing = np.clip(config.bedDepthShare * depth, 0.25 * config.crestSpacing, config.bedSpacing)
        density = np.concatenate([[0.0], np.cumsum(0.5 * (1.0 / spacing[1:] + 1.0 / spacing[:-1]) * np.diff(x))])
        count = max(4, int(round(density[-1])))
        nodesX = np.interp(np.linspace(0.0, density[-1], count + 1), density, x)
        # Keep every vertex of the bed profile, so the bed's corners are corners of the boundary
        nodesX = np.unique(np.concatenate([nodesX, bathymetry.x]))

        return np.column_stack([nodesX, bathymetry.elevation(nodesX)])

    def _spacing(self, surface: np.ndarray):

        '''

        Target spacing along the arc length: fine around the highest node, the surface spacing near
        the wave, and growing toward the far spacing beyond it. The crest takes its full fineness
        only once the face steepens; while it is gentle, twice the crest spacing resolves it.
        Wherever the surface curves sharply or folds back over itself the spacing falls further, to
        a share of the radius of curvature or of the distance across the fold.

        '''

        config = self.config
        s = arcLength(surface)
        crest = s[crestIndex(surface, config.crestRange)]
        low, high = crest - config.crestBehind, crest + config.crestAhead
        span = crestSpan(surface, config.crestRange)
        steep = frontSlopeMinimum(surface[span]) < math.cos(math.radians(config.steepAngle))
        crestSpacing = config.crestSpacing if steep else min(2.0 * config.crestSpacing, config.surfaceSpacing)
        farSpacing = max(config.farSpacing, config.surfaceSpacing)

        # Curvature from splines in arc length, smoothed over a few nodes so one noisy node does not
        # call for a cluster
        xSpline, zSpline = CubicSpline(s, surface[:, 0]), CubicSpline(s, surface[:, 1])
        curvature = np.abs(xSpline(s, 1) * zSpline(s, 2) - zSpline(s, 1) * xSpline(s, 2))
        curvature = np.convolve(np.pad(curvature, 2, mode = 'edge'), np.ones(5) / 5.0, mode = 'valid')
        smallest = config.minimumSpacing or 0.5 * config.crestSpacing

        # A jet nearing the water below sits over elements cut for still water; spacing no longer
        # than a share of the gap keeps the boundary solution resolving the water in between
        fold = np.minimum(config.foldShare * foldDistance(surface), 1e6)

        def spacing(arc):
            outside = np.maximum(low - arc, 0.0) + np.maximum(arc - high, 0.0)
            blend = np.clip(outside / config.transition, 0.0, 1.0)
            near = crestSpacing + (config.surfaceSpacing - crestSpacing) * blend
            far = (np.clip((outside - config.transition - config.nearDistance) / config.nearDistance, 0.0, 1.0)
                   if config.nearDistance > 0.0 else 0.0)
            base = near + (farSpacing - config.surfaceSpacing) * far
            curved = config.curvatureShare / np.maximum(np.interp(arc, s, curvature), 1e-12)
            folded = np.interp(arc, s, fold)
            return np.maximum(np.minimum(np.minimum(base, curved), folded), smallest)

        return spacing

    def _rates(self, surface: np.ndarray, potential: np.ndarray) -> tuple:

        '''Velocity of the surface nodes and the rate of change of their potential.'''

        solution = solveBoundary(surface, potential, self.bed, self.config.surfaceSpacing)
        velocity = surfaceVelocity(surface, potential, solution.normalDerivative)
        potentialRate = (-self.config.gravity * (surface[:, 1] - self.stillWaterLevel)
                         + 0.5 * np.sum(velocity**2, axis = 1))

        return velocity, potentialRate

    # -- Public Methods -- #
    def setSurface(self, surface: np.ndarray, potential: np.ndarray, regrid: bool = True) -> None:

        '''

        Set the free surface and its potential, optionally regridded to the configured spacing.

        Raises:
        -------
        ValueError : If the surface does not run from the left wall to the right wall.

        '''

        surface = np.asarray(surface, dtype = float)
        bathymetry = self.bathymetry
        if not (np.isclose(surface[0, 0], bathymetry.startX) and np.isclose(surface[-1, 0], bathymetry.endX)):
            raise ValueError('The free surface must run from the left wall to the right wall.')

        potential = np.asarray(potential, dtype = float)
        if regrid:
            surface, potential = regridSurface(surface, potential, self._spacing(surface))
        self.surface, self.potential = surface, potential
        self.velocity = None

    def stableTimeStep(self, velocity: np.ndarray) -> float:

        '''

        Time step: a share of sqrt(spacing / g) on the shortest surface element, and no node moving
        more than the motion limit in shortest spacings [s].

        '''

        config = self.config
        shortest = float(np.min(np.hypot(*np.diff(self.surface, axis = 0).T)))
        fastest = float(np.max(np.hypot(velocity[:, 0], velocity[:, 1])))
        timeStep = config.courantNumber * math.sqrt(shortest / config.gravity)
        # A jet tip that moves a whole spacing in a step can fold the intermediate Runge-Kutta
        # surfaces, and the boundary solutions on them tangle the tip; half a spacing keeps it clean
        limit = config.steepMotionLimit if frontSlopeMinimum(self.surface) <= 0.0 else config.motionLimit
        if fastest > 0.0:
            timeStep = min(timeStep, limit * shortest / fastest)

        return timeStep

    def step(self, timeStep: float | None = None) -> float:

        '''

        Advance one fourth-order Runge-Kutta step, regridding at the configured interval.

        Returns:
        --------
        float : The step taken [s]

        '''

        x0, phi0 = self.surface, self.potential
        k1x, k1p = self._rates(x0, phi0)
        dt = self.stableTimeStep(k1x) if timeStep is None else timeStep
        k2x, k2p = self._rates(x0 + 0.5 * dt * k1x, phi0 + 0.5 * dt * k1p)
        k3x, k3p = self._rates(x0 + 0.5 * dt * k2x, phi0 + 0.5 * dt * k2p)
        k4x, k4p = self._rates(x0 + dt * k3x, phi0 + dt * k3p)

        surface = x0 + dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        potential = phi0 + dt / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        surface[0, 0], surface[-1, 0] = x0[0, 0], x0[-1, 0]

        self.time += dt
        self.steps += 1
        self.timeStep = dt
        self.velocity = k1x
        overhanging = frontSlopeMinimum(surface) <= 0.0
        smoothing = self.config.steepSmoothingInterval if overhanging else self.config.smoothingInterval
        if smoothing and self.steps % smoothing == 0:
            surface, potential = smoothSurface(surface, potential)
        interval = self.config.steepRegridInterval if overhanging else self.config.regridInterval
        regrid = interval and self.steps % interval == 0
        self.setSurface(surface, potential, regrid = bool(regrid))

        return dt

    def volume(self) -> float:

        '''Water above still water level per unit width, the signed area under the surface [m^2].'''

        surface = self.surface
        dx = np.diff(surface[:, 0])
        height = 0.5 * (surface[1:, 1] + surface[:-1, 1]) - self.stillWaterLevel

        return float(np.sum(height * dx))

    def energy(self, density: float = constants.freshwaterDensity) -> tuple[float, float]:

        '''

        Kinetic and potential energy per unit width [J/m]: rho / 2 times the boundary integral of
        phi dphi/dn, and rho g / 2 times the integral of (z - SWL)^2 dx along the surface.

        '''

        solution = solveBoundary(self.surface, self.potential, self.bed, self.config.surfaceSpacing)

        return flowEnergy(self.surface, self.potential, solution.normalDerivative, self.stillWaterLevel,
                          self.config.gravity, density)

def flowEnergy(surface: np.ndarray, potential: np.ndarray, normalDerivative: np.ndarray, stillWaterLevel: float,
               gravity: float = constants.gravity, density: float = constants.freshwaterDensity) -> tuple[float, float]:

    '''

    Kinetic and potential energy of the water per unit width [J/m]: rho / 2 times the integral of
    phi dphi/dn over the free surface (the bed and walls carry no flux), and rho g / 2 times the
    integral of (z - SWL)^2 dx along it. Taken along the surface with dx signed, the second is the
    potential energy over still water even where the surface overhangs.

    '''

    segment = np.hypot(*np.diff(surface, axis = 0).T)
    flux = potential * normalDerivative
    kinetic = 0.5 * density * float(np.sum(0.5 * (flux[1:] + flux[:-1]) * segment))
    elevation = surface[:, 1] - stillWaterLevel
    potentialEnergy = 0.5 * density * gravity * float(np.sum(0.5 * (elevation[1:]**2 + elevation[:-1]**2)
                                                             * np.diff(surface[:, 0])))

    return kinetic, potentialEnergy

#----------------------------------------------------------------------#
# -- Flow Inside the Water -- #
#----------------------------------------------------------------------#

class FlowField:

    '''

    The potential flow inside the water at one instant: potential, velocity and pressure anywhere
    in it, from the boundary data.

    The velocity is grad(phi) from the boundary integral representation. The pressure follows from
    the unsteady Bernoulli equation,

        p / rho = -phi_t - |grad(phi)|^2 / 2 - g (z - SWL),

    which needs phi_t inside the water. phi_t is harmonic too, and solves the same mixed problem as
    phi with its own boundary data: on the free surface, where the pressure is zero, phi_t =
    -g (z - SWL) - |u|^2 / 2 with u the surface velocity; on the bed and the walls, which do not
    move, d(phi_t)/dn = 0. One more boundary solution gives it, and the representation carries it
    inside like phi.

    Parameters:
    -----------
    surface : np.ndarray
        Free-surface nodes from wall to wall, shape (N, 2) [m]
    potential : np.ndarray
        phi at the surface nodes [m^2/s]
    bed : np.ndarray
        Bed nodes from wall to wall, shape (Nb, 2) [m]
    wallSpacing : float
        Node spacing up the walls [m]
    stillWaterLevel : float
        Elevation of still water [m]
    gravity : float
        Gravitational acceleration [m/s^2]
    density : float
        Water density [kg/m^3]

    '''

    def __init__(self, surface: np.ndarray, potential: np.ndarray, bed: np.ndarray, wallSpacing: float,
                 stillWaterLevel: float, gravity: float = constants.gravity,
                 density: float = constants.freshwaterDensity) -> None:

        self.surface = np.asarray(surface, dtype = float)
        self.surfacePotential = np.asarray(potential, dtype = float)     # [m^2/s]
        self.stillWaterLevel = stillWaterLevel      # [m]
        self.gravity = gravity                      # [m/s^2]
        self.density = density                      # [kg/m^3]
        self.flow = solveBoundary(self.surface, potential, bed, wallSpacing)
        self.surfaceVelocity = surfaceVelocity(self.surface, potential, self.flow.normalDerivative)     # [m/s]
        rate = -gravity * (self.surface[:, 1] - stillWaterLevel) - 0.5 * np.sum(self.surfaceVelocity**2, axis = 1)
        self.unsteady = solveBoundary(self.surface, rate, bed, wallSpacing)

    @classmethod
    def fromSolver(cls, solver: BemSolver, density: float = constants.freshwaterDensity) -> FlowField:

        '''The flow at a solver's current state.'''

        return cls(solver.surface, solver.potential, solver.bed, solver.config.surfaceSpacing,
                   solver.stillWaterLevel, solver.config.gravity, density)

    def energy(self) -> tuple[float, float]:

        '''Kinetic and potential energy of the water per unit width [J/m]; see `flowEnergy`.'''

        return flowEnergy(self.surface, self.surfacePotential, self.flow.normalDerivative, self.stillWaterLevel,
                          self.gravity, self.density)

    def inside(self, points: np.ndarray) -> np.ndarray:

        '''True for points inside the water.'''

        return windingShare(self.flow, points) > 0.5

    def potential(self, points: np.ndarray) -> np.ndarray:

        '''phi at points inside the water [m^2/s].'''

        return interiorPotential(self.flow, points)

    def velocity(self, points: np.ndarray) -> np.ndarray:

        '''grad(phi) at points inside the water, shape (M, 2) [m/s].'''

        return interiorVelocity(self.flow, points)

    def pressure(self, points: np.ndarray, velocity: np.ndarray | None = None) -> np.ndarray:

        '''

        Pressure at points inside the water [Pa], from the unsteady Bernoulli equation; `velocity`
        there may be passed in when already known.

        '''

        if velocity is None:
            velocity = self.velocity(points)
        rate = interiorPotential(self.unsteady, points)

        return -self.density * (rate + 0.5 * np.sum(velocity**2, axis = 1)
                                + self.gravity * (points[:, 1] - self.stillWaterLevel))
