
# -- pySurf: SGN to Boundary Element Hand-off -- #

'''

A wave train from the Serre-Green-Naghdi model handed to the boundary element model shortly before
a wave breaks, so its overturn comes from fully nonlinear potential flow.

The SGN model carries the depth h and the discharge q = h u over a fixed grid; the boundary element
model needs the free surface and the velocity potential along it. The surface is the SGN surface.
The potential follows from how the surface moves. SGN conserves mass exactly, so the surface rises
at eta_t = -dq/dx; the kinematic condition makes the water's normal velocity there

    dphi/dn = eta_t / sqrt(1 + eta_x^2),

and with no flux through the bed and the walls the potential solves a Neumann problem
(`bemSolver.solveNeumann`). No expansion in depth over wavelength enters, so the hand-off is as
good on a 1:10 reef face as on a gentle beach. The tank's walls stand where the discharge crosses
zero, so they carry no flux at the hand-off, as walls must.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

tests/testSgnHandoff.py hands off a linear progressive wave, whose discharge, rise and surface
potential are known in closed form, and checks the potential to 0.2 percent of g a / omega; the
Neumann solve is checked against the same wave in tests/testBemSolver.py. How faithfully the SGN
model carries a wave train to the hand-off is its own validation (`pysurf.sgnSolver`).

Sean Bowman [10/14/2026]

'''

# Standard library imports
from __future__ import annotations

# Third-party imports
import numpy as np

# Local imports
from pysurf.bathymetry import Bathymetry
from pysurf.bemSolver import solveNeumann

def dischargeZeroCrossing(x: np.ndarray, discharge: np.ndarray, near: float, reach: float = 0.0) -> float:

    '''

    Where a wall can stand: a zero crossing of the discharge, interpolated between cells [m].

    With no `reach` it is the crossing nearest to x = `near`; with one, the crossing within `reach`
    of `near` where the surface rises slowest, |eta_t| = |dq/dx| least. The wall carries no flux at
    the hand-off either way, but water rising fast against it piles up there: at a bore's front the
    discharge can cross zero while the surface rises at its fastest. In a progressive wave, where q
    follows eta, the slowest rise is where the surface is flattest.

    Raises:
    -------
    ValueError : If the discharge does not cross zero.

    '''

    sign = np.sign(discharge)
    crossings = np.flatnonzero(sign[:-1] * sign[1:] < 0.0)
    if crossings.size == 0:
        raise ValueError('The discharge does not cross zero, so no wall can stand without flux through it.')
    share = discharge[crossings] / (discharge[crossings] - discharge[crossings + 1])
    positions = x[crossings] + share * (x[crossings + 1] - x[crossings])
    nearest = float(positions[int(np.argmin(np.abs(positions - near)))])
    if reach <= 0.0:
        return nearest

    within = np.abs(positions - near) <= reach
    if not within.any():
        return nearest
    rise = np.abs(np.interp(positions[within], x, np.gradient(discharge, x)))

    return float(positions[within][int(np.argmin(rise))])

def smoothSteepFronts(x: np.ndarray, surface: np.ndarray, discharge: np.ndarray, fromX: float, width: float,
                      steepest: float = 0.2, passes: int = 20) -> tuple[np.ndarray, np.ndarray]:

    '''

    The surface and discharge with every front beyond x = `fromX` steeper than `steepest` smoothed
    out: a running mean `width` wide, applied to both alike so the water's mass and flux keep their
    totals, over the stretch around each steep front, until none is left. An SGN bore keeps a front
    of 30 to 40 degrees that potential flow cannot start from; a front of 11 degrees it can.

    Returns:
    --------
    tuple : (surface [m], discharge [m^2/s]) on the same cells

    '''

    surface, discharge = surface.copy(), discharge.copy()
    cells = max(3, int(round(width / (x[1] - x[0]))) | 1)
    kernel = np.ones(cells) / cells
    for _ in range(passes):
        steep = (x > fromX) & (np.abs(np.gradient(surface, x)) > steepest)
        if not steep.any():
            break
        # The stretch a width either side of the steep cells
        near = np.convolve(steep.astype(float), np.ones(2 * cells + 1), mode = 'same') > 0.0
        near &= x > fromX
        for values in (surface, discharge):
            smoothed = np.convolve(np.pad(values, cells, mode = 'edge'), kernel, mode = 'same')[cells:-cells]
            values[near] = smoothed[near]

    return surface, discharge

def sgnSurfaceState(x: np.ndarray, surface: np.ndarray, discharge: np.ndarray, bathymetry: Bathymetry,
                    bedSpacing: float) -> tuple[np.ndarray, np.ndarray]:

    '''

    The free surface and its potential over a tank from the bathymetry's start to its end, from an
    SGN state on a grid that covers it.

    Parameters:
    -----------
    x : np.ndarray
        SGN cell centers, evenly spaced [m]
    surface : np.ndarray
        Free-surface elevation at the cell centers [m]
    discharge : np.ndarray
        Discharge q = h u at the cell centers [m^2/s]
    bathymetry : Bathymetry
        The tank's bed, whose ends are its walls; each should stand where the discharge crosses
        zero (`dischargeZeroCrossing`)
    bedSpacing : float
        Node spacing along the bed for the potential's solve [m]

    Returns:
    --------
    tuple : (surface nodes from wall to wall, shape (N, 2) [m], potential at them [m^2/s])

    '''

    low, high = bathymetry.startX, bathymetry.endX
    rise = -np.gradient(discharge, x)
    inside = (x > low) & (x < high)
    nodesX = np.concatenate([[low], x[inside], [high]])
    elevation = np.interp(nodesX, x, surface)
    nodeRise = np.interp(nodesX, x, rise)
    slope = np.gradient(elevation, nodesX)
    nodes = np.column_stack([nodesX, elevation])

    # The bed sampled evenly, its vertices kept so its corners are the boundary's
    sampleX = np.linspace(low, high, max(3, int(np.ceil((high - low) / bedSpacing)) + 1))
    bedX = np.unique(np.concatenate([sampleX, bathymetry.x[(bathymetry.x > low) & (bathymetry.x < high)]]))
    bed = np.column_stack([bedX, bathymetry.elevation(bedX)])

    solution = solveNeumann(nodes, nodeRise / np.sqrt(1.0 + slope**2), bed, bedSpacing)

    return nodes, solution.surfacePotential
