
# -- pySurf: SPH Energy Budget -- #

'''

The energy budget of the weakly compressible SPH step: how much each part of the scheme adds to
the fluid's kinetic, potential and elastic energy, step by step.

In weakly compressible SPH the mechanical energy E_M = K + P and the elastic energy E_C, the work
stored by compressing the fluid, change together by the power of the dissipative terms and of the
walls (Antuono et al. 2015; Meringolo et al. 2019, eqs. 9 to 15):

    dE_M/dt + dE_C/dt = P_v + P_delta + P_s

P_v is the power the artificial viscosity dissipates, P_delta the density diffusion's and P_s the
exchange with the walls. The momentum equation's pressure term and the continuity equation trade
energy between E_M and E_C, and the trade cancels when the two are a variational pair: the
symmetric pressure gradient and the plain continuity sum, as pySurf writes them. pySurf's
continuity sum carries the kernel gradient correction of Bonet and Lok (1999), which breaks the
pairing, so the correction's share is a term of its own.

The budget is taken from the step's own arithmetic, not from those identities. In one symplectic
Euler step, v' = v + a dt, rho' = rho + R dt with R from v', x' = x + v' dt:

    the kinetic energy changes by sum_i m_i a_i . (v_i + v_i') / 2 dt exactly, and a_i splits into
        the pressure and the viscosity of fluid and of boundary neighbors, and gravity;
    the elastic energy changes by sum_i m_i (e(rho_i') - e(rho_i)), with e the elastic energy per
        unit mass (`elasticEnergy`), and R_i splits into the plain continuity sum over fluid and over
        boundary neighbors, the correction's part and the diffusion, each over the secant slope
        (e(rho') - e(rho)) / (rho' - rho) (`elasticSlope`);
    the potential energy changes by the drift.

The floor, the boundary's enforcement, shifting, the Shepard filter, splitting and merging each
add the change they make. The terms then sum to the change of K + P + E_C to rounding, which is the
budget's check on itself. Terms, in `energyTerms`, each the energy [J] it adds to K + P + E_C, per
unit depth [J/m] in 2D:

    pressureFluid      pressure forces between fluid particles, on K
    viscosityFluid     the artificial viscosity between fluid particles, on K: P_v
    pressureWall       pressure forces of boundary particles on the fluid, on K
    viscosityWall      the artificial viscosity against boundary particles, on K
    gravityWork        gravity, on K
    continuityFluid    the plain continuity sum over fluid pairs, on E_C
    continuityWall     the plain continuity sum over fluid-boundary pairs, on E_C
    correction         the kernel gradient correction's part of the continuity sum, on E_C
    diffusion          the density diffusion, on E_C: P_delta
    floor              the density floor, on E_C
    drift              moving the fluid, on P
    clamp              the boundary's enforcement, on K and P
    shifting           particle shifting, on K, P and E_C
    filter             the Shepard filter, on E_C
    refinement         splitting and merging, on K, P and E_C

`budgetGroups` gathers them by cause: viscous (P_v), diffusive (P_delta), walls (P_s, with the
enforcement), correction, time stepping (the pressure-continuity and gravity-drift pairs, which
cancel as dt goes to zero) and other.

Only fluid particles carry energy here; boundary particles have no kinetic, potential or elastic
energy of their own, as in Antuono et al.'s sums over the fluid.

References:
-----------
Antuono, M., Marrone, S., Colagrossi, A. & Bouscasse, B. (2015) -- Energy balance in the delta-SPH
    scheme, Comput. Methods Appl. Mech. Eng. 289, 209-226
Meringolo, D.D., Marrone, S., Colagrossi, A. & Liu, Y. (2019) -- A dynamic delta-SPH model: how to
    get rid of diffusive parameter tuning, Comput. Fluids 179, 334-355
Bonet, J. & Lok, T.-S.L. (1999) -- Variational and momentum preservation aspects of smooth particle
    hydrodynamic formulations, Comput. Methods Appl. Mech. Eng. 180

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

An accounting of the scheme's own terms, not a physical model, so it is checked against
identities rather than a reference. tests/testSphEnergy.py checks the elastic energy against a
numerical integral of p / rho^2 and its slope against p / rho^2, the budget's terms against the
change of K + P + E_C over steps of the array solver and of the device-resident steps (closure),
the device's terms against the array solver's, the viscous term against ever adding energy, the
correction's term against being zero without the correction, and the device step against
changing when the budget is on.

Sean Bowman [11/22/2026]

'''

# Standard library imports
from __future__ import annotations

# Third-party imports
import numpy as np

energyTerms: tuple = ('pressureFluid', 'viscosityFluid', 'pressureWall', 'viscosityWall', 'gravityWork',
                      'continuityFluid', 'continuityWall', 'correction', 'diffusion', 'floor', 'drift', 'clamp',
                      'shifting', 'filter', 'refinement')

# The terms the device steps add, in the order of the device's per-particle columns; refinement
# happens on the host between device calls
deviceTerms: tuple = energyTerms[:14]

# The groups `budgetGroups` reports, each the sum of its terms
budgetGroupTerms: dict = {
    'viscous': ('viscosityFluid',),
    'diffusive': ('diffusion',),
    'walls': ('pressureWall', 'viscosityWall', 'continuityWall', 'clamp'),
    'correction': ('correction',),
    'timeStepping': ('pressureFluid', 'continuityFluid', 'gravityWork', 'drift'),
    'other': ('floor', 'shifting', 'filter', 'refinement'),
}

# Below this share of rho0 a density change takes the slope p / rho^2 at its midpoint, where the
# secant's two energies would lose their digits to cancellation
secantThreshold: float = 1e-10

def zeroBudget() -> dict:

    '''A budget with every term at zero [J].'''

    return {name: 0.0 for name in energyTerms}

def elasticEnergy(densities: np.ndarray, referenceDensity: float, soundSpeed: float,
                  tension: bool = False) -> np.ndarray:

    '''

    The elastic energy per unit mass the Tait equation of state stores at a density [J/kg]:
    e(rho) = integral from rho0 to rho of p(s) / s^2 ds, with p = B ((rho / rho0)^7 - 1) and
    B = rho0 c0^2 / 7 (`sphSolver.taitEquationOfState`).

    Integrated, e = (c0^2 / 7) ((1 + x)^7 - 1 - 7 x) / (6 (1 + x)) with x = rho / rho0 - 1, and the
    numerator is written x^2 (21 + 35 x + 35 x^2 + 21 x^3 + 7 x^4 + x^5), so no digits cancel near
    rho0, where e = c0^2 x^2 / 2, linear acoustics' energy. Below rho0 the clamped pressure is zero
    and so is e, unless the fluid carries `tension`.

    Parameters:
    -----------
    densities : np.ndarray
        Densities [kg/m^3]
    referenceDensity : float
        rho0 [kg/m^3]
    soundSpeed : float
        c0 [m/s]
    tension : bool
        Whether the pressure goes negative below rho0 (`SimulationConfig.negativePressure`)

    Returns:
    --------
    np.ndarray : e for each density [J/kg]

    '''

    x = np.asarray(densities, dtype = float) / referenceDensity - 1.0
    polynomial = 21.0 + x * (35.0 + x * (35.0 + x * (21.0 + x * (7.0 + x))))
    energy = soundSpeed**2 / 7.0 * x * x * polynomial / (6.0 * (1.0 + x))
    if not tension:
        energy = np.where(x < 0.0, 0.0, energy)

    return energy

def elasticSlope(before: np.ndarray, after: np.ndarray, referenceDensity: float, soundSpeed: float,
                 tension: bool = False) -> np.ndarray:

    '''

    The secant slope of the elastic energy between two densities, (e(after) - e(before)) /
    (after - before) [J/kg per kg/m^3], so a density change times it is exactly the energy change.
    A change under `secantThreshold` rho0 takes p / rho^2 at the midpoint instead, the slope's limit.

    '''

    before, after = np.asarray(before, dtype = float), np.asarray(after, dtype = float)
    change = after - before
    large = np.abs(change) > secantThreshold * referenceDensity
    secant = np.divide(elasticEnergy(after, referenceDensity, soundSpeed, tension)
                       - elasticEnergy(before, referenceDensity, soundSpeed, tension),
                       change, out = np.zeros_like(change), where = large)
    middle = 0.5 * (before + after)
    pressure = referenceDensity * soundSpeed**2 / 7.0 * ((middle / referenceDensity)**7 - 1.0)
    if not tension:
        pressure = np.maximum(pressure, 0.0)

    return np.where(large, secant, pressure / middle**2)

def budgetGroups(terms: dict) -> dict:

    '''

    The budget's terms gathered by cause (`budgetGroupTerms`), with their total, the change of
    K + P + E_C [J].

    Parameters:
    -----------
    terms : dict
        Energy of each term in `energyTerms` [J]

    Returns:
    --------
    dict : viscous, diffusive, walls, correction, timeStepping, other and total [J]

    '''

    groups = {name: float(sum(terms[term] for term in members)) for name, members in budgetGroupTerms.items()}
    groups['total'] = float(sum(terms[term] for term in energyTerms))

    return groups
