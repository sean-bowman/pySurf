# -- Item 8a: the delta-LES-SPH closure a priori, on Mostert and Deike's beach -- #

'''

What the delta-LES-SPH closure (Meringolo et al. 2019, eqs. 19 to 28) would remove in active
breaking, read from the current scheme's own flow. The run is `mostertEnergy.py`'s, with the
energy budget: a solitary wave of H0/h0 = 0.3 on a 3 degree beach, `BreakingWave` on the 'deltaSph'
scheme from bed-fitted columns, with its still-water reference.

The closure sets each particle's viscosity and density diffusion from its strain rate:

    ||D|| = sqrt(2 D:D), D the symmetric part of the renormalized velocity gradient over fluid
        neighbors (`sphSolver.velocityGradients`), so free-slip walls read no shear
    nu_T = (C_S l)^2 ||D||, C_S = 0.12;  nu_delta = (C_delta l)^2 ||D||, C_delta = 1.5;  l = 2h
    alpha_i = K nu_T / (c0 h), K = 2 (n + 2);  delta_i = nu_delta / (c0 h);  each capped at 0.2
    alpha_ij = K nu / (c0 h) + 2 alpha_i alpha_j / (alpha_i + alpha_j), nu water's;
    delta_ij = 2 delta_i delta_j / (delta_i + delta_j)

and dissipates, per unit depth,

    P_alpha = h c0 rho0 sum_i sum_j alpha_ij (pi_ij / 2) (u_i - u_j) . grad_i W_ij V_i V_j    (eq. 27)
    P_delta = h c0 sum_i (p_i / rho_i^2) m_i sum_j delta_ij D_ij . grad_i W_ij V_j           (eq. 28)

with pi_ij = (u_j - u_i) . (r_j - r_i) / |r_j - r_i|^2 over every fluid pair, approaching or not, and
D_ij Antuono et al.'s (2010) pair difference (`sphSolver.densityDiffusionRate` with the renormalized
gradients, item 7's term). p / rho^2 is the Tait equation's energy per unit mass and density.

The run captures the whole particle state every `waveEvery` tracker updates from `captureFrom` to
`captureTo` (every `stillEvery` for the still reference), with the budget's running terms. Each
capture gives the closure's two powers and the current scheme's own two, the limited artificial
viscosity and the hydrostatic diffusion, rebuilt from the same state. Over active breaking, t1 to
t2 as `BreakingWave.breakingDissipation` reads them, the trapezoid rule integrates each, the still
reference's taken out.

The estimate is a priori: the flow, t1 and t2 stay as the current scheme makes them, and the
closure's dissipation replaces the scheme's:

    share_pred = share - (Q_v + Q_delta) / E_b + (Q_alpha + Q_delta,LES) / E_b

Q_v and Q_delta from the budget (`BreakingWave.breakingBudget`), the closure's from the captures,
E_b the mechanical energy at breaking. Meringolo et al. find the viscous and the diffusive
dissipation coupled, so the flow's response could move either term; this estimate leaves it out.

Readings, set before the runs (2026-11-27):

    sampling    the current scheme's two terms rebuilt from the captures within 10 percent of the
                budget's over t1 to t2; otherwise the prediction is not read against the gate
    gate        item 8 is built only if share_pred at 64 ppd exceeds 0.174
    without criteria: share_pred at 32 ppd; the closure's split between Q_alpha and Q_delta,LES;
                the share of particles at either cap; the continuum form sum rho nu_T ||D||^2 V,
                against phase 1's 114 J/m at 64 ppd with the filter width h

Checks: tests/testClosureAPriori.py holds P_alpha to mu gamma^2 per unit area for a simple shear,
mu = alpha h c0 rho0 / K; ||D|| exact for a linear velocity field on jittered particles; P_delta zero
for a linear density field; the current scheme's two powers rebuilt from a state against the array
solver's own budget over one step.

Results (2026-11-27), energies per unit width over active breaking:

    ppd   t1 to t2 (s)    share   sampling (v, delta)   scheme (v + delta)   closure (alpha + delta)   share_pred
    32    6.397 to 9.262  0.202   +0.1, -1.1 percent    91.8 + 329.8         486.1 + 285.2             0.331
    64    6.259 to 9.223  0.165   +0.2, +4.5 percent    118.7 + 337.8        466.4 + 369.6             0.303

The sampling passes at both resolutions and the gate passes at 64 ppd. No particle reaches the
alpha cap; 0.20 and 0.12 percent of the fluid reach the delta cap. The continuum form gives 634 and
534 J/m; phase 1's 114 J/m at 64 ppd, from the boundary element hand-off runs with the filter width
h, scales by (2h / h)^2 to 456 J/m, and 534 is 17 percent above it. That compares two of pySurf's
own estimates on different flows, so it is a cross-check, not a validation. Over the 0.35 s captured
before t1 at 64 ppd the closure removes 7.0 J/m and the scheme 5.1, a difference of 0.07 percent of
E_b.

Validation status: share_pred is unvalidated. No reference measures what a closure would remove
from a frozen SPH flow; Mostert and Deike's 0.31 is the target of the dynamic model, not a check of
this estimate. Its terms are checked against closed forms (the tests above) and against the
solver's own budget (the sampling check).

    python closureAPriori.py <ppd> [--captures DIR] [--workers N] [--keepCaptures]

Writes results/closureAPriori_<ppd>.json. The captures go to DIR/<ppd> (by default
D:/pySurfRuns/item8a) and are deleted after the analysis unless kept.

References:
-----------
Meringolo, D.D., Marrone, S., Colagrossi, A. & Liu, Y. (2019) -- A dynamic delta-SPH model: how to
    get rid of diffusive parameter tuning, Comput. Fluids 179, 334-355
Di Mascio, A., Antuono, M., Colagrossi, A. & Marrone, S. (2017) -- Smoothed particle hydrodynamics
    method from a large eddy simulation perspective, Phys. Fluids 29, 035102
Antuono, M., Colagrossi, A., Marrone, S. & Molteni, D. (2010) -- Free-surface flows solved by means
    of SPH schemes with numerical diffusive terms, Comput. Phys. Commun. 181, 532-549
Mostert, W. & Deike, L. (2020) -- Inertial energy dissipation in shallow-water breaking waves,
    J. Fluid Mech. 890, A12

Sean Bowman [11/27/2026]

'''

import argparse
import gc
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import common
from common import writeResult

from pysurf import constants
from pysurf.sphEnergy import energyTerms
from pysurf.sphKernels import createKernel
from pysurf.sphNeighbors import neighborPairs
from pysurf.sphSolver import (artificialViscosity, correctionMatrices, densityDiffusionRate, limitedVelocityDifference,
                              pairAccelerations, renormalizedDensityGradient, taitEquationOfState, velocityGradients)

smagorinskyConstant = 0.12      # [-] C_S
diffusionConstant = 1.5         # [-] C_delta
filterWidthRatio = 2.0          # [-] l / h, the kernel radius
coefficientCap = 0.2            # [-] the limit on alpha_i and delta_i
captureFrom = 5.9               # [s]
captureTo = 10.2                # [s]
waveEvery = 5                   # tracker updates between captures of the wave, 0.05 sqrt(h0/g)
stillEvery = 10                 # tracker updates between captures of the still reference
samplingLimit = 10.0            # [percent]
gateShare = 0.174               # [-] the plain artificial viscosity's share at 64 ppd
progressShare = 0.217           # [-] the Mostert criterion's edge, 0.31 less 30 percent
phaseOneContinuum = 114.0       # [J/m] phase 1's continuum estimate at 64 ppd, filter width h
defaultCaptures = Path('D:/pySurfRuns/item8a')

#----------------------------------------------------------------------#
# -- Scheme and state -- #
#----------------------------------------------------------------------#

@dataclass
class Scheme:

    '''

    The scalars of the scheme a capture is read with.

    Parameters:
    -----------
    smoothingLength, soundSpeed, referenceDensity : float
        h [m], c0 [m/s] and rho0 [kg/m^3]
    alpha, delta : float
        The current scheme's artificial viscosity and diffusion coefficients
    gravity : float
        Gravity's magnitude [m/s^2], for the hydrostatic subtraction
    viscosityLimiter, negativePressure : bool
        The current scheme's options
    kinematicViscosity : float
        Water's nu [m^2/s], the closure's physical part
    dimensions : int
        2 or 3

    '''

    smoothingLength: float
    soundSpeed: float
    referenceDensity: float
    alpha: float
    delta: float
    gravity: float
    viscosityLimiter: bool = True
    negativePressure: bool = False
    kinematicViscosity: float = constants.freshwaterKinematicViscosity
    dimensions: int = 2

    @classmethod
    def fromSolver(cls, solver) -> 'Scheme':
        '''The scheme a `WcsphSolver` runs.'''
        config = solver.config
        return cls(smoothingLength = config.smoothingLength, soundSpeed = solver.speedOfSound,
                   referenceDensity = config.referenceDensity, alpha = config.alphaViscosity,
                   delta = config.diffusionDelta, gravity = abs(float(np.asarray(config.gravity)[-1])),
                   viscosityLimiter = config.viscosityLimiter, negativePressure = config.negativePressure,
                   dimensions = len(np.asarray(config.gravity)))

    @property
    def factorK(self) -> float:
        '''K = 2 (n + 2), the viscous term's factor (Meringolo et al., eq. 16).'''
        return 2.0 * (self.dimensions + 2)

@dataclass
class Pairs:

    '''Every pair within the kernel's support, with r_i - r_j, its length and grad_i W_ij.'''

    i: np.ndarray
    j: np.ndarray
    separation: np.ndarray
    distance: np.ndarray
    gradients: np.ndarray
    fluid: np.ndarray           # fluid-fluid pairs

def pairsOf(positions: np.ndarray, isFluid: np.ndarray, scheme: Scheme) -> Pairs:

    '''The pairs of a particle set under the Wendland C2 kernel at h.'''

    h = scheme.smoothingLength
    i, j = neighborPairs(positions, 2.0 * h)
    separation = positions[i] - positions[j]
    distance = np.linalg.norm(separation, axis = 1)
    gradients = createKernel('wendlandC2', scheme.dimensions).gradientBatch(separation, distance, h)

    return Pairs(i, j, separation, distance, gradients, isFluid[i] & isFluid[j])

def fluidPressures(densities: np.ndarray, scheme: Scheme) -> np.ndarray:

    '''The Tait pressure of each density, clamped at zero unless the scheme carries tension [Pa].'''

    return taitEquationOfState(densities, scheme.referenceDensity, scheme.soundSpeed,
                               clampNegative = not scheme.negativePressure)

def harmonicMean(first: np.ndarray, second: np.ndarray) -> np.ndarray:

    '''2 a b / (a + b), zero where both are zero.'''

    total = first + second

    return np.divide(2.0 * first * second, total, out = np.zeros_like(total), where = total > 0.0)

def pairToParticles(values: np.ndarray, i: np.ndarray, j: np.ndarray, count: int) -> np.ndarray:

    '''Each pair's value split evenly between its two members.'''

    return 0.5 * (np.bincount(i, weights = values, minlength = count) + np.bincount(j, weights = values, minlength = count))

#----------------------------------------------------------------------#
# -- The closure -- #
#----------------------------------------------------------------------#

def strainRateNorms(masses: np.ndarray, densities: np.ndarray, velocities: np.ndarray, isFluid: np.ndarray,
                    pairs: Pairs) -> tuple[np.ndarray, np.ndarray]:

    '''

    Each fluid particle's ||D|| = sqrt(2 D:D) [1/s], with its velocity gradient (N, dim, dim) [1/s].

    D is the symmetric part of the gradient renormalized over fluid neighbors (Meringolo et al., eq.
    23, with `sphSolver.velocityGradients`): exact for a linear field wherever the matrix inverts.

    '''

    gradient = velocityGradients(masses, densities, velocities, pairs.i, pairs.j, pairs.separation, pairs.gradients,
                                 isFluid)
    strain = 0.5 * (gradient + np.transpose(gradient, (0, 2, 1)))

    return np.sqrt(2.0 * np.sum(strain * strain, axis = (1, 2))), gradient

def closureCoefficients(norms: np.ndarray, scheme: Scheme) -> tuple[np.ndarray, np.ndarray]:

    '''

    Each particle's alpha_i and delta_i, capped at `coefficientCap` (Meringolo et al., eqs. 21, 25
    and 26).

    '''

    h, c0 = scheme.smoothingLength, scheme.soundSpeed
    width = filterWidthRatio * h
    eddyViscosity = (smagorinskyConstant * width)**2 * norms
    diffusivity = (diffusionConstant * width)**2 * norms
    alphas = np.minimum(scheme.factorK * eddyViscosity / (c0 * h), coefficientCap)
    deltas = np.minimum(diffusivity / (c0 * h), coefficientCap)

    return alphas, deltas

def closureViscousPower(masses: np.ndarray, densities: np.ndarray, velocities: np.ndarray, pairs: Pairs,
                        alphas: np.ndarray, scheme: Scheme) -> np.ndarray:

    '''

    The power the closure's viscous term adds to each particle's kinetic energy [W/m in 2D],
    Meringolo et al.'s eq. 27 over fluid pairs, each pair's share split between its members. Never
    positive.

    '''

    h, c0, rho0 = scheme.smoothingLength, scheme.soundSpeed, scheme.referenceDensity
    fluid = pairs.fluid
    i, j = pairs.i[fluid], pairs.j[fluid]
    separation, distance, gradients = pairs.separation[fluid], pairs.distance[fluid], pairs.gradients[fluid]
    volumes = masses / densities
    water = scheme.factorK * scheme.kinematicViscosity / (c0 * h)
    coefficient = water + harmonicMean(alphas[i], alphas[j])
    approach = np.sum((velocities[i] - velocities[j]) * separation, axis = 1)
    # pi_ij = (u_j - u_i) . (r_j - r_i) / r^2 = (u_i - u_j) . (r_i - r_j) / r^2
    pi = approach / distance**2
    power = h * c0 * rho0 * coefficient * pi * np.sum((velocities[i] - velocities[j]) * gradients, axis = 1) \
        * volumes[i] * volumes[j]

    return pairToParticles(power, i, j, len(densities))

def closureDiffusivePower(masses: np.ndarray, densities: np.ndarray, isFluid: np.ndarray, pairs: Pairs,
                          deltas: np.ndarray, scheme: Scheme) -> np.ndarray:

    '''

    The power the closure's density diffusion adds to each fluid particle's elastic energy [W/m in
    2D], Meringolo et al.'s eq. 28: m_i (p_i / rho_i^2) D_i, with D_i item 7's term at delta_ij over
    fluid pairs and the renormalized gradients over every neighbor, as the solver builds them.

    '''

    count = len(densities)
    corrections = correctionMatrices(masses, densities, pairs.i, pairs.j, pairs.separation, pairs.gradients, isFluid)
    densityGradients = renormalizedDensityGradient(masses, densities, pairs.i, pairs.j, pairs.gradients, corrections,
                                                   isFluid)
    fluid = pairs.fluid
    i, j = pairs.i[fluid], pairs.j[fluid]
    pairDelta = harmonicMean(deltas[i], deltas[j])
    rate = densityDiffusionRate(masses, densities, i, j, pairs.separation[fluid], pairs.distance[fluid],
                                pairs.gradients[fluid], count, scheme.smoothingLength, scheme.soundSpeed, pairDelta,
                                scheme.referenceDensity, gradients = densityGradients)

    return np.where(isFluid, elasticPower(masses, densities, rate, scheme), 0.0)

#----------------------------------------------------------------------#
# -- The current scheme's terms -- #
#----------------------------------------------------------------------#

def schemeViscousAccelerations(masses: np.ndarray, densities: np.ndarray, velocities: np.ndarray, isFluid: np.ndarray,
                               pairs: Pairs, velocityGradient: np.ndarray, scheme: Scheme) -> np.ndarray:

    '''

    The current scheme's artificial viscosity between fluid particles, as each fluid particle's
    acceleration [m/s^2], as `WcsphSolver._computeRates` builds the term: Monaghan's form on
    approaching pairs, with the limiter's reconstruction when the scheme has it.

    '''

    h, count = scheme.smoothingLength, len(densities)
    relative = velocities[pairs.i] - velocities[pairs.j]
    reconstructed = None
    if scheme.viscosityLimiter:
        reconstructed = limitedVelocityDifference(relative, pairs.separation, pairs.distance, h,
                                                  velocityGradient[pairs.i], velocityGradient[pairs.j])
        reconstructed[~pairs.fluid] = relative[~pairs.fluid]
    viscosity = artificialViscosity(relative, pairs.separation, pairs.distance,
                                    0.5 * (densities[pairs.i] + densities[pairs.j]), h, scheme.soundSpeed, scheme.alpha,
                                    reconstructed)
    fluid = pairs.fluid
    accelerations = pairAccelerations(masses, densities, np.zeros(count), pairs.i[fluid], pairs.j[fluid],
                                      pairs.gradients[fluid], viscosity[fluid], count)
    accelerations[~isFluid] = 0.0

    return accelerations

def schemeDiffusionRate(masses: np.ndarray, densities: np.ndarray, isFluid: np.ndarray, pairs: Pairs,
                        scheme: Scheme) -> np.ndarray:

    '''The current scheme's hydrostatic diffusion, each fluid particle's density rate over fluid pairs [kg/(m^3 s)].'''

    rate = densityDiffusionRate(masses, densities, pairs.i, pairs.j, pairs.separation, pairs.distance, pairs.gradients,
                                len(densities), scheme.smoothingLength, scheme.soundSpeed, scheme.delta,
                                scheme.referenceDensity, scheme.gravity, active = pairs.fluid)

    return np.where(isFluid, rate, 0.0)

def elasticPower(masses: np.ndarray, densities: np.ndarray, rates: np.ndarray, scheme: Scheme) -> np.ndarray:

    '''The power a density rate adds to each particle's elastic energy, m (p / rho^2) R [W/m in 2D].'''

    return masses * fluidPressures(densities, scheme) / densities**2 * rates

def statePowers(positions: np.ndarray, velocities: np.ndarray, densities: np.ndarray, masses: np.ndarray,
                isFluid: np.ndarray, scheme: Scheme) -> dict:

    '''

    The closure's and the current scheme's powers for one particle state, summed over the fluid
    [W/m in 2D], with the closure's readings.

    '''

    pairs = pairsOf(positions, isFluid, scheme)
    norms, gradient = strainRateNorms(masses, densities, velocities, isFluid, pairs)
    alphas, deltas = closureCoefficients(norms, scheme)
    h, c0 = scheme.smoothingLength, scheme.soundSpeed
    width = filterWidthRatio * h
    eddyViscosity = alphas * c0 * h / scheme.factorK
    uncapped = (smagorinskyConstant * width)**2 * norms
    volumes = masses / densities
    fluidNorms = norms[isFluid]

    return {
        'closureViscous': float(np.sum(closureViscousPower(masses, densities, velocities, pairs, alphas, scheme))),
        'closureDiffusive': float(np.sum(closureDiffusivePower(masses, densities, isFluid, pairs, deltas, scheme))),
        'schemeViscous': float(np.sum(masses * np.sum(velocities * schemeViscousAccelerations(
            masses, densities, velocities, isFluid, pairs, gradient, scheme), axis = 1))),
        'schemeDiffusive': float(np.sum(elasticPower(masses, densities, schemeDiffusionRate(
            masses, densities, isFluid, pairs, scheme), scheme))),
        # The continuum form's power, sum rho nu_T ||D||^2 V, removed energy counted negative
        'continuum': -float(np.sum((densities * eddyViscosity * norms**2 * volumes)[isFluid])),
        'continuumUncapped': -float(np.sum((densities * uncapped * norms**2 * volumes)[isFluid])),
        'alphaCapped': float(np.mean(alphas[isFluid] >= coefficientCap)),
        'deltaCapped': float(np.mean(deltas[isFluid] >= coefficientCap)),
        'strainRateP99': float(np.percentile(fluidNorms, 99.0)),
        'largestEddyViscosity': float(np.max(eddyViscosity[isFluid])),
    }

#----------------------------------------------------------------------#
# -- Capture and analysis -- #
#----------------------------------------------------------------------#

def attachCapture(wave, directory: Path, every: int) -> None:

    '''

    Save the whole particle state and the budget's running terms at every `every`-th tracker update
    between `captureFrom` and `captureTo`. The wave must be built; the solver pulls the full state
    from the device at each tracker instant.

    '''

    directory.mkdir(parents = True, exist_ok = True)
    particles = wave.particles
    np.savez(directory / 'meta.npz', masses = particles.masses, isFluid = particles.isFluid)
    tracker = wave.tracker
    original = tracker.update
    counter = {'updates': 0}

    def update(trackTime, positions, volumes = None):
        original(trackTime, positions, volumes)
        counter['updates'] += 1
        if captureFrom <= trackTime <= captureTo and counter['updates'] % every == 0:
            p, budget = wave.particles, wave.solver.energyBudget
            np.savez(directory / f'capture_{counter["updates"]:06d}.npz', time = wave.solver.time,
                     positions = p.positions, velocities = p.velocities, densities = p.densities,
                     budget = np.array([budget[name] for name in energyTerms]))

    tracker.update = update

def analyzeCapture(path: str, scheme: dict) -> dict:

    '''One capture's powers (`statePowers`), with its time and the budget's running terms.'''

    meta = np.load(Path(path).parent / 'meta.npz')
    capture = np.load(path)
    powers = statePowers(capture['positions'], capture['velocities'], capture['densities'], meta['masses'],
                         meta['isFluid'], Scheme(**scheme))
    powers['time'] = float(capture['time'])
    powers['budget'] = dict(zip(energyTerms, capture['budget'].tolist()))

    return powers

def analyzeDirectory(directory: Path, scheme: Scheme, workers: int) -> list[dict]:

    '''Every capture in a directory, in time order, over `workers` processes.'''

    paths = sorted(str(path) for path in directory.glob('capture_*.npz'))
    settings = dict(scheme.__dict__)
    if workers <= 1:
        return [analyzeCapture(path, settings) for path in paths]
    from joblib import Parallel, delayed
    # The worker processes find pysurf and this folder through the environment
    os.environ['PYTHONPATH'] = os.pathsep.join([str(common.repoRoot / 'src'), str(common.here),
                                                os.environ.get('PYTHONPATH', '')])

    return list(Parallel(n_jobs = workers)(delayed(analyzeCapture)(path, settings) for path in paths))

def integrateWindow(times: np.ndarray, values: np.ndarray, start: float, end: float) -> float:

    '''The trapezoid integral of a sampled series from `start` to `end`, the ends interpolated.'''

    if start < times[0] or end > times[-1]:
        raise ValueError(f'The window {start:.3f} to {end:.3f} s runs past the captures ({times[0]:.3f} to '
                         f'{times[-1]:.3f} s).')
    inside = (times > start) & (times < end)
    nodes = np.concatenate([[start], times[inside], [end]])
    samples = np.concatenate([[np.interp(start, times, values)], values[inside], [np.interp(end, times, values)]])

    return float(np.sum(0.5 * (samples[1:] + samples[:-1]) * np.diff(nodes)))

def windowEnergies(series: list[dict], start: float, end: float) -> dict:

    '''Each power in a capture series integrated over the window [J/m].'''

    times = np.array([entry['time'] for entry in series])
    names = ('closureViscous', 'closureDiffusive', 'schemeViscous', 'schemeDiffusive', 'continuum', 'continuumUncapped')

    return {name: integrateWindow(times, np.array([entry[name] for entry in series]), start, end) for name in names}

#----------------------------------------------------------------------#
# -- The case -- #
#----------------------------------------------------------------------#

def main(ppd: int, captureRoot: Path, workers: int, keepCaptures: bool) -> dict:

    '''Run the case at `ppd`, read the closure a priori and report it against the readings.'''

    from mostertEnergy import makeWave

    started = time.perf_counter()
    directory = captureRoot / str(ppd)
    if directory.exists():
        shutil.rmtree(directory)
    still = makeWave(ppd, 'still', None, energyBudget = True)
    still.build()
    attachCapture(still, directory / 'still', stillEvery)
    still.run(progress = False)
    print(f'{ppd} ppd still water: {time.perf_counter() - started:.0f} s', flush = True)
    wave = makeWave(ppd, 'exact', None, energyBudget = True)
    wave.build()
    attachCapture(wave, directory / 'wave', waveEvery)
    wave.run(progress = True)
    print(f'{ppd} ppd wave: {time.perf_counter() - started:.0f} s', flush = True)

    dissipation = wave.breakingDissipation(reference = still)
    budget = wave.breakingBudget(reference = still)
    start, end = dissipation['breakingTime'], dissipation['activeEndTime']
    scheme = Scheme.fromSolver(wave.solver)
    # The runs' frames and device arrays are not needed past here; the analysis wants the memory
    del still, wave
    gc.collect()
    waveSeries = analyzeDirectory(directory / 'wave', scheme, workers)
    stillSeries = analyzeDirectory(directory / 'still', scheme, workers)
    print(f'{ppd} ppd analysis: {len(waveSeries)} and {len(stillSeries)} captures, '
          f'{time.perf_counter() - started:.0f} s', flush = True)
    waveEnergy, stillEnergy = windowEnergies(waveSeries, start, end), windowEnergies(stillSeries, start, end)
    # Energy removed over active breaking, the still reference's taken out [J/m]
    removed = {name: -(waveEnergy[name] - stillEnergy[name]) for name in waveEnergy}

    breakingEnergy = dissipation['breakingEnergy']
    share = dissipation['energyShareLost']
    budgetViscous, budgetDiffusive = -budget['groups']['viscous'], -budget['groups']['diffusive']
    sampling = {
        'viscous': {'captures': removed['schemeViscous'], 'budget': budgetViscous,
                    'error': common.percent(removed['schemeViscous'], budgetViscous)},
        'diffusive': {'captures': removed['schemeDiffusive'], 'budget': budgetDiffusive,
                      'error': common.percent(removed['schemeDiffusive'], budgetDiffusive)},
    }
    samplingPasses = all(abs(entry['error']) <= samplingLimit for entry in sampling.values())
    closureRemoved = removed['closureViscous'] + removed['closureDiffusive']
    predicted = share + (closureRemoved - budgetViscous - budgetDiffusive) / breakingEnergy
    predictedFromCaptures = share + (closureRemoved - removed['schemeViscous'] - removed['schemeDiffusive']) / breakingEnergy

    inWindow = [entry for entry in waveSeries if start <= entry['time'] <= end]
    record = {
        'case': 'closureAPriori', 'particlesPerDepth': ppd,
        'closure': {'smagorinskyConstant': smagorinskyConstant, 'diffusionConstant': diffusionConstant,
                    'filterWidthOverSmoothingLength': filterWidthRatio, 'coefficientCap': coefficientCap},
        'scheme': dict(scheme.__dict__),
        'window': {'breakingTime': start, 'activeEndTime': end, 'waveCaptures': len(waveSeries),
                   'stillCaptures': len(stillSeries), 'capturesInWindow': len(inWindow)},
        'shareLost': share, 'breakingEnergy': breakingEnergy,
        'removed': removed, 'budget': {'viscous': budgetViscous, 'diffusive': budgetDiffusive,
                                       'shareOfLoss': budget['shareOfLoss']},
        'sampling': dict(sampling, limit = samplingLimit, passes = samplingPasses),
        'prediction': {'share': predicted, 'shareFromCaptures': predictedFromCaptures,
                       'closureViscousShare': removed['closureViscous'] / closureRemoved,
                       'gateShare': gateShare, 'progressShare': progressShare},
        'readings': {
            'alphaCappedMean': float(np.mean([entry['alphaCapped'] for entry in inWindow])),
            'deltaCappedMean': float(np.mean([entry['deltaCapped'] for entry in inWindow])),
            'strainRateP99Max': float(max(entry['strainRateP99'] for entry in inWindow)),
            'largestEddyViscosity': float(max(entry['largestEddyViscosity'] for entry in inWindow)),
            'schemeEffectiveViscosity': scheme.alpha * scheme.soundSpeed * scheme.smoothingLength / scheme.factorK,
            'phaseOneContinuumAt64': phaseOneContinuum,
        },
        'series': {'wave': waveSeries, 'still': stillSeries},
    }
    if ppd == 64:
        record['gate'] = {'passes': bool(samplingPasses and predicted > gateShare), 'samplingPasses': samplingPasses}
    record['wall'] = time.perf_counter() - started

    print(f'{ppd} ppd, active breaking {start:.3f} to {end:.3f} s ({len(inWindow)} captures): share lost {share:.4f}, '
          f'energy at breaking {breakingEnergy:.1f} J/m', flush = True)
    print(f'  current scheme removes: viscosity {budgetViscous:.1f} J/m (captures {removed["schemeViscous"]:.1f}, '
          f'{sampling["viscous"]["error"]:+.1f} percent), diffusion {budgetDiffusive:.1f} J/m (captures '
          f'{removed["schemeDiffusive"]:.1f}, {sampling["diffusive"]["error"]:+.1f} percent); sampling within '
          f'{samplingLimit:.0f} percent: {"passes" if samplingPasses else "fails"}', flush = True)
    print(f'  closure would remove: viscosity {removed["closureViscous"]:.1f} J/m, diffusion '
          f'{removed["closureDiffusive"]:.1f} J/m; continuum form {removed["continuum"]:.1f} J/m '
          f'(uncapped {removed["continuumUncapped"]:.1f})', flush = True)
    print(f'  predicted share {predicted:.4f} (from the captures alone {predictedFromCaptures:.4f}), against the '
          f'gate {gateShare} and Mostert and Deike 0.31', flush = True)
    print(f'  caps: alpha {100 * record["readings"]["alphaCappedMean"]:.2f} and delta '
          f'{100 * record["readings"]["deltaCappedMean"]:.2f} percent of the fluid; largest eddy viscosity '
          f'{record["readings"]["largestEddyViscosity"]:.2e} m2/s against the scheme alpha c0 h / K '
          f'{record["readings"]["schemeEffectiveViscosity"]:.2e}', flush = True)
    if 'gate' in record:
        print(f'  gate: {"build item 8" if record["gate"]["passes"] else "do not build item 8"}', flush = True)

    writeResult(f'closureAPriori_{ppd}', record)
    if not keepCaptures:
        shutil.rmtree(directory)

    return record

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Item 8a: the delta-LES-SPH closure a priori on Mostert and '
                                                   "Deike's beach.")
    parser.add_argument('ppd', type = int, help = 'particles per depth')
    parser.add_argument('--captures', type = Path, default = defaultCaptures, help = 'folder for the captures')
    parser.add_argument('--workers', type = int, default = 3, help = 'processes for the analysis')
    parser.add_argument('--keepCaptures', action = 'store_true', help = 'keep the captures after the analysis')
    arguments = parser.parse_args()
    main(arguments.ppd, arguments.captures, arguments.workers, arguments.keepCaptures)
