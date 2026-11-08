# -- pySurf: SPH Particle Refinement -- #

'''

Particle splitting for weakly compressible SPH: a finer spacing where the flow needs one, such as
the sharpening crest and the jet of a breaking wave, while the rest of the water keeps the coarser
spacing it was laid on.

    splitParticles      each chosen fluid particle into 2^dim daughters on its own lattice refined by two
    refinementLevels    how many times each particle has split, from its mass

A fluid particle of mass m, smoothing length h and spacing s = (m / rho0)^(1/dim) splits into
2^dim daughters at the corners of a square (a cube in 3D) s/2 across, centred on the mother, each
with mass m / 2^dim and the mother's velocity, acceleration, density and pressure. Mass, momentum,
kinetic energy and the centre of mass are conserved exactly, and with the centre of mass the
potential energy, which is linear in height. The continuity equation carries each particle's
density through the split.

The daughters keep the mother's smoothing length by default, so h / s doubles where particles split
and every pair still shares one kernel. A daughter may take a shorter one (`lengthRatio`), and
particles of different lengths interact at the mean, h_ij = (h_i + h_j) / 2 (`pysurf.sphSolver`,
`pysurf.sphGpu`), but a jump in h at the edge of a split region spoils the pressure term: with two
kernels across the edge, sum_j V_j grad W_ij is no longer zero in an even lattice, and the
symmetric pressure term turns the absolute pressure into a force. Under a still water column 7.5
smoothing lengths deep, halving h over half the width drove flow at 0.07 sqrt(g d) and moved
particles 1.6 spacings in 0.2 s; the pressure-difference form, which is free of that error, let the
particles clump and the run diverge. Keeping h, the same tank stays at rest. On the Surf Ranch
crest the spacing, not h, limits the sharpening before breaking (the Shepard scheme at h = 1.3
spacings and the deltaSph scheme at 3 lose the same height at 64 particles per depth), so halving
the spacing alone does what refinement is for.

The first daughter takes the mother's place in the particle arrays and the others are appended,
so the indices of every other particle, the boundary's among them, are unchanged.

Daughters are not merged back: a particle stays fine once split, and the fine region grows behind
a moving refinement zone.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

Bookkeeping, checked exactly in tests/testSphRefinement.py: mass, momentum, kinetic energy and
centre of mass through a split, the daughters' masses, lengths and positions, and that the other
particles keep their indices. The sums with mixed smoothing lengths are checked there against the
array formulation, device against host, and the numpy and GPU solvers step a tank with mixed
lengths alike. A still tank of the deltaSph scheme (h = 2 spacings), split over half its width with
h kept, keeps its fastest particle at 0.0067 sqrt(g d) after 0.2 s against 0.0059 unsplit, inside
the uniform tank's 0.01; its particles move up to 0.37 coarse spacings, against 0.22 unsplit, as
the daughters settle out of their squares.

Not validated on breaking. On the Surf Ranch overturn (`BreakingWave` with the fluid split once in
a zone that follows the crest), four of the five touchdown criteria of Feddersen et al. (2024)
pass at 64 particles per depth and two at 128; the overturn area is 18 and 21 percent small
against a criterion of 15.

Sean Bowman [11/07/2026]

'''

# Standard library imports
from __future__ import annotations

import itertools

# Third-party imports
import numpy as np

# Local imports
from pysurf.sphParticles import ParticleSystem

def refinementLevels(particles: ParticleSystem, referenceMass: float) -> np.ndarray:

    '''

    How many times each particle has split, shape (N,): the base-2^dim logarithm of the mass of a
    particle that has not split over its own, rounded. Boundary particles of the fluid's spacing
    read zero.

    '''

    ratio = referenceMass / particles.masses
    return np.rint(np.log2(ratio) / particles.dimensions).astype(int)

def splitParticles(particles: ParticleSystem, mask: np.ndarray, referenceDensity: float,
                   referenceLength: float, lengthRatio: float = 1.0) -> int:

    '''

    Split the fluid particles `mask` selects, in place, each into 2^dim daughters.

    Parameters:
    -----------
    particles : ParticleSystem
        The particle state; its arrays are replaced by longer ones
    mask : np.ndarray
        Particles to split, shape (N,); boundary particles in it are left alone
    referenceDensity : float
        rho0, which with a particle's mass gives its spacing (m / rho0)^(1/dim) [kg/m^3]
    referenceLength : float
        The smoothing length of a particle that has not split [m]
    lengthRatio : float
        A daughter's smoothing length over its mother's: 1 keeps h, 0.5 keeps h / s

    Returns:
    --------
    int : How many particles split

    '''

    chosen = np.flatnonzero(np.asarray(mask, dtype = bool) & particles.isFluid)
    if chosen.size == 0:
        return 0
    dimensions = particles.dimensions
    count = 2**dimensions
    lengths = particles.smoothingLengths(referenceLength).astype(float, copy = True)

    # The corners of a square (cube) s/2 across around each mother
    corners = np.array(list(itertools.product((-0.25, 0.25), repeat = dimensions)))
    spacing = (particles.masses[chosen] / referenceDensity)**(1.0 / dimensions)        # [m]
    daughters = particles.positions[chosen][:, None, :] + spacing[:, None, None] * corners[None]
    mass = particles.masses[chosen] / count                                             # [kg]
    length = lengths[chosen] * lengthRatio                                              # [m]

    # The first daughter takes the mother's place, the rest go at the end
    particles.positions[chosen] = daughters[:, 0]
    particles.masses[chosen] = mass
    lengths[chosen] = length
    repeat = lambda values: np.repeat(values, count - 1, axis = 0)
    particles.positions = np.vstack([particles.positions, daughters[:, 1:].reshape(-1, dimensions)])
    particles.velocities = np.vstack([particles.velocities, repeat(particles.velocities[chosen])])
    particles.accelerations = np.vstack([particles.accelerations, repeat(particles.accelerations[chosen])])
    particles.densities = np.concatenate([particles.densities, repeat(particles.densities[chosen])])
    particles.pressures = np.concatenate([particles.pressures, repeat(particles.pressures[chosen])])
    particles.masses = np.concatenate([particles.masses, repeat(mass)])
    particles.isFluid = np.concatenate([particles.isFluid, np.ones(chosen.size * (count - 1), dtype = bool)])
    particles.lengths = np.concatenate([lengths, repeat(length)])

    return int(chosen.size)
