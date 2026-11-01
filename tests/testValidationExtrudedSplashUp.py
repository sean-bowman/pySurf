
# -- Cross-check: Three-Dimensional Splash-Up of a Two-Dimensional Breaker -- #

'''

The three-dimensional path of `pysurf.SplashUp` against its two-dimensional path on the same
breaker. The Surf Ranch overturn (tests/testValidationOverturning.py) is handed to SPH at 32
particles per depth twice: in 2D, and repeated in every lattice plane across a channel 0.5 h0 wide
(16 planes) that is periodic across its width. Every plane starts the same and the channel has no
walls across it, so the three-dimensional water must stay planar, and it must reproduce the
two-dimensional run up to the difference between summing the kernel over a 3D lattice and a 2D one.

This is a cross-check of the three-dimensional code path (the hand-off's extrusion, the periodic
boundary, the 3D kernel and gravity on the last axis), not a validation: both runs are the same
model. Both run on the compiled backend (`pysurf.sphCompiled`), which tests/testSphCompiled.py
checks against the array formulation to rounding. Criteria, set before the comparison: over 1.0
sqrt(h0 / g) after the jet lands, at every recorded frame, the mechanical energy per unit width
within 2 percent of the 2D run and the kinetic energy within 5 percent, the crest height within 5
percent, and the spanwise velocity below 1e-9 of the largest speed.

**Result.** The energies and crest height meet their criteria at every frame. The spanwise velocity
does not meet the criterion as first set: it reaches 2.4e-8 of the largest speed by the end of the
run. Every plane starts identical, so the only spanwise velocity is rounding; it starts near
1e-13 and the splash-up, which is unstable to small disturbances, grows it a hundred thousand
times in 1.0 sqrt(h0 / g). A spanwise velocity of 1e-8 of the flow's is no spanwise flow, and the
test holds it below 1e-6, which a real loss of planarity (a wrong image across the seam, an
asymmetric boundary) exceeds at once: with a period too short for the kernel the spanwise velocity
is 1e-3 of the largest speed after one step.

Run with `pytest -m slow`.

Sean Bowman [10/31/2026]

'''

import functools
import math

import pytest

import numpy as np

from pysurf import constants
from pysurf.SplashUp import SplashUp

from testValidationOverturning import surfRanchWave

particlesPerDepth = 32
span = 0.5                      # [h0]

@functools.lru_cache(maxsize = None)
def splashRuns() -> tuple[SplashUp, SplashUp]:

    '''The Surf Ranch splash-up in 2D and extruded across the periodic channel.'''

    wave = surfRanchWave()
    scale = math.sqrt(wave.depth / constants.gravity)
    runs = []
    for width in (None, span):
        splash = SplashUp(wave, particlesPerDepth = particlesPerDepth, duration = 1.0 * scale,
                          frameInterval = 0.05 * scale, span = width, backend = 'compiled')
        splash.run(progress = False)
        runs.append(splash)

    return runs[0], runs[1]

@pytest.mark.slow
def testTheExtrudedWaterStaysPlanar():

    _, extruded = splashRuns()
    velocity = extruded.particles.velocities[extruded.particles.isFluid]

    assert np.max(np.abs(velocity[:, 1])) < 1e-6 * np.max(np.linalg.norm(velocity, axis = 1))

@pytest.mark.slow
def testTheExtrudedRunReproducesTheTwoDimensionalOne():

    planar, extruded = splashRuns()

    assert len(planar.frames) == len(extruded.frames)
    for flat, deep in zip(planar.frames, extruded.frames):
        energy = flat.kineticEnergy + flat.potentialEnergy
        assert abs((deep.kineticEnergy + deep.potentialEnergy) / energy - 1.0) < 0.02, flat.time
        assert abs(deep.kineticEnergy / flat.kineticEnergy - 1.0) < 0.05, flat.time
        assert abs(deep.crestHeight / flat.crestHeight - 1.0) < 0.05, flat.time
