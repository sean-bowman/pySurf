
# -- Validation: Solitary Wave Celerity -- #

'''

The SPH solitary wave against its theoretical celerity over a flat bottom.

A solitary wave of permanent form travels at c = sqrt(g (d + H)) to first order (Rayleigh,
Boussinesq); for H/d = 0.3 the second-order (Laitone) and fully nonlinear values differ from it by
under 0.5 percent, so the first-order value is the reference here to that precision.

The wave runs down a flat channel 32 depths long with walls at both ends, built from the same
modules `pysurf.BreakingWave` uses: `waveMakers.solitaryWaveFluid` lays it in, `BedBoundary` with
an added far wall holds it, and the solver runs it at the same speed of sound and viscosity. Its
crest is the vertex of a parabola fitted through the surface within 30 percent of the peak
(`waveBreaking.profileCrest`), and its
speed is a least-squares fit of that position over 3 s, while the crest travels about 11 depths and
its leading edge stays well clear of the far wall.

At finite resolution the wave loses height as it travels, and a lower wave is a slower one. The
measured speed is therefore also compared with sqrt(g (d + H_mean)) at the mean measured height,
which separates the speed-amplitude relation from the height decay. For H/d = 0.3:

    particles per depth     8       12      16
    c / sqrt(g (d + H))     0.948   0.995   0.997
    c / sqrt(g (d + Hmean)) 0.957   0.998   0.998
    crest height lost       20 %    12 %    6.5 %   (over 3 s, about 11 depths)

The speed converges to the reference. The height decay is numerical damping; it falls by a factor of
three from 8 to 16 particles per depth, and it is the larger error for a wave that must travel tens
of depths.

Run with `pytest -m slow`.

Sean Bowman [10/11/2026]

'''

import numpy as np
import pytest

from pysurf import constants, waveTheory
from pysurf.bathymetry import Bathymetry
from pysurf.sphBoundaries import BedBoundary
from pysurf.sphParticles import ParticleSystem
from pysurf.sphSolver import SimulationConfig, WcsphSolver
from pysurf.waveBreaking import freeSurfaceProfile, profileCrest
from pysurf.waveMakers import solitaryWaveFluid

def measureCelerity(particlesPerDepth: int, heightRatio: float = 0.3, channelDepths: float = 32.0,
                    fitStart: float = 0.5, fitEnd: float = 3.5) -> dict:

    '''

    Run a solitary wave down a flat channel and fit its speed. Lengths are in depths and times in
    seconds for a 1 m depth.

    Returns:
    --------
    dict : measured and reference celerity [m/s], their ratio, and the wave height at the start and
        end of the fit window [m]

    '''

    depth, height = 1.0, heightRatio
    spacing = depth / particlesPerDepth
    reference = waveTheory.solitaryWaveSpeed(height, depth)
    crestX = waveTheory.solitaryWaveHalfLength(height, depth, 0.01)

    channel = Bathymetry([0.0, channelDepths * depth], [0.0, 0.0], stillWaterLevel = depth)
    wallTop = depth + 3.0 * height
    boundary = BedBoundary(channel, spacing, nLayers = 3, wallTop = wallTop)
    heights = np.arange(-2.5 * spacing, wallTop, spacing)
    farWall = np.vstack([np.column_stack([np.full_like(heights, channel.endX + (layer + 0.5) * spacing), heights])
                         for layer in range(3)])

    fluid, velocities = solitaryWaveFluid(channel, height, crestX, spacing)
    particles = ParticleSystem.assemble(fluid, np.vstack([boundary.positions(), farWall]), spacing,
                                        constants.freshwaterDensity, velocities)
    config = SimulationConfig(
        domainMin        = np.array([0.0, 0.0]),
        domainMax        = np.array([channel.endX, wallTop]),
        particleSpacing  = spacing,
        referenceDensity = constants.freshwaterDensity,
        maxTimeStep      = 1.0,
        speedOfSound     = 10.0 * reference,
        alphaViscosity   = 0.02,
        initialSurface   = lambda x: depth + waveTheory.solitaryWaveProfile(x, height, depth, crestX),
    )
    solver = WcsphSolver(config, boundary = boundary)
    solver.initialize(particles)

    times, crests, crestHeights = [], [], []
    nextSample = fitStart
    while solver.time < fitEnd:
        solver.step()
        if solver.time >= nextSample:
            positions = particles.positions[particles.isFluid]
            centers, elevation = freeSurfaceProfile(positions, channel.startX, channel.endX, spacing)
            eta = np.nan_to_num(elevation - depth)
            times.append(solver.time)
            crests.append(profileCrest(centers, eta))
            crestHeights.append(np.max(eta))
            nextSample += 0.05

    celerity = float(np.polyfit(times, crests, 1)[0])
    meanHeight = float(np.mean(crestHeights))

    return {'celerity': celerity, 'reference': reference, 'ratio': celerity / reference,
            'ratioAtMeanHeight': celerity / waveTheory.solitaryWaveSpeed(meanHeight, depth),
            'startHeight': float(crestHeights[0]), 'endHeight': float(crestHeights[-1]), 'meanHeight': meanHeight}

@pytest.mark.slow
def testTheSolitaryWaveTravelsAtItsCelerity():

    '''At 12 particles per depth the measured celerity is within 3 percent of sqrt(g (d + H)).'''

    result = measureCelerity(12)

    assert abs(result['ratio'] - 1.0) < 0.03, result
