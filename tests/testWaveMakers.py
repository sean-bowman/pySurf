
# -- pySurf: Wave Maker Tests -- #

'''

Checks of pysurf.waveMakers against closed forms.

The piston stroke comes from the Biesel transfer function, which reduces to H/S = kd in shallow
water and H/S = 2 in deep water; the stroke is checked against both, and the wavenumber against
the dispersion relation. The paddle's displacement and velocity are checked against each other by
central difference once the ramp is complete.

The solitary initial condition is checked against its definition: the particles fill the profile to
within the lattice spacing, their count carries the still water plus the wave volume 2 H / kappa,
and the velocity field satisfies continuity (du/dx + dw/dy = 0), the bed condition (w = 0) and the
kinematic surface condition (w = u eta_x + eta_t with eta_t = -c eta_x).

Sean Bowman [10/11/2026]

'''

import math

import numpy as np
import pytest

from pysurf import constants, waveTheory
from pysurf.bathymetry import Bathymetry
from pysurf.sphParticles import ParticleSystem
from pysurf.waveMakers import PistonWaveMaker, WaveMakerConfig, solitaryWaveFluid, solitaryWaveVelocityField

#----------------------------------------------------------------------#
# -- Piston -- #
#----------------------------------------------------------------------#

def pistonFor(waveHeight: float, wavePeriod: float, waterDepth: float, **options) -> PistonWaveMaker:

    return PistonWaveMaker(WaveMakerConfig(waveHeight = waveHeight, wavePeriod = wavePeriod, waterDepth = waterDepth,
                                           **options))

def testTheWavenumberSolvesTheDispersionRelation():

    piston = pistonFor(0.05, 1.5, 0.5)
    omega = 2.0 * math.pi / 1.5

    assert omega**2 == pytest.approx(constants.gravity * piston.wavenumber * math.tanh(piston.wavenumber * 0.5), rel = 1e-12)
    assert piston.phaseSpeed * 1.5 == pytest.approx(piston.wavelength)

def testTheStrokeMatchesBieselInShallowWater():

    '''For kd << 1 the transfer function reduces to H/S = kd.'''

    piston = pistonFor(0.01, 30.0, 0.2)
    kd = piston.wavenumber * 0.2

    assert kd < 0.05
    assert 0.01 / piston.stroke == pytest.approx(kd, rel = 1e-3)

def testTheStrokeMatchesBieselInDeepWater():

    '''For kd >> 1 the transfer function reduces to H/S = 2.'''

    piston = pistonFor(0.05, 0.5, 5.0)

    assert piston.wavenumber * 5.0 > 20.0
    assert 0.05 / piston.stroke == pytest.approx(2.0, rel = 1e-9)

@pytest.mark.parametrize('secondOrder', [False, True])
def testThePaddleVelocityIsTheDerivativeOfItsDisplacement(secondOrder):

    piston = pistonFor(0.05, 1.5, 0.5, rampCycles = 1, useSecondOrder = secondOrder)
    dt = 1e-6

    for time in (1.6, 2.1, 2.9):
        derivative = (piston.getDisplacement(time + dt) - piston.getDisplacement(time - dt)) / (2.0 * dt)
        assert piston.getVelocity(time) == pytest.approx(derivative, rel = 1e-6, abs = 1e-9)

def testTheRampStartsThePaddleAtRest():

    piston = pistonFor(0.05, 1.5, 0.5, rampCycles = 2)

    assert piston.getDisplacement(0.0) == 0.0
    assert piston.getVelocity(0.0) == 0.0

def testThePaddleMovesOnlyTheParticlesBehindItsFace():

    fluid = np.array([[0.1, 0.1], [0.2, 0.1]])
    boundary = np.array([[-0.01, 0.1], [-0.03, 0.1], [-0.2, 0.1], [0.1, -0.01]])
    particles = ParticleSystem.assemble(fluid, boundary, 0.02, 1000.0)
    piston = pistonFor(0.05, 1.5, 0.5, rampCycles = 0)
    piston.initializeBoundaryParticles(particles, pistonX = 0.0, tolerance = 0.05)

    piston.updateBoundaryMotion(particles, 0.3)
    shift = piston.getDisplacement(0.3)

    assert particles.positions[2:4, 0] == pytest.approx([-0.01 + shift, -0.03 + shift])
    assert particles.positions[4, 0] == -0.2
    assert particles.positions[5, 0] == 0.1
    assert particles.positions[:2, 0] == pytest.approx([0.1, 0.2])

def testAnUninitializedPaddleRefusesToMove():

    with pytest.raises(RuntimeError):
        pistonFor(0.05, 1.5, 0.5).updateBoundaryMotion(ParticleSystem.assemble(np.zeros((1, 2)), np.zeros((0, 2)), 0.02, 1000.0), 0.1)

#----------------------------------------------------------------------#
# -- Solitary Wave -- #
#----------------------------------------------------------------------#

height, depth, crestX = 0.3, 1.0, 6.0

def flatBed() -> Bathymetry:

    return Bathymetry([0.0, 30.0], [0.0, 0.0], stillWaterLevel = depth)

def testTheVelocityFieldSatisfiesContinuityAndTheBedCondition():

    x = np.linspace(2.0, 10.0, 41)
    step = 1e-5
    for y in (0.0, 0.4, 0.9):
        bed = np.zeros_like(x)
        velocity = solitaryWaveVelocityField(x, np.full_like(x, y), height, depth, crestX, bed)
        dudx = (solitaryWaveVelocityField(x + step, np.full_like(x, y), height, depth, crestX, bed)[:, 0]
                - solitaryWaveVelocityField(x - step, np.full_like(x, y), height, depth, crestX, bed)[:, 0]) / (2.0 * step)
        dwdy = (solitaryWaveVelocityField(x, np.full_like(x, y + step), height, depth, crestX, bed)[:, 1]
                - solitaryWaveVelocityField(x, np.full_like(x, y - step), height, depth, crestX, bed)[:, 1]) / (2.0 * step)

        assert dudx + dwdy == pytest.approx(0.0, abs = 1e-8)
        if y == 0.0:
            assert velocity[:, 1] == pytest.approx(0.0, abs = 1e-15)

def testTheSurfaceMovesWithTheWave():

    '''At y = d + eta, w = eta_t + u eta_x, with eta_t = -c eta_x for a wave translating at c.'''

    x = np.linspace(2.0, 10.0, 41)
    eta = waveTheory.solitaryWaveProfile(x, height, depth, crestX)
    step = 1e-6
    etaSlope = (waveTheory.solitaryWaveProfile(x + step, height, depth, crestX)
                - waveTheory.solitaryWaveProfile(x - step, height, depth, crestX)) / (2.0 * step)
    u, w = solitaryWaveVelocityField(x, depth + eta, height, depth, crestX, np.zeros_like(x)).T
    celerity = waveTheory.solitaryWaveSpeed(height, depth)

    assert w == pytest.approx(-celerity * etaSlope + u * etaSlope, abs = 1e-8)

def testTheFluidFillsTheProfile():

    spacing = 0.05
    positions, velocities = solitaryWaveFluid(flatBed(), height, crestX, spacing)
    eta = waveTheory.solitaryWaveProfile(positions[:, 0], height, depth, crestX)

    assert positions.shape == velocities.shape
    assert np.min(positions[:, 1]) == pytest.approx(0.5 * spacing)
    assert np.all(positions[:, 1] < depth + eta)

    # The top particle of each column lies within one spacing of the surface
    columns = np.round(positions[:, 0] / spacing).astype(int)
    for column in np.unique(columns)[::25]:
        inColumn = columns == column
        top = positions[inColumn, 1].max()
        assert depth + eta[inColumn][0] - top < spacing

def testTheFluidCarriesTheWaveVolume():

    '''Particle count times cell area against d L + 2 H / kappa, to within one cell per column.'''

    spacing = 0.025
    bed = flatBed()
    positions, _ = solitaryWaveFluid(bed, height, crestX, spacing)
    expected = depth * (bed.endX - bed.startX) + 2.0 * height / waveTheory.solitaryWaveDecay(height, depth)
    columns = (bed.endX - bed.startX) / spacing

    assert abs(len(positions) * spacing**2 - expected) < columns * spacing**2
