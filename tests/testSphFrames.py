# -- pySurf: SPH Frame File Tests -- #

'''

Checks of pysurf.sphFrames: frames written and read back.

Sean Bowman [11/05/2026]

'''

import numpy as np
import pytest

from pysurf.sphFrames import FrameWriter, readFrame, readIndex
from pysurf.sphParticles import ParticleSystem

def particleState(seed: int) -> ParticleSystem:

    rng = np.random.default_rng(seed)
    particles = ParticleSystem.assemble(rng.uniform(size = (50, 3)), rng.uniform(size = (20, 3)), 0.1, 1000.0,
                                        rng.normal(size = (50, 3)))
    return particles

@pytest.mark.parametrize('compress', [False, True])
def testFramesReadBackAsWritten(tmp_path, compress):

    writer = FrameWriter(tmp_path / 'run', compress = compress)
    states = [particleState(seed) for seed in range(3)]
    for number, particles in enumerate(states):
        writer.write(0.1 * number, particles, step = 10 * number, fields = {'tag': np.arange(50) + number},
                     scalars = {'crestHeight': 0.5 + number})
    index = readIndex(tmp_path / 'run')

    assert [entry['time'] for entry in index] == pytest.approx([0.0, 0.1, 0.2])
    assert [entry['step'] for entry in index] == [0, 10, 20]
    assert [entry['crestHeight'] for entry in index] == pytest.approx([0.5, 1.5, 2.5])
    for number, particles in enumerate(states):
        frame = readFrame(tmp_path / 'run', number)
        fluid = particles.isFluid
        assert frame['positions'].dtype == np.float32
        assert frame['positions'] == pytest.approx(particles.positions[fluid], rel = 1e-6)
        assert frame['speeds'] == pytest.approx(np.linalg.norm(particles.velocities[fluid], axis = 1), rel = 1e-6)
        assert np.array_equal(frame['tag'], np.arange(50) + number)

def testARunCutShortReadsToItsLastFrame(tmp_path):

    # The index is rewritten after every frame
    writer = FrameWriter(tmp_path)
    writer.write(0.0, particleState(0))
    writer.write(0.5, particleState(1))
    del writer

    assert len(readIndex(tmp_path)) == 2
    assert readFrame(tmp_path, 1)['positions'].shape == (50, 3)

def testAFieldOfTheWrongLengthIsRejected(tmp_path):

    with pytest.raises(ValueError):
        FrameWriter(tmp_path).write(0.0, particleState(0), fields = {'tag': np.arange(70)})
