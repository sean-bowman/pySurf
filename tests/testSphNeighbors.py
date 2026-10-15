
# -- pySurf: SPH Neighbor Search Tests -- #

'''

Checks of pysurf.sphNeighbors.neighborPairs against a brute-force distance matrix.

Sean Bowman [10/11/2026]

'''

import numpy as np
import pytest

from pysurf.sphNeighbors import neighborPairs

def bruteForcePairs(positions: np.ndarray, radius: float) -> set:

    difference = positions[:, None, :] - positions[None, :, :]
    distance = np.linalg.norm(difference, axis = 2)
    i, j = np.nonzero(np.triu(distance < radius, k = 1))

    return set(zip(i.tolist(), j.tolist()))

@pytest.mark.parametrize('dimensions', [2, 3])
def testPairsMatchBruteForce(dimensions):

    rng = np.random.default_rng(11)
    positions = rng.uniform(0.0, 1.0, size = (400, dimensions))
    i, j = neighborPairs(positions, 0.12)

    assert np.all(i < j)
    assert len(set(zip(i.tolist(), j.tolist()))) == len(i)
    assert set(zip(i.tolist(), j.tolist())) == bruteForcePairs(positions, 0.12)

def testALatticeHasItsKnownNeighborCount():

    '''An interior point of a unit square lattice has 8 neighbors within 1.5 spacings.'''

    axis = np.arange(5.0)
    positions = np.column_stack([grid.ravel() for grid in np.meshgrid(axis, axis)])
    i, j = neighborPairs(positions, 1.5)
    center = 12
    count = np.sum(i == center) + np.sum(j == center)

    assert count == 8

def testPairsAtTheRadiusCarryNoWeight():

    '''Points exactly at the radius may be listed; any listed pair lies within it to rounding.'''

    positions = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    i, j = neighborPairs(positions, 1.0)
    distances = np.linalg.norm(positions[i] - positions[j], axis = 1)

    assert np.all(distances <= 1.0 + 1e-12)
    assert (0, 2) not in set(zip(i.tolist(), j.tolist()))

@pytest.mark.parametrize('positions', [np.zeros((0, 2)), np.zeros((1, 2)), np.array([[0.0, 0.0], [5.0, 5.0]])])
def testNoPairsGiveEmptyIndexArrays(positions):

    i, j = neighborPairs(positions, 1.0)

    assert i.size == 0 and j.size == 0
    assert i.dtype == np.intp
