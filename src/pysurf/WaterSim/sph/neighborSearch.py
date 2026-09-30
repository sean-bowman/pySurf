# -- Spatial Hash Grid for Neighbor Search -- #

'''

Cell-linked list spatial hashing for O(N) neighbor search in SPH.

Divides the domain into uniform grid cells of size equal to the
kernel support radius. For each particle query, only the cell
containing the particle and its immediate neighbors (9 cells in 2D,
27 in 3D) are searched.

The pair finding uses vectorized NumPy distance computations
within cell groups for performance.

References:
-----------
Ihmsen et al. (2011) -- Parallel Neighbor-Search for SPH
Green (2010) -- Particle Simulation using CUDA

Sean Bowman [02/05/2026]

'''

from __future__ import annotations

from typing import Protocol

import numpy as np

#--------------------------------------------------------------------#
# -- Neighbor Search Protocol -- #
#--------------------------------------------------------------------#

class NeighborSearch(Protocol):

    '''Protocol for neighbor search algorithms.'''

    def build(self, positions: np.ndarray) -> None:

        '''Build spatial data structure from particle positions.'''
        ...

    def queryPairs(self, radius: float) -> tuple[np.ndarray, np.ndarray]:

        '''

        Find all particle pairs within the given radius.

        Returns:
        --------
        tuple[np.ndarray, np.ndarray] :
            (i_indices, j_indices) where particle i and j are neighbors.
            Each pair appears once with i < j.

        '''
        ...

#--------------------------------------------------------------------#
# -- Spatial Hash Grid -- #
#--------------------------------------------------------------------#

class SpatialHashGrid:

    '''

    Uniform grid spatial hashing for 2D/3D neighbor search.

    Cell size equals the kernel support radius. Particles are
    binned into cells using integer coordinates. For neighbor
    queries, only the 9 (2D) or 27 (3D) adjacent cells are searched.

    The pair distance checks are vectorized per cell-pair group
    using NumPy broadcasting for fast batch distance computation.

    Parameters:
    -----------
    cellSize : float
        Grid cell size [m], should equal the kernel support radius
    dimensions : int
        Number of spatial dimensions (2 or 3)

    '''

    def __init__(self, cellSize: float, dimensions: int = 2) -> None:
        self._cellSize = cellSize
        self._dimensions = dimensions
        self._positions: np.ndarray | None = None
        self._cells: dict[tuple, np.ndarray] = {}

        # Pre-compute the half-stencil offsets (to avoid double-counting pairs)
        self._halfStencil = self._computeHalfStencil()

    def build(self, positions: np.ndarray) -> None:

        '''

        Build the spatial hash grid from particle positions.

        Bins all particles into grid cells based on their
        integer cell coordinates.

        Parameters:
        -----------
        positions : np.ndarray
            Particle positions, shape (N, dim)

        '''
        self._positions = positions
        self._cells.clear()

        # Compute cell indices for all particles (vectorized)
        cellIndices = np.floor(positions / self._cellSize).astype(np.int32)

        # Build a dict mapping cell key -> array of particle indices
        # Use a defaultdict-like approach with Python dict
        cellDict: dict[tuple, list[int]] = {}
        for i in range(len(positions)):
            key = tuple(cellIndices[i])
            if key not in cellDict:
                cellDict[key] = []
            cellDict[key].append(i)

        # Convert lists to numpy arrays for faster indexing
        self._cells = {k: np.array(v, dtype=np.int32) for k, v in cellDict.items()}

    def queryPairs(self, radius: float) -> tuple[np.ndarray, np.ndarray]:

        '''

        Find all unique particle pairs (i, j) within the given radius.

        Uses half-stencil traversal to ensure each pair is found exactly
        once. Distance checks are vectorized using NumPy broadcasting
        within each cell-pair group.

        Parameters:
        -----------
        radius : float
            Search radius [m]

        Returns:
        --------
        tuple[np.ndarray, np.ndarray] :
            (iIndices, jIndices) arrays of neighbor pair indices

        '''
        if self._positions is None:
            return (np.array([], dtype=np.int32), np.array([], dtype=np.int32))

        radiusSq = radius * radius
        positions = self._positions
        iChunks: list[np.ndarray] = []
        jChunks: list[np.ndarray] = []

        for cellKey, cellParticles in self._cells.items():
            # --- Self-pairs within the same cell --- #
            nCell = len(cellParticles)
            if nCell > 1:
                # Get positions for this cell
                cellPos = positions[cellParticles]  # shape (nCell, dim)

                # All pairs within the cell using upper triangle
                # Compute pairwise squared distances using broadcasting
                diff = cellPos[:, np.newaxis, :] - cellPos[np.newaxis, :, :]  # (n, n, dim)
                distSq = np.sum(diff * diff, axis=2)  # (n, n)

                # Upper triangle mask (i < j)
                rowIdx, colIdx = np.triu_indices(nCell, k=1)
                pairDistSq = distSq[rowIdx, colIdx]

                # Filter by radius
                withinRadius = pairDistSq < radiusSq
                if np.any(withinRadius):
                    # Map local indices back to global
                    globalI = cellParticles[rowIdx[withinRadius]]
                    globalJ = cellParticles[colIdx[withinRadius]]
                    # Ensure i < j for consistency
                    swapMask = globalI > globalJ
                    globalI[swapMask], globalJ[swapMask] = globalJ[swapMask], globalI[swapMask]
                    iChunks.append(globalI)
                    jChunks.append(globalJ)

            # --- Cross-pairs with neighbor cells (half-stencil only) --- #
            for offset in self._halfStencil:
                neighborKey = tuple(cellKey[d] + offset[d] for d in range(self._dimensions))
                neighborParticles = self._cells.get(neighborKey)
                if neighborParticles is None:
                    continue

                # Vectorized pairwise distance check between two cell groups
                cellPos = positions[cellParticles]      # (nA, dim)
                neighborPos = positions[neighborParticles]  # (nB, dim)

                diff = cellPos[:, np.newaxis, :] - neighborPos[np.newaxis, :, :]  # (nA, nB, dim)
                distSq = np.sum(diff * diff, axis=2)  # (nA, nB)

                # Find pairs within radius
                withinRadius = distSq < radiusSq
                localI, localJ = np.where(withinRadius)

                if len(localI) > 0:
                    globalI = cellParticles[localI]
                    globalJ = neighborParticles[localJ]
                    # Ensure i < j
                    swapMask = globalI > globalJ
                    globalI[swapMask], globalJ[swapMask] = globalJ[swapMask], globalI[swapMask]
                    iChunks.append(globalI)
                    jChunks.append(globalJ)

        if not iChunks:
            return (np.array([], dtype=np.int32), np.array([], dtype=np.int32))

        iAll = np.concatenate(iChunks)
        jAll = np.concatenate(jChunks)

        return (iAll, jAll)

    def _computeHalfStencil(self) -> list[tuple[int, ...]]:

        '''

        Compute the half-stencil: neighbor offsets that avoid double-counting.

        For the full 3x3 (2D) or 3x3x3 (3D) stencil, we only keep offsets
        that are lexicographically positive (i.e., the offset vector comes
        after (0,0,...) in lexicographic order). This gives us exactly half
        the non-self neighbors, so each pair is found exactly once.

        Returns:
        --------
        list[tuple[int, ...]] : Half-stencil offsets

        '''
        if self._dimensions == 2:
            # 4 neighbors in the "positive half" of the 3x3 stencil
            return [
                (1, -1), (1, 0), (1, 1),
                (0, 1),
            ]
        else:
            # 13 neighbors in the "positive half" of the 3x3x3 stencil
            offsets = []
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        if (dx, dy, dz) > (0, 0, 0):
                            offsets.append((dx, dy, dz))
            return offsets
