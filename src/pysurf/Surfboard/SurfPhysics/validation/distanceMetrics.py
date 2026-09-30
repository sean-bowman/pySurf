# -- Distance Metrics Module -- #

'''
SDF computation and deviation statistics for mesh comparison.

Computes signed distances between meshes and provides statistical
analysis broken down by board region.

Sean Bowman [02/05/2026]
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None


#--------------------------------------------------------------------#
# -- Data Classes -- #
#--------------------------------------------------------------------#

@dataclass
class DistanceStatistics:
    '''Statistics for signed distance field comparison.'''
    minMm: float
    maxMm: float
    meanMm: float
    rmsMm: float
    stdMm: float
    percentile95Mm: float
    percentile99Mm: float
    positiveCount: int      # Points where generated is OUTSIDE reference
    negativeCount: int      # Points where generated is INSIDE reference
    totalPoints: int

    def toDict(self) -> dict:
        '''Convert to dictionary for JSON serialization.'''
        return {
            'minMm': self.minMm,
            'maxMm': self.maxMm,
            'meanMm': self.meanMm,
            'rmsMm': self.rmsMm,
            'stdMm': self.stdMm,
            'percentile95Mm': self.percentile95Mm,
            'percentile99Mm': self.percentile99Mm,
            'positiveCount': self.positiveCount,
            'negativeCount': self.negativeCount,
            'totalPoints': self.totalPoints,
        }


@dataclass
class RegionalDeviation:
    '''Deviation statistics for a specific board region.'''
    region: str             # 'nose', 'tail', 'left_rail', etc.
    tRange: tuple[float, float]  # Normalized longitudinal range
    stats: DistanceStatistics
    maxDeviationPoint: np.ndarray  # Location of maximum deviation

    def toDict(self) -> dict:
        '''Convert to dictionary for JSON serialization.'''
        return {
            'region': self.region,
            'tRange': list(self.tRange),
            'stats': self.stats.toDict(),
            'maxDeviationPoint': self.maxDeviationPoint.tolist(),
        }


#--------------------------------------------------------------------#
# -- Distance Analyzer -- #
#--------------------------------------------------------------------#

class DistanceAnalyzer:
    '''
    Computes signed distance fields and deviation metrics between meshes.

    Uses trimesh proximity queries for fast distance computation.
    '''

    def __init__(
        self,
        referenceMesh: 'trimesh.Trimesh',
        generatedMesh: 'trimesh.Trimesh',
    ) -> None:
        '''
        Initialize with aligned meshes.

        Parameters:
        -----------
        referenceMesh : trimesh.Trimesh
            The reference (target) mesh
        generatedMesh : trimesh.Trimesh
            The generated (source) mesh to compare
        '''
        if trimesh is None:
            raise ImportError(
                'trimesh is required for distance computation. '
                'Install with: pip install trimesh'
            )

        self._refMesh = referenceMesh
        self._genMesh = generatedMesh
        self._refQuery = trimesh.proximity.ProximityQuery(referenceMesh)

    def sampleFromSurface(
        self,
        mesh: 'trimesh.Trimesh',
        nPoints: int,
        method: str = 'uniform',
    ) -> np.ndarray:
        '''
        Sample points from mesh surface for distance computation.

        Parameters:
        -----------
        mesh : trimesh.Trimesh
            Source mesh to sample from
        nPoints : int
            Number of sample points
        method : str
            'uniform' - uniform random sampling

        Returns:
        --------
        np.ndarray : (N, 3) sample point coordinates
        '''
        if method == 'uniform':
            points, _ = trimesh.sample.sample_surface(mesh, nPoints)
            return points
        else:
            raise ValueError(f'Unknown sampling method: {method}')

    def computeSignedDistances(
        self,
        samplePoints: np.ndarray,
    ) -> np.ndarray:
        '''
        Compute signed distances from sample points to reference mesh.

        Positive = point is outside reference (generated is too large)
        Negative = point is inside reference (generated is too small)

        Parameters:
        -----------
        samplePoints : np.ndarray
            (N, 3) array of query points (typically from generated mesh)

        Returns:
        --------
        np.ndarray : Signed distances in same units as mesh (mm)
        '''
        # Get unsigned distances to closest points on reference surface
        closestPoints, distances, _ = self._refQuery.on_surface(samplePoints)

        # Determine sign by checking if points are inside reference mesh
        # Points inside have negative distance, outside have positive
        containment = self._refMesh.contains(samplePoints)
        signedDistances = np.where(containment, -distances, distances)

        return signedDistances

    def computeStatistics(self, distances: np.ndarray) -> DistanceStatistics:
        '''
        Compute summary statistics for distance array.

        Parameters:
        -----------
        distances : np.ndarray
            Array of signed distances

        Returns:
        --------
        DistanceStatistics : Summary statistics
        '''
        absDistances = np.abs(distances)

        return DistanceStatistics(
            minMm=float(np.min(distances)),
            maxMm=float(np.max(distances)),
            meanMm=float(np.mean(distances)),
            rmsMm=float(np.sqrt(np.mean(distances ** 2))),
            stdMm=float(np.std(distances)),
            percentile95Mm=float(np.percentile(absDistances, 95)),
            percentile99Mm=float(np.percentile(absDistances, 99)),
            positiveCount=int(np.sum(distances > 0)),
            negativeCount=int(np.sum(distances < 0)),
            totalPoints=len(distances),
        )

    def computeBidirectionalDistance(
        self,
        nPoints: int = 50000,
    ) -> tuple[DistanceStatistics, np.ndarray, np.ndarray]:
        '''
        Compute distances from generated mesh surface to reference.

        Parameters:
        -----------
        nPoints : int
            Number of points to sample from generated surface

        Returns:
        --------
        tuple : (statistics, samplePoints, signedDistances)
        '''
        samplePoints = self.sampleFromSurface(self._genMesh, nPoints)
        signedDistances = self.computeSignedDistances(samplePoints)
        stats = self.computeStatistics(signedDistances)

        return stats, samplePoints, signedDistances

    def computeHausdorffDistance(self, nPoints: int = 10000) -> float:
        '''
        Compute approximate Hausdorff distance (maximum deviation).

        Samples both directions and returns the maximum.

        Parameters:
        -----------
        nPoints : int
            Number of points to sample from each mesh

        Returns:
        --------
        float : Maximum one-sided distance in mm
        '''
        # Generated -> Reference
        genPoints = self.sampleFromSurface(self._genMesh, nPoints)
        _, genToRefDist, _ = self._refQuery.on_surface(genPoints)
        maxGenToRef = float(np.max(genToRefDist))

        # Reference -> Generated
        refPoints = self.sampleFromSurface(self._refMesh, nPoints)
        genQuery = trimesh.proximity.ProximityQuery(self._genMesh)
        _, refToGenDist, _ = genQuery.on_surface(refPoints)
        maxRefToGen = float(np.max(refToGenDist))

        return max(maxGenToRef, maxRefToGen)

    def computeRegionalDeviations(
        self,
        samplePoints: np.ndarray,
        distances: np.ndarray,
        boardLengthMm: float,
    ) -> list[RegionalDeviation]:
        '''
        Break down deviations by board region.

        Regions by longitudinal position:
        - Nose: t = 0.0 to 0.15
        - Forward: t = 0.15 to 0.35
        - Middle: t = 0.35 to 0.65
        - Rear: t = 0.65 to 0.85
        - Tail: t = 0.85 to 1.0

        Parameters:
        -----------
        samplePoints : np.ndarray
            (N, 3) sample point coordinates
        distances : np.ndarray
            Signed distances for each sample point
        boardLengthMm : float
            Board length for normalizing X positions

        Returns:
        --------
        list[RegionalDeviation] : Statistics per region
        '''
        regions = [
            ('nose', (0.0, 0.15)),
            ('forward', (0.15, 0.35)),
            ('middle', (0.35, 0.65)),
            ('rear', (0.65, 0.85)),
            ('tail', (0.85, 1.0)),
        ]

        results = []
        xCoords = samplePoints[:, 0]
        tCoords = xCoords / boardLengthMm  # Normalize to 0-1

        for regionName, tRange in regions:
            # Find points in this region
            mask = (tCoords >= tRange[0]) & (tCoords < tRange[1])
            if np.sum(mask) < 10:
                continue  # Skip if too few points

            regionDistances = distances[mask]
            regionPoints = samplePoints[mask]

            # Find point with maximum absolute deviation
            maxIdx = np.argmax(np.abs(regionDistances))
            maxPoint = regionPoints[maxIdx]

            stats = self.computeStatistics(regionDistances)

            results.append(RegionalDeviation(
                region=regionName,
                tRange=tRange,
                stats=stats,
                maxDeviationPoint=maxPoint,
            ))

        return results
