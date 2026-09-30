# -- Mesh Alignment Module -- #

'''
Mesh alignment using PCA and ICP for geometry comparison.

Aligns generated surfboard mesh to reference mesh coordinate system
before computing deviation metrics.

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
# -- Alignment Result -- #
#--------------------------------------------------------------------#

@dataclass
class AlignmentResult:
    '''Result of mesh alignment operation.'''
    alignedMesh: 'trimesh.Trimesh'
    transform: np.ndarray           # 4x4 transformation matrix
    cost: float                     # Final alignment cost (lower = better)
    method: str                     # 'pca', 'icp', or 'combined'
    boundingBoxIou: float           # Intersection over union of bounding boxes

    def toDict(self) -> dict:
        '''Convert to dictionary for JSON serialization.'''
        return {
            'transform': self.transform.tolist(),
            'cost': self.cost,
            'method': self.method,
            'boundingBoxIou': self.boundingBoxIou,
        }


#--------------------------------------------------------------------#
# -- Mesh Aligner -- #
#--------------------------------------------------------------------#

class MeshAligner:
    '''
    Aligns two meshes to a common coordinate system for comparison.

    Supports PCA-based coarse alignment and ICP refinement.
    '''

    def __init__(self) -> None:
        '''Initialize the mesh aligner.'''
        if trimesh is None:
            raise ImportError(
                'trimesh is required for mesh alignment. '
                'Install with: pip install trimesh'
            )

    def alignPCA(
        self,
        sourceMesh: 'trimesh.Trimesh',
        targetMesh: 'trimesh.Trimesh',
    ) -> AlignmentResult:
        '''
        Align using Principal Component Analysis.

        Centers both meshes and aligns principal axes. Fast but may
        have rotation ambiguity.

        Parameters:
        -----------
        sourceMesh : trimesh.Trimesh
            Mesh to transform (typically generated)
        targetMesh : trimesh.Trimesh
            Reference mesh (alignment target)

        Returns:
        --------
        AlignmentResult : Aligned mesh and transformation
        '''
        # Center both meshes
        sourceCentroid = sourceMesh.centroid
        targetCentroid = targetMesh.centroid

        # Compute principal axes via covariance eigendecomposition
        sourceCoords = sourceMesh.vertices - sourceCentroid
        targetCoords = targetMesh.vertices - targetCentroid

        _, sourceAxes = np.linalg.eigh(np.cov(sourceCoords.T))
        _, targetAxes = np.linalg.eigh(np.cov(targetCoords.T))

        # Flip axes to ensure consistent orientation (longest axis positive)
        sourceAxes = self._consistentAxes(sourceAxes, sourceCoords)
        targetAxes = self._consistentAxes(targetAxes, targetCoords)

        # Build rotation matrix to align source axes to target axes
        rotation = targetAxes @ sourceAxes.T

        # Build 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = targetCentroid - rotation @ sourceCentroid

        # Apply transformation
        alignedMesh = sourceMesh.copy()
        alignedMesh.apply_transform(transform)

        # Compute alignment quality metrics
        cost = self._computeAlignmentCost(alignedMesh, targetMesh)
        iou = self._computeBoundingBoxIou(alignedMesh, targetMesh)

        return AlignmentResult(
            alignedMesh=alignedMesh,
            transform=transform,
            cost=cost,
            method='pca',
            boundingBoxIou=iou,
        )

    def alignICP(
        self,
        sourceMesh: 'trimesh.Trimesh',
        targetMesh: 'trimesh.Trimesh',
        initialTransform: np.ndarray | None = None,
        maxIterations: int = 100,
        threshold: float = 1e-5,
    ) -> AlignmentResult:
        '''
        Iterative Closest Point alignment.

        More accurate than PCA but requires good initial guess.

        Parameters:
        -----------
        sourceMesh : trimesh.Trimesh
            Mesh to transform
        targetMesh : trimesh.Trimesh
            Reference mesh
        initialTransform : np.ndarray | None
            Initial 4x4 transformation (uses identity if None)
        maxIterations : int
            Maximum ICP iterations
        threshold : float
            Convergence threshold

        Returns:
        --------
        AlignmentResult : Aligned mesh and transformation
        '''
        # Sample points for ICP (subsampling for speed)
        nPoints = min(10000, len(sourceMesh.vertices), len(targetMesh.vertices))
        sourcePoints, _ = trimesh.sample.sample_surface(sourceMesh, nPoints)
        targetPoints, _ = trimesh.sample.sample_surface(targetMesh, nPoints)

        # Apply initial transform if provided
        if initialTransform is not None:
            sourcePoints = trimesh.transformations.transform_points(
                sourcePoints, initialTransform
            )
            currentTransform = initialTransform.copy()
        else:
            currentTransform = np.eye(4)

        # Run ICP using trimesh
        transform, transformed, cost = trimesh.registration.icp(
            sourcePoints,
            targetPoints,
            initial=np.eye(4),
            max_iterations=maxIterations,
            threshold=threshold,
        )

        # Combine with initial transform
        finalTransform = transform @ currentTransform

        # Apply to mesh
        alignedMesh = sourceMesh.copy()
        alignedMesh.apply_transform(finalTransform)

        iou = self._computeBoundingBoxIou(alignedMesh, targetMesh)

        return AlignmentResult(
            alignedMesh=alignedMesh,
            transform=finalTransform,
            cost=float(cost),
            method='icp',
            boundingBoxIou=iou,
        )

    def combinedAlignment(
        self,
        sourceMesh: 'trimesh.Trimesh',
        targetMesh: 'trimesh.Trimesh',
    ) -> AlignmentResult:
        '''
        Two-stage alignment: coarse PCA followed by fine ICP.

        Most robust approach for surfboard comparison.

        Parameters:
        -----------
        sourceMesh : trimesh.Trimesh
            Mesh to transform
        targetMesh : trimesh.Trimesh
            Reference mesh

        Returns:
        --------
        AlignmentResult : Aligned mesh and transformation
        '''
        # Stage 1: Coarse PCA alignment
        pcaResult = self.alignPCA(sourceMesh, targetMesh)

        # Stage 2: Fine ICP refinement
        icpResult = self.alignICP(
            sourceMesh,
            targetMesh,
            initialTransform=pcaResult.transform,
            maxIterations=100,
        )

        # Update method name
        icpResult.method = 'combined'

        return icpResult

    def _consistentAxes(
        self,
        axes: np.ndarray,
        coords: np.ndarray,
    ) -> np.ndarray:
        '''
        Ensure consistent axis orientation.

        Flips axes so that the majority of points have positive projection.
        '''
        for i in range(3):
            projections = coords @ axes[:, i]
            if np.sum(projections > 0) < len(projections) / 2:
                axes[:, i] *= -1
        return axes

    def _computeAlignmentCost(
        self,
        sourceMesh: 'trimesh.Trimesh',
        targetMesh: 'trimesh.Trimesh',
        nPoints: int = 5000,
    ) -> float:
        '''
        Compute alignment cost as mean point-to-surface distance.
        '''
        sourcePoints, _ = trimesh.sample.sample_surface(sourceMesh, nPoints)
        targetQuery = trimesh.proximity.ProximityQuery(targetMesh)
        _, distances, _ = targetQuery.on_surface(sourcePoints)
        return float(np.mean(distances))

    def _computeBoundingBoxIou(
        self,
        mesh1: 'trimesh.Trimesh',
        mesh2: 'trimesh.Trimesh',
    ) -> float:
        '''
        Compute intersection over union of axis-aligned bounding boxes.
        '''
        # Get bounding boxes
        min1, max1 = mesh1.bounds
        min2, max2 = mesh2.bounds

        # Intersection
        intMin = np.maximum(min1, min2)
        intMax = np.minimum(max1, max2)

        # Check for no intersection
        if np.any(intMax <= intMin):
            return 0.0

        intVol = np.prod(intMax - intMin)

        # Union = Vol1 + Vol2 - Intersection
        vol1 = np.prod(max1 - min1)
        vol2 = np.prod(max2 - min2)
        unionVol = vol1 + vol2 - intVol

        return float(intVol / unionVol) if unionVol > 0 else 0.0
