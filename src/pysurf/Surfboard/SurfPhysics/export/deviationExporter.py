# -- Deviation Exporter -- #

'''

Exports mesh comparison results as JSON for the Three.js deviation heatmap.

Converts ComparisonResult from MeshComparisonAnalyzer into a format suitable
for vertex-colored rendering in the viewer.

Sean Bowman [02/05/2026]

'''

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None


#--------------------------------------------------------------------#
# -- Deviation Exporter -- #
#--------------------------------------------------------------------#

class DeviationExporter:

    '''

    Exports mesh comparison results for Three.js deviation visualization.

    Produces a JSON file containing the reference mesh geometry and
    per-vertex deviation values for colormap rendering.

    '''

    def exportDeviationData(
        self,
        comparisonResult,
        outputDir: str = 'src/pysurf/Surfboard/SurfViewer/data',
        filename: str = 'deviationData.json',
        decimateFactor: float = 0.1,
    ) -> str:

        '''

        Export comparison results as JSON for the Three.js heatmap.

        Parameters:
        -----------
        comparisonResult : ComparisonResult
            Results from MeshComparisonAnalyzer.runFullComparison()
        outputDir : str
            Output directory for the JSON file
        filename : str
            Output filename
        decimateFactor : float
            Decimation factor for the exported mesh (0.1 = 10% of original)

        Returns:
        --------
        str : Path to the exported JSON file

        '''
        if trimesh is None:
            raise ImportError('trimesh is required for deviation export')

        # Get the aligned mesh and distances
        alignedMesh = comparisonResult.alignedMesh
        distances = comparisonResult.signedDistances

        # Decimate mesh for web performance
        if decimateFactor < 1.0:
            exportMesh = self._decimateMesh(alignedMesh, decimateFactor)

            # Recompute distances for decimated mesh vertices
            # Use the reference mesh to compute SDF
            refMesh = comparisonResult.referenceMesh
            newDistances = self._computeDistances(exportMesh, refMesh)
        else:
            exportMesh = alignedMesh
            newDistances = distances

        # Compute statistics
        stats = self._computeStats(newDistances)

        # Build export structure
        exportData = {
            'meta': {
                'exportTimestamp': datetime.now().isoformat(),
                'referenceMeshVertices': len(comparisonResult.referenceMesh.vertices),
                'alignedMeshVertices': len(alignedMesh.vertices),
                'exportedMeshVertices': len(exportMesh.vertices),
                'decimationFactor': decimateFactor,
            },
            'surfaceMesh': {
                'vertices': exportMesh.vertices.tolist(),
                'faces': exportMesh.faces.tolist(),
            },
            'deviations': {
                'vertexDistances': newDistances.tolist(),
                'minMm': float(stats['minMm']),
                'maxMm': float(stats['maxMm']),
                'meanMm': float(stats['meanMm']),
                'rmsMm': float(stats['rmsMm']),
                'colorScale': 'RdYlBu',
            },
            'overallStats': {
                'rmsMm': comparisonResult.overallStats.rmsMm,
                'meanMm': comparisonResult.overallStats.meanMm,
                'maxMm': comparisonResult.overallStats.maxMm,
                'minMm': comparisonResult.overallStats.minMm,
            },
            'regionalDeviations': [
                {
                    'region': rd.region,
                    'rmsMm': rd.stats.rmsMm,
                    'meanMm': rd.stats.meanMm,
                }
                for rd in comparisonResult.regionalDeviations
            ],
        }

        # Write JSON
        os.makedirs(outputDir, exist_ok=True)
        outputPath = os.path.join(outputDir, filename)

        with open(outputPath, 'w') as f:
            json.dump(exportData, f, indent=2)

        print(f'  Deviation data exported: {outputPath}')
        print(f'    Mesh: {len(exportMesh.vertices)} vertices, {len(exportMesh.faces)} faces')
        print(f'    Stats: RMS={stats["rmsMm"]:.2f}mm, range=[{stats["minMm"]:.1f}, {stats["maxMm"]:.1f}]mm')

        return outputPath

    def exportFromMeshes(
        self,
        referenceMesh: 'trimesh.Trimesh',
        generatedMesh: 'trimesh.Trimesh',
        outputDir: str = 'src/pysurf/Surfboard/SurfViewer/data',
        filename: str = 'deviationData.json',
        decimateFactor: float = 0.1,
    ) -> str:

        '''

        Export deviation data directly from two meshes.

        This is a convenience method for when you have meshes but not
        a full ComparisonResult object.

        Parameters:
        -----------
        referenceMesh : trimesh.Trimesh
            Reference mesh
        generatedMesh : trimesh.Trimesh
            Generated mesh (will show deviations relative to reference)
        outputDir : str
            Output directory for the JSON file
        filename : str
            Output filename
        decimateFactor : float
            Decimation factor for the exported mesh

        Returns:
        --------
        str : Path to the exported JSON file

        '''
        if trimesh is None:
            raise ImportError('trimesh is required for deviation export')

        # Compute distances
        distances = self._computeDistances(generatedMesh, referenceMesh)

        # Decimate if needed
        if decimateFactor < 1.0:
            exportMesh = self._decimateMesh(generatedMesh, decimateFactor)
            newDistances = self._computeDistances(exportMesh, referenceMesh)
        else:
            exportMesh = generatedMesh
            newDistances = distances

        # Compute statistics
        stats = self._computeStats(newDistances)

        # Build export structure
        exportData = {
            'meta': {
                'exportTimestamp': datetime.now().isoformat(),
                'referenceMeshVertices': len(referenceMesh.vertices),
                'generatedMeshVertices': len(generatedMesh.vertices),
                'exportedMeshVertices': len(exportMesh.vertices),
                'decimationFactor': decimateFactor,
            },
            'surfaceMesh': {
                'vertices': exportMesh.vertices.tolist(),
                'faces': exportMesh.faces.tolist(),
            },
            'deviations': {
                'vertexDistances': newDistances.tolist(),
                'minMm': float(stats['minMm']),
                'maxMm': float(stats['maxMm']),
                'meanMm': float(stats['meanMm']),
                'rmsMm': float(stats['rmsMm']),
                'colorScale': 'RdYlBu',
            },
        }

        # Write JSON
        os.makedirs(outputDir, exist_ok=True)
        outputPath = os.path.join(outputDir, filename)

        with open(outputPath, 'w') as f:
            json.dump(exportData, f, indent=2)

        print(f'  Deviation data exported: {outputPath}')

        return outputPath

    def _computeDistances(
        self,
        queryMesh: 'trimesh.Trimesh',
        referenceMesh: 'trimesh.Trimesh',
    ) -> np.ndarray:

        '''

        Compute signed distances from query mesh vertices to reference mesh.

        Parameters:
        -----------
        queryMesh : trimesh.Trimesh
            Mesh to query distances from
        referenceMesh : trimesh.Trimesh
            Reference mesh to measure distances to

        Returns:
        --------
        np.ndarray : Signed distances in mm for each query vertex

        '''
        # Find closest points on reference mesh
        closest, distancesSq, triangleIds = referenceMesh.nearest.on_surface(
            queryMesh.vertices
        )

        # Get face normals at closest points
        faceNormals = referenceMesh.face_normals[triangleIds]

        # Compute vectors from closest points to query points
        vectors = queryMesh.vertices - closest

        # Sign based on whether query point is outside (positive) or inside (negative)
        signs = np.sign(np.sum(vectors * faceNormals, axis=1))
        signs[signs == 0] = 1  # Handle exactly-on-surface case

        distances = np.sqrt(distancesSq) * signs

        return distances

    def _computeStats(self, distances: np.ndarray) -> dict:

        '''

        Compute deviation statistics.

        Parameters:
        -----------
        distances : np.ndarray
            Signed distances

        Returns:
        --------
        dict : Statistics dictionary

        '''
        return {
            'minMm': float(np.min(distances)),
            'maxMm': float(np.max(distances)),
            'meanMm': float(np.mean(distances)),
            'rmsMm': float(np.sqrt(np.mean(distances ** 2))),
        }

    def _decimateMesh(self, mesh, decimateFactor: float):

        '''

        Decimate mesh to reduce face count.

        Parameters:
        -----------
        mesh : trimesh.Trimesh
            Input mesh
        decimateFactor : float
            Target face count as fraction of original (0.1 = 10%)

        Returns:
        --------
        trimesh.Trimesh : Decimated mesh

        '''
        targetFaces = max(int(len(mesh.faces) * decimateFactor), 1000)

        # Try different API variants based on trimesh version/backend
        try:
            # Ratio-based API (some backends expect reduction ratio 0-1)
            # decimateFactor of 0.1 means keep 10%, so reduction is 0.9
            reduction = max(0.01, min(0.99, 1.0 - decimateFactor))
            return mesh.simplify_quadric_decimation(target_reduction=reduction)
        except (TypeError, AttributeError):
            pass

        try:
            # Try with face_count parameter (newer trimesh API)
            return mesh.simplify_quadric_decimation(face_count=targetFaces)
        except (TypeError, ValueError):
            pass

        try:
            # Try with positional argument (older API)
            return mesh.simplify_quadric_decimation(targetFaces)
        except (TypeError, ValueError):
            pass

        # Fallback: return original mesh if decimation fails
        print('  Warning: mesh decimation failed, using original mesh')
        return mesh.copy()
