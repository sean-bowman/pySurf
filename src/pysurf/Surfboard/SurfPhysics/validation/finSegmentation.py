# -- Fin Segmentation Module -- #

'''
Segments surfboard mesh into body and fin components.

Isolates fins from the main board body for separate comparison analysis.
Supports height-based and bounding-box-based segmentation strategies.

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
class SegmentedMesh:
    '''Result of mesh segmentation into body and fins.'''
    bodyMesh: 'trimesh.Trimesh'
    finMeshes: list['trimesh.Trimesh']
    finBoundingBoxes: list[np.ndarray]  # Each is (2, 3) for min/max corners
    finConfiguration: str               # 'thruster', 'twin', 'quad', 'single', 'unknown'

    def toDict(self) -> dict:
        '''Convert to dictionary for JSON serialization (without meshes).'''
        return {
            'nFins': len(self.finMeshes),
            'finConfiguration': self.finConfiguration,
            'finBoundingBoxes': [bb.tolist() for bb in self.finBoundingBoxes],
        }


#--------------------------------------------------------------------#
# -- Default Fin Regions (from FinSystem.cs) -- #
#--------------------------------------------------------------------#

# Fin positions for a standard 6'0" (1828mm) shortboard with thruster fins
# Adjust these based on board length using scale factor
THRUSTER_FIN_REGIONS = [
    {
        'name': 'center_fin',
        'xMinOffset': 150,    # Distance from tail
        'xMaxOffset': 10,
        'yMin': -25,
        'yMax': 25,
        'zThreshold': -15,    # Must extend below this Z
    },
    {
        'name': 'right_side_fin',
        'xMinOffset': 450,
        'xMaxOffset': 250,
        'yMin': 60,
        'yMax': 180,
        'zThreshold': -15,
    },
    {
        'name': 'left_side_fin',
        'xMinOffset': 450,
        'xMaxOffset': 250,
        'yMin': -180,
        'yMax': -60,
        'zThreshold': -15,
    },
]


#--------------------------------------------------------------------#
# -- Fin Segmenter -- #
#--------------------------------------------------------------------#

class FinSegmenter:
    '''
    Segments surfboard mesh into body and fin components.
    '''

    def __init__(self, mesh: 'trimesh.Trimesh') -> None:
        '''
        Load mesh for segmentation.

        Parameters:
        -----------
        mesh : trimesh.Trimesh
            Complete surfboard mesh including fins
        '''
        if trimesh is None:
            raise ImportError(
                'trimesh is required for fin segmentation. '
                'Install with: pip install trimesh'
            )

        self._mesh = mesh
        self._bounds = mesh.bounds
        self._boardLength = self._bounds[1, 0] - self._bounds[0, 0]

    def segmentByHeight(
        self,
        finThresholdMm: float = 20.0,
    ) -> SegmentedMesh:
        '''
        Segment fins from body using Z-height threshold.

        Fins extend below the bottom surface (more negative Z), so
        vertices significantly below the main body bottom are fins.

        Parameters:
        -----------
        finThresholdMm : float
            Distance below main body bottom to consider as fins

        Returns:
        --------
        SegmentedMesh : Body and fin meshes separated
        '''
        vertices = self._mesh.vertices
        zCoords = vertices[:, 2]

        # Find the main body bottom by looking at the distribution of Z values
        # The body bottom is typically around the 10th percentile
        bodyBottomZ = np.percentile(zCoords, 10)

        # Vertices below this threshold are potential fin vertices
        finThreshold = bodyBottomZ - finThresholdMm
        isFinVertex = zCoords < finThreshold

        # Get faces where all vertices are fin vertices
        faceVertices = self._mesh.faces
        isFinFace = np.all(isFinVertex[faceVertices], axis=1)

        if np.sum(isFinFace) == 0:
            # No fins detected
            return SegmentedMesh(
                bodyMesh=self._mesh.copy(),
                finMeshes=[],
                finBoundingBoxes=[],
                finConfiguration='unknown',
            )

        # Create fin submesh
        finMesh = self._mesh.submesh([isFinFace], append=True)

        # Create body submesh (faces that are not fins)
        bodyMesh = self._mesh.submesh([~isFinFace], append=True)

        # Split fin mesh into connected components (individual fins)
        finComponents = finMesh.split(only_watertight=False)

        # Filter out tiny components (noise)
        minFinVolume = 100.0  # mm^3
        validFins = []
        finBounds = []

        for comp in finComponents:
            if comp.is_watertight and comp.volume > minFinVolume:
                validFins.append(comp)
                finBounds.append(comp.bounds)
            elif not comp.is_watertight and len(comp.vertices) > 100:
                # Non-watertight but significant geometry
                validFins.append(comp)
                finBounds.append(comp.bounds)

        # Infer fin configuration from count
        config = self._inferConfiguration(len(validFins))

        return SegmentedMesh(
            bodyMesh=bodyMesh,
            finMeshes=validFins,
            finBoundingBoxes=finBounds,
            finConfiguration=config,
        )

    def segmentByRegion(
        self,
        finRegions: list[dict] | None = None,
    ) -> SegmentedMesh:
        '''
        Segment using predefined bounding box regions.

        Parameters:
        -----------
        finRegions : list[dict] | None
            List of fin regions with keys: name, xMinOffset, xMaxOffset,
            yMin, yMax, zThreshold. If None, uses thruster defaults.

        Returns:
        --------
        SegmentedMesh : Body and fin meshes separated
        '''
        if finRegions is None:
            finRegions = THRUSTER_FIN_REGIONS

        vertices = self._mesh.vertices
        faces = self._mesh.faces
        boardLength = self._boardLength
        tailX = self._bounds[1, 0]  # Maximum X is tail

        finMeshes = []
        finBounds = []
        allFinFaceMask = np.zeros(len(faces), dtype=bool)

        for region in finRegions:
            # Convert offsets to absolute X positions
            xMin = tailX - region['xMinOffset']
            xMax = tailX - region['xMaxOffset']

            # Create vertex mask for this region
            inRegion = (
                (vertices[:, 0] >= xMin) & (vertices[:, 0] <= xMax) &
                (vertices[:, 1] >= region['yMin']) & (vertices[:, 1] <= region['yMax']) &
                (vertices[:, 2] < region['zThreshold'])
            )

            # Get faces where all vertices are in region
            isFinFace = np.all(inRegion[faces], axis=1)

            if np.sum(isFinFace) > 0:
                finMesh = self._mesh.submesh([isFinFace], append=True)
                finMeshes.append(finMesh)
                finBounds.append(finMesh.bounds)
                allFinFaceMask |= isFinFace

        # Body is everything not identified as fins
        if np.sum(allFinFaceMask) > 0:
            bodyMesh = self._mesh.submesh([~allFinFaceMask], append=True)
        else:
            bodyMesh = self._mesh.copy()

        config = self._inferConfiguration(len(finMeshes))

        return SegmentedMesh(
            bodyMesh=bodyMesh,
            finMeshes=finMeshes,
            finBoundingBoxes=finBounds,
            finConfiguration=config,
        )

    def _inferConfiguration(self, nFins: int) -> str:
        '''
        Infer fin configuration from detected fin count.

        Parameters:
        -----------
        nFins : int
            Number of fins detected

        Returns:
        --------
        str : Configuration name
        '''
        configMap = {
            1: 'single',
            2: 'twin',
            3: 'thruster',
            4: 'quad',
        }
        return configMap.get(nFins, 'unknown')
