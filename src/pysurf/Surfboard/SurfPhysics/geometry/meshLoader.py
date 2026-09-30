# -- STL Mesh Board Geometry -- #

'''

Board geometry derived from an STL mesh file exported by the C# generator.

Uses trimesh for mesh loading, repair, and volume/area computations.
Provides an alternative to the parametric BoardGeometry for validation
and when accurate voxel-based volumes are needed.

Sean Bowman [02/03/2026]

'''

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None


class MeshBoardGeometry:

    '''

    Board geometry derived from an STL mesh file.

    Provides volume, surface area, and submerged volume queries
    from the actual mesh data rather than parametric reconstruction.

    '''

    def __init__(self, stlFilePath: str) -> None:

        '''

        Load an STL mesh file and prepare it for geometry queries.

        Applies automatic repair (hole filling, normal fixing) since
        voxel-to-mesh conversion pipelines may produce non-watertight meshes.

        Parameters:
        -----------
        stlFilePath : str
            Path to the STL file

        '''

        if trimesh is None:
            raise ImportError(
                'trimesh is required for STL mesh loading. '
                'Install with: pip install trimesh'
            )

        self._mesh: trimesh.Trimesh = trimesh.load(stlFilePath)

        # Auto-repair: fill holes and fix normals for volume computation
        if not self._mesh.is_watertight:
            trimesh.repair.fill_holes(self._mesh)
            trimesh.repair.fix_normals(self._mesh)

        # Convert from mm to meters
        self._meshM: trimesh.Trimesh = self._mesh.copy()
        self._meshM.apply_scale(0.001)

    @property
    def isWatertight(self) -> bool:
        '''Whether the mesh is watertight (required for volume computation).'''
        return self._meshM.is_watertight

    def computeVolume(self) -> float:

        '''

        Exact volume in m^3 from the mesh.

        Returns:
        --------
        float : Volume in m^3 (returns 0 if mesh is not watertight)

        '''

        if not self._meshM.is_watertight:
            return 0.0
        return float(self._meshM.volume)

    def computeSurfaceArea(self) -> float:

        '''

        Total surface area in m^2.

        Returns:
        --------
        float : Surface area in m^2

        '''

        return float(self._meshM.area)

    def computeSubmergedVolume(self, waterplaneZ: float) -> float:

        '''

        Volume below a horizontal plane at a given Z height.

        Uses trimesh slice_plane to cut the mesh at the waterplane
        and compute the volume of the lower portion.

        Parameters:
        -----------
        waterplaneZ : float
            Z-coordinate of the waterplane in meters

        Returns:
        --------
        float : Submerged volume in m^3

        '''

        # Slice the mesh at the waterplane (plane normal = +Z, point on plane)
        planeOrigin = np.array([0.0, 0.0, waterplaneZ])
        planeNormal = np.array([0.0, 0.0, 1.0])

        try:
            # trimesh.intersections.slice_mesh_plane returns the portion
            # below the plane (in the direction opposite to the normal)
            submerged = trimesh.intersections.slice_mesh_plane(
                self._meshM,
                plane_normal=planeNormal,
                plane_origin=planeOrigin,
                cap=True,
            )
            if submerged is not None and submerged.is_watertight:
                return float(submerged.volume)
        except Exception:
            pass

        return 0.0

    def getCrossSectionAtX(self, xPositionM: float) -> Optional[np.ndarray]:

        '''

        Slice the mesh at a given X position and return the 2D cross-section.

        Parameters:
        -----------
        xPositionM : float
            X-coordinate to slice at, in meters

        Returns:
        --------
        Optional[np.ndarray] : Array of (y, z) vertices defining the cross-section,
                               or None if no intersection

        '''

        planeOrigin = np.array([xPositionM, 0.0, 0.0])
        planeNormal = np.array([1.0, 0.0, 0.0])

        try:
            lines = trimesh.intersections.mesh_plane(
                self._meshM,
                plane_normal=planeNormal,
                plane_origin=planeOrigin,
            )
            if lines is not None and len(lines) > 0:
                # Extract unique Y, Z points from intersection lines
                points = lines.reshape(-1, 3)[:, 1:]  # drop X, keep Y and Z
                return np.unique(points, axis=0)
        except Exception:
            pass

        return None

    def getBoundingBox(self) -> tuple[np.ndarray, np.ndarray]:

        '''

        Get the axis-aligned bounding box in meters.

        Returns:
        --------
        tuple[np.ndarray, np.ndarray] : (min_corner, max_corner), each shape (3,)

        '''

        return (self._meshM.bounds[0].copy(), self._meshM.bounds[1].copy())
