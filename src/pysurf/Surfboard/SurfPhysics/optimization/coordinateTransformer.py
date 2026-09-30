# -- Coordinate Transformer -- #

'''
Auto-detection of scale and axis transformations between meshes.

Handles unit conversion and axis remapping for reference meshes
that use different coordinate systems than the generated geometry.

Sean Bowman [02/05/2026]
'''

from __future__ import annotations

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None


#--------------------------------------------------------------------#
# -- Coordinate Transformer -- #
#--------------------------------------------------------------------#

class CoordinateTransformer:
    '''
    Handles scale and axis transformations between reference and generated meshes.

    The referenceThruster.stl uses different units (~0.0024mm) and axis orientation
    (length on Z) than the generated meshes (mm, length on X).
    '''

    @staticmethod
    def detectScaleFactor(
        mesh: 'trimesh.Trimesh',
        expectedLengthMm: float,
    ) -> float:
        '''
        Auto-detect scale factor by comparing mesh bounding box to expected length.

        Parameters:
        -----------
        mesh : trimesh.Trimesh
            Raw reference mesh
        expectedLengthMm : float
            Expected board length in mm (e.g., 1828 for 6'0')

        Returns:
        --------
        float : Scale factor to multiply reference coordinates
        '''
        bounds = mesh.bounds
        size = bounds[1] - bounds[0]

        # Find the longest dimension (should be board length)
        maxDim = float(max(size))

        if maxDim == 0:
            raise ValueError('Mesh has zero size')

        return expectedLengthMm / maxDim

    @staticmethod
    def detectAxisMapping(
        mesh: 'trimesh.Trimesh',
    ) -> str:
        '''
        Auto-detect axis remapping based on bounding box aspect ratios.

        Surfboards have a predictable aspect ratio:
        - Length is longest (~6x width)
        - Width is medium (~10x thickness)
        - Thickness is smallest

        Parameters:
        -----------
        mesh : trimesh.Trimesh
            Raw reference mesh

        Returns:
        --------
        str : Axis mapping string (e.g., 'zxy' means new_X = old_Z, etc.)
        '''
        bounds = mesh.bounds
        size = bounds[1] - bounds[0]

        # Sort dimensions to find: largest (length), medium (width), smallest (thickness)
        dimIndices = np.argsort(size)[::-1]  # Largest first

        # Map sorted order back to axis characters
        # dimIndices[0] is the original axis containing the largest dimension (length → should be X)
        # dimIndices[1] is the original axis containing the medium dimension (width → should be Y)
        # dimIndices[2] is the original axis containing the smallest dimension (thickness → should be Z)
        axisChars = ['x', 'y', 'z']

        # Build mapping: for each new axis (X, Y, Z), which old axis provides the value?
        mapping = ''.join([axisChars[dimIndices[i]] for i in range(3)])

        return mapping

    @staticmethod
    def autoDetect(
        mesh: 'trimesh.Trimesh',
        expectedLengthMm: float = 1828.0,
    ) -> tuple[float, str]:
        '''
        Auto-detect both scale factor and axis mapping.

        Parameters:
        -----------
        mesh : trimesh.Trimesh
            Raw reference mesh
        expectedLengthMm : float
            Expected board length in mm

        Returns:
        --------
        tuple[float, str] : (scaleFactor, axisMapping)
        '''
        scale = CoordinateTransformer.detectScaleFactor(mesh, expectedLengthMm)
        axisMapping = CoordinateTransformer.detectAxisMapping(mesh)

        return scale, axisMapping

    @staticmethod
    def validateTransformation(
        originalMesh: 'trimesh.Trimesh',
        transformedMesh: 'trimesh.Trimesh',
        expectedAspectRatio: tuple[float, float] = (0.27, 0.034),
        tolerance: float = 0.3,
    ) -> tuple[bool, str]:
        '''
        Validate that transformation produces reasonable board dimensions.

        Parameters:
        -----------
        originalMesh : trimesh.Trimesh
            Mesh before transformation
        transformedMesh : trimesh.Trimesh
            Mesh after transformation
        expectedAspectRatio : tuple[float, float]
            Expected (width/length, thickness/length) ratios for shortboard
        tolerance : float
            Allowable deviation from expected ratios

        Returns:
        --------
        tuple[bool, str] : (isValid, message)
        '''
        bounds = transformedMesh.bounds
        size = bounds[1] - bounds[0]

        # After transformation, X should be length, Y width, Z thickness
        length = size[0]
        width = size[1]
        thickness = size[2]

        if length <= 0:
            return False, 'Transformed length is zero or negative'

        actualWidthRatio = width / length
        actualThicknessRatio = thickness / length

        expectedWidthRatio, expectedThicknessRatio = expectedAspectRatio

        widthError = abs(actualWidthRatio - expectedWidthRatio) / expectedWidthRatio
        thicknessError = abs(actualThicknessRatio - expectedThicknessRatio) / expectedThicknessRatio

        if widthError > tolerance:
            return False, (
                f'Width/length ratio {actualWidthRatio:.3f} differs from '
                f'expected {expectedWidthRatio:.3f} by {widthError*100:.1f}%'
            )

        if thicknessError > tolerance:
            return False, (
                f'Thickness/length ratio {actualThicknessRatio:.3f} differs from '
                f'expected {expectedThicknessRatio:.3f} by {thicknessError*100:.1f}%'
            )

        return True, (
            f'Dimensions look valid: {length:.0f} x {width:.0f} x {thickness:.0f} mm'
        )
