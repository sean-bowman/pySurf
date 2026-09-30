
# -- Reverse Engineering Tool -- #

'''
Reverse-engineers surfboard shape from an STL mesh.

Slices a reference STL into cross-sections, extracts outline, rocker,
and thickness profiles, and optionally fits SurfboardParameters to
match the extracted shape. This enables:

  - Understanding the geometry of an existing board
  - Calibrating parametric model parameters to replicate a reference
  - Validating the parametric model by round-tripping through extraction

The workflow:
  1. Load and transform reference STL into project coordinates
  2. Optionally segment fins from the body
  3. Slice the body mesh at many X stations to get 2D cross-sections
  4. Extract outline, rocker, thickness, deck, and bottom profiles
  5. Fit SurfboardParameters from the extracted profiles

Sean Bowman [02/07/2026]
'''

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

from pysurf.Surfboard.SurfPhysics.geometry.parameters import SurfboardParameters
from pysurf.Surfboard.SurfPhysics.geometry.outline import Outline
from pysurf.Surfboard.SurfPhysics.geometry.rocker import RockerProfile
from pysurf.Surfboard.SurfPhysics.geometry.crossSection import CrossSection
from pysurf.Surfboard.SurfPhysics.optimization.coordinateTransformer import CoordinateTransformer
from pysurf.Surfboard.SurfPhysics.validation.finSegmentation import FinSegmenter


#--------------------------------------------------------------------#
# -- Data Classes -- #
#--------------------------------------------------------------------#

@dataclass
class ExtractedProfiles:
    '''
    Shape profiles extracted from a reference STL mesh.

    All spatial values are in millimeters, matching the project convention.
    The profiles are sampled at uniform longitudinal stations from nose to tail.
    '''

    # Normalized longitudinal positions (0 = nose, 1 = tail)
    tValues: np.ndarray

    # Half-width at each station (mm): the outline curve
    halfWidths: np.ndarray

    # Bottom centerline Z at each station (mm): the rocker curve
    rockerHeights: np.ndarray

    # Total thickness (deck - bottom at centerline) at each station (mm)
    thicknesses: np.ndarray

    # Deck Z profiles: list of arrays, each shape (nLateralSamples,)
    # Values are Z height relative to the rocker baseline at each lateral fraction
    deckProfiles: list[np.ndarray]

    # Bottom Z profiles: same shape as deckProfiles
    # Values are Z height relative to the rocker baseline (negative = below)
    bottomProfiles: list[np.ndarray]

    # Lateral fractions used for deck/bottom profiles (0 = center, 1 = rail)
    lateralFractions: np.ndarray

    # Overall board dimensions
    boardLengthMm: float
    maxWidthMm: float
    maxThicknessMm: float


#--------------------------------------------------------------------#
# -- Reverse Engineer -- #
#--------------------------------------------------------------------#

class ReverseEngineer:
    '''
    Reverse-engineers surfboard shape from an STL mesh.

    Slices the mesh into cross-sections, extracts shape profiles,
    and optionally fits SurfboardParameters to match the extracted shape.

    Examples:
    ---------
    >>> re = ReverseEngineer('referenceThruster.stl', expectedLengthMm=1828.0)
    >>> profiles = re.extractProfiles()
    >>> params = re.fitParameters(profiles)
    >>> params.printSummary()
    '''

    def __init__(
        self,
        stlPath: str,
        expectedLengthMm: float = 1828.0,
        removeFins: bool = True,
        autoTransform: bool = True,
    ) -> None:
        '''
        Load and prepare a reference STL for reverse engineering.

        Applies auto-detected scale and axis transformations to bring
        the mesh into the project coordinate system (X=length, Y=width,
        Z=thickness, mm). Optionally removes fins for cleaner analysis.

        Parameters:
        -----------
        stlPath : str
            Path to the reference STL file
        expectedLengthMm : float
            Expected board length in mm (used for scale detection)
        removeFins : bool
            If True, segment and remove fins before analysis
        autoTransform : bool
            If True, auto-detect and apply scale/axis transformations
        '''
        if trimesh is None:
            raise ImportError(
                'trimesh is required for reverse engineering. '
                'Install with: pip install trimesh'
            )

        self._stlPath = Path(stlPath)
        self._expectedLengthMm = expectedLengthMm

        # Load raw mesh
        rawMesh = trimesh.load(str(self._stlPath))
        print(f'Loaded {self._stlPath.name}: '
              f'{len(rawMesh.vertices)} vertices, {len(rawMesh.faces)} faces')

        # Auto-detect and apply coordinate transformation
        if autoTransform:
            scaleFactor, axisMapping = CoordinateTransformer.autoDetect(
                rawMesh, expectedLengthMm
            )
            self._mesh = self._applyTransform(rawMesh, scaleFactor, axisMapping)

            isValid, msg = CoordinateTransformer.validateTransformation(
                rawMesh, self._mesh
            )
            print(f'  Transform: scale={scaleFactor:.4f}, axes={axisMapping}: {msg}')
        else:
            self._mesh = rawMesh

        # Repair the mesh for reliable slicing
        if not self._mesh.is_watertight:
            trimesh.repair.fill_holes(self._mesh)
            trimesh.repair.fix_normals(self._mesh)

        # Optionally remove fins
        self._bodyMesh = self._mesh
        if removeFins:
            segmenter = FinSegmenter(self._mesh)
            segmented = segmenter.segmentByHeight()
            if segmented.finMeshes:
                self._bodyMesh = segmented.bodyMesh
                print(f'  Removed {len(segmented.finMeshes)} fins '
                      f'({segmented.finConfiguration} configuration)')
            else:
                print('  No fins detected: using full mesh')

        # Center the body mesh so the nose is at X=0
        bounds = self._bodyMesh.bounds
        self._noseX = float(bounds[0, 0])
        self._tailX = float(bounds[1, 0])
        self._boardLength = self._tailX - self._noseX

    @property
    def boardLengthMm(self) -> float:
        '''Detected board length in mm.'''
        return self._boardLength

    @property
    def bodyMesh(self) -> 'trimesh.Trimesh':
        '''The body mesh (fins removed if requested).'''
        return self._bodyMesh

    #--------------------------------------------------------------------#
    # -- Profile Extraction -- #
    def getContourAt(self, t: float) -> Optional[np.ndarray]:
        '''
        Return the raw (Y, Z) mesh intersection contour at normalized position t.

        Thin public wrapper around _sliceAtX for external callers (e.g. the
        cross-section comparison viewer). The contour points are ordered by
        angle from the centroid and form a closed loop.

        Parameters:
        -----------
        t : float
            Normalized longitudinal position (0.0 = nose, 1.0 = tail)

        Returns:
        --------
        Optional[np.ndarray] : (N, 2) array of (Y, Z) contour points, or None
        '''
        xPos = self._noseX + t * self._boardLength
        return self._sliceAtX(float(xPos))

    #--------------------------------------------------------------------#

    def extractProfiles(
        self,
        nStations: int = 200,
        nLateralSamples: int = 50,
    ) -> ExtractedProfiles:
        '''
        Slice the mesh at many X stations and extract shape profiles.

        At each station, the mesh is sliced to obtain a 2D cross-section.
        The cross-section is analyzed to extract the outline half-width,
        rocker height, thickness, and deck/bottom surface profiles.

        Parameters:
        -----------
        nStations : int
            Number of longitudinal slicing stations
        nLateralSamples : int
            Number of lateral sample points for deck/bottom profiles

        Returns:
        --------
        ExtractedProfiles : Extracted shape data
        '''
        # Set up slicing positions: use a small margin at each end to avoid
        # degenerate slices right at the nose/tail points, but keep it tight
        # so we capture the steep taper at the tail tip accurately
        marginMm = 3.0  # 3mm from each end
        xPositions = np.linspace(
            self._noseX + marginMm,
            self._tailX - marginMm,
            nStations,
        )

        tValues = (xPositions - self._noseX) / self._boardLength
        lateralFractions = np.linspace(0.0, 1.0, nLateralSamples)

        halfWidths = np.zeros(nStations)
        rockerHeights = np.zeros(nStations)
        thicknesses = np.zeros(nStations)
        deckProfiles = []
        bottomProfiles = []

        for i, xPos in enumerate(xPositions):
            contour = self._sliceAtX(float(xPos))

            if contour is None or len(contour) < 3:
                # Degenerate slice: use zero values
                deckProfiles.append(np.zeros(nLateralSamples))
                bottomProfiles.append(np.zeros(nLateralSamples))
                continue

            # Extract envelope data from the contour
            hw, rz, thickness, deckZ, bottomZ = self._extractEnvelope(
                contour, lateralFractions
            )

            halfWidths[i] = hw
            rockerHeights[i] = rz
            thicknesses[i] = thickness
            deckProfiles.append(deckZ)
            bottomProfiles.append(bottomZ)

        # Normalize rocker: shift so the minimum (flat spot) is at Z=0
        rockerMin = np.min(rockerHeights)
        rockerHeights -= rockerMin

        return ExtractedProfiles(
            tValues=tValues,
            halfWidths=halfWidths,
            rockerHeights=rockerHeights,
            thicknesses=thicknesses,
            deckProfiles=deckProfiles,
            bottomProfiles=bottomProfiles,
            lateralFractions=lateralFractions,
            boardLengthMm=self._boardLength,
            maxWidthMm=float(2.0 * np.max(halfWidths)),
            maxThicknessMm=float(np.max(thicknesses)),
        )

    def _sliceAtX(self, xPosition: float) -> Optional[np.ndarray]:
        '''
        Slice the body mesh at a given X position and return an ordered
        2D cross-section contour.

        Parameters:
        -----------
        xPosition : float
            X-coordinate to slice at (mm)

        Returns:
        --------
        Optional[np.ndarray] : Ordered (Y, Z) contour points, or None
        '''
        planeOrigin = np.array([xPosition, 0.0, 0.0])
        planeNormal = np.array([1.0, 0.0, 0.0])

        try:
            lines = trimesh.intersections.mesh_plane(
                self._bodyMesh,
                plane_normal=planeNormal,
                plane_origin=planeOrigin,
            )
        except Exception:
            return None

        if lines is None or len(lines) == 0:
            return None

        # Collect unique (Y, Z) points from the intersection line segments
        points = lines.reshape(-1, 3)[:, 1:]  # drop X, keep Y and Z
        points = np.unique(np.round(points, 6), axis=0)

        if len(points) < 3:
            return None

        # Order points by angle from centroid to form a closed contour
        centroid = points.mean(axis=0)
        angles = np.arctan2(
            points[:, 1] - centroid[1],
            points[:, 0] - centroid[0],
        )
        sortedIndices = np.argsort(angles)
        return points[sortedIndices]

    def _extractEnvelope(
        self,
        contour: np.ndarray,
        lateralFractions: np.ndarray,
    ) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        '''
        Extract envelope data from an ordered (Y, Z) cross-section contour.

        Separates the contour into upper (deck) and lower (bottom) envelopes,
        then interpolates onto a normalized lateral fraction grid.

        Parameters:
        -----------
        contour : np.ndarray
            Ordered (Y, Z) cross-section points, shape (N, 2)
        lateralFractions : np.ndarray
            Lateral fraction grid (0 = center, 1 = rail)

        Returns:
        --------
        tuple : (halfWidth, rockerHeight, thickness, deckProfile, bottomProfile)
            - halfWidth: max |Y| extent (mm)
            - rockerHeight: bottom Z at centerline (mm)
            - thickness: deck Z - bottom Z at centerline (mm)
            - deckProfile: deck Z at each lateral fraction (mm)
            - bottomProfile: bottom Z at each lateral fraction (mm)
        '''
        yCoords = contour[:, 0]
        zCoords = contour[:, 1]

        # Half-width: maximum |Y| extent
        halfWidth = float(np.max(np.abs(yCoords)))

        if halfWidth < 0.1:
            # Degenerate cross-section
            nLateral = len(lateralFractions)
            return 0.0, 0.0, 0.0, np.zeros(nLateral), np.zeros(nLateral)

        # Use the positive-Y half for profile extraction (board is symmetric)
        # Include points near centerline (Y >= -1mm) to capture center values
        rightMask = yCoords >= -1.0
        rightY = yCoords[rightMask]
        rightZ = zCoords[rightMask]

        if len(rightY) < 3:
            # Fall back to all points
            rightY = yCoords
            rightZ = zCoords

        # Build upper (deck) and lower (bottom) envelopes
        # Bin points by Y position and find max Z (deck) and min Z (bottom)
        nBins = max(20, len(lateralFractions))
        yBins = np.linspace(0.0, halfWidth, nBins)
        binWidth = halfWidth / (nBins - 1) if nBins > 1 else halfWidth

        deckEnvelope = np.full(nBins, np.nan)
        bottomEnvelope = np.full(nBins, np.nan)

        for j in range(nBins):
            yCenter = yBins[j]
            # Find points within this Y bin
            inBin = np.abs(rightY - yCenter) <= binWidth * 0.75
            if np.any(inBin):
                deckEnvelope[j] = float(np.max(rightZ[inBin]))
                bottomEnvelope[j] = float(np.min(rightZ[inBin]))

        # Fill NaN gaps by linear interpolation
        deckEnvelope = self._fillNans(deckEnvelope, yBins)
        bottomEnvelope = self._fillNans(bottomEnvelope, yBins)

        # Rocker height: midpoint of deck and bottom at centerline.
        # This corresponds to the parametric model's center plane Z, which
        # is what RockerProfile.getRockerHeight(t) defines. Using the midpoint
        # avoids the crown/concave offsets that shift deck and bottom surfaces.
        rockerHeight = float((deckEnvelope[0] + bottomEnvelope[0]) / 2.0)

        # Structural thickness: measure at ~85% of the half-width where
        # crown and concave contributions are negligible but rail rounding
        # (if present) has not yet started tapering. This avoids both the
        # centerline measurement (which includes crown + concave) and the
        # rail-edge measurement (which may be tapered by rounding).
        railMeasureIdx = int(0.85 * (nBins - 1))
        nearRailThickness = float(
            deckEnvelope[railMeasureIdx] - bottomEnvelope[railMeasureIdx]
        )
        centerThickness = float(deckEnvelope[0] - bottomEnvelope[0])
        # The near-rail measurement should be close to the structural
        # thickness. If it looks reasonable, use it; otherwise use center.
        if 0.1 < nearRailThickness < centerThickness:
            thickness = nearRailThickness
        else:
            thickness = centerThickness

        # Interpolate onto the lateral fraction grid
        # lateralFractions: 0 → center, 1 → rail
        yTarget = lateralFractions * halfWidth
        deckProfile = np.interp(yTarget, yBins, deckEnvelope)
        bottomProfile = np.interp(yTarget, yBins, bottomEnvelope)

        # Convert profiles to be relative to the rocker baseline (midZ)
        # so they represent local cross-section shape above/below center plane
        deckProfile -= rockerHeight
        bottomProfile -= rockerHeight

        return halfWidth, rockerHeight, thickness, deckProfile, bottomProfile

    @staticmethod
    def _fillNans(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
        '''
        Fill NaN values in an array by linear interpolation.

        Parameters:
        -----------
        values : np.ndarray
            Array with potential NaN gaps
        positions : np.ndarray
            Corresponding position values for interpolation

        Returns:
        --------
        np.ndarray : Array with NaNs filled
        '''
        valid = ~np.isnan(values)
        if not np.any(valid):
            return np.zeros_like(values)
        if np.all(valid):
            return values

        filled = values.copy()
        filled[~valid] = np.interp(
            positions[~valid], positions[valid], values[valid]
        )
        return filled

    #--------------------------------------------------------------------#
    # -- Parameter Fitting -- #
    #--------------------------------------------------------------------#

    def fitParameters(
        self,
        profiles: Optional[ExtractedProfiles] = None,
    ) -> SurfboardParameters:
        '''
        Fit SurfboardParameters from extracted profiles.

        Uses direct measurement extraction for most parameters:
        board dimensions from bounding extents, outline control points
        from the half-width curve, rocker heights from curve endpoints.

        Parameters:
        -----------
        profiles : ExtractedProfiles | None
            Pre-extracted profiles. If None, extracts them first.

        Returns:
        --------
        SurfboardParameters : Fitted parameters matching the reference shape
        '''
        if profiles is None:
            profiles = self.extractProfiles()

        length = profiles.boardLengthMm
        maxWidth = profiles.maxWidthMm
        maxThickness = profiles.maxThicknessMm

        # -- Outline parameters --

        # Wide point: position of maximum half-width
        widePointIdx = int(np.argmax(profiles.halfWidths))
        widePointT = float(profiles.tValues[widePointIdx])
        # Convert to offset from board center
        widePointOffset = (widePointT - 0.5) * length

        # Nose width: interpolate half-width at 305mm from nose
        noseStationT = 305.0 / length
        noseHalfWidth = float(np.interp(
            noseStationT, profiles.tValues, profiles.halfWidths
        ))
        noseWidth = 2.0 * noseHalfWidth

        # Tail width: interpolate half-width at 305mm from tail
        tailStationT = 1.0 - (305.0 / length)
        tailHalfWidth = float(np.interp(
            tailStationT, profiles.tValues, profiles.halfWidths
        ))
        tailWidth = 2.0 * tailHalfWidth

        # Tail tip half-width: the outline tapers steeply at the tail tip,
        # so polynomial extrapolation is unreliable. Instead, use the
        # minimum halfWidth observed in the tail region (last 5% of board),
        # which approximates the asymptotic tail tip value.
        nFit = min(10, len(profiles.tValues) // 5)
        tailRegion = profiles.halfWidths[-nFit:]
        tailTipHalfWidth = float(np.min(tailRegion))
        tailTipHalfWidth = max(0.0, tailTipHalfWidth)

        # -- Rocker parameters --
        # Extrapolate to t=0 and t=1 using linear fits of the endpoints

        # Nose rocker: extrapolate from first few points to t=0
        noseRocker = self._extrapolateToEndpoint(
            profiles.tValues[:nFit], profiles.rockerHeights[:nFit], targetT=0.0
        )
        noseRocker = max(0.0, noseRocker)

        # Tail rocker: extrapolate from last few points to t=1
        tailRocker = self._extrapolateToEndpoint(
            profiles.tValues[-nFit:], profiles.rockerHeights[-nFit:], targetT=1.0
        )
        tailRocker = max(0.0, tailRocker)

        # -- Cross-section parameters --

        # Find the thickest station for crown/concave analysis
        thickestIdx = int(np.argmax(profiles.thicknesses))

        # Deck crown: difference between deck Z at center vs. near-rail
        # at the thickest station. Measure at ~85% of the half-width to
        # avoid rail rounding effects at the edge, while still being far
        # enough from center that the crown cosine has mostly decayed.
        deckCrown = 0.0
        if thickestIdx < len(profiles.deckProfiles):
            deckProfile = profiles.deckProfiles[thickestIdx]
            if len(deckProfile) > 1:
                nearRailIdx = int(0.85 * (len(deckProfile) - 1))
                deckCrown = max(0.0, float(
                    deckProfile[0] - deckProfile[nearRailIdx]
                ))

        # Bottom concave: difference between bottom Z near-rail vs. center
        # at the thickest station. Same 85% measurement point to avoid
        # rail rounding artifacts.
        bottomConcave = 0.0
        if thickestIdx < len(profiles.bottomProfiles):
            bottomProfile = profiles.bottomProfiles[thickestIdx]
            if len(bottomProfile) > 1:
                nearRailIdx = int(0.85 * (len(bottomProfile) - 1))
                centerBottom = float(bottomProfile[0])
                nearRailBottom = float(bottomProfile[nearRailIdx])
                bottomConcave = max(0.0, nearRailBottom - centerBottom)

        # Clamp to reasonable ranges
        deckCrown = min(deckCrown, 20.0)
        bottomConcave = min(bottomConcave, 10.0)

        # Assemble the fitted parameters
        params = SurfboardParameters(
            length=round(length, 1),
            maxWidth=round(maxWidth, 1),
            maxThickness=round(maxThickness, 1),
            noseWidth=round(noseWidth, 1),
            tailWidth=round(tailWidth, 1),
            widePointOffset=round(widePointOffset, 1),
            tailTipHalfWidth=round(tailTipHalfWidth, 1),
            noseRocker=round(noseRocker, 1),
            tailRocker=round(tailRocker, 1),
            deckCrown=round(deckCrown, 1),
            bottomConcave=round(bottomConcave, 1),
        )

        return params

    #--------------------------------------------------------------------#
    # -- Parametric Comparison -- #
    #--------------------------------------------------------------------#

    def compareToParametric(
        self,
        params: SurfboardParameters,
        profiles: Optional[ExtractedProfiles] = None,
    ) -> dict:
        '''
        Compare extracted profiles against a parametric model.

        Generates outline, rocker, and thickness curves from the given
        parameters and computes per-curve RMS errors vs. the extracted
        profiles.

        Parameters:
        -----------
        params : SurfboardParameters
            Parameters to evaluate
        profiles : ExtractedProfiles | None
            Pre-extracted profiles. If None, extracts them first.

        Returns:
        --------
        dict : Comparison results with per-curve RMS errors in mm
        '''
        if profiles is None:
            profiles = self.extractProfiles()

        outline = Outline(params)
        rocker = RockerProfile(params)
        crossSection = CrossSection(params)

        nStations = len(profiles.tValues)

        # Outline comparison
        paramHalfWidths = np.array([
            outline.getHalfWidth(float(t)) for t in profiles.tValues
        ])
        outlineRms = float(np.sqrt(np.mean(
            (profiles.halfWidths - paramHalfWidths) ** 2
        )))

        # Rocker comparison
        paramRocker = np.array([
            rocker.getRockerHeight(float(t)) for t in profiles.tValues
        ])
        rockerRms = float(np.sqrt(np.mean(
            (profiles.rockerHeights - paramRocker) ** 2
        )))

        # Thickness comparison
        paramThicknesses = np.array([
            crossSection.getLocalThickness(float(t)) for t in profiles.tValues
        ])
        thicknessRms = float(np.sqrt(np.mean(
            (profiles.thicknesses - paramThicknesses) ** 2
        )))

        # Cross-section profile comparison (at the thickest station)
        thickestIdx = int(np.argmax(profiles.thicknesses))
        tThickest = float(profiles.tValues[thickestIdx])

        # Parametric deck profile at the thickest station
        paramDeck = np.array([
            crossSection.getDeckHeight(tThickest, float(lf))
            for lf in profiles.lateralFractions
        ])
        extractedDeck = profiles.deckProfiles[thickestIdx]
        deckRms = float(np.sqrt(np.mean((extractedDeck - paramDeck) ** 2)))

        # Parametric bottom profile at the thickest station
        paramBottom = np.array([
            crossSection.getBottomHeight(tThickest, float(lf))
            for lf in profiles.lateralFractions
        ])
        extractedBottom = profiles.bottomProfiles[thickestIdx]
        bottomRms = float(np.sqrt(np.mean((extractedBottom - paramBottom) ** 2)))

        return {
            'outlineRmsMm': round(outlineRms, 3),
            'rockerRmsMm': round(rockerRms, 3),
            'thicknessRmsMm': round(thicknessRms, 3),
            'deckProfileRmsMm': round(deckRms, 3),
            'bottomProfileRmsMm': round(bottomRms, 3),
            'nStations': nStations,
        }

    #--------------------------------------------------------------------#
    # -- Helpers -- #
    #--------------------------------------------------------------------#

    @staticmethod
    def _extrapolateToEndpoint(
        tValues: np.ndarray,
        values: np.ndarray,
        targetT: float,
    ) -> float:
        '''
        Extrapolate a curve to a target t value using a polynomial fit.

        Uses a quadratic fit (degree 2) of the provided data points to
        predict the value at targetT, which lies just beyond the data range.

        Parameters:
        -----------
        tValues : np.ndarray
            t positions of the data points
        values : np.ndarray
            Curve values at those positions
        targetT : float
            Target t value to extrapolate to

        Returns:
        --------
        float : Extrapolated value at targetT
        '''
        if len(tValues) < 2:
            return float(values[0]) if len(values) > 0 else 0.0

        # Use quadratic fit for smooth extrapolation
        degree = min(2, len(tValues) - 1)
        coeffs = np.polyfit(tValues, values, degree)
        return float(np.polyval(coeffs, targetT))

    @staticmethod
    def _applyTransform(
        mesh: 'trimesh.Trimesh',
        scaleFactor: float,
        axisMapping: str,
    ) -> 'trimesh.Trimesh':
        '''
        Apply scale and axis remapping to a mesh.

        Parameters:
        -----------
        mesh : trimesh.Trimesh
            Input mesh
        scaleFactor : float
            Scale multiplier
        axisMapping : str
            Axis remapping string (e.g., 'zyx')

        Returns:
        --------
        trimesh.Trimesh : Transformed mesh copy
        '''
        transformed = mesh.copy()
        vertices = transformed.vertices.copy()

        # Apply axis remapping
        axisMap = {'x': 0, 'y': 1, 'z': 2}
        if len(axisMapping) == 3:
            newVertices = np.zeros_like(vertices)
            for newAxis, oldAxisChar in enumerate(axisMapping):
                oldAxis = axisMap[oldAxisChar]
                newVertices[:, newAxis] = vertices[:, oldAxis]
            vertices = newVertices

        # Apply scale
        vertices *= scaleFactor

        transformed.vertices = vertices
        return transformed
