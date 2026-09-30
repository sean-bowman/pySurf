# -- Parametric Surface Mesh Generator -- #

'''

Generates a watertight triangulated surface mesh from the parametric surfboard model.

Stitches cross-section contours from nose to tail into a closed mesh,
with fan caps at the nose and tail tips. A pure-Python approach with
direct STL export and no external geometry kernel dependency.

The algorithm:
  1. Sample N longitudinal stations along the board length
  2. At each station, build a closed cross-section contour from the
     deck and bottom surface profiles, mirrored for left/right symmetry
  3. Triangulate quad strips between adjacent contour rings
  4. Fan-cap the nose tip and planar-cap the tail end
  5. Assemble into a trimesh.Trimesh for export

All geometry is in millimeters (matching the project coordinate system).

Sean Bowman [02/07/2026]

'''

from __future__ import annotations

import math
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


#--------------------------------------------------------------------#
# -- Resolution Presets -- #
#--------------------------------------------------------------------#

RESOLUTION_PRESETS = {
    'draft': {'nLongitudinal': 100, 'nLateral': 25},
    'standard': {'nLongitudinal': 300, 'nLateral': 60},
    'high': {'nLongitudinal': 500, 'nLateral': 100},
}

# Minimum half-width below which a station collapses to a tip point (mm)
_TIP_THRESHOLD_MM = 0.1


class SurfaceMeshGenerator:

    '''

    Generates a watertight triangulated surface mesh from the parametric
    surfboard model by stitching cross-section contours from nose to tail.

    The mesh is built by:
      1. Sampling closed cross-section contours at many longitudinal stations
      2. Triangulating quad strips between adjacent contours
      3. Capping the nose and tail to form a watertight solid

    All internal geometry and exported vertices are in millimeters.

    Examples:
    ---------
    >>> from pysurf.Surfboard.SurfPhysics.geometry.parameters import SurfboardParameters
    >>> params = SurfboardParameters.shortboard()
    >>> gen = SurfaceMeshGenerator(params, nLongitudinal=300, nLateral=60)
    >>> gen.exportStl('shortboard.stl')

    '''

    def __init__(
        self,
        params: SurfboardParameters,
        nLongitudinal: int = 300,
        nLateral: int = 60,
    ) -> None:

        '''

        Initialize the mesh generator from surfboard parameters.

        Parameters:
        -----------
        params : SurfboardParameters
            Board dimensions (all in mm)
        nLongitudinal : int
            Number of cross-section stations along the board length.
            More stations = smoother longitudinal resolution.
        nLateral : int
            Number of sample points across the half-width for each
            surface (deck and bottom). More points = smoother
            cross-section contour resolution.

        '''

        if trimesh is None:
            raise ImportError(
                'trimesh is required for surface mesh generation. '
                'Install with: pip install trimesh'
            )

        self._params = params
        self._nLongitudinal = nLongitudinal
        self._nLateral = nLateral

        # Reuse existing parametric geometry classes
        self._outline = Outline(params)
        self._rocker = RockerProfile(params)
        self._crossSection = CrossSection(params)

        # Cached mesh: generated on first call to generate() or getMesh()
        self._mesh: Optional[trimesh.Trimesh] = None

    @classmethod
    def fromPreset(
        cls,
        params: SurfboardParameters,
        preset: str = 'standard',
    ) -> SurfaceMeshGenerator:

        '''

        Create a mesh generator using a named resolution preset.

        Parameters:
        -----------
        params : SurfboardParameters
            Board dimensions
        preset : str
            Resolution preset name: 'draft', 'standard', or 'high'

        Returns:
        --------
        SurfaceMeshGenerator : Configured generator instance

        '''

        if preset not in RESOLUTION_PRESETS:
            raise ValueError(
                f'Unknown preset \'{preset}\'. '
                f'Available: {list(RESOLUTION_PRESETS.keys())}'
            )
        settings = RESOLUTION_PRESETS[preset]
        return cls(params, **settings)

    @property
    def pointsPerContour(self) -> int:

        '''

        Number of unique vertices in each cross-section contour ring.

        The contour traces the full board perimeter at a station:
        right deck + right bottom + left bottom + left deck,
        with shared centerline and rail transition points removed.

        Returns:
        --------
        int : M = 4 * nLateral - 2

        '''

        return 4 * self._nLateral - 2

    #--------------------------------------------------------------------#
    # -- Public Generation & Export -- #
    #--------------------------------------------------------------------#

    def generate(self) -> trimesh.Trimesh:

        '''

        Generate the complete surfboard surface mesh.

        Samples cross-section contours at each longitudinal station,
        triangulates between them, and caps the nose and tail.

        Returns:
        --------
        trimesh.Trimesh : Watertight triangulated surface mesh (vertices in mm)

        '''

        stations = self._sampleStations()
        vertices, faces = self._buildMesh(stations)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        trimesh.repair.fix_normals(mesh)

        # Attempt to fill any small holes from numeric edge cases
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)

        self._mesh = mesh
        return mesh

    def getMesh(self) -> trimesh.Trimesh:

        '''

        Get the generated mesh, creating it on first access.

        Returns:
        --------
        trimesh.Trimesh : The surface mesh

        '''

        if self._mesh is None:
            self.generate()
        return self._mesh

    def exportStl(self, filePath: str, binary: bool = True) -> None:

        '''

        Export the surfboard mesh as an STL file.

        Parameters:
        -----------
        filePath : str
            Output file path (should end in .stl)
        binary : bool
            If True, write binary STL (smaller). If False, write ASCII STL.

        '''

        mesh = self.getMesh()
        fileType = 'stl' if binary else 'stl_ascii'
        mesh.export(filePath, file_type=fileType)

    def computeVolume(self) -> float:

        '''

        Compute the mesh volume in mm^3.

        Useful for validation against BoardGeometry.computeVolume().

        Returns:
        --------
        float : Volume in mm^3 (0 if mesh is not watertight)

        '''

        mesh = self.getMesh()
        if not mesh.is_watertight:
            return 0.0
        return float(abs(mesh.volume))

    def computeVolumeLiters(self) -> float:

        '''

        Compute the mesh volume in liters.

        Returns:
        --------
        float : Volume in liters

        '''

        return self.computeVolume() / 1_000_000.0

    #--------------------------------------------------------------------#
    # -- Station Sampling -- #
    #--------------------------------------------------------------------#

    def _sampleStations(self) -> list[dict]:

        '''

        Sample cross-section contours at each longitudinal station.

        At each station, queries the parametric model for the half-width,
        rocker height, and deck/bottom surface profiles. Stations where
        the half-width falls below the tip threshold are recorded as
        single-point tips (used for nose/tail fan capping).

        Returns:
        --------
        list[dict] : Station data, each containing:
            - 't': normalized position (0 = nose, 1 = tail)
            - 'x': x position in mm
            - 'isTip': True if the station is a degenerate tip point
            - 'contour': np.ndarray of shape (M, 3) or (1, 3) for tips

        '''

        tValues = np.linspace(0.0, 1.0, self._nLongitudinal)
        stations = []

        for t in tValues:
            x = float(t) * self._params.length
            hw = self._outline.getHalfWidth(float(t))
            rz = self._rocker.getRockerHeight(float(t))

            if hw < _TIP_THRESHOLD_MM:
                # Degenerate station: collapse to a single tip point
                stations.append({
                    't': float(t),
                    'x': x,
                    'isTip': True,
                    'contour': np.array([[x, 0.0, rz]]),
                })
            else:
                contour = self._buildContour(float(t), x, hw, rz)
                stations.append({
                    't': float(t),
                    'x': x,
                    'isTip': False,
                    'contour': contour,
                })

        return stations

    def _buildContour(
        self, t: float, x: float, hw: float, rz: float
    ) -> np.ndarray:

        '''

        Build a closed cross-section contour at a given station.

        The contour traces the full perimeter of the cross-section as a
        closed polygon. Starting from the deck centerline, it proceeds:

          1. Right deck   (lf 0→1) : centerline to right rail, deck surface
          2. Right bottom (lf 1→0) : right rail to centerline, bottom surface
          3. Left bottom  (lf 0→1) : centerline to left rail, bottom surface
          4. Left deck    (lf 1→0) : left rail to centerline, deck surface

        Shared centerline points between segments 2→3 and 4→1 are deduplicated,
        yielding M = 4*nLateral - 2 unique vertices per contour ring.

        Rail rounding is applied near the rail edge (lf close to 1.0) using
        an elliptical taper based on the railRadius parameter. The deck and
        bottom surfaces converge toward their midpoint, producing the rounded
        rail profile characteristic of real surfboards.

        Parameters:
        -----------
        t : float
            Normalized longitudinal position (0 = nose, 1 = tail)
        x : float
            X position in mm along the board length
        hw : float
            Half-width at this station in mm
        rz : float
            Rocker Z-offset at this station in mm

        Returns:
        --------
        np.ndarray : Contour vertices, shape (M, 3) where M = 4*nLateral - 2

        '''

        n = self._nLateral
        lfs = np.linspace(0.0, 1.0, n)
        points = []

        # Pre-compute rail blend parameters (constant for this station).
        # The blend zone is the region near the rail edge where elliptical
        # rounding tapers the deck and bottom toward each other.
        railRadius = self._params.railRadius
        if railRadius > 0.1:
            # Clamp effective radius to 40% of half-width for narrow stations
            effectiveRadius = min(railRadius, 0.4 * hw)
            # Blend threshold: below this lf, no rounding is applied
            lfBlend = max(1.0 - effectiveRadius / hw, 0.6)
            # Cross-section heights at the blend boundary (the "entry" values
            # that define full thickness before the taper begins)
            entryDeckZ = self._crossSection.getDeckHeight(t, lfBlend)
            entryBottomZ = self._crossSection.getBottomHeight(t, lfBlend)
            entryMidZ = (entryDeckZ + entryBottomZ) / 2.0
        else:
            effectiveRadius = 0.0
            lfBlend = 1.0
            entryDeckZ = 0.0
            entryBottomZ = 0.0
            entryMidZ = 0.0

        # Segment 1: Right deck, lf 0→1 (n points)
        # Deck centerline → right deck rail
        for lf in lfs:
            y = float(lf) * hw
            deckZ = self._railRoundedDeck(
                t, float(lf), effectiveRadius, lfBlend,
                entryDeckZ, entryMidZ,
            )
            points.append([x, y, rz + deckZ])

        # Segment 2: Right bottom, lf 1→0 (n points)
        # Right bottom rail → bottom centerline
        for lf in reversed(lfs):
            y = float(lf) * hw
            bottomZ = self._railRoundedBottom(
                t, float(lf), effectiveRadius, lfBlend,
                entryBottomZ, entryMidZ,
            )
            points.append([x, y, rz + bottomZ])

        # Segment 3: Left bottom, lf 0→1 with Y negated (n-1 points)
        # Skip lf=0: that point duplicates the last point of segment 2
        # (bottom centerline at y=0)
        for lf in lfs[1:]:
            y = -float(lf) * hw
            bottomZ = self._railRoundedBottom(
                t, float(lf), effectiveRadius, lfBlend,
                entryBottomZ, entryMidZ,
            )
            points.append([x, y, rz + bottomZ])

        # Segment 4: Left deck, lf 1→0 with Y negated (n-1 points)
        # Skip lf=0: that point would duplicate the first point of
        # segment 1 (deck centerline at y=0), closing the loop implicitly
        for lf in reversed(lfs[1:]):
            y = -float(lf) * hw
            deckZ = self._railRoundedDeck(
                t, float(lf), effectiveRadius, lfBlend,
                entryDeckZ, entryMidZ,
            )
            points.append([x, y, rz + deckZ])

        return np.array(points)

    def _railRoundedDeck(
        self,
        t: float,
        lf: float,
        effectiveRadius: float,
        lfBlend: float,
        entryDeckZ: float,
        entryMidZ: float,
    ) -> float:

        '''

        Get deck height with rail rounding applied.

        For lf values inside the blend zone (lf > lfBlend), the deck height
        is tapered toward the entry midpoint using an elliptical profile.
        For lf values outside the blend zone, the raw cross-section value
        is returned unmodified.

        Parameters:
        -----------
        t : float
            Normalized longitudinal position
        lf : float
            Lateral fraction (0 = centerline, 1 = rail edge)
        effectiveRadius : float
            Clamped rail radius for this station (0 = no rounding)
        lfBlend : float
            Lateral fraction threshold where blending begins
        entryDeckZ : float
            Deck height at the blend boundary (lfBlend)
        entryMidZ : float
            Midpoint of deck and bottom at the blend boundary

        Returns:
        --------
        float : Deck height in mm relative to center plane

        '''

        if effectiveRadius > 0.0 and lf > lfBlend:
            blendT = (lf - lfBlend) / (1.0 - lfBlend)
            railFactor = math.sqrt(max(0.0, 1.0 - blendT * blendT))
            # Floor to avoid degenerate triangles at the rail tip (~0.6mm)
            railFactor = max(railFactor, 0.01)
            return entryMidZ + (entryDeckZ - entryMidZ) * railFactor
        return self._crossSection.getDeckHeight(t, lf)

    def _railRoundedBottom(
        self,
        t: float,
        lf: float,
        effectiveRadius: float,
        lfBlend: float,
        entryBottomZ: float,
        entryMidZ: float,
    ) -> float:

        '''

        Get bottom height with rail rounding applied.

        Mirror of _railRoundedDeck for the bottom surface.

        Parameters:
        -----------
        t : float
            Normalized longitudinal position
        lf : float
            Lateral fraction (0 = centerline, 1 = rail edge)
        effectiveRadius : float
            Clamped rail radius for this station (0 = no rounding)
        lfBlend : float
            Lateral fraction threshold where blending begins
        entryBottomZ : float
            Bottom height at the blend boundary (lfBlend)
        entryMidZ : float
            Midpoint of deck and bottom at the blend boundary

        Returns:
        --------
        float : Bottom height in mm relative to center plane

        '''

        if effectiveRadius > 0.0 and lf > lfBlend:
            blendT = (lf - lfBlend) / (1.0 - lfBlend)
            railFactor = math.sqrt(max(0.0, 1.0 - blendT * blendT))
            railFactor = max(railFactor, 0.01)
            return entryMidZ + (entryBottomZ - entryMidZ) * railFactor
        return self._crossSection.getBottomHeight(t, lf)

    #--------------------------------------------------------------------#
    # -- Mesh Assembly -- #
    #--------------------------------------------------------------------#

    def _buildMesh(
        self, stations: list[dict]
    ) -> tuple[np.ndarray, np.ndarray]:

        '''

        Assemble the complete mesh from sampled station contours.

        Combines quad-strip triangulation between adjacent contours
        with fan capping at the nose/tail tips and planar capping
        at any open terminal contour.

        Parameters:
        -----------
        stations : list[dict]
            Station data from _sampleStations()

        Returns:
        --------
        tuple[np.ndarray, np.ndarray] :
            (vertices shape (V, 3), faces shape (F, 3))

        '''

        # Collect vertices and faces as Python lists for flexible appending,
        # then convert to numpy arrays at the end
        vertexList: list[list[float]] = []
        faceList: list[list[int]] = []

        # Track the starting vertex index for each station
        stationOffsets: list[int] = []

        for station in stations:
            stationOffsets.append(len(vertexList))
            for pt in station['contour']:
                vertexList.append(pt.tolist())

        # Triangulate between each pair of adjacent stations
        for i in range(len(stations) - 1):
            stationA = stations[i]
            stationB = stations[i + 1]
            offA = stationOffsets[i]
            offB = stationOffsets[i + 1]

            tipA = stationA['isTip']
            tipB = stationB['isTip']

            if tipA and tipB:
                # Both degenerate: no surface between two coincident tips
                continue

            elif tipA and not tipB:
                # Nose-side fan: tip vertex → first full contour ring
                self._addFanCap(faceList, offA, offB, len(stationB['contour']),
                                tipBeforeContour=True)

            elif not tipA and tipB:
                # Tail-side fan: last full contour ring → tip vertex
                self._addFanCap(faceList, offB, offA, len(stationA['contour']),
                                tipBeforeContour=False)

            else:
                # Both full contours: quad-strip triangulation
                self._addQuadStrip(faceList, offA, offB,
                                   len(stationA['contour']),
                                   len(stationB['contour']))

        # Cap the nose end if the first station is a full contour (not a tip)
        # This is unusual, normally t=0 has halfWidth=0, but handle it
        if not stations[0]['isTip']:
            self._addPlanarCap(vertexList, faceList, stationOffsets[0],
                               len(stations[0]['contour']), facingNose=True)

        # Cap the tail end if the last station is a full contour (not a tip)
        # This is the common case: tailTipHalfWidth > 0 for most boards
        if not stations[-1]['isTip']:
            self._addPlanarCap(vertexList, faceList, stationOffsets[-1],
                               len(stations[-1]['contour']), facingNose=False)

        vertices = np.array(vertexList, dtype=np.float64)
        faces = np.array(faceList, dtype=np.int64)

        return vertices, faces

    def _addQuadStrip(
        self,
        faceList: list[list[int]],
        offsetA: int,
        offsetB: int,
        nPointsA: int,
        nPointsB: int,
    ) -> None:

        '''

        Triangulate a quad strip between two contour rings of equal size.

        Each pair of corresponding edges on the two rings forms a quad,
        which is split into two triangles.

        Parameters:
        -----------
        faceList : list[list[int]]
            Accumulator for face triples (modified in place)
        offsetA : int
            Starting vertex index for the first (nose-side) contour
        offsetB : int
            Starting vertex index for the second (tail-side) contour
        nPointsA : int
            Number of vertices in contour A
        nPointsB : int
            Number of vertices in contour B

        '''

        # Use the smaller contour size if they differ (safety fallback:
        # in normal operation all contours have the same M points)
        n = min(nPointsA, nPointsB)

        for j in range(n):
            jNext = (j + 1) % n

            v0 = offsetA + j
            v1 = offsetA + jNext
            v2 = offsetB + j
            v3 = offsetB + jNext

            # Two triangles per quad
            faceList.append([v0, v2, v1])
            faceList.append([v1, v2, v3])

    def _addFanCap(
        self,
        faceList: list[list[int]],
        tipOffset: int,
        contourOffset: int,
        nContourPoints: int,
        tipBeforeContour: bool,
    ) -> None:

        '''

        Fan-triangulate from a single tip vertex to a contour ring.

        Used when a station collapses to a single point (nose or tail tip)
        and the adjacent station is a full contour.

        Parameters:
        -----------
        faceList : list[list[int]]
            Accumulator for face triples (modified in place)
        tipOffset : int
            Vertex index of the tip point
        contourOffset : int
            Starting vertex index of the contour ring
        nContourPoints : int
            Number of vertices in the contour ring
        tipBeforeContour : bool
            True if the tip is on the nose side (before the contour in X).
            Affects face winding for consistent outward normals.

        '''

        for j in range(nContourPoints):
            jNext = (j + 1) % nContourPoints

            if tipBeforeContour:
                # Tip is at lower X (nose), contour is at higher X
                faceList.append([tipOffset, contourOffset + j,
                                 contourOffset + jNext])
            else:
                # Tip is at higher X (tail), contour is at lower X
                faceList.append([tipOffset, contourOffset + jNext,
                                 contourOffset + j])

    def _addPlanarCap(
        self,
        vertexList: list[list[float]],
        faceList: list[list[int]],
        contourOffset: int,
        nPoints: int,
        facingNose: bool,
    ) -> None:

        '''

        Close an open terminal contour ring with a planar fan cap.

        Computes the centroid of the contour, adds it as a new vertex,
        and fan-triangulates from the centroid to each edge of the ring.
        This is more robust than fan from a ring vertex because it handles
        mildly non-convex cross-sections (e.g., bottom concave).

        Parameters:
        -----------
        vertexList : list[list[float]]
            Accumulator for vertices (modified in place: centroid appended)
        faceList : list[list[int]]
            Accumulator for face triples (modified in place)
        contourOffset : int
            Starting vertex index of the contour ring
        nPoints : int
            Number of vertices in the contour ring
        facingNose : bool
            True if the cap should face toward the nose (-X direction).
            False for a tail-facing cap (+X direction).

        '''

        # Compute centroid of the contour ring
        centroid = np.zeros(3)
        for i in range(nPoints):
            centroid += np.array(vertexList[contourOffset + i])
        centroid /= nPoints

        # Add centroid as a new vertex
        centroidIdx = len(vertexList)
        vertexList.append(centroid.tolist())

        # Fan-triangulate from centroid to each contour edge
        for j in range(nPoints):
            jNext = (j + 1) % nPoints

            if facingNose:
                # Normal should face -X (toward nose)
                faceList.append([centroidIdx, contourOffset + jNext,
                                 contourOffset + j])
            else:
                # Normal should face +X (toward tail)
                faceList.append([centroidIdx, contourOffset + j,
                                 contourOffset + jNext])
