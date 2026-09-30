# -- Mesh Comparison Analyzer -- #

'''
Main orchestrator for SDF-based mesh comparison.

Coordinates alignment, distance computation, fin segmentation,
visualization, and report generation for geometry validation.

Sean Bowman [02/05/2026]
'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

from pysurf.Surfboard.SurfPhysics.validation.meshAlignment import MeshAligner, AlignmentResult
from pysurf.Surfboard.SurfPhysics.validation.distanceMetrics import (
    DistanceAnalyzer, DistanceStatistics, RegionalDeviation
)
from pysurf.Surfboard.SurfPhysics.validation.finSegmentation import FinSegmenter, SegmentedMesh
from pysurf.Surfboard.SurfPhysics.validation.deviationVisualizer import DeviationVisualizer
from pysurf.Surfboard.SurfPhysics.validation.comparisonReport import (
    ComparisonReportGenerator, ComparisonReportData
)


#--------------------------------------------------------------------#
# -- Comparison Result -- #
#--------------------------------------------------------------------#

@dataclass
class ComparisonResult:
    '''Complete results from mesh comparison analysis.'''
    # Meshes
    referenceMesh: 'trimesh.Trimesh'
    generatedMesh: 'trimesh.Trimesh'
    alignedMesh: 'trimesh.Trimesh'

    # Alignment
    alignmentResult: AlignmentResult

    # Distance metrics
    samplePoints: np.ndarray
    signedDistances: np.ndarray
    overallStats: DistanceStatistics
    regionalDeviations: list[RegionalDeviation]
    hausdorffMm: float

    # Fin analysis (optional)
    referenceSegmented: SegmentedMesh | None
    generatedSegmented: SegmentedMesh | None
    finComparisonStats: dict | None

    # Report data
    reportData: ComparisonReportData | None

    def toDict(self) -> dict:
        '''Convert key metrics to dictionary for quick inspection.'''
        return {
            'alignment': {
                'method': self.alignmentResult.method,
                'cost': self.alignmentResult.cost,
                'boundingBoxIou': self.alignmentResult.boundingBoxIou,
            },
            'deviations': {
                'rmsMm': self.overallStats.rmsMm,
                'maxMm': max(abs(self.overallStats.minMm), abs(self.overallStats.maxMm)),
                'meanMm': self.overallStats.meanMm,
                'hausdorffMm': self.hausdorffMm,
            },
            'finComparison': self.finComparisonStats,
        }


#--------------------------------------------------------------------#
# -- Mesh Comparison Analyzer -- #
#--------------------------------------------------------------------#

class MeshComparisonAnalyzer:
    '''
    Orchestrates complete mesh comparison workflow.

    Typical usage:
        analyzer = MeshComparisonAnalyzer('reference.stl', 'generated.stl')
        results = analyzer.runFullComparison()
        analyzer.generateReport(results, outputPath='reports/')
    '''

    def __init__(
        self,
        referencePath: str | Path,
        generatedPath: str | Path,
        decimateFactor: float = 0.05,
        forceDecimation: bool = False,
        referenceScaleFactor: float = 1.0,
        referenceAxisMapping: str | None = None,
    ) -> None:
        '''
        Initialize with mesh file paths.

        Parameters:
        -----------
        referencePath : str | Path
            Path to reference STL file
        generatedPath : str | Path
            Path to generated STL file
        decimateFactor : float
            Target face count as fraction of original (0.05 = 5%)
            Only applied to meshes with >100k faces
        forceDecimation : bool
            If True, always decimate regardless of mesh size
        referenceScaleFactor : float
            Scale factor to apply to reference mesh (e.g., 1000 to convert m to mm)
        referenceAxisMapping : str | None
            Axis remapping for reference mesh: 'zyx' means ref Z->X, Y->Y, X->Z
            Use when reference has different orientation than generated.
            None means no remapping.
        '''
        if trimesh is None:
            raise ImportError(
                'trimesh is required for mesh comparison. '
                'Install with: pip install trimesh'
            )

        self._referencePath = Path(referencePath)
        self._generatedPath = Path(generatedPath)
        self._decimateFactor = decimateFactor
        self._forceDecimation = forceDecimation
        self._referenceScaleFactor = referenceScaleFactor
        self._referenceAxisMapping = referenceAxisMapping

        # Load meshes
        print(f'Loading reference mesh: {self._referencePath.name}')
        self._referenceMesh = trimesh.load(str(self._referencePath))

        print(f'Loading generated mesh: {self._generatedPath.name}')
        self._generatedMesh = trimesh.load(str(self._generatedPath))

        # Apply unit conversion and axis remapping to reference if needed
        needsTransform = (referenceScaleFactor is not None and referenceScaleFactor != 1.0) or referenceAxisMapping is not None
        if needsTransform:
            self._referenceMesh = self._transformReference(self._referenceMesh)

        # Apply decimation if needed
        self._preparedReference = self._prepareMesh(self._referenceMesh, 'reference')
        self._preparedGenerated = self._prepareMesh(self._generatedMesh, 'generated')

        # Initialize components
        self._aligner = MeshAligner()
        self._visualizer = DeviationVisualizer()
        self._reportGenerator = ComparisonReportGenerator()

    def _transformReference(
        self,
        mesh: 'trimesh.Trimesh',
    ) -> 'trimesh.Trimesh':
        '''
        Apply scale and axis remapping to reference mesh.

        Parameters:
        -----------
        mesh : trimesh.Trimesh
            Reference mesh

        Returns:
        --------
        trimesh.Trimesh : Transformed mesh
        '''
        transformed = mesh.copy()
        vertices = transformed.vertices.copy()

        # Apply axis remapping first
        if self._referenceAxisMapping is not None:
            axisMap = {'x': 0, 'y': 1, 'z': 2}
            mapping = self._referenceAxisMapping.lower()
            if len(mapping) == 3:
                newVertices = np.zeros_like(vertices)
                # mapping 'zyx' means: new_X = old_Z, new_Y = old_Y, new_Z = old_X
                for newAxis, oldAxisChar in enumerate(mapping):
                    oldAxis = axisMap[oldAxisChar]
                    newVertices[:, newAxis] = vertices[:, oldAxis]
                vertices = newVertices
                print(f'  Applied axis remapping: {self._referenceAxisMapping}')

        # Apply scale factor
        if self._referenceScaleFactor is not None and self._referenceScaleFactor != 1.0:
            vertices *= self._referenceScaleFactor
            print(f'  Applied scale factor: {self._referenceScaleFactor}')

        transformed.vertices = vertices
        return transformed

    def _prepareMesh(
        self,
        mesh: 'trimesh.Trimesh',
        name: str,
    ) -> 'trimesh.Trimesh':
        '''
        Prepare mesh for analysis (decimate if too large).

        Parameters:
        -----------
        mesh : trimesh.Trimesh
            Input mesh
        name : str
            Mesh name for logging

        Returns:
        --------
        trimesh.Trimesh : Prepared mesh (possibly decimated)
        '''
        nFaces = len(mesh.faces)
        threshold = 100000  # 100k faces

        if self._forceDecimation or nFaces > threshold:
            targetFaces = int(nFaces * self._decimateFactor)
            print(f'Decimating {name} mesh: {nFaces:,} -> {targetFaces:,} faces')

            # Use quadric decimation for quality with explicit face count
            decimated = mesh.simplify_quadric_decimation(face_count=targetFaces)
            print(f'  Result: {len(decimated.faces):,} faces')
            return decimated

        print(f'{name.title()} mesh: {nFaces:,} faces (no decimation needed)')
        return mesh.copy()

    def runAlignment(
        self,
        method: str = 'combined',
    ) -> AlignmentResult:
        '''
        Align generated mesh to reference coordinate system.

        Parameters:
        -----------
        method : str
            'pca' - PCA only (fast, less accurate)
            'icp' - ICP only (needs good initial guess)
            'combined' - PCA then ICP (recommended)

        Returns:
        --------
        AlignmentResult : Alignment result with transformed mesh
        '''
        print(f'Running {method} alignment...')

        if method == 'pca':
            result = self._aligner.alignPCA(
                self._preparedGenerated,
                self._preparedReference,
            )
        elif method == 'icp':
            result = self._aligner.alignICP(
                self._preparedGenerated,
                self._preparedReference,
            )
        elif method == 'combined':
            result = self._aligner.combinedAlignment(
                self._preparedGenerated,
                self._preparedReference,
            )
        else:
            raise ValueError(f'Unknown alignment method: {method}')

        print(f'  Alignment cost: {result.cost:.4f}')
        print(f'  Bounding box IoU: {result.boundingBoxIou:.3f}')

        return result

    def runDistanceAnalysis(
        self,
        alignedMesh: 'trimesh.Trimesh',
        nSamplePoints: int = 50000,
    ) -> tuple[DistanceStatistics, np.ndarray, np.ndarray, float]:
        '''
        Compute distance metrics between aligned and reference meshes.

        Parameters:
        -----------
        alignedMesh : trimesh.Trimesh
            Aligned generated mesh
        nSamplePoints : int
            Number of surface sample points

        Returns:
        --------
        tuple : (overallStats, samplePoints, distances, hausdorffMm)
        '''
        print(f'Computing distances ({nSamplePoints:,} sample points)...')

        analyzer = DistanceAnalyzer(self._preparedReference, alignedMesh)

        stats, samplePoints, distances = analyzer.computeBidirectionalDistance(nSamplePoints)
        hausdorff = analyzer.computeHausdorffDistance(nPoints=10000)

        print(f'  RMS deviation: {stats.rmsMm:.2f} mm')
        print(f'  Max deviation: {max(abs(stats.minMm), abs(stats.maxMm)):.2f} mm')
        print(f'  Hausdorff distance: {hausdorff:.2f} mm')

        return stats, samplePoints, distances, hausdorff

    def runRegionalAnalysis(
        self,
        samplePoints: np.ndarray,
        distances: np.ndarray,
    ) -> list[RegionalDeviation]:
        '''
        Compute per-region deviation statistics.

        Parameters:
        -----------
        samplePoints : np.ndarray
            Sample point coordinates
        distances : np.ndarray
            Signed distances at sample points

        Returns:
        --------
        list[RegionalDeviation] : Statistics per board region
        '''
        print('Computing regional deviations...')

        # Get board length from reference mesh
        refBounds = self._preparedReference.bounds
        boardLength = refBounds[1, 0] - refBounds[0, 0]

        analyzer = DistanceAnalyzer(self._preparedReference, self._preparedReference)
        regions = analyzer.computeRegionalDeviations(samplePoints, distances, boardLength)

        for region in regions:
            print(f'  {region.region}: RMS={region.stats.rmsMm:.2f}mm')

        return regions

    def runFinSegmentation(
        self,
        alignedMesh: 'trimesh.Trimesh',
    ) -> tuple[SegmentedMesh | None, SegmentedMesh | None, dict | None]:
        '''
        Segment and compare fins between reference and generated meshes.

        Parameters:
        -----------
        alignedMesh : trimesh.Trimesh
            Aligned generated mesh

        Returns:
        --------
        tuple : (refSegmented, genSegmented, comparisonStats)
        '''
        print('Segmenting fins...')

        try:
            refSegmenter = FinSegmenter(self._preparedReference)
            refSegmented = refSegmenter.segmentByHeight()
            print(f'  Reference: {len(refSegmented.finMeshes)} fins ({refSegmented.finConfiguration})')
        except Exception as e:
            print(f'  Reference fin segmentation failed: {e}')
            refSegmented = None

        try:
            genSegmenter = FinSegmenter(alignedMesh)
            genSegmented = genSegmenter.segmentByHeight()
            print(f'  Generated: {len(genSegmented.finMeshes)} fins ({genSegmented.finConfiguration})')
        except Exception as e:
            print(f'  Generated fin segmentation failed: {e}')
            genSegmented = None

        # Comparison stats
        comparisonStats = None
        if refSegmented and genSegmented:
            comparisonStats = {
                'referenceFinCount': len(refSegmented.finMeshes),
                'generatedFinCount': len(genSegmented.finMeshes),
                'referenceConfiguration': refSegmented.finConfiguration,
                'generatedConfiguration': genSegmented.finConfiguration,
                'configurationMatch': (
                    refSegmented.finConfiguration == genSegmented.finConfiguration
                ),
            }

        return refSegmented, genSegmented, comparisonStats

    def runFullComparison(
        self,
        alignMethod: str = 'combined',
        segmentFins: bool = True,
        nSamplePoints: int = 50000,
    ) -> ComparisonResult:
        '''
        Run complete comparison pipeline.

        Parameters:
        -----------
        alignMethod : str
            Alignment method ('pca', 'icp', 'combined')
        segmentFins : bool
            Whether to segment and compare fins
        nSamplePoints : int
            Number of surface sample points

        Returns:
        --------
        ComparisonResult : Complete analysis results
        '''
        print('\n' + '='*60)
        print('MESH COMPARISON ANALYSIS')
        print('='*60)
        print(f'Reference: {self._referencePath.name}')
        print(f'Generated: {self._generatedPath.name}')
        print('='*60 + '\n')

        # 1. Alignment
        alignmentResult = self.runAlignment(alignMethod)
        alignedMesh = alignmentResult.alignedMesh

        # 2. Distance analysis
        overallStats, samplePoints, distances, hausdorff = self.runDistanceAnalysis(
            alignedMesh, nSamplePoints
        )

        # 3. Regional analysis
        regionalDeviations = self.runRegionalAnalysis(samplePoints, distances)

        # 4. Fin segmentation (optional)
        refSegmented = None
        genSegmented = None
        finComparisonStats = None
        if segmentFins:
            refSegmented, genSegmented, finComparisonStats = self.runFinSegmentation(alignedMesh)

        # 5. Build report data
        print('\nBuilding report data...')
        reportData = self._reportGenerator.buildReportData(
            referenceMesh=self._preparedReference,
            generatedMesh=alignedMesh,
            alignmentResult=alignmentResult,
            overallStats=overallStats,
            regionalDeviations=regionalDeviations,
            hausdorffMm=hausdorff,
            finComparison=finComparisonStats,
        )

        print('\n' + '='*60)
        print('COMPARISON COMPLETE')
        print('='*60)
        print(f'Overall RMS: {overallStats.rmsMm:.2f} mm')
        status = 'PASS' if overallStats.rmsMm < 5.0 else 'NEEDS IMPROVEMENT'
        print(f'Status: {status}')
        print('='*60 + '\n')

        return ComparisonResult(
            referenceMesh=self._referenceMesh,
            generatedMesh=self._generatedMesh,
            alignedMesh=alignedMesh,
            alignmentResult=alignmentResult,
            samplePoints=samplePoints,
            signedDistances=distances,
            overallStats=overallStats,
            regionalDeviations=regionalDeviations,
            hausdorffMm=hausdorff,
            referenceSegmented=refSegmented,
            generatedSegmented=genSegmented,
            finComparisonStats=finComparisonStats,
            reportData=reportData,
        )

    def generateReport(
        self,
        results: ComparisonResult,
        outputPath: str | Path = 'validation_reports',
        baseName: str | None = None,
    ) -> dict[str, Path]:
        '''
        Generate and save comparison report.

        Parameters:
        -----------
        results : ComparisonResult
            Comparison results
        outputPath : str | Path
            Output directory
        baseName : str | None
            Base filename (auto-generated if None)

        Returns:
        --------
        dict[str, Path] : Paths to generated report files
        '''
        if results.reportData is None:
            raise ValueError('No report data available in results')

        if baseName is None:
            refName = self._referencePath.stem
            genName = self._generatedPath.stem
            baseName = f'comparison_{refName}_vs_{genName}'

        paths = self._reportGenerator.saveReport(
            results.reportData,
            outputPath,
            baseName,
        )

        print(f'Reports saved:')
        for fmt, path in paths.items():
            print(f'  {fmt}: {path}')

        return paths

    def createVisualization(
        self,
        results: ComparisonResult,
        vizType: str = 'dashboard',
    ):
        '''
        Create visualization from comparison results.

        Parameters:
        -----------
        results : ComparisonResult
            Comparison results
        vizType : str
            Type of visualization:
            - 'dashboard': Combined multi-panel view
            - 'heatmap': 3D deviation heatmap
            - 'outline': Top-view outline overlay
            - 'rocker': Side-view rocker overlay
            - 'histogram': Distance distribution
            - 'regional': Regional bar chart

        Returns:
        --------
        go.Figure : Plotly figure
        '''
        if vizType == 'dashboard':
            return self._visualizer.createComparisonDashboard(
                referenceMesh=self._preparedReference,
                generatedMesh=results.alignedMesh,
                samplePoints=results.samplePoints,
                distances=results.signedDistances,
                stats=results.overallStats,
                regionalDeviations=results.regionalDeviations,
            )
        elif vizType == 'heatmap':
            # Need to compute per-vertex distances for heatmap
            analyzer = DistanceAnalyzer(self._preparedReference, results.alignedMesh)
            vertexDistances = analyzer.computeSignedDistances(results.alignedMesh.vertices)
            return self._visualizer.createDeviationHeatmap(
                results.alignedMesh,
                vertexDistances,
            )
        elif vizType == 'outline':
            return self._visualizer.createOutlineOverlay(
                self._preparedReference,
                results.alignedMesh,
            )
        elif vizType == 'rocker':
            return self._visualizer.createRockerOverlay(
                self._preparedReference,
                results.alignedMesh,
            )
        elif vizType == 'histogram':
            return self._visualizer.createDeviationHistogram(
                results.signedDistances,
                results.overallStats,
            )
        elif vizType == 'regional':
            return self._visualizer.createRegionalBarChart(results.regionalDeviations)
        else:
            raise ValueError(f'Unknown visualization type: {vizType}')
