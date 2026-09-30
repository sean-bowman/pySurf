# -- Comparison Report Generator -- #

'''
Generates markdown and JSON reports for mesh comparison results.

Produces structured documentation of geometry deviations with
recommendations for improving generation parameters.

Sean Bowman [02/05/2026]
'''

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

from pysurf.Surfboard.SurfPhysics.validation.distanceMetrics import DistanceStatistics, RegionalDeviation
from pysurf.Surfboard.SurfPhysics.validation.meshAlignment import AlignmentResult
from pysurf.Surfboard.SurfPhysics.validation.finSegmentation import SegmentedMesh


#--------------------------------------------------------------------#
# -- Report Data Classes -- #
#--------------------------------------------------------------------#

@dataclass
class MeshProperties:
    '''Properties of a mesh for reporting.'''
    name: str
    nVertices: int
    nFaces: int
    isWatertight: bool
    volumeMm3: float | None
    surfaceAreaMm2: float
    boundingBoxMm: list[list[float]]  # [[minX, minY, minZ], [maxX, maxY, maxZ]]

    def toDict(self) -> dict:
        '''Convert to dictionary for JSON serialization.'''
        return asdict(self)


@dataclass
class ComparisonSummary:
    '''High-level summary of comparison results.'''
    overallRmsMm: float
    overallMaxDevMm: float
    hausdorffMm: float
    alignmentIou: float
    primaryIssueRegion: str
    primaryIssueDescription: str
    passesThreshold: bool  # True if RMS < 5mm

    def toDict(self) -> dict:
        '''Convert to dictionary for JSON serialization.'''
        return asdict(self)


@dataclass
class ComparisonReportData:
    '''Complete data structure for a comparison report.'''
    timestamp: str
    referenceMeshProps: MeshProperties
    generatedMeshProps: MeshProperties
    alignment: dict
    overallStats: dict
    regionalStats: list[dict]
    finComparison: dict | None
    summary: ComparisonSummary
    recommendations: list[str]

    def toDict(self) -> dict:
        '''Convert to dictionary for JSON serialization.'''
        return {
            'timestamp': self.timestamp,
            'referenceMesh': self.referenceMeshProps.toDict(),
            'generatedMesh': self.generatedMeshProps.toDict(),
            'alignment': self.alignment,
            'overallStats': self.overallStats,
            'regionalStats': self.regionalStats,
            'finComparison': self.finComparison,
            'summary': self.summary.toDict(),
            'recommendations': self.recommendations,
        }


#--------------------------------------------------------------------#
# -- Report Generator -- #
#--------------------------------------------------------------------#

class ComparisonReportGenerator:
    '''
    Generates structured reports from mesh comparison results.

    Supports markdown (human-readable) and JSON (programmatic) output.
    '''

    def __init__(self) -> None:
        '''Initialize the report generator.'''
        if trimesh is None:
            raise ImportError(
                'trimesh is required for report generation. '
                'Install with: pip install trimesh'
            )

    def extractMeshProperties(
        self,
        mesh: 'trimesh.Trimesh',
        name: str = 'Mesh',
    ) -> MeshProperties:
        '''
        Extract reportable properties from a mesh.

        Parameters:
        -----------
        mesh : trimesh.Trimesh
            Source mesh
        name : str
            Display name for the mesh

        Returns:
        --------
        MeshProperties : Extracted properties
        '''
        volume = None
        if mesh.is_watertight:
            try:
                volume = float(mesh.volume)
            except Exception:
                pass

        return MeshProperties(
            name=name,
            nVertices=len(mesh.vertices),
            nFaces=len(mesh.faces),
            isWatertight=mesh.is_watertight,
            volumeMm3=volume,
            surfaceAreaMm2=float(mesh.area),
            boundingBoxMm=mesh.bounds.tolist(),
        )

    def generateRecommendations(
        self,
        overallStats: DistanceStatistics,
        regionalDeviations: list[RegionalDeviation],
    ) -> list[str]:
        '''
        Generate improvement recommendations based on deviation patterns.

        Parameters:
        -----------
        overallStats : DistanceStatistics
            Overall deviation statistics
        regionalDeviations : list[RegionalDeviation]
            Per-region statistics

        Returns:
        --------
        list[str] : List of actionable recommendations
        '''
        recommendations = []

        # Overall assessment
        if overallStats.rmsMm > 10:
            recommendations.append(
                'High overall deviation detected. Consider reviewing the fundamental '
                'geometry parameters (length, width, thickness).'
            )
        elif overallStats.rmsMm > 5:
            recommendations.append(
                'Moderate deviation detected. Fine-tuning of profile curves may help.'
            )
        else:
            recommendations.append(
                'Overall deviation is within acceptable range (<5mm RMS).'
            )

        # Analyze bias
        biasRatio = overallStats.positiveCount / max(overallStats.negativeCount, 1)
        if biasRatio > 2:
            recommendations.append(
                f'Generated board is consistently LARGER than reference '
                f'(bias ratio: {biasRatio:.1f}). Consider reducing volume parameters.'
            )
        elif biasRatio < 0.5:
            recommendations.append(
                f'Generated board is consistently SMALLER than reference '
                f'(bias ratio: {biasRatio:.1f}). Consider increasing volume parameters.'
            )

        # Regional analysis
        regionRms = {rd.region: rd.stats.rmsMm for rd in regionalDeviations}

        # Check nose
        if 'nose' in regionRms and regionRms['nose'] > 8:
            recommendations.append(
                f'Nose region has high deviation ({regionRms["nose"]:.1f}mm RMS). '
                'Adjust nose width taper or nose rocker curve.'
            )

        # Check tail
        if 'tail' in regionRms and regionRms['tail'] > 8:
            recommendations.append(
                f'Tail region has high deviation ({regionRms["tail"]:.1f}mm RMS). '
                'Adjust tail width, tail rocker, or tail kick parameters.'
            )

        # Check rails (middle region lateral deviation)
        if 'middle' in regionRms and regionRms['middle'] > 6:
            recommendations.append(
                f'Middle section has notable deviation ({regionRms["middle"]:.1f}mm RMS). '
                'Check wide point position and cross-section rail profile.'
            )

        # Find worst region
        if regionalDeviations:
            worstRegion = max(regionalDeviations, key=lambda rd: rd.stats.rmsMm)
            if worstRegion.stats.rmsMm > 5:
                recommendations.append(
                    f'Focus improvement efforts on the {worstRegion.region} region '
                    f'(highest RMS: {worstRegion.stats.rmsMm:.1f}mm).'
                )

        return recommendations

    def buildReportData(
        self,
        referenceMesh: 'trimesh.Trimesh',
        generatedMesh: 'trimesh.Trimesh',
        alignmentResult: AlignmentResult,
        overallStats: DistanceStatistics,
        regionalDeviations: list[RegionalDeviation],
        hausdorffMm: float,
        finComparison: dict | None = None,
    ) -> ComparisonReportData:
        '''
        Build complete report data structure.

        Parameters:
        -----------
        referenceMesh : trimesh.Trimesh
            Reference mesh
        generatedMesh : trimesh.Trimesh
            Generated (aligned) mesh
        alignmentResult : AlignmentResult
            Alignment results
        overallStats : DistanceStatistics
            Overall deviation statistics
        regionalDeviations : list[RegionalDeviation]
            Per-region statistics
        hausdorffMm : float
            Hausdorff distance
        finComparison : dict | None
            Fin comparison results (if segmented)

        Returns:
        --------
        ComparisonReportData : Complete report data
        '''
        # Find primary issue
        if regionalDeviations:
            worstRegion = max(regionalDeviations, key=lambda rd: rd.stats.rmsMm)
            primaryRegion = worstRegion.region
            primaryDescription = (
                f'{worstRegion.region.title()} has highest RMS deviation '
                f'({worstRegion.stats.rmsMm:.1f}mm)'
            )
        else:
            primaryRegion = 'unknown'
            primaryDescription = 'No regional analysis available'

        summary = ComparisonSummary(
            overallRmsMm=overallStats.rmsMm,
            overallMaxDevMm=max(abs(overallStats.minMm), abs(overallStats.maxMm)),
            hausdorffMm=hausdorffMm,
            alignmentIou=alignmentResult.boundingBoxIou,
            primaryIssueRegion=primaryRegion,
            primaryIssueDescription=primaryDescription,
            passesThreshold=overallStats.rmsMm < 5.0,
        )

        recommendations = self.generateRecommendations(overallStats, regionalDeviations)

        return ComparisonReportData(
            timestamp=datetime.now().isoformat(),
            referenceMeshProps=self.extractMeshProperties(referenceMesh, 'Reference'),
            generatedMeshProps=self.extractMeshProperties(generatedMesh, 'Generated'),
            alignment=alignmentResult.toDict(),
            overallStats=overallStats.toDict(),
            regionalStats=[rd.toDict() for rd in regionalDeviations],
            finComparison=finComparison,
            summary=summary,  # Keep as dataclass object for easier access
            recommendations=recommendations,
        )

    def generateMarkdownReport(
        self,
        reportData: ComparisonReportData,
    ) -> str:
        '''
        Generate a human-readable markdown report.

        Parameters:
        -----------
        reportData : ComparisonReportData
            Complete report data

        Returns:
        --------
        str : Markdown-formatted report
        '''
        lines = []

        # Title
        lines.append('# Surfboard Geometry Comparison Report')
        lines.append('')
        lines.append(f'**Generated:** {reportData.timestamp}')
        lines.append('')

        # Executive Summary
        lines.append('## Executive Summary')
        lines.append('')
        summary = reportData.summary
        status = '✅ PASS' if summary.passesThreshold else '❌ NEEDS IMPROVEMENT'
        lines.append(f'**Status:** {status}')
        lines.append('')
        lines.append(f'| Metric | Value |')
        lines.append(f'|--------|-------|')
        lines.append(f'| Overall RMS | {summary.overallRmsMm:.2f} mm |')
        lines.append(f'| Maximum Deviation | {summary.overallMaxDevMm:.2f} mm |')
        lines.append(f'| Hausdorff Distance | {summary.hausdorffMm:.2f} mm |')
        lines.append(f'| Alignment IoU | {summary.alignmentIou:.3f} |')
        lines.append(f'| Primary Issue | {summary.primaryIssueDescription} |')
        lines.append('')

        # Mesh Properties
        lines.append('## Mesh Properties')
        lines.append('')
        lines.append('| Property | Reference | Generated |')
        lines.append('|----------|-----------|-----------|')
        refProps = reportData.referenceMeshProps
        genProps = reportData.generatedMeshProps
        lines.append(f'| Vertices | {refProps.nVertices:,} | {genProps.nVertices:,} |')
        lines.append(f'| Faces | {refProps.nFaces:,} | {genProps.nFaces:,} |')
        lines.append(f'| Watertight | {refProps.isWatertight} | {genProps.isWatertight} |')
        lines.append(f'| Surface Area | {refProps.surfaceAreaMm2:,.0f} mm² | {genProps.surfaceAreaMm2:,.0f} mm² |')
        if refProps.volumeMm3 and genProps.volumeMm3:
            lines.append(f'| Volume | {refProps.volumeMm3:,.0f} mm³ | {genProps.volumeMm3:,.0f} mm³ |')
        lines.append('')

        # Bounding box comparison
        refBounds = np.array(refProps.boundingBoxMm)
        genBounds = np.array(genProps.boundingBoxMm)
        refSize = refBounds[1] - refBounds[0]
        genSize = genBounds[1] - genBounds[0]
        lines.append('### Bounding Box Dimensions')
        lines.append('')
        lines.append('| Dimension | Reference | Generated | Difference |')
        lines.append('|-----------|-----------|-----------|------------|')
        for i, dim in enumerate(['Length (X)', 'Width (Y)', 'Height (Z)']):
            diff = genSize[i] - refSize[i]
            lines.append(f'| {dim} | {refSize[i]:.1f} mm | {genSize[i]:.1f} mm | {diff:+.1f} mm |')
        lines.append('')

        # Alignment Quality
        lines.append('## Alignment Quality')
        lines.append('')
        alignment = reportData.alignment
        lines.append(f'- **Method:** {alignment["method"]}')
        lines.append(f'- **Final Cost:** {alignment["cost"]:.4f}')
        lines.append(f'- **Bounding Box IoU:** {alignment["boundingBoxIou"]:.3f}')
        lines.append('')

        # Overall Statistics
        lines.append('## Deviation Statistics')
        lines.append('')
        stats = reportData.overallStats
        lines.append('### Overall')
        lines.append('')
        lines.append(f'| Statistic | Value |')
        lines.append(f'|-----------|-------|')
        lines.append(f'| Mean | {stats["meanMm"]:.2f} mm |')
        lines.append(f'| RMS | {stats["rmsMm"]:.2f} mm |')
        lines.append(f'| Std Dev | {stats["stdMm"]:.2f} mm |')
        lines.append(f'| Min | {stats["minMm"]:.2f} mm |')
        lines.append(f'| Max | {stats["maxMm"]:.2f} mm |')
        lines.append(f'| 95th Percentile | {stats["percentile95Mm"]:.2f} mm |')
        lines.append(f'| 99th Percentile | {stats["percentile99Mm"]:.2f} mm |')
        lines.append(f'| Points Outside | {stats["positiveCount"]:,} ({100*stats["positiveCount"]/stats["totalPoints"]:.1f}%) |')
        lines.append(f'| Points Inside | {stats["negativeCount"]:,} ({100*stats["negativeCount"]/stats["totalPoints"]:.1f}%) |')
        lines.append('')

        # Regional breakdown
        if reportData.regionalStats:
            lines.append('### By Region')
            lines.append('')
            lines.append('| Region | RMS (mm) | Mean (mm) | Max |Dev| (mm) |')
            lines.append('|--------|----------|-----------|----------------|')
            for region in reportData.regionalStats:
                rStats = region['stats']
                maxAbs = max(abs(rStats['minMm']), abs(rStats['maxMm']))
                lines.append(
                    f'| {region["region"].title()} | '
                    f'{rStats["rmsMm"]:.2f} | {rStats["meanMm"]:.2f} | {maxAbs:.2f} |'
                )
            lines.append('')

        # Fin Comparison
        if reportData.finComparison:
            lines.append('## Fin Comparison')
            lines.append('')
            finData = reportData.finComparison
            lines.append(f'- **Reference Fins:** {finData.get("referenceFinCount", "N/A")}')
            lines.append(f'- **Generated Fins:** {finData.get("generatedFinCount", "N/A")}')
            lines.append(f'- **Configuration Match:** {finData.get("configurationMatch", "N/A")}')
            lines.append('')

        # Recommendations
        lines.append('## Recommendations')
        lines.append('')
        for i, rec in enumerate(reportData.recommendations, 1):
            lines.append(f'{i}. {rec}')
        lines.append('')

        # Footer
        lines.append('---')
        lines.append('')
        lines.append('*Report generated by SurfPhysics validation module*')

        return '\n'.join(lines)

    def saveReport(
        self,
        reportData: ComparisonReportData,
        outputDir: str | Path,
        baseName: str = 'comparison_report',
    ) -> dict[str, Path]:
        '''
        Save report to markdown and JSON files.

        Parameters:
        -----------
        reportData : ComparisonReportData
            Complete report data
        outputDir : str | Path
            Output directory path
        baseName : str
            Base filename (without extension)

        Returns:
        --------
        dict[str, Path] : Paths to generated files
        '''
        outputDir = Path(outputDir)
        outputDir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Save markdown
        mdPath = outputDir / f'{baseName}.md'
        mdContent = self.generateMarkdownReport(reportData)
        mdPath.write_text(mdContent, encoding='utf-8')
        paths['markdown'] = mdPath

        # Save JSON
        jsonPath = outputDir / f'{baseName}.json'
        jsonContent = json.dumps(reportData.toDict(), indent=2)
        jsonPath.write_text(jsonContent, encoding='utf-8')
        paths['json'] = jsonPath

        return paths
