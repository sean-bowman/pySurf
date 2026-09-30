
# -- STL Export Wrapper -- #

'''

Convenience wrapper for exporting parametric surfboard geometry as STL files.

Wraps SurfaceMeshGenerator with higher-level methods for exporting presets,
custom parameter sets, and optionally running validation against a reference
STL in a single call.

Sean Bowman [02/07/2026]

'''

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pysurf.Surfboard.SurfPhysics.geometry.parameters import SurfboardParameters
from pysurf.Surfboard.SurfPhysics.geometry.surfaceMeshGenerator import (
    SurfaceMeshGenerator,
    RESOLUTION_PRESETS,
)


#--------------------------------------------------------------------#
# -- Preset Name to Factory Mapping -- #
#--------------------------------------------------------------------#

_PRESET_FACTORIES = {
    'shortboard': SurfboardParameters.shortboard,
    'longboard': SurfboardParameters.longboard,
    'fish': SurfboardParameters.fish,
}

class StlExporter:

    '''

    Exports parametric surfboard geometry as STL files.

    Wraps SurfaceMeshGenerator with convenience methods for quick
    export of presets, custom parameters, and validation workflows.

    Examples:
    ---------
    >>> exporter = StlExporter()
    >>> exporter.exportPreset('shortboard', 'output/')
    >>> exporter.exportBoard(customParams, 'output/custom.stl')

    '''

    def exportBoard(
        self,
        params: SurfboardParameters,
        outputPath: str,
        resolution: str = 'standard',
        binary: bool = True,
    ) -> str:

        '''

        Export a surfboard from parameters to an STL file.

        Parameters:
        -----------
        params : SurfboardParameters
            Board dimensions
        outputPath : str
            Output file path (should end in .stl)
        resolution : str
            Mesh resolution preset: 'draft', 'standard', or 'high'
        binary : bool
            If True, write binary STL. If False, write ASCII STL.

        Returns:
        --------
        str : Path to the exported STL file

        '''
        gen = SurfaceMeshGenerator.fromPreset(params, resolution)
        gen.exportStl(outputPath, binary=binary)

        volumeL = gen.computeVolumeLiters()
        print(f'Exported {Path(outputPath).name}: '
              f'{volumeL:.2f}L, '
              f'{len(gen.getMesh().vertices)} vertices, '
              f'{len(gen.getMesh().faces)} faces')

        return str(outputPath)

    def exportPreset(
        self,
        presetName: str,
        outputDir: str,
        resolution: str = 'standard',
        binary: bool = True,
    ) -> str:

        '''

        Export a named board preset to an STL file.

        Parameters:
        -----------
        presetName : str
            Preset name: 'shortboard', 'longboard', or 'fish'
        outputDir : str
            Output directory (file named '{presetName}_parametric.stl')
        resolution : str
            Mesh resolution preset
        binary : bool
            If True, write binary STL

        Returns:
        --------
        str : Path to the exported STL file

        '''
        if presetName not in _PRESET_FACTORIES:
            raise ValueError(
                f'Unknown preset \'{presetName}\'. '
                f'Available: {list(_PRESET_FACTORIES.keys())}'
            )

        params = _PRESET_FACTORIES[presetName]()
        outDir = Path(outputDir)
        outDir.mkdir(parents=True, exist_ok=True)
        outputPath = str(outDir / f'{presetName}_parametric.stl')

        return self.exportBoard(params, outputPath, resolution, binary)

    def exportWithValidation(
        self,
        params: SurfboardParameters,
        referencePath: str,
        outputDir: str,
        resolution: str = 'standard',
    ) -> dict:

        '''

        Export a board and compare it against a reference STL.

        Generates the parametric mesh, exports it, then runs the
        MeshComparisonAnalyzer to compute deviation metrics.

        Parameters:
        -----------
        params : SurfboardParameters
            Board dimensions
        referencePath : str
            Path to reference STL for comparison
        outputDir : str
            Output directory for the generated STL
        resolution : str
            Mesh resolution preset

        Returns:
        --------
        dict : Results containing:
            - 'outputPath': path to exported STL
            - 'volumeLiters': board volume
            - 'watertight': whether mesh is watertight
            - 'comparison': comparison metrics dict (if available)

        '''
        outDir = Path(outputDir)
        outDir.mkdir(parents=True, exist_ok=True)
        outputPath = str(outDir / 'generated_parametric.stl')

        # Generate and export
        gen = SurfaceMeshGenerator.fromPreset(params, resolution)
        gen.exportStl(outputPath)
        volumeL = gen.computeVolumeLiters()
        watertight = gen.getMesh().is_watertight

        result = {
            'outputPath': outputPath,
            'volumeLiters': round(volumeL, 2),
            'watertight': watertight,
            'comparison': None,
        }

        # Run comparison if reference exists
        refPath = Path(referencePath)
        if refPath.exists():
            try:
                from pysurf.Surfboard.SurfPhysics.validation.meshComparison import MeshComparisonAnalyzer

                analyzer = MeshComparisonAnalyzer(
                    referencePath=str(refPath),
                    generatedPath=outputPath,
                )
                comparisonResult = analyzer.runFullComparison()
                result['comparison'] = comparisonResult.toDict()
            except Exception as e:
                print(f'Validation comparison failed: {e}')

        return result

    def exportAllPresets(
        self,
        outputDir: str,
        resolution: str = 'standard',
    ) -> list[str]:

        '''

        Export all available board presets as STL files.

        Parameters:
        -----------
        outputDir : str
            Output directory
        resolution : str
            Mesh resolution preset

        Returns:
        --------
        list[str] : Paths to all exported STL files

        '''
        paths = []
        for presetName in _PRESET_FACTORIES:
            path = self.exportPreset(presetName, outputDir, resolution)
            paths.append(path)
        return paths
