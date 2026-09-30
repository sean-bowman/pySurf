
# -- Viewer Data Exporter -- #

'''

Exports surfboard analysis results as JSON for the Three.js interactive viewer.

Consumes the structured results dictionary from PhysicsAnalyzer and the
parametric surface data from meshSampler to produce a single JSON file
containing everything the viewer needs: board parameters, sampled surface
geometry, physics results, force vectors, and waterline position.

Sean Bowman [02/04/2026]

'''

from __future__ import annotations

import base64
import json
import math
import os
from datetime import datetime

import numpy as np

from pysurf.Surfboard.SurfPhysics.geometry.board import BoardGeometry
from pysurf.Surfboard.SurfPhysics.geometry.parameters import SurfboardParameters
from pysurf.Surfboard.SurfPhysics.export.meshSampler import sampleParametricSurface
from pysurf.Surfboard.SurfPhysics.units import mToMm

class ViewerExporter:

    '''

    Exports analysis results as JSON for the Three.js 3D viewer.

    Produces a single JSON file per board configuration containing
    parametric surface geometry, physics data, and force vector
    information for rendering and overlay display.

    '''

    def exportForViewer(
        self,
        results: dict,
        stlDir: str = 'src/pysurf/Surfboard/SurfPhysics/output',
        outputDir: str = 'src/pysurf/Surfboard/SurfViewer/data',
        embedStl: bool = False,
        nLongitudinal: int = 100,
        nLateral: int = 25,
    ) -> str:

        '''

        Export analysis results as JSON for the Three.js viewer.

        Parameters:
        -----------
        results : dict
            Structured results from PhysicsAnalyzer.results.
            Must contain keys: params, waveConditions, riderMass,
            board, buoyancy, waves, performance
        stlDir : str
            Directory containing exported STL files
        outputDir : str
            Output directory for the JSON file
        embedStl : bool
            If True, base64-encode the STL file into the JSON
        nLongitudinal : int
            Longitudinal sampling resolution for parametric surface
        nLateral : int
            Lateral sampling resolution for parametric surface

        Returns:
        --------
        str : Path to the exported JSON file

        '''
        params: SurfboardParameters = results['params']
        board = BoardGeometry(params)

        # Determine board type and STL filename
        boardType = self._inferBoardType(params)
        stlFilename = self._resolveStlFilename(boardType, params, stlDir)

        #--------------------------------------------------------------------#
        # -- Parametric Surface Sampling -- #
        #--------------------------------------------------------------------#
        surfaceData = sampleParametricSurface(
            board, params, nLongitudinal, nLateral
        )

        #--------------------------------------------------------------------#
        # -- Build Export Data -- #
        #--------------------------------------------------------------------#
        exportData = {
            'meta': {
                'boardType': boardType,
                'finConfiguration': params.defaultFinConfiguration,
                'exportTimestamp': datetime.now().isoformat(),
                'unitSystem': 'mm',
            },
            'parameters': self._buildParametersSection(params, results),
            'parametricSurface': surfaceData,
            'stlFile': stlFilename,
            'physics': self._buildPhysicsSection(results, board, params),
        }

        # Optionally embed STL as base64
        if embedStl and stlFilename:
            stlPath = os.path.join(stlDir, stlFilename)
            if os.path.isfile(stlPath):
                exportData['stlEmbedded'] = True
                exportData['stlBase64'] = self._encodeStlBase64(stlPath)
            else:
                exportData['stlEmbedded'] = False

        #--------------------------------------------------------------------#
        # -- Write JSON -- #
        #--------------------------------------------------------------------#
        os.makedirs(outputDir, exist_ok=True)
        outputFilename = f'boardData_{boardType}.json'
        outputPath = os.path.join(outputDir, outputFilename)

        with open(outputPath, 'w') as f:
            json.dump(exportData, f, indent=2, default=self._jsonSerializer)

        print(f'  Viewer data exported: {outputPath}')

        return outputPath

    #--------------------------------------------------------------------#
    # -- Private Helpers -- #
    #--------------------------------------------------------------------#

    def _inferBoardType(self, params: SurfboardParameters) -> str:

        '''

        Infer the board type name from parameters.

        Compares against known presets to determine the type.
        Falls back to "custom" if no preset matches.

        Parameters:
        -----------
        params : SurfboardParameters
            Board parameters

        Returns:
        --------
        str : Board type identifier

        '''
        presetLengths = {
            'shortboard': 1828.0,
            'longboard': 2743.0,
            'fish': 1676.0,
        }
        for name, length in presetLengths.items():
            if abs(params.length - length) < 1.0:
                return name
        return 'custom'

    def _resolveStlFilename(
        self,
        boardType: str,
        params: SurfboardParameters,
        stlDir: str,
    ) -> str | None:

        '''

        Find the matching STL file for the given board configuration.

        Checks for fin-configuration-specific files first, then
        board-type-specific files, then falls back to generic.

        Parameters:
        -----------
        boardType : str
            Board type identifier
        params : SurfboardParameters
            Board parameters
        stlDir : str
            Directory containing STL files

        Returns:
        --------
        str | None : STL filename if found, None otherwise

        '''
        # Try fin-configuration-specific name first
        candidates = [
            f'surfboard_{params.defaultFinConfiguration}.stl',
            f'surfboard_{boardType}.stl',
            'surfboard.stl',
        ]

        # Special case: fish board with RNF naming
        if boardType == 'fish':
            candidates.insert(0, 'surfboard_RNF.stl')

        for candidate in candidates:
            if os.path.isfile(os.path.join(stlDir, candidate)):
                return candidate

        return None

    def _buildParametersSection(
        self,
        params: SurfboardParameters,
        results: dict,
    ) -> dict:

        '''

        Build the parameters section of the export JSON.

        Parameters:
        -----------
        params : SurfboardParameters
            Board parameters
        results : dict
            Analysis results

        Returns:
        --------
        dict : Parameters section

        '''
        boardData = results.get('board', {})

        return {
            'lengthMm': params.length,
            'maxWidthMm': params.maxWidth,
            'maxThicknessMm': params.maxThickness,
            'noseRockerMm': params.noseRocker,
            'tailRockerMm': params.tailRocker,
            'deckCrownMm': params.deckCrown,
            'bottomConcaveMm': params.bottomConcave,
            'tailShape': params.tailShape,
            'finConfiguration': params.defaultFinConfiguration,
            'foamType': params.foamType,
            'volumeLiters': round(boardData.get('volumeL', params.approxVolumeLiters), 2),
            'planformAreaCm2': round(boardData.get('planformAreaCm2', 0.0), 1),
            'wettedAreaCm2': round(boardData.get('wettedAreaCm2', 0.0), 1),
        }

    def _buildPhysicsSection(
        self,
        results: dict,
        board: BoardGeometry,
        params: SurfboardParameters,
    ) -> dict:

        '''

        Build the physics section of the export JSON.

        Includes buoyancy data, waterline position, wave conditions,
        performance curves, and pre-computed force vectors for the
        physics overlay.

        Parameters:
        -----------
        results : dict
            Analysis results from PhysicsAnalyzer
        board : BoardGeometry
            Board geometry for position calculations
        params : SurfboardParameters
            Board parameters

        Returns:
        --------
        dict : Physics section

        '''
        buoyancyData = results.get('buoyancy', {})
        wavesData = results.get('waves', {})
        perfData = results.get('performance', {})
        riderMass = results.get('riderMass', 75.0)
        boardMass = results.get('board', {}).get('massKg', 3.0)

        # Waterline position in mm (draft is in cm in results)
        riderDraftCm = buoyancyData.get('riderDraftCm', 5.0)
        waterplaneZMm = riderDraftCm * 10.0  # cm to mm

        # Build force vectors for physics overlay
        forceVectors = self._computeForceVectors(
            results, board, params, waterplaneZMm
        )

        # Convert numpy arrays to lists for JSON serialization
        speeds = perfData.get('speeds', [])
        if isinstance(speeds, np.ndarray):
            speeds = speeds.tolist()

        physicsSection = {
            'riderMassKg': riderMass,
            'boardMassKg': round(boardMass, 3),
            'buoyancy': {
                'maxBuoyancyN': round(buoyancyData.get('maxBuoyancyN', 0.0), 1),
                'totalWeightN': round(buoyancyData.get('totalWeightN', 0.0), 1),
                'boardDraftCm': round(buoyancyData.get('boardDraftCm', 0.0), 2),
                'riderDraftCm': round(riderDraftCm, 2),
                'buoyancyRatio': round(buoyancyData.get('buoyancyRatio', 0.0), 3),
                'floats': buoyancyData.get('floats', True),
            },
            'waterline': {
                'waterplaneZMm': round(waterplaneZMm, 2),
                'draftM': round(riderDraftCm / 100.0, 4),
                'trimAngleDeg': 0.0,
            },
            'waves': {
                'wavelengthM': round(wavesData.get('wavelengthM', 0.0), 2),
                'phaseSpeedMs': round(wavesData.get('phaseSpeedMs', 0.0), 2),
                'surfableSpeedMs': round(wavesData.get('surfableSpeedMs', 0.0), 2),
                'depthClass': wavesData.get('depthClass', 'unknown'),
                'isBroken': wavesData.get('isBroken', False),
            },
            'performance': {
                'speeds': [round(v, 3) for v in speeds],
                'trimAnglesDeg': self._arrayToRoundedList(perfData.get('trimAnglesDeg', [])),
                'liftN': self._arrayToRoundedList(perfData.get('liftN', [])),
                'dragN': self._arrayToRoundedList(perfData.get('dragN', [])),
                'liftToDrag': self._arrayToRoundedList(perfData.get('liftToDrag', [])),
                'planingThresholdMs': round(perfData.get('planingThresholdMs', 0.0), 2),
            },
            'forceVectors': forceVectors,
        }

        return physicsSection

    def _computeForceVectors(
        self,
        results: dict,
        board: BoardGeometry,
        params: SurfboardParameters,
        waterplaneZMm: float,
    ) -> dict:

        '''

        Compute force vector origins and directions for the physics overlay.

        Force vectors are positioned relative to the board coordinate system
        and represent the forces at paddling equilibrium (no forward speed).

        Parameters:
        -----------
        results : dict
            Analysis results
        board : BoardGeometry
            Board geometry
        params : SurfboardParameters
            Board parameters
        waterplaneZMm : float
            Waterplane Z height in mm

        Returns:
        --------
        dict : Force vector definitions with origin, direction, magnitude

        '''
        buoyancyData = results.get('buoyancy', {})
        totalWeightN = buoyancyData.get('totalWeightN', 0.0)
        maxBuoyancyN = buoyancyData.get('maxBuoyancyN', 0.0)

        # Center of mass approximation: 50% length, centerline, on deck
        centerX = params.length * 0.5
        deckHeightAtCenter = mToMm(board.getDeckHeightM(0.5, 0.0))
        rockerAtCenter = mToMm(board.getRockerHeightM(0.5))
        comZ = rockerAtCenter + deckHeightAtCenter

        # Center of buoyancy: slightly aft of center (40% of board = thickest)
        cobX = params.length * 0.42
        rockerAtCob = mToMm(board.getRockerHeightM(0.42))
        cobZ = rockerAtCob

        # Drag application point: forward of center, at waterline
        dragX = params.length * 0.35

        # Lift application point: at the planing section (center to tail)
        liftX = params.length * 0.55

        # Get representative drag/lift at surfing speed (~4 m/s)
        perfData = results.get('performance', {})
        speeds = perfData.get('speeds', [])
        dragValues = perfData.get('dragN', [])
        liftValues = perfData.get('liftN', [])

        # Find values at ~4 m/s (typical surfing speed)
        refDragN = 0.0
        refLiftN = 0.0
        if len(speeds) > 0 and len(dragValues) > 0:
            speedArr = np.array(speeds) if not isinstance(speeds, np.ndarray) else speeds
            idx = int(np.argmin(np.abs(speedArr - 4.0)))
            if idx < len(dragValues):
                refDragN = float(dragValues[idx]) if isinstance(dragValues[idx], (int, float, np.floating)) else 0.0
            if idx < len(liftValues):
                refLiftN = float(liftValues[idx]) if isinstance(liftValues[idx], (int, float, np.floating)) else 0.0

        return {
            'weight': {
                'origin': [round(centerX, 1), 0.0, round(comZ, 1)],
                'direction': [0.0, 0.0, -1.0],
                'magnitudeN': round(totalWeightN, 1),
                'label': 'Weight',
            },
            'buoyancy': {
                'origin': [round(cobX, 1), 0.0, round(cobZ, 1)],
                'direction': [0.0, 0.0, 1.0],
                'magnitudeN': round(maxBuoyancyN, 1),
                'label': 'Buoyancy',
            },
            'drag': {
                'origin': [round(dragX, 1), 0.0, round(waterplaneZMm * 0.5, 1)],
                'direction': [-1.0, 0.0, 0.0],
                'magnitudeN': round(refDragN, 1),
                'label': 'Drag',
            },
            'lift': {
                'origin': [round(liftX, 1), 0.0, round(cobZ, 1)],
                'direction': [0.0, 0.0, 1.0],
                'magnitudeN': round(refLiftN, 1),
                'label': 'Planing Lift',
            },
        }

    def _encodeStlBase64(self, stlPath: str) -> str:

        '''

        Read an STL file and return its contents as a base64 string.

        Parameters:
        -----------
        stlPath : str
            Path to the STL file

        Returns:
        --------
        str : Base64-encoded STL data

        '''
        with open(stlPath, 'rb') as f:
            rawBytes = f.read()
        return base64.b64encode(rawBytes).decode('ascii')

    @staticmethod
    def _arrayToRoundedList(arr, decimals: int = 2) -> list:

        '''

        Convert a numpy array or list to a rounded Python list.

        Parameters:
        -----------
        arr : array-like
            Input array or list
        decimals : int
            Number of decimal places

        Returns:
        --------
        list : Rounded values

        '''
        if isinstance(arr, np.ndarray):
            return [round(float(v), decimals) for v in arr]
        elif isinstance(arr, list):
            return [round(float(v), decimals) for v in arr]
        return []

    @staticmethod
    def _jsonSerializer(obj):

        '''

        Custom JSON serializer for numpy types and other non-serializable objects.

        Parameters:
        -----------
        obj : any
            Object to serialize

        Returns:
        --------
        JSON-serializable type

        Raises:
        -------
        TypeError : If the object type is not handled

        '''
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f'Object of type {type(obj)} is not JSON serializable')
