# -- Physics Analyzer -- #

'''

Top-level physics analysis class for surfboard hydrodynamics.

Encapsulates the full SurfPhysics pipeline: board geometry construction,
buoyancy analysis, wave physics, and hydrodynamic performance computation.
Reads configuration from JSON files and stores structured results.

Sean Bowman [02/04/2026]

'''

from __future__ import annotations

import json
import math

import numpy as np

from pysurf.Surfboard.SurfPhysics import constants as const
from pysurf.Surfboard.SurfPhysics.geometry.parameters import SurfboardParameters
from pysurf.Surfboard.SurfPhysics.geometry.board import BoardGeometry
from pysurf.Surfboard.SurfPhysics.waves.waveConditions import WaveConditions
from pysurf.Surfboard.SurfPhysics.waves.linearWaveTheory import LinearWaveTheory
from pysurf.Surfboard.SurfPhysics.hydrodynamics.buoyancy import BuoyancyModel
from pysurf.Surfboard.SurfPhysics.hydrodynamics.forceBalance import ForceBalance

class PhysicsAnalyzer:

    '''

    Surfboard hydrodynamic physics analysis.

    Reads a JSON configuration file, constructs board geometry,
    and runs the full analysis pipeline (buoyancy, wave physics,
    hydrodynamic performance). Results are stored as a structured
    dictionary accessible via the results property.

    '''

    def __init__(self) -> None:
        self._results: dict | None = None
        self._params: SurfboardParameters | None = None
        self._waveConditions: WaveConditions | None = None

    @property
    def results(self) -> dict:

        '''

        Structured analysis results.

        Keys:
        -----
        'params'         : SurfboardParameters instance
        'waveConditions' : WaveConditions instance
        'riderMass'      : float (kg)
        'board'          : dict of board geometry values
        'buoyancy'       : dict of buoyancy analysis values
        'waves'          : dict of wave physics values
        'performance'    : dict of hydrodynamic performance curves

        Raises:
        -------
        RuntimeError : If runAnalysis() has not been called yet

        '''
        if self._results is None:
            raise RuntimeError('No results available. Call runAnalysis() first.')
        return self._results

    @property
    def params(self) -> SurfboardParameters | None:

        '''Board parameters from the last analysis run.'''
        return self._params

    @property
    def waveConditions(self) -> WaveConditions | None:

        '''Wave conditions from the last analysis run.'''
        return self._waveConditions

    def runAnalysis(self, configPath: str) -> dict:

        '''

        Run the full physics analysis pipeline from a JSON config.

        Reads "board", "waves", "rider", and "analysis" sections from
        the config file. Prints a formatted console summary and stores
        the results.

        Parameters:
        -----------
        configPath : str
            Path to the JSON configuration file

        Returns:
        --------
        dict : Structured analysis results (also stored in self.results)

        '''
        config = self._loadConfig(configPath)
        self._params = self._buildBoardParams(config)
        self._waveConditions = WaveConditions(
            height=config['waveHeight'],
            period=config['wavePeriod'],
            depth=config['waterDepth'],
        )

        riderMass = config['riderMass']
        speedRange = config['speedRange']
        nSpeedPoints = config['nSpeedPoints']

        board = BoardGeometry(self._params)

        #--------------------------------------------------------------------#
        # -- Board Geometry -- #
        #--------------------------------------------------------------------#
        volume = board.computeVolume()
        volumeL = volume * 1e3
        planformArea = board.computePlanformArea()
        wettedArea = board.computeWettedSurfaceArea()

        #--------------------------------------------------------------------#
        # -- Buoyancy -- #
        #--------------------------------------------------------------------#
        buoyancy = BuoyancyModel(board, self._params)
        boardMass = buoyancy.estimateBoardMass()
        totalMass = boardMass + riderMass
        totalWeight = totalMass * const.gravity

        boardDraft = buoyancy.findEquilibriumDraft(boardMass)
        riderDraft = buoyancy.findEquilibriumDraft(totalMass)
        maxBuoyancy = buoyancy.buoyancyForce(volume)

        #--------------------------------------------------------------------#
        # -- Wave Physics -- #
        #--------------------------------------------------------------------#
        waveModel = LinearWaveTheory()
        wavelength = waveModel.waveLength(self._waveConditions)
        phaseSpeed = waveModel.waveSpeed(self._waveConditions)
        groupSpeed = waveModel.groupSpeed(self._waveConditions)
        energy = waveModel.energyDensity(self._waveConditions)
        power = waveModel.energyFlux(self._waveConditions)
        depthClass = waveModel.depthClassification(self._waveConditions)
        isBroken = waveModel.isBroken(self._waveConditions)
        surfSpeed = waveModel.surfableWaveSpeed(self._waveConditions)

        #--------------------------------------------------------------------#
        # -- Hydrodynamic Performance -- #
        #--------------------------------------------------------------------#
        forceBalance = ForceBalance(board, self._params, riderMass)

        testSpeeds = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
        equilibriumStates = [forceBalance.findEquilibrium(v) for v in testSpeeds]

        curves = forceBalance.performanceCurves(
            minSpeed=speedRange[0],
            maxSpeed=speedRange[1],
            nPoints=nSpeedPoints,
        )

        planingSpeed = math.sqrt(const.gravity * board.getLengthM())

        #--------------------------------------------------------------------#
        # -- Console Output -- #
        #--------------------------------------------------------------------#
        self._printSummary(
            params=self._params,
            volumeL=volumeL,
            planformArea=planformArea,
            wettedArea=wettedArea,
            boardMass=boardMass,
            riderMass=riderMass,
            totalMass=totalMass,
            totalWeight=totalWeight,
            maxBuoyancy=maxBuoyancy,
            boardDraft=boardDraft,
            riderDraft=riderDraft,
            waveConditions=self._waveConditions,
            wavelength=wavelength,
            phaseSpeed=phaseSpeed,
            groupSpeed=groupSpeed,
            energy=energy,
            power=power,
            depthClass=depthClass,
            isBroken=isBroken,
            surfSpeed=surfSpeed,
            testSpeeds=testSpeeds,
            equilibriumStates=equilibriumStates,
            planingSpeed=planingSpeed,
        )

        #--------------------------------------------------------------------#
        # -- Build Results -- #
        #--------------------------------------------------------------------#
        self._results = {
            'params': self._params,
            'waveConditions': self._waveConditions,
            'riderMass': riderMass,
            'board': {
                'volumeL': volumeL,
                'planformAreaCm2': planformArea * 1e4,
                'wettedAreaCm2': wettedArea * 1e4,
                'massKg': boardMass,
            },
            'buoyancy': {
                'maxBuoyancyN': maxBuoyancy,
                'totalWeightN': totalWeight,
                'boardDraftCm': boardDraft * 100,
                'riderDraftCm': riderDraft * 100,
                'buoyancyRatio': maxBuoyancy / totalWeight,
                'floats': maxBuoyancy > totalWeight,
            },
            'waves': {
                'wavelengthM': wavelength,
                'phaseSpeedMs': phaseSpeed,
                'groupSpeedMs': groupSpeed,
                'energyDensityJm2': energy,
                'energyFluxWm': power,
                'depthClass': depthClass,
                'isBroken': isBroken,
                'surfableSpeedMs': surfSpeed,
            },
            'performance': {
                'speeds': curves['speed'],
                'trimAnglesDeg': curves['trimAngleDeg'],
                'liftN': curves['liftN'],
                'dragN': curves['dragN'],
                'liftToDrag': curves['liftToDrag'],
                'wettedLengthM': curves['wettedLengthM'],
                'planingThresholdMs': planingSpeed,
            },
        }

        print()
        print('=' * 62)
        print('  Analysis complete.')
        print('=' * 62)

        return self._results

    #--------------------------------------------------------------------#
    # -- Private Helpers -- #
    #--------------------------------------------------------------------#

    def _loadConfig(self, configPath: str) -> dict:

        '''

        Load and flatten relevant config sections from JSON.

        Parameters:
        -----------
        configPath : str
            Path to the JSON configuration file

        Returns:
        --------
        dict : Flattened config values

        '''
        with open(configPath, 'r') as f:
            data = json.load(f)

        board = data.get('board', {})
        geometry = data.get('geometry', {})
        waves = data.get('waves', {})
        rider = data.get('rider', {})
        analysis = data.get('analysis', {})

        speedRange = analysis.get('speedRange', [0.5, 10.0])

        return {
            'boardType': board.get('type', 'shortboard'),
            'finConfiguration': board.get('finConfiguration', 'default'),
            'foamType': board.get('foamType', 'pu'),
            'customParametersFile': board.get('customParametersFile', None),
            'waveHeight': waves.get('height', 1.5),
            'wavePeriod': waves.get('period', 10.0),
            'waterDepth': waves.get('depth', 2.5),
            'riderMass': rider.get('mass', 75.0),
            'speedRange': tuple(speedRange),
            'nSpeedPoints': analysis.get('nSpeedPoints', 50),
        }

    def _buildBoardParams(self, config: dict) -> SurfboardParameters:

        '''

        Build SurfboardParameters from config values.

        Selects a preset board or loads custom parameters, then
        applies fin configuration and foam type overrides.

        Parameters:
        -----------
        config : dict
            Flattened config from _loadConfig()

        Returns:
        --------
        SurfboardParameters : Board parameters

        Raises:
        -------
        ValueError : If boardType is not recognized

        '''
        boardType = config['boardType']
        customFile = config.get('customParametersFile')

        if boardType == 'custom' and customFile:
            params = SurfboardParameters.fromJson(customFile)
        else:
            presets = {
                'shortboard': SurfboardParameters.shortboard,
                'longboard': SurfboardParameters.longboard,
                'fish': SurfboardParameters.fish,
            }
            factory = presets.get(boardType)
            if factory is None:
                raise ValueError(
                    f'Unknown board type: {boardType}. '
                    f'Valid options: {list(presets.keys())} or "custom"'
                )
            params = factory()

        # Apply overrides
        finConfig = config.get('finConfiguration', 'default')
        if finConfig != 'default':
            params.defaultFinConfiguration = finConfig

        params.foamType = config.get('foamType', 'pu')

        return params

    def _printSummary(
        self,
        *,
        params: SurfboardParameters,
        volumeL: float,
        planformArea: float,
        wettedArea: float,
        boardMass: float,
        riderMass: float,
        totalMass: float,
        totalWeight: float,
        maxBuoyancy: float,
        boardDraft: float,
        riderDraft: float,
        waveConditions: WaveConditions,
        wavelength: float,
        phaseSpeed: float,
        groupSpeed: float,
        energy: float,
        power: float,
        depthClass: str,
        isBroken: bool,
        surfSpeed: float,
        testSpeeds: list[float],
        equilibriumStates: list,
        planingSpeed: float,
    ) -> None:

        '''Print formatted analysis summary to console.'''

        print()
        print('=' * 62)
        print('  SURFBOARD ANALYSIS')
        print('=' * 62)
        print()

        params.printSummary()
        print()

        print(f'  Computed Volume:    {volumeL:8.2f} L  ({volumeL * 1000:.0f} cm^3)')
        print(f'  Planform Area:     {planformArea * 1e4:8.1f} cm^2')
        print(f'  Wetted Surface:    {wettedArea * 1e4:8.1f} cm^2')
        print()

        # Buoyancy
        print('-' * 62)
        print('  BUOYANCY ANALYSIS')
        print('-' * 62)
        print(f'  Board Mass:        {boardMass:8.2f} kg  ({params.foamType.upper()} foam)')
        print(f'  Rider Mass:        {riderMass:8.1f} kg')
        print(f'  Total Mass:        {totalMass:8.1f} kg')
        print(f'  Total Weight:      {totalWeight:8.1f} N')
        print()
        print(f'  Max Buoyancy:      {maxBuoyancy:8.1f} N  (fully submerged)')
        print(f'  Board-only Draft:  {boardDraft * 100:8.2f} cm')
        print(f'  With Rider Draft:  {riderDraft * 100:8.2f} cm')
        print(f'  Buoyancy Ratio:    {maxBuoyancy / totalWeight:8.2f}x  '
              f'({"floats" if maxBuoyancy > totalWeight else "SINKS"})')
        print()

        # Wave Physics
        print('-' * 62)
        print('  WAVE ANALYSIS')
        print('-' * 62)
        print(f'  Wave Height:       {waveConditions.height:8.2f} m')
        print(f'  Wave Period:       {waveConditions.period:8.1f} s')
        print(f'  Water Depth:       {waveConditions.depth:8.1f} m')
        print(f'  Wavelength:        {wavelength:8.1f} m')
        print(f'  Phase Speed:       {phaseSpeed:8.2f} m/s  ({phaseSpeed * 3.6:.1f} km/h)')
        print(f'  Group Speed:       {groupSpeed:8.2f} m/s')
        print(f'  Energy Density:    {energy:8.1f} J/m^2')
        print(f'  Energy Flux:       {power:8.1f} W/m')
        print(f'  Depth Class:       {depthClass:>8}  (d/L = {waveConditions.depth / wavelength:.3f})')
        print(f'  Breaking:          {"YES" if isBroken else "No":>8}')
        print(f'  Surfable Speed:    {surfSpeed:8.2f} m/s  ({surfSpeed * 3.6:.1f} km/h)')
        print()

        # Hydrodynamic Performance
        print('-' * 62)
        print('  HYDRODYNAMIC PERFORMANCE')
        print('-' * 62)
        print(f'  {"Speed":>6}  {"Trim":>6}  {"Lift":>8}  {"Drag":>8}  {"L/D":>6}  {"Planing":>8}')
        print(f'  {"(m/s)":>6}  {"(deg)":>6}  {"(N)":>8}  {"(N)":>8}  {"":>6}  {"":>8}')
        print('  ' + '-' * 52)

        for speed, state in zip(testSpeeds, equilibriumStates):
            print(
                f'  {speed:6.1f}  {state.trimAngleDeg:6.1f}  {state.liftForceN:8.1f}  '
                f'{state.dragForceN:8.1f}  {state.liftToDrag:6.1f}  '
                f'{"Yes" if state.isPlaning else "No":>8}'
            )

        print()
        print(f'  Planing Threshold: {planingSpeed:8.2f} m/s  (Fn = 1.0)')
        print()
