# -- Water Simulation Runner -- #

'''
Command-line entry point for running SPH water simulations.

Loads scenario configuration, runs the WCSPH simulation,
displays progress, and optionally exports frame data for
Manim or Three.js visualization.

Usage:
    python -m WaterSim                                    # Default small 2D sloshing
    python -m WaterSim --preset standard                  # Standard quality 2D sloshing
    python -m WaterSim --config configs/sloshing_2d_default.json
    python -m WaterSim --no-export                        # Skip frame export

Sean Bowman [02/05/2026]
'''

from __future__ import annotations

import argparse
import json
import time as timeModule

import numpy as np

from pysurf.WaterSim import constants as const
from pysurf.WaterSim.sph.protocols import SimulationConfig, SimulationState
from pysurf.WaterSim.sph.wcsphSolver import WcsphSolver
from pysurf.WaterSim.sph.particles import ParticleSystem
from pysurf.WaterSim.sph.boundaryHandling import BoundaryHandler
from pysurf.WaterSim.scenarios.sloshingTank import SloshingTankConfig, createSloshingTank
from pysurf.WaterSim.scenarios.numericalWaveTank import WaveTankConfig, createNumericalWaveTank
from pysurf.WaterSim.export.frameExporter import FrameExporter


#--------------------------------------------------------------------#
# -- CLI Argument Parser -- #
#--------------------------------------------------------------------#

def buildParser() -> argparse.ArgumentParser:
    '''Build the CLI argument parser.'''
    parser = argparse.ArgumentParser(
        description='WaterSim -- SPH water simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to JSON configuration file',
    )
    parser.add_argument(
        '--scenario', type=str, default='sloshing',
        choices=['sloshing', 'waveTank'],
        help='Simulation scenario type (default: sloshing)',
    )
    parser.add_argument(
        '--preset', type=str, default='small',
        choices=['small', 'standard', 'small3D', 'standard3D'],
        help='Scenario preset (default: small)',
    )
    parser.add_argument(
        '--no-export', action='store_true',
        help='Skip frame data export',
    )
    parser.add_argument(
        '--output-dir', type=str, default='src/pysurf/WaterSim/output',
        help='Output directory for exported frames (default: WaterSim/output)',
    )

    return parser


#--------------------------------------------------------------------#
# -- Runner Class -- #
#--------------------------------------------------------------------#

class WaterSimRunner:
    '''
    Runs an SPH simulation and stores results.

    Handles the full pipeline: scenario setup, simulation loop
    with progress reporting, and optional frame export.
    '''

    def __init__(self) -> None:
        self._frames: list[SimulationState] = []
        self._exporter: FrameExporter = FrameExporter()

    def runFromConfig(self, configPath: str, exportDir: str = 'src/pysurf/WaterSim/output') -> dict:
        '''
        Run simulation from a JSON configuration file.

        Automatically detects scenario type from config.

        Parameters:
        -----------
        configPath : str
            Path to the JSON configuration file
        exportDir : str
            Output directory for frame export

        Returns:
        --------
        dict : Simulation results summary
        '''
        with open(configPath, 'r') as f:
            data = json.load(f)

        simSection = data.get('simulation', {})
        scenarioType = simSection.get('type', 'sloshingTank')

        if scenarioType == 'waveTank':
            return self._runWaveTankFromConfig(data, exportDir)
        else:
            return self._runSloshingFromConfig(data, exportDir)

    def _runSloshingFromConfig(self, data: dict, exportDir: str) -> dict:
        '''Run sloshing tank from parsed config data.'''
        tankSection = data.get('tank', {})
        simSection = data.get('simulation', {})
        sphSection = data.get('sph', {})

        tankConfig = SloshingTankConfig(
            tankWidth=tankSection.get('width', 0.5),
            tankHeight=tankSection.get('height', 0.3),
            fillRatio=tankSection.get('fillRatio', 0.6),
            initialTiltDeg=tankSection.get('initialTiltDeg', 5.0),
            particleSpacing=sphSection.get('particleSpacing', 0.005),
            smoothingLengthRatio=sphSection.get('smoothingLengthRatio', 1.3),
            endTime=simSection.get('endTime', 3.0),
            outputInterval=simSection.get('outputInterval', 0.02),
        )

        return self.runSloshing(tankConfig, exportDir=exportDir)

    def _runWaveTankFromConfig(self, data: dict, exportDir: str) -> dict:
        '''Run wave tank from parsed config data.'''
        tankSection = data.get('tank', {})
        waveSection = data.get('wave', {})
        simSection = data.get('simulation', {})
        sphSection = data.get('sph', {})

        tankConfig = WaveTankConfig(
            tankLength=tankSection.get('length', 5.0),
            tankWidth=tankSection.get('width', 0.5),
            stillWaterDepth=tankSection.get('stillWaterDepth', 0.5),
            beachSlope=tankSection.get('beachSlope', 0.1),
            beachStartRatio=tankSection.get('beachStartRatio', 0.6),
            waveHeight=waveSection.get('height', 0.1),
            wavePeriod=waveSection.get('period', 1.5),
            particleSpacing=sphSection.get('particleSpacing', 0.02),
            smoothingLengthRatio=sphSection.get('smoothingLengthRatio', 1.3),
            endTime=simSection.get('endTime', 10.0),
            outputInterval=simSection.get('outputInterval', 0.05),
            dimensions=simSection.get('dimensions', 3),
            rampCycles=waveSection.get('rampCycles', 3),
            useSecondOrderWaves=waveSection.get('useSecondOrder', True),
        )

        return self.runWaveTank(tankConfig, exportDir=exportDir)

    def runSloshing(
        self,
        tankConfig: SloshingTankConfig,
        doExport: bool = True,
        exportDir: str = 'src/pysurf/WaterSim/output',
    ) -> dict:
        '''
        Run a sloshing tank simulation.

        Parameters:
        -----------
        tankConfig : SloshingTankConfig
            Sloshing tank configuration
        doExport : bool
            Whether to export frame data
        exportDir : str
            Output directory for frame export

        Returns:
        --------
        dict : Simulation results summary
        '''
        print()
        print('=' * 62)
        print('  WATERSIM -- SPH SLOSHING SIMULATION')
        print('=' * 62)
        print()

        #--------------------------------------------------------------------#
        # Scenario Setup
        #--------------------------------------------------------------------#
        print('-' * 62)
        print('  SCENARIO SETUP')
        print('-' * 62)

        simConfig, particles, boundaryHandler = createSloshingTank(tankConfig)

        print(f'  Tank Width:        {tankConfig.tankWidth:8.3f} m')
        print(f'  Tank Height:       {tankConfig.tankHeight:8.3f} m')
        print(f'  Fill Ratio:        {tankConfig.fillRatio:8.2f}')
        print(f'  Initial Tilt:      {tankConfig.initialTiltDeg:8.1f} deg')
        print(f'  Particle Spacing:  {tankConfig.particleSpacing:8.4f} m')
        print(f'  Smoothing Length:  {simConfig.smoothingLength:8.4f} m')
        print(f'  Fluid Particles:   {particles.nFluid:8d}')
        print(f'  Boundary Particles:{particles.nBoundary:8d}')
        print(f'  Total Particles:   {particles.nParticles:8d}')
        print(f'  End Time:          {simConfig.endTime:8.2f} s')
        print()

        #--------------------------------------------------------------------#
        # Initialize Solver
        #--------------------------------------------------------------------#
        print('-' * 62)
        print('  INITIALIZING SOLVER')
        print('-' * 62)

        solver = WcsphSolver(
            config=simConfig,
            boundaryHandler=boundaryHandler,
        )
        solver.initialize(particles)

        print(f'  Speed of Sound:    {solver.speedOfSound:8.2f} m/s')
        print(f'  EOS constant B:    {solver._eosB:12.1f} Pa')
        print()

        # Record initial frame
        self._exporter.addFrame(solver.currentState, particles)

        #--------------------------------------------------------------------#
        # Simulation Loop
        #--------------------------------------------------------------------#
        print('-' * 62)
        print('  RUNNING SIMULATION')
        print('-' * 62)
        print()
        print(f'  {"Time":>8}  {"Step":>8}  {"dt":>10}  {"MaxVel":>8}  {"DensErr":>8}  {"Energy":>10}')
        print(f'  {"(s)":>8}  {"":>8}  {"(s)":>10}  {"(m/s)":>8}  {"(%)":>8}  {"(J)":>10}')
        print('  ' + '-' * 58)

        wallClockStart = timeModule.time()
        nextOutputTime = simConfig.outputInterval
        printInterval = max(0.1, simConfig.endTime / 20.0)
        nextPrintTime = printInterval

        while solver.time < simConfig.endTime:
            state = solver.step()

            # Export frame at output intervals
            if solver.time >= nextOutputTime:
                self._exporter.addFrame(state, particles)
                nextOutputTime += simConfig.outputInterval

            # Print progress at regular intervals
            if solver.time >= nextPrintTime or state.step % 500 == 0:
                print(
                    f'  {state.time:8.4f}  {state.step:8d}  {state.dt:10.2e}  '
                    f'{state.maxVelocity:8.4f}  {state.maxDensityError * 100:8.3f}  '
                    f'{state.totalEnergy:10.4f}'
                )
                nextPrintTime += printInterval

        wallClockEnd = timeModule.time()
        wallClockSeconds = wallClockEnd - wallClockStart

        # Final frame
        finalState = solver.currentState
        self._exporter.addFrame(finalState, particles)

        print()
        print(f'  Simulation complete.')
        print(f'  Total steps:       {finalState.step:8d}')
        print(f'  Wall-clock time:   {wallClockSeconds:8.1f} s')
        print(f'  Frames exported:   {self._exporter.nFrames:8d}')
        print()

        #--------------------------------------------------------------------#
        # Export
        #--------------------------------------------------------------------#
        exportPath = None
        if doExport:
            print('-' * 62)
            print('  EXPORTING FRAME DATA')
            print('-' * 62)

            exportPath = self._exporter.export(
                config=simConfig,
                outputDir=exportDir,
                scenarioName='sloshing',
            )
            print(f'  Exported to: {exportPath}')
            print()

        #--------------------------------------------------------------------#
        # Summary
        #--------------------------------------------------------------------#
        print('=' * 62)
        print('  SIMULATION SUMMARY')
        print('=' * 62)
        print(f'  Final KE:          {finalState.kineticEnergy:10.6f} J')
        print(f'  Final PE:          {finalState.potentialEnergy:10.6f} J')
        print(f'  Final Total E:     {finalState.totalEnergy:10.6f} J')
        print(f'  Max Density Error: {finalState.maxDensityError * 100:8.3f} %')
        print(f'  Max Velocity:      {finalState.maxVelocity:8.4f} m/s')
        print('=' * 62)
        print()

        return {
            'finalState': finalState,
            'wallClockSeconds': wallClockSeconds,
            'nFrames': self._exporter.nFrames,
            'exportPath': exportPath,
        }

    def runWaveTank(
        self,
        tankConfig: WaveTankConfig,
        doExport: bool = True,
        exportDir: str = 'src/pysurf/WaterSim/output',
    ) -> dict:
        '''
        Run a numerical wave tank simulation.

        Parameters:
        -----------
        tankConfig : WaveTankConfig
            Wave tank configuration
        doExport : bool
            Whether to export frame data
        exportDir : str
            Output directory for frame export

        Returns:
        --------
        dict : Simulation results summary
        '''
        print()
        print('=' * 62)
        print('  WATERSIM -- SPH WAVE TANK SIMULATION')
        print('=' * 62)
        print()

        #--------------------------------------------------------------------#
        # Scenario Setup
        #--------------------------------------------------------------------#
        print('-' * 62)
        print('  SCENARIO SETUP')
        print('-' * 62)

        result = createNumericalWaveTank(tankConfig)
        simConfig = result.config
        particles = result.particles
        boundaryHandler = result.boundaryHandler
        waveMaker = result.waveMaker
        breakingDetector = result.breakingDetector

        print(f'  Tank Length:       {tankConfig.tankLength:8.3f} m')
        print(f'  Tank Width:        {tankConfig.tankWidth:8.3f} m')
        print(f'  Still Water Depth: {tankConfig.stillWaterDepth:8.3f} m')
        print(f'  Beach Slope:       {tankConfig.beachSlope:8.3f}')
        print(f'  Wave Height:       {tankConfig.waveHeight:8.3f} m')
        print(f'  Wave Period:       {tankConfig.wavePeriod:8.2f} s')
        print(f'  Wavelength:        {waveMaker.wavelength:8.3f} m')
        print(f'  Phase Speed:       {waveMaker.phaseSpeed:8.3f} m/s')
        print(f'  Piston Stroke:     {waveMaker.stroke:8.4f} m')
        print(f'  Dimensions:        {tankConfig.dimensions:8d}D')
        print(f'  Particle Spacing:  {tankConfig.particleSpacing:8.4f} m')
        print(f'  Smoothing Length:  {simConfig.smoothingLength:8.4f} m')
        print(f'  Fluid Particles:   {particles.nFluid:8d}')
        print(f'  Boundary Particles:{particles.nBoundary:8d}')
        print(f'  Total Particles:   {particles.nParticles:8d}')
        print(f'  End Time:          {simConfig.endTime:8.2f} s')
        print()

        #--------------------------------------------------------------------#
        # Initialize Solver
        #--------------------------------------------------------------------#
        print('-' * 62)
        print('  INITIALIZING SOLVER')
        print('-' * 62)

        solver = WcsphSolver(
            config=simConfig,
            boundaryHandler=boundaryHandler,
        )
        solver.initialize(particles)

        print(f'  Speed of Sound:    {solver.speedOfSound:8.2f} m/s')
        print(f'  EOS constant B:    {solver._eosB:12.1f} Pa')
        print()

        # Record initial frame
        self._exporter.addFrame(solver.currentState, particles)

        #--------------------------------------------------------------------#
        # Simulation Loop with Wave-Maker
        #--------------------------------------------------------------------#
        print('-' * 62)
        print('  RUNNING SIMULATION')
        print('-' * 62)
        print()
        print(f'  {"Time":>8}  {"Step":>8}  {"dt":>10}  {"MaxVel":>8}  {"DensErr":>8}  {"Breaking":>8}')
        print(f'  {"(s)":>8}  {"":>8}  {"(s)":>10}  {"(m/s)":>8}  {"(%)":>8}  {"events":>8}')
        print('  ' + '-' * 58)

        wallClockStart = timeModule.time()
        nextOutputTime = simConfig.outputInterval
        printInterval = max(0.1, simConfig.endTime / 20.0)
        nextPrintTime = printInterval
        totalBreakingEvents = 0

        while solver.time < simConfig.endTime:
            # Update wave-maker boundary motion BEFORE step
            waveMaker.updateBoundaryMotion(particles, solver.time)

            # Advance solver
            state = solver.step()

            # Detect breaking events
            events = breakingDetector.detectBreaking(particles, solver.time)
            totalBreakingEvents += len(events)

            # Export frame at output intervals
            if solver.time >= nextOutputTime:
                self._exporter.addFrame(state, particles)
                nextOutputTime += simConfig.outputInterval

            # Print progress at regular intervals
            if solver.time >= nextPrintTime or state.step % 500 == 0:
                print(
                    f'  {state.time:8.4f}  {state.step:8d}  {state.dt:10.2e}  '
                    f'{state.maxVelocity:8.4f}  {state.maxDensityError * 100:8.3f}  '
                    f'{totalBreakingEvents:8d}'
                )
                nextPrintTime += printInterval

        wallClockEnd = timeModule.time()
        wallClockSeconds = wallClockEnd - wallClockStart

        # Final frame
        finalState = solver.currentState
        self._exporter.addFrame(finalState, particles)

        print()
        print(f'  Simulation complete.')
        print(f'  Total steps:       {finalState.step:8d}')
        print(f'  Wall-clock time:   {wallClockSeconds:8.1f} s')
        print(f'  Frames exported:   {self._exporter.nFrames:8d}')
        print(f'  Breaking events:   {totalBreakingEvents:8d}')
        print()

        #--------------------------------------------------------------------#
        # Export
        #--------------------------------------------------------------------#
        exportPath = None
        if doExport:
            print('-' * 62)
            print('  EXPORTING FRAME DATA')
            print('-' * 62)

            exportPath = self._exporter.export(
                config=simConfig,
                outputDir=exportDir,
                scenarioName='waveTank',
            )
            print(f'  Exported to: {exportPath}')
            print()

        #--------------------------------------------------------------------#
        # Summary
        #--------------------------------------------------------------------#
        breakingStats = breakingDetector.getStatistics()

        print('=' * 62)
        print('  SIMULATION SUMMARY')
        print('=' * 62)
        print(f'  Final KE:          {finalState.kineticEnergy:10.6f} J')
        print(f'  Final PE:          {finalState.potentialEnergy:10.6f} J')
        print(f'  Final Total E:     {finalState.totalEnergy:10.6f} J')
        print(f'  Max Density Error: {finalState.maxDensityError * 100:8.3f} %')
        print(f'  Max Velocity:      {finalState.maxVelocity:8.4f} m/s')
        print()
        print('  BREAKING STATISTICS')
        print(f'  Total Events:      {breakingStats["totalEvents"]:8d}')
        if breakingStats['firstBreakingTime'] is not None:
            print(f'  First Breaking:    {breakingStats["firstBreakingTime"]:8.3f} s')
            print(f'  First Break Pos:   x={breakingStats["firstBreakingPosition"][0]:.3f} m')
        if breakingStats['byType']:
            for bType, count in breakingStats['byType'].items():
                print(f'  {bType.capitalize():16s} {count:8d}')
        print('=' * 62)
        print()

        return {
            'finalState': finalState,
            'wallClockSeconds': wallClockSeconds,
            'nFrames': self._exporter.nFrames,
            'exportPath': exportPath,
            'breakingStats': breakingStats,
        }


#--------------------------------------------------------------------#
# -- CLI Entry Point -- #
#--------------------------------------------------------------------#

def main() -> None:
    '''CLI entry point.'''
    parser = buildParser()
    args = parser.parse_args()

    runner = WaterSimRunner()

    if args.config:
        runner.runFromConfig(args.config, exportDir=args.output_dir)
    elif args.scenario == 'waveTank':
        # Wave tank presets
        waveTankPresets = {
            'small': WaveTankConfig.small2D,
            'standard': WaveTankConfig.standard2D,
            'small3D': WaveTankConfig.small3D,
            'standard3D': WaveTankConfig.standard3D,
        }
        preset = args.preset if args.preset in waveTankPresets else 'small'
        tankConfig = waveTankPresets[preset]()
        runner.runWaveTank(
            tankConfig,
            doExport=not args.no_export,
            exportDir=args.output_dir,
        )
    else:
        # Sloshing tank presets
        sloshingPresets = {
            'small': SloshingTankConfig.small2D,
            'standard': SloshingTankConfig.standard2D,
        }
        preset = args.preset if args.preset in sloshingPresets else 'small'
        tankConfig = sloshingPresets[preset]()
        runner.runSloshing(
            tankConfig,
            doExport=not args.no_export,
            exportDir=args.output_dir,
        )


if __name__ == '__main__':
    main()
