# -- SurfPhysics Analysis Runner -- #

'''

Command-line entry point for running surfboard physics analyses.

Loads board parameters (from JSON or preset), configures wave conditions,
computes forces, and generates visualization dashboards.

Usage:
    python -m SurfPhysics                          # Shortboard, typical waves
    python -m SurfPhysics --board fish             # Fish preset
    python -m SurfPhysics --json board.json        # Custom board from JSON
    python -m SurfPhysics --wave-height 1.5 --wave-period 10 --depth 2.5
    python -m SurfPhysics --rider-mass 80

Sean Bowman [02/03/2026]

'''

from __future__ import annotations

import argparse
import math
from typing import Optional

import numpy as np

from pysurf.Surfboard.SurfPhysics import constants as const
from pysurf.Surfboard.SurfPhysics.geometry.parameters import SurfboardParameters
from pysurf.Surfboard.SurfPhysics.geometry.board import BoardGeometry
from pysurf.Surfboard.SurfPhysics.waves.waveConditions import WaveConditions
from pysurf.Surfboard.SurfPhysics.waves.linearWaveTheory import LinearWaveTheory
from pysurf.Surfboard.SurfPhysics.hydrodynamics.buoyancy import BuoyancyModel
from pysurf.Surfboard.SurfPhysics.hydrodynamics.forceBalance import ForceBalance
from pysurf.Surfboard.SurfPhysics.visualization.dashboard import createAnalysisDashboard

def buildParser() -> argparse.ArgumentParser:

    '''Build the CLI argument parser.'''
    parser = argparse.ArgumentParser(
        description='SurfPhysics -- Surfboard physics analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--board', type=str, default='shortboard',
        choices=['shortboard', 'longboard', 'fish'],
        help='Board preset (default: shortboard)',
    )
    parser.add_argument(
        '--json', type=str, default=None,
        help='Path to custom board parameters JSON file',
    )
    parser.add_argument(
        '--wave-height', type=float, default=1.5,
        help='Wave height in meters (default: 1.5)',
    )
    parser.add_argument(
        '--wave-period', type=float, default=10.0,
        help='Wave period in seconds (default: 10)',
    )
    parser.add_argument(
        '--depth', type=float, default=2.5,
        help='Water depth in meters (default: 2.5)',
    )
    parser.add_argument(
        '--rider-mass', type=float, default=75.0,
        help='Rider mass in kg (default: 75)',
    )
    parser.add_argument(
        '--no-dashboard', action='store_true',
        help='Skip dashboard generation (console output only)',
    )

    return parser

def loadBoard(args: argparse.Namespace) -> SurfboardParameters:

    '''Load board parameters from args.'''
    if args.json:
        return SurfboardParameters.fromJson(args.json)

    presets = {
        'shortboard': SurfboardParameters.shortboard,
        'longboard': SurfboardParameters.longboard,
        'fish': SurfboardParameters.fish,
    }
    return presets[args.board]()

def runAnalysis(
    params: SurfboardParameters,
    waveConditions: WaveConditions,
    riderMassKg: float = 75.0,
    showDashboard: bool = True,
) -> None:

    '''

    Run the full analysis pipeline.

    1. Construct BoardGeometry from parameters
    2. Compute board mass and static buoyancy
    3. Find rest draft (paddling position)
    4. Compute wave properties for given conditions
    5. Sweep forces over speed range
    6. Generate visualizations

    Parameters:
    -----------
    params : SurfboardParameters
        Board parameters
    waveConditions : WaveConditions
        Wave conditions
    riderMassKg : float
        Rider mass in kg
    showDashboard : bool
        Whether to generate and show the interactive dashboard

    '''
    print()
    print('=' * 62)
    print('  SURFPHYSICS ANALYSIS')
    print('=' * 62)
    print()

    #--------------------------------------------------------------------#
    # Board Geometry
    #--------------------------------------------------------------------#
    params.printSummary()
    print()

    board = BoardGeometry(params)
    volume = board.computeVolume()
    volumeL = volume * 1e3  # m^3 to liters
    planformArea = board.computePlanformArea()
    wettedArea = board.computeWettedSurfaceArea()

    print(f'  Computed Volume:    {volumeL:8.2f} L  ({volume * 1e6:.0f} cm^3)')
    print(f'  Planform Area:     {planformArea * 1e4:8.1f} cm^2')
    print(f'  Wetted Surface:    {wettedArea * 1e4:8.1f} cm^2')
    print()

    #--------------------------------------------------------------------#
    # Board Mass & Buoyancy
    #--------------------------------------------------------------------#
    print('-' * 62)
    print('  BUOYANCY ANALYSIS')
    print('-' * 62)

    buoyancy = BuoyancyModel(board, params)
    boardMass = buoyancy.estimateBoardMass()
    totalMass = boardMass + riderMassKg
    totalWeight = totalMass * const.gravity

    boardDraft = buoyancy.findEquilibriumDraft(boardMass)
    riderDraft = buoyancy.findEquilibriumDraft(totalMass)

    maxBuoyancy = buoyancy.buoyancyForce(volume)

    print(f'  Board Mass:        {boardMass:8.2f} kg  ({params.foamType.upper()} foam)')
    print(f'  Rider Mass:        {riderMassKg:8.1f} kg')
    print(f'  Total Mass:        {totalMass:8.1f} kg')
    print(f'  Total Weight:      {totalWeight:8.1f} N')
    print()
    print(f'  Max Buoyancy:      {maxBuoyancy:8.1f} N  (fully submerged)')
    print(f'  Board-only Draft:  {boardDraft * 100:8.2f} cm')
    print(f'  With Rider Draft:  {riderDraft * 100:8.2f} cm')
    print(f'  Buoyancy Ratio:    {maxBuoyancy / totalWeight:8.2f}x  '
          f'({"floats" if maxBuoyancy > totalWeight else "SINKS"})')
    print()

    #--------------------------------------------------------------------#
    # Wave Physics
    #--------------------------------------------------------------------#
    print('-' * 62)
    print('  WAVE ANALYSIS')
    print('-' * 62)

    waveModel = LinearWaveTheory()
    wavelength = waveModel.waveLength(waveConditions)
    phaseSpeed = waveModel.waveSpeed(waveConditions)
    groupSpeed = waveModel.groupSpeed(waveConditions)
    energy = waveModel.energyDensity(waveConditions)
    power = waveModel.energyFlux(waveConditions)
    depthClass = waveModel.depthClassification(waveConditions)
    isBroken = waveModel.isBroken(waveConditions)
    surfSpeed = waveModel.surfableWaveSpeed(waveConditions)

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

    #--------------------------------------------------------------------#
    # Hydrodynamic Forces
    #--------------------------------------------------------------------#
    print('-' * 62)
    print('  HYDRODYNAMIC PERFORMANCE')
    print('-' * 62)

    forceBalance = ForceBalance(board, params, riderMassKg)

    # Performance at key speeds
    testSpeeds = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    print(f'  {"Speed":>6}  {"Trim":>6}  {"Lift":>8}  {"Drag":>8}  {"L/D":>6}  {"Planing":>8}')
    print(f'  {"(m/s)":>6}  {"(deg)":>6}  {"(N)":>8}  {"(N)":>8}  {"":>6}  {"":>8}')
    print('  ' + '-' * 52)

    for v in testSpeeds:
        state = forceBalance.findEquilibrium(v)
        print(
            f'  {v:6.1f}  {state.trimAngleDeg:6.1f}  {state.liftForceN:8.1f}  '
            f'{state.dragForceN:8.1f}  {state.liftToDrag:6.1f}  '
            f'{"Yes" if state.isPlaning else "No":>8}'
        )

    print()

    # Planing threshold
    planingSpeed = math.sqrt(const.gravity * board.getLengthM())
    print(f'  Planing Threshold: {planingSpeed:8.2f} m/s  (Fn = 1.0)')
    print()

    #--------------------------------------------------------------------#
    # Dashboard
    #--------------------------------------------------------------------#
    if showDashboard:
        print('-' * 62)
        print('  Generating interactive dashboard...')
        fig = createAnalysisDashboard(params, waveConditions, riderMassKg)
        fig.show()
        print('  Dashboard opened in browser.')
    else:
        print('  Dashboard generation skipped (--no-dashboard).')

    print()
    print('=' * 62)
    print('  Analysis complete.')
    print('=' * 62)

def main() -> None:

    '''CLI entry point.'''
    parser = buildParser()
    args = parser.parse_args()

    params = loadBoard(args)
    waveConditions = WaveConditions(
        height=args.wave_height,
        period=args.wave_period,
        depth=args.depth,
    )

    runAnalysis(
        params=params,
        waveConditions=waveConditions,
        riderMassKg=args.rider_mass,
        showDashboard=not args.no_dashboard,
    )

if __name__ == '__main__':
    main()
