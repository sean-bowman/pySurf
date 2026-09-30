# -- Surfboard Parameters Dataclass -- #

'''

Parametric surfboard dimensions matching the C# SurfboardParameters.cs.

All spatial dimensions stored in millimeters to match the C# convention.
Conversion to SI meters happens at the computation boundary in BoardGeometry.

Sean Bowman [02/03/2026]

'''

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict


@dataclass
class SurfboardParameters:

    '''

    Parameters defining a parametric surfboard shape.
    All dimensions in millimeters.

    Mirrors SurfboardGeometry/Surfboard/SurfboardParameters.cs exactly.
    Factory classmethods provide the same presets (Shortboard, Longboard, Fish).

    '''

    #--------------------------------------------------------------------#
    # -- Primary Dimensions -- #
    #--------------------------------------------------------------------#
    # Total board length from nose to tail [mm]
    length: float = 1828.0           # 6'0"

    # Maximum width at the wide point [mm]
    maxWidth: float = 495.0          # 19.5"

    # Maximum thickness at the thickest point [mm]
    maxThickness: float = 62.0       # 2.44"

    #--------------------------------------------------------------------#
    # -- Outline Control -- #
    #--------------------------------------------------------------------#
    # Width measured 305mm (12") from the nose tip [mm]
    noseWidth: float = 300.0         # ~11.8"

    # Width at the tail [mm]
    tailWidth: float = 380.0         # ~15"

    # Wide point offset from board center [mm]
    # Negative = toward nose, positive = toward tail
    widePointOffset: float = -25.0

    # Half-width at the tail tip before any notch carving [mm]
    tailTipHalfWidth: float = 15.0

    #--------------------------------------------------------------------#
    # -- Rocker Profile -- #
    #--------------------------------------------------------------------#
    # Nose rocker height at the tip [mm]
    noseRocker: float = 120.0        # ~4.7"

    # Tail rocker height at the tail tip [mm]
    tailRocker: float = 40.0         # ~1.6"

    #--------------------------------------------------------------------#
    # -- Cross-Section Profile -- #
    #--------------------------------------------------------------------#
    # Deck dome height at center [mm]
    deckCrown: float = 8.0

    # Bottom single concave depth [mm]
    bottomConcave: float = 2.0

    # Rail edge radius [mm]
    railRadius: float = 12.0

    #--------------------------------------------------------------------#
    # -- Tail Shape -- #
    #--------------------------------------------------------------------#
    # Tail shape type: 'standard' or 'swallow'
    tailShape: str = 'standard'

    # Swallow tail notch depth from the tail tip [mm]
    swallowNotchDepth: float = 0.0

    # Half-width of the swallow notch at the tail tip [mm]
    swallowNotchHalfWidth: float = 0.0

    #--------------------------------------------------------------------#
    # -- Fin Configuration -- #
    #--------------------------------------------------------------------#
    # Default fin configuration: 'thruster', 'twin', 'quad', or 'single'
    defaultFinConfiguration: str = 'thruster'

    #--------------------------------------------------------------------#
    # -- Construction -- #
    #--------------------------------------------------------------------#
    # Foam core type: 'pu' or 'eps'
    foamType: str = 'pu'

    #--------------------------------------------------------------------#
    # -- Computed Properties -- #
    #--------------------------------------------------------------------#
    @property
    def halfWidth(self) -> float:
        '''Half the maximum width [mm].'''
        return self.maxWidth / 2.0

    @property
    def halfLength(self) -> float:
        '''Half the total length [mm].'''
        return self.length / 2.0

    @property
    def widePointX(self) -> float:
        '''X position of the wide point from the nose [mm].'''
        return self.halfLength + self.widePointOffset

    @property
    def halfThickness(self) -> float:
        '''Half the maximum thickness [mm].'''
        return self.maxThickness / 2.0

    @property
    def approxVolumeLiters(self) -> float:

        '''

        Approximate board volume [liters] using ellipsoid estimation.
        V = (4/3) * pi * (L/2) * (W/2) * (T/2) * packingFactor / 1e6

        '''

        return (
            (4.0 / 3.0) * math.pi
            * (self.length / 2.0)
            * (self.maxWidth / 2.0)
            * (self.maxThickness / 2.0)
            * 0.35
            / 1_000_000.0
        )

    @property
    def approxPlanformAreaMm2(self) -> float:
        '''Planform area estimate [mm^2].'''
        return self.length * self.maxWidth * 0.65

    #--------------------------------------------------------------------#
    # -- Factory Presets -- #
    #--------------------------------------------------------------------#
    @classmethod
    def shortboard(cls) -> SurfboardParameters:

        '''

        Standard performance shortboard (6\'0" x 19.5" x 2.44").
        High-performance board for experienced surfers in good waves.

        '''

        return cls()

    @classmethod
    def longboard(cls) -> SurfboardParameters:

        '''

        Classic longboard (9\'0" x 22.5" x 3").
        Traditional longboard for nose riding and glide.

        '''

        return cls(
            length=2743.0,           # 9'0"
            maxWidth=570.0,          # 22.5"
            maxThickness=75.0,       # 3.0"
            noseWidth=430.0,         # ~17" wide nose for nose riding
            tailWidth=370.0,         # ~14.5"
            widePointOffset=0.0,     # centered wide point
            noseRocker=180.0,        # ~7" generous nose rocker
            tailRocker=25.0,         # ~1" minimal tail rocker for speed
            deckCrown=10.0,
            bottomConcave=1.0,       # subtle concave
            railRadius=18.0,         # soft, round rails
            defaultFinConfiguration='single',
        )

    @classmethod
    def fish(cls) -> SurfboardParameters:

        '''

        Round Nose Fish / RNF (5\'6" x 21" x 2.56").
        Wide, flat, and fast with swallow tail and twin fins.

        '''

        return cls(
            length=1676.0,           # 5'6"
            maxWidth=533.0,          # 21"
            maxThickness=65.0,       # 2.56"
            noseWidth=400.0,         # ~15.75" fuller/rounder nose
            tailWidth=430.0,         # ~16.9" wide tail
            widePointOffset=-25.0,   # near center
            noseRocker=80.0,         # ~3.1" minimal nose rocker
            tailRocker=30.0,         # ~1.2" low tail rocker
            deckCrown=6.0,
            bottomConcave=3.0,       # deeper concave for speed
            railRadius=10.0,         # medium-hard rails
            tailShape='swallow',
            tailTipHalfWidth=95.0,           # ~3.7" half-width at each lobe
            swallowNotchDepth=110.0,         # ~4.3" deep U-notch
            swallowNotchHalfWidth=55.0,      # ~2.2" notch half-width
            defaultFinConfiguration='twin',
        )

    #--------------------------------------------------------------------#
    # -- JSON I/O -- #
    #--------------------------------------------------------------------#
    def toJson(self, filePath: str) -> None:

        '''

        Export parameters to JSON file for C# interop.

        Parameters:
        -----------
        filePath : str
            Output file path

        '''

        data = {
            'boardType': self.defaultFinConfiguration,
            'dimensions': {
                'lengthMm': self.length,
                'maxWidthMm': self.maxWidth,
                'maxThicknessMm': self.maxThickness,
            },
            'outline': {
                'noseWidthMm': self.noseWidth,
                'tailWidthMm': self.tailWidth,
                'widePointOffsetMm': self.widePointOffset,
                'tailTipHalfWidthMm': self.tailTipHalfWidth,
            },
            'rocker': {
                'noseRockerMm': self.noseRocker,
                'tailRockerMm': self.tailRocker,
            },
            'crossSection': {
                'deckCrownMm': self.deckCrown,
                'bottomConcaveMm': self.bottomConcave,
                'railRadiusMm': self.railRadius,
            },
            'tail': {
                'tailShape': self.tailShape,
                'swallowNotchDepthMm': self.swallowNotchDepth,
                'swallowNotchHalfWidthMm': self.swallowNotchHalfWidth,
            },
            'fins': {
                'defaultConfiguration': self.defaultFinConfiguration,
            },
            'construction': {
                'foamType': self.foamType,
            },
        }
        with open(filePath, 'w') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def fromJson(cls, filePath: str) -> SurfboardParameters:

        '''

        Load parameters from a JSON file.

        Parameters:
        -----------
        filePath : str
            Path to JSON parameter file

        Returns:
        --------
        SurfboardParameters : Loaded parameters

        '''

        with open(filePath, 'r') as f:
            data = json.load(f)

        dims = data.get('dimensions', {})
        outline = data.get('outline', {})
        rocker = data.get('rocker', {})
        cross = data.get('crossSection', {})
        tail = data.get('tail', {})
        fins = data.get('fins', {})
        construction = data.get('construction', {})

        return cls(
            length=dims.get('lengthMm', 1828.0),
            maxWidth=dims.get('maxWidthMm', 495.0),
            maxThickness=dims.get('maxThicknessMm', 62.0),
            noseWidth=outline.get('noseWidthMm', 300.0),
            tailWidth=outline.get('tailWidthMm', 380.0),
            widePointOffset=outline.get('widePointOffsetMm', -25.0),
            tailTipHalfWidth=outline.get('tailTipHalfWidthMm', 15.0),
            noseRocker=rocker.get('noseRockerMm', 120.0),
            tailRocker=rocker.get('tailRockerMm', 40.0),
            deckCrown=cross.get('deckCrownMm', 8.0),
            bottomConcave=cross.get('bottomConcaveMm', 2.0),
            railRadius=cross.get('railRadiusMm', 12.0),
            tailShape=tail.get('tailShape', 'standard'),
            swallowNotchDepth=tail.get('swallowNotchDepthMm', 0.0),
            swallowNotchHalfWidth=tail.get('swallowNotchHalfWidthMm', 0.0),
            defaultFinConfiguration=fins.get('defaultConfiguration', 'thruster'),
            foamType=construction.get('foamType', 'pu'),
        )

    #--------------------------------------------------------------------#
    # -- Display -- #
    #--------------------------------------------------------------------#
    def printSummary(self) -> None:
        '''Print surfboard parameters to console in a formatted table.'''
        def mmToFeetInches(mm: float) -> str:
            totalInches = mm / 25.4
            feet = int(totalInches // 12)
            inches = totalInches - feet * 12
            if inches >= 11.95:
                feet += 1
                inches = 0.0
            return f'{feet}\'{inches:.1f}"'

        def mmToInches(mm: float) -> str:
            return f'{mm / 25.4:.2f}"'

        print('=' * 58)
        print('  SURFBOARD PARAMETERS SUMMARY')
        print('=' * 58)
        print(f'  Length:            {self.length:8.1f} mm  ({mmToFeetInches(self.length):>8})')
        print(f'  Max Width:         {self.maxWidth:8.1f} mm  ({mmToInches(self.maxWidth):>8})')
        print(f'  Max Thickness:     {self.maxThickness:8.1f} mm  ({mmToInches(self.maxThickness):>8})')
        print('-' * 58)
        print(f'  Nose Width (12"):  {self.noseWidth:8.1f} mm  ({mmToInches(self.noseWidth):>8})')
        print(f'  Tail Width:        {self.tailWidth:8.1f} mm  ({mmToInches(self.tailWidth):>8})')
        print(f'  Wide Pt Offset:    {self.widePointOffset:8.1f} mm')
        print('-' * 58)
        print(f'  Nose Rocker:       {self.noseRocker:8.1f} mm  ({mmToInches(self.noseRocker):>8})')
        print(f'  Tail Rocker:       {self.tailRocker:8.1f} mm  ({mmToInches(self.tailRocker):>8})')
        print('-' * 58)
        print(f'  Deck Crown:        {self.deckCrown:8.1f} mm')
        print(f'  Bottom Concave:    {self.bottomConcave:8.1f} mm')
        print(f'  Rail Radius:       {self.railRadius:8.1f} mm')
        print('-' * 58)
        print(f'  Approx Volume:     {self.approxVolumeLiters:8.1f} L')
        print(f'  Tail Shape:        {self.tailShape:>8}')
        print(f'  Fins:              {self.defaultFinConfiguration:>8}')
        print('=' * 58)
