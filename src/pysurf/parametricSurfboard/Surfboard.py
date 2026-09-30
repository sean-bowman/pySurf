
# -- Class Definition for the Generation of Parametric Surfboard Geometry -- #

'''

This module defines the Surfboard class, which encapsulates the geometry and parameters of a surfboard.
It provides methods for generating the surfboard geometry based on a set of parameters, as well as utilities
for exporting the geometry to STL format for use in CFD simulations or 3D printing.

Sean Bowman [04/01/2026]

'''

# Global Imports
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pysurf.utilsCE import bezierCurve, bezierSubdivide, bezierEval

class Surfboard:

    '''
    
    This class represents a parametric surfboard geometry. It includes methods for generating the surfboard
    geometry based on a set of parameters, as well as utilities for exporting the geometry to STL format.
    
    '''

    def __init__(self,
                 totalLength      : float = 6.0,
                 maxWidth         : float = 20.0,
                 midpointThickness: float = 2.5,
                 noseShape        : str   = 'point',
                 tailShape        : str   = 'round',
                 rockerProfile    : str   = 'aggressive',
                 finConfig        : str   = 'thruster') -> None:

        '''

        Initializes the Surfboard object with the given parameters.

        Parameters:
        -----------
        totalLength : float
            Board length in feet (default: 6.0).
        maxWidth : float
            Maximum width in inches (default: 20.0).
        midpointThickness : float
            Thickness at the midpoint in inches (default: 2.5).
        noseShape : str
            Nose shape preset: 'point', 'round', or 'fish' (default: 'point').
        tailShape : str
            Tail shape preset: 'squash', 'round', 'pin', or 'fish' (default: 'round').
        rockerProfile : str
            Rocker preset: 'flat', 'medium', or 'aggressive' (default: 'aggressive').
        finConfig : str
            Fin configuration: 'thruster', 'quad', 'single', or 'twin' (default: 'thruster').

        Settable properties (set per board-type preset or override after construction):
        ------------------------------------------------------------------------------
        resolution               : int: axial station count (default 1000)
        outlineMag               : list: Bezier handle magnitudes [dep, arr] for outline (default [0.35, 0.35])
        rockerMag                : list: Bezier handle magnitudes [dep, arr] for rocker  (default [0.35, 0.35])
        showControlPoints        : bool: overlay Bezier control points in plotGeometry (default True)

        Set by noseShape preset (overridable):
        noseOutlineMidY          : float|None: Y (ft) of nose outline subdivision midpoint
        noseOutlineMidAngle      : float|None: tangent angle (deg) at nose outline midpoint (None = avg dep+arr)

        Set by tailShape preset (overridable):
        tailOutlineMidY          : float|None: Y (ft) of tail outline subdivision midpoint
        tailOutlineMidAngle      : float|None: tangent angle (deg) at tail outline midpoint

        Set by rockerProfile preset (overridable, independent per surface):
        tailTopRockerAngle       : float: departure angle (deg) at tail tip, top surface
        tailBottomRockerAngle    : float: departure angle (deg) at tail tip, bottom surface
        noseTopRockerAngle       : float: arrival angle (deg) at nose tip, top surface
        noseBottomRockerAngle    : float: arrival angle (deg) at nose tip, bottom surface
        tailTopRockerMidY        : float|None: Z (ft) of tail rocker midpoint, top surface
        tailBottomRockerMidY     : float|None: Z (ft) of tail rocker midpoint, bottom surface
        noseTopRockerMidY        : float|None: Z (ft) of nose rocker midpoint, top surface
        noseBottomRockerMidY     : float|None: Z (ft) of nose rocker midpoint, bottom surface
        tailTopRockerMidAngle    : float|None: tangent angle (deg) at tail rocker midpoint, top surface
        tailBottomRockerMidAngle : float|None: tangent angle (deg) at tail rocker midpoint, bottom surface
        noseTopRockerMidAngle    : float|None: tangent angle (deg) at nose rocker midpoint, top surface
        noseBottomRockerMidAngle : float|None: tangent angle (deg) at nose rocker midpoint, bottom surface

        '''

        # Initialize the surfboard parameters as object properties
        self.totalLength       = totalLength
        self.maxWidth          = maxWidth
        self.midpointThickness = midpointThickness
        self.noseShape         = noseShape
        self.tailShape         = tailShape
        self.rockerProfile     = rockerProfile
        self.finConfig         = finConfig

        # Set default derived parameters based on the input parameters

        # -- Nose/Tail Shapes and Board Outline -- #

        match self.noseShape:

            case 'point':

                self.maxWidthFrac       = 0.50   # fraction of totalLength where max width occurs
                self.noseWidthFrac      = 0.01   # fraction of maxWidth for nose tip half-width (0 = perfect point)
                self.noseAngle          = -42.0  # arrival angle at nose tip
                self.noseOutlineMidY    = 0.7
                self.noseOutlineMidAngle = -11.0

            case 'round':

                self.maxWidthFrac       = 0.55
                self.noseWidthFrac      = 0.0
                self.noseAngle          = -85.0
                self.noseOutlineMidY    = 0.5
                self.noseOutlineMidAngle = None

            case 'fish':

                self.maxWidthFrac       = 0.60
                self.noseWidthFrac      = 0.01
                self.noseAngle          = -75.0
                self.noseOutlineMidY    = 0.5
                self.noseOutlineMidAngle = None

        match self.tailShape:

            case 'squash':

                self.tailWidthFrac      = 0.40  # fraction of maxWidth for tail tip half-width
                self.tailAngle          = 78.0  # departure angle at tail tip (high = board widens rapidly from tail)
                self.tailOutlineMidY    = 0.5
                self.tailOutlineMidAngle = None

            case 'round':

                self.tailWidthFrac      = 0.0
                self.tailAngle          = 82.0  # rounder tail widens even more steeply from the very tip
                self.tailOutlineMidY    = 0.675
                self.tailOutlineMidAngle = 12.0

            case 'pin':

                self.tailWidthFrac      = 0.01
                self.tailAngle          = 60.0  # pin tail tapers gradually: shallower departure angle
                self.tailOutlineMidY    = 0.5
                self.tailOutlineMidAngle = None

            case 'fish':

                self.tailWidthFrac      = 0.40
                self.tailAngle          = 70.0
                self.tailOutlineMidY    = 0.5
                self.tailOutlineMidAngle = None
                
        # -- Rocker Profiles -- #

        match self.rockerProfile:

            case 'flat':

                self.noseBottomRockerFt       = 0.5
                self.tailBottomRockerFt       = 0.25
                self.noseTopRockerFt          = 0.4
                self.tailTopRockerFt          = 0.15
                self.rockerFlatSpotFrac       = 0.40   # fraction of totalLength where flat spot (minimum) occurs
                self.tailTopRockerAngle       = -10.0  # departure angle at tail tip, top surface
                self.tailBottomRockerAngle    = -10.0  # departure angle at tail tip, bottom surface
                self.noseTopRockerAngle       = 30.0   # arrival angle at nose tip, top surface
                self.noseBottomRockerAngle    = 30.0   # arrival angle at nose tip, bottom surface
                self.tailTopRockerMidY        = 0.025
                self.tailBottomRockerMidY     = 0.025
                self.noseTopRockerMidY        = 0.05
                self.noseBottomRockerMidY     = 0.05
                self.tailTopRockerMidAngle    = None
                self.tailBottomRockerMidAngle = None
                self.noseTopRockerMidAngle    = None
                self.noseBottomRockerMidAngle = None

            case 'medium':

                self.noseBottomRockerFt       = 2.5
                self.tailBottomRockerFt       = 1.5
                self.noseTopRockerFt          = 2.25
                self.tailTopRockerFt          = 1.25
                self.rockerFlatSpotFrac       = 0.40
                self.tailTopRockerAngle       = -15.0
                self.tailBottomRockerAngle    = -15.0
                self.noseTopRockerAngle       = 50.0
                self.noseBottomRockerAngle    = 50.0
                self.tailTopRockerMidY        = 0.025
                self.tailBottomRockerMidY     = 0.025
                self.noseTopRockerMidY        = 0.05
                self.noseBottomRockerMidY     = 0.05
                self.tailTopRockerMidAngle    = None
                self.tailBottomRockerMidAngle = None
                self.noseTopRockerMidAngle    = None
                self.noseBottomRockerMidAngle = None

            case 'aggressive':

                self.noseBottomRockerFt       = 0.435
                self.tailBottomRockerFt       = 0.225
                self.noseTopRockerFt          = 0.45
                self.tailTopRockerFt          = 0.285
                self.rockerFlatSpotFrac       = 0.40
                self.tailTopRockerAngle       = -2.0
                self.tailBottomRockerAngle    = -8.0
                self.noseTopRockerAngle       = 15.0
                self.noseBottomRockerAngle    = 25.0
                self.tailTopRockerMidY        = 0.25
                self.tailBottomRockerMidY     = 0.075
                self.noseTopRockerMidY        = 0.25
                self.noseBottomRockerMidY     = 0.05
                self.tailTopRockerMidAngle    = -1.0
                self.tailBottomRockerMidAngle = -6.0
                self.noseTopRockerMidAngle    = 2.0
                self.noseBottomRockerMidAngle = 6.0

        # Bezier magnitude parameters: fraction of chord length for interior control point offsets.
        # Larger values produce a fuller curve through the middle; typical range 0.2–0.6.
        self.outlineMag = [0.35, 0.35]  # [departure side, arrival side] for outline segments
        self.rockerMag  = [0.35, 0.35]  # [departure side, arrival side] for rocker segments

        # Geometry resolution and display flags
        self.resolution            = 1000
        self.numCrossSectionPoints = 200
        self.debugMode             = False
        self.plotsEnabled          = False
        self.showControlPoints     = True

        # Create placeholders for the geometry data
        self.geometryData      = None
        self.referenceGeometry = None

    #------------------------------------------------------------------------------#
    # -- Private Methods -- #
    #------------------------------------------------------------------------------#

    def _validateInputs(self) -> None:

        '''

        Validates the input parameters to ensure they are within reasonable bounds and of the correct type.
        
        Raises:
            ValueError: If any of the input parameters are invalid.
        
        '''
        
        if self.totalLength <= 0:
            raise ValueError("Total length must be a positive value.")
        if self.maxWidth <= 0:
            raise ValueError("Maximum width must be a positive value.")
        if self.midpointThickness <= 0:
            raise ValueError("Midpoint thickness must be a positive value.")
        if self.noseShape not in ['point', 'round', 'fish']:
            raise ValueError("Nose shape must be 'point', 'round', or 'fish'.")
        if self.tailShape not in ['squash', 'round', 'pin', 'fish']:
            raise ValueError("Tail shape must be 'squash', 'round', 'pin', or 'fish'.")
        if self.rockerProfile not in ['flat', 'medium', 'aggressive']:
            raise ValueError("Rocker profile must be 'flat', 'medium', or 'aggressive'.")
        
    def _convertUnits(self, outputUnits: str = 'feet') -> dict:

        '''

        Converts the input parameters to consistent units for geometry generation. This method converts
        the total length from whatever they currently are to the desired output units.

        Returns:
            A dictionary containing the converted parameters.
        
        '''

        match outputUnits:

            case 'feet':
                
                convertedValues = {
                    'totalLength': self.totalLength,
                    'maxWidth': self.maxWidth / 12,  # Convert inches to feet
                    'midpointThickness': self.midpointThickness / 12  # Convert inches to feet
                }

            case 'inches':
        
                convertedValues = {
                    'totalLength': self.totalLength * 12,  # Convert feet to inches
                    'maxWidth': self.maxWidth,
                    'midpointThickness': self.midpointThickness
                }
        
        return convertedValues

    #------------------------------------------------------------------------------#
    # -- Geometry Generation Methods -- #
    #------------------------------------------------------------------------------#

    def _splitBezierSection(self, p1: list, p4: list, thetaDep: float, thetaArr: float,
                            mag: list, midAngleOverride: float | None,
                            resInternal: int, midYOverride: float | None = None) -> tuple[list, list]:

        '''

        Generate a Bezier curve section as two linked cubic sub-segments joined at the de
        Casteljau midpoint (t=0.5 of the equivalent single cubic), with optional tangent
        angle override at that midpoint.

        When midAngleOverride is None the subdivision is exact: the two cubics together are
        mathematically identical to the single cubic that bezierCurve would produce, so there
        is zero visual change from using this helper. When midAngleOverride is a float the
        midpoint position stays fixed at the subdivision point but the tangent is rotated to
        the specified angle, bending the curve locally while preserving C1 continuity at both
        the midpoint and the outer endpoints.

        Parameters:
        -----------
        p1, p4 : list[float]
            Start and end points as [x, y].
        thetaDep, thetaArr : float
            Departure angle at p1 and arrival angle at p4 (degrees from +x axis).
        mag : list[float]
            Bezier magnitude list [mag1, mag2] as fractions of chord length.
        midAngleOverride : float or None
            Tangent angle at the subdivision midpoint. None = exact de Casteljau subdivision.
        resInternal : int
            Total sample count; each sub-segment gets half.

        Returns:
        --------
        combinedCurve : list[np.ndarray]
            [x, y] arrays for the full section at resInternal resolution.
        cpList : list[dict]
            Two control-point dicts (left sub-segment, right sub-segment), each with
            keys p1, p2, p3, p4, tangent1, tangent2.

        '''

        resHalf = max(2, resInternal // 2)

        # Build full single cubic control points for subdivision
        _, cpFull = bezierCurve(p1, p4, thetaDep, thetaArr, list(mag), 2, returnControlPoints=True)

        # De Casteljau subdivision at t=0.5
        leftCtrl, rightCtrl = bezierSubdivide(
            cpFull['p1'], cpFull['p2'], cpFull['p3'], cpFull['p4']
        )
        midPoint = list(leftCtrl[3])   # point on original curve at t=0.5

        # Shift the on-curve midpoint in Y if requested. When no explicit midAngleOverride
        # is given, the fallback tangent is the average of the section's departure and arrival
        # angles. This is always sign-consistent with the section's travel direction (unlike
        # the de Casteljau internal tangent, which can have the wrong sign when the control
        # polygon dips below the chord: e.g. nose rocker with a steep arrival angle).
        if midYOverride is not None:
            midPoint[1] = midYOverride
            if midAngleOverride is None:
                midAngleOverride = 0.5 * (thetaDep + thetaArr)

        if midAngleOverride is None:
            # Exact subdivision: curve is identical to the original single cubic
            curveL, cpL = bezierEval(*leftCtrl,  resHalf, returnControlPoints=True)
            curveR, cpR = bezierEval(*rightCtrl, resHalf, returnControlPoints=True)
        else:
            # Custom angle at midpoint (or Y-override with preserved tangent direction):
            # rebuild each half with bezierCurve so the curve passes through midPoint
            curveL, cpL = bezierCurve(
                p1, midPoint, thetaDep, midAngleOverride, list(mag), resHalf,
                returnControlPoints=True,
            )
            curveR, cpR = bezierCurve(
                midPoint, p4, midAngleOverride, thetaArr, list(mag), resHalf,
                returnControlPoints=True,
            )

        # Concatenate, trimming the duplicate midpoint from the right sub-segment
        combinedX = np.concatenate([np.array(curveL[0]),  np.array(curveR[0])[1:]])
        combinedY = np.concatenate([np.array(curveL[1]),  np.array(curveR[1])[1:]])

        return [combinedX, combinedY], [cpL, cpR]

    def _importGeometry(self, filePath: str) -> None:
        
        '''
        
        Imports surfboard geometry from an STL file. This reads an STL file and extracts the geometry data.

        Parameters:
            filePath: The path to the STL file containing the surfboard geometry.
        
        '''
        
        # Read the STL file and extract geometry data
        pass

    def _exportGeometry(self, filePath: str) -> None:
        
        '''
        
        Exports the generated surfboard geometry to an STL file. This writes the geometry data to an STL file format.

        Parameters:
            filePath: The path to the output STL file.
        
        '''
        
        # Write the geometry data to an STL file
        pass

    def _generateRockerProfile(self, isTopOrBottom: str = 'top') -> tuple[np.ndarray, np.ndarray, dict]:

        '''

        Generates the rocker profile (side view height above the baseline) using two cubic Bezier
        segments joined with C1 continuity at the flat spot.

        Segment 1 (tail tip → flat spot): departure = self.tail{Top|Bottom}RockerAngle, arrival = 0°
        Segment 2 (flat spot → nose tip): departure = 0°, arrival = self.nose{Top|Bottom}RockerAngle

        The zero-slope join at the flat spot (both segments arrive/depart horizontally) ensures
        a smooth transition with no kink, consistent with the physical flat spot on a real board.

        For the top surface, a thickness offset is added: constant (= midpointThickness/12) through
        the tail region, tapering linearly to zero at the nose tip so the top and bottom curves
        converge at the nose.

        Parameters:
        -----------
        isTopOrBottom : str
            'top' for deck surface, 'bottom' for bottom surface.

        Returns:
        --------
        xRocker : np.ndarray
            Axially-uniform x coordinates (feet).
        rRocker : np.ndarray
            Rocker height at each x station (feet).
        controlPoints : dict
            Bezier control point data: {'tail': {...}, 'nose': {...}}.

        '''

        # Convert to feet (local vars only: do not mutate self)
        convertedValues  = self._convertUnits(outputUnits='feet')
        totalLength      = convertedValues['totalLength']

        isTop            = isTopOrBottom == 'top'
        noseRockerFt     = self.noseTopRockerFt  if isTop else self.noseBottomRockerFt
        tailRockerFt     = self.tailTopRockerFt  if isTop else self.tailBottomRockerFt

        xFlat            = self.rockerFlatSpotFrac * totalLength
        baseThicknessOff = self.midpointThickness / 12.0

        # Internal Bezier resolution; each section splits into two cubics internally
        resInternal = self.resolution * 8

        # All rocker endpoint and midpoint heights are physical Z values above the flat datum.
        # The only place midpointThickness (via baseThicknessOff) appears is the flat spot:
        # bottom surface flat spot = 0, top surface flat spot = baseThicknessOff.
        # Tail tips, nose tips, and subdivision midpoints are all user-specified physical heights.
        tailStartZ = tailRockerFt
        tailEndZ   = baseThicknessOff if isTop else 0.0

        # -- Select per-surface parameters -- #
        tailMidY     = self.tailTopRockerMidY        if isTop else self.tailBottomRockerMidY
        noseMidY     = self.noseTopRockerMidY        if isTop else self.noseBottomRockerMidY
        tailMidAngle = self.tailTopRockerMidAngle    if isTop else self.tailBottomRockerMidAngle
        noseMidAngle = self.noseTopRockerMidAngle    if isTop else self.noseBottomRockerMidAngle
        tailAngle    = self.tailTopRockerAngle       if isTop else self.tailBottomRockerAngle
        noseAngle    = self.noseTopRockerAngle       if isTop else self.noseBottomRockerAngle

        # -- Tail section: tail tip → flat spot (two linked cubics) --
        tailCurve, cpTailList = self._splitBezierSection(
            [0.0,   tailStartZ],
            [xFlat, tailEndZ  ],
            tailAngle,  # departure at tail tip (negative = pointing down from raised tail)
            0.0,        # horizontal arrival at flat spot
            self.rockerMag, tailMidAngle, resInternal,
            midYOverride=tailMidY,
        )

        # -- Nose section: flat spot → nose tip (two linked cubics) --
        noseCurve, cpNoseList = self._splitBezierSection(
            [xFlat,       tailEndZ    ],
            [totalLength, noseRockerFt],
            0.0,        # horizontal departure from flat spot (C1 with tail section)
            noseAngle,  # arrival at nose tip (positive = rising into nose)
            self.rockerMag, noseMidAngle, resInternal,
            midYOverride=noseMidY,
        )

        # Resample onto uniform x grid: split at xFlat to keep each section's x-range clean
        xRocker   = np.linspace(0.0, totalLength, self.resolution)
        xTailGrid = xRocker[xRocker <= xFlat]
        xNoseGrid = xRocker[xRocker >  xFlat]

        sortTail  = np.argsort(tailCurve[0])
        sortNose  = np.argsort(noseCurve[0])
        rTailGrid = np.interp(xTailGrid, tailCurve[0][sortTail], tailCurve[1][sortTail])
        rNoseGrid = np.interp(xNoseGrid, noseCurve[0][sortNose], noseCurve[1][sortNose])

        rRocker = np.concatenate([rTailGrid, rNoseGrid])

        # Debug plot
        if self.debugMode:
            plt.style.use('dark_background')
            plt.figure(figsize=(10, 4))
            plt.plot(xRocker, rRocker, label=f'Rocker ({isTopOrBottom})')
            plt.title('Surfboard Rocker Profile (Side View)')
            plt.xlabel('Length (ft)')
            plt.ylabel('Rocker Height (ft)')
            plt.gca().set_aspect('auto')
            plt.show(block=False)

        # Control points stored as lists of two dicts per section (one per sub-segment)
        controlPoints = {'tail': cpTailList, 'nose': cpNoseList}

        return xRocker, rRocker, controlPoints

    def _generateNoseToTailProfile(self) -> tuple[np.ndarray, np.ndarray, dict]:

        '''

        Generates the nose-to-tail planform outline of the surfboard using two cubic Bezier segments
        joined with C1 continuity at the widest point.

        Segment 1 (tail → wide point): departure angle = self.tailAngle, arrival angle = 0°
        Segment 2 (wide point → nose): departure angle = 0°, arrival angle = self.noseAngle

        The C1 join (both segments tangent to horizontal at the wide point) ensures no kink at
        the widest point regardless of shape parameters.

        Each segment is generated at high internal parametric resolution, then resampled onto a
        uniform axial (x) grid via interpolation. The number of output points per segment is
        proportional to its axial length so that overall point density is consistent.

        Returns:
        --------
        xBoard : np.ndarray
            Axially-uniform x coordinates (tail → nose, feet).
        rBoard : np.ndarray
            Corresponding half-width values (feet).
        controlPoints : dict
            Bezier control point data for both segments, suitable for visualization:
            {'tail': {...}, 'nose': {...}} where each sub-dict has p1, p2, p3, p4, tangent1, tangent2.

        '''

        # Convert to feet (local vars only: do not mutate self)
        convertedValues = self._convertUnits(outputUnits='feet')
        totalLengthFt   = convertedValues['totalLength']
        maxWidthFt      = convertedValues['maxWidth']

        xWidePoint = self.maxWidthFrac * totalLengthFt
        tailOffset = (self.tailWidthFrac * maxWidthFt) if self.tailShape != 'fish' else 0.0
        noseOffset = self.noseWidthFrac  * maxWidthFt

        # Points per segment proportional to axial length so output density is uniform
        nTail = max(2, round(self.resolution * (xWidePoint / totalLengthFt)))
        nNose = max(2, self.resolution - nTail)

        # High internal resolution for accurate interpolation onto the uniform x grid.
        # Each section uses two linked cubics (see _splitBezierSection), so resInternal
        # per section is split in half internally: use 8× to keep density high.
        resInternal = self.resolution * 8

        # -- Tail section: tail tip → wide point (two linked cubics) --
        tailCurve, cpTailList = self._splitBezierSection(
            [0.0,        tailOffset    ],
            [xWidePoint, maxWidthFt / 2],
            self.tailAngle, 0.0,
            self.outlineMag, self.tailOutlineMidAngle, resInternal,
            midYOverride=self.tailOutlineMidY,
        )

        # -- Nose section: wide point → nose tip (two linked cubics) --
        noseCurve, cpNoseList = self._splitBezierSection(
            [xWidePoint,    maxWidthFt / 2],
            [totalLengthFt, noseOffset    ],
            0.0, self.noseAngle,
            self.outlineMag, self.noseOutlineMidAngle, resInternal,
            midYOverride=self.noseOutlineMidY,
        )

        # Resample each section onto a uniform x grid via linear interpolation.
        # Sort by x to guard against any floating-point non-monotonicity.
        xTailGrid = np.linspace(0.0,        xWidePoint,    nTail)
        xNoseGrid = np.linspace(xWidePoint, totalLengthFt, nNose + 1)[1:]  # trim shared wide-point

        sortTail  = np.argsort(tailCurve[0])
        rTailGrid = np.interp(xTailGrid, tailCurve[0][sortTail], tailCurve[1][sortTail])

        sortNose  = np.argsort(noseCurve[0])
        rNoseGrid = np.interp(xNoseGrid, noseCurve[0][sortNose], noseCurve[1][sortNose])

        xBoard = np.concatenate([xTailGrid, xNoseGrid])
        rBoard = np.concatenate([rTailGrid, rNoseGrid])

        # Control points stored as lists of two dicts per section (one per sub-segment)
        controlPoints = {'tail': cpTailList, 'nose': cpNoseList}

        # Debug plot
        if self.debugMode:
            plt.style.use('dark_background')
            plt.figure(figsize=(10, 4))
            plt.plot(xBoard,  rBoard, color='cyan', label='outline')
            plt.plot(xBoard, -rBoard, color='cyan')
            plt.title('Surfboard Nose-to-Tail Profile (Top View)')
            plt.xlabel('Length (ft)')
            plt.ylabel('Half-Width (ft)')
            plt.gca().set_aspect('equal')
            plt.show(block=False)

        return xBoard, rBoard, controlPoints

    def _generateBoardCrossSections(self, geometryData: dict) -> dict:

        '''
        
        Generates the axial cross sections of the surfboard based on the nose-to-tail profile and rocker profiles.
        
        This method creates a 3D representation of the surfboard geometry by combining the top view outline with the side view rocker profiles.

        Board cross sections are built from the tail to the nose (x = 0 to x = totalLength) using the board outline and two rocker profiles
        to determine the bounding box for each cross section. The purpose of this method is to create a smooth profile between
        the cross section boundaries that is representative of the board deck, bottom, and rails.

        The coordinate frame for the 3D surfboard is as follows:
            X (i-hat): Runs from tail (0) to nose (totalLength) along the centerline of the board
            Y (j-hat): Runs from the centerline to the right rail (positive) and left rail (negative)
            Z (k-hat): Runs from the bottom of the board (0) to the deck (positive), with the midpoint
                       of the thickness defining the approximate z=0 plane for the cross sections
        
        To that end, each board cross section is drawn in the YZ plane and each subsequent cross section represents
        a step in the X direction.

        The board cross sections in general look like:
                    ________________
                   /                \
        Left Rail (                  ) Right Rail
                   \________________/
        
        '''

        # Unpack and locally scope board geometry data for easier access
        xRockerBottom, rRockerBottom  = geometryData['xRockerBottom'], geometryData['rRockerBottom']
        xRockerTop, rRockerTop        = geometryData['xRockerTop'], geometryData['rRockerTop']
        xBoardOutline, rBoardOutline  = geometryData['xBoardOutline'], geometryData['rBoardOutline']

        # Build the board wireframe by combining the board outline with the top and bottom rocker profiles
        # Pre-allocate the arrays for the board geometry as (numCrossSections, pointsPerSection)
        xBoard = np.zeros((self.resolution, int(self.resolution/2)))
        yBoard = np.zeros((self.resolution, int(self.resolution/2)))
        zBoard = np.zeros((self.resolution, int(self.resolution/2)))

        xBoundingBox = np.zeros((self.resolution, 4))
        yBoundingBox = np.zeros((self.resolution, 4))
        zBoundingBox = np.zeros((self.resolution, 4))

        # Build the bounding box for each cross section
        for i, x in enumerate(xRockerTop):

            # Bounding box is drawn starting in the bottom left and moving around the cross section in a
            # clockwise direction: bottom left --> top left --> top right --> bottom right
            xBoundingBox[i,:] = x # x coordinate is constant along the cross section
            yBoundingBox[i]   = np.array([-rBoardOutline[i], -rBoardOutline[i], rBoardOutline[i], rBoardOutline[i]])
            zBoundingBox[i]   = np.array([rRockerBottom[i], rRockerTop[i], rRockerTop[i], rRockerBottom[i]])

        # Debug plot of board wireframe bounding boxes
        if self.debugPlots:
            plt.style.use('dark_background')
            # Plot the 3d points of bounding box for each cross section
            ax = plt.figure().add_subplot(111, projection='3d')
            for i in range(self.resolution):
                ax.plot(xBoundingBox[i], yBoundingBox[i], zBoundingBox[i], '*c', alpha=0.1)
            plt.gca().set_aspect('equal')
            plt.show(block = False)

            debug = 1

        # Cross-section shape fractions (fraction of local board thickness H at each x-station).
        # Scaling with H ensures the crown and concavity taper naturally to zero at nose and tail where H→0.
        deckHeightFrac      = 0.10  # deck crown height as fraction of local thickness
        bottomConcavityFrac = 0.05  # bottom concavity depth as fraction of local thickness

        # Pre-allocate cross section arrays
        crossSectionX = np.zeros(self.numCrossSectionPoints)
        crossSectionY = np.zeros((self.resolution, int(self.resolution/2)))

        for i, x in enumerate(xRockerTop):

            halfWidth      = rBoardOutline[i]
            rBottom        = rRockerBottom[i]
            rTop           = rRockerTop[i]
            H              = rTop - rBottom                           # local board thickness

            deckCrown      = deckHeightFrac      * H                  # crown height at centerline
            concavityDepth = bottomConcavityFrac * H                  # concavity depth at centerline

            # Deck: parabolic crown: Z_deck(η) = rTop + deckCrown*(1-η²)
            # Horizontal tangent at η=0 (centerline), meets rail exactly at η=1 (Z = rTop)
            deckX[i]   = x
            deckY[i]   = halfWidth * eta
            deckZ[i]   = rTop    + deckCrown      * (1.0 - eta**2)

            # Bottom: parabolic concavity: Z_bottom(η) = rBottom - concavityDepth*(1-η²)
            # Horizontal tangent at η=0 (centerline groove), meets rail exactly at η=1 (Z = rBottom)
            bottomX[i] = x
            bottomY[i] = halfWidth * eta
            bottomZ[i] = rBottom - concavityDepth * (1.0 - eta**2)

        # Debug plot of the center board cross section
        if self.plotsEnabled:
            plt.style.use('dark_background')
            plt.plot()
            plt.gca().set_aspect('equal')
            plt.show(block = False)

            debug = 1

        return {
            'deckX'   : deckX,
            'deckY'   : deckY,
            'deckZ'   : deckZ,
            'bottomX' : bottomX,
            'bottomY' : bottomY,
            'bottomZ' : bottomZ,
        }

    def generateGeometry(self):
        
        '''
        
        Wrapper that calls required private methods to generate the surfboard geometry based on the initialized parameters.
        
        Sean Bowman [04/01/2026]
        
        '''
        
        # Validate input parameters
        self._validateInputs()

        # First: Rocker profiles (side view): returns (x, r, controlPoints)
        xRockerBottom, rRockerBottom, cpRockerBottom = self._generateRockerProfile(isTopOrBottom='bottom')
        xRockerTop,    rRockerTop,    cpRockerTop    = self._generateRockerProfile(isTopOrBottom='top')

        # Second: Nose-to-tail planform outline: returns (x, r, controlPoints)
        xBoardOutline, rBoardOutline, cpOutline = self._generateNoseToTailProfile()

        self.geometryData = {
            'xRockerBottom'        : xRockerBottom,
            'rRockerBottom'        : rRockerBottom,
            'xRockerTop'           : xRockerTop,
            'rRockerTop'           : rRockerTop,
            'xBoardOutline'        : xBoardOutline,
            'rBoardOutline'        : rBoardOutline,
            'cpOutline'            : cpOutline,            # {'tail': {...}, 'nose': {...}}
            'cpRockerBottom'       : cpRockerBottom,       # {'tail': {...}, 'nose': {...}}
            'cpRockerTop'          : cpRockerTop,          # {'tail': {...}, 'nose': {...}}
        }

        # Third: Generate axial board cross sections using board outline and rocker profiles
        crossSectionData = self._generateBoardCrossSections(self.geometryData)

        # Append the generated geometry to the geometry data dictionary for potential export or plotting
        self.geometryData.update(crossSectionData)

        # If plots are enabled
        if self.plotsEnabled:
            self.plotGeometry()

    #------------------------------------------------------------------------------#
    # -- Public Methods -- #
    #------------------------------------------------------------------------------#

    def plotGeometry(self) -> None:

        '''

        Renders a two-subplot figure showing the top-view planform outline and the
        side-view rocker profile, with the thruster reference image overlaid on each
        plot for direct shape comparison.

        Reference image axes are remapped from pixel space to physical coordinates (feet)
        using estimated real-board dimensions loaded from thrusterReferenceImageDimensions.json.
        Adjust the JSON values to dial in the alignment between the image and the generated outline.

        '''

        if self.geometryData is not None:
            xRockerBottom, rRockerBottom  = self.geometryData['xRockerBottom'], self.geometryData['rRockerBottom']
            xRockerTop, rRockerTop        = self.geometryData['xRockerTop'], self.geometryData['rRockerTop']
            xBoardOutline,  rBoardOutline = self.geometryData['xBoardOutline'],  self.geometryData['rBoardOutline']

        # Local working values in feet (read-only, do not mutate self)
        convertedValues = self._convertUnits(outputUnits = 'feet')
        totalLength     = convertedValues['totalLength']
        maxWidthFt      = convertedValues['maxWidth']

        # -- Load reference board physical dimensions for image axis calibration -- #

        assetDir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'referenceMaterial')
        jsonPath  = os.path.join(assetDir, 'thrusterReferenceImageDimensions.json')

        # Fall back to self dimensions if the JSON is missing
        refTotalLength = totalLength
        refMaxWidthFt  = maxWidthFt

        if os.path.exists(jsonPath):
            with open(jsonPath, 'r') as f:
                refDims = json.load(f)
            refTotalLength = refDims.get('totalLength', totalLength)
            refMaxWidthFt  = refDims.get('maxWidth', maxWidthFt * 12.0) / 12.0  # JSON in inches → feet

        # Top-view extent: x = board length, y centred at 0 = board half-width
        topExtent = [0, refTotalLength, -refMaxWidthFt / 2.0, refMaxWidthFt / 2.0]

        # -- Plot: shared x-axis, constrained layout -- #

        # Collapsible plot formatting container for board geometry plot
        if True:

            plt.style.use('dark_background')
            _, (axTop, axSide, axBottom) = plt.subplots(3, 1, sharex = True, constrained_layout = True, figsize = (12, 8))

            topViewPath    = os.path.join(assetDir, 'thrusterReferenceImage_pointedNose_roundTail_topView.png')
            sideViewPath   = os.path.join(assetDir, 'thrusterReferenceImage_pointedNose_roundTail_sideView.png')
            bottomViewPath = os.path.join(assetDir, 'thrusterReferenceImage_pointedNose_roundTail_bottomView.png')

            if os.path.exists(topViewPath):
                topView = plt.imread(topViewPath)
                axTop.imshow(topView, extent = topExtent, aspect = 'equal', alpha = 0.45)

            if os.path.exists(sideViewPath):
                sideView = plt.imread(sideViewPath)
                # y-extent derived from pixel aspect ratio: x-axis in feet, y in image-scaled units
                h, w       = sideView.shape[:2]
                sideYRange = refTotalLength * (h / w)
                sideExtent = [0, refTotalLength, 0, sideYRange]
                axSide.imshow(sideView, extent = sideExtent, aspect = 'equal', alpha = 0.45)

            if os.path.exists(bottomViewPath):
                bottomView = plt.imread(bottomViewPath)
                # Same extent mapping as top view: bottom board has the same planform outline
                h, w          = bottomView.shape[:2]
                bottomYRange  = refTotalLength * (h / w)
                bottomExtent  = [0, refTotalLength, -bottomYRange / 2.0, bottomYRange / 2.0]
                axBottom.imshow(bottomView, extent = bottomExtent, aspect = 'equal', alpha = 0.45)

            # -- Top view: full mirrored planform -- #
            axTop.plot(xBoardOutline,  rBoardOutline,  color = 'cyan', linewidth = 2)
            axTop.plot(xBoardOutline, -rBoardOutline,  color = 'cyan', linewidth = 2)
            axTop.fill_between(xBoardOutline, rBoardOutline, -rBoardOutline, alpha = 0.08, color = 'cyan')
            axTop.set_title('Surfboard Outline (Top View): reference overlay at 45% alpha')
            axTop.set_xlabel('Length (ft)')
            axTop.set_ylabel('Half-Width (ft)')
            axTop.grid(True, alpha = 0.2)

            # -- Side view: rocker profiles -- #
            axSide.plot(xRockerBottom, rRockerBottom, color = 'cyan', linewidth = 2)
            axSide.plot(xRockerTop, rRockerTop, color = 'cyan', linewidth = 2)
            axSide.axhline(0, color = 'white', linewidth = 0.5, linestyle = '--', alpha = 0.4)
            axSide.set_title('Rocker Profile (Side View): reference overlay at 45% alpha')
            axSide.set_xlabel('Length (ft)')
            axSide.set_ylabel('Rocker Height (ft)')
            axSide.grid(True, alpha = 0.2)

            # -- Bottom view: planform outline (same geometry as top view, different reference image) --
            axBottom.plot(xBoardOutline,  rBoardOutline,  color = 'cyan', linewidth = 2)
            axBottom.plot(xBoardOutline, -rBoardOutline,  color = 'cyan', linewidth = 2)
            axBottom.fill_between(xBoardOutline, rBoardOutline, -rBoardOutline, alpha = 0.08, color = 'cyan')
            axBottom.set_title('Surfboard Outline (Bottom View): reference overlay at 45% alpha')
            axBottom.set_xlabel('Length (ft)')
            axBottom.set_ylabel('Half-Width (ft)')
            axBottom.grid(True, alpha = 0.2)

            # -- Bezier control point overlay (only when showControlPoints is enabled) -- #

            if self.showControlPoints and self.geometryData is not None:

                cpColor   = '#FF9500'   # orange: distinct from cyan board curves
                cpKwArgs  = dict(color=cpColor, zorder=5)

                def drawControlPoints(ax: plt.Axes, cp: dict, mirrorY: bool = False) -> None:
                    '''
                    Draw the four Bezier control points and two tangent lines for one curve segment.
                    p1/p4 are the on-curve endpoints; p2/p3 are the off-curve interior handles.
                    mirrorY = True negates y-coordinates for the mirrored lower half of the outline.
                    '''
                    sign = -1.0 if mirrorY else 1.0

                    p1x, p1y = cp['p1'][0], sign * cp['p1'][1]
                    p2x, p2y = cp['p2'][0], sign * cp['p2'][1]
                    p3x, p3y = cp['p3'][0], sign * cp['p3'][1]
                    p4x, p4y = cp['p4'][0], sign * cp['p4'][1]

                    # Tangent lines: dashed connection from endpoint to its interior handle
                    ax.plot([p1x, p2x], [p1y, p2y], '--', linewidth=1.0, alpha=0.7, **cpKwArgs)
                    ax.plot([p3x, p4x], [p3y, p4y], '--', linewidth=1.0, alpha=0.7, **cpKwArgs)

                    # On-curve endpoints: filled circles
                    ax.scatter([p1x, p4x], [p1y, p4y], marker='o', s=40, zorder=6, color=cpColor)

                    # Interior handles: hollow squares
                    ax.scatter([p2x, p3x], [p2y, p3y], marker='s', s=35, zorder=6,
                            facecolors='none', edgecolors=cpColor, linewidths=1.5)

                # Outline control points on top and bottom views (both halves).
                # Each section ('tail', 'nose') now holds a list of two cp dicts: one per sub-segment.
                if 'cpOutline' in self.geometryData:
                    for segment in ('tail', 'nose'):
                        for cp in self.geometryData['cpOutline'][segment]:
                            for ax in (axTop, axBottom):
                                drawControlPoints(ax, cp, mirrorY=False)  # upper half
                                drawControlPoints(ax, cp, mirrorY=True)   # mirrored lower half

                # Rocker control points on side view: same list-of-two-dicts structure
                for cpKey in ('cpRockerBottom', 'cpRockerTop'):
                    if cpKey in self.geometryData:
                        for segment in ('tail', 'nose'):
                            for cp in self.geometryData[cpKey][segment]:
                                drawControlPoints(axSide, cp)

            plt.show(block = False)

        # -- Create board cross section view -- #

        # Collapsible plot formatting container for board cross section plot
        if True:

            plt.style.use('dark_background')

            plt.plot(xBoardOutline,  rBoardOutline,  color = 'cyan', linewidth = 2)

        # Create 3D plotly plot of the 3D geometry if the cross sections have been generated
        # if 'deckX' in self.geometryData:
        if False:
            deckX   = self.geometryData['deckX']
            deckY   = self.geometryData['deckY']
            deckZ   = self.geometryData['deckZ']
            bottomX = self.geometryData['bottomX']
            bottomY = self.geometryData['bottomY']
            bottomZ = self.geometryData['bottomZ']

            fig3d = go.Figure()

            # Render deck and bottom surfaces: each mirrored across centerline (Y and -Y) for full board
            for surfaceName, xArr, yArr, zArr, color in [
                ('Deck',   deckX,   deckY,   deckZ,   'rgba(100,200,255,0.7)'),
                ('Bottom', bottomX, bottomY, bottomZ, 'rgba(255,160,80,0.7)'),
            ]:
                colorscale = [[0, color], [1, color]]

                # Right half (positive Y)
                fig3d.add_trace(go.Surface(
                    x          = xArr,
                    y          = yArr,
                    z          = zArr,
                    colorscale = colorscale,
                    showscale  = False,
                    name       = f'{surfaceName} (right)',
                ))

                # Left half (mirror Y → negative)
                fig3d.add_trace(go.Surface(
                    x          = xArr,
                    y          = -yArr,
                    z          = zArr,
                    colorscale = colorscale,
                    showscale  = False,
                    name       = f'{surfaceName} (left)',
                    showlegend = False,
                ))

            fig3d.update_layout(
                title    = 'Surfboard 3D Geometry',
                scene    = dict(
                    xaxis_title = 'Length (ft)',
                    yaxis_title = 'Width (ft)',
                    zaxis_title = 'Height (ft)',
                    aspectmode  = 'data',
                ),
                template = 'plotly_dark',
            )
            fig3d.show()

        debug = 1
