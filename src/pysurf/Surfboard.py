
# -- Class Definition for the Generation of Parametric Surfboard Geometry -- #

'''

This module defines the Surfboard class, which encapsulates the geometry and parameters of a surfboard.
It provides methods for generating the surfboard geometry based on a set of parameters, as well as utilities
for exporting the geometry to STL format for use in CFD simulations or 3D printing.

The board is built from the three views a shaper works from. The planform and the deck and bottom
rockers are C1-continuous cubic Bezier sections. Each cross section spans the planform's half-width
and the rockers' thickness at its station: a deck contour and a bottom contour meet at the rail apex,
the widest point of the rail, halfway up the local thickness. Measured from the apex, each contour
follows (z / b)^2 + |y / a|^p = 1, elliptic where it turns vertical at the rail and flatter across
the board the larger its exponent p. The exponents vary along the board.

Validation:
    Not validated. The planform and rocker are matched by eye to photographs of a reference
    thruster (assets/referenceMaterial). The cross-section exponents are calibrated to the
    sections of the reference thruster's STL (assets/referenceMaterial/referenceThruster.stl),
    fitted station by station with the rail apex at half the thickness. Scaled to each station's
    half-width and thickness, the generated sections lie within 0.9 percent of the local
    thickness (rms) of the STL's from 0.2 to 0.9 of the length, the worst point within 2.1
    percent, and within 2.0 percent rms (worst point 3.7 percent) toward the tips;
    tests/testSurfboardSections.py checks it. That is a calibration to the board the exponents
    came from: no independent section data are in the repository, and a board of another type
    takes the thruster's sections.

Sean Bowman [04/01/2026]

'''

# Standard library imports
import os

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# Local imports
from pysurf import exports, figures, palette
from pysurf.errors import GeometryError, InvalidInputError
from pysurf.geometryTools import bezierCurve, bezierSubdivide, bezierEval
from pysurf.units import convert
from pysurf.validation import applyRules, arrayRule, choiceRule, numericRule

# Inputs a board must satisfy before any geometry is generated, from the most basic to the most
# specific. Lengths are in the units the constructor takes them in.
inputRules = (
    numericRule('totalLength',       'Board length',       units = 'ft', minimum = 0.0),
    numericRule('maxWidth',          'Maximum width',      units = 'in', minimum = 0.0),
    numericRule('midpointThickness', 'Midpoint thickness', units = 'in', minimum = 0.0),
    choiceRule('noseShape',     'Nose shape',     choices = ('point', 'round', 'fish')),
    choiceRule('tailShape',     'Tail shape',     choices = ('squash', 'round', 'pin', 'fish')),
    choiceRule('rockerProfile', 'Rocker profile', choices = ('flat', 'medium', 'aggressive')),
    choiceRule('finConfig',     'Fin setup',      choices = ('thruster', 'quad', 'single', 'twin')),
)

# The cross-section shape settings, which the constructor presets rather than takes
sectionRules = (
    arrayRule('sectionStations', 'Cross-section stations', minimumLength = 2),
    arrayRule('deckFlatness',    'Deck flatness',          sameLengthAs = 'sectionStations'),
    arrayRule('bottomFlatness',  'Bottom flatness',        sameLengthAs = 'sectionStations'),
    numericRule('railApexFrac',  'Rail apex height',       minimum = 0.0, maximum = 1.0),
)

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
        numCrossSectionPoints    : int: lateral points per deck and bottom contour (default 200)

        Cross-section shape (overridable; fitted to the reference thruster STL):
        sectionStations          : list: stations along the board, fractions of the length from the tail
        deckFlatness             : list: deck exponent p at each station (2 = elliptic, larger = flatter)
        bottomFlatness           : list: bottom exponent p at each station
        railApexFrac             : float: rail apex height above the bottom, fraction of local thickness

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

        # -- Cross Sections -- #

        # Exponents of the deck and bottom contours, (z / b)^2 + |y / a|^p = 1 from the rail apex,
        # fitted to the sections of the reference thruster STL with the apex at half the thickness.
        # The deck is flattest just ahead of the tail and through the middle, where the STL's deck
        # is broad and its rails full; it domes toward the nose. The bottom is flat with a tight
        # edge in the tail and nose and rolls softly into the rail just ahead of the middle.
        self.sectionStations = [0.0,  0.05, 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  0.95, 1.0]
        self.deckFlatness    = [4.50, 4.50, 7.16, 5.69, 5.03, 5.40, 6.02, 6.44, 6.32, 4.85, 3.31, 4.90, 9.22]
        self.bottomFlatness  = [4.24, 4.24, 5.32, 5.38, 4.77, 4.16, 3.49, 2.90, 2.63, 3.27, 5.26, 4.56, 2.52]
        self.railApexFrac    = 0.5

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

        Validates the input parameters against the module's rule table: each must be present, of
        the correct type, and inside the range it may take.

        Raises:
            InvalidInputError: If any of the input parameters are invalid. It is also a ValueError.

        '''

        applyRules(self, inputRules)
        applyRules(self, sectionRules)

        stations = np.asarray(self.sectionStations, dtype = float)
        if stations[0] != 0.0 or stations[-1] != 1.0 or np.any(np.diff(stations) <= 0.0):
            raise InvalidInputError('Cross-section stations must rise from 0 at the tail to 1 at the nose.',
                                    parameterName = 'sectionStations', value = list(stations),
                                    validRange = 'increasing, from 0 to 1')
        for name in ('deckFlatness', 'bottomFlatness'):
            exponents = np.asarray(getattr(self, name), dtype = float)
            if np.any(exponents < 1.0):
                raise InvalidInputError(f'{name} below 1 pinches the section into a cusp at the centerline.',
                                        parameterName = name, value = float(exponents.min()), validRange = '>= 1')

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
                    'maxWidth': convert(self.maxWidth, 'in', 'ft'),
                    'midpointThickness': convert(self.midpointThickness, 'in', 'ft')
                }

            case 'inches':

                convertedValues = {
                    'totalLength': convert(self.totalLength, 'ft', 'in'),
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
        baseThicknessOff = convert(self.midpointThickness, 'in', 'ft')

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
            figures.applyMplStyle()
            plt.figure(figsize=(10, 4))
            plt.plot(xRocker, rRocker, color=palette.colors()['accentText'], label=f'Rocker ({isTopOrBottom})')
            plt.title('Surfboard Rocker Profile (Side View)')
            plt.xlabel('Length (ft)')
            plt.ylabel('Rocker Height (ft)')
            plt.gca().set_aspect('auto')
            figures.showFigure(plt.gcf())

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
            figures.applyMplStyle()
            plt.figure(figsize=(10, 4))
            plt.plot(xBoard,  rBoard, color=palette.colors()['accentText'], label='outline')
            plt.plot(xBoard, -rBoard, color=palette.colors()['accentText'])
            plt.title('Surfboard Nose-to-Tail Profile (Top View)')
            plt.xlabel('Length (ft)')
            plt.ylabel('Half-Width (ft)')
            plt.gca().set_aspect('equal')
            figures.showFigure(plt.gcf())

        return xBoard, rBoard, controlPoints

    def _generateBoardCrossSections(self, geometryData: dict) -> dict:

        '''

        Generates the axial cross sections of the surfboard from the planform outline and the deck
        and bottom rocker profiles.

        Each station's section spans the outline's half-width and the thickness between the two
        rockers: the deck meets the deck rocker and the bottom meets the bottom rocker at the
        centerline, and the two contours meet at the rail apex, at the outline's half-width and
        railApexFrac of the thickness above the bottom (`crossSectionShape`). The sections run from
        the tail to the nose (x = 0 to x = totalLength), one per rocker station.

        The coordinate frame for the 3D surfboard is as follows:
            X (i-hat): Runs from tail (0) to nose (totalLength) along the centerline of the board
            Y (j-hat): Runs from the centerline to the right rail (positive) and left rail (negative)
            Z (k-hat): Height above the bottom rocker's flat spot

        Each section is drawn in the YZ plane from the centerline to the right rail; the left half
        mirrors it.

        The board cross sections in general look like:
                    ________________
                   /                \
        Left Rail (                  ) Right Rail
                   \________________/

        Returns:
        --------
        dict : 'deckX', 'deckY', 'deckZ', 'bottomX', 'bottomY', 'bottomZ', each shaped
            (stations, numCrossSectionPoints) [ft], from the centerline to the rail apex

        Raises:
        -------
        GeometryError : If the deck rocker dips below the bottom rocker anywhere.

        '''

        x            = np.asarray(geometryData['xRockerTop'], dtype = float)
        top          = np.asarray(geometryData['rRockerTop'], dtype = float)
        bottom       = np.interp(x, geometryData['xRockerBottom'], geometryData['rRockerBottom'])
        halfWidth    = np.interp(x, geometryData['xBoardOutline'], geometryData['rBoardOutline'])
        thickness    = top - bottom
        if np.any(thickness < 0.0):
            raise GeometryError('The deck rocker dips below the bottom rocker, so the board has no thickness there.')

        # Lateral positions from the centerline to the rail apex, closing in toward the rail where
        # the contours turn vertical
        eta = np.sin(0.5 * np.pi * np.linspace(0.0, 1.0, self.numCrossSectionPoints))
        deckShare, bottomShare = self.crossSectionShape(x / x[-1], eta)

        deckX   = np.repeat(x[:, None], eta.size, axis = 1)
        deckY   = halfWidth[:, None] * eta[None, :]
        deckZ   = bottom[:, None] + thickness[:, None] * deckShare
        bottomZ = bottom[:, None] + thickness[:, None] * bottomShare

        return {
            'deckX'   : deckX,
            'deckY'   : deckY,
            'deckZ'   : deckZ,
            'bottomX' : deckX.copy(),
            'bottomY' : deckY.copy(),
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

    def crossSectionShape(self, stationFraction, eta) -> tuple[np.ndarray, np.ndarray]:

        '''

        Deck and bottom heights of the cross section, as fractions of the local thickness above the
        bottom centerline.

        From the rail apex, at railApexFrac of the thickness, the deck rises and the bottom falls
        to the centerline along (z / b)^2 + |eta|^p = 1, with b the apex's height below the deck or
        above the bottom and p the deck or bottom flatness interpolated to the station (monotone
        cubic through the table). At the centerline the deck share is 1 and the bottom share 0;
        both reach the apex height at the rail with a vertical tangent.

        Parameters:
        -----------
        stationFraction : float or np.ndarray
            Station along the board, as a fraction of the length from the tail
        eta : np.ndarray
            Lateral position as a fraction of the local half-width, 0 at the centerline and 1 at
            the rail apex

        Returns:
        --------
        tuple : (deck share, bottom share), shaped like eta for a single station, or (stations,
            len(eta)) for an array of them

        '''

        stations = np.asarray(self.sectionStations, dtype = float)
        fraction = np.clip(np.atleast_1d(np.asarray(stationFraction, dtype = float)), 0.0, 1.0)
        deckP    = PchipInterpolator(stations, np.asarray(self.deckFlatness, dtype = float))(fraction)[:, None]
        bottomP  = PchipInterpolator(stations, np.asarray(self.bottomFlatness, dtype = float))(fraction)[:, None]
        eta      = np.clip(np.asarray(eta, dtype = float), 0.0, 1.0)[None, :]
        apex     = self.railApexFrac

        deck   = apex + (1.0 - apex) * np.sqrt(np.clip(1.0 - eta**deckP, 0.0, 1.0))
        bottom = apex * (1.0 - np.sqrt(np.clip(1.0 - eta**bottomP, 0.0, 1.0)))

        if np.ndim(stationFraction) == 0:
            return deck[0], bottom[0]
        return deck, bottom

    def exportGeometry(self, filePath: str, units: str = 'in') -> str:

        '''

        Writes the board to a binary STL file as one closed surface. Each station's section becomes
        a loop out along the right deck to the rail, back along the right bottom, out along the left
        bottom and back along the left deck, and the loops join station to station from tail to
        nose. An end with width, such as a squash tail or the nose tip's last fraction of an inch,
        is closed with a fan of triangles; an end of zero width folds onto the line between deck and
        bottom and closes without one. Facet normals point out of the board.

        Parameters:
        -----------
        filePath : str
            Output path; '.stl' is appended when it has no extension, and missing folders are made
        units : str
            Length unit the file is written in: 'in', 'mm' or 'ft' (default 'in')

        Returns:
        --------
        str : The path written

        Raises:
        -------
        GeometryError : If the board has no cross sections yet.

        '''

        data = self.geometryData
        if data is None or 'deckZ' not in data:
            raise GeometryError('The board has no cross sections to export: call generateGeometry() first.')

        scale = convert(1.0, 'ft', units)
        deckY, deckZ = np.asarray(data['deckY']), np.asarray(data['deckZ'])
        bottomY, bottomZ = np.asarray(data['bottomY']), np.asarray(data['bottomZ'])

        # One loop per station from the deck centerline, each point once: the rail apexes and the
        # bottom centerline are shared by the contours that meet there
        loopY = np.concatenate([deckY, bottomY[:, -2::-1], -bottomY[:, 1:-1], -deckY[:, :0:-1]], axis = 1)
        loopZ = np.concatenate([deckZ, bottomZ[:, -2::-1], bottomZ[:, 1:-1], deckZ[:, :0:-1]], axis = 1)
        loopY[loopY == 0.0] = 0.0       # no negative zeros, so both halves' centerline points coincide
        stations, count = loopY.shape
        loopX = np.repeat(np.asarray(data['deckX'])[:, :1], count, axis = 1)
        vertices = scale * np.stack([loopX, loopY, loopZ], axis = -1).reshape(-1, 3)

        # Two triangles between each pair of neighboring loops
        i, j = np.meshgrid(np.arange(stations - 1), np.arange(count), indexing = 'ij')
        here, ahead = i * count + j, i * count + (j + 1) % count
        faces = [np.stack([here, here + count, ahead + count], axis = -1).reshape(-1, 3),
                 np.stack([here, ahead + count, ahead], axis = -1).reshape(-1, 3)]

        # Fans over the ends that have width, from a point midway up the end's thickness
        for station, flip in ((0, False), (stations - 1, True)):
            if deckY[station, -1] <= 0.0:
                continue
            center = scale * np.array([loopX[station, 0], 0.0, 0.5 * (deckZ[station, 0] + bottomZ[station, 0])])
            vertices = np.vstack([vertices, center])
            ring = station * count + np.arange(count)
            fan = np.stack([np.full(count, len(vertices) - 1), ring, np.roll(ring, -1)], axis = -1)
            faces.append(fan[:, ::-1] if flip else fan)

        if not os.path.splitext(filePath)[1]:
            filePath += '.stl'
        folder = os.path.dirname(filePath)
        if folder:
            os.makedirs(folder, exist_ok = True)
        exports.writeStl(filePath, vertices, np.vstack(faces))

        return filePath

    def plotGeometry(self, show3D: bool = False) -> None:

        '''

        Renders the planform, rocker and bottom views with the thruster reference images overlaid,
        then the station cross sections and, on request, the 3D surfaces once the board has them.
        Drawing lives in pysurf.figures; this method shows what it draws. The interactive HTML
        companions come from pysurf.figures.exportInteractiveFigures.

        Parameters:
        -----------
        show3D : bool
            Also open the 3D deck and bottom surfaces in a browser, when cross sections exist
            (default False).

        Raises:
        -------
        GeometryError : If the outline and rocker have not been generated.

        '''

        # Planform, rocker and bottom views
        profiles = figures.drawBoardProfileFigure(self)
        if profiles is None:
            raise GeometryError('The board has no profiles to draw: call generateGeometry() first.')
        figures.showFigure(profiles)

        # Station cross sections, once the cross-section step has added them to the geometry data
        sections = figures.drawCrossSectionFigure(self)
        if sections is not None:
            figures.showFigure(sections)

        # 3D deck and bottom surfaces
        if show3D:
            surface = figures.boardSurfaceFigure(self)
            if surface is not None:
                figures.showFigure(surface)
