# -- Board Profile Components -- #

'''

Manim mobjects for rendering surfboard profiles in 2D side-view
and 3D planform outline.

Queries BoardGeometry for rocker, deck, and bottom heights to build
closed polygons representing the board cross-section profile.

Sean Bowman [02/04/2026]

'''

import math
import os
import sys

import numpy as np
from manim import VMobject, Polygon, VGroup, np as mnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pysurf.Surfboard.SurfPhysics.geometry.board import BoardGeometry
from pysurf.Surfboard.SurfPhysics.geometry.parameters import SurfboardParameters
from pysurf.Surfboard.SurfPhysics.waves.linearWaveTheory import LinearWaveTheory
from pysurf.Surfboard.SurfPhysics.waves.waveConditions import WaveConditions
from pysurf.Surfboard.SurfAnimations.utils.manimTheme import BOARD_COLOR

#--------------------------------------------------------------------#
# -- 2D Side-View Profile -- #
#--------------------------------------------------------------------#

def createBoardProfile(
    board: BoardGeometry,
    params: SurfboardParameters,
    color=None,
    fillOpacity: float = 0.8,
    strokeWidth: float = 1.5,
    nStations: int = 200,
) -> Polygon:

    '''

    Create a 2D side-view board profile (centerline cross-section).

    The profile shows the board from the side: deck on top, bottom below,
    with rocker curvature defining the overall shape. Units are meters
    (1 Manim unit = 1 meter).

    Parameters:
    -----------
    board : BoardGeometry
        Board geometry instance
    params : SurfboardParameters
        Board parameters
    color : ManimColor
        Fill/stroke color (defaults to BOARD_COLOR)
    fillOpacity : float
        Fill transparency
    strokeWidth : float
        Outline stroke width
    nStations : int
        Number of sample stations along the length

    Returns:
    --------
    Polygon : Closed board profile shape centered at origin

    '''
    if color is None:
        color = BOARD_COLOR

    lengthM = board.getLengthM()
    tVals = np.linspace(0, 1, nStations)

    # Build deck (top) and bottom curves along centerline
    deckPoints = []
    bottomPoints = []

    for t in tVals:
        x = t * lengthM
        rocker = board.getRockerHeightM(t)
        deck = rocker + board.getDeckHeightM(t, 0.0)
        bottom = rocker + board.getBottomHeightM(t, 0.0)
        deckPoints.append([x, deck, 0])
        bottomPoints.append([x, bottom, 0])

    # Close the polygon: deck forward, then bottom reversed
    allPoints = deckPoints + list(reversed(bottomPoints))
    vertices = [np.array(p) for p in allPoints]

    boardShape = Polygon(
        *vertices,
        fill_color=color,
        fill_opacity=fillOpacity,
        stroke_color=color,
        stroke_width=strokeWidth,
    )

    # Center the board at the origin
    boardShape.move_to(np.array([0, 0, 0]))

    return boardShape

def getBoardCenterOffset(board: BoardGeometry) -> np.ndarray:

    '''

    Get the center point of the board profile in local coordinates.

    Returns:
    --------
    np.ndarray : [centerX, centerY, 0] in meters

    '''
    lengthM = board.getLengthM()
    # Center at 40% from nose (thickest point / wide point)
    centerT = 0.4
    centerX = centerT * lengthM
    rocker = board.getRockerHeightM(centerT)
    deck = rocker + board.getDeckHeightM(centerT, 0.0)
    bottom = rocker + board.getBottomHeightM(centerT, 0.0)
    centerY = (deck + bottom) / 2.0
    return np.array([centerX, centerY, 0])

def positionBoardOnWave(
    boardMobject: VMobject,
    board: BoardGeometry,
    waveTheory: LinearWaveTheory,
    wc: WaveConditions,
    boardX: float,
    t: float,
    trimAngleDeg: float = 0.0,
    draftM: float = 0.0,
) -> VMobject:

    '''

    Position a board profile on the wave surface.

    Places the board at position boardX on the wave, rotated by
    the trim angle, with the bottom at the wave surface minus draft.

    Parameters:
    -----------
    boardMobject : VMobject
        Board profile mobject (will be moved/rotated in-place)
    board : BoardGeometry
        Board geometry for computing offsets
    waveTheory : LinearWaveTheory
        Wave theory instance
    wc : WaveConditions
        Wave conditions
    boardX : float
        Board center X position on the wave (meters)
    t : float
        Time in seconds
    trimAngleDeg : float
        Board trim angle in degrees (nose up = positive)
    draftM : float
        Draft depth in meters (how far bottom sits below surface)

    Returns:
    --------
    VMobject : The board mobject (modified in-place)

    '''
    # Wave elevation at board position
    eta = waveTheory.surfaceElevation(boardX, t, wc)

    # Reset board to origin before repositioning
    boardMobject.move_to(np.array([0, 0, 0]))

    # Apply trim angle (rotate about center, nose up = counter-clockwise)
    trimRad = math.radians(trimAngleDeg)
    boardMobject.rotate(trimRad, about_point=boardMobject.get_center())

    # Position: board bottom sits at wave surface minus draft
    # Move board center to (boardX, eta - draftM + halfThickness)
    halfThickness = board.getThicknessM(0.4) / 2.0
    targetY = eta - draftM + halfThickness
    boardMobject.move_to(np.array([boardX, targetY, 0]))

    return boardMobject

#--------------------------------------------------------------------#
# -- 3D Planform Outline -- #
#--------------------------------------------------------------------#

def createBoardOutline3D(
    board: BoardGeometry,
    params: SurfboardParameters,
    color=None,
    strokeWidth: float = 2.0,
    fillOpacity: float = 0.6,
    nStations: int = 200,
) -> Polygon:

    '''

    Create a 3D planform outline for ThreeDScene.

    Shows the board from above as a filled shape lying on the XY plane
    (X = length, Y = width). The Z coordinate follows the rocker profile.

    Parameters:
    -----------
    board : BoardGeometry
        Board geometry instance
    params : SurfboardParameters
        Board parameters
    color : ManimColor
        Color for the outline
    strokeWidth : float
        Outline stroke width
    fillOpacity : float
        Fill opacity
    nStations : int
        Number of sample stations

    Returns:
    --------
    Polygon : 3D board planform shape

    '''
    if color is None:
        color = BOARD_COLOR

    lengthM = board.getLengthM()
    tVals = np.linspace(0, 1, nStations)

    # Positive-Y side (right rail)
    rightPoints = []
    for t in tVals:
        x = t * lengthM
        y = board.getHalfWidthM(t)
        z = board.getRockerHeightM(t)
        rightPoints.append([x, y, z])

    # Negative-Y side (left rail), reversed
    leftPoints = []
    for t in reversed(tVals):
        x = t * lengthM
        y = -board.getHalfWidthM(t)
        z = board.getRockerHeightM(t)
        leftPoints.append([x, y, z])

    allPoints = rightPoints + leftPoints
    vertices = [np.array(p) for p in allPoints]

    boardOutline = Polygon(
        *vertices,
        fill_color=color,
        fill_opacity=fillOpacity,
        stroke_color=color,
        stroke_width=strokeWidth,
    )

    # Center at origin
    boardOutline.move_to(np.array([0, 0, 0]))

    return boardOutline

def positionBoardOnWave3D(
    boardMobject: VMobject,
    board: BoardGeometry,
    waveTheory: LinearWaveTheory,
    wc: WaveConditions,
    boardX: float,
    t: float,
    draftM: float = 0.0,
) -> VMobject:

    '''

    Position a 3D board outline on the wave surface.

    Parameters:
    -----------
    boardMobject : VMobject
        3D board planform mobject
    board : BoardGeometry
        Board geometry
    waveTheory : LinearWaveTheory
        Wave theory instance
    wc : WaveConditions
        Wave conditions
    boardX : float
        Board center X position on the wave
    t : float
        Time in seconds
    draftM : float
        Draft depth in meters

    Returns:
    --------
    VMobject : Modified board mobject

    '''
    eta = waveTheory.surfaceElevation(boardX, t, wc)
    boardMobject.move_to(np.array([boardX, 0, eta - draftM]))
    return boardMobject
