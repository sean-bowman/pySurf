# -- Force Arrow Components -- #

'''

Manim mobjects for rendering force vector arrows with labels.

Creates styled arrows representing weight, buoyancy, planing lift,
and drag forces on a surfboard, with magnitude labels in Newtons.

Sean Bowman [02/04/2026]

'''

import math
import os
import sys

import numpy as np
from manim import (
    Arrow, Arrow3D, Text, VGroup,
    UP, DOWN, LEFT, RIGHT,
    ManimColor,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pysurf.Surfboard.SurfPhysics.hydrodynamics.protocols import PlaningState
from pysurf.Surfboard.SurfPhysics import constants as const
from pysurf.Surfboard.SurfAnimations.utils.manimTheme import (
    WEIGHT_COLOR, BUOYANCY_COLOR, DRAG_COLOR, LIFT_COLOR,
)

# Scale factor: Manim arrow length per Newton of force
# Calibrated so ~830N (typical weight) produces ~2.5 Manim unit arrow
FORCE_SCALE = 0.003

#--------------------------------------------------------------------#
# -- Single Force Arrow (2D) -- #
#--------------------------------------------------------------------#

def createForceArrow(
    origin: np.ndarray,
    direction: np.ndarray,
    magnitudeN: float,
    color: ManimColor,
    label: str,
    scaleFactor: float = FORCE_SCALE,
    maxLength: float = 3.0,
    strokeWidth: float = 5,
    fontSize: float = 18,
) -> VGroup:

    '''

    Create a 2D force arrow with a magnitude label.

    Parameters:
    -----------
    origin : np.ndarray
        Arrow start point [x, y, 0]
    direction : np.ndarray
        Unit direction vector [dx, dy, 0]
    magnitudeN : float
        Force magnitude in Newtons
    color : ManimColor
        Arrow and label color
    label : str
        Force name (e.g., 'Weight', 'Buoyancy')
    scaleFactor : float
        Manim units per Newton
    maxLength : float
        Maximum arrow length in Manim units
    strokeWidth : float
        Arrow stroke width
    fontSize : float
        Label font size

    Returns:
    --------
    VGroup : Arrow + text label

    '''
    # Compute arrow length (capped)
    arrowLength = min(abs(magnitudeN) * scaleFactor, maxLength)
    if arrowLength < 0.1:
        arrowLength = 0.1

    # Normalize direction
    dirNorm = np.array(direction, dtype=float)
    norm = np.linalg.norm(dirNorm)
    if norm > 0:
        dirNorm = dirNorm / norm

    endPoint = np.array(origin, dtype=float) + dirNorm * arrowLength

    arrow = Arrow(
        start=origin,
        end=endPoint,
        color=color,
        stroke_width=strokeWidth,
        buff=0,
        max_tip_length_to_length_ratio=0.2,
    )

    # Label at arrow tip
    labelText = Text(
        f'{label}: {magnitudeN:.0f} N',
        font_size=fontSize,
        color=color,
    )

    # Position label just past the arrow tip
    labelOffset = dirNorm * 0.3
    # Add perpendicular offset to avoid overlap with arrow
    perpendicular = np.array([-dirNorm[1], dirNorm[0], 0])
    labelOffset += perpendicular * 0.2
    labelText.move_to(endPoint + labelOffset)

    return VGroup(arrow, labelText)

#--------------------------------------------------------------------#
# -- Force Balance Group (2D) -- #
#--------------------------------------------------------------------#

def createForceBalance(
    planingState: PlaningState,
    totalWeightN: float,
    boardCenter: np.ndarray,
    trimAngleDeg: float = 0.0,
    scaleFactor: float = FORCE_SCALE,
    fontSize: float = 18,
) -> dict:

    '''

    Create a complete set of force arrows for a board in equilibrium.

    Returns a dict of VGroups keyed by force name, so individual
    forces can be animated independently.

    Parameters:
    -----------
    planingState : PlaningState
        Equilibrium state from ForceBalance.findEquilibrium()
    totalWeightN : float
        Total weight (board + rider) in Newtons
    boardCenter : np.ndarray
        Position of the board center [x, y, 0]
    trimAngleDeg : float
        Board trim angle in degrees
    scaleFactor : float
        Manim units per Newton
    fontSize : float
        Label font size

    Returns:
    --------
    dict : Keys 'weight', 'buoyancy', 'lift', 'drag' -> VGroup

    '''
    origin = np.array(boardCenter, dtype=float)
    trimRad = math.radians(trimAngleDeg)

    forces = {}

    # Weight: always straight down
    forces['weight'] = createForceArrow(
        origin=origin,
        direction=np.array([0, -1, 0]),
        magnitudeN=totalWeightN,
        color=WEIGHT_COLOR,
        label='Weight',
        scaleFactor=scaleFactor,
        fontSize=fontSize,
    )

    # Buoyancy: straight up, applied near center
    buoyancyN = totalWeightN - planingState.liftForceN
    if buoyancyN < 0:
        buoyancyN = 0
    forces['buoyancy'] = createForceArrow(
        origin=origin + np.array([0, -0.05, 0]),
        direction=np.array([0, 1, 0]),
        magnitudeN=buoyancyN,
        color=BUOYANCY_COLOR,
        label='Buoyancy',
        scaleFactor=scaleFactor,
        fontSize=fontSize,
    )

    # Planing lift: perpendicular to board bottom (angled by trim)
    liftDir = np.array([
        -math.sin(trimRad),
        math.cos(trimRad),
        0,
    ])
    forces['lift'] = createForceArrow(
        origin=origin + np.array([0.05, 0, 0]),
        direction=liftDir,
        magnitudeN=planingState.liftForceN,
        color=LIFT_COLOR,
        label='Lift',
        scaleFactor=scaleFactor,
        fontSize=fontSize,
    )

    # Drag: opposite to motion direction (leftward = negative X)
    forces['drag'] = createForceArrow(
        origin=origin + np.array([0, 0.05, 0]),
        direction=np.array([-1, 0, 0]),
        magnitudeN=planingState.dragForceN,
        color=DRAG_COLOR,
        label='Drag',
        scaleFactor=scaleFactor,
        fontSize=fontSize,
    )

    return forces

#--------------------------------------------------------------------#
# -- 3D Force Arrows -- #
#--------------------------------------------------------------------#

def createForceArrow3D(
    origin: np.ndarray,
    direction: np.ndarray,
    magnitudeN: float,
    color: ManimColor,
    scaleFactor: float = FORCE_SCALE,
    maxLength: float = 3.0,
) -> Arrow3D:

    '''

    Create a 3D force arrow for ThreeDScene.

    Parameters:
    -----------
    origin : np.ndarray
        Arrow start point [x, y, z]
    direction : np.ndarray
        Unit direction vector [dx, dy, dz]
    magnitudeN : float
        Force magnitude in Newtons
    color : ManimColor
        Arrow color
    scaleFactor : float
        Manim units per Newton
    maxLength : float
        Maximum arrow length

    Returns:
    --------
    Arrow3D : 3D arrow

    '''
    arrowLength = min(abs(magnitudeN) * scaleFactor, maxLength)
    if arrowLength < 0.1:
        arrowLength = 0.1

    dirNorm = np.array(direction, dtype=float)
    norm = np.linalg.norm(dirNorm)
    if norm > 0:
        dirNorm = dirNorm / norm

    endPoint = np.array(origin, dtype=float) + dirNorm * arrowLength

    arrow = Arrow3D(
        start=origin,
        end=endPoint,
        color=color,
    )
    return arrow

def createForceBalance3D(
    planingState: PlaningState,
    totalWeightN: float,
    boardCenter: np.ndarray,
    scaleFactor: float = FORCE_SCALE,
) -> dict:

    '''

    Create 3D force arrows for ThreeDScene.

    Parameters:
    -----------
    planingState : PlaningState
        Equilibrium state
    totalWeightN : float
        Total weight in Newtons
    boardCenter : np.ndarray
        Board center [x, y, z]
    scaleFactor : float
        Manim units per Newton

    Returns:
    --------
    dict : Keys 'weight', 'buoyancy', 'lift', 'drag' -> Arrow3D

    '''
    origin = np.array(boardCenter, dtype=float)
    forces = {}

    # Weight: down (negative Z in 3D)
    forces['weight'] = createForceArrow3D(
        origin=origin,
        direction=np.array([0, 0, -1]),
        magnitudeN=totalWeightN,
        color=WEIGHT_COLOR,
        scaleFactor=scaleFactor,
    )

    # Buoyancy: up (positive Z)
    buoyancyN = totalWeightN - planingState.liftForceN
    if buoyancyN < 0:
        buoyancyN = 0
    forces['buoyancy'] = createForceArrow3D(
        origin=origin,
        direction=np.array([0, 0, 1]),
        magnitudeN=buoyancyN,
        color=BUOYANCY_COLOR,
        scaleFactor=scaleFactor,
    )

    # Lift: up (positive Z)
    forces['lift'] = createForceArrow3D(
        origin=origin,
        direction=np.array([0, 0, 1]),
        magnitudeN=planingState.liftForceN,
        color=LIFT_COLOR,
        scaleFactor=scaleFactor,
    )

    # Drag: backward (negative X)
    forces['drag'] = createForceArrow3D(
        origin=origin,
        direction=np.array([-1, 0, 0]),
        magnitudeN=planingState.dragForceN,
        color=DRAG_COLOR,
        scaleFactor=scaleFactor,
    )

    return forces
