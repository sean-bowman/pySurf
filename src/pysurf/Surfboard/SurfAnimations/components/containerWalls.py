# -- Container Wall Components -- #

'''

Reusable Manim mobjects for rendering rectangular container walls.

Provides functions to create open-top tank outlines for
SPH sloshing tank visualization.

Sean Bowman [02/05/2026]

'''

import os
import sys

import numpy as np
from manim import VGroup, Line

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pysurf.Surfboard.SurfAnimations.utils.manimTheme import WHITE

def createContainerWalls(
    width: float,
    height: float,
    scale: float = 1.0,
    offset: np.ndarray | None = None,
    color = None,
    strokeWidth: float = 3.0,
) -> VGroup:

    '''

    Create container walls (open-top rectangle) in Manim coordinates.

    Draws three lines: left wall, bottom, and right wall.
    The top is left open for the free surface.

    Parameters:
    -----------
    width : float
        Container width in simulation units [m]
    height : float
        Container height in simulation units [m]
    scale : float
        Scale factor: Manim units per simulation meter
    offset : np.ndarray | None
        Translation offset in Manim units (default: None)
    color : ManimColor
        Wall color (default: WHITE)
    strokeWidth : float
        Line thickness in Manim units

    Returns:
    --------
    VGroup : Left wall + bottom + right wall

    '''
    color = color or WHITE
    off = offset if offset is not None else np.array([0.0, 0.0])

    # Container corners in Manim coordinates
    botLeft = np.array([0.0, 0.0]) * scale + off
    botRight = np.array([width, 0.0]) * scale + off
    topLeft = np.array([0.0, height]) * scale + off
    topRight = np.array([width, height]) * scale + off

    # Convert 2D to 3D for Manim
    def to3d(p: np.ndarray) -> np.ndarray:
        return np.array([p[0], p[1], 0.0])

    leftWall = Line(
        start=to3d(topLeft), end=to3d(botLeft),
        color=color, stroke_width=strokeWidth,
    )
    bottom = Line(
        start=to3d(botLeft), end=to3d(botRight),
        color=color, stroke_width=strokeWidth,
    )
    rightWall = Line(
        start=to3d(botRight), end=to3d(topRight),
        color=color, stroke_width=strokeWidth,
    )

    return VGroup(leftWall, bottom, rightWall)
