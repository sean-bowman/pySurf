# -- Wave Surface Components -- #

'''

Reusable Manim mobjects for rendering wave surfaces in 2D and 3D.

Wraps LinearWaveTheory to produce animated wave lines, water fills,
and 3D parametric surfaces for use across all animation scenes.

Sean Bowman [02/04/2026]

'''

import os
import sys

import numpy as np
from manim import (
    VMobject, Polygon, Surface, VGroup,
    ORIGIN, UP, RIGHT, DOWN,
    interpolate_color,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pysurf.Surfboard.SurfPhysics.waves.linearWaveTheory import LinearWaveTheory
from pysurf.Surfboard.SurfPhysics.waves.waveConditions import WaveConditions
from pysurf.Surfboard.SurfAnimations.utils.manimTheme import WAVE_COLOR, WATER_FILL

#--------------------------------------------------------------------#
# -- 2D Wave Line -- #
#--------------------------------------------------------------------#

def createWaveLine(
    waveTheory: LinearWaveTheory,
    wc: WaveConditions,
    t: float,
    xMin: float = -7.0,
    xMax: float = 7.0,
    nPoints: int = 200,
    color = None,
    strokeWidth: float = 2.5,
) -> VMobject:

    '''

    Create a 2D wave surface line from LinearWaveTheory.

    Parameters:
    -----------
    waveTheory : LinearWaveTheory
        Wave theory instance
    wc : WaveConditions
        Wave conditions (height, period, depth)
    t : float
        Time in seconds
    xMin : float
        Left edge of wave domain (Manim units = meters)
    xMax : float
        Right edge of wave domain
    nPoints : int
        Number of sample points along the wave
    color : ManimColor
        Line color (defaults to WAVE_COLOR)
    strokeWidth : float
        Line stroke width

    Returns:
    --------
    VMobject : Wave polyline in the XY plane

    '''
    if color is None:
        color = WAVE_COLOR

    xVals = np.linspace(xMin, xMax, nPoints)
    points = []
    for x in xVals:
        eta = waveTheory.surfaceElevation(x, t, wc)
        points.append([x, eta, 0])

    wave = VMobject()
    wave.set_points_as_corners(points)
    wave.set_stroke(color=color, width=strokeWidth)
    return wave

def updateWaveLine(
    wave: VMobject,
    waveTheory: LinearWaveTheory,
    wc: WaveConditions,
    t: float,
    xMin: float = -7.0,
    xMax: float = 7.0,
    nPoints: int = 200,
) -> None:

    '''

    Update an existing wave VMobject to a new time value.

    Modifies the wave in-place by recomputing corner points.

    Parameters:
    -----------
    wave : VMobject
        Existing wave line to update
    waveTheory : LinearWaveTheory
        Wave theory instance
    wc : WaveConditions
        Wave conditions
    t : float
        New time value in seconds
    xMin : float
        Left edge of wave domain
    xMax : float
        Right edge of wave domain
    nPoints : int
        Number of sample points

    '''
    xVals = np.linspace(xMin, xMax, nPoints)
    points = []
    for x in xVals:
        eta = waveTheory.surfaceElevation(x, t, wc)
        points.append([x, eta, 0])

    wave.set_points_as_corners(points)

#--------------------------------------------------------------------#
# -- Water Fill -- #
#--------------------------------------------------------------------#

def createWaterFill(
    waveTheory: LinearWaveTheory,
    wc: WaveConditions,
    t: float,
    xMin: float = -7.0,
    xMax: float = 7.0,
    bottomY: float = -4.0,
    nPoints: int = 200,
    color = None,
    opacity: float = 0.3,
) -> Polygon:

    '''

    Create a filled polygon below the wave surface.

    Parameters:
    -----------
    waveTheory : LinearWaveTheory
        Wave theory instance
    wc : WaveConditions
        Wave conditions
    t : float
        Time in seconds
    xMin : float
        Left edge
    xMax : float
        Right edge
    bottomY : float
        Bottom of the water fill region
    nPoints : int
        Number of surface sample points
    color : ManimColor
        Fill color (defaults to WATER_FILL)
    opacity : float
        Fill opacity

    Returns:
    --------
    Polygon : Filled water region

    '''
    if color is None:
        color = WATER_FILL

    xVals = np.linspace(xMin, xMax, nPoints)

    # Surface points (left to right)
    surfacePoints = []
    for x in xVals:
        eta = waveTheory.surfaceElevation(x, t, wc)
        surfacePoints.append([x, eta, 0])

    # Bottom corners (right to left)
    bottomPoints = [
        [xMax, bottomY, 0],
        [xMin, bottomY, 0],
    ]

    allPoints = surfacePoints + bottomPoints
    fill = Polygon(
        *[np.array(p) for p in allPoints],
        fill_color=color,
        fill_opacity=opacity,
        stroke_width=0,
    )
    return fill

def updateWaterFill(
    fill: Polygon,
    waveTheory: LinearWaveTheory,
    wc: WaveConditions,
    t: float,
    xMin: float = -7.0,
    xMax: float = 7.0,
    bottomY: float = -4.0,
    nPoints: int = 200,
) -> None:

    '''

    Update an existing water fill polygon to a new time value.

    Parameters:
    -----------
    fill : Polygon
        Existing fill polygon
    waveTheory : LinearWaveTheory
        Wave theory instance
    wc : WaveConditions
        Wave conditions
    t : float
        New time value
    xMin : float
        Left edge
    xMax : float
        Right edge
    bottomY : float
        Bottom of fill region
    nPoints : int
        Number of surface sample points

    '''
    xVals = np.linspace(xMin, xMax, nPoints)
    surfacePoints = []
    for x in xVals:
        eta = waveTheory.surfaceElevation(x, t, wc)
        surfacePoints.append([x, eta, 0])

    bottomPoints = [
        [xMax, bottomY, 0],
        [xMin, bottomY, 0],
    ]

    allPoints = surfacePoints + bottomPoints
    newVertices = [np.array(p) for p in allPoints]

    # Rebuild the polygon by setting new corners
    fill.set_points_as_corners(newVertices + [newVertices[0]])

#--------------------------------------------------------------------#
# -- 3D Wave Surface -- #
#--------------------------------------------------------------------#

def createWaveSurface3D(
    waveTheory: LinearWaveTheory,
    wc: WaveConditions,
    t: float,
    xRange: tuple = (-7.0, 7.0),
    yRange: tuple = (-3.0, 3.0),
    resolution: tuple = (80, 30),
    color = None,
    opacity: float = 0.5,
) -> Surface:

    '''

    Create a 3D parametric wave surface for ThreeDScene.

    The surface elevation varies with x (propagation) but is constant
    in y for our linear wave model. The y-dimension provides visual depth.

    Parameters:
    -----------
    waveTheory : LinearWaveTheory
        Wave theory instance
    wc : WaveConditions
        Wave conditions
    t : float
        Time in seconds
    xRange : tuple
        (xMin, xMax) extent
    yRange : tuple
        (yMin, yMax) lateral extent
    resolution : tuple
        (uRes, vRes) grid resolution
    color : ManimColor
        Surface color
    opacity : float
        Surface opacity

    Returns:
    --------
    Surface : 3D wave surface

    '''
    if color is None:
        color = WAVE_COLOR

    xMin, xMax = xRange
    yMin, yMax = yRange

    def surfaceFunc(u, v):
        x = xMin + u * (xMax - xMin)
        y = yMin + v * (yMax - yMin)
        z = waveTheory.surfaceElevation(x, t, wc)
        return np.array([x, y, z])

    surface = Surface(
        surfaceFunc,
        u_range=[0, 1],
        v_range=[0, 1],
        resolution=resolution,
        fill_color=color,
        fill_opacity=opacity,
        stroke_width=0.5,
        stroke_color=color,
    )
    return surface
