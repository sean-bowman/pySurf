# -- Velocity Field Components -- #

'''

Manim mobjects for rendering wave-induced velocity fields.

Generates quiver-style arrow grids showing the orbital velocity
pattern beneath a propagating wave, with color mapping by magnitude.

Sean Bowman [02/04/2026]

'''

import os
import sys

import numpy as np
from manim import Arrow, VGroup, interpolate_color, ManimColor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pysurf.Surfboard.SurfPhysics.waves.linearWaveTheory import LinearWaveTheory
from pysurf.Surfboard.SurfPhysics.waves.waveConditions import WaveConditions
from pysurf.Surfboard.SurfAnimations.utils.manimTheme import BLUE, CYAN

#--------------------------------------------------------------------#
# -- 2D Velocity Quiver Field -- #
#--------------------------------------------------------------------#

def createVelocityField(
    waveTheory: LinearWaveTheory,
    wc: WaveConditions,
    t: float,
    xMin: float = -6.0,
    xMax: float = 6.0,
    zMin: float = -2.5,
    zMax: float = -0.1,
    nX: int = 16,
    nZ: int = 6,
    scaleFactor: float = 0.8,
    maxArrowLength: float = 0.6,
    strokeWidth: float = 2.0,
    colorLow: ManimColor = BLUE,
    colorHigh: ManimColor = CYAN,
) -> VGroup:

    '''

    Create a 2D velocity quiver field below the wave surface.

    Samples the velocity field on a grid of (x, z) points and creates
    small arrows showing flow direction and magnitude.

    Parameters:
    -----------
    waveTheory : LinearWaveTheory
        Wave theory instance
    wc : WaveConditions
        Wave conditions
    t : float
        Time in seconds
    xMin : float
        Left boundary of the field
    xMax : float
        Right boundary
    zMin : float
        Bottom boundary (negative = below surface)
    zMax : float
        Top boundary (should be below wave surface)
    nX : int
        Number of horizontal sample points
    nZ : int
        Number of vertical sample points
    scaleFactor : float
        Arrow length multiplier (Manim units per m/s)
    maxArrowLength : float
        Maximum individual arrow length
    strokeWidth : float
        Arrow stroke width
    colorLow : ManimColor
        Color for low-magnitude velocities
    colorHigh : ManimColor
        Color for high-magnitude velocities

    Returns:
    --------
    VGroup : Collection of velocity arrows

    '''
    arrows = VGroup()

    xVals = np.linspace(xMin, xMax, nX)
    zVals = np.linspace(zMin, zMax, nZ)

    # Find max velocity magnitude for color normalization
    maxMag = 0.0
    velocities = []
    for x in xVals:
        eta = waveTheory.surfaceElevation(x, t, wc)
        for z in zVals:
            # Skip points above the wave surface
            if z > eta - 0.05:
                velocities.append(None)
                continue
            u, w = waveTheory.velocityField(x, z, t, wc)
            mag = (u * u + w * w) ** 0.5
            maxMag = max(maxMag, mag)
            velocities.append((x, z, u, w, mag))

    if maxMag < 1e-10:
        return arrows

    # Create arrows
    for entry in velocities:
        if entry is None:
            continue
        x, z, u, w, mag = entry

        # Skip tiny velocities
        if mag < maxMag * 0.05:
            continue

        # Arrow length proportional to magnitude
        arrowLen = min(mag * scaleFactor, maxArrowLength)
        if arrowLen < 0.05:
            continue

        # Direction
        dirX = u / mag
        dirZ = w / mag

        start = np.array([x, z, 0])
        end = start + np.array([dirX, dirZ, 0]) * arrowLen

        # Color by magnitude
        colorAlpha = mag / maxMag
        color = interpolate_color(colorLow, colorHigh, colorAlpha)

        arrow = Arrow(
            start=start,
            end=end,
            color=color,
            stroke_width=strokeWidth,
            buff=0,
            max_tip_length_to_length_ratio=0.3,
            max_stroke_width_to_length_ratio=8,
        )
        arrows.add(arrow)

    return arrows


def updateVelocityField(
    fieldGroup: VGroup,
    waveTheory: LinearWaveTheory,
    wc: WaveConditions,
    t: float,
    xMin: float = -6.0,
    xMax: float = 6.0,
    zMin: float = -2.5,
    zMax: float = -0.1,
    nX: int = 16,
    nZ: int = 6,
    scaleFactor: float = 0.8,
    maxArrowLength: float = 0.6,
) -> VGroup:

    '''

    Create a new velocity field VGroup at a new time value.

    Rather than updating arrows in-place (complex with variable
    visibility), this returns a fresh VGroup. Use with become()
    or replace in the scene.

    Parameters:
    -----------
    fieldGroup : VGroup
        Previous field (not modified)
    waveTheory : LinearWaveTheory
        Wave theory instance
    wc : WaveConditions
        Wave conditions
    t : float
        New time value

    Returns:
    --------
    VGroup : New velocity field arrows

    '''
    return createVelocityField(
        waveTheory, wc, t,
        xMin=xMin, xMax=xMax,
        zMin=zMin, zMax=zMax,
        nX=nX, nZ=nZ,
        scaleFactor=scaleFactor,
        maxArrowLength=maxArrowLength,
    )

#--------------------------------------------------------------------#
# -- Particle Orbital Paths -- #
#--------------------------------------------------------------------#

def createParticleOrbits(
    waveTheory: LinearWaveTheory,
    wc: WaveConditions,
    xCenter: float = 0.0,
    depths: list = None,
    nPoints: int = 100,
    color: ManimColor = CYAN,
    strokeWidth: float = 1.5,
    opacity: float = 0.6,
) -> VGroup:

    '''

    Create circular/elliptical orbital paths at different depths.

    Traces particle trajectories by integrating the velocity field
    over one wave period. In deep water these are circles that
    decay exponentially with depth.

    Parameters:
    -----------
    waveTheory : LinearWaveTheory
        Wave theory instance
    wc : WaveConditions
        Wave conditions
    xCenter : float
        X position for the orbital display
    depths : list
        Z depths for each orbit (negative values)
    nPoints : int
        Points per orbit
    color : ManimColor
        Orbit path color
    strokeWidth : float
        Path stroke width
    opacity : float
        Path opacity

    Returns:
    --------
    VGroup : Collection of orbit path VMobjects

    '''
    from manim import VMobject

    if depths is None:
        depths = [-0.3, -0.8, -1.5, -2.2]

    orbits = VGroup()
    period = wc.period
    tVals = np.linspace(0, period, nPoints)

    for z0 in depths:
        # Skip if below seabed
        if z0 < -wc.depth:
            continue

        # Integrate velocity to get displacement
        pathPoints = []
        dx, dz = 0.0, 0.0

        for i, tVal in enumerate(tVals):
            u, w = waveTheory.velocityField(xCenter, z0, tVal, wc)
            if i > 0:
                dt = tVals[i] - tVals[i - 1]
                dx += u * dt
                dz += w * dt

            pathPoints.append([xCenter + dx, z0 + dz, 0])

        # Close the orbit
        pathPoints.append(pathPoints[0])

        orbit = VMobject()
        orbit.set_points_as_corners([np.array(p) for p in pathPoints])
        orbit.set_stroke(color=color, width=strokeWidth, opacity=opacity)

        # Fade deeper orbits
        depthFactor = max(0.2, 1.0 + z0 / wc.depth)
        orbit.set_stroke(opacity=opacity * depthFactor)

        orbits.add(orbit)

    return orbits
