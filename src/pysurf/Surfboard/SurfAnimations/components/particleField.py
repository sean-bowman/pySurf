# -- SPH Particle Field Components -- #

'''

Reusable Manim mobjects for rendering SPH particle fields.

Provides functions to create groups of colored dots representing
SPH particles, with color mapping by velocity magnitude or
other scalar fields. Supports frame-by-frame animation updates.

Sean Bowman [02/05/2026]

'''

import os
import sys

import numpy as np
from manim import (
    VGroup, Dot,
    interpolate_color,
    ORIGIN, UP, RIGHT,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pysurf.Surfboard.SurfAnimations.utils.manimTheme import BLUE, RED, CYAN, WHITE

#--------------------------------------------------------------------#
# -- Particle Field Creation -- #
#--------------------------------------------------------------------#

def createParticleField(
    positions: np.ndarray,
    scalarField: np.ndarray | None = None,
    radius: float = 0.04,
    colorMin = None,
    colorMax = None,
    scalarMin: float | None = None,
    scalarMax: float | None = None,
    opacity: float = 0.85,
    scale: float = 1.0,
    offset: np.ndarray | None = None,
) -> VGroup:

    '''

    Create a group of Manim dots representing SPH particles.

    Positions are mapped from simulation coordinates to Manim
    scene coordinates using the given scale and offset.

    Parameters:
    -----------
    positions : np.ndarray
        Particle positions (N, 2) in simulation coordinates [m]
    scalarField : np.ndarray | None
        Scalar values for color mapping (e.g., velocity magnitude).
        If None, all particles use colorMin.
    radius : float
        Dot radius in Manim units
    colorMin : ManimColor
        Color for minimum scalar value (default: CYAN)
    colorMax : ManimColor
        Color for maximum scalar value (default: RED)
    scalarMin : float | None
        Minimum scalar for normalization (default: auto from data)
    scalarMax : float | None
        Maximum scalar for normalization (default: auto from data)
    opacity : float
        Dot opacity (0-1)
    scale : float
        Scale factor: Manim units per simulation meter
    offset : np.ndarray | None
        Translation offset in Manim units (default: center on origin)

    Returns:
    --------
    VGroup : Group of colored dots

    '''
    colorMin = colorMin or CYAN
    colorMax = colorMax or RED

    nParticles = len(positions)
    if nParticles == 0:
        return VGroup()

    # Scale positions to Manim coordinates
    manimPositions = positions * scale
    if offset is not None:
        manimPositions = manimPositions + offset

    # Compute colors from scalar field
    colors = _computeColors(scalarField, colorMin, colorMax, scalarMin, scalarMax, nParticles)

    # Create dots
    dots = VGroup()
    for i in range(nParticles):
        x, y = manimPositions[i]
        dot = Dot(
            point=np.array([x, y, 0.0]),
            radius=radius,
            color=colors[i],
            fill_opacity=opacity,
        )
        dots.add(dot)

    return dots

#--------------------------------------------------------------------#
# -- Particle Field Update -- #
#--------------------------------------------------------------------#

def updateParticleField(
    dotGroup: VGroup,
    positions: np.ndarray,
    scalarField: np.ndarray | None = None,
    colorMin = None,
    colorMax = None,
    scalarMin: float | None = None,
    scalarMax: float | None = None,
    scale: float = 1.0,
    offset: np.ndarray | None = None,
) -> None:

    '''

    Update existing dot positions and colors from new frame data.

    Modifies the dots in-place for efficient animation. The number
    of particles must match the existing VGroup size.

    Parameters:
    -----------
    dotGroup : VGroup
        Existing dot group from createParticleField
    positions : np.ndarray
        New particle positions (N, 2) in simulation coordinates
    scalarField : np.ndarray | None
        New scalar values for color mapping
    colorMin : ManimColor
        Color for minimum scalar value
    colorMax : ManimColor
        Color for maximum scalar value
    scalarMin : float | None
        Minimum scalar for normalization
    scalarMax : float | None
        Maximum scalar for normalization
    scale : float
        Scale factor: Manim units per simulation meter
    offset : np.ndarray | None
        Translation offset in Manim units

    '''
    colorMin = colorMin or CYAN
    colorMax = colorMax or RED

    nParticles = len(positions)

    # Scale positions
    manimPositions = positions * scale
    if offset is not None:
        manimPositions = manimPositions + offset

    # Compute colors
    colors = _computeColors(scalarField, colorMin, colorMax, scalarMin, scalarMax, nParticles)

    # Update each dot
    for i in range(min(nParticles, len(dotGroup))):
        x, y = manimPositions[i]
        dotGroup[i].move_to(np.array([x, y, 0.0]))
        dotGroup[i].set_color(colors[i])

#--------------------------------------------------------------------#
# -- Helper Functions -- #
#--------------------------------------------------------------------#

def _computeColors(
    scalarField: np.ndarray | None,
    colorMin,
    colorMax,
    scalarMin: float | None,
    scalarMax: float | None,
    nParticles: int,
) -> list:

    '''

    Compute per-particle colors from a scalar field.

    Parameters:
    -----------
    scalarField : np.ndarray | None
        Scalar values to map to colors
    colorMin, colorMax : ManimColor
        Color range endpoints
    scalarMin, scalarMax : float | None
        Normalization range (auto if None)
    nParticles : int
        Number of particles

    Returns:
    --------
    list : Per-particle ManimColor values

    '''
    if scalarField is None or len(scalarField) == 0:
        return [colorMin] * nParticles

    sMin = scalarMin if scalarMin is not None else float(np.min(scalarField))
    sMax = scalarMax if scalarMax is not None else float(np.max(scalarField))

    if abs(sMax - sMin) < 1e-12:
        return [colorMin] * nParticles

    # Normalize to [0, 1] and interpolate colors
    normalized = np.clip((scalarField - sMin) / (sMax - sMin), 0.0, 1.0)

    colors = []
    for t in normalized:
        colors.append(interpolate_color(colorMin, colorMax, float(t)))

    return colors
