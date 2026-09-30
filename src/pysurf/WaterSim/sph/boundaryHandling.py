# -- SPH Boundary Conditions -- #

'''

Boundary particle generation and enforcement for SPH.

Uses fixed boundary particles arranged in multiple layers along
container walls. Boundary particles participate in density and
pressure computations but do not move. This creates a repulsive
pressure barrier that prevents fluid particles from penetrating
the walls.

For a sloshing tank, boundary particles line the left wall,
bottom wall, and right wall (open top).

References:
-----------
Morris et al. (1997) -- Modeling low Reynolds number incompressible
    flows using SPH
Monaghan & Kos (1999) -- Solitary waves on a Cretan beach

Sean Bowman [02/05/2026]

'''

from __future__ import annotations

import numpy as np

from pysurf.WaterSim.sph.particles import ParticleSystem


class BoundaryHandler:

    '''

    Manages boundary particles for a rectangular container.

    Generates layers of fixed boundary particles along container
    walls and enforces boundary conditions on fluid particles
    that attempt to penetrate walls.

    Parameters:
    -----------
    containerMin : np.ndarray
        Lower corner of the container [m]
    containerMax : np.ndarray
        Upper corner of the container [m]
    spacing : float
        Inter-particle spacing [m] (matches fluid spacing)
    nLayers : int
        Number of boundary particle layers (default 3)
    dimensions : int
        Number of spatial dimensions (2 or 3)
    openTop : bool
        If True, no boundary particles on the top face (default True)

    '''

    def __init__(
        self,
        containerMin: np.ndarray,
        containerMax: np.ndarray,
        spacing: float,
        nLayers: int = 3,
        dimensions: int = 2,
        openTop: bool = True,
    ) -> None:
        self._containerMin = containerMin.copy()
        self._containerMax = containerMax.copy()
        self._spacing = spacing
        self._nLayers = nLayers
        self._dimensions = dimensions
        self._openTop = openTop

    def generateBoundaryParticles(
        self, referenceDensity: float
    ) -> tuple[np.ndarray, np.ndarray]:

        '''

        Generate boundary particle positions and masses.

        Creates multiple layers of particles along each wall face.
        Layers extend outward from the container boundary.

        Parameters:
        -----------
        referenceDensity : float
            Reference density for computing particle mass [kg/m^3]

        Returns:
        --------
        tuple[np.ndarray, np.ndarray] :
            (positions, masses) of boundary particles
            positions shape: (M, dim), masses shape: (M,)

        '''
        if self._dimensions == 2:
            return self._generate2D(referenceDensity)
        else:
            return self._generate3D(referenceDensity)

    def _generate2D(
        self, referenceDensity: float
    ) -> tuple[np.ndarray, np.ndarray]:

        '''

        Generate 2D boundary particles for a rectangular container.

        Walls: left, bottom, right (and optionally top).
        Each wall has nLayers of particles extending outward.

        Parameters:
        -----------
        referenceDensity : float
            Reference density [kg/m^3]

        Returns:
        --------
        tuple[np.ndarray, np.ndarray] : (positions, masses)

        '''
        s = self._spacing
        xMin, yMin = self._containerMin
        xMax, yMax = self._containerMax

        allPositions: list[np.ndarray] = []

        for layer in range(self._nLayers):
            offset = (layer + 1) * s

            # Bottom wall: y = yMin - offset, x from (xMin - nLayers*s) to (xMax + nLayers*s)
            xCoords = np.arange(
                xMin - self._nLayers * s + s / 2.0,
                xMax + self._nLayers * s,
                s,
            )
            bottomY = np.full_like(xCoords, yMin - offset + s / 2.0)
            bottomPositions = np.column_stack([xCoords, bottomY])
            allPositions.append(bottomPositions)

            # Left wall: x = xMin - offset, y from yMin to yMax
            yCoords = np.arange(yMin + s / 2.0, yMax, s)
            leftX = np.full_like(yCoords, xMin - offset + s / 2.0)
            leftPositions = np.column_stack([leftX, yCoords])
            allPositions.append(leftPositions)

            # Right wall: x = xMax + offset, y from yMin to yMax
            rightX = np.full_like(yCoords, xMax + offset - s / 2.0)
            rightPositions = np.column_stack([rightX, yCoords])
            allPositions.append(rightPositions)

            # Top wall (optional)
            if not self._openTop:
                topY = np.full_like(xCoords, yMax + offset - s / 2.0)
                topPositions = np.column_stack([xCoords, topY])
                allPositions.append(topPositions)

        positions = np.vstack(allPositions)
        particleVolume = s ** self._dimensions
        mass = referenceDensity * particleVolume
        masses = np.full(len(positions), mass)

        return (positions, masses)

    def _generate3D(
        self, referenceDensity: float
    ) -> tuple[np.ndarray, np.ndarray]:

        '''

        Generate 3D boundary particles for a rectangular container.

        Creates layers of particles on floor (z=zMin), left wall (x=xMin),
        right wall (x=xMax), front wall (y=yMin), and back wall (y=yMax).
        Open top (z=zMax) has no particles if openTop=True.

        Coordinate convention:
            x: Length (wave propagation direction)
            y: Width (lateral)
            z: Height (vertical, gravity in -z)

        Parameters:
        -----------
        referenceDensity : float
            Reference density [kg/m^3]

        Returns:
        --------
        tuple[np.ndarray, np.ndarray] : (positions, masses)

        '''
        s = self._spacing
        xMin, yMin, zMin = self._containerMin
        xMax, yMax, zMax = self._containerMax

        allPositions: list[np.ndarray] = []

        # Extended domain for corners (boundary layers extend outward)
        xMinExt = xMin - self._nLayers * s
        xMaxExt = xMax + self._nLayers * s
        yMinExt = yMin - self._nLayers * s
        yMaxExt = yMax + self._nLayers * s

        for layer in range(self._nLayers):
            offset = (layer + 1) * s

            # ----------------------------------------------------------------
            # Floor (z = zMin - offset), covers full extended xy range
            # ----------------------------------------------------------------
            xCoords = np.arange(xMinExt + s / 2.0, xMaxExt, s)
            yCoords = np.arange(yMinExt + s / 2.0, yMaxExt, s)
            xx, yy = np.meshgrid(xCoords, yCoords, indexing='xy')
            floorZ = np.full_like(xx, zMin - offset + s / 2.0)
            floorPositions = np.column_stack([
                xx.ravel(), yy.ravel(), floorZ.ravel()
            ])
            allPositions.append(floorPositions)

            # ----------------------------------------------------------------
            # Left wall (x = xMin - offset), yz plane, z from zMin to zMax
            # ----------------------------------------------------------------
            yWall = np.arange(yMinExt + s / 2.0, yMaxExt, s)
            zWall = np.arange(zMin + s / 2.0, zMax, s)
            yyWall, zzWall = np.meshgrid(yWall, zWall, indexing='xy')
            leftX = np.full_like(yyWall, xMin - offset + s / 2.0)
            leftPositions = np.column_stack([
                leftX.ravel(), yyWall.ravel(), zzWall.ravel()
            ])
            allPositions.append(leftPositions)

            # ----------------------------------------------------------------
            # Right wall (x = xMax + offset), yz plane
            # ----------------------------------------------------------------
            rightX = np.full_like(yyWall, xMax + offset - s / 2.0)
            rightPositions = np.column_stack([
                rightX.ravel(), yyWall.ravel(), zzWall.ravel()
            ])
            allPositions.append(rightPositions)

            # ----------------------------------------------------------------
            # Front wall (y = yMin - offset), xz plane, interior x only
            # (to avoid overlap with left/right walls)
            # ----------------------------------------------------------------
            xWall = np.arange(xMin + s / 2.0, xMax, s)
            xxWall, zzFront = np.meshgrid(xWall, zWall, indexing='xy')
            frontY = np.full_like(xxWall, yMin - offset + s / 2.0)
            frontPositions = np.column_stack([
                xxWall.ravel(), frontY.ravel(), zzFront.ravel()
            ])
            allPositions.append(frontPositions)

            # ----------------------------------------------------------------
            # Back wall (y = yMax + offset), xz plane
            # ----------------------------------------------------------------
            backY = np.full_like(xxWall, yMax + offset - s / 2.0)
            backPositions = np.column_stack([
                xxWall.ravel(), backY.ravel(), zzFront.ravel()
            ])
            allPositions.append(backPositions)

            # ----------------------------------------------------------------
            # Top wall (optional, z = zMax + offset)
            # ----------------------------------------------------------------
            if not self._openTop:
                topZ = np.full_like(xx, zMax + offset - s / 2.0)
                topPositions = np.column_stack([
                    xx.ravel(), yy.ravel(), topZ.ravel()
                ])
                allPositions.append(topPositions)

        positions = np.vstack(allPositions)
        particleVolume = s ** self._dimensions
        mass = referenceDensity * particleVolume
        masses = np.full(len(positions), mass)

        return (positions, masses)

    def enforceBoundary(self, particles: ParticleSystem) -> None:

        '''

        Enforce boundary conditions on fluid particles.

        Clamps fluid particle positions to stay within the container
        and reflects velocities for particles that hit walls.

        Parameters:
        -----------
        particles : ParticleSystem
            The particle system to enforce boundaries on

        '''
        fluidMask = particles.isFluid
        positions = particles.positions
        velocities = particles.velocities

        for d in range(self._dimensions):
            # Check lower boundary
            belowMin = fluidMask & (positions[:, d] < self._containerMin[d])
            positions[belowMin, d] = self._containerMin[d]
            velocities[belowMin, d] = abs(velocities[belowMin, d]) * 0.5

            # Check upper boundary (skip top if open)
            # Vertical axis: y (d=1) in 2D, z (d=2) in 3D
            verticalAxis = 2 if self._dimensions == 3 else 1
            if d == verticalAxis and self._openTop:
                # Vertical axis open top -- no upper clamping
                continue

            aboveMax = fluidMask & (positions[:, d] > self._containerMax[d])
            positions[aboveMax, d] = self._containerMax[d]
            velocities[aboveMax, d] = -abs(velocities[aboveMax, d]) * 0.5
