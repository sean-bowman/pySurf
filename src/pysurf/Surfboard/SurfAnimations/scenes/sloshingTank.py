# -- Sloshing Tank Animation Scene -- #

'''

Manim animation showing SPH sloshing tank simulation.

Loads pre-computed frame data from JSON exports and animates
particles sloshing inside a container. Color-codes particles
by velocity magnitude with an energy tracker overlay.

Usage:
    python SurfAnimations/render.py --scene sloshing_tank
    manim render -qm SurfAnimations/scenes/sloshingTank.py SloshingTankAnimation

Sean Bowman [02/05/2026]

'''

import json
import os
import sys

import numpy as np
from manim import (
    Scene, Text, VGroup, Dot, Line, FadeIn, FadeOut,
    Write, Create, Transform,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
    config as manimConfig,
    rate_functions,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pysurf.Surfboard.SurfAnimations.utils.manimTheme import (
    BG_COLOR, WHITE, BLUE, CYAN, RED, WAVE_COLOR,
)
from pysurf.Surfboard.SurfAnimations.components.particleField import (
    createParticleField, updateParticleField,
)
from pysurf.Surfboard.SurfAnimations.components.containerWalls import createContainerWalls


class SloshingTankAnimation(Scene):

    '''

    Animated sloshing tank SPH simulation.

    Loads exported frame JSON data and plays back the particle
    positions with velocity-based color mapping. Shows the
    container walls and a title card.

    To use: export simulation data with WaterSim, then render
    this scene with the exported JSON path.

    '''

    def construct(self) -> None:

        '''Build and play the animation sequence.'''

        # --- Configuration --- #
        manimConfig.background_color = BG_COLOR

        # Find the most recent exported simulation file
        dataDir = os.path.join(
            os.path.dirname(__file__), '..', '..', 'WaterSim', 'output'
        )
        jsonPath = self._findLatestExport(dataDir)

        if jsonPath is None:
            # No data file found -- show error message
            errorText = Text(
                'No simulation data found.\n'
                'Run: python -m WaterSim --preset small',
                font_size=28,
                color=WHITE,
            )
            self.play(Write(errorText))
            self.wait(3)
            return

        # --- Load simulation data --- #
        with open(jsonPath, 'r') as f:
            data = json.load(f)

        frames = data['frames']
        simConfig = data['config']
        meta = data['meta']
        nFrames = len(frames)

        domainMin = np.array(simConfig['domainMin'])
        domainMax = np.array(simConfig['domainMax'])
        tankWidth = domainMax[0] - domainMin[0]
        tankHeight = domainMax[1] - domainMin[1]

        # Scale to fit Manim scene (target ~10 units wide)
        targetWidth = 8.0
        simScale = targetWidth / tankWidth
        offset = np.array([-targetWidth / 2.0, -tankHeight * simScale / 2.0])

        #--------------------------------------------------------------------#
        # Title Card
        #--------------------------------------------------------------------#
        title = Text(
            'SPH Sloshing Simulation',
            font_size=36,
            color=WHITE,
        ).to_edge(UP, buff=0.4)

        subtitle = Text(
            f'{meta.get("nFluidParticles", "?")} particles  |  '
            f'dx = {simConfig.get("particleSpacing", "?")}m  |  '
            f'{nFrames} frames',
            font_size=20,
            color=BLUE,
        ).next_to(title, DOWN, buff=0.2)

        self.play(Write(title), run_time=0.8)
        self.play(FadeIn(subtitle), run_time=0.5)

        #--------------------------------------------------------------------#
        # Container walls
        #--------------------------------------------------------------------#
        walls = createContainerWalls(
            width=tankWidth,
            height=tankHeight,
            scale=simScale,
            offset=offset,
            strokeWidth=3.0,
        )
        self.play(Create(walls), run_time=0.6)

        #--------------------------------------------------------------------#
        # Initial particle field
        #--------------------------------------------------------------------#
        firstFrame = frames[0]
        positions = np.array(firstFrame['positions'])
        velocities = np.array(firstFrame['velocityMagnitudes'])

        # Determine global max velocity for consistent color scaling
        globalMaxVel = 0.0
        for frame in frames:
            vels = frame['velocityMagnitudes']
            if vels:
                globalMaxVel = max(globalMaxVel, max(vels))
        globalMaxVel = max(globalMaxVel, 0.1)  # Prevent zero range

        dotRadius = max(0.02, simScale * simConfig.get('particleSpacing', 0.005) * 0.4)

        particleDots = createParticleField(
            positions=positions,
            scalarField=velocities,
            radius=dotRadius,
            scalarMin=0.0,
            scalarMax=globalMaxVel,
            scale=simScale,
            offset=offset,
            opacity=0.85,
        )
        self.play(FadeIn(particleDots), run_time=0.8)
        self.wait(0.3)

        #--------------------------------------------------------------------#
        # Time label
        #--------------------------------------------------------------------#
        timeLabel = Text(
            f't = {firstFrame["time"]:.3f} s',
            font_size=22,
            color=WHITE,
        ).to_corner(DOWN + RIGHT, buff=0.5)
        self.add(timeLabel)

        #--------------------------------------------------------------------#
        # Animate frames
        #--------------------------------------------------------------------#
        # Target ~15-20 seconds of animation, skip frames if needed
        targetDuration = 15.0
        frameInterval = max(1, nFrames // int(targetDuration * 30))
        frameDt = 1.0 / 30.0  # 30fps target

        for frameIdx in range(1, nFrames, frameInterval):
            frame = frames[frameIdx]
            newPositions = np.array(frame['positions'])
            newVelocities = np.array(frame['velocityMagnitudes'])

            # Update particles
            updateParticleField(
                dotGroup=particleDots,
                positions=newPositions,
                scalarField=newVelocities,
                scalarMin=0.0,
                scalarMax=globalMaxVel,
                scale=simScale,
                offset=offset,
            )

            # Update time label
            newTimeLabel = Text(
                f't = {frame["time"]:.3f} s',
                font_size=22,
                color=WHITE,
            ).to_corner(DOWN + RIGHT, buff=0.5)
            self.remove(timeLabel)
            timeLabel = newTimeLabel
            self.add(timeLabel)

            self.wait(frameDt)

        #--------------------------------------------------------------------#
        # End card
        #--------------------------------------------------------------------#
        self.wait(0.5)
        endText = Text(
            'Simulation Complete',
            font_size=28,
            color=BLUE,
        ).to_edge(DOWN, buff=1.0)
        self.play(FadeIn(endText), run_time=0.6)
        self.wait(1.5)

    def _findLatestExport(self, dataDir: str) -> str | None:

        '''

        Find the most recently exported simulation JSON file.

        Parameters:
        -----------
        dataDir : str
            Directory to search for JSON exports

        Returns:
        --------
        str | None : Path to the most recent file, or None

        '''
        if not os.path.isdir(dataDir):
            return None

        jsonFiles = [
            os.path.join(dataDir, f)
            for f in os.listdir(dataDir)
            if f.startswith('waterSim_') and f.endswith('.json')
        ]

        if not jsonFiles:
            return None

        # Sort by modification time, most recent first
        jsonFiles.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        return jsonFiles[0]
