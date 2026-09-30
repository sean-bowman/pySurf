# -- Board on Wave Scenes -- #

'''

Manim animations showing a surfboard riding a wave.

Contains two scene classes:
- BoardOnWaveSideView: 2D side profile with force vectors and velocity field
- BoardOnWavePerspective: 2.5D ThreeDScene with 3D wave surface and board outline

Sean Bowman [02/04/2026]

'''

import math
import os
import sys

import numpy as np
from manim import (
    Scene, ThreeDScene, Text, MathTex, VGroup, VMobject,
    ValueTracker, Arrow, DashedLine, Arc, Polygon,
    Create, FadeIn, FadeOut, Write,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
    linear, config,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pysurf.Surfboard.SurfPhysics.waves.linearWaveTheory import LinearWaveTheory
from pysurf.Surfboard.SurfPhysics.waves.waveConditions import WaveConditions
from pysurf.Surfboard.SurfPhysics.geometry.board import BoardGeometry
from pysurf.Surfboard.SurfPhysics.geometry.parameters import SurfboardParameters
from pysurf.Surfboard.SurfPhysics.hydrodynamics.forceBalance import ForceBalance
from pysurf.Surfboard.SurfPhysics import constants as const

from pysurf.Surfboard.SurfAnimations.utils.manimTheme import (
    WAVE_COLOR, WATER_FILL, BG_COLOR, WHITE, CYAN, BLUE,
    RED, GREEN, ORANGE, REFERENCE_LINE, BOARD_COLOR,
    SHORTBOARD_COLOR,
)
from pysurf.Surfboard.SurfAnimations.components.waveSurface import (
    createWaveLine, updateWaveLine,
    createWaterFill, updateWaterFill,
    createWaveSurface3D,
)
from pysurf.Surfboard.SurfAnimations.components.boardProfile import (
    createBoardProfile, positionBoardOnWave,
    createBoardOutline3D, positionBoardOnWave3D,
)
from pysurf.Surfboard.SurfAnimations.components.forceArrows import (
    createForceArrow, createForceBalance,
    createForceBalance3D,
)
from pysurf.Surfboard.SurfAnimations.components.velocityField import createVelocityField

#--------------------------------------------------------------------#
# -- Scene A: 2D Side View -- #
#--------------------------------------------------------------------#

class BoardOnWaveSideView(Scene):

    '''

    2D side-view animation of a shortboard riding a wave.

    Shows the board profile on the wave face with force vectors
    (weight, buoyancy, planing lift, drag) and velocity field.

    '''

    def construct(self) -> None:
        self.camera.background_color = BG_COLOR

        # Physics setup
        waveTheory = LinearWaveTheory()
        wc = WaveConditions.typicalBeachBreak()
        params = SurfboardParameters.shortboard()
        board = BoardGeometry(params)
        surfSpeed = waveTheory.surfableWaveSpeed(wc)
        forceBalance = ForceBalance(board, params, riderMassKg=const.defaultSurferMassKg)
        state = forceBalance.findEquilibrium(surfSpeed)
        totalWeightN = forceBalance.totalWeightN

        # Scene domain
        xMin, xMax = -7.0, 7.0
        bottomY = -3.5

        time = ValueTracker(0)

        # -------------------------------------------------------
        # 1. Wave + Board Appear (0-3s)
        # -------------------------------------------------------

        # Title
        title = Text(
            'Surfboard on Wave: Side View',
            font_size=30,
            color=WHITE,
        ).to_corner(UP + LEFT, buff=0.3).scale(0.8)

        self.play(Write(title), run_time=0.8)

        # Wave and water
        waveLine = createWaveLine(waveTheory, wc, 0, xMin, xMax)
        waterFill = createWaterFill(waveTheory, wc, 0, xMin, xMax, bottomY)

        # Still water line
        stillWater = DashedLine(
            start=np.array([xMin, 0, 0]),
            end=np.array([xMax, 0, 0]),
            color=REFERENCE_LINE,
            dash_length=0.15,
            stroke_width=1,
            stroke_opacity=0.3,
        )

        self.play(
            Create(stillWater),
            Create(waveLine),
            FadeIn(waterFill),
            run_time=1.0,
        )

        # Board profile
        boardShape = createBoardProfile(
            board, params,
            color=SHORTBOARD_COLOR,
            fillOpacity=0.85,
        )

        # Position board on wave face
        boardX = 1.0
        positionBoardOnWave(
            boardShape, board, waveTheory, wc,
            boardX=boardX, t=0,
            trimAngleDeg=state.trimAngleDeg,
            draftM=state.draftM,
        )

        self.play(FadeIn(boardShape), run_time=1.0)
        self.wait(0.5)

        # -------------------------------------------------------
        # 2. Board Starts Riding (3-8s)
        # -------------------------------------------------------

        # Set up updaters for wave propagation
        def waveUpdater(mob):
            t = time.get_value()
            updateWaveLine(mob, waveTheory, wc, t, xMin, xMax)

        def fillUpdater(mob):
            t = time.get_value()
            updateWaterFill(mob, waveTheory, wc, t, xMin, xMax, bottomY)

        def boardUpdater(mob):
            t = time.get_value()
            # Board moves with wave at surf speed
            # Wave phase velocity moves the wave profile rightward
            omega = wc.angularFrequency
            k = waveTheory.solveDispersionRelation(omega, wc.depth)
            phaseX = omega * t / k
            # Board rides on the face, offset from the crest
            currentBoardX = boardX + (phaseX % (2 * math.pi / k)) - (2 * math.pi / k) / 4
            # Keep board in visible range
            while currentBoardX > xMax - 2:
                currentBoardX -= 2 * math.pi / k
            while currentBoardX < xMin + 2:
                currentBoardX += 2 * math.pi / k

            positionBoardOnWave(
                mob, board, waveTheory, wc,
                boardX=currentBoardX, t=t,
                trimAngleDeg=state.trimAngleDeg,
                draftM=state.draftM,
            )

        waveLine.add_updater(waveUpdater)
        waterFill.add_updater(fillUpdater)
        boardShape.add_updater(boardUpdater)

        # Speed info
        speedText = Text(
            f'Speed: {surfSpeed:.1f} m/s ({surfSpeed * 3.6:.0f} km/h)',
            font_size=18,
            color=CYAN,
        ).to_corner(UP + RIGHT, buff=0.3)

        self.play(Write(speedText), run_time=0.5)
        self.play(
            time.animate.set_value(5.0),
            run_time=5.0,
            rate_func=linear,
        )

        # -------------------------------------------------------
        # 3. Force Vectors Appear (8-13s)
        # -------------------------------------------------------

        # Pause wave for force display
        waveLine.remove_updater(waveUpdater)
        waterFill.remove_updater(fillUpdater)
        boardShape.remove_updater(boardUpdater)

        currentTime = time.get_value()
        boardCenter = boardShape.get_center()

        forces = createForceBalance(
            state, totalWeightN, boardCenter,
            trimAngleDeg=state.trimAngleDeg,
            fontSize=16,
        )

        # Animate forces one by one
        forceOrder = ['weight', 'buoyancy', 'lift', 'drag']
        forceLabels = {
            'weight': 'Weight (Gravity)',
            'buoyancy': 'Buoyancy (Archimedes)',
            'lift': 'Planing Lift (Hydrodynamic)',
            'drag': 'Drag (Resistance)',
        }

        for forceName in forceOrder:
            self.play(FadeIn(forces[forceName]), run_time=1.0)
            self.wait(0.3)

        # -------------------------------------------------------
        # 4. Velocity Field (13-17s)
        # -------------------------------------------------------

        velField = createVelocityField(
            waveTheory, wc, currentTime,
            xMin=-5.0, xMax=5.0,
            zMin=-2.5, zMax=-0.2,
            nX=12, nZ=4,
            scaleFactor=0.6,
        )

        velLabel = Text(
            'Velocity Field',
            font_size=18,
            color=CYAN,
        ).to_edge(DOWN, buff=0.3)

        self.play(FadeIn(velField), Write(velLabel), run_time=1.0)
        self.wait(2.0)

        # -------------------------------------------------------
        # 5. Trim Angle + L/D (17-20s)
        # -------------------------------------------------------

        self.play(FadeOut(velField), FadeOut(velLabel), run_time=0.5)

        # Trim angle arc
        trimText = Text(
            f'Trim Angle: {state.trimAngleDeg:.1f}\u00b0',
            font_size=20,
            color=ORANGE,
        ).to_edge(DOWN, buff=0.5).shift(LEFT * 2)

        ldText = Text(
            f'L/D Ratio: {state.liftToDrag:.1f}',
            font_size=20,
            color=GREEN,
        ).next_to(trimText, RIGHT, buff=1.0)

        planingText = Text(
            'Planing' if state.isPlaning else 'Displacement',
            font_size=20,
            color=CYAN if state.isPlaning else ORANGE,
        ).next_to(ldText, RIGHT, buff=1.0)

        self.play(
            Write(trimText), Write(ldText), Write(planingText),
            run_time=1.0,
        )
        self.wait(2.0)

        # -------------------------------------------------------
        # Final Fade
        # -------------------------------------------------------

        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=1.5,
        )
        self.wait(0.5)

#--------------------------------------------------------------------#
# -- Scene B: 2.5D Perspective View -- #
#--------------------------------------------------------------------#

class BoardOnWavePerspective(ThreeDScene):

    '''

    2.5D perspective animation of a surfboard riding a wave.

    Shows a 3D wave surface with the board planform outline
    tracking the wave, 3D force arrows, and camera orbit.

    '''

    def construct(self) -> None:
        self.camera.background_color = BG_COLOR

        # Physics setup
        waveTheory = LinearWaveTheory()
        wc = WaveConditions.typicalBeachBreak()
        params = SurfboardParameters.shortboard()
        board = BoardGeometry(params)
        surfSpeed = waveTheory.surfableWaveSpeed(wc)
        forceBalance = ForceBalance(board, params, riderMassKg=const.defaultSurferMassKg)
        state = forceBalance.findEquilibrium(surfSpeed)
        totalWeightN = forceBalance.totalWeightN

        time = ValueTracker(0)

        # -------------------------------------------------------
        # 1. Camera Setup (0-1s)
        # -------------------------------------------------------

        self.set_camera_orientation(
            phi=60 * math.pi / 180,
            theta=-45 * math.pi / 180,
            zoom=0.7,
        )

        # -------------------------------------------------------
        # 2. Wave Surface Appears (1-4s)
        # -------------------------------------------------------

        waveSurface = createWaveSurface3D(
            waveTheory, wc, 0,
            xRange=(-6.0, 6.0),
            yRange=(-3.0, 3.0),
            resolution=(60, 20),
            opacity=0.4,
        )

        self.play(Create(waveSurface), run_time=2.0)
        self.wait(1.0)

        # -------------------------------------------------------
        # 3. Board Outline Appears (4-7s)
        # -------------------------------------------------------

        boardOutline = createBoardOutline3D(
            board, params,
            color=SHORTBOARD_COLOR,
            fillOpacity=0.7,
        )

        boardX = 1.0
        positionBoardOnWave3D(
            boardOutline, board, waveTheory, wc,
            boardX=boardX, t=0,
            draftM=state.draftM,
        )

        self.play(FadeIn(boardOutline), run_time=1.5)
        self.wait(1.0)

        # -------------------------------------------------------
        # 4. Wave Propagates with Board (7-14s)
        # -------------------------------------------------------

        # Animate by rebuilding surface at each timestep
        nSteps = 14
        totalDuration = 7.0
        dt = totalDuration / nSteps

        for i in range(nSteps):
            t = (i + 1) * dt
            time.set_value(t)

            newSurface = createWaveSurface3D(
                waveTheory, wc, t,
                xRange=(-6.0, 6.0),
                yRange=(-3.0, 3.0),
                resolution=(60, 20),
                opacity=0.4,
            )

            # Move board with wave
            omega = wc.angularFrequency
            k = waveTheory.solveDispersionRelation(omega, wc.depth)
            phaseShift = omega * t / k
            currentBoardX = boardX + (phaseShift % (2 * math.pi / k)) - (2 * math.pi / k) / 4
            while currentBoardX > 4.0:
                currentBoardX -= 2 * math.pi / k
            while currentBoardX < -4.0:
                currentBoardX += 2 * math.pi / k

            positionBoardOnWave3D(
                boardOutline, board, waveTheory, wc,
                boardX=currentBoardX, t=t,
                draftM=state.draftM,
            )

            self.play(
                waveSurface.animate.become(newSurface),
                run_time=dt,
                rate_func=linear,
            )

        # -------------------------------------------------------
        # 5. Force Arrows in 3D (14-18s)
        # -------------------------------------------------------

        boardCenter3D = boardOutline.get_center()
        forces3D = createForceBalance3D(
            state, totalWeightN, boardCenter3D,
        )

        for forceName in ['weight', 'buoyancy', 'lift', 'drag']:
            self.play(FadeIn(forces3D[forceName]), run_time=0.8)

        self.wait(0.5)

        # -------------------------------------------------------
        # 6. Camera Orbit (18-20s)
        # -------------------------------------------------------

        self.begin_ambient_camera_rotation(rate=0.3)
        self.wait(2.0)
        self.stop_ambient_camera_rotation()

        # -------------------------------------------------------
        # Final Fade
        # -------------------------------------------------------

        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=1.5,
        )
        self.wait(0.5)
