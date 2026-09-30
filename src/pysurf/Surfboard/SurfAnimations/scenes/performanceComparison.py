# -- Performance Comparison Scene -- #

'''

Manim animation comparing three surfboard types riding the same wave.

Shows shortboard, longboard, and fish side-by-side with force vectors,
performance metrics table, and L/D ratio bar chart comparison.

Sequence (~45 seconds):
1. Title card
2. Three board profiles appear stacked
3. Wave appears, boards ride wave
4. Force vectors on each board
5. Performance metrics table
6. L/D ratio bar chart
7. Summary insight

Sean Bowman [02/04/2026]

'''

import math
import os
import sys

import numpy as np
from manim import (
    Scene, Text, MathTex, VGroup, VMobject,
    ValueTracker, Arrow, Rectangle, Line,
    Create, FadeIn, FadeOut, Write, GrowFromEdge, Transform,
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
    SHORTBOARD_COLOR, LONGBOARD_COLOR, FISH_COLOR,
    BOARD_COLORS,
)
from pysurf.Surfboard.SurfAnimations.components.waveSurface import (
    createWaveLine, updateWaveLine,
    createWaterFill, updateWaterFill,
)
from pysurf.Surfboard.SurfAnimations.components.boardProfile import createBoardProfile, positionBoardOnWave
from pysurf.Surfboard.SurfAnimations.components.forceArrows import createForceBalance

#--------------------------------------------------------------------#
# -- Board Data Container -- #
#--------------------------------------------------------------------#

class _BoardData:

    '''Internal container for computed board data.'''

    def __init__(self, name, params, color):
        self.name = name
        self.params = params
        self.color = color
        self.board = BoardGeometry(params)
        self.forceBalance = ForceBalance(
            self.board, params,
            riderMassKg=const.defaultSurferMassKg,
        )
        self.state = None
        self.totalWeightN = self.forceBalance.totalWeightN
        self.shape = None

    def computeEquilibrium(self, speed: float) -> None:

        '''Compute planing state at the given speed.'''
        self.state = self.forceBalance.findEquilibrium(speed)

    def lengthFt(self) -> float:

        '''Board length in feet (for display).'''
        return self.params.length / 304.8

    def widthIn(self) -> float:

        '''Board width in inches (for display).'''
        return self.params.maxWidth / 25.4

    def thicknessIn(self) -> float:

        '''Board thickness in inches (for display).'''
        return self.params.maxThickness / 25.4

#--------------------------------------------------------------------#
# -- Performance Comparison Scene -- #
#--------------------------------------------------------------------#

class PerformanceComparison(Scene):

    '''

    Multi-board comparison animation.

    Compares shortboard, longboard, and fish side-by-side,
    showing force balance differences and performance metrics.

    '''

    def construct(self) -> None:
        self.camera.background_color = BG_COLOR

        # Physics setup
        waveTheory = LinearWaveTheory()
        wc = WaveConditions.typicalBeachBreak()
        surfSpeed = waveTheory.surfableWaveSpeed(wc)

        # Build board data
        boards = [
            _BoardData('Shortboard', SurfboardParameters.shortboard(), SHORTBOARD_COLOR),
            _BoardData('Longboard', SurfboardParameters.longboard(), LONGBOARD_COLOR),
            _BoardData('Fish', SurfboardParameters.fish(), FISH_COLOR),
        ]

        for bd in boards:
            bd.computeEquilibrium(surfSpeed)

        # Scene domain
        xMin, xMax = -7.0, 7.0
        bottomY = -3.5

        # -------------------------------------------------------
        # 1. Title (0-2s)
        # -------------------------------------------------------

        title = Text(
            'Board Comparison',
            font_size=42,
            color=WHITE,
        )
        subtitle = Text(
            'Shortboard vs Longboard vs Fish',
            font_size=24,
            color=CYAN,
        ).next_to(title, DOWN, buff=0.3)

        titleGroup = VGroup(title, subtitle)

        self.play(Write(title), run_time=0.8)
        self.play(Write(subtitle), run_time=0.8)
        self.wait(0.5)

        self.play(
            titleGroup.animate.scale(0.45).to_corner(UP + LEFT, buff=0.2),
            run_time=0.7,
        )

        # -------------------------------------------------------
        # 2. Three Boards Appear (2-6s)
        # -------------------------------------------------------

        # Create board profiles stacked vertically
        yPositions = [1.5, 0.0, -1.5]

        for bd, yPos in zip(boards, yPositions):
            bd.shape = createBoardProfile(
                bd.board, bd.params,
                color=bd.color,
                fillOpacity=0.8,
            )
            bd.shape.move_to(np.array([0, yPos, 0]))

            # Board label
            dimText = (
                f'{bd.name}\n'
                f'{bd.lengthFt():.0f}\' x {bd.widthIn():.1f}" x {bd.thicknessIn():.2f}"'
            )
            label = Text(
                dimText,
                font_size=14,
                color=bd.color,
            ).next_to(bd.shape, RIGHT, buff=0.3)

            self.play(
                FadeIn(bd.shape),
                Write(label),
                run_time=0.8,
            )

        self.wait(0.5)

        # -------------------------------------------------------
        # 3. Wave Appears (6-8s)
        # -------------------------------------------------------

        # Transition: move boards to wave-riding layout
        # Three separate wave+board rows
        waveRows = VGroup()
        rowYPositions = [2.0, 0.0, -2.0]

        # Fade existing labels and reposition
        self.play(
            *[FadeOut(mob) for mob in self.mobjects
              if mob not in [bd.shape for bd in boards] and mob != titleGroup],
            run_time=0.5,
        )

        # Create waves for each row
        wavesAndFills = []
        for bd, rowY in zip(boards, rowYPositions):
            # Move board to row position
            self.play(
                bd.shape.animate.move_to(np.array([1.0, rowY, 0])),
                run_time=0.3,
            )

            # Small wave for each row
            waveLine = createWaveLine(
                waveTheory, wc, 0,
                xMin=xMin, xMax=xMax,
                nPoints=150,
                strokeWidth=1.5,
            )
            waveLine.shift(UP * rowY)

            waterFill = createWaterFill(
                waveTheory, wc, 0,
                xMin=xMin, xMax=xMax,
                bottomY=bottomY,
                nPoints=150,
                opacity=0.15,
            )
            waterFill.shift(UP * rowY)

            self.play(
                Create(waveLine),
                FadeIn(waterFill),
                run_time=0.3,
            )
            wavesAndFills.append((waveLine, waterFill))

        # Position boards on waves
        for bd, rowY in zip(boards, rowYPositions):
            positionBoardOnWave(
                bd.shape, bd.board, waveTheory, wc,
                boardX=1.0, t=0,
                trimAngleDeg=bd.state.trimAngleDeg,
                draftM=bd.state.draftM,
            )
            bd.shape.shift(UP * rowY)

        self.wait(0.5)

        # -------------------------------------------------------
        # 4. Boards Ride Wave (8-14s)
        # -------------------------------------------------------

        time = ValueTracker(0)

        # Set up updaters for all rows
        for idx, (bd, rowY) in enumerate(zip(boards, rowYPositions)):
            wl, wf = wavesAndFills[idx]

            def makeWaveUpdater(waveMob, rowOffset):
                def updater(mob):
                    t = time.get_value()
                    updateWaveLine(mob, waveTheory, wc, t, xMin, xMax)
                    mob.shift(UP * rowOffset - mob.get_center()[1] * UP)
                    # Re-center vertically at rowOffset
                    currentCenter = mob.get_center()
                    mob.shift(UP * (rowOffset - currentCenter[1]))
                return updater

            def makeFillUpdater(fillMob, rowOffset):
                def updater(mob):
                    t = time.get_value()
                    updateWaterFill(mob, waveTheory, wc, t, xMin, xMax, bottomY)
                    currentCenter = mob.get_center()
                    mob.shift(UP * (rowOffset - currentCenter[1]))
                return updater

            def makeBoardUpdater(boardData, rowOffset):
                def updater(mob):
                    t = time.get_value()
                    omega = wc.angularFrequency
                    k = waveTheory.solveDispersionRelation(omega, wc.depth)
                    phaseX = omega * t / k
                    currentBoardX = 1.0 + (phaseX % (2 * math.pi / k)) - (2 * math.pi / k) / 4
                    while currentBoardX > xMax - 2:
                        currentBoardX -= 2 * math.pi / k
                    while currentBoardX < xMin + 2:
                        currentBoardX += 2 * math.pi / k

                    positionBoardOnWave(
                        mob, boardData.board, waveTheory, wc,
                        boardX=currentBoardX, t=t,
                        trimAngleDeg=boardData.state.trimAngleDeg,
                        draftM=boardData.state.draftM,
                    )
                    mob.shift(UP * rowOffset)
                return updater

            wl.add_updater(makeWaveUpdater(wl, rowY))
            wf.add_updater(makeFillUpdater(wf, rowY))
            bd.shape.add_updater(makeBoardUpdater(bd, rowY))

        self.play(
            time.animate.set_value(6.0),
            run_time=6.0,
            rate_func=linear,
        )

        # Remove updaters
        for idx, (bd, rowY) in enumerate(zip(boards, rowYPositions)):
            wl, wf = wavesAndFills[idx]
            wl.clear_updaters()
            wf.clear_updaters()
            bd.shape.clear_updaters()

        # -------------------------------------------------------
        # 5. Force Balance Comparison (14-22s)
        # -------------------------------------------------------

        forceGroups = {}
        for bd, rowY in zip(boards, rowYPositions):
            boardCenter = bd.shape.get_center()
            forces = createForceBalance(
                bd.state, bd.totalWeightN, boardCenter,
                trimAngleDeg=bd.state.trimAngleDeg,
                scaleFactor=0.002,
                fontSize=12,
            )
            forceGroups[bd.name] = forces

        # Show forces on all boards simultaneously
        for forceName in ['weight', 'buoyancy', 'lift', 'drag']:
            animations = []
            for bd in boards:
                animations.append(FadeIn(forceGroups[bd.name][forceName]))
            self.play(*animations, run_time=1.0)
            self.wait(0.5)

        self.wait(1.0)

        # -------------------------------------------------------
        # 6. Performance Metrics Table (22-30s)
        # -------------------------------------------------------

        # Fade forces and waves, transition to metrics
        fadeAnimations = []
        for bd in boards:
            for fName in ['weight', 'buoyancy', 'lift', 'drag']:
                fadeAnimations.append(FadeOut(forceGroups[bd.name][fName]))
        for wl, wf in wavesAndFills:
            fadeAnimations.extend([FadeOut(wl), FadeOut(wf)])
        for bd in boards:
            fadeAnimations.append(FadeOut(bd.shape))

        self.play(*fadeAnimations, run_time=0.8)

        # Build metrics table
        tableHeader = Text(
            'Board         Trim    L/D    Lift(N)  Drag(N)  Planing',
            font_size=16,
            font='Consolas',
            color=WHITE,
        ).to_edge(UP, buff=1.2)

        headerLine = Line(
            start=tableHeader.get_left() + DOWN * 0.2,
            end=tableHeader.get_right() + DOWN * 0.2,
            color=REFERENCE_LINE,
            stroke_width=1,
        )

        self.play(Write(tableHeader), Create(headerLine), run_time=0.8)

        tableRows = []
        for idx, bd in enumerate(boards):
            s = bd.state
            planingStr = 'Yes' if s.isPlaning else 'No'
            rowText = (
                f'{bd.name:<14s}{s.trimAngleDeg:>5.1f}\u00b0  '
                f'{s.liftToDrag:>5.1f}   '
                f'{s.liftForceN:>7.0f}  '
                f'{s.dragForceN:>7.0f}  '
                f'{planingStr:>7s}'
            )
            row = Text(
                rowText,
                font_size=15,
                font='Consolas',
                color=bd.color,
            )
            row.next_to(
                headerLine if idx == 0 else tableRows[-1],
                DOWN,
                buff=0.3,
            )
            tableRows.append(row)
            self.play(Write(row), run_time=0.8)

        self.wait(2.0)

        # -------------------------------------------------------
        # 7. L/D Ratio Bar Chart (30-38s)
        # -------------------------------------------------------

        # Fade table
        tableMobs = [tableHeader, headerLine] + tableRows
        self.play(*[FadeOut(m) for m in tableMobs], run_time=0.5)

        # Bar chart title
        chartTitle = Text(
            'Lift-to-Drag Ratio Comparison',
            font_size=28,
            color=WHITE,
        ).to_edge(UP, buff=0.8)

        self.play(Write(chartTitle), run_time=0.5)

        # Build bars
        maxLD = max(bd.state.liftToDrag for bd in boards)
        barWidth = 1.2
        barSpacing = 2.0
        barMaxHeight = 4.0
        barBottom = -2.5
        barStartX = -(len(boards) - 1) * barSpacing / 2

        barGroups = []
        for idx, bd in enumerate(boards):
            barX = barStartX + idx * barSpacing
            barHeight = (bd.state.liftToDrag / maxLD) * barMaxHeight
            if barHeight < 0.1:
                barHeight = 0.1

            bar = Rectangle(
                width=barWidth,
                height=barHeight,
                fill_color=bd.color,
                fill_opacity=0.8,
                stroke_color=bd.color,
                stroke_width=1,
            )
            bar.move_to(np.array([barX, barBottom + barHeight / 2, 0]))

            barLabel = Text(
                bd.name,
                font_size=16,
                color=bd.color,
            ).next_to(bar, DOWN, buff=0.2)

            barValue = Text(
                f'{bd.state.liftToDrag:.1f}',
                font_size=20,
                color=WHITE,
            ).next_to(bar, UP, buff=0.15)

            barGroup = VGroup(bar, barLabel, barValue)
            barGroups.append(barGroup)

        # Animate bars growing
        for bg in barGroups:
            self.play(
                GrowFromEdge(bg[0], DOWN),
                Write(bg[1]),
                Write(bg[2]),
                run_time=1.0,
            )
            self.wait(0.3)

        # Highlight best L/D
        bestIdx = max(range(len(boards)), key=lambda i: boards[i].state.liftToDrag)
        bestBar = barGroups[bestIdx][0]

        self.wait(1.5)

        # -------------------------------------------------------
        # 8. Summary (38-42s)
        # -------------------------------------------------------

        # Fade bars
        self.play(
            *[FadeOut(bg) for bg in barGroups],
            FadeOut(chartTitle),
            run_time=0.5,
        )

        bestBoard = boards[bestIdx]
        summaryText = Text(
            f'The {bestBoard.name.lower()} achieves the highest L/D ratio\n'
            f'({bestBoard.state.liftToDrag:.1f}) at surf speed ({surfSpeed:.1f} m/s),\n'
            f'meaning it converts forward motion to lift most efficiently.',
            font_size=22,
            color=WHITE,
            line_spacing=1.4,
        ).move_to(ORIGIN)

        self.play(Write(summaryText), run_time=2.0)
        self.wait(2.0)

        # -------------------------------------------------------
        # 9. Fade Out (42-45s)
        # -------------------------------------------------------

        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=1.5,
        )
        self.wait(0.5)
