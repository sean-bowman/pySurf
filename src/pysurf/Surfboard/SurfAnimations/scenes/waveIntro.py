# -- Wave Physics Introduction Scene -- #

'''

Manim animation introducing linear wave theory concepts.

Sequence (~30 seconds):
1. Title card with wave equation
2. Animated propagating wave with water fill
3. Labeled wave properties (H, L, T, c)
4. Particle orbital motions at depth
5. Velocity field visualization
6. Wave classification and energy

Sean Bowman [02/04/2026]

'''

import os
import sys

import numpy as np
from manim import (
    Scene, Text, MathTex, VGroup, VMobject,
    ValueTracker, Arrow, DashedLine, Dot, Line,
    Create, FadeIn, FadeOut, Write, Indicate, MoveAlongPath,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
    linear, rate_functions,
    config,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pysurf.Surfboard.SurfPhysics.waves.linearWaveTheory import LinearWaveTheory
from pysurf.Surfboard.SurfPhysics.waves.waveConditions import WaveConditions
from pysurf.Surfboard.SurfAnimations.utils.manimTheme import (
    WAVE_COLOR, WATER_FILL, BG_COLOR, WHITE, CYAN, BLUE,
    RED, GREEN, ORANGE, REFERENCE_LINE,
)
from pysurf.Surfboard.SurfAnimations.components.waveSurface import (
    createWaveLine, updateWaveLine,
    createWaterFill, updateWaterFill,
)
from pysurf.Surfboard.SurfAnimations.components.velocityField import (
    createVelocityField, createParticleOrbits,
)

class WavePhysicsIntro(Scene):

    '''

    Animation introducing linear (Airy) wave theory.

    Shows a propagating wave with labeled properties, particle
    orbital motions, and a velocity field visualization.

    '''

    def construct(self) -> None:
        # Dark background
        self.camera.background_color = BG_COLOR

        # Physics setup
        waveTheory = LinearWaveTheory()
        wc = WaveConditions.typicalBeachBreak()  # H=1.5m, T=10s, d=2.5m

        # Precompute wave properties
        waveLen = waveTheory.waveLength(wc)
        waveSpd = waveTheory.waveSpeed(wc)
        energy = waveTheory.energyDensity(wc)
        depthClass = waveTheory.depthClassification(wc)

        # Scene domain
        xMin, xMax = -7.0, 7.0
        bottomY = -3.5

        # Time tracker for wave animation
        time = ValueTracker(0)

        # -------------------------------------------------------
        # 1. Title Card (0-2s)
        # -------------------------------------------------------
        title = Text(
            'Linear Wave Theory',
            font_size=48,
            color=WHITE,
        )
        title.to_edge(UP, buff=0.5)

        equation = MathTex(
            r'\eta(x,t) = \frac{H}{2} \cos(kx - \omega t)',
            font_size=36,
            color=CYAN,
        )
        equation.next_to(title, DOWN, buff=0.4)

        self.play(Write(title), run_time=1.0)
        self.play(Write(equation), run_time=1.0)
        self.wait(0.5)

        # Fade title up and out of the way
        titleGroup = VGroup(title, equation)
        self.play(
            titleGroup.animate.scale(0.5).to_corner(UP + RIGHT, buff=0.3),
            run_time=0.8,
        )

        # -------------------------------------------------------
        # 2. Wave Appears + Propagation (2-6s)
        # -------------------------------------------------------

        # Create initial wave line and water fill
        waveLine = createWaveLine(waveTheory, wc, 0, xMin, xMax)
        waterFill = createWaterFill(waveTheory, wc, 0, xMin, xMax, bottomY)

        # Still water reference line
        stillWater = DashedLine(
            start=np.array([xMin, 0, 0]),
            end=np.array([xMax, 0, 0]),
            color=REFERENCE_LINE,
            dash_length=0.15,
            stroke_width=1,
            stroke_opacity=0.4,
        )
        swlLabel = Text(
            'SWL',
            font_size=14,
            color=REFERENCE_LINE,
        ).next_to(stillWater, RIGHT, buff=0.2)

        self.play(
            Create(stillWater),
            FadeIn(swlLabel),
            run_time=0.5,
        )
        self.play(
            Create(waveLine),
            FadeIn(waterFill),
            run_time=1.5,
        )

        # Animate wave propagation for a couple seconds
        def waveUpdater(mob):
            t = time.get_value()
            updateWaveLine(mob, waveTheory, wc, t, xMin, xMax)

        def fillUpdater(mob):
            t = time.get_value()
            updateWaterFill(mob, waveTheory, wc, t, xMin, xMax, bottomY)

        waveLine.add_updater(waveUpdater)
        waterFill.add_updater(fillUpdater)

        self.play(
            time.animate.set_value(2.0),
            run_time=2.0,
            rate_func=linear,
        )

        # -------------------------------------------------------
        # 3. Wave Property Labels (6-12s)
        # -------------------------------------------------------

        # Pause wave for labeling
        waveLine.remove_updater(waveUpdater)
        waterFill.remove_updater(fillUpdater)
        currentTime = time.get_value()

        # Find a crest and trough position for labeling
        # Crest is where cos(kx - wt) = 1, i.e., kx - wt = 0
        omega = wc.angularFrequency
        k = waveTheory.solveDispersionRelation(omega, wc.depth)
        crestX = (omega * currentTime) / k
        # Keep crest in visible range
        while crestX > xMax - 1:
            crestX -= 2 * np.pi / k
        while crestX < xMin + 1:
            crestX += 2 * np.pi / k

        troughX = crestX + np.pi / k
        crestY = wc.amplitude
        troughY = -wc.amplitude

        # Wave Height H: vertical double-arrow
        hArrowStart = np.array([crestX + 1.5, troughY, 0])
        hArrowEnd = np.array([crestX + 1.5, crestY, 0])
        hArrow = Arrow(
            start=hArrowStart, end=hArrowEnd,
            color=RED, stroke_width=3, buff=0,
        )
        hArrowDown = Arrow(
            start=hArrowEnd, end=hArrowStart,
            color=RED, stroke_width=3, buff=0,
        )
        hLabel = Text(
            f'H = {wc.height:.1f} m',
            font_size=20,
            color=RED,
        ).next_to(hArrow, RIGHT, buff=0.15)

        self.play(Create(hArrow), Create(hArrowDown), Write(hLabel), run_time=1.0)

        # Wavelength L: horizontal double-arrow between crests
        nextCrestX = crestX + 2 * np.pi / k
        lArrowY = crestY + 0.4
        lArrowStart = np.array([crestX, lArrowY, 0])
        lArrowEnd = np.array([nextCrestX, lArrowY, 0])

        if nextCrestX <= xMax:
            lArrow = Arrow(
                start=lArrowStart, end=lArrowEnd,
                color=GREEN, stroke_width=3, buff=0,
            )
            lArrowBack = Arrow(
                start=lArrowEnd, end=lArrowStart,
                color=GREEN, stroke_width=3, buff=0,
            )
            lLabel = Text(
                f'L = {waveLen:.1f} m',
                font_size=20,
                color=GREEN,
            )
            lLabel.move_to((lArrowStart + lArrowEnd) / 2 + np.array([0, 0.35, 0]))

            self.play(
                Create(lArrow), Create(lArrowBack), Write(lLabel),
                run_time=1.0,
            )
        else:
            # If wavelength doesn't fit, just show text
            lLabel = Text(
                f'L = {waveLen:.1f} m',
                font_size=20,
                color=GREEN,
            ).move_to(np.array([0, crestY + 0.8, 0]))
            lArrow = None
            lArrowBack = None
            self.play(Write(lLabel), run_time=1.0)

        # Period T and Phase Speed c: text annotations
        periodLabel = Text(
            f'T = {wc.period:.0f} s',
            font_size=20,
            color=ORANGE,
        ).to_edge(LEFT, buff=0.5).shift(DOWN * 0.5)

        speedArrow = Arrow(
            start=np.array([crestX - 0.8, crestY + 1.0, 0]),
            end=np.array([crestX + 0.8, crestY + 1.0, 0]),
            color=CYAN,
            stroke_width=3,
        )
        speedLabel = Text(
            f'c = {waveSpd:.1f} m/s',
            font_size=20,
            color=CYAN,
        ).next_to(speedArrow, UP, buff=0.1)

        self.play(
            Write(periodLabel),
            Create(speedArrow), Write(speedLabel),
            run_time=1.0,
        )
        self.wait(0.5)

        # -------------------------------------------------------
        # 4. Particle Orbits (12-18s)
        # -------------------------------------------------------

        # Fade out property labels
        labelsToFade = [hArrow, hArrowDown, hLabel, periodLabel, speedArrow, speedLabel]
        if lArrow is not None:
            labelsToFade.extend([lArrow, lArrowBack])
        labelsToFade.append(lLabel)

        self.play(
            *[FadeOut(m) for m in labelsToFade],
            run_time=0.8,
        )

        # Resume wave propagation
        waveLine.add_updater(waveUpdater)
        waterFill.add_updater(fillUpdater)

        # Create particle orbits
        orbitPaths = createParticleOrbits(
            waveTheory, wc,
            xCenter=0.0,
            depths=[-0.3, -0.8, -1.5, -2.2],
        )

        # Particle dots
        particles = VGroup()
        for orbit in orbitPaths:
            dot = Dot(
                point=orbit.get_start(),
                radius=0.06,
                color=CYAN,
            )
            particles.add(dot)

        orbitLabel = Text(
            'Orbital Motion',
            font_size=22,
            color=CYAN,
        ).to_edge(LEFT, buff=0.5).shift(DOWN * 2.5)

        self.play(
            FadeIn(orbitPaths),
            FadeIn(particles),
            Write(orbitLabel),
            run_time=1.0,
        )

        # Animate particles along orbits
        self.play(
            time.animate(rate_func=linear).set_value(currentTime + 12.0),
            *[
                MoveAlongPath(dot, orbit, rate_func=linear)
                for dot, orbit in zip(particles, orbitPaths)
            ],
            run_time=4.0,
        )
        currentTime = time.get_value()

        # -------------------------------------------------------
        # 5. Velocity Field (18-24s)
        # -------------------------------------------------------

        # Fade orbits
        self.play(
            FadeOut(orbitPaths), FadeOut(particles), FadeOut(orbitLabel),
            run_time=0.8,
        )

        # Create velocity field snapshot
        velField = createVelocityField(
            waveTheory, wc, currentTime,
            xMin=-5.0, xMax=5.0,
            zMin=-2.5, zMax=-0.2,
            nX=14, nZ=5,
        )

        velLabel = Text(
            'Velocity Field',
            font_size=22,
            color=CYAN,
        ).to_edge(LEFT, buff=0.5).shift(DOWN * 2.5)

        self.play(FadeIn(velField), Write(velLabel), run_time=1.0)

        # Animate velocity field updating with wave
        for _ in range(4):
            self.play(
                time.animate.set_value(time.get_value() + wc.period / 4),
                run_time=1.0,
                rate_func=linear,
            )
            newField = createVelocityField(
                waveTheory, wc, time.get_value(),
                xMin=-5.0, xMax=5.0,
                zMin=-2.5, zMax=-0.2,
                nX=14, nZ=5,
            )
            self.play(velField.animate.become(newField), run_time=0.01)
            velField = newField

        # -------------------------------------------------------
        # 6. Wave Classification (24-28s)
        # -------------------------------------------------------

        self.play(FadeOut(velField), FadeOut(velLabel), run_time=0.5)

        classText = Text(
            f'Depth Classification: {depthClass.capitalize()} Water',
            font_size=24,
            color=WHITE,
        ).move_to(np.array([0, -2.0, 0]))

        energyText = Text(
            f'Energy Density: {energy:.0f} J/m\u00b2',
            font_size=20,
            color=ORANGE,
        ).next_to(classText, DOWN, buff=0.3)

        conditionsText = Text(
            f'H = {wc.height:.1f} m    T = {wc.period:.0f} s    d = {wc.depth:.1f} m',
            font_size=18,
            color=REFERENCE_LINE,
        ).next_to(energyText, DOWN, buff=0.3)

        self.play(
            Write(classText),
            Write(energyText),
            Write(conditionsText),
            run_time=1.5,
        )

        # Continue wave propagation during display
        self.play(
            time.animate.set_value(time.get_value() + 3.0),
            run_time=3.0,
            rate_func=linear,
        )

        # -------------------------------------------------------
        # 7. Fade Out (28-30s)
        # -------------------------------------------------------

        waveLine.remove_updater(waveUpdater)
        waterFill.remove_updater(fillUpdater)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=1.5,
        )
        self.wait(0.5)
