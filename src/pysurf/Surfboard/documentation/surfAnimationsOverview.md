# SurfAnimations -- Manim Physics Education Module

An overview of the SurfAnimations module: what it does, how the scenes and
components fit together, and how to render or extend them. This module uses
[Manim](https://www.manim.community/) to produce animated videos that visualize
the wave physics and surfboard hydrodynamics computed by SurfPhysics.

Sean Bowman [02/05/2026]

---

## 1. What It Does

SurfAnimations turns the math from SurfPhysics into watchable animations.
Instead of staring at static plots or scrolling through tables of numbers,
you get video sequences that show waves propagating, boards riding those
waves, force vectors growing and shrinking in real time, and particles
sloshing around a tank.

The motivation is educational. Wave theory equations are hard to internalize
from a textbook. Seeing a propagating sinusoidal wave with labeled height,
wavelength, period, and phase speed -- then watching the orbital particle
motions decay with depth -- makes the physics click in a way that static
diagrams cannot. The same logic applies to planing hydrodynamics: watching
a shortboard ride a wave face with live force arrows for weight, buoyancy,
planing lift, and drag communicates the force balance far more effectively
than a free-body diagram on a whiteboard.

Every scene in this module pulls its physics data directly from SurfPhysics.
The wave elevations come from `LinearWaveTheory`, the board shapes come from
`BoardGeometry`, the forces come from `ForceBalance`. The animations do not
hardcode any physics -- they query the simulation models at render time, so
if you change a wave height or board parameter, the animation reflects it
automatically.

---

## 2. Available Scenes

The module ships with five registered scenes. Each one targets a specific
physics concept:

| Scene Key          | Class                     | File                                   | What It Shows                                                                 | Duration |
|--------------------|---------------------------|----------------------------------------|-------------------------------------------------------------------------------|----------|
| `wave_intro`       | `WavePhysicsIntro`        | `scenes/waveIntro.py`                  | Linear wave theory intro: propagating wave, labeled properties (H, L, T, c), particle orbits, velocity field, depth classification | ~30s     |
| `board_side`       | `BoardOnWaveSideView`     | `scenes/boardOnWave.py`                | Shortboard riding a wave in 2D side view with force vectors and velocity field | ~20s     |
| `board_perspective`| `BoardOnWavePerspective`  | `scenes/boardOnWave.py`                | Same board-on-wave concept but in a 2.5D ThreeDScene with a 3D wave surface, planform outline, 3D force arrows, and camera orbit | ~20s     |
| `performance`      | `PerformanceComparison`   | `scenes/performanceComparison.py`      | Shortboard vs longboard vs fish: three boards ride the same wave, force comparison, metrics table, and L/D ratio bar chart | ~45s     |
| `sloshing_tank`    | `SloshingTankAnimation`   | `scenes/sloshingTank.py`               | SPH sloshing tank playback: loads pre-computed JSON frame data, animates particles color-coded by velocity magnitude inside container walls | ~15-20s  |

---

## 3. How It Works

### Manim in 30 Seconds

Manim is a Python animation engine. You define a `Scene` subclass with a
`construct()` method, and inside that method you create graphical objects
(called *mobjects*), then call `self.play(...)` to animate them. Manim
renders each frame to image files, then stitches them into an MP4 using
ffmpeg.

The key concept is the **updater**: you attach a function to a mobject that
gets called every frame, and that function can recompute the mobject's
geometry based on a `ValueTracker`. This is how wave propagation works --
a `ValueTracker` holds the current simulation time, and updater functions
on the wave line and water fill polygon recompute the surface elevation
at each frame.

### The Component Architecture

Rather than having each scene build every graphical element from scratch,
SurfAnimations uses a **component pattern**. Reusable factory functions
in the `components/` directory create and update specific mobjects:

```
Scene (construct method)
  |
  +-- components/waveSurface    --> wave line, water fill, 3D surface
  +-- components/boardProfile   --> board side profile, 3D planform outline
  +-- components/forceArrows    --> weight/buoyancy/lift/drag arrows
  +-- components/velocityField  --> quiver field, particle orbits
  +-- components/particleField  --> SPH particle dots with color mapping
  +-- components/containerWalls --> open-top tank outline
```

A scene's `construct()` method orchestrates the sequence: create components,
attach updaters, play animations, detach updaters, create the next set of
components, and so on. The components themselves know nothing about
sequencing -- they just know how to draw one thing well.

This separation matters because multiple scenes reuse the same components.
The wave surface appears in `WavePhysicsIntro`, `BoardOnWaveSideView`,
`BoardOnWavePerspective`, and `PerformanceComparison`. The force arrows
appear in both board-on-wave scenes and the performance comparison. If you
improve the wave rendering (say, adding a gradient fill), every scene that
uses `createWaveLine` benefits automatically.

---

## 4. API Quick Start

The simplest way to render a scene from Python code:

```python
from SurfAnimations import renderScene, SCENES

# Render a single scene at high quality (1080p, 60fps)
renderScene('wave_intro', quality='high')

# See what scenes are available
for name, info in SCENES.items():
    print(f'{name}: {info["description"]}')

# Render everything
from SurfAnimations import renderAll
results = renderAll(quality='medium')
```

The `renderScene` function builds a Manim CLI command under the hood and
runs it as a subprocess. Output videos land in `SurfAnimations/media/`.

You can also import the scene classes directly if you want to work with
them in a Manim workflow:

```python
from SurfAnimations import WavePhysicsIntro, BoardOnWaveSideView
```

---

## 5. CLI Usage

The `render.py` script provides a command-line interface:

```bash
# Render a specific scene
python SurfAnimations/render.py --scene wave_intro

# Render at a specific quality
python SurfAnimations/render.py --scene board_side --quality low

# Render all scenes
python SurfAnimations/render.py --all

# Render all at 4K
python SurfAnimations/render.py --all --quality fourk

# List available scenes
python SurfAnimations/render.py --list
```

Short flags work too: `-s` for `--scene`, `-q` for `--quality`, `-a` for
`--all`, `-l` for `--list`.

On Windows with a conda environment, the script automatically searches for
ffmpeg in the conda `Library/bin` directory and adds it to PATH if needed.
If ffmpeg is not found, you will see a warning with install instructions.

---

## 6. Scene Architecture

Each scene follows the same structural pattern. Here is the flow using
`WavePhysicsIntro` as an example:

### 1. Physics Setup

The scene creates instances of the SurfPhysics models it needs:

```python
waveTheory = LinearWaveTheory()
wc = WaveConditions.typicalBeachBreak()  # H=1.5m, T=10s, d=2.5m
```

Everything downstream -- wave shape, velocities, orbital paths, depth
classification -- comes from these objects. The scene never hardcodes
a wave height or wavelength; it asks the physics model.

### 2. Time Tracking

A Manim `ValueTracker` holds the simulation clock:

```python
time = ValueTracker(0)
```

Updater functions read `time.get_value()` to know what instant to render.
When the scene calls `time.animate.set_value(5.0)` with `run_time=5.0`,
Manim smoothly increments the tracker from 0 to 5 over five seconds of
real-time video, and every mobject with an updater redraws itself each frame.

### 3. Sequenced Sections

Scenes are divided into timestamped sections using comment blocks:

```python
# -------------------------------------------------------
# 1. Title Card (0-2s)
# -------------------------------------------------------
```

Each section creates mobjects, plays animations, and optionally pauses or
removes updaters. This makes the animation timeline readable in the source
code -- you can scan the section headers to understand the narrative flow.

### 4. Component Calls

Scenes delegate all geometry creation to component functions:

```python
waveLine = createWaveLine(waveTheory, wc, 0, xMin, xMax)
waterFill = createWaterFill(waveTheory, wc, 0, xMin, xMax, bottomY)
velField = createVelocityField(waveTheory, wc, currentTime, ...)
```

### 5. Physics-Driven Data

The `BoardOnWaveSideView` scene demonstrates this most clearly. It creates
a `ForceBalance` instance, calls `findEquilibrium(surfSpeed)` to get the
planing state (trim angle, draft, lift force, drag force, L/D ratio), and
then passes that state to both the board positioning function and the force
arrow factory. The animation is not artistic license -- it is a direct
visualization of the computed equilibrium.

---

## 7. Reusable Components

All components live in `SurfAnimations/components/`. Each file exports
factory functions that return Manim mobjects.

### waveSurface (`components/waveSurface.py`)

Creates and updates wave geometry in both 2D and 3D.

- **`createWaveLine()`** -- Produces a `VMobject` polyline by sampling
  `LinearWaveTheory.surfaceElevation()` across the scene width. Used for
  the 2D water surface in every wave scene.
- **`updateWaveLine()`** -- Recomputes the polyline points at a new time
  value. Called from updater functions each frame.
- **`createWaterFill()`** -- Builds a filled `Polygon` from the wave
  surface down to the scene floor, giving the visual impression of a body
  of water beneath the wave.
- **`updateWaterFill()`** -- Updates the fill polygon for a new time step.
- **`createWaveSurface3D()`** -- Produces a Manim `Surface` for
  `ThreeDScene`. The elevation varies with x (propagation direction) and
  is constant in y, since the linear wave model is 2D. The y axis provides
  visual depth for the perspective view.

### boardProfile (`components/boardProfile.py`)

Renders surfboard shapes by querying `BoardGeometry`.

- **`createBoardProfile()`** -- Creates a 2D side-view board profile as a
  closed `Polygon`. Samples deck and bottom heights along the centerline,
  including rocker curvature. Units are meters (1 Manim unit = 1 meter).
- **`positionBoardOnWave()`** -- Places a board profile mobject on the wave
  surface at a given x position, rotated by the equilibrium trim angle and
  offset by the computed draft depth.
- **`createBoardOutline3D()`** -- Creates a 3D planform outline (top-down
  view) by sampling half-widths along the board length. The z coordinate
  follows the rocker profile so the outline sits on the wave surface.
- **`positionBoardOnWave3D()`** -- Positions the 3D outline on the wave
  surface at a given x and time.

### forceArrows (`components/forceArrows.py`)

Draws force vector arrows with magnitude labels.

- **`createForceArrow()`** -- Produces a single 2D `Arrow` with a text
  label showing the force name and magnitude in Newtons. Arrow length is
  proportional to force magnitude (capped at a maximum).
- **`createForceBalance()`** -- Creates the full set of four arrows
  (weight, buoyancy, planing lift, drag) from a `PlaningState` object.
  Returns a dict keyed by force name so each arrow can be animated
  independently.
- **`createForceArrow3D()` / `createForceBalance3D()`** -- 3D equivalents
  using `Arrow3D` for the perspective scene.

The force scale factor is calibrated so that a typical combined weight of
~830 N produces an arrow about 2.5 Manim units long. Lift acts perpendicular
to the board bottom (angled by trim), drag acts opposite to the direction of
motion, buoyancy acts straight up, and weight acts straight down.

### velocityField (`components/velocityField.py`)

Visualizes the orbital velocity pattern beneath the wave.

- **`createVelocityField()`** -- Samples `LinearWaveTheory.velocityField()`
  on a grid of (x, z) points and creates a quiver-style arrow field. Arrow
  length is proportional to velocity magnitude, and color interpolates from
  blue (slow) to cyan (fast). Points above the wave surface are skipped.
- **`createParticleOrbits()`** -- Traces particle trajectories at specified
  depths by numerically integrating the velocity field over one wave period.
  In deep water these are circles; they flatten to ellipses and shrink with
  depth. Deeper orbits are rendered with lower opacity to reinforce the
  exponential decay visually.

### particleField (`components/particleField.py`)

Renders SPH simulation particle data as colored dots.

- **`createParticleField()`** -- Creates a `VGroup` of `Dot` mobjects from
  an array of particle positions. Supports color mapping by a scalar field
  (typically velocity magnitude), with linear interpolation from cyan (slow)
  to red (fast). Positions are transformed from simulation coordinates to
  Manim coordinates using a scale factor and offset.
- **`updateParticleField()`** -- Updates existing dot positions and colors
  in-place from new frame data. This avoids recreating hundreds of dots
  every frame, which would be prohibitively slow.

### containerWalls (`components/containerWalls.py`)

Draws a simple open-top rectangular tank.

- **`createContainerWalls()`** -- Creates three `Line` mobjects (left wall,
  bottom, right wall) representing a tank cross-section. The top is left
  open for the free surface. Used by the sloshing tank scene.

---

## 8. Adding New Scenes

To add a new animation, follow this pattern:

### Step 1: Create a Scene File

Add a new file in `SurfAnimations/scenes/`. Your scene class subclasses
Manim's `Scene` (or `ThreeDScene` for 3D) and implements `construct()`:

```python
# SurfAnimations/scenes/myNewScene.py

from manim import Scene, Text, Write, FadeOut
from SurfAnimations.utils.manimTheme import BG_COLOR, WHITE
from SurfAnimations.components.waveSurface import createWaveLine

class MyNewScene(Scene):
    '''Describe what physics this scene demonstrates.'''

    def construct(self):
        self.camera.background_color = BG_COLOR

        # 1. Set up physics models
        # 2. Create components
        # 3. Animate sequence
        # 4. Fade out

        title = Text('My New Scene', font_size=42, color=WHITE)
        self.play(Write(title))
        self.wait(2.0)
        self.play(FadeOut(title))
```

### Step 2: Register in the SCENES Dict

Open `SurfAnimations/render.py` and add an entry to the `SCENES` dict:

```python
SCENES = {
    # ... existing scenes ...
    'my_new_scene': {
        'file': 'SurfAnimations/scenes/myNewScene.py',
        'class': 'MyNewScene',
        'description': 'Brief description of the scene',
    },
}
```

### Step 3: Export from the Package

Add the import to `SurfAnimations/__init__.py`:

```python
from SurfAnimations.scenes.myNewScene import MyNewScene
```

### Step 4: Render and Iterate

```bash
python SurfAnimations/render.py --scene my_new_scene --quality low
```

Start with `low` quality (480p, 15fps) during development. Renders take
seconds instead of minutes, and you can iterate quickly on timing and layout.
Switch to `high` or `fourk` once the content is finalized.

---

## 9. Quality Settings

The render system supports four quality presets, each mapping to a Manim
quality flag:

| Preset   | Flag  | Resolution | FPS | Use Case                                       |
|----------|-------|------------|-----|-------------------------------------------------|
| `low`    | `-ql` | 480p       | 15  | Fast iteration during development                |
| `medium` | `-qm` | 720p       | 30  | Draft review, sharing previews                   |
| `high`   | `-qh` | 1080p      | 60  | Final output for presentations                   |
| `fourk`  | `-qk` | 4K         | 60  | High-resolution archival, large displays          |

Pass the quality via the `--quality` CLI flag or the `quality` parameter in
the Python API:

```python
renderScene('wave_intro', quality='low')      # quick test
renderScene('wave_intro', quality='fourk')    # publication quality
```

Rendered videos are written to `SurfAnimations/media/`, organized by quality
level in Manim's default directory structure.

---

## 10. Theme and Visual Consistency

All color definitions live in `SurfAnimations/utils/manimTheme.py`. The
palette uses Material Design colors and mirrors the SurfPhysics Plotly theme
so that animations, dashboards, and the Three.js viewer all share a
consistent visual language.

Key semantic color assignments:

| Color Constant      | Hex       | Used For                          |
|---------------------|-----------|-----------------------------------|
| `WAVE_COLOR`        | `#42A5F5` | Wave surface line                 |
| `WATER_FILL`        | `#1a3a5c` | Filled water region               |
| `BG_COLOR`          | `#1a1a2e` | Scene background                  |
| `WEIGHT_COLOR`      | `#EF5350` | Gravity / weight force arrows     |
| `BUOYANCY_COLOR`    | `#66BB6A` | Buoyancy force arrows             |
| `LIFT_COLOR`        | `#42A5F5` | Planing lift force arrows         |
| `DRAG_COLOR`        | `#FFA726` | Drag force arrows                 |
| `SHORTBOARD_COLOR`  | `#42A5F5` | Shortboard in comparisons         |
| `LONGBOARD_COLOR`   | `#66BB6A` | Longboard in comparisons          |
| `FISH_COLOR`        | `#FFA726` | Fish in comparisons               |

If you add a new scene, use these constants rather than hardcoding hex values.
The theme file is the single source of truth for the project's visual identity.
