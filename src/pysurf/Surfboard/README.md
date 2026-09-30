# Surfboard Domain

**Computational surfboard design, physics simulation, 3D geometry generation, and reference validation.**

This domain covers the full pipeline from parametric board shape definition through physics analysis, animated visualizations, interactive 3D viewing, and reverse-engineering validation against real reference boards.

---

## Quick Start

### Run the reverse-engineering demo

```bash
# From the repo root -- fits the Python parametric geometry to
# referenceThruster.stl, then launches the interactive comparison viewer
python codeInterface.py
```

### Generate a board with Python (parametric surface mesh)

```python
from pysurf.Surfboard import *

params = SurfboardParameters.shortboard()  # or .longboard(), .fish()
gen    = SurfaceMeshGenerator.fromPreset(params, 'standard')
gen.exportStl('shortboard.stl')
```

### Launch the interactive 3D web viewer

```bash
python -m http.server 8080
# Open: http://localhost:8080/src/pysurf/Surfboard/SurfViewer/viewer.html
```

### Render physics animations

```bash
python src/pysurf/Surfboard/SurfAnimations/render.py --scene wave_intro
python src/pysurf/Surfboard/SurfAnimations/render.py --all
python src/pysurf/Surfboard/SurfAnimations/render.py --list
```

---

## Module Guide

```
Surfboard/
├── SurfPhysics/          Python physics, geometry, validation, visualization
├── SurfAnimations/       Manim physics animation scenes
├── SurfViewer/           Three.js interactive 3D web viewer
├── configs/              JSON configuration files
└── documentation/        Technical writeups
```

### SurfPhysics

The core Python package. Import everything with:

```python
from pysurf.Surfboard import *
# or
from pysurf.Surfboard.SurfPhysics.geometry.parameters import SurfboardParameters
```

| Subpackage | Purpose |
|---|---|
| `geometry/` | Parametric board model: outline, rocker, cross-section, mesh generation |
| `waves/` | Linear wave theory: dispersion, phase/group velocity, pressure |
| `hydrodynamics/` | Drag, lift, buoyancy, planing force models |
| `visualization/` | Plotly analysis dashboards and the reference comparison viewer |
| `export/` | Viewer data and STL export utilities |
| `validation/` | Reverse engineering, mesh alignment, and deviation analysis |
| `optimization/` | Coordinate transform utilities (scale/axis auto-detection) |

### SurfAnimations

Manim scenes for physics education. Available scenes:

| Scene | Description |
|---|---|
| `wave_intro` | Linear wave theory: propagating wave, particle orbits, velocity field |
| `board_side` | 2D side view with force vectors and velocity field |
| `board_perspective` | 3D perspective view with camera orbit |
| `performance` | Shortboard vs longboard vs fish performance comparison |
| `sloshing_tank` | SPH water simulation particle playback |

### SurfViewer

Three.js web viewer with dual geometry modes (STL mesh or parametric surface), solid/wireframe/glass render modes, physics overlays, board comparison, and deviation heatmap. Reads JSON data from `SurfViewer/data/` exported by the Python pipeline.

---

## Board Parameters

All parameters are in millimeters. The `SurfboardParameters` dataclass holds the full board definition:

| Parameter | Description | Default (shortboard) |
|---|---|---|
| `length` | Total board length | 1828 mm (6'0") |
| `maxWidth` | Maximum beam width | 495 mm (19.5") |
| `maxThickness` | Maximum thickness | 62 mm (2.44") |
| `noseWidth` | Width 305 mm from nose | 300 mm |
| `tailWidth` | Width 305 mm from tail | 380 mm |
| `widePointOffset` | Wide point from board center (negative = toward nose) | -25 mm |
| `tailTipHalfWidth` | Half-width at tail tip | 15 mm |
| `noseRocker` | Nose rocker height | 120 mm (4.7") |
| `tailRocker` | Tail rocker height | 40 mm (1.6") |
| `deckCrown` | Deck dome height at centerline | 8 mm |
| `bottomConcave` | Single concave channel depth | 2 mm |

### Presets

| Parameter | Shortboard | Longboard | Fish |
|---|---|---|---|
| Length | 1828 mm (6'0") | 2743 mm (9'0") | 1676 mm (5'6") |
| Width | 495 mm (19.5") | 570 mm (22.5") | 533 mm (21") |
| Thickness | 62 mm (2.44") | 75 mm (3.0") | 65 mm (2.56") |
| Nose Rocker | 120 mm (4.7") | 180 mm (7.0") | 80 mm (3.1") |
| Tail Rocker | 40 mm (1.6") | 25 mm (1.0") | 30 mm (1.2") |
| Default Fins | Thruster | Single | Twin |

### Fin Configurations

| Configuration | Count | Characteristics |
|---|---|---|
| `thruster` | 3 (center + 2 side) | Versatile, balanced drive and pivot |
| `twin` | 2 (side only) | Fast and loose, skatey feel |
| `quad` | 4 (2 front + 2 rear) | Speed and hold in large surf |
| `single` | 1 (center) | Smooth, flowing turns; stable |

---

## Validation and Reverse Engineering

The validation pipeline compares any generated board against a reference STL:

```python
from pysurf.Surfboard.SurfPhysics.validation.reverseEngineer import ReverseEngineer

re       = ReverseEngineer('referenceThruster.stl', expectedLengthMm=1828.0)
profiles = re.extractProfiles(nStations=200)
params   = re.fitParameters(profiles)
errors   = re.compareToParametric(params, profiles)
print(f'Outline RMS: {errors["outlineRmsMm"]:.2f} mm')
```

The interactive comparison viewer (launched by `codeInterface.py`) overlays the Python parametric board on the reference with per-vertex deviation coloring: green inside the reference, red outside.

---

## Config Reference

Configs in `configs/*.json` control the full pipeline via the Python entry point:

```json
{
  "board": {
    "type": "shortboard",
    "finConfiguration": "thruster",
    "foamType": "pu"
  },
  "validation": {
    "run": true,
    "referencePath": "..."
  },
  "waves": { "height": 1.5, "period": 10.0, "depth": 2.5 },
  "rider": { "mass": 75.0 },
  "analysis": { "run": true, "showDashboard": true }
}
```

Preset configs: `shortboard_default.json`, `longboard_small.json`, `fish_overhead.json`.

---

## Documentation

Detailed writeups in [`documentation/`](documentation/):

- **[API Reference](documentation/apiReference.md)**: All importable classes and functions
- **[SurfPhysics Overview](documentation/surfPhysicsOverview.md)**: Wave theory, buoyancy, and force balance
- **[Parametric Surface Mesh](documentation/parametricSurfaceMeshOverview.md)**: Python STL generation algorithm
- **[SurfAnimations Overview](documentation/surfAnimationsOverview.md)**: Manim scene architecture
- **[Fin Dimensions Reference](documentation/finDimensionsReference.md)**: Reference fin measurements
- **[Surfboard Shaping References](documentation/surfboardShapingReferences.md)**: Industry shaping conventions

---

Sean Bowman
