# Parametric Surface Mesh Generator -- Overview & Theory

A detailed walkthrough of the parametric surface mesh generation pipeline:
how it builds surfboard geometry from cross-section contours, why the approach
works, and how the reverse engineering tool closes the loop between generated
and reference geometry.

For the physics pipeline, see `documentation/surfPhysicsOverview.md`.

Sean Bowman [02/07/2026]

---

## 1. What It Does

The parametric surface mesh generator builds a watertight triangle mesh of
a surfboard directly from the parametric model -- no voxels, no PicoGK,
no .NET runtime. You hand it a `SurfboardParameters` object, and it returns
a `trimesh.Trimesh` ready for STL export, volume computation, or
visualization.

The core idea is simple: define the board as a series of closed cross-section
contour rings from nose to tail, then stitch adjacent rings together with
triangle strips. The nose and tail are capped to form a closed solid. The
result is a smooth, watertight mesh with exactly the geometric fidelity you
ask for -- hundreds of longitudinal stations and dozens of lateral sample
points per cross-section.

The surface mesh generator is a pure-Python STL generation pipeline: no
external geometry kernel, no compiled runtime dependency, no boolean
operations. It trades away boolean-based features (fin attachment, swallow
tail notches) for a direct, inspectable triangulation that runs in under a
second at standard resolution.

The surface mesh generator lives in `SurfPhysics/geometry/surfaceMeshGenerator.py`.
The reverse engineering tool lives in `SurfPhysics/validation/reverseEngineer.py`.
The export wrapper lives in `SurfPhysics/export/stlExporter.py`.

---

## 2. Why Direct Triangulation?

A voxel-based approach has one significant trade-off: resolution. Voxels
discretize space onto a regular grid, so the achievable surface smoothness
is bounded by voxel size. A 0.25 mm voxel grid produces excellent results
but requires substantial memory and computation time. And because the voxel
representation is inherently volumetric, you cannot easily inspect
intermediate geometric quantities like cross-section profiles or rocker
curves -- you only get the final mesh.

Direct triangulation sidesteps these issues:

- **Exact parametric geometry.** Each vertex sits exactly on the
  parametric surface. There is no discretization error beyond the
  station/lateral sampling density, and that density is user-controlled.

- **Lightweight.** No geometry kernel needed. The mesh is built from numpy
  arrays and assembled by trimesh. Generation takes under a second for
  standard resolution on modest hardware.

- **Inspectable.** The intermediate cross-section contours are available
  for visualization, validation, and comparison against reference geometry.
  This is what makes the reverse engineering workflow possible.

- **Predictable mesh topology.** The triangle count is deterministic:
  `2 * M * (N - 1)` body faces plus cap faces, where `M` is points per
  contour and `N` is the number of non-degenerate stations. There are no
  marching-cubes artifacts, no degenerate triangles from voxel boundaries.

The trade-off is that direct triangulation does not support boolean
operations. Fin attachment, swallow tail notches, and other boolean
modifications require voxels (or a proper B-rep kernel). The surface mesh
generator produces the board body only.

---

## 3. The Cross-Section Stitching Algorithm

The algorithm has four stages: station sampling, contour construction,
quad-strip triangulation, and end capping.

### 3.1 Station Sampling

The board length is divided into `N` equally-spaced longitudinal stations
from `t = 0` (nose tip) to `t = 1` (tail tip). At each station, the
existing parametric classes provide the geometric quantities:

- `Outline.getHalfWidth(t)` -- the board's half-width at this station (mm)
- `RockerProfile.getRockerHeight(t)` -- the bottom Z-offset from the rocker
  datum (mm)
- `CrossSection.getDeckHeight(t, lf)` -- the deck surface Z above the rocker
  baseline at lateral fraction `lf`
- `CrossSection.getBottomHeight(t, lf)` -- the bottom surface Z below the
  rocker baseline at lateral fraction `lf`

If the half-width at a station is less than 0.1 mm, the station is considered
a **tip** -- it collapses to a single point at `[x, 0, rockerZ]`. This
happens at the nose tip (t = 0) and at the tail tip (t = 1) for boards with
a pointed tail. Tip stations are used for fan capping rather than quad-strip
triangulation.

### 3.2 Contour Construction

At each non-tip station, a closed cross-section contour is built by tracing
four segments around the full perimeter of the board:

```
                      Segment 1: Right Deck (lf 0→1)
                        centerline → right rail
    ┌─────────────────────────────────────────────────┐
    │                   DECK (top)                    │
    │                                                 │ Segment 4
    │            Segment 4:                           │ (left deck)
    │            Left Deck                            │ lf 1→0
    │            (lf 1→0, Y negated)                  │
    │                                                 │
    ├─ ─ ─ ─ ─ ─ ─ ─ ─ centerline ─ ─ ─ ─ ─ ─ ─ ─ ─┤
    │                                                 │
    │            Segment 3:                           │ Segment 2
    │            Left Bottom                          │ (right bottom)
    │            (lf 0→1, Y negated)                  │ lf 1→0
    │                                                 │
    │                 BOTTOM (underside)              │
    └─────────────────────────────────────────────────┘
                      Segment 2: Right Bottom (lf 1→0)
                        right rail → centerline
```

The four segments are:

1. **Right deck** (lf 0 → 1): From deck centerline to right rail.
   `n` points with `y = lf * halfWidth`, `z = rockerZ + getDeckHeight(t, lf)`.

2. **Right bottom** (lf 1 → 0): From right rail back to bottom centerline.
   `n` points with `y = lf * halfWidth`, `z = rockerZ + getBottomHeight(t, lf)`.

3. **Left bottom** (lf 0 → 1, Y negated): Centerline to left rail.
   `n - 1` points (skip `lf = 0` to avoid duplicating the bottom centerline
   point from segment 2).

4. **Left deck** (lf 1 → 0, Y negated): Left rail back toward deck centerline.
   `n - 1` points (skip `lf = 0` to avoid duplicating the deck centerline
   point from segment 1, which closes the loop).

The total number of unique vertices per contour ring is:

```
M = n + n + (n - 1) + (n - 1) = 4n - 2
```

where `n = nLateral` (the number of lateral sample points per half-surface).

**Why not deduplicate at the rail?** At the rail edge (`lf = 1`), the deck
and bottom surfaces have different Z values -- the rail has finite thickness.
So segment 1's last point and segment 2's first point are distinct vertices
at the same `(x, y)` but different `z`. No deduplication is needed there.

**Why deduplicate at the centerline?** At `lf = 0`, both deck and bottom
are at the board's center stringer. The bottom centerline point at the end
of segment 2 is identical to the bottom centerline point at the start of
segment 3 (both at `y = 0`). Similarly, the deck centerline at the end of
segment 4 would duplicate segment 1's first point. Skipping these avoids
duplicate vertices that would cause degenerate triangles in the mesh.

### 3.3 Quad-Strip Triangulation

Between each pair of adjacent non-tip stations, the contour rings are
connected by a **quad strip**. Each pair of corresponding edges on the two
rings forms a quadrilateral, which is split into two triangles:

```
    Station i         Station i+1
    (contour A)       (contour B)

      v0 ─────────── v2
      │ \            │
      │   \  tri 1   │
      │     \        │
      │       \      │
      │  tri 0  \    │
      │           \  │
      v1 ─────────── v3
```

For each edge `j` in the contour (wrapping around with `j_next = (j+1) % M`):

```
tri 0 = [v0, v2, v1]    where v0 = offsetA + j
tri 1 = [v1, v2, v3]          v1 = offsetA + j_next
                               v2 = offsetB + j
                               v3 = offsetB + j_next
```

This produces `2 * M` triangles per station pair. For `N` non-degenerate
stations, the body has `2 * M * (N - 1)` triangles total.

### 3.4 End Capping

The nose and tail need special treatment to close the mesh into a
watertight solid.

**Fan cap (nose tip):** When the nose station is a degenerate tip point
and the next station is a full contour, fan triangulation connects the
single tip vertex to each edge of the contour ring. This produces `M`
triangles radiating from the tip, like the ribs of an umbrella.

**Planar cap (tail end):** When the tail station has a non-zero half-width
(which is the case for shortboard, longboard, and fish presets -- they all
have `tailTipHalfWidth > 0`), the mesh has an open terminal contour ring.
This is closed by computing the centroid of the ring, adding it as a new
vertex, and fan-triangulating from the centroid to each contour edge.
Using the centroid rather than a ring vertex handles mildly non-convex
cross-sections (e.g., boards with bottom concave) without producing
self-intersecting faces.

If the tail also collapses to a tip (like an extremely pointed pintail),
it gets the same fan treatment as the nose.

---

## 4. Resolution Presets

Three named presets control the mesh density:

| Preset | Longitudinal Stations | Lateral Points | Contour Size (M) | Approx. Vertices | Approx. Faces |
|--------|----------------------|----------------|-------------------|-------------------|---------------|
| `draft` | 100 | 25 | 98 | ~9,800 | ~19,600 |
| `standard` | 300 | 60 | 238 | ~71,000 | ~142,000 |
| `high` | 500 | 100 | 398 | ~199,000 | ~398,000 |

- **Draft** is for quick previews and interactive visualization. Mesh
  generation takes a fraction of a second.
- **Standard** is the default. Produces smooth surfaces suitable for STL
  export and 3D printing. Generation takes under a second.
- **High** is for validation and comparison against reference geometry.
  The extra resolution helps when computing surface deviation metrics.

All presets produce watertight meshes.

---

## 5. Reverse Engineering

The `ReverseEngineer` class provides a pipeline for extracting parametric
shape profiles from an existing STL mesh. The motivation is twofold:

1. **Calibration.** Given a reference board (e.g., a scanned reference
   thruster STL), extract the profiles and fit `SurfboardParameters` to
   match. This validates that the parametric model can reproduce known shapes.

2. **Understanding.** Given an unknown board STL (e.g., a 3D scan of a
   physical board), extract its key shape curves to understand its design
   properties.

### 5.1 Profile Extraction

The extraction pipeline:

1. **Load and transform.** The STL is loaded with trimesh. If `autoTransform`
   is enabled, the `CoordinateTransformer` auto-detects scale and axis mapping
   to bring the mesh into the project coordinate system (X = length, Y = width,
   Z = thickness, mm).

2. **Fin segmentation.** If `removeFins` is enabled, the `FinSegmenter`
   separates fins from the body using height-based clustering. Only the body
   mesh is analyzed further.

3. **Slicing.** The body mesh is sliced at `N` X positions using
   `trimesh.intersections.mesh_plane()`. Each slice produces a set of line
   segments in the YZ plane, which are collected into an ordered 2D contour
   by angular sorting from the centroid.

4. **Envelope extraction.** Each 2D contour is separated into upper (deck)
   and lower (bottom) envelopes by binning points by Y position and taking
   the max Z (deck) and min Z (bottom) in each bin. NaN gaps from sparse
   regions are filled by linear interpolation.

5. **Profile measurement.** From the envelopes, the code extracts:
   - **Half-width**: maximum |Y| extent of the contour
   - **Rocker height**: midpoint of deck and bottom Z at the centerline
   - **Thickness**: deck Z minus bottom Z at the rail (lf = 1)
   - **Deck/bottom profiles**: Z values at normalized lateral fractions

### 5.2 Measurement Design Decisions

Two non-obvious choices in the extraction pipeline deserve explanation:

**Rocker is measured as the midpoint of deck and bottom at the centerline.**
The parametric model's `RockerProfile.getRockerHeight(t)` defines the center
plane of the board -- equidistant from the deck and bottom surfaces. Using
the raw bottom Z at centerline would include half the thickness, the concave
offset, and half the crown offset, producing a systematic bias. The midpoint
measurement cancels these additive offsets:

```
midZ = (deckZ_center + bottomZ_center) / 2
     = (rockerZ + halfThickness + crown + rockerZ - halfThickness + concave) / 2
     = rockerZ + (crown + concave) / 2
```

Since crown and concave are typically small (8 mm and 2 mm respectively), the
residual `(crown + concave) / 2 = 5 mm` is much smaller than the bias from
using raw bottom Z directly, and it gets absorbed into the rocker offset
normalization (the rocker curve is shifted so its minimum is zero).

**Thickness is measured at the rail, not the centerline.** At the centerline,
the measured thickness includes deck crown (cosine dome above the base
half-thickness) and bottom concave (scoop below the base half-thickness).
This gives a centerline measurement like:

```
centerlineThickness = 2 * halfThickness + crown + concave
                    = 62 + 8 + 2 = 72 mm  (for a 62mm thick shortboard)
```

At the rail (`lf = 1`), both crown and concave evaluate to zero in the
parametric model, so the measurement gives the pure structural thickness:

```
railThickness = 2 * halfThickness = 62 mm
```

This matches the `maxThickness` parameter definition, which is the rail-to-rail
structural thickness, not the centerline peak.

### 5.3 Parameter Fitting

`fitParameters()` extracts `SurfboardParameters` from the profiles using
direct measurement rather than optimization:

| Parameter | Extraction Method |
|---|---|
| `length` | Board extent in X |
| `maxWidth` | 2 x max(halfWidths) |
| `maxThickness` | max(railThicknesses) |
| `widePointOffset` | Position of max halfWidth, converted to offset from center |
| `noseWidth` | 2 x halfWidth at 305mm from nose (12" station) |
| `tailWidth` | 2 x halfWidth at 305mm from tail |
| `tailTipHalfWidth` | Minimum halfWidth in the last 5% of the board |
| `noseRocker` | Quadratic extrapolation of rocker curve to t = 0 |
| `tailRocker` | Quadratic extrapolation of rocker curve to t = 1 |
| `deckCrown` | Deck center Z minus deck rail Z at thickest station |
| `bottomConcave` | Bottom rail Z minus bottom center Z at thickest station |

Endpoint extrapolation is needed because the slicing margins (3mm from each
end) prevent direct measurement at the actual nose and tail tips. A quadratic
polynomial is fit to the first/last 10 data points and evaluated at `t = 0`
or `t = 1`.

### 5.4 Known Limitations

- **Tail tip half-width** is the least accurate extracted parameter. The
  outline tapers steeply at the tail tip (power-law exponent ~2.5), so the
  half-width changes dramatically over the last few millimeters. The 3mm
  slicing margin means the true tip value is never directly observed. The
  minimum-value heuristic typically underestimates the actual tail width by
  15-25mm for boards with non-zero `tailTipHalfWidth`.

- **Swallow tail notches** are not modeled. The parametric `Outline` class
  produces a smooth taper without the notch. A fish board's swallow tail
  would need either a modified outline curve or a boolean subtraction.

- **Fins** must be removed before extraction. If `removeFins` is disabled
  on a finned board, the fin geometry inflates the apparent thickness and
  contaminates the bottom profiles.

---

## 6. STL Export

The `StlExporter` class wraps `SurfaceMeshGenerator` with convenience methods:

```python
from SurfPhysics.export import StlExporter

exporter = StlExporter()

# Export a single preset
exporter.exportPreset('shortboard', 'output/')

# Export all three presets at high resolution
exporter.exportAllPresets('output/', resolution='high')

# Export custom parameters
from SurfPhysics.geometry.parameters import SurfboardParameters
params = SurfboardParameters(length=1750, maxWidth=510, maxThickness=65, ...)
exporter.exportBoard(params, 'output/custom.stl')

# Export and validate against a reference STL
result = exporter.exportWithValidation(
    params, referencePath='reference.stl', outputDir='output/'
)
print(f'RMS deviation: {result["comparison"]["overallRmsMm"]:.2f} mm')
```

The `exportWithValidation()` method generates the parametric mesh, exports
it, then runs `MeshComparisonAnalyzer` against the reference STL to produce
deviation metrics in a single call.

---

## 7. API Quick Start

### Direct Mesh Generation

```python
from SurfPhysics.geometry.parameters import SurfboardParameters
from SurfPhysics.geometry.surfaceMeshGenerator import SurfaceMeshGenerator

# Create from a preset
params = SurfboardParameters.shortboard()
gen = SurfaceMeshGenerator(params, nLongitudinal=300, nLateral=60)
mesh = gen.generate()

print(f'Vertices:   {len(mesh.vertices):,}')
print(f'Faces:      {len(mesh.faces):,}')
print(f'Watertight: {mesh.is_watertight}')
print(f'Volume:     {gen.computeVolumeLiters():.2f} L')

# Export as STL
gen.exportStl('shortboard.stl')
```

### Using Resolution Presets

```python
from SurfPhysics.geometry.surfaceMeshGenerator import (
    SurfaceMeshGenerator, RESOLUTION_PRESETS,
)

# Factory method with named preset
gen = SurfaceMeshGenerator.fromPreset(params, preset='high')
mesh = gen.generate()
```

### Reverse Engineering

```python
from SurfPhysics.validation.reverseEngineer import ReverseEngineer

# Load a reference STL and extract shape profiles
re = ReverseEngineer(
    'reference_board.stl',
    expectedLengthMm=1828.0,
    removeFins=True,
    autoTransform=True,
)

profiles = re.extractProfiles(nStations=300, nLateralSamples=50)
print(f'Board length: {profiles.boardLengthMm:.1f} mm')
print(f'Max width:    {profiles.maxWidthMm:.1f} mm')
print(f'Max thickness:{profiles.maxThicknessMm:.1f} mm')

# Fit parametric parameters to the extracted shape
fittedParams = re.fitParameters(profiles)
fittedParams.printSummary()

# Compare fitted model against extracted profiles
comparison = re.compareToParametric(fittedParams, profiles)
print(f'Outline RMS:   {comparison["outlineRmsMm"]:.2f} mm')
print(f'Rocker RMS:    {comparison["rockerRmsMm"]:.2f} mm')
print(f'Thickness RMS: {comparison["thicknessRmsMm"]:.2f} mm')
```

### Round-Trip Validation

```python
from SurfPhysics.geometry.parameters import SurfboardParameters
from SurfPhysics.geometry.surfaceMeshGenerator import SurfaceMeshGenerator
from SurfPhysics.validation.reverseEngineer import ReverseEngineer

# Generate → export → reverse-engineer → compare
params = SurfboardParameters.shortboard()
gen = SurfaceMeshGenerator.fromPreset(params, 'standard')
gen.exportStl('shortboard.stl')

re = ReverseEngineer('shortboard.stl', expectedLengthMm=params.length,
                     removeFins=False, autoTransform=False)
profiles = re.extractProfiles()
fitted = re.fitParameters(profiles)
comparison = re.compareToParametric(fitted, profiles)
```

---

## 8. Validation Results

### Volume Accuracy

All three presets generate watertight meshes whose volume matches the
parametric integration (via `BoardGeometry.computeVolume()`) to within
0.01%:

| Preset | Mesh Volume (L) | Parametric Volume (L) | Error |
|--------|------------------|-----------------------|-------|
| Shortboard | 34.03 | 34.03 | 0.010% |
| Longboard | 77.72 | 77.73 | 0.010% |
| Fish | 33.54 | 33.54 | 0.009% |

### Round-Trip Parameter Recovery

Generating an STL from the shortboard preset, reverse-engineering it, and
fitting parameters produces the following recovery accuracy:

| Parameter | Fitted | Original | Difference |
|---|---|---|---|
| Length | 1828.0 mm | 1828.0 mm | 0.0 mm |
| Max Width | 494.5 mm | 495.0 mm | -0.5 mm |
| Max Thickness | 62.0 mm | 62.0 mm | 0.0 mm |
| Nose Width | 300.0 mm | 300.0 mm | 0.0 mm |
| Tail Width | 361.8 mm | 380.0 mm | -18.2 mm |
| Wide Point Offset | -21.3 mm | -25.0 mm | +3.7 mm |
| Nose Rocker | 117.9 mm | 120.0 mm | -2.1 mm |
| Tail Rocker | 37.6 mm | 40.0 mm | -2.4 mm |
| Deck Crown | 7.9 mm | 8.0 mm | -0.1 mm |
| Bottom Concave | 2.0 mm | 2.0 mm | 0.0 mm |

Most parameters are recovered within 1-3 mm. The tail width discrepancy
(18.2 mm) is the known tail-tip limitation described in Section 5.4 -- the
steep outline taper means the 305mm-from-tail measurement station captures a
different part of the taper curve than where the parametric model measures it.

### Curve-Level RMS Errors

| Curve | RMS Error |
|-------|-----------|
| Outline | 2.10 mm |
| Rocker | 0.58 mm |
| Thickness | 0.64 mm |
| Deck Profile | 2.82 mm |
| Bottom Profile | 2.96 mm |

Rocker and thickness are recovered with sub-1mm RMS, confirming that the
midpoint rocker measurement and rail thickness measurement are effective.
The outline RMS is slightly higher due to the tail tip approximation. Deck
and bottom profile RMS reflect the lateral interpolation between discrete
bins in the envelope extraction.

---

## 9. The Demo Script

The top-level `codeInterface.py` is configured as a demo script for the
mesh generation pipeline. Running it produces:

1. A parametric surfboard mesh from the selected preset
2. STL export to `SurfPhysics/output/`
3. A 4-panel interactive Plotly figure:
   - **3D Mesh3d** -- the full triangulated surface, rotatable
   - **Planform outline** -- half-width vs. length from nose
   - **Rocker profile** -- bottom curvature from side view
   - **Cross-sections** -- deck/bottom profiles at 6 stations along the board
4. (Optional) Reverse engineering validation with parameter comparison and
   4-panel curve overlay plots

Configuration is via variables at the top of the file:

```python
boardPreset = 'shortboard'        # 'shortboard', 'longboard', or 'fish'
meshResolution = 'standard'       # 'draft', 'standard', or 'high'
outputDir = 'SurfPhysics/output'
runReverseEngineering = True      # Set to False to skip validation
```

Run with:

```bash
python codeInterface.py
```

---

## 10. Coordinate System

All geometry in this pipeline uses the same coordinate system as the rest
of the project:

- **X** -- Along the board length. `X = 0` at the nose tip, increasing
  toward the tail.
- **Y** -- Across the board width. `Y = 0` at the centerline (stringer).
  Positive Y is to the right when viewed from above.
- **Z** -- Vertical. `Z = 0` at the rocker datum (the flat spot, which
  sits at 60% from the nose). Positive Z is upward (toward the deck).

All dimensions are in **millimeters**, matching `SurfboardParameters`.
Conversion to SI meters happens at the `BoardGeometry` boundary for
physics calculations.

---

## 11. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | >= 1.20 | Array operations, contour construction |
| `trimesh` | >= 3.0 | STL I/O, mesh repair, plane intersection, volume |
| `scipy` | >= 1.7 | Used by trimesh internally; also used in reverseEngineer for interpolation |
| `plotly` | >= 5.0 | Interactive visualization in codeInterface.py |

---

## Key Source Files

| File | Purpose |
|------|---------|
| `SurfPhysics/geometry/surfaceMeshGenerator.py` | Core mesh generator (cross-section stitching) |
| `SurfPhysics/validation/reverseEngineer.py` | STL slicing, profile extraction, parameter fitting |
| `SurfPhysics/export/stlExporter.py` | Convenience export wrapper |
| `SurfPhysics/geometry/parameters.py` | `SurfboardParameters` dataclass and presets |
| `SurfPhysics/geometry/outline.py` | Outline half-width parametric curve |
| `SurfPhysics/geometry/rocker.py` | Rocker height parametric curve |
| `SurfPhysics/geometry/crossSection.py` | Deck/bottom surface profiles |
| `SurfPhysics/geometry/board.py` | `BoardGeometry` facade (SI unit conversions) |
| `SurfPhysics/validation/finSegmentation.py` | Fin detection and removal |
| `SurfPhysics/optimization/coordinateTransformer.py` | Auto-detect scale and axis mapping |
| `codeInterface.py` | Demo script with Plotly visualization |
