# API Reference

Quick reference for all importable classes and functions in the
Computational Engineering Toolkit. Each module exposes its public
API through package-level imports.

Sean Bowman [02/05/2026]

---

## Import Patterns

```python
# Package-level imports (recommended)
from SurfPhysics import PhysicsAnalyzer, SurfboardParameters, WaveConditions, Visualizer, ViewerExporter
from WaterSim import WaterSimRunner, SloshingTankConfig, FrameExporter
from SurfAnimations import renderScene, renderAll, SCENES

# Subpackage imports (advanced use)
from SurfPhysics.geometry import SurfboardParameters, BoardGeometry
from SurfPhysics.waves import WaveConditions, LinearWaveTheory
from SurfPhysics.hydrodynamics import BuoyancyModel, ForceBalance
from SurfPhysics.visualization import Visualizer, createAnalysisDashboard
from SurfPhysics.export import ViewerExporter
from WaterSim.scenarios import SloshingTankConfig, createSloshingTank
from WaterSim.export import FrameExporter
from WaterSim.sph import SimulationConfig, SimulationState
```

---

## SurfPhysics

### PhysicsAnalyzer

Full analysis pipeline from JSON config to results dictionary.

```python
from SurfPhysics import PhysicsAnalyzer

analyzer = PhysicsAnalyzer()
results = analyzer.runAnalysis('configs/shortboard_default.json')

analyzer.results       # dict -- analysis results (same as return value)
analyzer.params        # SurfboardParameters -- board configuration
analyzer.waveConditions  # WaveConditions -- wave state
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `runAnalysis` | `(configPath: str) -> dict` | Results dictionary with board geometry, wave physics, force balance, and performance curves |

### SurfboardParameters

Parametric surfboard dimensions with factory presets.

```python
from SurfPhysics import SurfboardParameters

params = SurfboardParameters.shortboard()
params = SurfboardParameters.longboard()
params = SurfboardParameters.fish()
params = SurfboardParameters.fromJson('path/to/custom.json')
```

| Factory Method | Returns |
|----------------|---------|
| `shortboard()` | 6'0" shortboard with thruster fins |
| `longboard()` | 9'0" longboard with single fin |
| `fish()` | 5'6" fish with twin fins |
| `fromJson(filePath: str)` | Custom board from JSON file |

Key properties: `lengthMm`, `widthMm`, `thicknessMm`, `noseRockerMm`, `tailRockerMm`, `widePointFraction`, `finConfiguration`.

### WaveConditions

Wave state definition (height, period, depth).

```python
from SurfPhysics import WaveConditions

waves = WaveConditions(height=1.5, period=10.0, depth=2.5)
```

| Field | Type | Description |
|-------|------|-------------|
| `height` | `float` | Wave height [m] |
| `period` | `float` | Wave period [s] |
| `depth` | `float` | Water depth [m] |

### Visualizer

Interactive Plotly dashboard creation.

```python
from SurfPhysics import Visualizer

visualizer = Visualizer()
fig = visualizer.createDashboard(results)
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `createDashboard` | `(results: dict) -> go.Figure` | Plotly figure with multi-panel dashboard |

### ViewerExporter

Export analysis results as JSON for the Three.js viewer.

```python
from SurfPhysics import ViewerExporter

exporter = ViewerExporter()
path = exporter.exportForViewer(results, outputDir='SurfViewer/data')
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `exportForViewer` | `(results, stlDir, outputDir, embedStl, nLongitudinal, nLateral) -> str` | Path to exported JSON file |

---

### Advanced: SurfPhysics Submodules

#### BoardGeometry (`SurfPhysics.geometry`)

Parametric board shape queries in SI units.

```python
from SurfPhysics.geometry import BoardGeometry, SurfboardParameters

board = BoardGeometry(SurfboardParameters.shortboard())
```

| Method | Returns |
|--------|---------|
| `getHalfWidthM(t: float)` | Half-width at normalized position t [m] |
| `getRockerHeightM(t: float)` | Rocker height at position t [m] |
| `getThicknessM(t: float)` | Thickness at position t [m] |
| `getDeckHeightM(t, lateralFraction)` | Deck height [m] |
| `getBottomHeightM(t, lateralFraction)` | Bottom height [m] |
| `computeVolume()` | Total board volume [m^3] |
| `computePlanformArea()` | Planform area [m^2] |
| `computeWettedSurfaceArea(draft)` | Wetted area at given draft [m^2] |
| `computeSubmergedVolume(draft)` | Submerged volume at given draft [m^3] |
| `getLengthM()` | Board length [m] |
| `getMaxWidthM()` | Maximum width [m] |

#### LinearWaveTheory (`SurfPhysics.waves`)

Linear (Airy) wave physics calculations.

```python
from SurfPhysics.waves import LinearWaveTheory, WaveConditions

lwt = LinearWaveTheory()
waves = WaveConditions(height=1.5, period=10.0, depth=2.5)
```

| Method | Returns |
|--------|---------|
| `waveLength(waveConditions)` | Wavelength [m] (iterative dispersion solve) |
| `waveSpeed(waveConditions)` | Phase speed [m/s] |
| `groupSpeed(waveConditions)` | Group speed [m/s] |
| `surfaceElevation(x, t, waveConditions)` | Free surface elevation [m] |
| `velocityField(x, z, t, waveConditions)` | (u, w) velocity components [m/s] |
| `energyDensity(waveConditions)` | Energy density [J/m^2] |
| `energyFlux(waveConditions)` | Energy flux [W/m] |
| `isBroken(waveConditions)` | True if wave has broken |
| `depthClassification(waveConditions)` | 'deep', 'intermediate', or 'shallow' |
| `surfableWaveSpeed(waveConditions)` | Speed a surfer must match [m/s] |

#### BuoyancyModel (`SurfPhysics.hydrodynamics`)

Archimedes-based buoyancy calculations.

```python
from SurfPhysics.hydrodynamics import BuoyancyModel
from SurfPhysics.geometry import BoardGeometry, SurfboardParameters

buoyancy = BuoyancyModel(BoardGeometry(SurfboardParameters.shortboard()))
```

| Method | Returns |
|--------|---------|
| `estimateBoardMass()` | Board mass estimate [kg] |
| `buoyancyForce(submergedVolume)` | Buoyancy force [N] |
| `findEquilibriumDraft(totalMassKg)` | Equilibrium draft depth [m] |
| `paddleDraftWithRider(riderMassKg)` | Draft with rider sitting [m] |

#### ForceBalance (`SurfPhysics.hydrodynamics`)

Full force analysis for a surfboard at speed.

```python
from SurfPhysics.hydrodynamics import ForceBalance

fb = ForceBalance(boardGeometry, waveConditions, riderMassKg=75.0)
```

| Method | Returns |
|--------|---------|
| `findEquilibrium(speed)` | `PlaningState` at equilibrium trim angle |
| `performanceCurves()` | Dict of arrays: speed, drag, lift, L/D, trim, draft |
| `totalDrag(speed, trimAngleDeg)` | Total drag force [N] |
| `totalLift(speed, trimAngleDeg)` | Total lift force [N] |
| `dragBreakdown(speed, trimAngleDeg)` | Dict of individual drag components [N] |

---

## WaterSim

### WaterSimRunner

SPH simulation pipeline runner.

```python
from WaterSim import WaterSimRunner, SloshingTankConfig

runner = WaterSimRunner()

# From preset
results = runner.runSloshing(SloshingTankConfig.small2D())

# From JSON config
results = runner.runFromConfig('configs/sloshing_2d_default.json')
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `runSloshing` | `(tankConfig, doExport=True, exportDir='WaterSim/output') -> dict` | Results with finalState, wallClockSeconds, nFrames, exportPath |
| `runFromConfig` | `(configPath: str, exportDir: str) -> dict` | Same as runSloshing |

### SloshingTankConfig

Sloshing tank scenario configuration with presets.

```python
from WaterSim import SloshingTankConfig

config = SloshingTankConfig.small2D()     # ~800 particles, fast
config = SloshingTankConfig.standard2D()  # ~3000 particles, standard
```

| Field | Type | Description |
|-------|------|-------------|
| `tankWidth` | `float` | Tank width [m] |
| `tankHeight` | `float` | Tank height [m] |
| `fillRatio` | `float` | Water fill fraction (0-1) |
| `initialTiltDeg` | `float` | Initial free surface tilt [deg] |
| `particleSpacing` | `float` | Inter-particle spacing [m] |
| `smoothingLengthRatio` | `float` | h / particleSpacing |
| `endTime` | `float` | Simulation end time [s] |
| `outputInterval` | `float` | Frame output interval [s] |

### FrameExporter

Export simulation frames as JSON.

```python
from WaterSim import FrameExporter

exporter = FrameExporter()
exporter.addFrame(state, particles)
path = exporter.export(config, outputDir='WaterSim/output', scenarioName='sloshing')
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `addFrame` | `(state: SimulationState, particles: ParticleSystem) -> None` | -- |
| `export` | `(config, outputDir, scenarioName) -> str` | Path to exported JSON |

---

## SurfAnimations

### renderScene

Render a single Manim animation scene via subprocess.

```python
from SurfAnimations import renderScene

success = renderScene('wave_intro', quality='high')
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `sceneName` | `str` | Key from SCENES dict |
| `quality` | `str` | `'low'` (480p), `'medium'` (720p), `'high'` (1080p), `'fourk'` (4K) |
| **Returns** | `bool` | True if rendering succeeded |

### renderAll

Render all registered scenes.

```python
from SurfAnimations import renderAll

results = renderAll(quality='high')  # {'wave_intro': True, 'board_side': True, ...}
```

### SCENES

Scene registry dictionary.

```python
from SurfAnimations import SCENES

for name, info in SCENES.items():
    print(f'{name}: {info["description"]}')
```

| Scene Key | Class | Description |
|-----------|-------|-------------|
| `wave_intro` | `WavePhysicsIntro` | Linear wave theory introduction |
| `board_side` | `BoardOnWaveSideView` | Board riding wave (2D side view) |
| `board_perspective` | `BoardOnWavePerspective` | Board riding wave (2.5D perspective) |
| `performance` | `PerformanceComparison` | Multi-board performance comparison |
| `sloshing_tank` | `SloshingTankAnimation` | SPH sloshing tank simulation |

---

## Configuration (JSON)

All modules are driven by JSON config files in `configs/`. Full schema:

```json
{
    "board":      { "type", "finConfiguration", "foamType", "customParametersFile" },
    "waves":      { "height", "period", "depth" },
    "rider":      { "mass" },
    "analysis":   { "run", "showDashboard", "speedRange", "nSpeedPoints" },
    "viewer":     { "export", "openInBrowser", "outputDir" },
    "waterSim":   { "run", "preset", "configPath", "export", "outputDir" },
    "animations": { "render", "scenes", "quality" }
}
```

See `configs/shortboard_default.json` for a complete example.
