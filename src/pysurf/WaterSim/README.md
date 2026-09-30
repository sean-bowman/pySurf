# WaterSim

**Weakly Compressible Smoothed Particle Hydrodynamics (WCSPH) simulation for free-surface water flows.**

A pure-Python, NumPy-vectorized SPH solver for 2D and 3D water sloshing and wave-tank scenarios. Primarily used for physics visualizations and as a computational benchmark: particularly the `sloshing_tank` Manim animation scene.

---

## Quick Start

### Run a simulation (CLI)

```bash
# Default: small 2D sloshing tank (fast, ~2s runtime)
python -m pysurf.WaterSim

# Standard quality preset
python -m pysurf.WaterSim --preset standard

# Load a JSON config
python -m pysurf.WaterSim --config src/pysurf/WaterSim/configs/sloshing_2d_default.json

# Wave tank scenario
python -m pysurf.WaterSim --scenario waveTank --preset small

# Skip frame export (for quick testing)
python -m pysurf.WaterSim --no-export
```

### Programmatic API

```python
from pysurf.WaterSim.runner import WaterSimRunner
from pysurf.WaterSim.scenarios.sloshingTank import SloshingTankConfig, createSloshingTank

# Configure a sloshing tank
config = SloshingTankConfig(
    tankWidth=0.5,
    tankHeight=0.3,
    fillRatio=0.6,
    initialTiltDeg=5.0,
    particleSpacing=0.005,
    endTime=3.0,
)

# Create and run
particles, boundary = createSloshingTank(config)
runner = WaterSimRunner(particles, boundary, config)
runner.run()
```

### Render the animation (Manim)

```bash
# Full quality
python src/pysurf/Surfboard/SurfAnimations/render.py --scene sloshing_tank

# Fast preview
python src/pysurf/Surfboard/SurfAnimations/render.py --scene sloshing_tank --quality low
```

---

## Architecture

```
WaterSim/
├── runner.py             CLI entry point and simulation orchestrator
├── constants.py          Physical constants (g, water density, etc.)
├── sph/                  Core SPH solver engine
│   ├── wcsphSolver.py    WCSPH equations: density, pressure, acceleration
│   ├── particles.py      ParticleSystem data structure (position, vel, density)
│   ├── kernels.py        Cubic spline and Wendland C2 kernel functions
│   ├── neighborSearch.py Spatial hash grid for O(N) neighbor lookup
│   ├── boundaryHandling.py Boundary particle repulsion and mirroring
│   ├── timeIntegration.py Symplectic Euler and Verlet integrators
│   ├── waveMaker.py      Piston wave-maker boundary condition
│   └── breakingDetection.py Wave breaking detection (velocity ratio)
├── scenarios/            Pre-configured simulation setups
│   ├── sloshingTank.py   2D/3D tilted tank sloshing
│   └── numericalWaveTank.py 2D wave tank with beach slope
├── export/               Frame data export for visualization
│   └── frameExporter.py  JSON/binary export for Manim and Three.js
├── configs/              JSON configuration files
└── documentation/        Technical writeups
```

### SPH Method

The solver implements **WCSPH** (Weakly Compressible SPH):

- **Density summation** via cubic spline kernel (smoothing length `h = 1.3 * particleSpacing`)
- **Pressure** from Tait equation of state: `P = B * ((ρ/ρ₀)^γ - 1)` with `γ = 7`
- **Speed of sound**: `c = 10 * sqrt(g * H)`: ensures density fluctuations < 1%
- **Momentum equation**: pressure gradient + viscosity (artificial + laminar)
- **Time integration**: symplectic Euler with CFL-adaptive time step
- **Boundary conditions**: fixed boundary particles with Lennard-Jones repulsion

---

## Scenarios

### Sloshing Tank

A rectangular tank filled to a set water depth and tilted at an initial angle. Good for validating energy conservation and the Manim animation.

```json
{
  "tank": {
    "width": 0.5,
    "height": 0.3,
    "fillRatio": 0.6,
    "initialTiltDeg": 5.0
  }
}
```

### Numerical Wave Tank

A 2D tank with a piston wave-maker on one end and a beach slope for energy absorption. Supports breaking wave detection.

```json
{
  "wave": {
    "height": 0.05,
    "period": 1.5,
    "rampCycles": 2,
    "useSecondOrder": false
  },
  "beachSlope": 0.15,
  "beachStartRatio": 0.6
}
```

---

## Config Reference

All config files share this top-level structure:

```json
{
  "simulation": {
    "type": "sloshingTank",
    "dimensions": 2,
    "endTime": 3.0,
    "outputInterval": 0.02
  },
  "sph": {
    "particleSpacing": 0.005,
    "smoothingLengthRatio": 1.3,
    "kernelType": "cubicSpline",
    "maxTimeStep": 0.0001
  },
  "fluid": {
    "density": 1000.0,
    "gravity": 9.81
  },
  "viewer": {
    "export": true,
    "outputDir": "src/pysurf/Surfboard/SurfViewer/data"
  }
}
```

Available presets: `sloshing_2d_default.json`, `wave_tank_2d_small.json`, `wave_tank_3d_default.json`.

---

## Performance Notes

- **~62 ms/step** with ~1200 fluid + ~400 boundary particles (vectorized NumPy)
- **~16 steps/s**: acceptable for short sims; consider Numba JIT for longer runs
- **Bottleneck**: neighbor search `build()` uses Python dict loop; all SPH solver loops are fully vectorized with `np.add.at` scatter-add
- **Density error**: ~35% at free-surface particles is expected (kernel truncation at free surface)
- **Energy conservation**: total energy fluctuates by ~1% over a simulation: acceptable for WCSPH

---

## Documentation

- **[WaterSim Overview](documentation/waterSimOverview.md)**: WCSPH theory, algorithm design, and comparison to grid-based methods

---

Sean Bowman
