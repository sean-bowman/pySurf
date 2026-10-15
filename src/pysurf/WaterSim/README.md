# WaterSim

**Ready-to-run SPH water simulations: a sloshing tank and a numerical wave tank, with a command-line runner and frame export.**

WaterSim assembles named problems from pySurf's water modules, which live one level up in `src/pysurf/` and work on their own. The breaking solitary wave on a beach is its own facade, [`pysurf.BreakingWave`](../BreakingWave.py), described in the [main README](../../../README.md#breaking-wave). WaterSim imports nothing from the board model.

---

## Quick Start

### Command line

```bash
# Default: small 2D sloshing tank, 2.0 s simulated (20,000 steps at about 7 ms each)
python -m pysurf.WaterSim

# Standard quality preset
python -m pysurf.WaterSim --preset standard

# Load a JSON config
python -m pysurf.WaterSim --config src/pysurf/WaterSim/configs/sloshing_2d_default.json

# Wave tank scenario
python -m pysurf.WaterSim --scenario waveTank --preset small

# Skip frame export
python -m pysurf.WaterSim --no-export
```

### Python

```python
from pysurf.WaterSim import WaterSimRunner, SloshingTankConfig

config = SloshingTankConfig(tankWidth = 0.5, tankHeight = 0.3, fillRatio = 0.6, initialTiltDeg = 5.0,
                            particleSpacing = 0.005, endTime = 3.0)
results = WaterSimRunner().runSloshing(config, doExport = True)
```

To step the solver directly:

```python
from pysurf.sphSolver import WcsphSolver
from pysurf.WaterSim.scenarios.sloshingTank import SloshingTankConfig, createSloshingTank

simConfig, particles, boundary = createSloshingTank(SloshingTankConfig.small2D())
solver = WcsphSolver(simConfig, boundary = boundary)
solver.initialize(particles)
state = solver.step()
```

Wave theory is a module of its own:

```python
from pysurf.waveTheory import LinearWaveTheory, WaveConditions

waves = WaveConditions(height = 1.5, period = 10.0, depth = 2.5)
theory = LinearWaveTheory()            # seawater density by default
theory.waveLength(waves)               # 48.69 m
theory.groupSpeed(waves)               # m/s
```

---

## Architecture

```txt
src/pysurf/
├── constants.py          Gravity and water properties
├── waveTheory.py         Linear and solitary waves, Biesel transfer function, breaking criteria
├── bathymetry.py         Bed profile: elevation, depth, slope, normal, shoreline
├── sphKernels.py         Cubic spline and Wendland C2 kernels
├── sphNeighbors.py       k-d tree pair search
├── sphParticles.py       Particle state, energy diagnostics, lattice seeding
├── sphBoundaries.py      Container and bed boundaries of fixed particles
├── sphSolver.py          WCSPH equations and the time-stepping loop
├── waveMakers.py         Piston wave maker; solitary wave initial condition
├── waveBreaking.py       Free surface, front face lean, breaking tracker, runup
├── BreakingWave.py       Facade: a solitary wave breaking on a plane beach
│
└── WaterSim/
    ├── runner.py         CLI entry point and simulation orchestrator
    ├── scenarios/
    │   ├── sloshingTank.py       Tilted-surface rectangular tank, 2D and 3D
    │   └── numericalWaveTank.py  Piston wave maker and beach, 2D and 3D
    ├── export/
    │   └── frameExporter.py      JSON and binary frames for Manim and Three.js
    ├── configs/          JSON configurations
    └── documentation/    Technical writeups and references
```

### SPH method

The solver is weakly compressible SPH:

- **Density** from the continuity equation, started hydrostatic beneath the configured free surface through the inverse Tait equation, and reset to its Shepard-filtered value every 30 steps (Dalrymple and Rogers 2006). Summation density is not used: a free-surface particle's kernel support is half empty, so summation reads it low and the surface layers collapse.
- **Pressure** from the Tait equation of state, `p = B ((rho / rho0)^7 - 1)` with `B = rho0 c0^2 / 7`; negative pressure is clamped to zero.
- **Speed of sound** `c0 = 10 sqrt(g H)`, which holds density variation near 1 percent.
- **Momentum**: the symmetric pressure gradient plus Monaghan's artificial viscosity; pair forces are equal and opposite, so momentum is conserved to rounding.
- **Time integration**: symplectic Euler with a CFL-adaptive step. The density rate uses the kicked velocity; computing it from the old velocity is forward Euler on the acoustic oscillator, which grows without bound at any step size.
- **XSPH** moves positions with a smoothed velocity and leaves particle velocities untouched.
- **Boundaries**: layers of fixed particles that take part in density and pressure (Crespo et al. 2007), with a projection that returns any particle that crosses a wall.
- **Neighbors**: `scipy.spatial.cKDTree.query_pairs`, each unique pair once.

---

## Scenarios

### Sloshing tank

A rectangular tank filled to a set depth with its surface tilted across it. The first sloshing mode has a closed-form linear period, $\omega_1^2 = g k_1 \tanh(k_1 d)$ with $k_1 = \pi / W$, which makes this the scenario the solver is validated on (`tests/testValidationSloshing.py`). The particle spacing must divide the tank width and the fill height: a lattice that does not fit stops short of one wall by more than the one-spacing gap the boundary expects, and the water slumps into it.

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

### Numerical wave tank

A tank with a piston wave maker at one end and a beach slope at the other. The stroke comes from Biesel's transfer function for the requested wave height, and the breaker type is predicted from the offshore Iribarren number.

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
    "outputDir": "src/pysurf/WaterSim/output"
  }
}
```

Available configs: `sloshing_2d_default.json`, `wave_tank_2d_small.json`, `wave_tank_3d_default.json`.

---

## Performance

- About 4.2 µs per particle per step in 2D: 6.8 ms per step for the small sloshing tank (1200 fluid and 438 boundary particles).
- The small presets cap the step at 1e-4 s; the CFL limit alone would allow about 1.2e-4 s.
- Free-surface particles read up to 35 percent below the reference density by construction, since their kernel support is partly empty; the interior density error is what measures compressibility.

---

## Documentation

- **[WaterSim Overview](documentation/waterSimOverview.md)**: WCSPH theory, algorithm design, and comparison to grid-based methods
- **[SPH Wave Breaking References](documentation/papers/papers/sph-wave-breaking-references.md)**: annotated bibliography

---

Sean Bowman
