# WaterSim -- SPH Water Simulation

Overview of the WaterSim Python module: a 2D Smoothed Particle Hydrodynamics
(SPH) simulation for free-surface water flows. Built from scratch in pure
Python + NumPy, designed as an educational implementation that still runs
at practical speeds through vectorized pair computations.

Sean Bowman [02/05/2026]

---

## 1. What It Does

WaterSim simulates water sloshing inside a rectangular tank. You define a
tank geometry, fill it partway with water, tilt the free surface a few
degrees, and let gravity do the rest. The water sloshes back and forth,
splashes off the walls, and gradually dissipates energy through artificial
viscosity -- exactly what you would see in a real tank experiment.

The simulation uses the **Weakly Compressible SPH (WCSPH)** method, a
meshfree Lagrangian particle approach where the fluid is represented by
discrete particles that carry mass, velocity, and density. There is no grid.
Particles move with the flow, interact with their neighbors through a
smoothing kernel, and obey discretized versions of the Navier-Stokes
equations.

The primary scenario right now is a sloshing tank (defined in
`WaterSim/scenarios/sloshingTank.py`), but the solver architecture is
general enough for dam breaks, wave tanks, or any 2D free-surface flow.

Output is a JSON file containing per-frame particle positions, velocity
magnitudes, densities, and an energy history -- ready for Manim animations
or the Three.js viewer.

---

## 2. The Physics

### Why Particles?

Traditional CFD uses Eulerian grids: you divide space into fixed cells and
track how fluid moves through them. That works well for problems with simple
boundaries and no free surface. But for water sloshing with splashing,
breaking waves, and a constantly deforming free surface, grids struggle.
You either need expensive interface tracking (Volume of Fluid, level sets)
or adaptive mesh refinement to follow the surface.

SPH sidesteps all of that. Because the particles *are* the fluid, the free
surface is just wherever the outermost particles happen to be. No interface
tracking, no mesh generation, no remeshing. The trade-off is that SPH needs
more work per degree of freedom (every particle interacts with ~30-50
neighbors), and maintaining accuracy near boundaries requires care. But for
free-surface flows at moderate resolution, the simplicity is hard to beat.

### Navier-Stokes in Particle Form

The governing equations are the standard incompressible Navier-Stokes
equations written in Lagrangian form (following the fluid parcels):

```
D(rho)/Dt = -rho * div(v)          (continuity)
Dv/Dt = -(1/rho) * grad(p) + nu * laplacian(v) + g    (momentum)
```

In SPH, field quantities are approximated by weighted sums over neighboring
particles. Any field `A` at position `r_i` is interpolated as:

```
A(r_i) = sum_j (m_j / rho_j) * A_j * W(|r_i - r_j|, h)
```

where `W` is the smoothing kernel and `h` is the smoothing length. Spatial
derivatives (gradient, divergence, Laplacian) are computed by differentiating
the kernel rather than the field -- which is the central trick of SPH.

### WCSPH: Weakly Compressible Approach

The hardest part of incompressible fluid simulation is enforcing the
divergence-free velocity constraint. Truly incompressible SPH methods (ISPH)
solve a pressure Poisson equation every time step, which is expensive.

WCSPH takes a shortcut: it allows the fluid to be *slightly* compressible
and computes pressure directly from density using an algebraic equation of
state. By choosing an artificially high speed of sound (much faster than
the actual flow velocities), density fluctuations are kept below ~1%,
so the fluid *behaves* incompressibly even though the solver never solves
a linear system.

The implementation lives in `WaterSim/sph/wcsphSolver.py` (`WcsphSolver`
class). Each time step follows this sequence:

1. Build neighbor list (spatial hash grid)
2. Compute density by SPH summation
3. Compute pressure via Tait equation of state
4. Compute accelerations (pressure gradient + artificial viscosity + gravity)
5. Compute adaptive time step (CFL condition)
6. Integrate (Symplectic Euler)
7. Enforce boundary conditions
8. Apply XSPH velocity correction
9. Periodically apply density re-initialization (Shepard filter)

### Kernel Functions -- How Particles "See" Each Other

The smoothing kernel `W(r, h)` is the weighting function that controls how
much influence a neighbor particle has. A valid SPH kernel must satisfy:

- **Normalization:** Integrates to 1 over the domain
- **Compact support:** Evaluates to zero beyond a cutoff radius (so each
  particle only interacts with nearby neighbors, not the entire domain)
- **Positivity:** Non-negative within the support
- **Delta limit:** Approaches a Dirac delta as h approaches 0

WaterSim uses the **cubic spline (M4) kernel** (implemented in
`WaterSim/sph/kernels.py`), the most widely used kernel in SPH literature.
It is a piecewise cubic polynomial with compact support at `q = 2` where
`q = r / h`:

```
W(q) = sigma * { 1 - (3/2)*q^2 + (3/4)*q^3    for 0 <= q < 1
               { (1/4)*(2 - q)^3               for 1 <= q < 2
               { 0                              for q >= 2
```

The 2D normalization constant is `sigma = 10 / (7 * pi * h^2)`. The support
radius is `2h`, which means a particle at the center of its support volume
"sees" everything within a circle of radius `2h` and ignores everything
beyond it. In practice, with a smoothing length ratio of 1.3 (i.e.,
`h = 1.3 * particleSpacing`), each fluid particle has roughly 20-40
neighbors in 2D.

The kernel gradient is what drives the force calculations. For the cubic
spline:

```
dW/dq = { -3*q + (9/4)*q^2            for 0 < q < 1
        { -(3/4)*(2 - q)^2            for 1 <= q < 2
```

The full gradient vector is `grad_W = (dW/dr) * (r_vec / |r_vec|)` --
pointing along the line connecting the two particles. Both scalar and
vectorized batch versions (`evaluateBatch`, `gradientBatch`) are provided
for performance.

### Artificial Viscosity (Monaghan)

Real water has very low viscosity, but SPH simulations at moderate resolution
need *artificial* viscosity for two reasons: stabilizing particle motion near
shocks or sharp pressure gradients, and dissipating unphysical high-frequency
noise that accumulates over many time steps.

WaterSim uses the Monaghan (1992) artificial viscosity formulation:

```
Pi_ij = { (-alpha * c * mu_ij) / rho_avg,   if v_ij . r_ij < 0
        { 0,                                  otherwise

mu_ij = h * (v_ij . r_ij) / (|r_ij|^2 + 0.01 * h^2)
```

The `v_ij . r_ij < 0` condition means viscosity is only applied when
particles approach each other (compression), not when they separate. This is
analogous to a Riemann solver in grid-based methods -- it acts like a shock
viscosity that smooths out discontinuities without over-damping smooth
regions.

The coefficient `alpha` (set to 0.1 in `WaterSim/constants.py`) controls
the dissipation strength. The `0.01 * h^2` term in the denominator prevents
a singularity when particles are very close. The speed of sound `c` enters
because the viscosity must be proportional to the wave speed to properly
dissipate shocks.

### XSPH Velocity Correction

Raw SPH particle trajectories can be noisy -- particles sometimes develop
erratic zig-zag paths even when the underlying velocity field is smooth.
XSPH (Monaghan 1989) corrects this by blending each particle's velocity
with a weighted average of its neighbors' velocities:

```
v_i += epsilon * sum_j (m_j / rho_avg) * (v_j - v_i) * W_ij
```

The epsilon parameter (0.5 in `constants.py`) controls how strongly
particles follow their neighbors. At `epsilon = 0`, particles move
independently; at `epsilon = 1`, particles move with the local average
velocity. The correction smooths trajectories without altering the
underlying physics (pressure forces, viscosity) because it only modifies
how particles *move*, not the forces they experience.

### Boundary Handling (Fixed Wall Particles)

Fluid-wall interaction in SPH is handled by lines of fixed "boundary
particles" that participate in density and pressure calculations but never
move (`WaterSim/sph/boundaryHandling.py`). When a fluid particle approaches
a wall, it "sees" the boundary particles as neighbors and computes a
repulsive pressure force that pushes it away from the wall.

The `BoundaryHandler` generates 3 layers of boundary particles along the
left, bottom, and right walls of the tank (the top is open). Multiple layers
are needed to ensure the density field is well-supported near walls -- a
single layer would cause density deficiency artifacts where fluid particles
near the wall underestimate their own density.

As a safety net, `enforceBoundary` also clamps particle positions that
somehow penetrate the wall and reflects their velocities with a 50% damping
factor. In a well-tuned simulation this rarely activates, but it prevents
runaway particles from escaping the domain.

### Key Equations Reference

| Quantity | Equation | Code Location |
|---|---|---|
| Density summation | `rho_i = sum_j m_j * W(\|r_i - r_j\|, h)` | `wcsphSolver._computeDensity()` |
| Pressure (Tait EOS) | `p = B * ((rho / rho_0)^gamma - 1)` | `wcsphSolver._computePressure()` |
| EOS constant | `B = rho_0 * c^2 / gamma` | `wcsphSolver.__init__()` |
| Speed of sound | `c = 10 * sqrt(g * H)` | `wcsphSolver.__init__()` |
| Pressure force | `-sum_j m_j * (p_i/rho_i^2 + p_j/rho_j^2) * grad_W_ij` | `wcsphSolver._computeAccelerations()` |
| Artificial viscosity | `Pi_ij = -alpha * c * mu_ij / rho_avg` | `wcsphSolver._computeAccelerations()` |
| XSPH correction | `v_i += eps * sum_j (m_j/rho_avg) * (v_j - v_i) * W_ij` | `wcsphSolver._applyXsphCorrection()` |
| Adaptive dt (CFL) | `dt = CFL * h / (c + max\|v\|)` | `wcsphSolver._computeAdaptiveTimeStep()` |
| Adaptive dt (force) | `dt = CFL * sqrt(h / max\|a\|)` | `wcsphSolver._computeAdaptiveTimeStep()` |
| Symplectic Euler | `v(t+dt) = v(t) + a(t)*dt; x(t+dt) = x(t) + v(t+dt)*dt` | `timeIntegration.SymplecticEuler` |

The Tait EOS exponent `gamma = 7` is the standard value for water in WCSPH.
Negative pressures are clamped to zero to prevent tensile instability (which
would cause particles to clump rather than repel). The CFL number is 0.25,
and the time step is the minimum of the CFL-based and force-based criteria,
capped at `maxTimeStep = 1e-4 s`.

---

## 3. Module Architecture

```
WaterSim/
  __init__.py               Package entry -- exports WaterSimRunner,
                              SloshingTankConfig, FrameExporter
  runner.py                  WaterSimRunner class -- CLI and pipeline
  constants.py               Physical & numerical constants (SI units)
  sph/
    protocols.py             SimulationConfig, SimulationState dataclasses,
                              SphSolver protocol
    kernels.py               CubicSplineKernel (scalar + vectorized batch)
    particles.py             ParticleSystem dataclass (positions, velocities,
                              densities, masses, isFluid mask)
    neighborSearch.py        SpatialHashGrid -- O(N) cell-linked list
    wcsphSolver.py           WcsphSolver -- full WCSPH algorithm
    boundaryHandling.py      BoundaryHandler -- wall particle generation
                              and clamping
    timeIntegration.py       SymplecticEuler integrator
  scenarios/
    sloshingTank.py          SloshingTankConfig dataclass + createSloshingTank()
                              factory function
  export/
    frameExporter.py         FrameExporter -- collects frames, writes JSON
```

**Key design decisions:**

- **Protocols over inheritance.** `SphKernel`, `NeighborSearch`,
  `TimeIntegrator`, and `SphSolver` are all defined as Python `Protocol`
  classes in their respective files. This makes it easy to swap components
  (e.g., a Wendland C2 kernel or a Verlet integrator) without touching the
  solver.

- **Fluid + boundary in one array.** The `ParticleSystem` stores both fluid
  and boundary particles in the same contiguous NumPy arrays, distinguished
  by the `isFluid` boolean mask. This avoids the bookkeeping complexity of
  separate arrays and lets the neighbor search and density computation treat
  all particles uniformly.

- **Vectorized pair operations.** The inner loops (density summation,
  pressure force, viscosity, XSPH) operate on arrays of neighbor pair
  indices `(iIdx, jIdx)` returned by the spatial hash grid. Kernel
  evaluations, distance computations, and force accumulations are all done
  with NumPy broadcasting and `np.add.at` scatter operations -- no Python
  `for` loops over pairs.

---

## 4. API Quick Start

### Programmatic Usage

```python
from WaterSim import WaterSimRunner, SloshingTankConfig

# Create runner and scenario
runner = WaterSimRunner()
tankConfig = SloshingTankConfig.small2D()

# Run simulation (prints progress, exports frames)
results = runner.runSloshing(tankConfig)

# Access results
print(f'Wall-clock time: {results["wallClockSeconds"]:.1f} s')
print(f'Frames exported: {results["nFrames"]}')
print(f'Export path: {results["exportPath"]}')
```

### CLI Usage

```bash
# Default small 2D sloshing (~800 particles)
python -m WaterSim

# Standard quality 2D sloshing (~2000 particles)
python -m WaterSim --preset standard

# Custom JSON config
python -m WaterSim --config configs/sloshing_2d_default.json

# Run without exporting frame data
python -m WaterSim --no-export

# Custom output directory
python -m WaterSim --output-dir path/to/output
```

### Custom Tank Configuration

```python
from WaterSim import WaterSimRunner, SloshingTankConfig

tankConfig = SloshingTankConfig(
    tankWidth=0.4,
    tankHeight=0.25,
    fillRatio=0.7,            # 70% fill
    initialTiltDeg=10.0,      # Aggressive initial tilt
    particleSpacing=0.003,    # Finer resolution
    smoothingLengthRatio=1.3,
    endTime=5.0,
    outputInterval=0.01,      # 100 fps output
)

runner = WaterSimRunner()
results = runner.runSloshing(tankConfig)
```

### Presets

| Preset | Method | Particle Spacing | Fluid Particles | End Time |
|---|---|---|---|---|
| Small 2D | `SloshingTankConfig.small2D()` | 0.005 m | ~800 | 2.0 s |
| Standard 2D | `SloshingTankConfig.standard2D()` | 0.004 m | ~2000 | 3.0 s |

---

## 5. Configuration

### JSON Config Format

Simulations can be configured via JSON files (loaded with `--config` or
`WaterSimRunner.runFromConfig()`). The file at
`configs/sloshing_2d_default.json` is the reference example:

```json
{
    "simulation": {
        "type": "sloshingTank",
        "dimensions": 2,
        "endTime": 3.0,
        "outputInterval": 0.02
    },
    "tank": {
        "width": 0.5,
        "height": 0.3,
        "fillRatio": 0.6,
        "initialTiltDeg": 5.0
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
        "outputDir": "SurfViewer/data"
    }
}
```

### Config Field Reference

| Section | Field | Type | Default | Description |
|---|---|---|---|---|
| `simulation` | `dimensions` | int | 2 | Spatial dimensions (2D only for now) |
| `simulation` | `endTime` | float | 3.0 | Simulation end time [s] |
| `simulation` | `outputInterval` | float | 0.02 | Time between exported frames [s] |
| `tank` | `width` | float | 0.5 | Tank width [m] |
| `tank` | `height` | float | 0.3 | Tank height [m] |
| `tank` | `fillRatio` | float | 0.6 | Water fill fraction (0-1) |
| `tank` | `initialTiltDeg` | float | 5.0 | Initial free-surface tilt [deg] |
| `sph` | `particleSpacing` | float | 0.005 | Inter-particle spacing [m] |
| `sph` | `smoothingLengthRatio` | float | 1.3 | h / particleSpacing ratio |
| `sph` | `kernelType` | str | `'cubicSpline'` | Smoothing kernel type |
| `sph` | `maxTimeStep` | float | 1e-4 | Maximum allowed dt [s] |
| `fluid` | `density` | float | 1000.0 | Reference fluid density [kg/m^3] |
| `fluid` | `gravity` | float | 9.81 | Gravitational acceleration [m/s^2] |

### Numerical Constants (`WaterSim/constants.py`)

These are hard-coded tuning parameters, not user-configurable via JSON.
Changing them requires editing the constants file.

| Constant | Value | Purpose |
|---|---|---|
| `speedOfSoundFactor` | 10.0 | `c = factor * sqrt(g * H)` -- keeps density error < 1% |
| `gamma` | 7.0 | Tait EOS exponent (standard for water) |
| `cflNumber` | 0.25 | CFL condition multiplier |
| `alphaViscosity` | 0.1 | Monaghan artificial viscosity coefficient |
| `xsphEpsilon` | 0.5 | XSPH velocity smoothing factor |
| `densityFilterInterval` | 30 | Shepard filter applied every N steps |
| `defaultBoundaryLayers` | 3 | Number of wall particle layers |

---

## 6. Output Format

The `FrameExporter` (`WaterSim/export/frameExporter.py`) writes a single
JSON file per simulation run. The file contains all frames, energy tracking,
and metadata.

### JSON Structure

```json
{
    "meta": {
        "type": "waterSim",
        "dimensions": 2,
        "nFrames": 101,
        "nFluidParticles": 800,
        "particleSpacing": 0.005,
        "created": "2026-02-05T14:30:00"
    },
    "config": {
        "domainMin": [0.0, 0.0],
        "domainMax": [0.3, 0.2],
        "particleSpacing": 0.005,
        "referenceDensity": 1000.0,
        "endTime": 2.0
    },
    "frames": [
        {
            "time": 0.0,
            "positions": [[0.0025, 0.0025], [0.0075, 0.0025], ...],
            "velocityMagnitudes": [0.0, 0.0, ...],
            "densities": [1000.0, 1000.0, ...]
        },
        ...
    ],
    "energy": {
        "times": [0.0, 0.02, 0.04, ...],
        "kinetic": [0.0, 0.000123, ...],
        "potential": [0.123456, 0.123400, ...],
        "total": [0.123456, 0.123523, ...]
    }
}
```

**Key details:**

- Only **fluid** particle data is exported (boundary particles are static
  and can be reconstructed from the config).
- Positions are rounded to 6 decimal places, densities to 2.
- The JSON is written with `indent=None` and minimal separators for compact
  file size. A 2-second simulation at 50 fps with 800 particles produces
  roughly a 5-10 MB file.
- The `energy` section provides a full time history of kinetic, potential,
  and total mechanical energy -- useful for verifying energy conservation
  and diagnosing numerical instability.
- Files are named `waterSim_{scenario}_{timestamp}.json` and written to
  the configured output directory (default: `WaterSim/output/`).

---

## 7. Performance Notes

### Particle Count vs. Wall-Clock Time

SPH is fundamentally O(N) per time step thanks to the spatial hash grid
neighbor search, but the constant factor is significant. Each time step
involves rebuilding the hash grid, finding all pairs within the support
radius, and performing vectorized kernel evaluations, distance
computations, and scatter-add operations for density, pressure forces,
viscosity, and XSPH.

The bottleneck is the spatial hash grid construction and pair finding in
`WaterSim/sph/neighborSearch.py`. The `build()` method uses a Python
`for` loop to bin particles into cells (this is the one unavoidable Python
loop in the pipeline). Pair queries are vectorized per cell-pair group
using NumPy broadcasting.

**Approximate wall-clock times (single-threaded, modern hardware):**

| Preset | Fluid Particles | Boundary Particles | Sim Duration | Typical Wall-Clock |
|---|---|---|---|---|
| Small 2D | ~800 | ~600 | 2.0 s | ~10-30 s |
| Standard 2D | ~2000 | ~800 | 3.0 s | ~2-5 min |
| Custom fine (0.002 m) | ~8000 | ~1200 | 3.0 s | ~30-60 min |

These are rough estimates. Actual times depend heavily on hardware, Python
version, and NumPy build (MKL vs. OpenBLAS). The adaptive time step means
more energetic simulations (higher tilt angle, larger tank) take more steps
and run longer.

### Vectorization Strategy

The solver avoids Python `for` loops over particles or pairs wherever
possible. The general pattern is:

1. The spatial hash grid returns arrays of pair indices `(iIdx, jIdx)`
2. Displacement vectors, distances, and kernel values are computed for all
   pairs simultaneously using NumPy slicing and broadcasting
3. Per-pair force contributions are computed as arrays
4. Results are scattered back to particle arrays using `np.add.at()`

The `CubicSplineKernel` provides both scalar methods (`evaluate`,
`gradient`) for single-pair lookups and batch methods (`evaluateBatch`,
`gradientBatch`) that process entire arrays of distances at once. The
solver exclusively uses the batch methods.

### Known Limitations

- **2D only.** The solver and kernel support 3D in principle (the
  `CubicSplineKernel` has a 3D normalization path, the spatial hash grid
  computes 3D half-stencils), but `BoundaryHandler._generate3D()` raises
  `NotImplementedError`. Extending to 3D is straightforward but will
  increase particle counts by ~50-100x.
- **Pure Python.** There is no C extension, Cython, or GPU acceleration.
  For serious production work you would want to move the neighbor search
  and pair computations to compiled code. But for educational purposes and
  moderate particle counts, NumPy vectorization keeps things practical.
- **No free-slip walls.** The boundary particle approach with velocity
  reflection provides approximate no-slip/partial-slip behavior. True
  free-slip or no-slip boundary conditions would require ghost particles
  with mirrored velocities.
- **Single-threaded.** NumPy operations benefit from multi-threaded BLAS
  under the hood, but the simulation loop itself is serial. For large
  particle counts, the `np.add.at` scatter operations become the
  bottleneck since they cannot be parallelized within NumPy.

---

## References

- Monaghan, J.J. (1992) -- *Smoothed Particle Hydrodynamics.* Annual Review
  of Astronomy and Astrophysics.
- Monaghan, J.J. (1994) -- *Simulating free surface flows with SPH.* Journal
  of Computational Physics.
- Monaghan, J.J. (2005) -- *Smoothed Particle Hydrodynamics.* Reports on
  Progress in Physics.
- Morris, J.P., Fox, P.J., & Zhu, Y. (1997) -- *Modeling low Reynolds
  number incompressible flows using SPH.* Journal of Computational Physics.
- Monaghan, J.J. & Kos, A. (1999) -- *Solitary waves on a Cretan beach.*
  Journal of Waterway, Port, Coastal and Ocean Engineering.
- Ihmsen, M. et al. (2011) -- *Parallel Neighbor-Search for SPH.*
- Batchelor, G.K. (1967) -- *An Introduction to Fluid Dynamics.* Cambridge
  University Press.
- Hairer, E., Lubich, C., & Wanner, G. (2003) -- *Geometric Numerical
  Integration.* Springer.
