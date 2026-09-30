# SurfPhysics -- Module Overview & Theory Reference

A conversational walkthrough of the SurfPhysics Python module: what it computes,
why the physics works the way it does, and how the code is organized.

Sean Bowman [02/05/2026]

---

## 1. What It Does

SurfPhysics is the analysis backend for the surfboard project. You hand it a
JSON configuration file describing a board shape and wave conditions, and it
runs the full hydrodynamic pipeline: construct parametric geometry, estimate
board mass, solve the wave field, compute buoyancy equilibrium, evaluate
planing forces across a speed range, and return everything as a structured
results dictionary.

The top-level entry point is `PhysicsAnalyzer` (`SurfPhysics/analyzer.py`).
A single call to `runAnalysis()` executes the entire pipeline and prints a
formatted console summary. From there you can feed the results into the
interactive Plotly dashboard (`Visualizer`) or export them as JSON for the
Three.js viewer (`ViewerExporter`).

The pipeline, in order:

1. **Board geometry** -- build a parametric surfboard shape from dimensions
   (outline, rocker, cross-section) and compute volume, planform area, and
   wetted surface area via numerical integration.
2. **Buoyancy** -- estimate board mass from material properties, then find the
   static equilibrium draft where Archimedes buoyancy equals weight.
3. **Wave physics** -- solve the dispersion relation for the configured wave
   conditions, compute wavelength, phase/group speed, energy density, breaking
   criteria, and surfable wave speed.
4. **Hydrodynamic performance** -- sweep a range of forward speeds, finding the
   equilibrium planing state (trim angle, wetted length, lift, drag, L/D ratio)
   at each speed. This uses the Savitsky planing hull method, ITTC 1957 friction
   correlation, and thin-airfoil fin force model.
5. **Output** -- results are stored in a structured dictionary, optionally
   visualized as a 6-panel Plotly dashboard, and optionally exported as JSON
   for the Three.js viewer.

---

## 2. The Physics

### 2.1 Linear Wave Theory

**File:** `SurfPhysics/waves/linearWaveTheory.py`

The wave model follows Dean & Dalrymple's treatment of small-amplitude (Airy)
wave theory. The core assumption is that wave height is small compared to both
wavelength and water depth -- the "linear" in the name comes from linearizing the
free-surface boundary conditions.

#### The Dispersion Relation

Everything starts with the dispersion relation, which ties wave frequency to
wavenumber for a given water depth:

```
omega^2 = g * k * tanh(k * d)
```

where `omega = 2*pi/T` is angular frequency, `k = 2*pi/L` is wavenumber, `g` is
gravity, and `d` is water depth. This is an implicit equation for `k` -- you
cannot solve it analytically. The code uses Newton-Raphson iteration with the
deep-water initial guess `k0 = omega^2 / g`, which converges to machine
precision in 5-10 iterations (`solveDispersionRelation()` in
`linearWaveTheory.py`).

From the wavenumber, wave properties follow directly:

- **Phase speed** (how fast a crest travels): `c = omega / k`
- **Group speed** (how fast energy propagates): `cg = c/2 * (1 + 2kd / sinh(2kd))`
- **Wavelength**: `L = 2*pi / k`

In deep water (`d/L > 0.5`), `tanh(kd) ~ 1` and the waves do not feel the
bottom. In shallow water (`d/L < 0.05`), `tanh(kd) ~ kd` and the phase speed
simplifies to `c = sqrt(g*d)` -- waves become non-dispersive. Most surf zone
conditions fall in the intermediate regime, which is why the full dispersion
solve is necessary.

#### Wave Kinematics

The surface elevation is a simple cosine:

```
eta(x, t) = (H/2) * cos(kx - omega*t)
```

Beneath the surface, the velocity field decays with depth following
hyperbolic profiles:

```
u = (H/2) * omega * cosh(k*(z+d)) / sinh(k*d) * cos(kx - omega*t)
w = (H/2) * omega * sinh(k*(z+d)) / sinh(k*d) * sin(kx - omega*t)
```

These are the orbital velocities that drive particle motion. Near the surface
the orbits are circular (deep water) or elliptical (intermediate depth),
becoming flatter with depth until motion is negligible at `z = -d/2` (the
"deep water" criterion). The code implements `velocityField()`,
`accelerationField()`, and `pressure()` for the full kinematic state at any
point.

#### Wave Energy

The wave carries energy proportional to the square of the wave height:

```
E = (1/8) * rho * g * H^2    [J/m^2]
```

This energy propagates at the group speed, giving an energy flux (power per
unit crest width):

```
P = E * cg    [W/m]
```

This matters for understanding the power available to propel a surfer -- a
1.5 m wave at 10 s period carries roughly 10-15 kW per meter of crest.

#### Breaking Criteria

The code checks two independent breaking conditions:

- **Depth-limited breaking** (McCowan 1894): `H/d > 0.78`
- **Steepness-limited breaking** (Miche 1944): `H/L > 1/7`

Either triggers `isBroken() = True`. The surfable wave speed is estimated as the
shallow-water phase speed at the breaking depth (`d_break = H / 0.78`):

```
c_surf = sqrt(g * H / 0.78)
```

This is the speed a surfer must match to ride the wave.

#### Depth Classification

The code classifies conditions as `'deep'` (`d/L > 0.5`), `'intermediate'`,
or `'shallow'` (`d/L < 0.05`), which tells you how much the seabed influences
the wave field.

---

### 2.2 Buoyancy

**File:** `SurfPhysics/hydrodynamics/buoyancy.py`

#### Archimedes' Principle

The buoyancy model is straightforward: a floating body displaces its own weight
in water. The buoyancy force is:

```
F_b = rho_water * g * V_submerged
```

where `rho_water = 1025 kg/m^3` (standard seawater) and `V_submerged` is the
volume of the board below the waterline. The question is: how deep does the
board sit?

#### Draft Calculation

The equilibrium draft is the waterline height where buoyancy exactly equals
the total weight (board + rider). The code solves this with Brent's method
(`scipy.optimize.brentq`) on the residual:

```
f(draft) = rho * g * V_sub(draft) - m_total * g
```

At each candidate draft, the submerged volume is computed by numerical
integration through `BoardGeometry.computeSubmergedVolume()`, which integrates
cross-sectional area below the waterplane along the board length, accounting
for rocker curvature and trim angle.

Two drafts are computed: board-only (how the board floats without a rider) and
with-rider (the paddling waterline). The buoyancy ratio -- max buoyancy divided
by total weight -- tells you whether the system floats at all. A shortboard
with a 75 kg rider typically has a buoyancy ratio around 1.1-1.3x, meaning the
board floats but sits low in the water.

#### Board Mass Estimation

Board mass comes from material properties:

```
mass = foamDensity * coreVolume + fiberglassDensity * shellVolume
```

The shell volume is `surfaceArea * shellThickness` (1.5 mm for a typical
4+4oz glass schedule). The core volume is `totalVolume - shellVolume`. Foam
density depends on construction: PU (polyurethane) at 35 kg/m^3 or EPS
(expanded polystyrene) at 20 kg/m^3. The fiberglass shell is much denser
(1800 kg/m^3) but very thin, so it contributes a disproportionate fraction of
the total mass.

A typical 6'0" shortboard in PU comes out around 2.5-3.5 kg, which matches
real-world measurements well.

---

### 2.3 Planing Hydrodynamics

**File:** `SurfPhysics/hydrodynamics/planing.py`

This is where the physics gets interesting. A surfboard at rest is a
displacement hull -- it sits in the water and buoyancy supports its weight. But
at surfing speeds, it transitions to a **planing hull**: hydrodynamic pressure
on the bottom surface generates lift, the board rises out of the water, and the
wetted area shrinks dramatically. Understanding this transition is central to
surfboard performance.

#### The Savitsky Method

The planing model adapts Savitsky's 1964 empirical method for prismatic planing
hulls. Originally developed for powerboat design, Savitsky's equations predict
lift coefficient, drag, and wetted area from hull geometry and speed. For
surfboards (which are decidedly non-prismatic), the code adapts the method by
averaging beam and deadrise angle over the wetted region.

The **lift coefficient** for a flat-bottomed planing surface is:

```
CL0 = tau^1.1 * (0.0120 * lambda^0.5 / Cv^2 + 0.0055 * lambda^2.5 / Cv^2)
```

where:
- `tau` = trim angle in degrees (the pitch of the board relative to the water)
- `lambda = wettedLength / beam` (the wetted length-to-beam ratio)
- `Cv = V / sqrt(g * beam)` (the speed coefficient, essentially a beam Froude number)

The **deadrise correction** accounts for the V-shape of the bottom
(concave/convex contour):

```
CL_beta = CL0 - 0.0065 * beta * CL0^0.60
```

where `beta` is the deadrise angle in degrees. Higher deadrise reduces lift
because less of the bottom surface is oriented horizontally.

The **lift force** is then:

```
L = 0.5 * rho * V^2 * beam^2 * CL
```

#### Drag Components

Planing drag has three components, all computed in
`PlaningModel.computePlaningDrag()`:

1. **Pressure drag**: `D_p = L * tan(tau)` -- the horizontal component of the
   lift force due to the tilted bottom surface. This is the dominant drag
   source at planing speeds.

2. **Friction drag**: Skin friction on the wetted bottom surface, computed
   using the ITTC 1957 correlation (see Section 2.4 below).

3. **Spray drag**: An empirical 10% addition to friction drag, accounting for
   energy lost to the spray sheet at the leading edge of the wetted area.

#### Finding Equilibrium

At each speed, the code finds the trim angle where planing lift equals the
total weight. This is the **planing equilibrium** -- the natural running
attitude of the board. The solver sweeps trim angles from 1 to 15 degrees,
finds the maximum lift, and then uses `scipy.optimize.fsolve` to zero in on
the exact trim where `lift = weight`.

The planing threshold speed is estimated from the Froude number criterion:

```
V_planing = sqrt(g * L_board)    (Fn = 1.0)
```

For a 6'0" shortboard (1.83 m), this gives about 4.2 m/s (15 km/h). Below
this speed the board is in displacement mode; above it, planing dominates.

---

### 2.4 Friction Drag

**File:** `SurfPhysics/hydrodynamics/friction.py`

Skin friction is computed using the ITTC 1957 model-ship correlation line,
which is the standard in naval architecture:

```
Cf = 0.075 / (log10(Re) - 2)^2
```

For the laminar regime (`Re < 5e5`), the Blasius flat-plate solution is used
instead:

```
Cf = 1.328 / sqrt(Re)
```

At surfing speeds (2-8 m/s) with typical board lengths (1.5-2.7 m), Reynolds
numbers range from 3e6 to 2e7 -- solidly turbulent. The friction drag force
is:

```
D_f = 0.5 * rho * V^2 * S_wetted * Cf
```

The code also computes **form drag** (pressure drag from the frontal area)
using an empirical form factor of 0.1, which is typical for streamlined bodies
like surfboards.

---

### 2.5 Fin Forces

**File:** `SurfPhysics/hydrodynamics/fins.py`

Fins are modeled using thin airfoil theory with finite-wing corrections. The
lift curve slope for a 2D thin airfoil is `dCl/d_alpha = 2*pi` per radian.
The finite-wing correction reduces this based on the aspect ratio:

```
dCL/d_alpha = 2*pi * AR / (AR + 2)
```

Typical surfboard fins have aspect ratios around 1.5-3.0, so the 3D correction
is significant -- lift per degree of angle of attack is roughly half the 2D
value.

Drag follows the classical decomposition:

```
CD = CD0 + CL^2 / (pi * e * AR)
```

where `CD0 ~ 0.01` is the parasitic (skin friction) drag and the second term
is induced drag with Oswald efficiency `e ~ 0.85`.

The `FinForceModel` builds the full fin configuration (thruster, twin, quad,
or single) with dimensions scaled from the C# FinSystem.cs, then sums forces
across all fins accounting for cant angle projection and differential yaw
angle of attack.

---

### 2.6 Force Balance

**File:** `SurfPhysics/hydrodynamics/forceBalance.py`

The `ForceBalance` class brings everything together. At a given speed and
trim angle, it computes:

- **Weight** (downward): `W = m_total * g`
- **Buoyancy** (upward): from Archimedes, dependent on draft
- **Planing lift** (upward): from Savitsky, dependent on speed and trim
- **Planing drag** (aft): pressure + friction + spray
- **Form drag** (aft): frontal area pressure drag
- **Fin drag** (aft): parasitic + induced drag from fin system
- **Fin side force** (lateral): for yaw/turn analysis

The `performanceCurves()` method sweeps speed from 0.5 to 10 m/s (configurable),
finding the planing equilibrium at each point and recording the resulting trim
angle, wetted length, lift, drag, and L/D ratio. These curves are the primary
performance characterization of the board.

---

## 3. Module Architecture

The package is organized into four subpackages plus top-level utilities:

```
SurfPhysics/
    __init__.py                         # Package exports
    analyzer.py                         # PhysicsAnalyzer (top-level pipeline)
    constants.py                        # Physical constants (SI units)
    units.py                            # Unit conversion helpers (mmToM, mToMm)

    geometry/
        parameters.py                   # SurfboardParameters dataclass
        board.py                        # BoardGeometry facade
        outline.py                      # Planform outline (half-width vs t)
        rocker.py                       # Rocker profile (Z-offset vs t)
        crossSection.py                 # Cross-section (deck/bottom heights)
        meshLoader.py                   # STL mesh-based geometry (optional)

    waves/
        waveConditions.py               # WaveConditions dataclass
        linearWaveTheory.py             # Airy wave theory implementation

    hydrodynamics/
        protocols.py                    # ForceResult, PlaningState, ForceModel
        buoyancy.py                     # BuoyancyModel (mass, draft, forces)
        planing.py                      # PlaningModel (Savitsky method)
        friction.py                     # FrictionDragModel (ITTC 1957)
        fins.py                         # FinForceModel (thin airfoil theory)
        forceBalance.py                 # ForceBalance (combined solver)

    visualization/
        visualizer.py                   # Visualizer class (dashboard wrapper)
        dashboard.py                    # 6-panel Plotly dashboard
        theme.py                        # Plotly theme/color constants

    export/
        viewerExporter.py               # ViewerExporter (JSON for Three.js)
        meshSampler.py                  # Parametric surface sampling
```

**Key design decisions:**

- **Geometry is parametric, not mesh-based.** `BoardGeometry` composes
  `Outline`, `RockerProfile`, and `CrossSection` from `SurfboardParameters`
  using the same power-curve and cosine formulas throughout, so physics and
  geometry are always consistent.

- **All geometry classes work in normalized coordinates.** Position along the
  board is always `t` in [0, 1] (0 = nose, 1 = tail). Lateral position is
  `lateralFraction` in [0, 1] (0 = centerline, 1 = rail). Conversion to SI
  meters happens at the `BoardGeometry` boundary.

- **Dimensions are stored in mm, computed in SI meters.** `SurfboardParameters`
  holds all dimensions in millimeters. `BoardGeometry`
  converts to meters at its public interface. This avoids unit confusion at the
  physics layer while keeping parameter files human-readable.

- **Force models are composable.** Each force model (buoyancy, planing,
  friction, fins) is an independent class. `ForceBalance` composes them and
  provides the equilibrium solver. This makes it easy to swap or extend
  individual models.

---

## 4. API Quick Start

The simplest way to use SurfPhysics is through the top-level `PhysicsAnalyzer`:

```python
from SurfPhysics import PhysicsAnalyzer

analyzer = PhysicsAnalyzer()
results = analyzer.runAnalysis('configs/shortboard_default.json')
```

This prints a formatted console summary and returns a results dictionary. To
visualize:

```python
from SurfPhysics import Visualizer

viz = Visualizer()
fig = viz.createDashboard(results)
```

This opens a 6-panel interactive Plotly dashboard in your browser showing
board outline, rocker profile, wave profile, drag breakdown, lift-to-drag
ratio, and planing equilibrium curves.

To export for the Three.js viewer:

```python
from SurfPhysics import ViewerExporter

exporter = ViewerExporter()
outputPath = exporter.exportForViewer(results)
print(f'Exported to: {outputPath}')
```

---

## 5. Programmatic Use

For more control, you can use individual components directly. This is useful
for parameter sweeps, custom analysis, or integrating specific models into
other tools.

### Board Geometry

```python
from SurfPhysics import SurfboardParameters
from SurfPhysics.geometry.board import BoardGeometry

params = SurfboardParameters.fish()
board = BoardGeometry(params)

volume = board.computeVolume()           # m^3
area = board.computePlanformArea()       # m^2
halfWidth = board.getHalfWidthM(0.5)     # m, at midpoint
rocker = board.getRockerHeightM(0.0)     # m, at nose tip
```

### Wave Analysis

```python
from SurfPhysics import WaveConditions
from SurfPhysics.waves.linearWaveTheory import LinearWaveTheory

waves = WaveConditions.typicalBeachBreak()  # H=1.5m, T=10s, d=2.5m
model = LinearWaveTheory()

wavelength = model.waveLength(waves)        # m
phaseSpeed = model.waveSpeed(waves)         # m/s
groupSpeed = model.groupSpeed(waves)        # m/s
energy = model.energyDensity(waves)         # J/m^2
isBroken = model.isBroken(waves)            # bool
surfSpeed = model.surfableWaveSpeed(waves)  # m/s

# Velocity at a point under the wave
u, w = model.velocityField(x=5.0, z=-1.0, t=0.0, waveConditions=waves)
```

### Custom Wave Conditions

```python
from SurfPhysics import WaveConditions

# Manual specification
bigDay = WaveConditions(height=2.5, period=14.0, depth=4.0)

# Or use presets
small = WaveConditions.smallDay()       # H=0.6m, T=8s, d=1.5m
overhead = WaveConditions.overhead()    # H=2.0m, T=12s, d=3.0m
```

### Buoyancy Analysis

```python
from SurfPhysics.hydrodynamics.buoyancy import BuoyancyModel

buoyancy = BuoyancyModel(board, params)
boardMass = buoyancy.estimateBoardMass()                     # kg
boardDraft = buoyancy.findEquilibriumDraft(boardMass)        # m
riderDraft = buoyancy.findEquilibriumDraft(boardMass + 75.0) # m
maxBuoyancy = buoyancy.buoyancyForce(volume)                 # N
```

### Force Balance and Performance Curves

```python
from SurfPhysics.hydrodynamics.forceBalance import ForceBalance

forces = ForceBalance(board, params, riderMassKg=75.0)
curves = forces.performanceCurves(minSpeed=0.5, maxSpeed=10.0, nPoints=50)

# curves is a dict with numpy arrays:
# curves['speed'], curves['trimAngleDeg'], curves['liftN'],
# curves['dragN'], curves['liftToDrag'], curves['wettedLengthM']
```

---

## 6. Results Structure

The dictionary returned by `PhysicsAnalyzer.runAnalysis()` has the following
structure:

```python
results = {
    'params': SurfboardParameters,       # The board parameters object
    'waveConditions': WaveConditions,     # The wave conditions object
    'riderMass': float,                   # Rider mass in kg

    'board': {
        'volumeL': float,                # Board volume in liters
        'planformAreaCm2': float,        # Planform area in cm^2
        'wettedAreaCm2': float,          # Wetted surface area in cm^2
        'massKg': float,                 # Estimated board mass in kg
    },

    'buoyancy': {
        'maxBuoyancyN': float,           # Max buoyancy force (fully submerged) in N
        'totalWeightN': float,           # Total weight (board + rider) in N
        'boardDraftCm': float,           # Board-only equilibrium draft in cm
        'riderDraftCm': float,           # With-rider equilibrium draft in cm
        'buoyancyRatio': float,          # maxBuoyancy / totalWeight (>1 = floats)
        'floats': bool,                  # True if buoyancy exceeds weight
    },

    'waves': {
        'wavelengthM': float,            # Wavelength in m
        'phaseSpeedMs': float,           # Phase speed in m/s
        'groupSpeedMs': float,           # Group speed in m/s
        'energyDensityJm2': float,       # Energy density in J/m^2
        'energyFluxWm': float,           # Energy flux in W/m
        'depthClass': str,               # 'deep', 'intermediate', or 'shallow'
        'isBroken': bool,                # True if breaking criteria met
        'surfableSpeedMs': float,        # Surfable wave speed in m/s
    },

    'performance': {
        'speeds': np.ndarray,            # Speed points in m/s
        'trimAnglesDeg': np.ndarray,     # Equilibrium trim angles in degrees
        'liftN': np.ndarray,             # Lift force at each speed in N
        'dragN': np.ndarray,             # Drag force at each speed in N
        'liftToDrag': np.ndarray,        # L/D ratio at each speed
        'wettedLengthM': np.ndarray,     # Wetted length at each speed in m
        'planingThresholdMs': float,     # Planing onset speed in m/s (Fn = 1.0)
    },
}
```

The `'performance'` arrays are all the same length (default 50 points from
0.5 to 10.0 m/s) and can be plotted directly or indexed into for specific
speed conditions.

---

## 7. Configuration

Analysis is driven by JSON config files in the `configs/` directory. Here is
the full schema with defaults:

```json
{
    "board": {
        "type": "shortboard",
        "finConfiguration": "thruster",
        "foamType": "pu",
        "customParametersFile": null
    },
    "waves": {
        "height": 1.5,
        "period": 10.0,
        "depth": 2.5
    },
    "rider": {
        "mass": 75.0
    },
    "analysis": {
        "speedRange": [0.5, 10.0],
        "nSpeedPoints": 50
    }
}
```

### Board Section

| Field | Type | Description |
|---|---|---|
| `type` | `str` | Board preset: `'shortboard'`, `'longboard'`, `'fish'`, or `'custom'` |
| `finConfiguration` | `str` | Fin setup: `'thruster'`, `'twin'`, `'quad'`, `'single'`, or `'default'` (uses preset) |
| `foamType` | `str` | Construction: `'pu'` (polyurethane, 35 kg/m^3) or `'eps'` (polystyrene, 20 kg/m^3) |
| `customParametersFile` | `str` | Path to a custom JSON parameters file (only used when type is `'custom'`) |

### Waves Section

| Field | Type | Default | Description |
|---|---|---|---|
| `height` | `float` | `1.5` | Wave height H in meters (trough to crest) |
| `period` | `float` | `10.0` | Wave period T in seconds |
| `depth` | `float` | `2.5` | Water depth d in meters |

These three values fully define the wave field under linear theory. Typical
surfing conditions range from H=0.5-3.0 m, T=6-16 s, d=1.0-5.0 m. The
defaults (H=1.5 m, T=10 s, d=2.5 m) represent chest-to-shoulder-high surf
at a typical beach break.

### Rider Section

| Field | Type | Default | Description |
|---|---|---|---|
| `mass` | `float` | `75.0` | Rider mass in kg |

### Analysis Section

| Field | Type | Default | Description |
|---|---|---|---|
| `speedRange` | `[float, float]` | `[0.5, 10.0]` | Min and max speed in m/s for performance curves |
| `nSpeedPoints` | `int` | `50` | Number of speed points in the sweep |

The parametric geometry pipeline is always used -- `SurfboardParameters`
defines the shape and `BoardGeometry` computes everything analytically.

---

## References

- Dean, R.G. & Dalrymple, R.A. -- *Water Wave Mechanics for Engineers and Scientists*
- Savitsky, D. (1964) -- Hydrodynamic Design of Planing Hulls, Marine Technology and SNAME News
- ITTC 1957 Friction Line -- International Towing Tank Conference
- Anderson, J. -- *Fundamentals of Aerodynamics* (thin airfoil theory)
- Gudimetla et al. (2019) -- CFD analysis of surfboard fin configurations
- Falk et al. (2020) -- CFD surfboard drag methodology
- Paine, F. -- *The Science of Surfing* (Cambridge University Press)
- Matveev (2024) -- Bottom modifications for planing boards
