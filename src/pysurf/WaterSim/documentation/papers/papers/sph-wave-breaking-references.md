# SPH Wave Breaking Simulation References

**Compiled:** February 2026
**Purpose:** Reference material for extending WaterSim to 3D with wave-maker boundaries

---

## Core SPH Formulation (DualSPHysics)

**Source:** [DualSPHysics Wiki - SPH Formulation](https://github.com/DualSPHysics/DualSPHysics/wiki/3.-SPH-formulation)

### Kernel Functions

**Cubic Spline (M4):**
```
W(q) = σ × {
    1 - (3/2)q² + (3/4)q³    for 0 ≤ q < 1
    (1/4)(2 - q)³             for 1 ≤ q < 2
    0                          for q ≥ 2
}

Normalization (σ):
    2D: σ = 10 / (7πh²)
    3D: σ = 1 / (πh³)
```

**Wendland C2 (Quintic):**
```
W(q) = σ × (1 - q/2)⁴ × (2q + 1)   for 0 ≤ q < 2

Normalization:
    2D: σ = 7 / (4πh²)
    3D: σ = 21 / (16πh³)
```

### Equation of State (Weakly Compressible)

```
P = (c₀²ρ₀/γ) × [(ρ/ρ₀)^γ - 1]

where:
    γ = 7 (typical for water)
    c₀ = artificial sound speed (chosen to keep Δρ/ρ₀ < 1%)
```

### Viscosity Models

1. **Artificial Viscosity** (Monaghan): α = 0.01 recommended
2. **Laminar Viscosity**: explicit viscous stress with kinematic viscosity
3. **SPS Turbulence**: Smagorinsky constant Cs = 0.12

### Density Diffusion (Delta-SPH)

Reduces numerical oscillations:
```
d(ρ)/dt += δ × h × c × Σⱼ (ρⱼ - ρᵢ) × rᵢⱼ · ∇Wᵢⱼ / |rᵢⱼ|²

with δ ≈ 0.1
```

---

## Piston-Type Wave-Maker

### Biesel Transfer Function (First Order)

Relates piston stroke to wave amplitude:
```
H/S = 2 × (cosh(2kd) - 1) / (sinh(2kd) + 2kd)

where:
    H = target wave height [m]
    S = piston stroke (peak-to-peak) [m]
    k = wavenumber [rad/m]
    d = water depth [m]
```

### Piston Motion

```
X(t) = (S/2) × sin(ωt) × ramp(t)
V(t) = (S/2) × ω × cos(ωt) × ramp(t)

ramp(t) = min(1, t / (n × T))  for n ramp-up cycles
```

### Second-Order Correction (Madsen 1971)

Adds harmonic terms to eliminate parasitic long waves:
```
X₂(t) = (S₂/2) × sin(2ωt)
S₂ ≈ 0.25 × S × kH
```

**Reference:** [Wave generation in SPH numerical tank](https://link.springer.com/article/10.1007/s40722-020-00163-x)

---

## Wave Breaking Criteria

### Geometric Criteria

1. **Depth-limited (McCowan 1894):** H/d > 0.78
2. **Steepness-limited (Miche 1944):** H/L > 1/7 ≈ 0.142

### Kinematic Criterion

Breaking onset when horizontal surface velocity approaches wave celerity:
```
u_crest > 0.85 × c
```

### Breaker Classification

Determined by surf similarity parameter (Iribarren number):
```
ξ = tan(β) / √(H/L)

where:
    β = beach slope angle
    H = wave height
    L = wavelength

ξ < 0.4  → Spilling breakers
ξ > 0.4  → Plunging breakers
ξ > 2.0  → Collapsing/surging breakers
```

**Reference:** [Coastal Wiki - Breaker Index](https://www.coastalwiki.org/wiki/Breaker_index)

---

## Neighbor Search Algorithms

### Spatial Hashing (Cell-Linked List)

- Time complexity: O(N) average case
- Memory: O(N + M) where M = grid cells
- Cell size = 2h (support radius of kernel)
- 3D half-stencil: 14 cells (vs 27 for full stencil)

### Compact Hashing

- Reduces memory for sparse domains
- Uses handle array pointing to compact list
- Memory: O(n×k + m) where n = occupied cells

**Reference:** [Compressed Neighbor Lists for SPH](https://cg.informatik.uni-freiburg.de/publications/2019_CGF_CompressedNeighbors.pdf)

---

## Validation Benchmarks

### 3D Dam Break

- **Reference:** Koshizuka & Oka (1996)
- Compare front position vs time
- Expected accuracy: < 10% error

### Wave Propagation Over Submerged Bar

- **Reference:** [Aalborg University benchmark (2025)](https://link.springer.com/article/10.1007/s42452-025-06651-9)
- Compare with VoF, sigma-transformation, and SPH
- Key metrics: wave height, phase, reflection

### Large-Scale Wave Breaking

- **Reference:** [DualSPHysics barred beach study (2023)](https://www.sciencedirect.com/science/article/pii/S0378383923000868)
- Metrics: breaking position, phase-averaged velocities, vorticity

---

## Key Academic Papers

### SPH Fundamentals
- Monaghan (1992) - Smoothed Particle Hydrodynamics
- Monaghan (1994) - Simulating Free Surface Flows with SPH

### Wave Generation
- Biesel (1951) - Wave maker theory
- Madsen (1971) - On the generation of long waves
- Liu & Frigaard (2001) - Irregular wave generation

### Wave Breaking in SPH
- [SPH wave breaking vorticity study (2019)](https://link.springer.com/article/10.1007/s10652-019-09699-5)
- [Breaking waves on vertical pile (2025)](https://www.mdpi.com/2077-1312/13/6/1005)

### General SPH Reviews
- [SPH for free-surface flows: past, present, future (2015)](https://www.tandfonline.com/doi/abs/10.1080/00221686.2015.1119209)
- [Review of SPH towards converged Lagrangian flow modelling (2019)](https://royalsocietypublishing.org/doi/10.1098/rspa.2019.0801)

---

## Software References

- **DualSPHysics:** Open-source WCSPH solver - [dual.sphysics.org](https://dual.sphysics.org/)
- **PySPH:** Python SPH framework - [pysph.readthedocs.io](https://pysph.readthedocs.io/)
- **SPHysics:** Original Fortran SPH code
