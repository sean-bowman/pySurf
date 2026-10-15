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

## Solitary Wave Breaking on Slopes

References behind `pysurf.BreakingWave`: the breaking criterion, the validation values and the runup law.

### Breaking criterion and characteristics for solitary waves on slopes

- **Citation:** Grilli, S.T., Svendsen, I.A. & Subramanya, R. (1997), J. Waterway, Port, Coastal, Ocean Eng. 123(3), 102-112. doi:10.1061/(ASCE)0733-950X(1997)123:3(102)
- **Accessed:** 2026-10-11, through the values tabulated by Senthilkumar (2016) below
- **Relevance:** The breaking criterion and breaking point that `waveTheory.solitarySlopeParameter`, `waveTheory.solitaryBreakerType` and the `BreakingWave` validation are built on.
- **Key findings:**
  - Slope parameter S0 = 1.521 tan(beta) / sqrt(H0 / h0) classifies the breaker: spilling below 0.025, plunging to 0.30, surging to 0.37, no breaking above.
  - No breaking for H0 / h0 < 16.9 tan^2(beta), the same bound written as a height.
  - The breaking point is where the front face of the wave first turns vertical, computed with a fully nonlinear potential flow model.
  - For H0 / h0 = 0.3 on a 1:35 slope, the breaking index is Hb / hb = 1.380.

### Solitary wave shoaling and breaking in a regularized Boussinesq system

- **URL:** <https://arxiv.org/abs/1601.06822>
- **Citation:** Senthilkumar, A. (2016), arXiv:1601.06822
- **Accessed:** 2026-10-11
- **Relevance:** Tabulates Grilli et al. (1997) breaking values alongside Boussinesq and experimental results, which gives the reference numbers for the `BreakingWave` validation case.
- **Key findings:**
  - Table 8 compares the relative breaking wave height Hb / hb for initial amplitudes 0.2, 0.25, 0.3 and 0.4 on a 1:35 slope against Grilli et al. (1997) and Chou et al.
  - For H0 / h0 = 0.3 on 1:35 the Grilli values are Hb / hb = 1.380 and Hb / h0 = 0.476.
  - The regularized Boussinesq model breaks with a lower Hb / hb than the fully nonlinear model, which the author attributes to its lower-order nonlinearity.

### The runup of solitary waves

- **Citation:** Synolakis, C.E. (1987), J. Fluid Mech. 185, 523-545. doi:10.1017/S002211208700329X
- **Accessed:** 2026-10-11, from the standard citation; not used for a pass/fail check
- **Relevance:** The runup law a non-breaking solitary wave follows, and the experimental breaking runup data on a 1:19.85 slope.
- **Key findings:**
  - Non-breaking runup R / d = 2.831 sqrt(cot beta) (H / d)^(5/4).
  - The law overpredicts the runup of breaking waves; breaking runup was measured separately on a 1:19.85 slope.

---

## Free-Surface (Depth-Integrated) Models

References behind `pysurf.sgnSolver` and `pysurf.SurfaceBreakingWave`: the Serre-Green-Naghdi equations, their numerical solution, the breaking switch, and the evidence that only fully nonlinear forms predict breaking.

### A new approach to handle wave breaking in fully non-linear Boussinesq models

- **Citation:** Tissier, M., Bonneton, P., Marche, F., Chazel, F. & Lannes, D. (2012), Coastal Engineering 67, 54-66. doi:10.1016/j.coastaleng.2012.04.004
- **URL:** <https://hal.archives-ouvertes.fr/hal-00798996>
- **Accessed:** 2026-10-11
- **Relevance:** The breaking switch `sgnSolver` implements, with its published parameters.
- **Key findings:**
  - Breaking is handled by switching locally from SGN to the nonlinear shallow-water equations near the wave front, so the front becomes a shock whose dissipation follows from shock theory with no tuned breaking coefficient.
  - Initiation: maximal local front slope above a critical angle of 30 degrees, the optimum Cienfuegos et al. (2010) found for their SGN model; Lynett (2006) identified the front-slope criterion as the least sensitive threshold.
  - Termination: bore Froude number Fr1 = sqrt(((2 h2 / h1 + 1)^2 - 1) / 8) below 1.3, from bore-cessation experiments (1.2 to 1.3).
  - The shallow-water zone is centered on the front, 2.5 roller lengths wide, with the roller 2.9 H long (Haller and Catalan 2010); results depend weakly on its width as long as it exceeds the roller.
  - Fronts are located from the energy-dissipation peaks of the shallow-water step; switching is abrupt, with small disturbances and no filtering.

### A splitting approach for the fully nonlinear and weakly dispersive Green-Naghdi model

- **Citation:** Bonneton, P., Chazel, F., Lannes, D., Marche, F. & Tissier, M. (2011), J. Comput. Phys. 230, 1479-1498
- **Accessed:** 2026-10-11, through Tissier et al. (2012), which summarizes the scheme
- **Relevance:** The hyperbolic-dispersive splitting `sgnSolver` follows.
- **Key findings:**
  - The SGN momentum equation separates into the shallow-water equations in conservative form and a dispersive correction solved as an elliptic (here tridiagonal) problem.
  - The hyperbolic part is solved with shock-capturing finite volumes and MUSCL reconstruction; the dispersive part with finite differences.
  - An optimization parameter alpha = 1.159 improves linear dispersion; `sgnSolver` uses alpha = 1, the original equations, which keep the exact solitary wave as a check.

### A fast and stable well-balanced scheme with hydrostatic reconstruction for shallow water flows

- **Citation:** Audusse, E., Bouchut, F., Bristeau, M.-O., Klein, R. & Perthame, B. (2004), SIAM J. Sci. Comput. 25(6), 2050-2065
- **Accessed:** 2026-10-11, from the standard citation; the scheme is checked in tests/testSgnSolver.py
- **Relevance:** The well-balanced, depth-positive finite volume scheme for the shallow-water part, including wet-dry fronts.
- **Key findings:**
  - Face depths are reconstructed against the higher bed of the two sides, and the hydrostatic pressure they leave out returns as a face correction, so a lake at rest stays at rest exactly.
  - The scheme keeps the depth non-negative under a CFL condition, which lets water run up a dry beach.

### Shoaling of solitary waves on plane beaches

- **Citation:** Grilli, S.T., Subramanya, R., Svendsen, I.A. & Veeramony, J. (1994), J. Waterway, Port, Coastal, Ocean Eng. 120(6), 609-628
- **URL:** <https://personal.egr.uri.edu/grilli/GSS94_abs.html>
- **Accessed:** 2026-10-11
- **Relevance:** Why the free-surface model has to be fully nonlinear.
- **Key findings:**
  - Fully nonlinear potential flow computations and laboratory measurements of solitary wave shoaling on a 1:35 slope agree closely up to breaking.
  - The nonlinear shallow-water equations and the standard Boussinesq approximation both fail to predict that shoaling.

### A fully nonlinear Boussinesq model for surface waves. Part 1. Highly nonlinear unsteady waves

- **Citation:** Wei, G., Kirby, J.T., Grilli, S.T. & Subramanya, R. (1995), J. Fluid Mech. 294, 71-92. doi:10.1017/S0022112095002813
- **URL:** <https://personal.egr.uri.edu/grilli/WKGS95_abs.html>
- **Accessed:** 2026-10-11
- **Relevance:** Evidence that fully nonlinear depth-integrated models track the potential flow solution for shoaling solitary waves.
- **Key findings:**
  - Compared with Grilli's boundary element solution for solitary waves shoaling on slopes, the fully nonlinear model predicts wave heights, phase speeds and particle kinematics more accurately than the standard Boussinesq approximation.
  - Removing the small-nonlinearity assumption allows strong wave interactions up to breaking.

### Modeling of depth-induced wave breaking in a fully nonlinear free-surface potential flow model

- **Citation:** Papoutsellis, C.E., Yates, M.L., Simon, B. & Benoit, M. (2019), arXiv:1910.08982
- **URL:** <https://arxiv.org/pdf/1910.08982>
- **Accessed:** 2026-10-11
- **Relevance:** The surface-variable potential flow alternative, and a review of breaking treatments.
- **Key findings:**
  - A fully nonlinear, fully dispersive potential flow model written in surface variables, with breaking added through an eddy viscosity.
  - Cites the transition time T* = 5 sqrt(h / g) of Kennedy et al. (2000) for their eddy-viscosity breaking model.

---

## Overturning Waves and Reef Breaks

References behind `pysurf.bemSolver`, `pysurf.OverturningWave` and `pysurf.ReefBreak`: the boundary element method for a free surface that folds, the internal wave source and breaking transition for swell, and the reef and surf-break data that would validate the reef cases.

### The deformation of steep surface waves on water I. A numerical method of computation

- **Citation:** Longuet-Higgins, M.S. & Cokelet, E.D. (1976), Proc. R. Soc. Lond. A 350, 1-26
- **Accessed:** 2026-10-11, from the standard citation
- **Relevance:** The mixed Eulerian-Lagrangian method `bemSolver` follows, and its five-point smoothing.
- **Key findings:**
  - Free-surface nodes move with the water while the potential is solved on the boundary at each step, which lets the surface overturn.
  - A two-node sawtooth instability grows on the surface; a five-point smoothing applied every few steps removes it.

### An efficient boundary element method for nonlinear water waves

- **Citation:** Grilli, S.T., Skourup, J. & Svendsen, I.A. (1989), Engineering Analysis with Boundary Elements 6(2), 97-107
- **Accessed:** 2026-10-11, from the standard citation
- **Relevance:** The numerical wave tank Grilli et al. (1997) computed the breaking criterion with, the same class of model as `bemSolver`.
- **Key findings:**
  - Boundary elements with higher-order shape functions and a second-order Taylor series in time.
  - `bemSolver` uses linear elements with closed-form integrals and fourth-order Runge-Kutta instead.

### Numerical study of three-dimensional overturning waves in shallow water

- **Citation:** Guyenne, P. & Grilli, S.T. (2006), J. Fluid Mech. 547, 361-388. doi:10.1017/S0022112005007317
- **URL:** <https://personal.egr.uri.edu/grilli//JFM-GG-6-30-05.pdf>
- **Accessed:** 2026-10-11
- **Relevance:** Supports running the barrel shape in 2D slices along the wave's path.
- **Key findings:**
  - Solitary waves shoaling and breaking over a sloping ridge in a 3D fully nonlinear potential flow tank, focused on plunging breakers.
  - The ridge's transverse variation changes the evolution, shape and kinematics of the breaker, but the jet shape and dynamics at the middle cross-section are remarkably similar to 2D results.

### Generation of waves in Boussinesq models using a source function method

- **Citation:** Wei, G., Kirby, J.T. & Sinha, A. (1999), Coastal Engineering 36, 271-299
- **Accessed:** 2026-10-11, from the standard citation; the source amplitude in `sgnSolver.WaveSource` is derived for the SGN equations from the linearized system and checked numerically in tests/testSgnSolver.py
- **Relevance:** The internal wave generator `ReefBreak` drives its swell with.
- **Key findings:**
  - A Gaussian mass source inside the domain radiates waves both ways, so absorbing layers can sit behind it and reflected waves pass through it.
  - The source width is set by gamma = 80 / (delta^2 L^2), with delta of 0.3 to 0.5.

### Boussinesq modeling of wave transformation, breaking, and runup. I: 1D

- **Citation:** Kennedy, A.B., Chen, Q., Kirby, J.T. & Dalrymple, R.A. (2000), J. Waterway, Port, Coastal, Ocean Eng. 126(1), 39-47
- **Accessed:** 2026-10-11, through Papoutsellis et al. (2019), which restates the transition time
- **Relevance:** The transition time `sgnSolver` waits before a weak bore stops breaking.
- **Key findings:**
  - An eddy-viscosity breaking model localized on the front face of breaking waves, with a slotted beach for runup.
  - A transition time T* = 5 sqrt(h / g) for the onset and cessation of breaking.

### The effect of relative crest submergence on wave breaking over submerged slopes

- **Citation:** Blenkinsopp, C.E. & Chaplin, J.R. (2008), Coastal Engineering 55(12), 967-974. doi:10.1016/j.coastaleng.2008.03.004
- **URL:** <https://researchportal.bath.ac.uk/en/publications/the-effect-of-relative-crest-submergence-on-wave-breaking-over-su/>
- **Accessed:** 2026-10-11; full text read from docs/papers/blenkinsopp2008.pdf on 2026-10-15
- **Relevance:** The flume `ReefOverturn` reproduces and is checked against (tests/testValidationReefOverturn.py). The reference values are its points for H0/L0 = 0.0174, digitized from figures 6, 7, 10 and 11 by marker color.
- **Key findings:**
  - Flume 17 m long and 0.44 m wide, 0.7 m deep, flap paddle with active absorption; reef 0.618 m high with a 1:10 face and a vertical back, its toe 3.8 m from the paddle; nine water levels give hc = 62 to 102 mm.
  - Break point where any part of the face first turns vertical (Iversen 1952); Hb from the trough ahead to the crest, hb the depth there. Hb/hb is near 0.85 at every submergence and steepness, nearer McCowan's 0.78 than the plane-slope relations; Hb/H0 = 0.5 (H0/L0)^-0.2.
  - Cavity at the plunge point fitted with the Longuet-Higgins curve: lc/wc 1.46 to 2.29 with no clear trend (mean 1.76, length error about 10 percent); Ac = (2 sqrt(3) / 5) lc wc, and Ac/Hb^2 rises as the submergence falls (error about 20 percent). Side-wall friction on the jet is visible in many cases.
  - Transmission Kt rises from about 0.54 to 0.84 over hc/Hi = 0.9 to 1.47, and reflection Kr from 12 to 20 percent over hc/H0 = 0.83 to 1.34, for H0/L0 = 0.0174; the foam beach reflects 0.8 to 8.5 percent.
  - The hc/Hi of figure 10 puts the flume's incident height for the 0.0174 waves at 68.4 to 69.6 mm, the deep-water 75.5 mm shoaled by linear theory to 0.7 m.

### Predicting the breaking intensity of surfing waves

- **Citation:** Mead, S.T. & Black, K.P. (2001), J. Coastal Research SI 29, 51-65
- **URL:** <https://ref.coastalrestorationtrust.org.nz/documents/predicting-the-breaking-intensity-of-surfing-waves/>
- **Accessed:** 2026-10-11, abstract and secondary sources
- **Relevance:** A field relation between reef shape and barrel shape at surf breaks.
- **Key findings:**
  - The seabed gradient along the wave's path (orthogonal gradient), not normal to the contours, controls breaking intensity at surf breaks, where waves meet the contours at an angle.
  - Vortex ratio (tube length over width) Y = 0.065 X + 0.821 against the orthogonal seabed gradient X, from world-class surf breaks; intensity also rises with period and falls with height.
  - Robertson et al. (2014, ICCE) found the vortex ratio cannot be predicted reliably from standard breaking parameters for irregular field waves, so it is guidance, not a validation target.

### Scaling the shape of shoaling and overturning solitary waves

- **Citation:** Pick, K. & Feddersen, F. (2026), J. Fluid Mech. 1040, A8. doi:10.1017/jfm.2026.11869 (open access, CC BY 4.0)
- **URL:** <https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/scaling-the-shape-of-shoaling-and-overturning-solitary-waves/D43EDB6346C8975E77258A291CFCB4EF>
- **Accessed:** 2026-10-12
- **Relevance:** Overturn shape at jet impact for solitary waves on plane slopes from a fully nonlinear potential flow model, the benchmark `OverturningWave` overturn measures are compared with across slopes.
- **Key findings:**
  - 30 cases, s = 1/100 to 1/10 and H0/h0 = 0.2 to 0.6, in the potential flow model surftank (Hanson 2023), starting from an essentially exact solitary wave (Dutykh & Clamond 2014); slope ends in a flat at hs/h0 = 0.05.
  - Overturn area A_O (enclosing the void), jet area A_J (water where the surface is multi-valued in x), both over H_I^2 with H_I the highest elevation at impact; W/L and theta_O from a Longuet-Higgins (1982) curve fitted to the overturn boundary.
  - With psi0 = s / (H0/h0)^(1/4): A_O/H_I^2 = 5.319 psi0 - 0.043 (RMSE 0.013); A_J/H_I^2 = 37.072 psi0^2 - 0.587 psi0 + 0.020 (RMSE 0.005); W/L = 1.661 psi0 + 0.298 (RMSE 0.010); theta_O = -5746.4 psi0^2 + 225.2 psi0 + 48.4 deg (RMSE 1.2 deg); gamma_b = 0.871 exp(11.874 psi0) (RMSE 0.078).
  - Restates Blenkinsopp & Chaplin (2008): periodic waves over a 1:10 wedge reef, A_O/H_I^2 0.05 to 0.35, W/L 0.43 to 0.67, H0/hc 0.625 to 1.25.

### Cross-shore wind-induced changes to field-scale overturning wave shape

- **Citation:** Feddersen, F., Fincham, A.M., Brodie, K.L., Young, A.P., Spydell, M.S., Grimes, D.J., Pieszka, M. & Hanson, K. (2023), J. Fluid Mech. 958, A4. doi:10.1017/jfm.2023.40
- **URL:** <https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/crossshore-windinduced-changes-to-fieldscale-overturning-wave-shape/4236FD21F9A9EA5AFCF6ED8966B93596>
- **Accessed:** 2026-10-12
- **Relevance:** Field-scale measurements of overturn shape for solitons at the Surf Ranch wave basin, the hardware data for the overturn measures.
- **Key findings:**
  - Hydrofoil-generated solitons of H about 2 m in 2.5 m of water, obliquely incident on a slope rising to a bar 0.95 m deep; 22 waves, overturns measured with a lidar and fitted to the Longuet-Higgins (1982) curve, A = (2 sqrt(3) / 5) L W.
  - A/H^2 from 0.2 to 0.42 and W/L from 0.26 to 0.5, both falling as onshore wind rises; cross-wave wind, not H/h, dominated the variation.
  - Zero-wind values quoted by Pick & Feddersen (2026): A/H^2 = 0.31, W/L = 0.38.

### Modelling wind-induced changes to overturning wave shape

- **Citation:** Feddersen, F., Hanson, K., Mostert, W. & Fincham, A. (2024), J. Fluid Mech. 1000, A48. doi:10.1017/jfm.2024.1012
- **URL:** <https://falk.ucsd.edu/pdf/Feddersen_etal2024JFM.pdf>
- **Accessed:** 2026-10-12
- **Relevance:** Two-phase DNS (Basilisk) of the Surf Ranch soliton in two dimensions, the reference case `OverturningWave` reproduces for overturn shape.
- **Key findings:**
  - Green-Naghdi solitary wave a0/h0 = 0.6 at x0/h0 = 15; flat to x/h0 = 30, slope 0.0693 up to a flat at hs/h0 = 0.371; walls at x/h0 = 0 and 60; wave Reynolds number 4 x 10^4.
  - Impact defined as the jet's lowest point within 0.015 h0 of the water below; geometry measured then, as in the 2023 experiment.
  - Zero wind: impact at x/h0 = 42.05, Hb/h0 = 0.64 (highest elevation at impact), A_O/H_b^2 = 0.352, W/L = 0.300; Pick & Feddersen (2026) quote A_J/H^2 = 0.22 and theta_O = 33 deg for this case.

### Modeling of breaking and post-breaking waves on slopes by coupling of BEM and VOF methods

- **Citation:** Lachaume, C., Biausser, B., Grilli, S.T., Fraunié, P. & Guignard, S. (2003), Proc. 13th Int. Offshore and Polar Eng. Conf. (ISOPE), Honolulu
- **URL:** <https://www.researchgate.net/publication/244470382_Modeling_of_Breaking_and_Post-breaking_Waves_on_Slopes_by_Coupling_of_BEM_and_VOF_Methods>
- **Accessed:** 2026-10-12, through search summaries (full text not retrieved)
- **Relevance:** The established way to carry a potential flow breaker past jet impact: hand the flow to a Navier-Stokes VOF model.
- **Key findings:**
  - A boundary element model computes shoaling to just before impact, then a two-phase VOF model continues through the splash-up, with air trapped under the jet and a rebound.
  - Tested on solitary waves shoaling on plane slopes; weak and strong coupling variants.

### Inertial energy dissipation in shallow-water breaking waves

- **Citation:** Mostert, W. & Deike, L. (2020), J. Fluid Mech. 890, A12. doi:10.1017/jfm.2020.83
- **URL:** <https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/inertial-energy-dissipation-in-shallowwater-breaking-waves/8B818C95BBB1C8BF2723DB611AFB9DCD>
- **Accessed:** 2026-10-12, abstract only; full text read from docs/papers/mostert2020.pdf on 2026-10-14
- **Relevance:** Two-phase DNS of solitary waves breaking on slopes through the active breaking phase, the reference `SplashUp` is checked against (tests/testValidationSplashUp.py).
- **Key findings:**
  - Active breaking runs from the face turning vertical, t1, to the first time after it that the kinetic energy has fallen to the potential, t2; the dissipation rate per unit crest width is the fall in mechanical energy over t2 - t1.
  - Across slopes of 2 to 5 degrees the rate collapses onto an inertial model built on Taylor's hypothesis, (epsilon / epsilon0)(db / d0) = 0.0281 (hb / d0)^3.152 with epsilon0 = rho g^1.5 d0^2.5.
  - The plunging case, H0/h0 = 0.3 on 3 degrees, falls from 0.96 E0 at t1 to 0.66 E0 at t2 (figure 8b), a share of 0.31 of the energy at breaking.
  - Empirical relations tie the local parameters to the offshore conditions, so the dissipation can be predicted from the incident wave.

### Experimental study on plunging breaking waves in deep water

- **Citation:** Lim et al. (2015), J. Geophys. Res. Oceans. doi:10.1002/2014JC010269 (full author list not retrieved)
- **URL:** <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2014JC010269>
- **Accessed:** 2026-10-12, through search summaries
- **Relevance:** Laboratory kinematics of the splash-up after a plunging jet lands, for judging the speeds `SplashUp` produces; deep water, so not a like-for-like reference.
- **Key findings:**
  - The first splash-up follows the jet's impact at once and rises higher than the overturning jet.
  - The splash-up starts at up to 2.14 times the wave phase speed.

### The effect of wave steepness on wave breaking properties over submerged reef

- **Citation:** Xia, K. & Wan, D. (2017), Proc. 8th Int. Conf. Computational Methods (ICCM 2017)
- **URL:** <https://www.sci-en-tech.com/ICCM2017/PDFs/2641-9438-1-PB.pdf>
- **Accessed:** 2026-10-12
- **Relevance:** States the Blenkinsopp & Chaplin (2008) flume conditions, which are behind a paywall.
- **Key findings:**
  - Reef 0.618 m high on a 1:10 slope, 3.8 m from the wavemaker, reference depth 0.7 m; periodic waves of steepness H0/L0 0.02 to 0.07.
  - Quotes Mead & Black's vortex ratio as l_c / w_c = 0.065 / m + 0.821 for seabed gradient m.

---

## Software References

- **DualSPHysics:** Open-source WCSPH solver - [dual.sphysics.org](https://dual.sphysics.org/)
- **PySPH:** Python SPH framework - [pysph.readthedocs.io](https://pysph.readthedocs.io/)
- **SPHysics:** Original Fortran SPH code
