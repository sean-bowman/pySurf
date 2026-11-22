# Particle Methods References -- Graphics SPH Against pySurf

Five sources from graphics SPH, set against pySurf's SPH scheme, with the DOIs of the references they lead to and of the references pySurf already cites. Every DOI below was resolved against OpenAlex or doi.org on 2026-11-21 and its title checked against the paper. "docs/papers" means a copy sits in `docs/papers/` (gitignored).

## Contents

- [Sources reviewed](#sources-reviewed)
- [References these sources lead to](#references-these-sources-lead-to)
- [DOIs of references pySurf cites](#dois-of-references-pysurf-cites)
- [Copies to procure](#copies-to-procure)
  - [Closed: DOIs to procure](#closed-dois-to-procure)
  - [Open copies that refuse automated download](#open-copies-that-refuse-automated-download)

## Sources reviewed

### Smoothed Particle Hydrodynamics Techniques for the Physics Based Simulation of Fluids and Solids

- **URL:** <https://doi.org/10.2312/egt.20191035> (Koschier, Bender, Solenthaler & Teschner, Eurographics 2019 tutorial; docs/papers/SPH_Tutorial.pdf)
- **Accessed:** 2026-11-21
- **Relevance:** The graphics community's survey of SPH: neighbor search, pressure solvers, boundaries, viscosity, vorticity, surface tension, multiphase, solids. It locates which graphics techniques pySurf has and which it lacks.
- **Key findings:**
  - Compact hashing (Ihmsen et al. 2011) sorts occupied cells along a Z-curve and the per-particle data the same way, re-sorted every ~1000 steps, for cache hits; sorting the cells alone leaves the particle data scattered (sec. 3.1).
  - Implicit pressure solvers (PCISPH, IISPH, DFSPH) take larger steps than an equation of state but iterate per step. The gain grows with the height of the fluid column in particles; for a shallow layer the equation of state is faster. DFSPH reports the best overall performance of the three (sec. 4.11).
  - Particle boundaries need dense, uniform sampling that costs neighbors and memory, and a bumpy sampling biases trajectories. Implicit boundaries (density maps, volume maps) replace the boundary particles with a precomputed field over a signed distance function (sec. 5).
  - Vorticity confinement and the micropolar model put back vorticity that numerical dissipation removes, by adding energy through a tuned coefficient (sec. 8).
  - Surface tension (color field, cohesion) and multiphase interfaces get full treatments (secs. 7, 9).

### Particle-Based Fluid Simulation for Interactive Applications

- **URL:** <https://doi.org/10.2312/SCA03/154-159> (Müller, Charypar & Gross, SCA 2003; docs/papers/sca03.pdf)
- **Accessed:** 2026-11-21
- **Relevance:** The paper that brought SPH into real-time graphics and the basis of the Lague simulation below.
- **Key findings:**
  - Poly6 kernel for density, spiky kernel for the pressure gradient (its gradient does not vanish at the center), a separate kernel for the viscosity Laplacian.
  - Ideal-gas pressure k(ρ - ρ0), symmetrized as the mean of the pair's pressures.
  - Surface tension from the curvature of a smoothed color field, applied where its gradient exceeds a threshold.
  - Free surface drawn by point splatting or by marching cubes on the color field; 5000 particles at interactive rates.

### Particle-based Viscoelastic Fluid Simulation

- **URL:** <https://doi.org/10.1145/1073368.1073400> (Clavet, Beaudoin & Poulin, SCA 2005; docs/papers/pvfs.pdf)
- **Accessed:** 2026-11-21
- **Relevance:** The source of the "near density" used by the Lague simulation, and an early two-way coupling of particles with rigid bodies.
- **Key findings:**
  - Prediction-relaxation: positions advance first, then position impulses relax them, with velocities taken from the displacement; stable at large steps where explicit forces are not.
  - Double density relaxation: a pressure k(ρ - ρ0) plus a repulsive near-pressure from a sharper kernel, which prevents clustering and makes surface tension emerge without a separate term.
  - Viscosity as radial impulses, linear and quadratic in the approach speed.
  - Springs with varying rest lengths give elasticity and plasticity; floating rigid bodies exchange impulses with the fluid.

### Particle Simulation using CUDA

- **URL:** NVIDIA CUDA SDK white paper, Simon Green, 2010 (no DOI; docs/papers/particles.pdf)
- **Accessed:** 2026-11-21
- **Relevance:** The uniform-grid GPU neighbor search that most GPU particle codes, Warp's hash grid among them, descend from.
- **Key findings:**
  - Hash each particle by its cell, radix sort the (hash, index) pairs, and find each cell's start in one pass over the sorted list.
  - Positions and velocities are then copied into sorted order, which makes neighbor reads coherent; a Z-order hash improves it further.
  - Interactions are spring-damper collisions (DEM), not SPH.

### SebLague/Fluid-Sim, branch Episode-01

- **URL:** <https://github.com/SebLague/Fluid-Sim/tree/Episode-01>
- **Accessed:** 2026-11-21
- **Relevance:** A Unity compute-shader SPH in 2D and 3D, built on the Müller and Clavet papers above.
- **Key findings:**
  - Density and near density from spiky kernels of power 2 and 3; pressure (ρ - ρ_target) k; near pressure ρ_near k_near; each pair takes the mean of the two pressures.
  - Densities and forces are evaluated at positions predicted 1/120 s ahead with the current velocity, a fixed-lookahead form of Clavet's prediction.
  - Viscosity pulls each velocity toward its neighbors' with the poly6 kernel (an XSPH-like smoothing applied to the velocity itself).
  - Neighbor search: cells of one smoothing radius hashed into a table of N entries, bitonic sort of (index, hash, key), offsets per key; hash collisions are skipped by comparing hashes. Particle data stays in its original order.
  - Walls: a particle outside the box is clamped back with its normal velocity reversed and damped. Particles are drawn as instanced spheres colored by speed.

## References these sources lead to

None of these is cited in pySurf yet. "Addresses" names the pySurf limit or option each bears on.

| Reference | DOI | Addresses | Copy |
| --- | --- | --- | --- |
| Ihmsen, Akinci, Becker & Teschner (2011), A parallel SPH implementation on multi-core CPUs, CGF 30(1) | 10.1111/j.1467-8659.2010.01832.x | Z-order sorting of particle data for cache coherence | closed |
| Koschier & Bender (2017), Density maps for improved SPH boundary handling, SCA | 10.1145/3099564.3099565 | Boundary particles (a third of the shoal run's particles) replaced by a precomputed field | docs/papers/koschier2017.pdf |
| Bender, Kugelstadt, Weiler & Koschier (2019), Volume maps: an implicit boundary representation for SPH, MIG | 10.1145/3359566.3360077 | Same, with volumes in place of densities, cheaper to precompute | closed |
| Band, Gissler, Ihmsen, Cornelis, Peer & Teschner (2018), Pressure boundaries for implicit incompressible SPH, ACM TOG 37(2) | 10.1145/3180486 | Wall pressure for single-layer boundaries | closed |
| Ferrand, Laurence, Rogers, Violeau & Kassiotis (2013), Unified semi-analytical wall boundary conditions in SPH, IJNMF 71 | 10.1002/fld.3666 | The marine-SPH counterpart of implicit boundaries | open copy (HAL, below) |
| Solenthaler & Pajarola (2009), Predictive-corrective incompressible SPH, ACM TOG 28(3) | 10.1145/1531326.1531346 | Implicit pressure: step set by flow speed, not sound speed | open copy at ACM (bronze) |
| Ihmsen, Cornelis, Solenthaler, Horvath & Teschner (2014), Implicit incompressible SPH, IEEE TVCG 20(3) | 10.1109/TVCG.2013.105 | Same | closed |
| Bender & Koschier (2017), Divergence-free SPH for incompressible and viscous fluids, IEEE TVCG 23(3) | 10.1109/TVCG.2016.2578335 | Same; the fastest of the three in the tutorial's comparison | docs/papers/bender2017.pdf |
| Macklin & Müller (2013), Position based fluids, ACM TOG 32(4) | 10.1145/2461912.2461984 | Clavet's relaxation made a constraint solver (graphics only) | closed |
| Winchenbach, Hochstetter & Kolb (2017), Infinite continuous adaptivity for incompressible SPH, ACM TOG 36(4) | 10.1145/3072959.3073713 | Continuous mass redistribution in place of discrete splitting and merging | closed |
| Akinci, Ihmsen, Akinci, Solenthaler & Teschner (2012), Versatile rigid-fluid coupling for incompressible SPH, ACM TOG 31(4) | 10.1145/2185520.2185558 | Two-way coupling of a rigid body (a surfboard) with the water | closed |
| Canelas, Crespo, Domínguez, Ferraz & Gómez-Gesteira (2016), SPH-DCDEM model for arbitrary geometries in free surface solid-fluid flows, CPC 202 | 10.1016/j.cpc.2016.01.006 | The DualSPHysics route to the same coupling | closed |
| van der Laan, Green & Sainz (2009), Screen space fluid rendering with curvature flow, I3D | 10.1145/1507149.1507164 | Rendering a smooth water surface from particles | closed |
| Zhu & Bridson (2005), Animating sand as a fluid, ACM TOG 24(3) | 10.1145/1073204.1073298 | Implicit surface from particles for marching cubes | closed |
| Yu & Turk (2013), Reconstructing surfaces of particle-based fluids using anisotropic kernels, ACM TOG 32(1) | 10.1145/2421636.2421641 | Same, sharper at thin sheets and jets | closed |
| Zhang, Hu & Adams (2017), A weakly compressible SPH method based on a low-dissipation Riemann solver, JCP 335 | 10.1016/j.jcp.2017.01.027 | The artificial viscosity behind the breaking-point lag at 32 ppd | closed |
| Di Mascio, Antuono, Colagrossi & Marrone (2017), SPH method from a large eddy simulation perspective, Phys. Fluids 29 | 10.1063/1.4978274 | A turbulence closure derived inside δ-SPH, for the energy kept after breaking | docs/papers/diMascio2017.pdf |
| Antuono, Marrone, Di Mascio & Colagrossi (2021), the same, generalized to a quasi-Lagrangian model, Phys. Fluids 33 | 10.1063/5.0034568 | Same, with shifting folded in (δ-LES-SPH) | docs/papers/antuono2021.pdf |
| Green, Vacondio & Peiró (2019), An SPH scheme with a consistent diffusion term for the continuity equation, Comput. Fluids 179 | 10.1016/j.compfluid.2018.11.020 | An alternative to the hydrostatic δ term | closed |
| Mokos, Rogers, Stansby & Domínguez (2015), Multi-phase SPH modelling of violent hydrodynamics on GPUs, CPC 196 | 10.1016/j.cpc.2015.06.020 | Air in the tube; the vacuum pocket of a single-phase model | closed |
| Lubin & Glockner (2015), Numerical simulations of three-dimensional plunging breaking waves: aerated vortex filaments, JFM 767 | 10.1017/jfm.2015.62 | A 3D reference for the flow after the jet lands | closed |
| Kiger & Duncan (2012), Air-entrainment mechanisms in plunging jets and breaking waves, Annu. Rev. Fluid Mech. 44 | 10.1146/annurev-fluid-122109-160724 | Same | closed |
| Deike, Melville & Popinet (2016), Air entrainment and bubble statistics in breaking waves, JFM 801 | 10.1017/jfm.2016.372 | Dissipation and air entrainment in 3D two-phase DNS | open copy (HAL, below) |

## DOIs of references pySurf cites

The SPH scheme (`sphSolver`, `sphGpu`, `sphKernels`, `sphBoundaries`, `sphPacking`, `sphRefinement`, README, gpuSph roadmap):

| Reference | DOI | Copy |
| --- | --- | --- |
| Monaghan (1989), On the problem of penetration in particle methods, JCP 82 | 10.1016/0021-9991(89)90032-6 | closed |
| Monaghan (1992), Smoothed particle hydrodynamics, ARA&A 30 | 10.1146/annurev.aa.30.090192.002551 | closed |
| Monaghan (1994), Simulating free surface flows with SPH, JCP 110 | 10.1006/jcph.1994.1034 | closed |
| Monaghan (2000), SPH without a tensile instability, JCP 159 | 10.1006/jcph.2000.6439 | closed |
| Wendland (1995), Piecewise polynomial, positive definite and compactly supported radial functions of minimal degree, Adv. Comput. Math. 4 | 10.1007/BF02123482 | closed |
| Bonet & Lok (1999), Variational and momentum preservation aspects of SPH formulations, CMAME 180 | 10.1016/S0045-7825(99)00051-1 | closed |
| Dalrymple & Rogers (2006), Numerical modeling of water waves with the SPH method, Coastal Eng. 53 | 10.1016/j.coastaleng.2005.10.004 | closed |
| Crespo, Gómez-Gesteira & Dalrymple (2007), Boundary conditions generated by dynamic particles in SPH methods, CMC 5(3) | 10.3970/cmc.2007.005.173 | docs/papers/crespo2007.pdf |
| Lee, Moulinec, Xu, Violeau, Laurence & Stansby (2008), Comparisons of weakly compressible and truly incompressible algorithms for SPH, JCP 227 | 10.1016/j.jcp.2008.06.005 | closed |
| Molteni & Colagrossi (2009), A simple procedure to improve the pressure evaluation in hydrodynamic context using the SPH, CPC 180 | 10.1016/j.cpc.2008.12.004 | closed |
| Antuono, Colagrossi, Marrone & Molteni (2010), Free-surface flows solved by means of SPH schemes with numerical diffusive terms, CPC 181 | 10.1016/j.cpc.2009.11.002 | closed |
| Marrone, Antuono, Colagrossi, Colicchio, Le Touzé & Graziani (2011), δ-SPH model for simulating violent impact flows, CMAME 200 | 10.1016/j.cma.2010.12.016 | closed |
| Lind, Xu, Stansby & Rogers (2012), Incompressible SPH for free-surface flows: a generalised diffusion-based algorithm, JCP 231 | 10.1016/j.jcp.2011.10.027 | closed |
| Adami, Hu & Adams (2012), A generalized wall boundary condition for SPH, JCP 231 | 10.1016/j.jcp.2012.05.005 | closed |
| Colagrossi, Bouscasse, Antuono & Marrone (2012), Particle packing algorithm for SPH schemes, CPC 183 | 10.1016/j.cpc.2012.02.032 | open at publisher (bronze) |
| Vacondio, Rogers, Stansby, Mignosa & Feldman (2013), Variable resolution for SPH: a dynamic particle coalescing and splitting scheme, CMAME 256 | 10.1016/j.cma.2012.12.014 | closed |
| Crespo et al. (2015), DualSPHysics: open-source parallel CFD solver based on SPH, CPC 187 | 10.1016/j.cpc.2014.10.004 | open copy (Universidade de Vigo, below) |
| Frontiere, Raskin & Owen (2017), CRKSPH, JCP 332 | 10.1016/j.jcp.2016.12.004 | docs/papers/frontiere2017.pdf (arXiv) |
| Sun, Colagrossi, Marrone & Zhang (2017), The δplus-SPH model, CMAME 315 | 10.1016/j.cma.2016.10.028 | closed |
| Chiron, Oger, de Leffe & Le Touzé (2018), Analysis and improvements of Adaptive Particle Refinement, JCP 354 | 10.1016/j.jcp.2017.10.041 | closed |
| Fourtakas, Domínguez, Vacondio & Rogers (2019), Local uniform stencil (LUST) boundary condition, Comput. Fluids 190 | 10.1016/j.compfluid.2019.06.009 | open at publisher (CC BY) |
| Rosswog (2020), The Lagrangian hydrodynamics code MAGMA2, MNRAS 498 | 10.1093/mnras/staa2591 | docs/papers/rosswog2020.pdf (arXiv) |
| Domínguez et al. (2022), DualSPHysics: from fluid dynamics to multiphysics problems, Comp. Part. Mech. 9 | 10.1007/s40571-021-00404-2 | docs/papers/dominguez2022.pdf (arXiv) |
| English et al. (2022), Modified dynamic boundary conditions (mDBC) for general-purpose SPH, Comp. Part. Mech. 9 | 10.1007/s40571-021-00403-3 | docs/papers/english2022.pdf |
| Koshizuka & Oka (1996), Moving-particle semi-implicit method for fragmentation of incompressible fluid, Nucl. Sci. Eng. 123 | 10.13182/NSE96-A24205 | closed |
| Violeau & Rogers (2016), SPH for free-surface flows: past, present and future, J. Hydraul. Res. 54 | 10.1080/00221686.2015.1119209 | closed |
| Lind, Rogers & Stansby (2020), Review of SPH: towards converged Lagrangian flow modelling, Proc. R. Soc. A 476 | 10.1098/rspa.2019.0801 | open copy (PubMed Central, below) |

Waves, BEM, SGN and wave makers:

| Reference | DOI | Copy |
| --- | --- | --- |
| Biesel & Suquet (1951), Les appareils générateurs de houle en laboratoire, La Houille Blanche 37(2), 147-165 | 10.1051/lhb/1951033 | closed |
| Serre (1953), Contribution à l'étude des écoulements permanents et variables dans les canaux, La Houille Blanche 39 | 10.1051/lhb/1953034 (pp. 374-388); 10.1051/lhb/1953058 (pp. 830-872) | closed |
| Madsen (1971), On the generation of long waves, JGR 76 | 10.1029/JC076i036p08672 | closed |
| Battjes (1974), Surf similarity, Proc. 14th ICCE | 10.9753/icce.v14.26 | docs/papers/battjes1974.pdf |
| Longuet-Higgins & Cokelet (1976), The deformation of steep surface waves on water I, Proc. R. Soc. A 350 | 10.1098/rspa.1976.0092 | closed |
| Green & Naghdi (1976), A derivation of equations for wave propagation in water of variable depth, JFM 78 | 10.1017/S0022112076002425 | closed |
| Keys (1981), Cubic convolution interpolation for digital image processing, IEEE TASSP 29 | 10.1109/TASSP.1981.1163711 | closed |
| Synolakis (1987), The runup of solitary waves, JFM 185 | 10.1017/S002211208700329X | closed |
| Grilli, Skourup & Svendsen (1989), An efficient boundary element method for nonlinear water waves, EABE 6 | 10.1016/0955-7997(89)90005-2 | closed |
| Dean & Dalrymple (1991), Water Wave Mechanics for Engineers and Scientists (book) | 10.1142/1232 | closed |
| Grilli, Subramanya, Svendsen & Veeramony (1994), Shoaling of solitary waves on plane beaches, JWPCOE 120(6) | 10.1061/(ASCE)0733-950X(1994)120:6(609) | open copy (URI, below) |
| Wei, Kirby, Grilli & Subramanya (1995), A fully nonlinear Boussinesq model for surface waves, part 1, JFM 294 | 10.1017/S0022112095002813 | closed |
| Grilli, Svendsen & Subramanya (1997), Breaking criterion and characteristics for solitary waves on slopes, JWPCOE 123(3) | 10.1061/(ASCE)0733-950X(1997)123:3(102) | open copy (URI, below) |
| Monaghan & Kos (1999), Solitary waves on a Cretan beach, JWPCOE 125(3) | 10.1061/(ASCE)0733-950X(1999)125:3(145) | closed |
| Wei, Kirby & Sinha (1999), Generation of waves in Boussinesq models using a source function method, Coastal Eng. 36 | 10.1016/S0378-3839(99)00009-5 | closed |
| Kennedy, Chen, Kirby & Dalrymple (2000), Boussinesq modeling of wave transformation, breaking, and runup. I: 1D, JWPCOE 126(1) | 10.1061/(ASCE)0733-950X(2000)126:1(39) | closed |
| Mead & Black (2001), Predicting the breaking intensity of surfing waves, J. Coastal Res. SI 29 | none registered | closed |
| Audusse, Bouchut, Bristeau, Klein & Perthame (2004), A fast and stable well-balanced scheme with hydrostatic reconstruction, SIAM J. Sci. Comput. 25 | 10.1137/S1064827503431090 | closed |
| Guyenne & Grilli (2006), Numerical study of three-dimensional overturning waves in shallow water, JFM 547 | 10.1017/S0022112005007317 | open copy (URI, below) |
| Blenkinsopp & Chaplin (2008), The effect of relative crest submergence on wave breaking over submerged slopes, Coastal Eng. 55 | 10.1016/j.coastaleng.2008.03.004 | docs/papers/blenkinsopp2008.pdf |
| Bonneton, Chazel, Lannes, Marche & Tissier (2011), A splitting approach for the fully nonlinear and weakly dispersive Green-Naghdi model, JCP 230 | 10.1016/j.jcp.2010.11.015 | docs/papers/bonneton2011.pdf (arXiv) |
| Tissier, Bonneton, Marche, Chazel & Lannes (2012), A new approach to handle wave breaking in fully non-linear Boussinesq models, Coastal Eng. 67 | 10.1016/j.coastaleng.2012.04.004 | closed |
| Clamond & Dutykh (2013), Fast accurate computation of the fully nonlinear solitary surface gravity waves, Comput. Fluids 84 | 10.1016/j.compfluid.2013.05.010 | docs/papers/clamond2013.pdf (arXiv) |
| Perlin, Choi & Tian (2013), Breaking waves in deep and intermediate waters, Annu. Rev. Fluid Mech. 45 | 10.1146/annurev-fluid-011212-140721 | closed |
| Quiroga & Cheung (2013), Laboratory study of solitary-wave transformation over bed-form roughness on fringing reefs, Coastal Eng. 80 | 10.1016/j.coastaleng.2013.05.002 | closed; the thesis is open (below) |
| Lim et al. (2015), Experimental study on plunging breaking waves in deep water, JGR Oceans 120 | 10.1002/2014JC010269 | closed |
| Senthilkumar (2016), Solitary wave shoaling and breaking in a regularized Boussinesq system, arXiv:1601.06822 | none (arXiv) | docs/papers/senthilkumar2016.pdf |
| Papoutsellis, Yates, Simon & Benoit (2019), Modelling of depth-induced wave breaking in a fully nonlinear free-surface potential flow model, Coastal Eng. 154, 103579 | 10.1016/j.coastaleng.2019.103579 | docs/papers/papoutsellis2019.pdf (arXiv) |
| Mostert & Deike (2020), Inertial energy dissipation in shallow-water breaking waves, JFM 890 | 10.1017/jfm.2020.83 | docs/papers/mostert2020.pdf |
| Feddersen et al. (2023), Cross-shore wind-induced changes to field-scale overturning wave shape, JFM 958 | 10.1017/jfm.2023.40 | docs/papers/feddersen2023.pdf |
| Feddersen, Hanson, Mostert & Fincham (2024), Modelling wind-induced changes to overturning wave shape, JFM 1000 | 10.1017/jfm.2024.1012 | docs/papers/feddersen2024.pdf |
| Pick & Feddersen (2026), Scaling the shape of shoaling and overturning solitary waves, JFM 1040 | 10.1017/jfm.2026.11869 | docs/papers/pick2026.pdf |

Cited without a registered DOI: Goring (1978), Caltech report KH-R-38; Hanson (2023), UCSD thesis; Quiroga (2012), University of Hawaii MS thesis; Lachaume et al. (2003), ISOPE; Xia & Wan (2017), ICCM.

## Copies to procure

None of the papers below has a copy in `docs/papers/`.

### Closed: DOIs to procure

These are the references marked "closed" in the tables above: no legal open copy was found on 2026-11-21.

**Leads the particle methods roadmap draws on** (`planning/particleMethodsRoadmap.md`):

- `10.1111/j.1467-8659.2010.01832.x`: Ihmsen et al. (2011), Z-order sorting; item 1
- `10.1145/3359566.3360077`: Bender et al. (2019), volume maps; item 2
- `10.1145/1073204.1073298`: Zhu & Bridson (2005), particle implicit surface; item 3
- `10.1145/2421636.2421641`: Yu & Turk (2013), anisotropic kernels; item 3
- `10.1016/j.jcp.2017.01.027`: Zhang, Hu & Adams (2017), low-dissipation Riemann solver; item 4a
- `10.1016/j.cpc.2015.06.020`: Mokos et al. (2015), multi-phase SPH on GPUs; item 5
- `10.1145/2185520.2185558`: Akinci et al. (2012), rigid-fluid coupling; item 6
- `10.1016/j.cpc.2016.01.006`: Canelas et al. (2016), SPH-DCDEM; item 6

**Other leads:**

- `10.1145/3180486`: Band et al. (2018), pressure boundaries for IISPH
- `10.1109/TVCG.2013.105`: Ihmsen et al. (2014), implicit incompressible SPH
- `10.1145/2461912.2461984`: Macklin & Müller (2013), position based fluids
- `10.1145/3072959.3073713`: Winchenbach et al. (2017), infinite continuous adaptivity
- `10.1145/1507149.1507164`: van der Laan et al. (2009), screen space fluid rendering
- `10.1016/j.compfluid.2018.11.020`: Green, Vacondio & Peiró (2019), consistent continuity diffusion
- `10.1017/jfm.2015.62`: Lubin & Glockner (2015), 3D plunging breakers
- `10.1146/annurev-fluid-122109-160724`: Kiger & Duncan (2012), air entrainment

**Cited by pySurf, SPH scheme:**

- `10.1016/0021-9991(89)90032-6`: Monaghan (1989), penetration in particle methods
- `10.1146/annurev.aa.30.090192.002551`: Monaghan (1992), SPH review
- `10.1006/jcph.1994.1034`: Monaghan (1994), free surface flows
- `10.1006/jcph.2000.6439`: Monaghan (2000), tensile instability
- `10.1007/BF02123482`: Wendland (1995), compactly supported radial functions
- `10.1016/S0045-7825(99)00051-1`: Bonet & Lok (1999), variational and momentum preservation
- `10.1016/j.coastaleng.2005.10.004`: Dalrymple & Rogers (2006), water waves with SPH
- `10.1016/j.jcp.2008.06.005`: Lee et al. (2008), WCSPH against ISPH
- `10.1016/j.cpc.2008.12.004`: Molteni & Colagrossi (2009), pressure evaluation
- `10.1016/j.cpc.2009.11.002`: Antuono et al. (2010), numerical diffusive terms
- `10.1016/j.cma.2010.12.016`: Marrone et al. (2011), δ-SPH for violent impact
- `10.1016/j.jcp.2011.10.027`: Lind et al. (2012), diffusion-based shifting
- `10.1016/j.jcp.2012.05.005`: Adami, Hu & Adams (2012), generalized wall boundary
- `10.1016/j.cma.2012.12.014`: Vacondio et al. (2013), coalescing and splitting
- `10.1016/j.cma.2016.10.028`: Sun et al. (2017), δplus-SPH
- `10.1016/j.jcp.2017.10.041`: Chiron et al. (2018), adaptive particle refinement
- `10.13182/NSE96-A24205`: Koshizuka & Oka (1996), MPS
- `10.1080/00221686.2015.1119209`: Violeau & Rogers (2016), SPH for free-surface flows

**Cited by pySurf, waves, BEM, SGN and wave makers:**

- `10.1051/lhb/1951033`: Biesel & Suquet (1951), laboratory wave generators
- `10.1051/lhb/1953034` and `10.1051/lhb/1953058`: Serre (1953), both parts
- `10.1029/JC076i036p08672`: Madsen (1971), generation of long waves
- `10.1098/rspa.1976.0092`: Longuet-Higgins & Cokelet (1976), steep surface waves I
- `10.1017/S0022112076002425`: Green & Naghdi (1976), variable depth wave equations
- `10.1109/TASSP.1981.1163711`: Keys (1981), cubic convolution interpolation
- `10.1017/S002211208700329X`: Synolakis (1987), solitary wave runup
- `10.1016/0955-7997(89)90005-2`: Grilli, Skourup & Svendsen (1989), BEM for nonlinear waves
- `10.1142/1232`: Dean & Dalrymple (1991), Water Wave Mechanics (book)
- `10.1017/S0022112095002813`: Wei et al. (1995), fully nonlinear Boussinesq model
- `10.1061/(ASCE)0733-950X(1999)125:3(145)`: Monaghan & Kos (1999), Cretan beach
- `10.1016/S0378-3839(99)00009-5`: Wei, Kirby & Sinha (1999), source function wave generation
- `10.1061/(ASCE)0733-950X(2000)126:1(39)`: Kennedy et al. (2000), Boussinesq breaking and runup
- `10.1137/S1064827503431090`: Audusse et al. (2004), hydrostatic reconstruction
- `10.1016/j.coastaleng.2012.04.004`: Tissier et al. (2012), breaking in Boussinesq models
- `10.1146/annurev-fluid-011212-140721`: Perlin, Choi & Tian (2013), breaking waves review
- `10.1016/j.coastaleng.2013.05.002`: Quiroga & Cheung (2013), solitary waves on fringing reefs
- `10.1002/2014JC010269`: Lim et al. (2015), plunging breakers in deep water
- No DOI: Mead & Black (2001), Predicting the breaking intensity of surfing waves, J. Coastal Res. SI 29

### Open copies that refuse automated download

These are legal open copies behind bot checks; a browser fetches them.

- Crespo et al. 2015: <https://www.investigo.biblioteca.uvigo.es/xmlui/bitstream/11093/6338/5/2015_crespo_dualsphysics.pdf>
- Fourtakas et al. 2019 (CC BY): <https://www.sciencedirect.com/science/article/pii/S0045793019301859/pdf>
- Colagrossi et al. 2012: <https://www.sciencedirect.com/science/article/pii/S0010465512001051>
- Lind, Rogers & Stansby 2020: <https://pmc.ncbi.nlm.nih.gov/articles/PMC7544338/>
- Grilli et al. 1994: <https://digitalcommons.uri.edu/oce_facpubs/211>
- Grilli, Svendsen & Subramanya 1997: <https://digitalcommons.uri.edu/oce_facpubs/207>
- Guyenne & Grilli 2006: <https://digitalcommons.uri.edu/oce_facpubs/167>
- Ferrand et al. 2013: <https://enpc.hal.science/hal-00691603>
- Deike, Melville & Popinet 2016: <https://hal.science/hal-01361280>
- Solenthaler & Pajarola 2009: <https://dl.acm.org/doi/pdf/10.1145/1531326.1531346>
- Quiroga 2012 thesis: <http://hdl.handle.net/10125/100952>
