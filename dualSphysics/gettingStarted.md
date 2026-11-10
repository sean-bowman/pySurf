# DualSPHysics -- Reference Runs of pySurf's Breaking-Wave Cases

This folder runs pySurf's breaking-wave gate cases in DualSPHysics (Domínguez et al. 2022), an established GPU SPH code, and measures them with pySurf's own tools. The runs show how a published weakly compressible SPH code behaves on these cases at the resolutions pySurf can afford. Where both codes fail a check, the failure is likely common to the method at that resolution; where DualSPHysics passes and pySurf fails, its source shows what differs. Every result here is a cross-check. None validates pySurf, and DualSPHysics is never a dependency: its source stays outside the repository, and nothing here imports or copies it.

DualSPHysics v5.4 is built from source in WSL2 with CUDA, on Windows 11 with an RTX 2070 Super (8 GB). The tools are Python run from Windows; they write each case from pySurf's geometry, drive the solver in WSL, and read its binary output back.

## Contents

- [DualSPHysics -- Reference Runs of pySurf's Breaking-Wave Cases](#dualsphysics----reference-runs-of-pysurfs-breaking-wave-cases)
  - [Contents](#contents)
  - [The tools](#the-tools)
  - [Install](#install)
    - [WSL2 and the CUDA toolkit](#wsl2-and-the-cuda-toolkit)
    - [The source](#the-source)
    - [Build](#build)
  - [Running from Windows](#running-from-windows)
  - [Build check: the dam break](#build-check-the-dam-break)
  - [The 2D cases](#the-2d-cases)
    - [The scheme](#the-scheme)
    - [The start: a piston tuned at a gauge](#the-start-a-piston-tuned-at-a-gauge)
    - [Measuring with pySurf's tracker](#measuring-with-pysurfs-tracker)
  - [The 3D cases](#the-3d-cases)
    - [Guyenne and Grilli's ridge](#guyenne-and-grillis-ridge)
    - [The Surf Ranch reef turned across the span](#the-surf-ranch-reef-turned-across-the-span)
  - [Results](#results)
  - [Troubleshooting](#troubleshooting)
  - [References](#references)

## The tools

| File | Does |
| --- | --- |
| `tools/buildDualSphysics.sh` | Builds the GPU and CPU solvers in WSL from a local clone |
| `tools/runDamBreak.sh`, `tools/checkDamBreak.py` | Runs the dam break DualSPHysics ships and checks the build against the CPU solver and the experiment |
| `tools/runCase.sh` | Runs one case in WSL: GenCase, then the GPU solver |
| `tools/dsphCase.py` | Writes a 2D solitary-wave case from a pySurf `Bathymetry`: bed band, piston, end wall, water, gauge, output windows |
| `tools/dsphRidge.py`, `tools/dsphReef.py` | Write the 3D ridge and reef tanks: bed geometry as STL bands and a VTK surface for the normals |
| `tools/dsphOutput.py` | Reads the solver's binary Part files directly, or PartVTK's CSV frames |
| `tools/gateRuns.py` | Tunes, runs and measures the 2D Grilli and Surf Ranch cases |
| `tools/mostertRuns.py` | Runs Mostert and Deike's energy-loss case: a wave run and a still-water run, energies from the particles |
| `tools/ridgeRuns.py`, `tools/reefRuns.py` | Tune, run and measure the 3D ridge and reef |

Each Python tool's module docstring gives its command line, its criteria and what it writes. Runs land in `~/dsphRuns` inside WSL; results and figures in `pySurf/temp/phase1_5` (not tracked).

## Install

### WSL2 and the CUDA toolkit

Install WSL2 and Ubuntu 24.04 as in [the OpenFOAM guide](../openFoam/gettingStarted.md#wsl2-and-ubuntu). The Windows NVIDIA driver serves the GPU to WSL; install no Linux driver. Inside Ubuntu, install the compilers and CUDA 12.8 from NVIDIA's WSL repository:

```bash
sudo apt update
sudo apt install -y build-essential git
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8
/usr/local/cuda-12.8/bin/nvcc --version
```

`nvidia-smi` inside WSL should list the card. The build targets compute capability 7.5 (Turing); for another card, change the `-gencode` entry of `NVFLAGS` in `buildDualSphysics.sh`.

### The source

Clone DualSPHysics beside pySurf, not inside it:

```bash
cd /path/to/personalCodeProjects/externalReferences
git clone --depth 1 https://github.com/DualSPHysics/DualSPHysics.git dualsphysics
```

The runs here used commit ef3721a (2025-04-16), version 5.4. The repository ships GenCase, PartVTK and MeasureTool prebuilt for Linux but no solver.

### Build

From Git Bash on Windows:

```bash
MSYS_NO_PATHCONV=1 wsl.exe --exec bash /mnt/c/<path to pySurf>/dualSphysics/tools/buildDualSphysics.sh
```

The script clones the Windows clone again to `~/externalReferences/dualsphysics` on WSL's own disk, so the build tree has Unix line endings, and builds both solvers into its `bin/linux`. The source stays as published. Two settings on the make command line handle g++ 13: the C++ compiler is a wrapper that forces `<cstdint>` in (JBinaryData.cpp uses `SIZE_MAX` without it), and nvcc keeps plain g++ as its host compiler. Chrono and MoorDynPlus are off; WaveGen, which carries the solitary-wave piston, stays on. The GPU solver took 16 minutes and the CPU solver 14.

## Running from Windows

The tools run under the Windows Python that runs pySurf, and call WSL for each solver run:

```bash
cd pySurf/dualSphysics/tools
python gateRuns.py grilli 32
```

`gateRuns.py` holds the WSL distribution and user (`wslDistribution`, `wslUser`); the other tools take their run folders from it. Windows reads the runs through `\\wsl.localhost\Ubuntu-24.04\home\<user>\dsphRuns`. The measuring runs on the CPU in Windows while the card runs the solver in WSL; cap pySurf's Numba threads (`NUMBA_NUM_THREADS`) when several runs share the machine.

A 3D case at 32 ppd runs for four to eight hours. Start long runs from a process that outlives the terminal, and log to a file.

## Build check: the dam break

```bash
MSYS_NO_PATHCONV=1 wsl.exe --exec bash /mnt/c/<path to pySurf>/dualSphysics/tools/runDamBreak.sh
python checkDamBreak.py <gpu run> <cpu run> <EXP_X-DamTipPosition file>
```

`runDamBreak.sh` runs `CaseDambreakVal2D` as shipped on both solvers (144 s on the GPU, 500 s on the CPU); `checkDamBreak.py` compares the fronts from the case's own gauge, `Swl_z003`, the free surface along z = 0.03 m.

- **GPU against CPU: passes.** The fronts agree within 0.021 percent at all 133 gauge times before the end wall (criterion 1 percent).
- **Against Koshizuka and Oka (1996): fails.** The simulated front leads the measured one by up to 17 percent (criterion 10). The lead is a steady 0.05 to 0.06 s time offset from 0.19 s on, with the two fronts moving at 5.63 and 5.68 m/s; a gate that takes time to lift would produce it, which the record cannot confirm.

The front was first defined as the largest x of the fluid within two spacings of the floor. The dynamic boundary particles hold the leading particles 2 to 6 spacings above the floor, so that definition missed the tip; the check uses the case's gauge instead, with the tolerances unchanged.

## The 2D cases

```bash
python gateRuns.py <grilli|surfRanch> <ppd> [paddle height to start from] [--reuse] [--window start end] [--measure]
python mostertRuns.py <ppd> <artificial|sps> [paddle height to start from] [--measure]
```

| Case | Bed | Reference |
| --- | --- | --- |
| `grilli` | H0/h0 = 0.3 on a 1:35 plane beach | Grilli et al. (1997): Hb/h0 within 10 percent of 0.476, hb/h0 within 15 percent of 0.345 |
| `surfRanch` | H0/h0 = 0.6 over the Surf Ranch reef | Feddersen et al. (2024): impact x within 0.5 h0, H within 5 percent, A_O/H² within 15, A_J/H² within 20, theta_O within 5 degrees; the surface 0.1 sqrt(h0/g) before the boundary element model's impact within two spacings of that model's |
| Mostert | H0/h0 = 0.3 on a 3 degree plane beach | Mostert and Deike (2020): the share of the wave's energy lost in active breaking within 30 percent of 0.31 |

`dsphCase.py` writes each tank from pySurf's own `Bathymetry`, extended flat offshore to the piston, so both codes see the same bed. The particle lattice sits half a spacing off the bed line, as pySurf's does.

### The scheme

DualSPHysics's own options, fixed before the first gate run: Wendland kernel at h = 1.5 dp, symplectic time stepping, Fourtakas density diffusion at 0.1 (`DensityDT` 2), artificial viscosity at alpha 0.01, sound speed 20 sqrt(g h0), CFL 0.2, and mDBC walls (English et al. 2022) with the boundary velocity of plain DBC (`SlipMode` 1). The energy-loss case also runs with laminar plus sub-particle-scale viscosity (`ViscoTreatment` 2; Dalrymple and Rogers 2006).

### The start: a piston tuned at a gauge

GenCase cannot lay the exact solitary wave in the tank, so each case starts from a piston (`piston_solitary`, Goring's 1978 first-order theory) two half lengths and a stroke offshore of the place pySurf's run puts its starting crest. A free-surface gauge there reads the wave's height, less its own still-water reading at t = 0. The tune stage scales the paddle's target height until the gauge reads the case's height within 2 percent; this matches the input, not the result. Times count from the crest's passage of the gauge, as pySurf's count from its starting crest.

At H0/h0 = 0.3 the paddle's wave matches the exact solitary wave ahead of the crest within 0.03 H at 32 ppd. At 0.6 it does not: Goring's theory is first order, and the wave at the gauge is narrow, its volume within 10 h0 of the crest 4 to 10 percent short of the exact wave's, by an amount that changes with resolution. pySurf starts the Surf Ranch case from its boundary element model's Serre wave, so the two codes' starts differ in shape as well as in method.

### Measuring with pySurf's tracker

The tools read the solver's Part files directly (`dsphOutput.readPartArrays`; within 5×10⁻⁸ m of PartVTK's CSV), shift their times to pySurf's, and feed the fluid through `waveBreaking.BreakingTracker` set up as `BreakingWave.build` sets it, with `faceMeasure = 'contour'`. `BreakingWave.summary`'s measures and the gate criteria follow. The breaking point is the start of the face's last vertical run before the jet's touchdown, or of its longest vertical run where no touchdown is read. The energy budget subtracts a still-water run of the same tank, which settles as the wave run does, and the water the piston's stroke lifts, rho0 g S h0² / 2.

`--measure` reads a finished run again with its saved tuning and window.

## The 3D cases

```bash
python ridgeRuns.py <k> <ppd> [paddle height to start from] [--measure]
python reefRuns.py <ppd> [paddle height to start from] [--measure]
```

Building in 3D needs five things the 2D cases do not:

- **Normals from the geometry.** GenCase computes mDBC's normals from the particles only in 2D. The bed's come from the bed surface itself (`bed.vtk`, triangles); the walls' and the paddle's from their planes.
- **A drawn bed band.** Layer m of the band is the bed surface lowered (m + 0.6 + (|dz_b/dx| + |dz_b/dy|) / 2) spacings, written as an STL in triangles that rise at most 0.1 spacings. GenCase keeps a lattice point when a triangle crosses the point's cell, which reaches further than half a spacing on a tilted surface. A band lowered by a fixed (m + 0.6) spacings in larger triangles left 1.3 percent of the ridge's top row up to 0.27 spacings above the bed at 32 ppd.
- **Boxes drawn through their inner rows.** A box keeps the lattice points within half a spacing of its faces, so a wall drawn up to its plane takes the water's first row. The fill box reaches a quarter spacing past the walls' planes.
- **Global slip.** Version 5.4 has no symmetry plane, and its slip mode applies to every wall. The half-tank ridge's plane at y = 0 is a free-slip wall (`SlipMode` 3), so the bed and the other walls are free-slip too. With normals from the geometry, free slip holds the still water level within 0.003 h0.
- **Slices, not the tank.** Part files keep only the fluid the measures need: two rows each side of a section for the breaking point (each particle standing for s³ / 2s of the plane), and a layer for velocities. That is about 1 MB per frame at 32 ppd against 105 MB for the whole tank.

The paddle is tuned in a slab of the tank 0.5 h0 wide, then the full tank's own gauge is held to the same 2 percent.

### Guyenne and Grilli's ridge

z_b / h0 = (x - 5.225) sech²(k y) / 15 for x > 5.225, flat offshore, a wall at x = 19 h0, half the tank (0 ≤ y ≤ 4 h0), H0/h0 = 0.6. The breaking point is read on the centre line from the two rows nearest y = 0 against Guyenne and Grilli's (2006) table 1: time and crest height within 5 percent, crest x within 0.5 h0. For k = 0.25 the horizontal velocity at z = -0.2 h0 at t = 7.973 sqrt(h0/g) on their figure 16 grid: the largest speed within 10 percent of 0.651 sqrt(g h0), the largest spanwise velocity within 30 percent of 0.085 of the largest streamwise one. One row alone is too ragged: it read a sloping face as vertical at t = 5.94 sqrt(h0/g).

At 32 ppd the half tank holds 3.52 million particles and ran in 13810 s at k = 0.25 with the card shared.

### The Surf Ranch reef turned across the span

The 2D Surf Ranch profile with its toe at x = 30 + y tan 25° across a span of 4 h0, the shelf 0.371 h0 deep running to an end wall at x = 52 h0, between free-slip side walls: the geometry of `openFoam/cases/breakingWave3D`. No measurement exists for it, so it is reported only. Nine two-row slices across the span are each read against their own section of the reef; a line through the breaking times of the seven sections clear of the side walls gives the speed at which the break crosses the span, set against c / tan 25° with c their mean crest speed. The slip walls mirror the reef into a chevron, so that speed gives the scale only.

At 32 ppd: 4.56 million water particles, 27327 s with the card shared.

## Results

All at the scheme above; breaking points read with the rule above.

| Case | Criterion | DualSPHysics |
| --- | --- | --- |
| Grilli 1:35, 32 ppd | Hb within 10 percent, hb within 15 | Hb/h0 0.355 (-25.5 percent), hb/h0 0.240 (-30.5): fails |
| Grilli 1:35, 64 ppd | The same | 0.427 (-10.4), 0.314 (-9.0): fails on Hb |
| Surf Ranch overturn, 32 ppd | Five criteria | None pass; breaks at Hb/h0 0.586 against the boundary element model's 0.763 |
| Surf Ranch overturn, 64 ppd | The same | One passes (impact x); H -12 percent, A_O/H² -50, A_J/H² -44, theta_O +6.7 degrees |
| Surf Ranch overturn, 128 ppd | The same | Two pass (impact x, theta_O); H -12 percent, A_O/H² -35, A_J/H² -29 |
| Surface before impact, 32 to 128 ppd | Two spacings | 0.65 h0 at every resolution: the wave breaks late |
| Mostert, 32 ppd, artificial viscosity | Share within 30 percent of 0.31 | 0.291 (-6.0 percent): passes, at 2.4 times the fit's dissipation rate |
| Mostert, 64 ppd, artificial viscosity | The same | 0.347 (+11.9): passes, at 1.6 times the fit's rate |
| Mostert, laminar plus SPS | The same | Not measurable: the still tank's thin water over the beach throws particles from 1.8 s |
| Ridge k = 0.25, 32 ppd | Breaking point; velocity | t +0.6 percent, x +0.001 h0, crest -3.5 percent; speed -8.5 percent, spanwise ratio +27: passes |
| Ridge k = 0.50, 32 ppd | Breaking point | t +2.6 percent, x +0.134 h0, crest -5.3 percent: fails on crest height |
| Reef, 32 ppd | Reported only | The seven inner sections plunge 0.8 to 1.4 h0 past the shelf's edge (A_O/H² 0.12 to 0.17, theta_O 34 to 47 degrees); the break crosses the span at 9.9 m/s, 11 percent faster than c / tan 25° |

From 32 to 128 ppd the 2D overturn's shape converges and its height and timing stall: the paddle's narrow wave at H0/h0 = 0.6 likely sets the remaining error. The 2D runs use slip mode 1 and the 3D runs slip mode 3, so the two sets differ in their walls as well as their dimension.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `bash: C:/Program Files/Git/mnt/c/...: No such file or directory` | Git Bash rewrote the `/mnt/c` path; prefix the command with `MSYS_NO_PATHCONV=1` |
| `'SIZE_MAX' was not declared in this scope` in JBinaryData.cpp | g++ 13 no longer includes `<cstdint>` through other headers; build with `buildDualSphysics.sh`, whose compiler wrapper forces it in |
| nvcc errors on a redefinition in `<cstdint>` | The forced header reached nvcc's host compiler; nvcc must keep plain g++ (`-ccbin /usr/bin/g++`), as the build script sets |
| `/bin/bash^M: bad interpreter` | Windows line endings in a shell script; this folder's `.gitattributes` keeps them out of checkouts, and `sed -i 's/\r$//' <script>` repairs a copy |
| `error while loading shared libraries` at launch | The solver's libraries in `bin/linux` are not on the path; `runCase.sh` adds them to `LD_LIBRARY_PATH` |
| A 2D tank loses its water, or its still level falls, with slip mode 2 or 3 | Free slip fails in 2D with normals computed from the particles; keep slip mode 1 in 2D |
| Boundary particles above a sloping bed, or normals pointing down | The bed band was lowered too little for the slope or drawn in triangles that rise too far; see the 3D bed band above |
| The water's first row missing beside a wall | The wall's box reached the wall plane; draw it with its water-side face through its inner row |
| The tracker fires early on a 3D slice | One row of particles reads a ragged surface as a vertical face; read two rows each side of the section |
| A wall slice records a touchdown with A_O/H² under 0.02 | The top particle layer lifting off the one under it reads as a closed pocket, and fixes a false breaking point; leave the wall slices out |

## References

- Dalrymple, R.A. & Rogers, B.D. (2006) -- Numerical modeling of water waves with the SPH method, Coastal Eng. 53, 141-147.
- Domínguez, J.M. et al. (2022) -- DualSPHysics: from fluid dynamics to multiphysics problems, Comput. Part. Mech. 9, 867-895.
- English, A. et al. (2022) -- Modified dynamic boundary conditions (mDBC) for general-purpose smoothed particle hydrodynamics (SPH): application to tank sloshing, dam break and fish pass problems, Comput. Part. Mech. 9, 1-15.
- Feddersen, F., Hanson, K., Mostert, W. & Fincham, A. (2024) -- Modelling wind-induced changes to overturning wave shape, J. Fluid Mech. 1000, A48.
- Fourtakas, G., Domínguez, J.M., Vacondio, R. & Rogers, B.D. (2019) -- Local uniform stencil (LUST) boundary condition for arbitrary 3-D boundaries in parallel smoothed particle hydrodynamics (SPH) models, Computers & Fluids 190, 346-361.
- Goring, D.G. (1978) -- Tsunamis: the propagation of long waves onto a shelf, Report KH-R-38, California Institute of Technology.
- Grilli, S.T., Svendsen, I.A. & Subramanya, R. (1997) -- Breaking criterion and characteristics for solitary waves on slopes, J. Waterway, Port, Coastal, Ocean Eng. 123(3), 102-112.
- Guyenne, P. & Grilli, S.T. (2006) -- Numerical study of three-dimensional overturning waves in shallow water, J. Fluid Mech. 547, 361-388.
- Koshizuka, S. & Oka, Y. (1996) -- Moving-particle semi-implicit method for fragmentation of incompressible fluid, Nucl. Sci. Eng. 123, 421-434.
- Mostert, W. & Deike, L. (2020) -- Inertial energy dissipation in shallow-water breaking waves, J. Fluid Mech. 890, A12.
- DualSPHysics source: <https://github.com/DualSPHysics/DualSPHysics>
- CUDA on WSL: <https://docs.nvidia.com/cuda/wsl-user-guide/>
