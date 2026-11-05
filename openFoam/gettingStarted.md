
# OpenFOAM -- From Installation to a 3D Breaking Wave

This guide takes OpenFOAM from nothing installed to a 3D plunging breaker peeling along an oblique reef. It runs on Windows 11 through WSL2, uses the ESI build of OpenFOAM (v2606) for its built-in wave makers, and validates the solver on a 2D breaking case against Grilli et al. (1997) before trusting it in 3D.

## Contents

- [OpenFOAM -- From Installation to a 3D Breaking Wave](#openfoam----from-installation-to-a-3d-breaking-wave)
  - [Contents](#contents)
  - [The path](#the-path)
    - [Which OpenFOAM](#which-openfoam)
  - [Install](#install)
    - [WSL2 and Ubuntu](#wsl2-and-ubuntu)
    - [OpenFOAM](#openfoam)
    - [ParaView](#paraview)
    - [Where cases live](#where-cases-live)
  - [First case: damBreak](#first-case-dambreak)
    - [Anatomy of a case](#anatomy-of-a-case)
    - [What interFoam solves](#what-interfoam-solves)
  - [Waves: the solitary wave tutorial](#waves-the-solitary-wave-tutorial)
  - [Validation: a solitary wave breaking on a 1:35 slope (2D)](#validation-a-solitary-wave-breaking-on-a-135-slope-2d)
    - [2D setup](#2d-setup)
    - [2D run](#2d-run)
    - [Acceptance criteria](#acceptance-criteria)
    - [Grid convergence](#grid-convergence)
  - [The 3D breaking wave: a soliton peeling along an oblique reef](#the-3d-breaking-wave-a-soliton-peeling-along-an-oblique-reef)
    - [3D setup](#3d-setup)
    - [3D run](#3d-run)
    - [Changing it](#changing-it)
    - [Validation status](#validation-status)
  - [Troubleshooting](#troubleshooting)
  - [References](#references)

## The path

| Step | Case | Cells | What it teaches |
| --- | --- | --- | --- |
| 1 | `damBreak` tutorial | 2,268 (2D) | Case structure, the `interFoam` VOF solver, ParaView |
| 2 | `solitaryGrimshaw` tutorial | 411,600 (3D) | Wave generation and absorption boundary conditions, parallel runs |
| 3 | `cases/solitaryBreaking2D` | 69,254 (2D) | A breaking wave validated against a published benchmark |
| 4 | `cases/breakingWave3D` | 1,444,320 (3D) | A 3D plunging breaker on a turned reef |

Steps 3 and 4 live in this folder. `tools/makeBlockMeshDict.py` writes their meshes and `tools/analyzeBreaking2D.py` measures the 2D breaking point against the benchmark.

### Which OpenFOAM

Two projects publish OpenFOAM. The OpenFOAM Foundation (openfoam.org) releases numbered versions (14, July 2026). OpenCFD, now distributed by Keysight (openfoam.com), releases `vYYMM` versions (v2606, June 2026). Both solve the same VOF equations. This guide uses the ESI build because its `waveModels` library generates solitary, cnoidal, Stokes and irregular waves at a boundary with active absorption of reflected waves (Higuera et al. 2013), and its tutorial set includes solitary wave flumes. The case files here use ESI names (`interFoam`, `waveVelocity`, `turbulenceProperties`) and will not run on the Foundation build without changes.

## Install

OpenFOAM runs on Linux. On Windows 11 the supported route is WSL2, a full Linux kernel in a lightweight VM; OpenFOAM inside it runs at native speed on all cores.

### WSL2 and Ubuntu

In PowerShell opened as Administrator:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart Windows when asked. Ubuntu opens on the next login and asks for a Linux user name and password; these are separate from the Windows account.

Give WSL enough of the machine. Create `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=12GB
processors=8
swap=8GB
```

then run `wsl --shutdown` in PowerShell for it to take effect. On an 8-core, 16 GB machine this leaves Windows 4 GB and gives OpenFOAM one process per physical core. In a new Ubuntu terminal, `nproc` should print 8 and `free -g` about 11 GB total; without the file WSL takes half the memory and every logical CPU.

### OpenFOAM

Inside Ubuntu:

```bash
sudo apt update && sudo apt upgrade -y

# Add the OpenCFD package repository and its signing key
curl -s https://dl.openfoam.com/add-debian-repo.sh | sudo bash
sudo apt update

# Python for the analysis scripts
sudo apt install -y python3-numpy python3-matplotlib
```

Adding the repository installs nothing by itself. List the releases it offers; the newest is the highest number:

```bash
apt-cache search openfoam | grep default
```

Install that release; OpenFOAM itself takes about 600 MB, and apt pulls in some 230 packages with it (MPI, Scotch, compilers):

```bash
sudo apt install -y openfoam2606-default
ls /usr/lib/openfoam          # must list openfoam2606 before the next step
```

Load the environment in every new shell by adding one line to `~/.bashrc`:

```bash
echo 'source /usr/lib/openfoam/openfoam2606/etc/bashrc' >> ~/.bashrc
source ~/.bashrc
interFoam -help          # prints the solver's options if the install works
echo $FOAM_TUTORIALS     # where the tutorials are
```

### ParaView

The packages do not include ParaView. Install the Windows build from paraview.org: the `.msi` without "MPI" in its name (`ParaView-6.2.0-Windows-Python3.12-msvc2017-AMD64.msi` in October 2026). The MPI build runs ParaView's server in parallel and needs Microsoft MPI; the plain build reads OpenFOAM cases directly. ParaView opens an OpenFOAM case through an empty file ending in `.foam`; each case here writes `case.foam` when its run finishes, and for a tutorial `touch case.foam` in the case folder makes one.

Windows reaches the Linux files through a network share, listed under **Linux** at the bottom of File Explorer's sidebar. ParaView's File > Open dialog is its own and does not list network locations, so type the full path into its File name box: `\\wsl.localhost\Ubuntu-24.04\home\<user>\foam\damBreak\case.foam`. If ParaView refuses the network path, give the share a drive letter in PowerShell, `net use L: \\wsl.localhost\Ubuntu-24.04 /persistent:yes`, and open `L:\home\<user>\foam\...` instead.

### Where cases live

Keep cases in the Linux file system, not under `/mnt/c`: OpenFOAM writes many small files, and the Windows mount is many times slower for that. Copy this folder in once:

```bash
mkdir -p ~/foam
cp -r /mnt/c/Users/seanb/Desktop/Main/personalCodeProjects/pySurf/openFoam ~/foam/
chmod +x ~/foam/openFoam/cases/*/All*
```

## First case: damBreak

```bash
cp -r $FOAM_TUTORIALS/multiphase/interFoam/laminar/damBreak/damBreak ~/foam/
cd ~/foam/damBreak
./Allrun
touch case.foam
```

It runs in about a minute. Open `case.foam` in ParaView, colour by `alpha.water` and step through time.

### Anatomy of a case

| Folder | Holds | Files that matter here |
| --- | --- | --- |
| `0.orig` | Initial and boundary conditions, one file per field; `Allrun` copies it to `0` | `U`, `p_rgh`, `alpha.water` |
| `constant` | The mesh (`polyMesh`, written by `blockMesh`) and physical properties | `transportProperties`, `g`, `turbulenceProperties`, `waveProperties` |
| `system` | Run control and numerics | `controlDict` (time, output, function objects), `fvSchemes` (discretization), `fvSolution` (linear solvers, PIMPLE), `blockMeshDict`, `setFieldsDict`, `decomposeParDict` |

`Allrun` chains the utilities and writes a `log.<application>` for each. `Allclean` returns the case to its initial state.

### What interFoam solves

`interFoam` solves the incompressible Navier-Stokes equations for two immiscible fluids treated as one mixture. The volume fraction `alpha.water` (1 in water, 0 in air) is transported with MULES, a flux limiter that keeps it between 0 and 1, plus a compression flux (`cAlpha`) that keeps the interface a cell or two thick. Pressure and velocity are coupled by PIMPLE. The pressure solved for is `p_rgh = p - rho g.h`, which takes hydrostatic pressure out of the boundary conditions. The time step adapts to hold the Courant number below `maxCo`; the cases here use 0.3, a conservative value for a breaking wave.

## Waves: the solitary wave tutorial

```bash
cp -r $FOAM_TUTORIALS/multiphase/interFoam/laminar/waves/solitaryGrimshaw ~/foam/
cd ~/foam/solitaryGrimshaw
foamDictionary -entry numberOfSubdomains -set 8 system/decomposeParDict
foamDictionary -entry coeffs/n -set '(8 1 1)' system/decomposeParDict
./Allrun                  # about 10 minutes on 8 cores
reconstructPar            # join the 8 processor folders into one case
rm -rf processor*         # the output now exists twice; free about 4 GB
touch case.foam
```

The tutorial's `Allrun` stops after the parallel run, so its results sit in `processor0` to `processor7` and the case folder holds only `0`. `reconstructPar` joins them; without it, ParaView shows only the initial state unless the reader's **Case Type** is set to **Decomposed Case**.

This is a 7 m long, 0.55 m wide flume, 0.15 m deep, with a 0.05 m solitary wave on a flat bed. The parts that make waves:

- `constant/waveProperties` names a wave model for each wave patch. The `left` patch generates a `Grimshaw` solitary wave of the given `waveHeight`; the `right` patch absorbs with `shallowWaterAbsorption`.
- On those patches `U` is `waveVelocity` and `alpha.water` is `waveAlpha`: the boundary imposes the wave's velocity profile and surface elevation and reads the still water depth from `alpha.water` on the patch.
- `activeAbsorption yes` lets the generating boundary also absorb what reflects back to it.
- `runParallel` runs on the subdomains `decomposePar` cut; `reconstructPar` joins them for viewing.

The other wave tutorials in that folder (`solitary`, `solitaryMcCowan`, `cnoidal`, `stokesV`, `irregularMultiDirection`, paddle wave makers) use the same structure with a different `waveModel`.

## Validation: a solitary wave breaking on a 1:35 slope (2D)

A breaking-wave simulation is credible only after the solver reproduces a known breaking point. This case reruns Grilli, Svendsen and Subramanya (1997): a solitary wave of H0/h0 = 0.3 shoaling up a 1:35 plane slope. They computed it with fully nonlinear potential flow and took breaking where the front face first stands vertical, giving Hb/h0 = 0.476 and Hb/hb = 1.380, so hb/h0 = 0.345. The same benchmark validates `pysurf.OverturningWave`, which breaks at Hb/h0 = 0.504 and hb/h0 = 0.359.

### 2D setup

| Item | Value |
| --- | --- |
| Depth h0 | 1 m |
| Wave | Grimshaw solitary wave, H = 0.3 m, generated at the inlet (x = 0) with active absorption |
| Bed | Flat to x = 10 m, 1:35 to a depth of 0.27 m at x = 35.55 m, flat to the outlet at x = 38.55 m |
| Outlet | Shallow water absorption |
| Bed condition | Free slip, as the inviscid reference assumes |
| Mesh | 0.02 m cells for x > 27 m (15 per wave height), 0.05 m offshore; one cell across, front and back `empty` |
| Flow model | Laminar; the overturn up to the jet's impact is close to inviscid |

The mesh has two block layers. The lower layer runs from the bed to z = 0.8 m with a fixed 12 cells, so it thins as the bed rises; the upper layer runs from 0.8 to 1.8 m with uniform 0.02 m cells everywhere, so the free surface is resolved at the same size from deep water to the breaking point.

### 2D run

```bash
cd ~/foam/openFoam/cases/solitaryBreaking2D
./Allrun
python3 ../../tools/analyzeBreaking2D.py .
```

`Allrun` runs to t = 9 s writing fields every second while the wave shoals. It then continues to 16 s, writing every 0.05 s through breaking, which should come near t = 12 s. Each step's progress is in `log.interFoam.shoaling` and `log.interFoam.breaking`. Time per step is the difference between successive `ExecutionTime` lines; multiply by the remaining steps to estimate the finish.

`analyzeBreaking2D.py` traces the free surface as the `alpha.water = 0.5` contour in every written frame. It takes the breaking point as the first instant the front face stands vertical, interpolated between frames, and prints the comparison:

```text
quantity  OpenFOAM  Grilli 1997    error  criterion  verdict  pySurf BEM
Hb/h0        .....        0.476     ...%       10 %     ....       0.504
hb/h0        .....        0.345     ...%       15 %     ....       0.359
```

It also writes `breaking2D.json` and `breaking2D.png`, the surface in the frames around breaking. On a synthetic case with a known breaking instant, it recovers the instant to 0.001 s, the crest position to within one cell and hb to 0.1 percent.

### Acceptance criteria

The criteria are fixed before the run, and are the ones pySurf applies to its own breaking models: Hb/h0 within 10 percent and hb/h0 within 15 percent of Grilli's. Grilli started from the exact potential flow solitary wave (Tanaka 1986); this case generates Grimshaw's third-order wave at the inlet. A few percent of difference in the breaking height therefore belongs to the starting wave rather than to the solver. The comparison with pySurf's BEM is a cross-check between two models, not a validation.

### Grid convergence

Repeat the case at 0.014 m cells. The two resolutions should agree on Hb and hb to within 3 percent before either is reported.

```bash
cd ~/foam/openFoam/cases
cp -r solitaryBreaking2D solitaryBreaking2D-fine && cd solitaryBreaking2D-fine
./Allclean
python3 ../../tools/makeBlockMeshDict.py solitaryBreaking2D --cell 0.014 --output system/blockMeshDict
./Allrun && python3 ../../tools/analyzeBreaking2D.py .
```

The analysis also prints the water the wave maker added against the volume a first-order solitary wave of the same height carries (2H/k per unit width). This is a sanity check on the wave maker, not a validation.

## The 3D breaking wave: a soliton peeling along an oblique reef

The wave is the Surf Ranch soliton of Feddersen et al. (2024): a0/h0 = 0.6 on a 0.0693 reef face rising to a flat 0.371 h0 deep. Their 2D two-phase DNS of it is the reference for overturn shape at jet impact that `pysurf.OverturningWave` is validated against. Here the toe of the face and the edge of the flat turn 25 degrees from the incoming crest across a 4 m span. The wave reaches the reef first at y = 0, breaks there first, and the break travels across the span. That is a peeling plunging breaker, the 3D structure a surfer rides.

### 3D setup

| Item | Value |
| --- | --- |
| Depth h0 | 1 m |
| Wave | Grimshaw solitary wave, H = 0.6 m, generated at x = 20 m with its crest square to the flume |
| Reef | Toe at x = 30 m + y tan 25 deg; 0.0693 face to the flat (0.371 m deep) at x = 39.08 m + y tan 25 deg; flat to the outlet at x = 52 m |
| Span | 4 m, free-slip side walls |
| Mesh | 0.05 m cells over the reef (12 per wave height), 0.1 m offshore; 1,444,320 cells |
| Output | Fields every 0.5 s (binary); the free surface every 0.05 s from 5 s as VTK, in `postProcessing/freeSurface` |

Grimshaw's third-order series loses accuracy as a solitary wave nears its breaking limit, and McCowan's model would be the closer approximation at this height. In v2606, though, McCowan's Newton-Raphson solve for the wave's coefficients diverges in this flume from H/h = 0.4 up, so the case uses Grimshaw's. The blocks over the reef are parallelograms in plan, so the bed planes are exact. Their faces sit up to 25 degrees from orthogonal, which `corrected` Laplacian schemes and one non-orthogonal corrector handle.

### 3D run

```bash
cd ~/foam/openFoam/cases/breakingWave3D
./Allrun
```

It runs on 8 cores to t = 11 s; breaking should start near 7 s. Expect a run measured in hours: check the time per step in `log.interFoam` after the first few minutes and extrapolate. The output takes about 4 GB.

To view it, open `case.foam` in ParaView for the fields. For the breaker itself, open the `postProcessing/freeSurface/<time>/alpha05` series, colour by `U` and play it from 6 to 10 s.

### Changing it

```bash
# Finer: 0.04 m cells, 2.66 million cells
python3 ../../tools/makeBlockMeshDict.py breakingWave3D --cell 0.04

# A different reef angle or span
python3 ../../tools/makeBlockMeshDict.py breakingWave3D --angle 15 --width 6
```

`makeBlockMeshDict.py` prints the cell count before anything runs. The wave height and model are in `constant/waveProperties`, the run length and output in `system/controlDict`, and the core count in `system/decomposeParDict`.

### Validation status

The 3D case is unvalidated: no published measurement or simulation of a solitary wave peeling along an oblique reef was found to compare with. Three checks bound it:

1. **2D section against the DNS (cross-check).** Run with `--angle 0 --width 0.5`; the flow is then 2D, and the jet's impact position and the highest elevation at impact can be compared with Feddersen et al. (2024): x/h0 = 42.05 and H/h0 = 0.64. This is a cross-check, not a validation, because the starting waves differ. The DNS started from a Green-Naghdi solitary wave at x = 15 h0; this case generates a Grimshaw wave at the inlet.
2. **Water volume.** `postProcessing/waterVolume` records the water volume; it should rise by the wave's volume as the wave enters and then hold steady.
3. **Resolution.** Rerun at 0.04 m cells; the position where breaking starts along the span should move by less than a cell size.

The 2D validation above shows the solver breaks a solitary wave in the right place. It does not show that the 3D splash-up, the spanwise structure of the broken wave, or the air entrained in the tube are right; at 12 cells per wave height they are not resolved.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `./Allrun: Permission denied` | The script lost its execute bit in the copy: `chmod +x Allrun Allclean` |
| `/bin/sh^M: bad interpreter` | Windows line endings: `sed -i 's/\r$//' Allrun Allclean`. The `.gitattributes` in this folder keeps them out of git checkouts |
| `bashrc: No such file or directory`, or `interFoam` not found | The OpenFOAM package is not installed; `ls /usr/lib/openfoam` lists nothing. Run `sudo apt install -y openfoam2606-default` |
| `Unknown patchField type waveVelocity` | The Foundation build is loaded; source the ESI `etc/bashrc` |
| `Newton-Raphson iterations diverging` or a floating point exception in `McCowan::newtonRapsonF1` | The McCowan wave model fails for large waves in v2606; use `Grimshaw` in `constant/waveProperties` |
| `Maximum number of iterations exceeded` or the time step collapsing to 1e-8 s | The jet is too thin for the mesh or the Courant number too high; lower `maxCo` and `maxAlphaCo` to 0.2 in `system/controlDict` and restart from the last write |
| A run killed with no error | Out of memory; raise `memory` in `.wslconfig`, or use a coarser `--cell` |
| `decomposePar` refuses because processor folders exist | Run `./Allclean` first, or `decomposePar -force` |
| ParaView does not see the case | Open it through `\\wsl.localhost\Ubuntu-24.04\...`, and run `reconstructPar` first, or tick "Decomposed Case" in the reader |
| Breaking analysis reports the face never turned vertical | The run stopped before breaking or writes too rarely; check `endTime` and `writeInterval` against the breaking time printed in the frame list |

## References

- Feddersen, F., Hanson, K., Mostert, W. & Fincham, A. (2024) -- Modelling wind-induced changes to overturning wave shape, J. Fluid Mech. 1000, A48.
- Grilli, S.T., Svendsen, I.A. & Subramanya, R. (1997) -- Breaking criterion and characteristics for solitary waves on slopes, J. Waterway, Port, Coastal, Ocean Eng. 123(3), 102-112.
- Higuera, P., Lara, J.L. & Losada, I.J. (2013) -- Realistic wave generation and active wave absorption for Navier-Stokes models, Coastal Engineering 71, 102-118.
- Senthilkumar, A. (2016) -- arXiv:1601.06822; Table 8 tabulates Grilli et al.'s breaking characteristics.
- Tanaka, M. (1986) -- The stability of solitary waves, Phys. Fluids 29, 650-655.
- OpenFOAM v2606 release notes: <https://www.openfoam.com/news/main-news/openfoam-v2606>
- OpenFOAM precompiled packages (Ubuntu): <https://gitlab.com/openfoam/core/openfoam/-/wikis/precompiled/debian>
- ESI wave tutorials: `$FOAM_TUTORIALS/multiphase/interFoam/laminar/waves`
