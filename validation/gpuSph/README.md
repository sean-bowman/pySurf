# GPU SPH Validation -- The δ-SPH Breaking-Wave Model

The runs that validate pySurf's particle model of a breaking wave: `BreakingWave` in 2D and `BreakingWave3D` in 3D, on the 'deltaSph' scheme and the GPU backend. Each script runs one case, prints its verdicts against criteria set before the runs, and writes its record to `results/`. The runs take from minutes to 32 GPU hours each, so they live here rather than in the test suite.

## Contents

- [Summary](#summary)
- [Running the cases](#running-the-cases)
- [The cases](#the-cases)
- [Validation status](#validation-status)
- [Files](#files)
- [References](#references)

## Summary

The model's claim is the breaking point: where and when a solitary wave breaks, in 2D from 64 particles per depth and in 3D at 32 particles per depth with the crest's water split once. In 2D the overturn's shape at the jet's landing also passes, at 128 particles per depth. After breaking, the model dissipates too little energy, and the shortfall grows as the spacing shrinks: bores, runup and splashes are unvalidated and probably high.

| Case | Script | Reference | Criteria | Recorded | Verdict |
| --- | --- | --- | --- | --- | --- |
| Breaking point, $H/h_0 = 0.3$ on 1:35 | `grilliBreaking.py` | Grilli et al. (1997), potential flow | $H_b$ within 10%, $h_b$ within 15% | 64 ppd: +4.3%, -8.4%; 128: +1.6%, -2.8%; 32: +2.1%, -19.5% | Pass at 64 and 128 |
| Overturn at the jet's landing, Surf Ranch soliton | `surfRanchOverturn.py` | Feddersen et al. (2024), DNS | Impact x 0.5 $h_0$; $H_I$ 5%, $A_O/H_I^2$ 15%, $A_J/H_I^2$ 20%, $\theta_O$ 5° | 128 ppd: all five; 64: four ($\theta_O$ -5.2°) | Pass at 128 |
| Surface 0.1 $\sqrt{h_0/g}$ before the jet lands | `surfRanchOverturn.py` | pySurf's boundary element model | Within 2 spacings | 11, 15, 14 spacings at 32, 64, 128 ppd | Fail |
| Energy lost in active breaking, $H_0/h_0 = 0.3$ on 3° | `mostertEnergy.py` | Mostert and Deike (2020), DNS | Share within 30% of 0.31 | 0.20, 0.17 at 32, 64 ppd (-35%, -47%) | Fail |
| The same with DualSPHysics's settings | `mostertEnergy.py --dualSphysicsSettings` | Mostert and Deike (2020) | The same | 0.607, 0.545 at 32, 64 ppd (+96%, +76%) | Fail, a diagnostic |
| Where that energy goes | `mostertEnergy.py --energyBudget` | The budget's own closure | Walls or time stepping over a tenth of the loss reads as part of the deficit | Diffusion 0.60, 0.74; viscosity 0.17, 0.26; walls 0.38, 0.14; correction -0.17 at 32, 64 ppd; closure under 1e-5 J/m | Walls part of the deficit, a diagnostic |
| Bore over a reef flat, $H_0/h_0 = 0.4$ | `reefFlat.py` | Quiroga and Cheung (2013), flume | WG12 to WG14 within 10% at 128 ppd | WG12 +24.5% to +29.3% over three runs, clear of their scatter | Fail |
| Exact solitary wave over 30 $h_0$ of flat bed | `solitaryWave.py` | Clamond and Dutykh (2013), exact | Height lost under 2% | 0.42 to 0.50% at 32 ppd, 0.76% at 64 | Pass |
| Still water after 20 $\sqrt{d/g}$ | `stillTank.py` | At rest | rms speed under $10^{-3}\sqrt{gd}$, fastest under $10^{-2}$ | 60 ppd: $7.1 \times 10^{-4}$, $3.2 \times 10^{-3}$; 30: rms $1.1 \times 10^{-3}$ | Pass at 60 |
| The same, the fluid sorted in cell order every 0.1 $\sqrt{d/g}$ | `stillTank.py --sorted` | At rest | The same | 60 ppd: $6.1 \times 10^{-4}$, $2.7 \times 10^{-3}$ | Pass |
| Breaking point, the fluid sorted every 10 breaking checks | `grilliBreaking.py --sortInterval 10` | The unsorted run; Grilli et al. (1997) | Time, $H_b$ and $h_b$ within 1% of the unsorted run; Grilli's criteria | 64 ppd: within $2 \times 10^{-6}$ of the unsorted run; +4.3%, -8.4% | Pass |
| 3D breaking point, sloping ridge, k = 0.10 to 0.50 | `ridge.py` | Guyenne and Grilli (2006), 3D potential flow | Time and crest height 5%, crest position 0.5 $h_0$ | 32 ppd: four of five k pass; k = 0.25 crest +0.53 $h_0$ | Fail at k = 0.25 |
| The same at k = 0.25 with the crest zone | `ridge.py --zone` | Guyenne and Grilli (2006) | The same | Time +2.9%, crest +0.39 $h_0$, height -0.2% | Pass |
| Flow under the crest, k = 0.25 | `ridge.py` | Guyenne and Grilli (2006), figure 16 | Largest speed 10%, spanwise ratio 30% | -0.3%, +0.1%; with the zone -0.6%, +1.0% | Pass |
| Crest crossing an oblique reef at 30° | `obliqueReef.py` | Closed form, $c_0 / \tan 30°$ | Read against the closed form | Within 1.7% at every distance, 16 ppd, 16 $h_0$ across | Pass |
| The Surf Ranch reef extruded across y | `extrudedReef.py` | pySurf's 2D model | Half the gates' tolerances | All eight readings, split, 32 ppd | Consistent |
| The Surf Ranch reef turned 25° | `shoalReef.py` | DualSPHysics, the same tank | None: a cross-check | Breaks within 0.04 s, 8% to 11% higher; lands 0.03 to 0.13 s later | Reported |
| Tube at touchdown, turned Surf Ranch reef, 3D | `cavityArea.py` | Blenkinsopp and Chaplin (2008), flume, a cross-check | $A_c/H_b^2$ within their points at $h_c/H_o \le 0.9$ (0.154 to 0.408) | 0.233 to 0.258 over the seven inner sections at 32 ppd with the zone | Consistent, a cross-check |

## Running the cases

The scripts need an NVIDIA GPU and pySurf's GPU dependency, NVIDIA Warp (`pip install -e .[gpu]` from the repository root). Run them from this folder:

```bash
python grilliBreaking.py 64 [--sortInterval 10]
python surfRanchOverturn.py 128
python mostertEnergy.py 64 [--dualSphysicsSettings] [--energyBudget]
python reefFlat.py 128 [--cflNumber 0.225] [--shift 0.5] [--energyBudget]
python cavityArea.py [<run folder>]
python solitaryWave.py 32
python stillTank.py 60 [--sorted]
python ridge.py 0.25 32 [--zone]
python obliqueReef.py 16 --span 16 --skip -3
python extrudedReef.py 2d 32 --split
python extrudedReef.py 3d 32 --split
python extrudedReef.py compare 32 --split
python shoalReef.py 32 --zone
```

Each script's docstring states its case, reference, criteria and recorded result. Records go to `results/<case>_<resolution>.json`; setting `PYSURF_VALIDATION_RESULTS` sends them elsewhere, for trial runs at low resolution. `surfRanchOverturn.py` runs pySurf's boundary element model once for its reference surface (about 20 minutes) and caches it in `results/surfRanchBem.pkl`.

Cost on an RTX 2070 Super (8 GB), one run at a time:

| Case | Resolution | GPU time |
| --- | --- | --- |
| Grilli breaking point | 64 ppd | 11 minutes |
| Reef flat | 64 / 128 ppd | 17 minutes / 2 hours |
| Mostert and Deike | 64 ppd | 23 minutes, the still-water reference included |
| Ridge | 32 ppd | 5 to 7 hours; 28 with the crest zone |
| Oblique reef, 16 $h_0$ across | 16 ppd | 3.8 hours |
| Extruded reef, 3D | 32 ppd | 4.7 hours; 8.5 split, 6.8 split and merged |
| Shoal reef | 32 ppd with the zone | 32 hours |

Runs of an hour or more are best started from a process that outlives the terminal, with the output sent to a log.

## The cases

The 2D cases use `BreakingWave` with its defaults for 'deltaSph': Wendland C2 at three spacings, hydrostatic δ-SPH density diffusion, the kernel gradient correction in the continuity equation, wall pressure from the water, a limited artificial viscosity at α = 0.01, a density floor of 0.9 ρ₀, and the water laid in columns fitted to the bed. The 3D cases use `BreakingWave3D` with the same scheme. The breaking point is the start of the front face's last vertical run before the jet lands, read on the smoothed surface; a run with no landing takes the longest vertical run.

- **Breaking point.** `grilliBreaking.py` is gate 1's breaking-point case and `ridge.py` gate 2's. In 3D the scheme's 2D lag at 32 particles per depth carries over: on the ridge's centre line, run in 2D, the wave breaks 0.37 $\sqrt{h_0/g}$ after Guyenne and Grilli's 2D result, against 0.10 at 64. The crest zone (`--zone`) closes most of it.
- **Overturn.** `surfRanchOverturn.py` reads five measures at the jet's landing. The pass at 128 particles per depth is not converged: the overturn area runs 0.495, 0.308 and 0.341 $H_I^2$ at 32, 64 and 128, against 0.352.
- **After breaking.** `mostertEnergy.py` and `reefFlat.py` both fail, in the same direction: the model keeps too much energy once the wave breaks. The model has no turbulence. Its artificial viscosity and density diffusion stand in for the loss, and both shrink with the spacing. The energy budget (`--energyBudget`, `pysurf.sphEnergy`) splits the loss by term: at 64 particles per depth the density diffusion takes three quarters of it and the limited viscosity a quarter, the continuity correction gives a sixth back, and the walls take 14% through the extrapolated wall pressure's work and the clamp at the bed, neither of which a free-slip wall would do. The share lost moves 6% under a rounding-level change to the device kernels (0.165 to 0.156), and the walls' share with it (14% to 8.5%). `reefFlat.py`'s scatter runs (`--cflNumber 0.225`, `--shift 0.5`) show the WG12 failure and its growth from 64 to 128 particles per depth are both larger than the run-to-run spread. The reef edge's position, 54.8 m, is scaled from the thesis's figure; the thesis bounds it to 54.4 to 54.9 m, which moves WG12 by 0.011 $H_0$ at most.
- **3D construction.** `obliqueReef.py` checks the bed's rotation and the crest's kinematics against a closed form, and `extrudedReef.py` checks the 3D form against the 2D one. In the oblique tank the break crosses 9 to 11% slower than the closed form: the free-slip walls mirror the bed into a chevron, a limit of a walled tank.
- **Cross-check.** `shoalReef.py` sets the 3D model beside DualSPHysics on a tank with no measurement. The two break together, but pySurf's jet lands after DualSPHysics's, where in 2D it lands 0.55 s before; the runs do not say which code departs from 2D in that tank. `cavityArea.py` reads the tube at touchdown from a 3D run that saved its whole fluid, against Blenkinsopp and Chaplin's (2008) flume cavities: the shelf's relative depth lies below their range, so the comparison takes their points nearest it, and their waves are regular where this one is solitary. DualSPHysics's runs of every gate case, with the settings they used and what they read, are in [reference/dualSphysics/](reference/dualSphysics/README.md).
- **Sorted particle data.** `--sorted` and `--sortInterval` reorder the fluid's state in the GPU hash grid's cell order (`WcsphSolver.sortParticles`). Sorted, the still tank passes and the breaking point at 64 particles per depth matches the unsorted run to $2 \times 10^{-6}$; after breaking the two part as rounding changes make them. Timed on the turned Surf Ranch reef at 16 particles per depth with the crest zone, from a checkpoint at $t' = 17$ to 18.5 (429 steps), sorting every 10 and 40 breaking checks took 0.3% and 4.1% less time than no sorting. The rule set before the runs made sorting the default only at 15% or more, so `sortInterval` is off by default.

## Validation status

- **Validated:**
  - the breaking point in 2D at 64 and 128 particles per depth;
  - the overturn's shape at the jet's landing in 2D at 128 particles per depth;
  - the breaking point and the flow under it in 3D at 32 particles per depth with the crest zone;
  - the solitary wave over a flat bed, and still water at 60 particles per depth, sorted or not.
- **Cross-checked, not validated:** the 3D model against 2D on an extruded reef, and against DualSPHysics on a turned reef.
- **Failed, standing as limits of the scheme:**
  - the surface before the jet lands;
  - the energy lost in active breaking;
  - the bore over a reef flat;
  - the overturn's shape in 3D, which no measurement covers and which the crest zone does not carry in 2D.

## Files

| File | Contents |
| --- | --- |
| `common.py` | Paths, the time unit, and the results writer the scripts share |
| `grilliBreaking.py`, `surfRanchOverturn.py`, `mostertEnergy.py`, `solitaryWave.py`, `stillTank.py` | Gate 1, 2D |
| `reefFlat.py` | The Oregon State reef flat, 2D |
| `ridge.py` | Gate 2, 3D |
| `obliqueReef.py`, `extrudedReef.py`, `shoalReef.py`, `cavityArea.py` | The 3D model over reefs |
| `reference/quiroga2012Fig25Bf0.json` | The flume's smooth-bed gauge peaks, digitized from Quiroga (2012) figure 25, with the gauge positions and the reef edge's bounds |
| `reference/dualSphysics/` | DualSPHysics's reference runs of the gate cases, the ridge, the turned reef and the reef flat: each run's record, the scheme's settings, results and example outputs (cross-checks; [its README](reference/dualSphysics/README.md)) |
| `results/` | Each run's record |

## References

- Clamond, D. & Dutykh, D. (2013). Fast accurate computation of the fully nonlinear solitary surface gravity waves. *Computers & Fluids* 84, 35-38.
- Domínguez, J.M. et al. (2022). DualSPHysics: from fluid dynamics to multiphysics problems. *Computational Particle Mechanics* 9, 867-895.
- Feddersen, F., Hanson, K., Mostert, W. & Fincham, A. (2024). Modelling wind-induced changes to overturning wave shape. *J. Fluid Mech.* 1000, A48.
- Grilli, S.T., Svendsen, I.A. & Subramanya, R. (1997). Breaking criterion and characteristics for solitary waves on slopes. *J. Waterway, Port, Coastal, Ocean Eng.* 123(3), 102-112.
- Guyenne, P. & Grilli, S.T. (2006). Numerical study of three-dimensional overturning waves in shallow water. *J. Fluid Mech.* 547, 361-388.
- Mostert, W. & Deike, L. (2020). Inertial energy dissipation in shallow-water breaking waves. *J. Fluid Mech.* 890, A12.
- Quiroga, P.D. & Cheung, K.F. (2013). Laboratory study of solitary-wave transformation over bed-form roughness on fringing reefs. *Coastal Engineering* 80, 35-48.
- Quiroga, P.D. (2012). *Effects of bed-form roughness on long-wave transformation over fringing reefs*. MS thesis, University of Hawaii at Manoa.
