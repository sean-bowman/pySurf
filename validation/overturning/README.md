# Boundary Element Model Runs

Runs of `OverturningWave`, pySurf's boundary element model of an overturning wave, that take longer than the test suite allows. The model's established validation cases (Grilli et al. 1997, the DNS of Feddersen et al. 2024, the fits of Pick and Feddersen 2026, Guyenne and Grilli 2006) run in the suite's slow tests (`python -m pytest -m slow`). The scripts here read how the model's numerics bear on those results, and test it against laboratory data the suite does not run.

## Smoothing energy

`smoothingEnergy.py` runs the five plane-slope cases of `tests/testValidationOverturning.py` and the Surf Ranch soliton with the five-point smoothing every 3, 6 and 10 steps once the face overhangs, records each pass's energy and volume before and after it (`OverturningWave(trackSmoothing = True)`, `smoothingBudget`), and reads the jet area at impact against Pick and Feddersen's fit by a rule set before the runs.

```bash
python smoothingEnergy.py run --workers 9     # 18 runs, 17 to 40 minutes each on one thread
python smoothingEnergy.py report              # the reading, from results/
```

| Reading | Result |
| --- | --- |
| Jet area, the three cases outside three RMSE | Falls 3 to 5 percent from interval 3 to 10; two of the three move a third of their gap to the fit, each just past that line, so the smoothing is part of the failure by the rule. At interval 10 they still stand 9 to 15 percent over the fit. |
| Other overturn measures | A_O, W/L, theta_O and Hb/hb pass at every interval |
| Surf Ranch soliton against the DNS | A_J falls from 0.195 to 0.187, further below the DNS's 0.22, inside its 20 percent; the other readings barely move |
| Energy the smoothing removes over the overturn | -0.05 to +0.07 percent of the wave's energy, under 0.0013 percent a pass (Longuet-Higgins and Cokelet 1976: 0.06 percent) |
| Energy everything else changes over the overturn | Gains 0.11 to 0.35 percent, more the less often the surface is smoothed |

The default interval stays at 3.

## Breaking on a steep reef face

`reefFaceBreaking.py` runs solitary waves onto the 1:12 face of the fringing reef Quiroga and Cheung (2013) built in Oregon State's Large Wave Flume, against their control tests' breaking height and depth, read from video with Grilli's definition (`reference/quiroga2013Fig11Fig12.json`, their figures 11 and 12 digitized). Sixteen cases over flats 0.1, 0.2 and 0.3 m deep, each from the exact solitary wave of the flume's gauge height; the dry flat is not run, since it needs a moving shoreline.

```bash
python reefFaceBreaking.py run --workers 9     # 16 runs, 20 to 80 minutes each on one thread
python reefFaceBreaking.py report
```

| Reading | Criterion | Result |
| --- | --- | --- |
| Breaking height H_b/h0, 16 cases | 10 percent | +2.5 to +23.7 percent, 11 of 16 beyond: fails |
| Breaking depth h_b/h0 under the front, the 9 cases that break on the face | 15 percent | +1.4 to +28.8 percent, deeper, 5 of 9 beyond: fails |
| Growth from the gauge to breaking, H_b/H0 | Reported | Flume 0.96 to 1.14 (mostly 1.01 to 1.08); model 1.13 to 1.20 |

The error has one sign: the model's wave breaks higher and further offshore, and it is already the larger wave before it breaks. The flume's wave came from a piston, with trailing waves and a shape of its own, where the model starts from the exact solitary wave of the same gauge height; that difference is the first suspect and is not yet tested.

Each script's docstring holds its criteria and recorded result; each run's record is in `results/`.

## References

- Longuet-Higgins, M.S. & Cokelet, E.D. (1976). The deformation of steep surface waves on water I. A numerical method of computation. *Proc. R. Soc. Lond. A* 350, 1-26.
- Pick, K. & Feddersen, F. (2026). Scaling the shape of shoaling and overturning solitary waves. *J. Fluid Mech.* 1040, A8.
- Feddersen, F., Hanson, K., Mostert, W. & Fincham, A. (2024). Modelling wind-induced changes to overturning wave shape. *J. Fluid Mech.* 1000, A48.
