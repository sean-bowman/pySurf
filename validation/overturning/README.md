# Boundary Element Model Diagnostics

Runs of `OverturningWave`, pySurf's boundary element model of an overturning wave, that take longer than the test suite allows. The model's validation cases (Grilli et al. 1997, the DNS of Feddersen et al. 2024, the fits of Pick and Feddersen 2026, Guyenne and Grilli 2006) run in the suite's slow tests (`python -m pytest -m slow`); the scripts here read how the model's numerics bear on those results.

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

The default interval stays at 3. Each script's docstring holds its criteria and recorded result; each run's record, with every smoothing pass, is in `results/`.

## References

- Longuet-Higgins, M.S. & Cokelet, E.D. (1976). The deformation of steep surface waves on water I. A numerical method of computation. *Proc. R. Soc. Lond. A* 350, 1-26.
- Pick, K. & Feddersen, F. (2026). Scaling the shape of shoaling and overturning solitary waves. *J. Fluid Mech.* 1040, A8.
- Feddersen, F., Hanson, K., Mostert, W. & Fincham, A. (2024). Modelling wind-induced changes to overturning wave shape. *J. Fluid Mech.* 1000, A48.
