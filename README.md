# Computational Engineering Portfolio

**Generation and analysis of physical models through computer algorithms.**

This repository demonstrates computational engineering: using software to define, generate, and analyze physical objects and their behavior. Rather than drawing shapes manually in CAD software, geometry is defined programmatically through mathematical functions and parameters, enabling precise control, reproducibility, and design exploration.

The package lives under `src/pysurf/`. It follows the layout of the NOVA nozzle suite: a facade class for each domain, and single-purpose shared modules (geometry, figures, units, errors, validation, export) that every domain calls rather than reimplementing.

---

## Domains

### Surfboard

Parametric surfboard geometry built from the three views a shaper works from: the planform (top-down outline), the rocker (side profile of deck and bottom), and the cross section with its rails. The planform and both rockers are C1-continuous cubic Bezier sections with controllable departure, arrival, and midpoint angles, overlaid on reference photographs of a thruster for comparison. Cross-section generation is in development.

```python
from pysurf import Surfboard

board = Surfboard(totalLength = 6.0, maxWidth = 20.0, midpointThickness = 2.5,
                  noseShape = 'point', tailShape = 'round', rockerProfile = 'aggressive')
board.plotsEnabled = True
board.generateGeometry()
```

Length is in feet; width and thickness are in inches.

### Breaking Wave

Breaking surf waves in 2D over a bed of any shape: a wave shoaling up a beach until it breaks onto the shore, and swell breaking in open water where a reef rises steeply under it. Three models cover it, each answering a different question.

- `OverturningWave` solves fully nonlinear potential flow with boundary elements. Its free surface can fold, so it gives the shape of the barrel: the lip curling over and the tube under it, up to the moment the jet lands. `SplashUp` hands that flow to particles and carries it on through the impact and the splash-up; `ReefOverturn` feeds it waves from an SGN train breaking on a reef.
- `SurfaceBreakingWave` and `ReefBreak` solve the Serre-Green-Naghdi equations for the free surface. The surface cannot fold, so a breaking wave becomes a bore, but where and how high waves break comes out in about a minute, for a single wave on a beach or a train of swell over a reef.
- `BreakingWave` simulates every parcel of water as a particle with weakly compressible SPH. At the resolution a desktop affords it breaks too late; it is kept as the particle reference.

Grilli et al.'s (1997) slope parameter $S_0 = 1.521 \tan\beta / \sqrt{H/d}$ predicts the breaker type before anything runs. The runs are Froude-similar, so every result scales with the depth $d$ and every time with $\sqrt{d/g}$. The GIFs show the whole bed above, its vertical scale stretched, and a true-scale close-up below; the red marker is the measured breaking point.

| | Potential flow (BEM) | Free surface (SGN) | Particles (SPH) |
| --- | --- | --- | --- |
| Unknowns, 1:35 case | 440 to 570 surface and 224 bed nodes | Depth and depth-averaged velocity on 3,869 cells | 7,748 particles |
| Run time, 1:35 case | 8 min | 70 s | 37 min |
| Breaking point vs Grilli (1997) | $H_b/d$ +5.8%, $h_b/d$ +4.1% | $H_b/d$ -4.5%, $h_b/d$ +2.0% | $H_b/d$ -32%, $h_b/d$ -72% |
| Overturn shape at impact vs DNS and field | $A_O/H_I^2$ -0.9%, $\theta_O$ -3.0°; $W/L$ +1.3% vs field | None: the front becomes a bore | Not measured |
| Overturning lip | Yes, until the jet lands | No: the front becomes a bore | Yes, a few particles thick |
| After breaking | Ends at touchdown; `SplashUp` carries it on with SPH | Bore and runup | Splash and runup, coarse |
| Breaking criterion | Front face vertical (Grilli's definition) | Front steeper than 30 degrees (Tissier et al. 2012) | Front face leaning past vertical |

#### Free-surface model (Serre-Green-Naghdi)

![Solitary wave breaking on a 1:35 beach, Serre-Green-Naghdi](assets/surfaceWave/surfaceWave.gif)

```python
from pysurf.SurfaceBreakingWave import SurfaceBreakingWave

wave = SurfaceBreakingWave(depth = 1.0, waveHeight = 0.3, beachSlope = 1 / 35, cellsPerDepth = 50)
wave.run()
wave.summary()                               # breaking point, breaker type
wave.writeAnimation('surfaceWave.gif')
```

The Serre-Green-Naghdi (SGN) equations keep the full nonlinearity of the shallow-water equations and add the pressure that the vertical acceleration of the water column carries, which gives the waves dispersion. `sgnSolver` splits them in two (Bonneton et al. 2011): the shallow-water part as finite volumes with the hydrostatic reconstruction of Audusse et al. (2004), which holds a lake at rest exactly and lets the water run up a dry beach, and the dispersive part as a tridiagonal solve for the extra acceleration. The wave starts as the exact SGN solitary wave. When its front passes 30 degrees, the dispersive part is switched off over a zone 2.5 roller lengths wide around the front (Tissier et al. 2012), and the shallow-water equations carry the front as a bore, which dissipates energy as a hydraulic jump does. The water is drawn as columns from the bed to the surface, colored by depth-averaged speed and whitened where the wave is breaking.

| Quantity | Reference | Reference value | Model value | Error | Verdict |
| --- | --- | --- | --- | --- | --- |
| Breaking height $H_b/d$, $H/d = 0.3$, slope 1:35, 50 cells per depth | Grilli et al. (1997) | 0.476 | 0.454 | -4.5% | Pass (10%) |
| Breaking depth $h_b/d$ | Grilli et al. (1997) | 0.345 | 0.352 | +2.0% | Pass (15%) |
| Breaking index $H_b/h_b$ | Grilli et al. (1997) | 1.380 | 1.291 | -6.5% | |
| Non-breaking runup $R/d$, $H/d = 0.019$, slope 1:19.85 | Synolakis (1987) | 0.0890 | 0.0874 | -1.8% | Pass (10%) |
| Exact SGN solitary wave, height and speed after 7 depths, 40 cells per depth | Closed form | | | < 0.01% | Pass, second order |
| Standing wave period at $kd = 1$ | SGN linear dispersion | | | < 0.01% | Pass |
| Dam break onto a dry bed | Ritter (1892) | | | 0.12% (L1) | Pass, first order |

The breaking criteria are the published values, not tuned to Grilli: Tissier et al. took the 30-degree angle from Cienfuegos et al.'s (2010) fit to laboratory breaking. The breaking point is converged in resolution: $H_b/d$ is 0.452, 0.454 and 0.455 at 25, 50 and 100 cells per depth. The criterion is the sensitivity. Grilli's breaking point is where the face turns vertical, which a single-valued surface cannot reach, and a 25- or 35-degree angle moves $h_b/d$ to 0.390 or 0.321. Not validated: the bore after breaking, and the runup of a breaking wave. The runup tongue has no bed friction, so it climbs as a sheet millimeters thick and still moving fast when the run ends; the runup the summary reports for a breaking wave is not a measurement.

#### Overturning (boundary element potential flow)

![Solitary wave overturning over a reef, potential flow](assets/overturningWave/overturningWave.gif)

```python
from pysurf.bathymetry import Bathymetry
from pysurf.OverturningWave import OverturningWave

reef = Bathymetry.reefBreak(depth = 1.0, toeX = 14.4, faceSlope = 0.1, reefDepth = 0.3, reefLength = 8.0)
wave = OverturningWave(depth = 1.0, waveHeight = 0.3, bathymetry = reef, beachSlope = 0.1)
wave.run()
wave.summary()                               # breaking point; tube and overturn measures at impact
wave.drawProfiles()                          # the overturn as a sequence of profiles
wave.drawOverturn()                          # the overturn at impact with its fitted curve
wave.writeAnimation('overturningWave.gif')
```

The water is inviscid and irrotational, so `bemSolver` solves Laplace's equation on the boundary alone: the free surface, the bed and a wall at each end. The boundary is cut into straight elements whose integrals have closed forms, so there is no quadrature error however close the jet comes to the water below. The free-surface nodes move with the water and carry the potential from the zero-pressure surface condition, stepped with fourth-order Runge-Kutta; once the face overhangs, no node moves more than half a spacing in a step. Because the nodes move with the water, the surface can fold into a jet. They are regridded finer around the crest, wherever the surface curves sharply, and wherever it folds back over itself, as the face does under a landing jet, and smoothed with Longuet-Higgins and Cokelet's five-point filter against the sawtooth that grows on a Lagrangian surface. The run ends when the jet comes within one element of the face: closer than that the elements cannot resolve the water between the two, and once the jet lands the flow has a broken surface. Running in 2D slices along the wave's path is a fair approximation for the barrel: Guyenne and Grilli (2006) found the jet at the middle of a 3D ridge closely matched 2D results.

The GIF is a solitary wave of $H/d = 0.3$ breaking in open water over a 1:10 reef face onto a reef flat $0.3d$ deep, the geometry of Blenkinsopp and Chaplin's (2008) reef flume. It breaks at the reef edge with $H_b/h_b = 1.41$. When the jet lands, 0.34 s later, the tube under it is $0.24d$ long and $0.25d$ high and encloses $0.029d^2$ of air.

| Quantity | Reference | Reference value | Model value | Error | Verdict |
| --- | --- | --- | --- | --- | --- |
| Breaking height $H_b/d$, $H/d = 0.3$, slope 1:35 | Grilli et al. (1997) | 0.476 | 0.504 | +5.8% | Pass (10%) |
| Breaking depth $h_b/d$ | Grilli et al. (1997) | 0.345 | 0.359 | +4.1% | Pass (15%) |
| Breaking index $H_b/h_b$ | Grilli et al. (1997) | 1.380 | 1.402 | +1.6% | Pass (10%) |
| Standing wave period, $kd = 1$ and $kd = 3$ | Airy | | | -0.03%, -0.11% | Pass |
| Harmonic field $\cosh k(z+d) \cos kx$ in a rectangle | Closed form | | | 4.8e-5 at 160 nodes | Pass, second order |
| Volume and energy over two periods | Conservation | | | 1e-8, 1e-5 | Pass |
| Impact location $x/h_0$, soliton $a_0/h_0 = 0.6$ on the Surf Ranch slope | Feddersen et al. (2024), DNS | 42.05 | 41.85 | $-0.20h_0$ | Pass ($0.5h_0$) |
| Impact height $H_I/h_0$ | Feddersen et al. (2024), DNS | 0.64 | 0.654 | +2.2% | Pass (5%) |
| Overturn area $A_O/H_I^2$ | Feddersen et al. (2024), DNS | 0.352 | 0.349 | -0.9% | Pass (15%) |
| Jet area $A_J/H_I^2$ | DNS, as quoted by Pick and Feddersen (2026) | 0.22 | 0.195 | -12% | Pass (20%) |
| Overturn angle $\theta_O$ | DNS, as quoted by Pick and Feddersen (2026) | 33° | 30.0° | -3.0° | Pass (5°) |
| Overturn aspect ratio $W/L$ | Feddersen et al. (2024), DNS | 0.300 | 0.385 | +28% | Fail (15%) |
| Overturn aspect ratio $W/L$, zero wind | Feddersen et al. (2023), field scale | 0.38 | 0.385 | +1.3% | |
| Overturn area $A/H^2$, zero wind | Feddersen et al. (2023), field scale | 0.31 | 0.349 | +13% | |
| $A_O/H_I^2$, $W/L$, $\theta_O$, $H_b/h_b$; slopes 1:30 to 1:15, $H_0/h_0$ = 0.2 to 0.6 | Pick and Feddersen (2026), potential flow | Fits | 5 cases | Within 6%, 6%, 2.3°, 4.1% | Pass (3 RMSE) |
| Jet area $A_J/H_I^2$, the same five cases | Pick and Feddersen (2026), potential flow | Fit | 5 cases | +2% to +19% | Fail in 3 of 5 (3 RMSE) |

The breaking point is converged in resolution: $H_b/d$ is 0.504 and $h_b/d$ 0.359 at crest spacings of both $0.02d$ and $0.014d$. The remaining difference from Grilli comes from the starting wave: Grilli's is the exact potential flow solitary wave, this one the exact SGN wave with potential flow's vertical velocity structure added, which adjusts by about 1% in height over the first depths of travel. On the 1:35 slope the jet lands 0.36 s after breaking, over a tube $0.26d$ long and $0.28d$ high enclosing $0.035d^2$ of air, the same to 1% at a crest spacing of $0.014d$.

![Overturning profiles of a solitary wave on a 1:35 slope](assets/overturningWave/grilliProfiles.png)

The overturn is measured as Feddersen et al. (2023, 2024) and Pick and Feddersen (2026) measure it at jet impact: the area $A_O$ enclosed by the overturn boundary, the area $A_J$ of the jet (the water above the tube, where the surface is multi-valued in $x$), both over the square of the highest elevation $H_I$, and the length $L$, width $W$ and angle $\theta_O$ of a Longuet-Higgins (1982) curve fitted to the boundary. Feddersen et al.'s (2024) DNS reproduces in 2D the soliton of their field-scale experiments in the Surf Ranch wave pool. The model runs the same case from the same starting wave, a Green-Naghdi solitary wave of $a_0/h_0 = 0.6$, and meets the criteria set before the comparison for the impact location, height, overturn area, jet area and angle. $W/L$ fails against the DNS but matches the field measurement at zero wind, and Pick and Feddersen's potential flow gives 0.41 for this case, so on this measure the DNS stands apart from both. Across plane slopes the overturn follows Pick and Feddersen's potential flow fits in every measure but the jet area, which is up to 19% above their fit while 12% below the DNS; it grows fast in the last instants before impact, which each study fixes in its own way.

![The Surf Ranch soliton at jet impact, with the fitted overturn curve](assets/overturningWave/surfRanchOverturn.png)

Not validated: the solitary wave over the reef in the GIF. Blenkinsopp and Chaplin's (2008) reef flume ran periodic waves, which `ReefOverturn` runs (below), and Mead and Black's (2001) relation between seabed gradient and tube shape comes from photographs of surf breaks rather than controlled conditions.

#### Splash-up (potential flow handed to SPH)

![The Surf Ranch soliton's jet landing and splashing up, SPH continued from the potential flow](assets/splashUp/splashUp.gif)

```python
from pysurf.OverturningWave import OverturningWave
from pysurf.SplashUp import SplashUp

wave = OverturningWave(depth = 1.0, waveHeight = 0.3, beachSlope = 1 / 35)
wave.run()                                   # to the instant the jet lands
splash = SplashUp(wave, particlesPerDepth = 64)
splash.run()
splash.summary()                             # the hand-off, the window, the energies
splash.writeAnimation('splashUp.gif', dpi = 72)
```

Potential flow ends where the jet lands: after that the surface encloses the tube and the impact makes vorticity. `SplashUp` hands the flow at that instant to the SPH solver, in a window from three depths behind the crest to six ahead of the impact. Particles fill the water on a lattice, and the tube under the jet starts empty. Each particle takes the potential flow's velocity, evaluated inside the water from the boundary integral representation, and its pressure, from the unsteady Bernoulli equation with $\partial\phi/\partial t$ from a second boundary solution, set as its density through the equation of state, so the water starts in the balance it had instead of settling. Behind the window a piston moves with the potential flow's depth-averaged velocity, the wave behind it treated as translating at its crest speed, so cutting the tank sends no reflection toward the jet. Ahead, the water is still at rest; the run ends before the broken water reaches the wall there.

The GIF continues the Surf Ranch overturn validated above ($a_0/h_0 = 0.6$ on a 0.0693 slope) at 64 particles per depth. The jet lands, and a splash-up shoots almost vertically from where it lands, rising to $1.33H_0$ ($0.80d$) at $t\sqrt{g/d} = 1.4$ after the impact, 23% above the jet's own crest. It curls into a second plunging jet over a second cavity while the first tube collapses behind it. The broken water's leading edge advances at 0.83 to 0.89 of the crest speed, and the fastest splash leaves at 2.0 times the crest speed. Lim et al. (2015) saw both in deep-water plunging breakers, a first splash-up higher than the overturning jet and starting at up to 2.14 times the phase speed; that is consistency, not validation.

| Check | Reference | Result | Verdict |
| --- | --- | --- | --- |
| Velocity and pressure inside the water | Exact harmonic fields | $2 \times 10^{-4} k$ and $1.5 \times 10^{-4} \rho u^2$ at 80 nodes, second order | Pass |
| Still water handed off | Hydrostatic | At rest, pressure exact | Pass |
| SPH over the last 0.1 s before impact, 32 to 64 particles per depth | The validated potential flow (`OverturningWave`) | Surface gap at most 0.66 spacings; no water more than 0.05 spacings outside it | Pass (1 spacing) |
| Leading edge of the broken water, 64 and 96 particles per depth | Resolution | 4.18 and 3.91 m/s, 0.89 and 0.83 of the crest speed | Converged to 7% |
| Splash-up height, 64 and 96 particles per depth | Resolution | $1.324H_0$ and $1.331H_0$, at $t\sqrt{g/d} = 1.50$ and 1.35 | Converged to 0.5% in height, 10% in time |
| Share of energy lost in active breaking, $H_0/h_0 = 0.3$ on 3°, 48 particles per depth | Mostert and Deike (2020), DNS | 0.277 against 0.31 | Pass (30%), -11% |
| Dissipation rate $(\epsilon/\epsilon_0)(d_b/d_0)$, the same case | Mostert and Deike (2020), inertial model fit | 0.00202 against 0.00217 | Pass (30%), -7% |
| Share lost and dissipation rate, 48 and 64 particles per depth | Resolution | 0.277 and 0.279; 0.00202 and 0.00200 | Converged to 1% |

The flow after the impact is checked against the energy Mostert and Deike (2020) measured breaking solitary waves lose in two-phase DNS. They take the active breaking time from the face turning vertical to the kinetic energy falling to the potential; their plunging case, $H_0/h_0 = 0.3$ on a 3° beach, loses 0.31 of its energy over it, and across slopes of 2° to 5° their dissipation rate collapses onto $(\epsilon/\epsilon_0)(d_b/d_0) = 0.0281(h_b/d_0)^{3.152}$. `SplashUp` runs the same case with the window's offshore wall standing still eight depths behind the crest, so the window's energy changes only by dissipation, and the beach running on eight depths past the shoreline. The SPH water loses 0.277 of its energy, 11% under theirs, at a rate 7% under the fit, within the 30% set before the comparison; at 64 particles per depth both move by under 1%. The SPH model's dissipation is its artificial viscosity, not resolved turbulence, so the agreement holds for this breaker rather than making the model a model of turbulence. Not validated: the shape of the splash-up itself, for which no profiles of a solitary wave's splash-up were found. The water is single-phase: the air in the tube neither cushions its collapse nor escapes, as it does in a real breaker, and the spray, a few percent of the particles, depends on the resolution rather than on the sea.

#### Reef break (swell, Serre-Green-Naghdi)

![Swell breaking on a reef, Serre-Green-Naghdi](assets/reefBreak/reefBreak.gif)

```python
from pysurf.ReefBreak import ReefBreak

reef = ReefBreak(depth = 10.0, waveHeight = 1.5, period = 12.0)
reef.run()
reef.summary()                               # breaking zones, where breaking stops
reef.writeAnimation('reefBreak.gif', lastPeriods = 3)
```

`ReefBreak` drives the SGN solver with regular swell over a reef from `Bathymetry.reefBreak`: 10 m offshore, a 1:5 reef face up to a reef flat 2.5 m deep, a 6 m lagoon and a 1:20 beach. A mass source offshore (Wei, Kirby and Sinha 1999) sends 1.5 m, 12 s waves at the reef; the half that runs seaward is absorbed in a layer at the offshore end. The dispersion parameter is Bonneton et al.'s 1.159, which holds the phase speed within 1% of Airy's out to $kd = 3$. Breaking is decided front by front: a front starts when it is steeper than 30 degrees and strong enough to be a breaking bore, and stops when its bore Froude number has stayed below 1.3 for $5\sqrt{h/g}$, the transition time of Kennedy et al. (2000).

Each wave shoals from 1.5 m to 2.1 m on the face and starts breaking at the reef edge, 379 m out, over 3.2 m of still water. The bore crosses the reef flat, weakens in the lagoon, and stops breaking at 463 m with a bore Froude number of 1.11. The wave reforms, crosses the lagoon, and breaks again as a shore break at 665 m in 1.2 m of water. The cycle repeats every period.

| Check | Reference | Result | Verdict |
| --- | --- | --- | --- |
| Wave height from the source, $kd$ = 0.5 to 1.5, 120 cells per wavelength | Linear SGN theory | within 0.4% | Pass |
| Reflection from the absorbing layer | None | 0.1% | Pass |
| Standing wave period with $\alpha = 1.159$, $kd = 2$ | $\alpha$-SGN dispersion; Airy | within 0.2%; 0.6% from Airy | Pass |

Not validated: breaking over the reef itself. Run in Blenkinsopp and Chaplin's (2008) flume by `ReefOverturn` (below), the same solver reflects 1.5 to 2.1 times what the flume reflected and passes up to 44% more over the reef: its breaking dissipates too little on a steep reef face. Fringing-reef experiments such as Yao et al. (2012) would test it at a scale nearer the sea's.

#### Reef overturn (SGN train handed to potential flow)

![A regular wave overturning on Blenkinsopp and Chaplin's reef, hc = 77 mm, at jet impact](assets/reefOverturn/reefOverturn.png)

```python
from pysurf.ReefOverturn import ReefOverturn

reef = ReefOverturn(crestDepth = 0.077, deepWaterHeight = 0.0755, frequency = 0.6)
reef.run()                                   # SGN train, hand-off, overturn to the jet's impact
reef.summary()                               # reflection, transmission, breaker indices, cavity
reef.wave.drawOverturn()
```

`ReefOverturn` builds Blenkinsopp and Chaplin's (2008) flume: a reef 0.618 m high with a 1:10 face in 0.68 to 0.72 m of water, its toe 3.8 m from the wave source. The SGN model runs a regular train over it until the train is periodic. A third of a period before SGN starts breaking a wave at the crest, the state goes to the boundary element model (`sgnHandoff`): the surface is the SGN surface, and the potential solves a Neumann problem whose flux is the rate the surface rises, $\eta_t = -\partial q/\partial x$, exact for any depth over wavelength. The tank's walls stand where the discharge crosses zero and the surface rises slowest. Behind the reef lie the bores of waves already broken, whose fronts of 30 to 40° potential flow cannot start from; they are smoothed to under 11°, keeping their mass, and the crest and the jet are sought no further than a quarter wavelength behind the reef. The reef drops behind its crest at 2:1 where the flume's drops vertically, since the SGN model's dispersive solve needs a bed of finite slope. Each water level runs in about three minutes.

The measures are the flume's. At the break point, where part of the face first turns vertical, $H_b$ runs from the trough ahead of the face to the crest and $h_b$ is the still water depth under the vertical face; at the plunge point the cavity is the Longuet-Higgins (1982) curve fitted to its boundary, $l_c$ long and $w_c$ wide, with area $A_c = (2\sqrt{3}/5) l_c w_c$. Ahead of the reef the train's first harmonic is split into incident and reflected waves; behind it the transmitted height is the mean crest-to-trough height a wavelength on. The reference values are Blenkinsopp and Chaplin's points for 0.6 Hz waves of $H_0 = 75.5$ mm in deep water ($H_0/L_0 = 0.0174$) at nine water levels, digitized from their figures by marker color. The criteria follow their stated errors: 20% on the cavity area, 10% on each length and so 15% on their ratio, 20% on the breaker indices and on $K_r$ and $K_t$, and 5% on the incident height.

| Quantity, $h_c$ = 62 to 102 mm | Reference | Reference value | Model value | Error | Verdict |
| --- | --- | --- | --- | --- | --- |
| Incident height $H_i$ | Blenkinsopp and Chaplin (2008), fig. 10 | 68.4 to 69.6 mm | 66.6 to 69.0 mm | -3.5% to +1% | Pass (5%) |
| Reflection from the absorbing layers, flat bed | None | | 1.4% | | Pass |
| Reflection coefficient $K_r$ | Fig. 11 | 0.12 to 0.20 | 0.26 to 0.33 | +46% to +111% | Fail (20%) |
| Transmission coefficient $K_t$ | Fig. 10 | 0.54 to 0.84 | 0.77 to 0.94 | +11% to +44% | Fail (20%) |
| Breaker depth index $H_b/h_b$ | Section 3.1.1 | 0.85 | 0.99 to 1.26 | +16% to +48% | Fail (20%) |
| Breaker height index $H_b/H_0$ | Eq. 3 | 1.12 | 1.23 to 1.76 | +9% to +56% | Fail (20%) |
| Cavity ratio $l_c/w_c$, $h_c/H_0 \le 0.95$ | Fig. 6 | 2.07 to 2.29 | 2.26 to 2.34 | +2% to +9% | Pass (15%) |
| Cavity ratio $l_c/w_c$, $h_c/H_0 \ge 1.02$ | Fig. 6 | 1.55 to 2.00 | 2.57 to 5.88 | +28% to +269% | Fail (15%) |
| Cavity area $A_c/H_b^2$ | Fig. 7 | 0.04 to 0.22 | 0.02 to 0.45 | -80% to +164% | Fail (20%) |

| $h_c$ [mm] | $h_c/H_0$ | $K_r$, model / flume | $K_t$, model / flume | $H_b/h_b$ | $H_b/H_0$ | $l_c/w_c$, model / flume | $A_c/H_b^2$, model / flume |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 62 | 0.82 | 0.26 / 0.12 | 0.77 / 0.54 | 0.99 | 1.23 | 2.29 / 2.10 | 0.42 / 0.22 |
| 67 | 0.89 | 0.27 / 0.14 | 0.78 / 0.57 | 1.07 | 1.40 | 2.34 / 2.29 | 0.35 / 0.21 |
| 72 | 0.95 | 0.31 / 0.15 | 0.81 / 0.60 | 1.25 | 1.63 | 2.26 / 2.07 | 0.28 / 0.18 |
| 77 | 1.02 | 0.33 / 0.16 | 0.81 / 0.65 | 1.20 | 1.54 | 2.62 / 1.90 | 0.45 / 0.20 |
| 82 | 1.09 | 0.26 / 0.18 | 0.82 / 0.72 | 1.03 | 1.37 | 4.02 / 1.82 | 0.42 / 0.16 |
| 87 | 1.15 | 0.27 / 0.19 | 0.84 / 0.75 | 1.10 | 1.50 | 5.88 / 1.60 | 0.15 / 0.13 |
| 92 | 1.22 | 0.29 / 0.19 | 0.88 / 0.80 | 1.16 | 1.63 | 4.00 / 1.56 | 0.024 / 0.12 |
| 97 | 1.28 | 0.30 / 0.20 | 0.94 / 0.82 | 1.26 | 1.76 | 3.24 / 1.73 | 0.019 / 0.06 |
| 102 | 1.35 | 0.33 / 0.19 | 0.94 / 0.84 | 1.21 | 1.69 | 2.57 / 2.00 | 0.023 / 0.04 |

The incident wave matches the flume; nothing after it does. The SGN train reflects 1.5 to 2.1 times what the flume reflected and passes up to 44% more over the reef. By the linear balance $1 - K_r^2 - K_t^2$ its breaking takes 0.34 of the incident energy at $h_c$ = 62 mm, where the flume's took 0.70, and 0.14 at 92 mm against 0.33. The waves it hands over break 16 to 48% higher against the depth than the flume's. Those breaker indices sit among the plane-slope relations Blenkinsopp and Chaplin plot (about 1.1 to 1.3 for $H_b/h_b$ at this steepness, all but McCowan's 0.78) rather than on their reef data, which they explain by the return flow from the wave before; here that return flow comes from a train that dissipates too little. The cavity ratio matches the flume within 9% up to $h_c/H_0 = 0.95$. From $h_c$ = 72 mm the jet lands past the crest, over the 2:1 back slope that stands in for the flume's vertical drop, and from $h_c/H_0 = 1.09$ the cavity grows long and thin. Its area falls with submergence as the flume's does, but it is 1.5 to 2.6 times the flume's up to $h_c/H_0 = 1.09$ and a fifth to a half of it from 1.22. The flume's 0.44 m width also puts side-wall friction on the jet, which Blenkinsopp and Chaplin report; neither that nor their stated errors cover these gaps. The overturn itself is validated (above); what fails here is the wave it is handed. Closing the gap needs SGN breaking that dissipates as a plunging breaker does on a steep reef, checked first against $K_r$ and $K_t$.

#### Particle model (SPH)

![Solitary wave breaking on a 1:15 beach](assets/breakingWave/breakingWave.gif)

```python
from pysurf.BreakingWave import BreakingWave

wave = BreakingWave(depth = 1.0, waveHeight = 0.3, beachSlope = 1 / 15, particlesPerDepth = 24)
wave.run()
wave.summary()                               # breaking point, breaker type, runup
wave.writeAnimation('breakingWave.gif')
```

`BreakingWave` is a facade like `Surfboard`: it lays out the beach (`bathymetry`), places the solitary wave already moving with its first-order velocity field (`waveMakers`), builds the sand and a seaward wall from boundary particles (`sphBoundaries`), steps the solver (`sphSolver`), and tracks the crest, the breaking point and the runup (`waveBreaking`). The breaking point is where the front face of the wave first leans forward of vertical, measured on the water binned into cells one spacing square. Particles are colored by speed. The GIF was written by `writeAnimation(..., frameStep = 2, dpi = 72)` from a 24 particles per depth run (12,121 fluid particles, 40 minutes).

| Quantity | Reference | Reference value | Model value | Error | Verdict |
| --- | --- | --- | --- | --- | --- |
| First-mode sloshing period, 0.3 m tank, 0.1 m deep, 5 mm spacing | Linear potential flow, $\omega_1^2 = g k_1 \tanh(k_1 d)$ | 0.7017 s | 0.6862 s | -2.2% | Pass (3%) |
| Solitary wave celerity, $H/d = 0.3$, 12 particles per depth | $\sqrt{g(d+H)}$ (Rayleigh, Boussinesq) | 3.571 m/s | 3.554 m/s | -0.5% | Pass (3%) |
| Solitary wave celerity, 16 particles per depth | $\sqrt{g(d+H)}$ | 3.571 m/s | 3.560 m/s | -0.3% | Pass (3%) |
| Breaking height $H_b/d$, $H/d = 0.3$, slope 1:35, 16 particles per depth | Grilli et al. (1997) | 0.476 | 0.325 | -32% | Fail (10%) |
| Breaking depth $h_b/d$ | Grilli et al. (1997) | 0.345 | 0.097 | -72% | Fail (15%) |
| Breaking index $H_b/h_b$ | Grilli et al. (1997) | 1.380 | 3.34 | +142% | Fail |

The sloshing tank and the solitary wave pass: the solver reproduces the dispersion of standing and progressive gravity waves to within 2.2% and 0.5%. The breaking point does not. At 16 particles per depth the wave breaks far up the beach, in water a third as deep as in Grilli's computation and at two thirds of its breaking height. With the artificial viscosity also acting at the bed, it broke lower still ($H_b/d = 0.262$, $h_b/d = 0.119$).

The cause is numerical damping. Over 3 s of travel on a flat bottom (about 11 depths), the crest loses 20%, 12% and 6.5% of its height at 8, 12 and 16 particles per depth. On the 1:35 slope the wave crosses about 30 depths before it breaks, and the loss there outweighs the shoaling that should raise it 30% (Green's law, $H \propto h^{-1/4}$). Particle size makes it worse, since it is fixed while the water shallows: at Grilli's breaking depth only 5.5 particles span the water column. Closing the gap takes either several times the resolution, which in 2D costs the cube of the factor in run time, or a density-diffusion scheme such as $\delta$-SPH (Molteni and Colagrossi 2009) in place of the Shepard filter.

The GIF is the steeper 1:15 beach at 24 particles per depth. The wave crosses 15 depths of slope instead of 30, and when it breaks its crest stands about ten particles above the bed, against seven in the 1:35 run, enough to see the face turn vertical and the jet thrown ahead of it. It still breaks late, in water $0.11d$ deep with $H_b/d = 0.33$, and has no published reference here; it is a qualitative picture of the breaking process.

Post-breaking flow (the bore and the runup tongue) has no reference here at all; at a few particles across the jet it shows where the water goes, not how fast.

### [WaterSim →](src/pysurf/WaterSim/README.md)

Ready-to-run SPH scenarios built from the water modules: a sloshing tank (2D and 3D) and a numerical wave tank with a piston wave maker sized by Biesel's transfer function. WaterSim imports nothing from the board model.

```bash
python -m pysurf.WaterSim   # Run default sloshing tank
```

### Water modules

| Module | Purpose |
| --- | --- |
| `waveTheory` | Linear dispersion, phase and group speed, kinematics and energy; first-order and exact SGN solitary waves; SGN linear dispersion; Biesel piston transfer function; sloshing frequencies; McCowan, Miche, Iribarren and Grilli breaking criteria |
| `sgnSolver` | Serre-Green-Naghdi equations: shallow-water finite volumes with hydrostatic reconstruction, the dispersive tridiagonal solve, and the Tissier breaking switch |
| `sgnHandoff` | An SGN state handed to the boundary element model: wall placement, bore smoothing, the surface potential from a Neumann solve |
| `bemSolver` | Fully nonlinear potential flow by boundary elements: the Lagrangian free surface, regridding and smoothing, jet and overturn measures, and the flow inside the water (velocity, and pressure from $\phi_t$) |
| `sphHandoff` | The potential flow handed to SPH particles in a window: seeding, density from the pressure, boundary pressure, piston |
| `bathymetry` | The bed as a 2D polyline: elevation, depth, slope, normal, shoreline, equal-arc sampling, sand outline |
| `sphKernels` | Cubic spline and Wendland C2 kernels, pair and batch forms |
| `sphNeighbors` | Every particle pair inside the kernel support, from a k-d tree |
| `sphParticles` | Particle state, energy diagnostics, lattice seeding and region fills |
| `sphBoundaries` | Container and bed boundaries of fixed particles, with enforcement |
| `sphSolver` | Tait equation of state, continuity density, Shepard filter, artificial viscosity, XSPH, CFL step, and the solver that composes them |
| `waveMakers` | Piston wave maker; solitary wave initial condition |
| `waveBreaking` | For particles: free-surface profile, front face lean, breaking tracker, waterline and runup, kinematic breaking detector. For surface profiles: crest, wave front and its steepest slope, breaking tracker |
| `constants` | Gravity and water properties |

### Shared modules

| Module | Purpose |
| --- | --- |
| `geometryTools` | Curve and surface math: Bezier construction and subdivision, offsets, intersections, fillets, arc-length resampling, revolution, rotation |
| `figures` | Every figure, described once as a dataclass and drawn by both Matplotlib and Plotly: board profiles against the reference photographs, cross sections, and the 3D surface. `exportInteractiveFigures` writes the HTML companions; `PYSURF_HEADLESS=1` suppresses every window |
| `palette` | The Surfy Pastels palette every figure is drawn in: sand and seafoam by default, night ocean with `PYSURF_FIGURE_MODE=dark`; nineteen named colors and the ocean, sunset and deviation colormaps |
| `exports` | Binary STL from structured grids; delimited text |
| `units` | One Pint registry and every conversion factor (ft, in, L, lb, kn, mph to SI) |
| `errors` | The exception family pySurf raises on purpose; `InvalidInputError` is also a `ValueError` |
| `validation` | Rule tables that check inputs before a calculation runs |

---

## Project Structure

```txt
repo_root/
├── pyproject.toml                   Package metadata, dependencies, pytest configuration
├── README.md
├── assets/
│   ├── referenceMaterial/           Thruster reference photographs, dimensions, and STL
│   ├── breakingWave/                Breaking wave animation, particle model
│   ├── surfaceWave/                 Breaking wave animation, free-surface model
│   ├── overturningWave/             Overturn animation and figures, potential flow
│   ├── splashUp/                    Splash-up animation, potential flow handed to SPH
│   ├── reefBreak/                   Swell over a reef animation, free-surface model
│   └── reefOverturn/                A reef overturn at jet impact
├── tests/                           pytest suite; `-m slow` runs the validation cases
│
└── src/
    └── pysurf/
        ├── __init__.py              Package facade: `from pysurf import Surfboard`
        ├── Surfboard.py             Parametric board: presets, profiles, cross sections
        ├── BreakingWave.py          Solitary wave breaking on a plane beach, particles (SPH)
        ├── SurfaceBreakingWave.py   Solitary wave breaking on a plane beach, free surface (SGN)
        ├── OverturningWave.py       Plunging jet over any bed, potential flow (BEM)
        ├── ReefBreak.py             Regular swell breaking over a reef, free surface (SGN)
        ├── ReefOverturn.py          Regular waves overturning on a reef: SGN train handed to BEM
        ├── SplashUp.py              The overturn through jet impact: BEM handed to SPH
        ├── geometryTools.py         Curve and surface primitives
        ├── figures.py               Figure descriptions, Matplotlib and Plotly renderers, HTML and GIF export
        ├── palette.py               Surfy Pastels figure colors
        ├── exports.py               STL and text output
        ├── units.py                 Unit registry and conversion
        ├── constants.py             Gravity and water properties
        ├── errors.py                Exception family
        ├── validation.py            Input rule vocabulary
        ├── waveTheory.py            Linear and solitary wave mechanics, breaking criteria
        ├── bathymetry.py            2D bed profile
        ├── sphKernels.py            SPH kernels
        ├── sphNeighbors.py          Pair search
        ├── sphParticles.py          Particle state and seeding
        ├── sphBoundaries.py         Container and bed boundaries
        ├── sphSolver.py             WCSPH equations and time stepping
        ├── sgnSolver.py             Serre-Green-Naghdi equations and the breaking switch
        ├── sgnHandoff.py            SGN state to the boundary element model
        ├── bemSolver.py             Boundary element potential flow, interior flow
        ├── sphHandoff.py            Potential flow to SPH particles
        ├── waveMakers.py            Piston wave maker, solitary initial condition
        ├── waveBreaking.py          Free surface, breaking point, runup
        │
        └── WaterSim/                Ready-to-run SPH scenarios → see WaterSim/README.md
            ├── runner.py             Command-line entry point
            ├── scenarios/            Sloshing tank, numerical wave tank
            ├── export/               Frame data export
            ├── configs/              Water sim JSON configs
            └── documentation/        Water sim technical writeups and references
```

---

## Prerequisites & Setup

```bash
git clone https://github.com/sean-bowman/pySurf.git
cd pySurf

# Editable install (src layout, makes `pysurf` importable everywhere)
pip install -e .
```

**Requirements:**

- Python 3.10 or newer
- NumPy, SciPy, Matplotlib, Plotly, Pint, trimesh (installed by the command above)

---

## Tests

```bash
python -m pytest            # unit and verification tests, about 2 minutes
python -m pytest -m slow    # validation runs, about two hours
```

The suite runs headless and needs no install: `pyproject.toml` puts `src/` on the path. The unit tests cover the shared modules, the board's input rules and figures, the wave theory against its closed forms and limits, each SPH equation against its closed form or a conservation law, the breaking measures on constructed particle arrangements, and a short breaking-wave run through to the GIF. The SGN solver is checked against a lake at rest, Ritter's dam break, the SGN standing-wave frequency and its exact solitary wave, and its wave source, absorbing layer and improved dispersion against linear theory. The boundary element solver is checked against quadrature, an exact harmonic field and the Airy standing-wave period, its jet measures against a constructed jet over a tube, and the flow it hands to SPH (velocity and pressure inside the water) against exact harmonic fields. The hand-off to SPH is checked on still water, on a jet over an empty tube, and on a solitary wave's mass flux; the hand-off from SGN against a linear progressive wave's potential, and its walls against a bore. The slow tests are the validation cases in the tables above: the sloshing period, the solitary celerity, the Grilli breaking point for all three models, the Synolakis runup, the overturn at jet impact against the DNS of Feddersen et al. (2024) and the potential flow scaling of Pick and Feddersen (2026), the SPH continuation against the potential flow over the last tenth of a second before the jet lands, the energy it loses in active breaking against the DNS of Mostert and Deike (2020), and the reef overturn against the flume of Blenkinsopp and Chaplin (2008).

---

## Technology

| Component | Language | Purpose |
| --- | --- | --- |
| Surfboard Geometry | Python / NumPy | Parametric Bezier profiles and cross sections |
| Water Simulation | Python / NumPy, SciPy | WCSPH sloshing, wave tanks and breaking waves; Serre-Green-Naghdi breaking waves |
| Wave Theory | Python | Linear and solitary waves, wave makers, breaking criteria |
| Visualization | Python / Matplotlib, Plotly, Pillow | Profile views, cross sections, 3D surfaces, breaking-wave GIFs |
| Units | Python / Pint | Conversion between shaping units and SI |

---

## Future Work

- [X] Document all work thoroughly with informational overviews of all modules
- [X] Prioritize API calls over CLI calls for code library interfacing
- [X] Update surfboard and fin shapes against reference STLs for maximum realism
- [X] Investigate PicoGK geometry generation methodologies
- [X] Parallel-develop surface-mesh geometry generation through Python
- [X] Consolidate references and documentation folders
- [X] Model a solitary wave shoaling and breaking on a beach, with a GIF of the transient
- [X] Free-surface (Serre-Green-Naghdi) model that reaches Grilli's breaking point
- [X] Barrel shape over any bed from fully nonlinear potential flow, up to touchdown
- [X] Swell breaking over a reef, with breaking that starts and stops front by front
- [X] Periodic waves overturning on a reef: an SGN train handed to potential flow, compared with Blenkinsopp and Chaplin's (2008) flume
- [ ] SGN breaking that dissipates as a plunging breaker does on a steep reef, bringing the reef train's reflection and transmission to the flume's
- [ ] Validate the shape of the splash-up against laboratory profiles of a breaking solitary wave
- [ ] Give the SPH splash-up an air phase, so the tube's air cushions its collapse
- [ ] Bed friction in the swash, so the runup of a breaking wave can be measured
- [ ] Reach Grilli's breaking point with particles: density diffusion ($\delta$-SPH) and higher resolution
- [ ] Extend the breaking wave to 3D and to periodic waves from the piston wave maker
- [ ] Add new geometry domains (boat hull, etc.) as sibling packages

---

## References

- Paine, Frank. *The Science of Surfing*. Cambridge University Press.
- Dean & Dalrymple. *Water Wave Mechanics for Engineers and Scientists*.
- Grilli, S.T., Svendsen, I.A. & Subramanya, R. (1997). Breaking criterion and characteristics for solitary waves on slopes. *J. Waterway, Port, Coastal, Ocean Eng.* 123(3), 102-112.
- Senthilkumar, A. (2016). Solitary wave shoaling and breaking in a regularized Boussinesq system. arXiv:1601.06822 (Table 8 tabulates the Grilli breaking values used above).
- Dalrymple, R.A. & Rogers, B.D. (2006). Numerical modeling of water waves with the SPH method. *Coastal Engineering* 53, 141-147.
- Tissier, M., Bonneton, P., Marche, F., Chazel, F. & Lannes, D. (2012). A new approach to handle wave breaking in fully non-linear Boussinesq models. *Coastal Engineering* 67, 54-66.
- Bonneton, P., Chazel, F., Lannes, D., Marche, F. & Tissier, M. (2011). A splitting approach for the fully nonlinear and weakly dispersive Green-Naghdi model. *J. Comput. Phys.* 230, 1479-1498.
- Audusse, E., Bouchut, F., Bristeau, M.-O., Klein, R. & Perthame, B. (2004). A fast and stable well-balanced scheme with hydrostatic reconstruction for shallow water flows. *SIAM J. Sci. Comput.* 25(6), 2050-2065.
- Synolakis, C.E. (1987). The runup of solitary waves. *J. Fluid Mech.* 185, 523-545.
- Molteni, D. & Colagrossi, A. (2009). A simple procedure to improve the pressure evaluation in hydrodynamic context using the SPH. *Comput. Phys. Commun.* 180, 861-872.
- Finney, Ben & Houston, James. *Surfing: A History of the Ancient Hawaiian Sport*.

---

Sean Bowman
