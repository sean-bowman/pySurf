
# -- pySurf: Reef Overturn -- #

'''

Regular waves overturning on a submerged reef: the Serre-Green-Naghdi model carries a wave train in
over the reef face, and a wave about to break is handed to the boundary element model, whose free
surface can fold, for its overturn and the cavity of air under the jet. Each part is a module:

    pysurf.sgnSolver        the wave train: a source, absorbing layers at both ends, the reef
    pysurf.sgnHandoff       the free surface and its potential from the SGN state
    pysurf.OverturningWave  the overturn by boundary elements, to the jet's impact
    pysurf.bemSolver        the cavity under the jet at impact

    reef = ReefOverturn(crestDepth = 0.082, deepWaterHeight = 0.0755, frequency = 0.6)
    reef.run()
    reef.summary()

The default reef is the laboratory reef of Blenkinsopp and Chaplin (2008): 0.618 m high with a 1:10
face, in 0.68 to 0.72 m of water, its toe 3.8 m from the wave source. Their reef drops vertically
behind the crest; here it drops at 2:1, the same in both models, since the SGN model's dispersive
solve needs a bed of finite slope. The wave source makes the height a deep-water wave of height H0
shoals to by linear theory at the flume's depth, H0 being their offshore height in deep water.

The SGN model runs until the train over the reef is periodic. A third of a period before SGN starts
breaking a wave at the crest (its front steeper than 30 degrees), the state goes to the boundary
element model, in a tank whose walls stand where the discharge crosses zero and the surface rises
slowest. Over that third of a period the wave turns its face vertical and plunges, by potential flow
alone. Behind the reef the state holds the bores of waves already broken, whose fronts of 30 to 40
degrees potential flow cannot start from; their fronts are smoothed over 0.4 m to under 11 degrees,
keeping their mass. They lie a metre or more ahead of the wave being measured and move away from
it, and the crest and the jet are sought only up to a quarter wavelength behind the reef.

The measures are Blenkinsopp and Chaplin's. At the break point, where part of the face first turns
vertical, the breaking height Hb runs from the trough just ahead of the face to the crest, and hb
is the still water depth under the vertical face. At the plunge point, where the jet lands, the
cavity is the Longuet-Higgins (1982) curve fitted to its boundary, of length lc and width wc, and its
area is Ac = (2 sqrt(3) / 5) lc wc.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

Not validated: the model fails against Blenkinsopp and Chaplin's (2008) flume everywhere but the
incident wave and the cavity's shape at low submergence. The reference values are their points for
0.6 Hz waves of H0 = 75.5 mm (H0/L0 = 0.0174) at nine water levels, digitized from their figures
6, 7, 10 and 11; the criteria follow their stated errors (tests/testValidationReefOverturn.py).

    quantity, hc = 62 to 102 mm     model           flume           error           verdict
    incident height Hi [mm]         66.6 to 69.0    68.4 to 69.6    -3.5 to +1 %    pass (5 %)
    reflection Kr                   0.26 to 0.33    0.12 to 0.20    +46 to +111 %   fail (20 %)
    transmission Kt                 0.77 to 0.94    0.54 to 0.84    +11 to +44 %    fail (20 %)
    Hb / hb                         0.99 to 1.26    0.85            +16 to +48 %    fail (20 %)
    Hb / H0                         1.23 to 1.76    1.12            +9 to +56 %     fail (20 %)
    lc / wc, hc / H0 <= 0.95        2.26 to 2.34    2.07 to 2.29    +2 to +9 %      pass (15 %)
    lc / wc, hc / H0 >= 1.02        2.57 to 5.88    1.55 to 2.00    +28 to +269 %   fail (15 %)
    Ac / Hb^2                       0.02 to 0.45    0.04 to 0.22    -80 to +164 %   fail (20 %)

The SGN train passes too much over the reef and reflects too much: by the linear balance
1 - Kr^2 - Kt^2 its breaking takes 0.34 of the incident energy at hc = 62 mm against the flume's
0.70. The waves it hands over break higher against the depth than the flume's, among the
plane-slope relations Blenkinsopp and Chaplin compare with rather than on their reef data. The
cavity area falls with submergence as the flume's does, but at 1.5 to 2.6 times its value up to
hc / H0 = 1.09 and a fifth to a half of it from 1.22, where the jet lands over the 2:1 back slope
that stands in for the flume's vertical drop. The overturn by boundary elements is validated in
`pysurf.OverturningWave`; what fails is the wave it is handed. On a flat bed the absorbing layers
reflect 1.4 percent of the train. The hand-off is checked in tests/testSgnHandoff.py.

References:
-----------
Blenkinsopp, C.E. & Chaplin, J.R. (2008) -- The effect of relative crest submergence on wave
    breaking over submerged slopes, Coastal Engineering 55(12), 967-974.
Longuet-Higgins, M.S. (1982) -- Parametric solutions for breaking waves, J. Fluid Mech. 121, 403-424.

Sean Bowman [10/14/2026]

'''

# Standard library imports
from __future__ import annotations

import math
import time as timeModule
from dataclasses import dataclass

# Third-party imports
import numpy as np

# Local imports
from pysurf import constants, waveTheory
from pysurf.bathymetry import Bathymetry
from pysurf.OverturningWave import OverturningWave
from pysurf.sgnHandoff import dischargeZeroCrossing, sgnSurfaceState, smoothSteepFronts
from pysurf.sgnSolver import SgnConfig, SgnSolver, WaveSource
from pysurf.sphHandoff import windowBathymetry
from pysurf.validation import applyRules, numericRule

# Inputs a reef overturn must satisfy before anything is built
inputRules = (
    numericRule('crestDepth',      'Water depth over the reef crest', units = 'm', minimum = 0.0),
    numericRule('deepWaterHeight', 'Deep-water wave height',          units = 'm', minimum = 0.0),
    numericRule('frequency',       'Wave frequency',                  units = 'Hz', minimum = 0.0),
    numericRule('reefHeight',      'Reef height',                     units = 'm', minimum = 0.0),
    numericRule('faceSlope',       'Reef face slope',                 minimum = 0.0),
    numericRule('backSlope',       'Reef back slope',                 minimum = 0.0),
    numericRule('lead',            'Hand-off lead, in periods',       minimum = 0.0, maximum = 1.0),
)

@dataclass
class SgnState:

    '''

    One stored instant of the SGN wave train.

    Parameters:
    -----------
    time : float
        Simulated time [s]
    surface, discharge : np.ndarray
        Free-surface elevation [m] and discharge [m^2/s] at the cell centers

    '''

    time: float
    surface: np.ndarray
    discharge: np.ndarray

class ReefOverturn:

    '''

    A regular wave train breaking on a submerged reef, its overturn by potential flow.

    Parameters:
    -----------
    crestDepth : float
        Still water depth over the reef crest hc [m]
    deepWaterHeight : float
        Deep-water wave height H0 [m]
    frequency : float
        Wave frequency [Hz]
    reefHeight : float
        Height of the reef above the flume floor [m]
    faceSlope : float
        Slope of the reef face [-]
    backSlope : float
        Slope of the reef's back, behind the crest [-]
    toeDistance : float
        Distance from the wave source to the reef's toe [m]
    cellsPerDepth : int
        SGN cells across the flume depth
    periods : float
        Wave periods the SGN train runs over the reef before the hand-off is chosen
    lead : float
        How long before SGN starts breaking the wave at the crest it is handed off, in periods
    crestSpacingRatio : float
        Boundary element node spacing around the crest, over the flume depth
    crestWindowRatio : float
        Scale on the boundary element model's crest window, over its two depths behind the crest;
        a reef breaker is a few tenths of the flume depth high

    '''

    # -- Constructor -- #
    def __init__(self, crestDepth: float = 0.082, deepWaterHeight: float = 0.0755, frequency: float = 0.6,
                 reefHeight: float = 0.618, faceSlope: float = 0.1, backSlope: float = 2.0, toeDistance: float = 3.8,
                 cellsPerDepth: int = 40, periods: float = 6.0, lead: float = 1.0 / 3.0,
                 crestSpacingRatio: float = 0.005, crestWindowRatio: float = 0.25) -> None:

        self.crestDepth = crestDepth                # [m]
        self.deepWaterHeight = deepWaterHeight      # [m]
        self.frequency = frequency                  # [Hz]
        self.reefHeight = reefHeight                # [m]
        self.faceSlope = faceSlope                  # [-]
        self.backSlope = backSlope                  # [-]
        self.toeDistance = toeDistance              # [m]
        self.cellsPerDepth = cellsPerDepth
        self.periods = periods
        self.lead = lead
        self.crestSpacingRatio = crestSpacingRatio
        self.crestWindowRatio = crestWindowRatio
        # Improved dispersion for the SGN train, as in `ReefBreak`
        self.alpha = 1.159

        self.bathymetry: Bathymetry | None = None
        self.sgn: SgnSolver | None = None
        self.states: list[SgnState] = []
        self.breakingStarts: list[tuple[float, float]] = []
        self.handoffState: SgnState | None = None
        # The hand-off state with the bores behind the reef smoothed, which the tank starts from
        self.tankState: SgnState | None = None
        self.wave: OverturningWave | None = None
        self.wallClockSeconds = 0.0                 # [s]

        self._validateInputs()

    # -- Private Helper Methods -- #
    def _validateInputs(self) -> None:

        '''

        Check the inputs against the rule table.

        Raises:
        -------
        InvalidInputError : If any input is out of range.

        '''

        applyRules(self, inputRules)

    # -- Geometry and the Wave -- #
    @property
    def depth(self) -> float:
        '''Still water depth in the flume [m].'''
        return self.reefHeight + self.crestDepth

    @property
    def period(self) -> float:
        '''Wave period [s].'''
        return 1.0 / self.frequency

    @property
    def deepWaterWavelength(self) -> float:
        '''Deep-water wavelength g T^2 / (2 pi) [m].'''
        return constants.gravity * self.period**2 / (2.0 * math.pi)

    @property
    def incidentHeight(self) -> float:

        '''

        The height the deep-water wave shoals to at the flume's depth by linear theory, H0 times
        sqrt(cg0 / cg) with the deep-water group speed cg0 = g T / (4 pi) [m].

        '''

        groupSpeed = float(waveTheory.linearGroupSpeed(self.period, self.depth))

        return self.deepWaterHeight * math.sqrt(constants.gravity * self.period / (4.0 * math.pi) / groupSpeed)

    @property
    def wavelength(self) -> float:
        '''Wavelength of the SGN train at the flume's depth [m].'''
        omega = 2.0 * math.pi / self.period
        return 2.0 * math.pi / waveTheory.serreWavenumber(omega, self.depth, alpha = self.alpha)

    @property
    def spongeWidth(self) -> float:
        '''Width of each absorbing layer, one and a half wavelengths [m].'''
        return 1.5 * self.wavelength

    @property
    def sourceX(self) -> float:
        '''Center of the wave source, half a wavelength inside the offshore layer [m].'''
        return self.spongeWidth + 0.5 * self.wavelength

    @property
    def crestX(self) -> float:
        '''The reef crest [m].'''
        return self.sourceX + self.toeDistance + self.reefHeight / self.faceSlope

    def reefBathymetry(self) -> Bathymetry:

        '''

        The flume: a flat floor from the offshore absorbing layer to the reef's toe, the face up to
        the crest, the back down to the floor, and two wavelengths of floor before the shoreward
        layer.

        '''

        toe = self.sourceX + self.toeDistance
        foot = self.crestX + self.reefHeight / self.backSlope
        end = foot + 2.0 * self.wavelength + self.spongeWidth

        return Bathymetry([0.0, toe, self.crestX, foot, end], [0.0, 0.0, self.reefHeight, 0.0, 0.0], self.depth)

    # -- Public Methods -- #
    def runWaveTrain(self, progress: bool = True) -> None:

        '''

        Run the SGN train until it is periodic over the reef, storing its state through the last
        two periods and every start of breaking at the reef.

        '''

        d, T = self.depth, self.period
        self.bathymetry = self.reefBathymetry()
        config = SgnConfig(cellSize = d / self.cellsPerDepth, dryDepth = 1e-5 * d, dispersiveDepth = 0.05 * d,
                           dispersionAlpha = self.alpha, spongeLeft = self.spongeWidth, spongeRight = self.spongeWidth,
                           spongeRate = 4.0 * 2.0 * math.pi / T)
        source = WaveSource(self.incidentHeight, T, d, self.sourceX, alpha = self.alpha)
        self.sgn = sgn = SgnSolver(self.bathymetry, config, source = source)

        x = np.linspace(self.sourceX, self.crestX, 400)
        groupSpeed = waveTheory.linearGroupSpeed(T, np.maximum(self.bathymetry.depth(x[1:]), 0.05 * d))
        arrival = float(np.sum(np.diff(x) / groupSpeed))
        endTime = source.rampPeriods * T + arrival + self.periods * T
        storeFrom = endTime - 2.0 * T
        nextStore = storeFrom
        started = timeModule.perf_counter()
        if progress:
            print(f'ReefOverturn SGN: {sgn.x.size} cells, H = {self.incidentHeight:.4f} m, T = {T:.3f} s, '
                  f'L = {self.wavelength:.3f} m, running to {endTime:.2f} s')

        self.states, self.breakingStarts = [], []
        while sgn.time < endTime:
            sgn.step()
            for frontX, _, _ in sgn.startedFronts:
                if self.crestX - 2.0 * self.reefHeight / self.faceSlope < frontX < self.crestX + self.wavelength:
                    self.breakingStarts.append((sgn.time, frontX))
            if sgn.time >= nextStore:
                self.states.append(SgnState(sgn.time, sgn.surface.copy(), sgn.discharge.copy()))
                nextStore += T / 60.0

        self.wallClockSeconds += timeModule.perf_counter() - started

    def handOff(self) -> None:

        '''

        Hand the wave that breaks last in the stored periods to the boundary element model: the state
        `lead` periods before SGN starts breaking it, in a tank from about a wavelength and a quarter
        offshore of the reef crest to about a wavelength behind its back, each wall where the
        discharge crosses zero and the surface rises slowest. Behind the reef, the bores of waves
        already broken have their fronts smoothed over 0.4 m first. The crest and the jet are
        sought no further than a quarter wavelength behind the reef.

        Raises:
        -------
        RuntimeError : If no wave started breaking at the reef while the state was stored.

        '''

        if not self.states:
            self.runWaveTrain(progress = False)
        stored = (self.states[0].time + self.lead * self.period, self.states[-1].time)
        starts = [time for time, _ in self.breakingStarts if stored[0] <= time <= stored[1]]
        if not starts:
            raise RuntimeError('No wave started breaking at the reef while the SGN state was stored.')
        target = starts[-1] - self.lead * self.period
        self.handoffState = min(self.states, key = lambda state: abs(state.time - target))

        x, L = self.sgn.x, self.wavelength
        state = self.handoffState
        foot = self.crestX + self.reefHeight / self.backSlope
        # Behind the reef lie the bores of waves already broken, whose fronts potential flow cannot
        # start from: they are smoothed, and the shoreward wall stands on the gentle back of the next
        # bore but one, so the bore nearest the reef runs a wavelength before it meets the wall
        surfaceSGN, dischargeSGN = smoothSteepFronts(x, state.surface, state.discharge, foot, 0.4)
        self.tankState = SgnState(state.time, surfaceSGN, dischargeSGN)
        low = dischargeZeroCrossing(x, dischargeSGN, self.crestX - 1.25 * L, 0.75 * L)
        high = dischargeZeroCrossing(x, dischargeSGN, foot + L, 0.5 * L)
        tank = windowBathymetry(self.bathymetry, low, high)
        surface, potential = sgnSurfaceState(x, surfaceSGN, dischargeSGN, tank, 0.1 * self.depth)

        # The crest and the jet are sought from the offshore wall to a quarter wavelength behind the
        # reef, clear of the bores and the shoreward wall they rise against
        self.wave = OverturningWave(depth = self.depth, waveHeight = self.incidentHeight, bathymetry = tank,
                                    beachSlope = self.faceSlope, crestSpacingRatio = self.crestSpacingRatio,
                                    crestWindowRatio = self.crestWindowRatio, frameInterval = self.period / 40.0,
                                    initialState = (surface, potential), crestRange = (low, foot + 0.25 * L))

    def trainCoefficients(self) -> dict:

        '''

        Height, reflection and transmission of the SGN train over its last stored period, measured as
        Blenkinsopp and Chaplin measured them. Ahead of the reef, from a fifth of a wavelength past
        the source to the toe, the first harmonic of the surface is split by least squares into
        incident and reflected waves, a exp(-i k x) + b exp(i k x). The transmitted height is the
        mean crest-to-trough height over the wavelength from half a wavelength behind the reef's
        foot, where the broken waves have reformed.

        Returns:
        --------
        dict : incident height Hi [m], reflection coefficient Kr = Hr / Hi and transmission
            coefficient Kt = Ht / Hi

        Raises:
        -------
        RuntimeError : If the train has not run.

        '''

        if not self.states:
            raise RuntimeError('The SGN train has not run.')
        T, L = self.period, self.wavelength
        omega, k = 2.0 * math.pi / T, 2.0 * math.pi / L
        times = np.array([state.time for state in self.states])
        last = times >= times[-1] - T * (1.0 - 0.5 / 60.0)
        times = times[last]
        elevation = np.array([state.surface for state, keep in zip(self.states, last) if keep]) - self.depth
        x = self.sgn.x

        toe = self.sourceX + self.toeDistance
        ahead = (x > self.sourceX + 0.2 * L) & (x < toe - 0.05 * L)
        # First-harmonic complex amplitude at each cell, then the two waves that best make it up
        harmonic = 2.0 / times.size * np.sum(elevation[:, ahead] * np.exp(-1j * omega * times)[:, None], axis = 0)
        waves = np.column_stack([np.exp(-1j * k * x[ahead]), np.exp(1j * k * x[ahead])])
        (incident, reflected), *_ = np.linalg.lstsq(waves, harmonic, rcond = None)

        foot = self.crestX + self.reefHeight / self.backSlope
        behind = (x > foot + 0.5 * L) & (x < foot + 1.5 * L)
        transmitted = float(np.mean(np.ptp(elevation[:, behind], axis = 0)))
        incidentHeight = 2.0 * float(abs(incident))

        return {'incidentHeight': incidentHeight, 'reflectionCoefficient': float(abs(reflected) / abs(incident)),
                'transmissionCoefficient': transmitted / incidentHeight}

    def run(self, progress: bool = True) -> None:

        '''Run the SGN train, hand off, and run the overturn to the jet's impact.'''

        self.runWaveTrain(progress)
        self.handOff()
        started = timeModule.perf_counter()
        self.wave.run(progress = progress)
        self.wallClockSeconds += timeModule.perf_counter() - started

    def summary(self) -> dict:

        '''

        Blenkinsopp and Chaplin's measures: hc / H0 and H0 / L0; the train's incident height and its
        reflection and transmission coefficients (`trainCoefficients`); at the break point Hb (the
        first trough ahead of the face to the crest), hb (still water depth under the vertical face),
        Hb / hb and Hb / H0; at the plunge point the fitted cavity's lc, wc, lc / wc and Ac / Hb^2;
        and the run's bookkeeping.

        '''

        result = {
            'crestDepthOverHeight': self.crestDepth / self.deepWaterHeight,
            'deepWaterSteepness': self.deepWaterHeight / self.deepWaterWavelength,
            'incidentHeight': self.incidentHeight,
            'wallClockSeconds': self.wallClockSeconds,
        }
        if self.states:
            result.update(self.trainCoefficients())
        wave = self.wave
        if wave is None or wave.breakingPoint is None:
            return result
        point = wave.breakingPoint
        frame = min(wave.frames, key = lambda f: abs(f.time - point.time))
        level = wave.bathymetry.stillWaterLevel
        # The trough just ahead of the face: the first low point of the surface past it
        nodeX = frame.surface[:, 0]
        ahead = frame.surface[(nodeX > point.faceX) & (nodeX < point.faceX + 0.5 * self.wavelength), 1]
        rising = np.flatnonzero(np.diff(ahead) > 0.0)
        trough = float(ahead[rising[0]] if rising.size else ahead[-1]) - level if ahead.size else 0.0
        breakingHeight = point.crestHeight - trough
        result.update({
            'handoffTime': self.handoffState.time,
            'breakingTime': point.time,
            'breakingHeight': breakingHeight,
            'breakingDepth': point.faceDepth,
            'breakerDepthIndex': breakingHeight / point.faceDepth if point.faceDepth > 0.0 else float('inf'),
            'breakerHeightIndex': breakingHeight / self.deepWaterHeight,
            'breakingX': point.faceX,
            'touchedDown': wave.touchedDown,
            'stopReason': wave.stopReason,
        })
        jet = wave.jet if wave.touchedDown else None
        if jet is not None:
            fittedArea = 2.0 * math.sqrt(3.0) / 5.0 * jet.overturnLength * jet.overturnWidth
            result.update({
                'cavityLength': jet.overturnLength,
                'cavityWidth': jet.overturnWidth,
                'cavityRatio': jet.overturnLength / jet.overturnWidth,
                'cavityAreaOverBreakingHeightSquared': fittedArea / breakingHeight**2,
                'enclosedAreaOverBreakingHeightSquared': jet.cavityArea / breakingHeight**2,
                'impactX': float(jet.impactPoint[0]),
            })

        return result
