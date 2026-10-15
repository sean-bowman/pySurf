
# -- pySurf: Wave Breaking Detection -- #

'''

Reading wave breaking out of a simulation: a particle cloud, or a free-surface profile eta(x).

A particle cloud has no surface mesh, so its breaking measures are built from the fluid particle
positions directly:

    freeSurfaceProfile      the highest fluid particle in each vertical column: eta(x) while the
                            surface is single-valued
    frontFaceLean           how far the shoreward face of the crest leans forward of vertical
    waterlineX              the landward tip of the water sheet on the beach
    BreakingTracker         a time history of the crest, the breaking point and the runup
    BreakingDetector        the kinematic criterion: particles moving faster than a fraction of
                            the wave speed

A depth-integrated model carries the surface as a single-valued profile, and its measures read that
profile:

    profileCrest            the crest of a profile, refined by a parabola through its top
    waveFront               the front of the wave, from the crest to the trough ahead, and its
                            steepest slope
    SurfaceBreakingTracker  a time history of the crest, the breaking point and the runup

Grilli et al. (1997) place the breaking point where the front face of the wave first turns
vertical. `frontFaceLean` measures that face. The water near the crest is binned into cells one
spacing square; in each horizontal band of cells from just above still water up to the crest, the
face is where the water running shoreward from the crest column ends. The lean is the largest
distance by which a band's face stands shoreward of the face in any band below it. A face sloping
back toward the sea has a negative lean, a vertical face zero, and a face curling over into a lip a
positive one.

The face is the end of the water that is continuous with the crest, not the most shoreward particle
in the band, because particle rows do not stay level: the wave lifts each row most under the crest,
so a deeper row rises into a band near the crest while the band's real face lies far out on the
front. Taking the most shoreward particle would put that band's face at the deeper row's crossing
and read a gently sloping face as leaning forward. Each particle marks every cell its own cell of
water overlaps, half a spacing to each side: rows sit at arbitrary heights against the bands, and a
row straddling a band edge would otherwise scatter between two bands and leave holes in both,
ending a run inside the water.

The tracker records breaking when the lean exceeds one particle spacing over several consecutive
updates, which keeps particle jitter from triggering it. The breaking point it reports follows the
vertical-face instant by the short time the lip takes to grow one spacing, a bias shoreward that
shrinks with resolution.

A particle stands for the cell of water around it, so the free surface sits half a spacing above
the topmost particle centers; elevations reported here include that half spacing. With it, still
water reads at the still water level.

References:
-----------
Grilli, S.T., Svendsen, I.A. & Subramanya, R. (1997) -- Breaking criterion and characteristics
    for solitary waves on slopes, J. Waterway, Port, Coastal, Ocean Eng. 123(3), 102-112.
Perlin, M., Choi, W. & Tian, Z. (2013) -- Breaking waves in deep and intermediate waters, Annu.
    Rev. Fluid Mech. 45, 115-145 (kinematic breaking criteria).

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

The detectors are measurement tools, tested on constructed particle arrangements in
tests/testWaveBreaking.py: a face sloping back has negative lean, a vertical face none, an
overhanging lip a lean equal to its overhang, still water ahead of the wave, a front running past
the search window and deeper rows rising into a band do not read as lean, and the profile and
waterline match the arrangement they were built from. The surface measures are tested on constructed profiles: the crest of a sech^2 profile
between grid points, the highest point of a bore, a front running from crest to trough, a ripple on
the face that does not end it, and a breaking point recorded once. Breaking measured with them is
validated against Grilli et al. (1997) through `pysurf.BreakingWave` and
`pysurf.SurfaceBreakingWave`.

Sean Bowman [10/11/2026]

'''

# Standard library imports
from __future__ import annotations

from dataclasses import dataclass, field

# Third-party imports
import numpy as np

# Local imports
from pysurf.bathymetry import Bathymetry
from pysurf.sphParticles import ParticleSystem

#----------------------------------------------------------------------#
# -- Free Surface Measures -- #
#----------------------------------------------------------------------#

def freeSurfaceProfile(positions: np.ndarray, x0: float, x1: float, columnWidth: float) -> tuple[np.ndarray, np.ndarray]:

    '''

    The free-surface elevation in each column: the highest fluid particle plus half a column
    width, the top of the cell that particle stands for.

    Parameters:
    -----------
    positions : np.ndarray
        Fluid particle positions, shape (N, 2) [m]
    x0, x1 : float
        Range of the profile [m]
    columnWidth : float
        Column width, one particle spacing [m]

    Returns:
    --------
    tuple : (column centers, elevation) [m]; elevation is NaN in empty columns

    '''

    count = max(1, int(np.ceil((x1 - x0) / columnWidth)))
    centers = x0 + (np.arange(count) + 0.5) * columnWidth
    elevation = np.full(count, np.nan)

    columns = np.floor((positions[:, 0] - x0) / columnWidth).astype(np.int64)
    keep = (columns >= 0) & (columns < count)
    if np.any(keep):
        top = np.full(count, -np.inf)
        np.maximum.at(top, columns[keep], positions[keep, 1])
        elevation = np.where(np.isfinite(top), top + 0.5 * columnWidth, np.nan)

    return centers, elevation

def frontFaceLean(positions: np.ndarray, crestX: float, stillWaterLevel: float, spacing: float,
                  reach: float, shoreLimit: float = np.inf) -> float:

    '''

    How far the shoreward face of a crest leans forward of vertical [m].

    The water from half a spacing behind the crest to `reach` shoreward of it, and more than half a
    spacing above still water, is binned into cells one spacing square; each particle marks the
    cells its own cell (half a spacing to each side) overlaps. In each horizontal band the face is
    the shoreward edge of the run of marked cells that starts at the crest column. The lean is the largest amount by
    which a band's face stands shoreward of the face in any band below it: negative for a face
    sloping back, zero for a vertical face, positive for a lip curling over.

    Only water above still water is used, so the undisturbed water ahead of the wave cannot be
    mistaken for the face. A band whose run reaches the end of the window is left out: its face lies
    beyond the reach, low on a long gentle front.

    Parameters:
    -----------
    positions : np.ndarray
        Fluid particle positions, shape (N, 2) [m]
    crestX : float
        Crest position [m]
    stillWaterLevel : float
        Still water elevation [m]
    spacing : float
        Particle spacing [m]
    reach : float
        How far shoreward of the crest the face may lie [m]
    shoreLimit : float
        Particles shoreward of this are ignored, so a runup sheet on the beach is not the face [m]

    Returns:
    --------
    float : The lean [m]; -inf when fewer than two bands have a face inside the window

    '''

    windowStart = crestX - 0.5 * spacing
    windowEnd = min(crestX + reach, shoreLimit)
    base = stillWaterLevel + 0.5 * spacing
    x, z = positions[:, 0], positions[:, 1]
    window = (x >= windowStart - 0.5 * spacing) & (x < windowEnd + 0.5 * spacing) & (z > base - 0.5 * spacing)
    if not np.any(window):
        return -np.inf

    # Lower-left cell each particle's own cell overlaps; it overlaps that cell and the next one up
    # and shoreward
    firstColumn = np.floor((x[window] - 0.5 * spacing - windowStart) / spacing).astype(np.int64)
    firstBand = np.floor((z[window] - 0.5 * spacing - base) / spacing).astype(np.int64)
    nColumns = max(1, int(np.ceil((windowEnd - windowStart) / spacing)))
    occupied = np.zeros((int(firstBand.max()) + 2, nColumns), dtype = bool)
    for columnStep in (0, 1):
        for bandStep in (0, 1):
            column, band = firstColumn + columnStep, firstBand + bandStep
            inside = (column >= 0) & (column < nColumns) & (band >= 0)
            occupied[band[inside], column[inside]] = True

    faces = []
    for row in occupied:
        # The run must start at the crest; a band with no water there is above it
        start = np.flatnonzero(row[:2])
        if start.size == 0:
            continue
        rest = row[start[0]:]
        if np.all(rest):
            continue
        faces.append(windowStart + (start[0] + int(np.argmin(rest))) * spacing)
    if len(faces) < 2:
        return -np.inf

    faces = np.asarray(faces)
    lowestBelow = np.minimum.accumulate(faces)

    return float(np.max(faces[1:] - lowestBelow[:-1]))

def waterlineX(positions: np.ndarray, bathymetry: Bathymetry, wetHeight: float, minimumParticles: int = 3) -> float:

    '''

    The landward tip of the water sheet on the bed [m].

    The tip is taken among particles within `wetHeight` of the bed, and is the `minimumParticles`-th
    most landward of them, so a lone droplet thrown ahead of the sheet does not set it.

    '''

    nearBed = positions[positions[:, 1] - bathymetry.elevation(positions[:, 0]) < wetHeight]
    if len(nearBed) < minimumParticles:
        return float('nan')

    return float(np.sort(nearBed[:, 0])[-minimumParticles])

#----------------------------------------------------------------------#
# -- Breaking History -- #
#----------------------------------------------------------------------#

@dataclass
class BreakingPoint:

    '''

    Where the wave breaks: its front face first leaning forward of vertical by a spacing, at the
    first of the consecutive updates that confirmed it.

    Parameters:
    -----------
    time : float
        Simulated time [s]
    lean : float
        Front face lean when breaking was recorded [m]
    crestX : float
        Crest position at that instant [m]
    crestHeight : float
        Crest elevation above still water, Hb [m]
    depth : float
        Still water depth under the crest, hb [m]

    '''

    time: float
    lean: float
    crestX: float
    crestHeight: float
    depth: float

    @property
    def breakingIndex(self) -> float:
        '''Hb / hb [-].'''
        return self.crestHeight / self.depth if self.depth > 0.0 else float('inf')

@dataclass
class BreakingTracker:

    '''

    Time history of a wave approaching a beach: crest position and height, the breaking point,
    and the highest runup.

    Parameters:
    -----------
    bathymetry : Bathymetry
        The bed and its still water level
    spacing : float
        Particle spacing; sets the column width, the face bands and the breaking threshold [m]
    leanThreshold : float
        Front face lean that counts as breaking, in spacings; the lean comes in whole spacings, so
        1 means a lean of at least one
    faceReach : float
        How far shoreward of the crest the face may lie [m]
    searchFromX : float
        A crest offshore of this position cannot register as breaking, so the start-up of the
        simulation cannot either [m]
    persistence : int
        Consecutive updates the lean must stay above its threshold to count as breaking

    '''

    bathymetry: Bathymetry
    spacing: float
    leanThreshold: float = 1.0
    faceReach: float = 1.0
    searchFromX: float = -np.inf
    persistence: int = 3
    times: list = field(default_factory = list)
    crestX: list = field(default_factory = list)
    crestHeight: list = field(default_factory = list)
    lean: list = field(default_factory = list)
    waterline: list = field(default_factory = list)
    breakingPoint: BreakingPoint | None = None
    maxRunup: float = float('nan')
    _candidate: BreakingPoint | None = field(default = None, repr = False)
    _streak: int = field(default = 0, repr = False)

    def update(self, time: float, positions: np.ndarray) -> None:

        '''

        Record the crest, the face lean and the waterline at `time`, and the breaking point once
        the lean has stayed above its threshold, with the crest past `searchFromX`, for
        `persistence` consecutive updates.

        Parameters:
        -----------
        time : float
            Simulated time [s]
        positions : np.ndarray
            Fluid particle positions, shape (N, 2) [m]

        '''

        bathymetry, s = self.bathymetry, self.spacing
        swl = bathymetry.stillWaterLevel

        centers, elevation = freeSurfaceProfile(positions, bathymetry.startX, bathymetry.endX, s)
        crestIndex = int(np.nanargmax(elevation))
        crestX, crestHeight = float(centers[crestIndex]), float(elevation[crestIndex] - swl)

        lean = frontFaceLean(positions, crestX, swl, s, self.faceReach)

        tip = waterlineX(positions, bathymetry, 2.0 * s)
        if np.isfinite(tip):
            runup = float(bathymetry.elevation(tip) - swl)
            self.maxRunup = runup if not np.isfinite(self.maxRunup) else max(self.maxRunup, runup)

        self.times.append(time)
        self.crestX.append(crestX)
        self.crestHeight.append(crestHeight)
        self.lean.append(lean)
        self.waterline.append(tip)

        if self.breakingPoint is not None:
            return
        # The lean is measured in whole cells; the half-cell margin keeps rounding from deciding a tie
        if crestX >= self.searchFromX and lean > (self.leanThreshold - 0.5) * s:
            if self._streak == 0:
                self._candidate = BreakingPoint(time = time, lean = lean, crestX = crestX, crestHeight = crestHeight,
                                                depth = float(bathymetry.depth(crestX)))
            self._streak += 1
            if self._streak >= self.persistence:
                self.breakingPoint = self._candidate
        else:
            self._streak, self._candidate = 0, None

#----------------------------------------------------------------------#
# -- Surface Profile Measures -- #
#----------------------------------------------------------------------#

def profileCrest(x: np.ndarray, eta: np.ndarray, fraction: float = 0.7) -> float:

    '''

    The crest of a surface profile: the vertex of a parabola fitted through the points above
    `fraction` of the peak elevation, contiguous with the peak.

    A sech^2 crest is parabolic near its top, so the fit locates it to a small part of a grid cell
    or particle spacing, and spreads the noise of any one point over all the points it uses. With
    fewer than three points above the threshold, or a vertex outside the points fitted (the flat top
    and cliff of a bore are not a parabola), the highest point is returned.

    Parameters:
    -----------
    x : np.ndarray
        Positions [m]
    eta : np.ndarray
        Elevation above still water at each position [m]; NaN where there is no water
    fraction : float
        Part of the peak elevation the fit starts from

    Returns:
    --------
    float : The crest position [m]

    '''

    eta = np.nan_to_num(np.asarray(eta, dtype = float), nan = -np.inf)
    peak = int(np.argmax(eta))
    threshold = fraction * eta[peak]
    low, high = peak, peak
    while low > 0 and eta[low - 1] > threshold:
        low -= 1
    while high < len(eta) - 1 and eta[high + 1] > threshold:
        high += 1
    if high - low < 2:
        return float(x[peak])

    a, b, _ = np.polyfit(x[low:high + 1], eta[low:high + 1], 2)
    vertex = -b / (2.0 * a) if a < 0.0 else np.nan

    return float(vertex) if x[low] <= vertex <= x[high] else float(x[peak])

@dataclass
class WaveFront:

    '''

    The front of a wave on a surface profile: the part between the crest and the trough ahead of it
    in the direction of travel, here toward +x.

    Parameters:
    -----------
    crestIndex, troughIndex : int
        Profile points at the crest and at the trough, or the last wet point, ahead of it
    steepestIndex : int
        Profile point where the front is steepest
    slope : float
        Steepest downward slope of the front, -d eta / dx [-]

    '''

    crestIndex: int
    troughIndex: int
    steepestIndex: int
    slope: float

    @property
    def angle(self) -> float:
        '''Steepest front slope as an angle from horizontal [deg].'''
        return float(np.degrees(np.arctan(max(self.slope, 0.0))))

def waveFront(x: np.ndarray, surface: np.ndarray, wet: np.ndarray, rippleShare: float = 0.1,
              crestIndex: int | None = None, minimumRise: float = 0.0) -> WaveFront | None:

    '''

    The front of the highest wave on a surface profile.

    The crest is the highest wet point. The front runs shoreward from it until the surface climbs
    back from its lowest level so far by more than `rippleShare` of the drop from the crest (and
    more than `minimumRise`, so ripples on a flat crest do not end it), or until the waterline. A ripple on the face does not end it: taken as the trough, the depths
    either side of it are nearly equal and a breaking bore would read as spent. The trough is the
    foot of the front, the first point within 2 percent of the drop of the lowest level, so still
    water ahead of a wave does not move it. The slope is the central difference of the surface.

    Parameters:
    -----------
    x : np.ndarray
        Uniformly spaced positions [m]
    surface : np.ndarray
        Free-surface elevation [m]
    wet : np.ndarray
        Boolean mask of the points that hold water
    rippleShare : float
        Rise, as a share of the drop from the crest, that ends the front
    crestIndex : int | None
        Crest to start from; the highest wet point by default
    minimumRise : float
        Smallest climb back that ends the front [m]

    Returns:
    --------
    WaveFront | None : The front, or None when there is no water or no front ahead of the crest

    '''

    if not np.any(wet):
        return None

    crest = int(np.argmax(np.where(wet, surface, -np.inf))) if crestIndex is None else int(crestIndex)
    ahead = np.arange(crest + 1, len(surface))
    if ahead.size == 0:
        return None
    # The front ends at the waterline, or where the surface climbs back from its lowest point so
    # far by more than the ripple share of the drop from the crest
    lowest = np.minimum.accumulate(surface[ahead])
    rise = surface[ahead] - lowest
    stops = (~wet[ahead]) | (rise > np.maximum(rippleShare * (surface[crest] - lowest), minimumRise))
    end = int(np.argmax(stops)) if np.any(stops) else ahead.size
    if end == 0:
        return None
    segment = surface[ahead[:end]]
    lowest = float(np.min(segment))
    foot = np.flatnonzero(segment <= lowest + 0.02 * (surface[crest] - lowest))[0]
    trough = int(ahead[foot])
    if trough - crest < 2:
        return None

    dx = x[1] - x[0]
    interior = np.arange(crest + 1, trough)
    slopes = -(surface[interior + 1] - surface[interior - 1]) / (2.0 * dx)
    steepest = int(interior[np.argmax(slopes)])

    return WaveFront(crestIndex = crest, troughIndex = trough, steepestIndex = steepest, slope = float(np.max(slopes)))

@dataclass
class SurfaceBreakingPoint:

    '''

    Where a wave on a surface model breaks: its front first steeper than the breaking angle.

    Parameters:
    -----------
    time : float
        Simulated time [s]
    frontAngle : float
        Steepest front slope when breaking started [deg]
    crestX : float
        Crest position at that instant [m]
    crestHeight : float
        Crest elevation above still water, Hb [m]
    depth : float
        Still water depth under the crest, hb [m]

    '''

    time: float
    frontAngle: float
    crestX: float
    crestHeight: float
    depth: float

    @property
    def breakingIndex(self) -> float:
        '''Hb / hb [-].'''
        return self.crestHeight / self.depth if self.depth > 0.0 else float('inf')

@dataclass
class SurfaceBreakingTracker:

    '''

    Time history of a wave on a surface model: crest position and height, front angle, waterline,
    the breaking point and the highest runup.

    Parameters:
    -----------
    bathymetry : Bathymetry
        The bed and its still water level
    searchFromX : float
        A crest offshore of this position cannot register as breaking [m]

    '''

    bathymetry: Bathymetry
    searchFromX: float = -np.inf
    times: list = field(default_factory = list)
    crestX: list = field(default_factory = list)
    crestHeight: list = field(default_factory = list)
    frontAngle: list = field(default_factory = list)
    waterline: list = field(default_factory = list)
    breakingPoint: SurfaceBreakingPoint | None = None
    maxRunup: float = float('nan')

    def update(self, time: float, x: np.ndarray, surface: np.ndarray, wet: np.ndarray, breaking: bool) -> None:

        '''

        Record the crest, the front angle and the waterline at `time`, and the breaking point the
        first time `breaking` is reported with the crest past `searchFromX`.

        Parameters:
        -----------
        time : float
            Simulated time [s]
        x : np.ndarray
            Uniformly spaced positions [m]
        surface : np.ndarray
            Free-surface elevation [m]
        wet : np.ndarray
            Boolean mask of the points that hold water
        breaking : bool
            Whether the model is treating the wave as breaking

        '''

        swl = self.bathymetry.stillWaterLevel
        eta = np.where(wet, surface - swl, np.nan)
        crestX = profileCrest(x, eta)
        crestHeight = float(np.nanmax(eta))
        front = waveFront(x, surface, wet)
        angle = front.angle if front is not None else 0.0

        # The waterline is the shoreward edge of the last wet cell
        lastWet = int(np.flatnonzero(wet)[-1])
        tip = float(x[lastWet] + 0.5 * (x[1] - x[0]))
        runup = float(self.bathymetry.elevation(tip) - swl)
        self.maxRunup = runup if not np.isfinite(self.maxRunup) else max(self.maxRunup, runup)

        self.times.append(time)
        self.crestX.append(crestX)
        self.crestHeight.append(crestHeight)
        self.frontAngle.append(angle)
        self.waterline.append(tip)

        if self.breakingPoint is None and breaking and crestX >= self.searchFromX:
            self.breakingPoint = SurfaceBreakingPoint(time = time, frontAngle = angle, crestX = crestX,
                                                      crestHeight = crestHeight,
                                                      depth = float(self.bathymetry.depth(crestX)))

#----------------------------------------------------------------------#
# -- Kinematic Breaking Detector -- #
#----------------------------------------------------------------------#

@dataclass
class BreakingEvent:

    '''

    Particles exceeding the kinematic breaking threshold at one instant.

    Parameters:
    -----------
    position : np.ndarray
        Centroid of the fast particles [m]
    time : float
        Simulated time [s]
    breakerType : str
        Breaker type expected for the incident wave and slope
    intensity : float
        Mean horizontal particle speed over the wave speed [-]
    nParticles : int
        Particles in the event

    '''

    position: np.ndarray
    time: float
    breakerType: str
    intensity: float
    nParticles: int = 1

class BreakingDetector:

    '''

    The kinematic breaking criterion: a wave crest breaks when its particles outrun the wave,
    u > threshold * c, with the threshold typically 0.85 to 1.

    The breaker type is a property of the incident wave and the slope, not of the particle
    velocities, so it is supplied by the caller (from `waveTheory.iribarrenBreakerType` or
    `waveTheory.solitaryBreakerType`) and attached to every event.

    Parameters:
    -----------
    waveSpeed : float
        Wave phase speed c [m/s]
    velocityThreshold : float
        Fraction of c that counts as breaking
    breakerType : str
        Breaker type expected for this wave on this slope
    clusterRadius : float
        Fast particles further apart than about three times this split into two events [m]
    cooldownTime : float
        Shortest time between events in the same neighborhood [s]

    '''

    def __init__(self, waveSpeed: float, velocityThreshold: float = 0.85, breakerType: str = 'unknown',
                 clusterRadius: float = 0.1, cooldownTime: float = 0.5) -> None:

        self.waveSpeed = waveSpeed
        self._velocityThreshold = velocityThreshold
        self._breakerType = breakerType
        self._clusterRadius = clusterRadius
        self._cooldownTime = cooldownTime
        self._events: list[BreakingEvent] = []
        self._lastEventTime: dict[tuple, float] = {}

    def _clusters(self, positions: np.ndarray, speeds: np.ndarray) -> list[tuple[np.ndarray, float, int]]:

        '''

        One cluster when the fast particles lie within three cluster radii of their centroid,
        otherwise two split at the median x.

        '''

        centroid = np.mean(positions, axis = 0)
        if len(positions) == 1 or np.max(np.linalg.norm(positions - centroid, axis = 1)) < 3.0 * self._clusterRadius:
            return [(centroid, float(np.mean(speeds)), len(positions))]

        left = positions[:, 0] < np.median(positions[:, 0])
        return [(np.mean(positions[mask], axis = 0), float(np.mean(speeds[mask])), int(np.sum(mask)))
                for mask in (left, ~left) if np.any(mask)]

    def detectBreaking(self, particles: ParticleSystem, time: float) -> list[BreakingEvent]:

        '''

        Events for the fluid particles currently faster than the threshold.

        Returns:
        --------
        list[BreakingEvent] : Events detected at this instant

        '''

        fluid = particles.isFluid
        horizontal = particles.velocities[fluid, 0]
        fast = horizontal > self._velocityThreshold * self.waveSpeed
        if not np.any(fast):
            return []

        events = []
        for centroid, meanSpeed, count in self._clusters(particles.positions[fluid][fast], horizontal[fast]):
            cell = tuple(np.floor(centroid / (2.0 * self._clusterRadius)).astype(int))
            if time - self._lastEventTime.get(cell, -np.inf) < self._cooldownTime:
                continue
            event = BreakingEvent(position = centroid, time = time, breakerType = self._breakerType,
                                  intensity = meanSpeed / self.waveSpeed, nParticles = count)
            events.append(event)
            self._events.append(event)
            self._lastEventTime[cell] = time

        return events

    @property
    def allEvents(self) -> list[BreakingEvent]:
        '''Every event detected so far.'''
        return list(self._events)

    @property
    def eventCount(self) -> int:
        '''Number of events detected so far.'''
        return len(self._events)

    def getStatistics(self) -> dict:

        '''Event count, mean and largest intensity, and the first event's time and position.'''

        if not self._events:
            return {'totalEvents': 0, 'breakerType': self._breakerType, 'averageIntensity': 0.0,
                    'firstBreakingTime': None, 'firstBreakingPosition': None}

        first = min(self._events, key = lambda event: event.time)
        intensities = [event.intensity for event in self._events]

        return {
            'totalEvents': len(self._events),
            'breakerType': self._breakerType,
            'averageIntensity': float(np.mean(intensities)),
            'maxIntensity': float(np.max(intensities)),
            'firstBreakingTime': first.time,
            'firstBreakingPosition': first.position.tolist(),
        }
