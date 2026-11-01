
# -- pySurf: Three-Dimensional Potential Flow from Two-Dimensional Slices -- #

'''

A three-dimensional potential flow assembled from two-dimensional boundary element runs, one per
spanwise station: strip theory. Each slice is an `OverturningWave` on the bed's cross-section at its
station y_k, run on its own; the slices never exchange water or momentum.

    SliceStack      slices at stations y_k across a bed z_b(x, y), run in parallel; at a recorded
                    time the flow at a point (x, y, z) takes u and w from the slices and the spanwise
                    velocity v = dphi/dy from a cubic spline through the slices' potentials at (x, z)

The potentials of different slices share one gauge, which is what makes dphi/dy across them
meaningful: every slice starts from the same solitary wave over the same flat offshore bed, and the
free-surface condition Dphi/Dt = -g eta + |u|^2 / 2 adds no free function of time, so phi at the
seaward wall, where the water stays at rest, keeps its starting value in every slice.

What strip theory leaves out is the spanwise curvature of the flow. Each slice satisfies Laplace's
equation in (x, z); the stacked field has the residual d2phi/dy2, and the bed condition and the
free-surface conditions miss the terms in v. Refraction and focusing by the bed act through exactly
these terms, so a bed whose contours curve (a ridge, a headland) is where strip theory errs most.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

Each slice is a run of `pysurf.OverturningWave`, validated in tests/testValidationOverturning.py.
tests/testValidationRidgeOverturn.py runs the sloping ridge of Guyenne and Grilli (2006): their
two-dimensional breaking points against these slices, and their fully three-dimensional boundary
element results against the stacked flow, which measures the error strip theory makes there. The
slices match the two-dimensional breaking points (the centre slice breaks at t = 7.572 against
7.629 sqrt(h0 / g), its crest 0.680 against 0.680 h0). Against the three-dimensional ridge the
centre slice is within 5 percent in breaking time and height for transverse modulations up to
k = 0.25 / h0 and outside it from 0.34, where the ridge focuses the wave on its centre line; the
largest in-plane speed at z = -0.2 h0 is within 10 percent. The spanwise velocity is not: the
stack gives 0.144 of the streamwise velocity where the three-dimensional flow has 0.085. Strip
theory has no spanwise momentum balance, so nothing resists the slices' potentials drifting apart.

Sean Bowman [10/30/2026]

'''

# Standard library imports
from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor

# Third-party imports
import numpy as np
from scipy.interpolate import CubicSpline

# Local imports
from pysurf.bathymetry import Bathymetry
from pysurf.bemSolver import FlowField
from pysurf.OverturningWave import OverturningWave

# Thread-count variables set to one in the worker processes, so parallel slices do not each start
# a full set of linear algebra threads
threadVariables = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')

def runSlice(bathymetry: Bathymetry, options: dict) -> OverturningWave:

    '''

    One slice's run, in a worker process.

    Parameters:
    -----------
    bathymetry : Bathymetry
        The bed's cross-section at the station
    options : dict
        Keyword arguments for `OverturningWave`

    Returns:
    --------
    OverturningWave : The finished run

    '''

    wave = OverturningWave(bathymetry = bathymetry, **options)
    wave.run(progress = False)

    return wave

def wetStretch(wet: np.ndarray, stations: np.ndarray, y: float, minimum: int = 3) -> slice | None:

    '''

    The longest unbroken run of wet stations that brackets y.

    Parameters:
    -----------
    wet : np.ndarray
        Whether each station's slice holds water at the point, shape (K,)
    stations : np.ndarray
        Station positions [m], increasing, shape (K,)
    y : float
        Spanwise position of the point [m]
    minimum : int
        Fewest stations a run may have

    Returns:
    --------
    slice | None : Indices of the run, or None if no run of `minimum` stations brackets y

    '''

    if y < stations[0] or y > stations[-1]:
        return None
    upper = int(np.searchsorted(stations, y))
    lower = upper if stations[upper] == y else upper - 1
    if not (wet[lower] and wet[upper]):
        return None
    while lower > 0 and wet[lower - 1]:
        lower -= 1
    while upper < len(stations) - 1 and wet[upper + 1]:
        upper += 1

    return slice(lower, upper + 1) if upper - lower + 1 >= minimum else None

def flowAt(wave: OverturningWave, time: float) -> FlowField:

    '''The potential flow inside a slice at one of its recorded times.'''

    frame = wave.frameAt(time)
    if frame is None:
        raise ValueError(f'The slice has no frame at t = {time} s; its run ended at {wave.solver.time} s '
                         f'({wave.stopReason}).')
    solver = wave.solver

    return FlowField(frame.surface, frame.potential, solver.bed, solver.config.surfaceSpacing,
                     solver.stillWaterLevel, solver.config.gravity)

class SliceStack:

    '''

    Two-dimensional boundary element runs stacked across the span.

    Parameters:
    -----------
    section : Callable[[float], Bathymetry]
        The bed's cross-section at a spanwise position y [m]
    stations : Sequence[float]
        Spanwise positions of the slices [m], increasing
    symmetric : bool
        The flow is mirror-symmetric about y = 0 and the stations cover y >= 0 only; the spline
        through the slices is then even in y
    workers : int | None
        Worker processes; by default the PYSURF_WORKERS environment variable, or else one per slice
        up to the processor count. Each worker runs dense linear algebra flat out, so on a machine
        that cannot hold every core at full load the variable caps it.
    waveOptions
        Keyword arguments for every slice's `OverturningWave`: depth, waveHeight, startX,
        recordTimes, endTime and the rest

    '''

    # -- Constructor -- #
    def __init__(self, section: Callable[[float], Bathymetry], stations: Sequence[float], symmetric: bool = False,
                 workers: int | None = None, **waveOptions) -> None:

        self.section = section
        self.stations = np.asarray(stations, dtype = float)     # [m]
        self.symmetric = symmetric
        self.workers = workers
        self.waveOptions = waveOptions
        self.slices: list[OverturningWave] = []
        self._fields: dict[float, list[FlowField]] = {}

        if np.any(np.diff(self.stations) <= 0.0):
            raise ValueError('Slice stations must increase.')
        if symmetric and self.stations[0] < 0.0:
            raise ValueError('A symmetric stack covers y >= 0 only.')

    # -- Public Methods -- #
    def run(self, progress: bool = True) -> None:

        '''Run every slice, in parallel worker processes.'''

        workers = self.workers or int(os.environ.get('PYSURF_WORKERS', 0)) or os.cpu_count() or 1
        workers = min(workers, len(self.stations))
        saved = {name: os.environ.get(name) for name in threadVariables}
        os.environ.update({name: '1' for name in threadVariables})
        try:
            with ProcessPoolExecutor(max_workers = workers) as pool:
                futures = [pool.submit(runSlice, self.section(float(y)), self.waveOptions) for y in self.stations]
                self.slices = []
                for y, future in zip(self.stations, futures):
                    wave = future.result()
                    self.slices.append(wave)
                    if progress:
                        point = wave.breakingPoint
                        broke = f'breaks at t = {point.time:.3f} s' if point is not None else 'does not break'
                        print(f'  slice y = {y:6.3f} m: {broke}, ends at t = {wave.solver.time:.3f} s '
                              f'({wave.stopReason}), {wave.wallClockSeconds:.0f} s')
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        self._fields = {}

    def fields(self, time: float) -> list[FlowField]:

        '''Every slice's flow at one of the recorded times, built once and kept.'''

        if time not in self._fields:
            self._fields[time] = [flowAt(wave, time) for wave in self.slices]

        return self._fields[time]

    def velocity(self, points: np.ndarray, time: float) -> np.ndarray:

        '''

        The stacked flow's velocity at points (x, y, z) at a recorded time.

        u and w at a point are the slices' values at its (x, z), interpolated across the span by a
        cubic spline; v is the spline's derivative of the slices' potentials. Each point's splines
        run through the unbroken stretch of stations around it whose slices hold water at its (x, z),
        so a point on the flank of a ridge whose crest is dry still has a velocity. Points with fewer
        than three such stations around them, or outside the stations' span, are NaN.

        Parameters:
        -----------
        points : np.ndarray
            Points, shape (M, 3) [m]
        time : float
            One of the slices' recorded times [s]

        Returns:
        --------
        np.ndarray : (u, v, w), shape (M, 3) [m/s]

        '''

        points = np.atleast_2d(np.asarray(points, dtype = float))
        planar = points[:, [0, 2]]
        fields = self.fields(time)
        count = len(points)
        potential = np.full((len(fields), count), np.nan)
        horizontal = np.full((len(fields), count), np.nan)
        vertical = np.full((len(fields), count), np.nan)
        for index, field in enumerate(fields):
            wet = field.inside(planar)
            if np.any(wet):
                potential[index, wet] = field.potential(planar[wet])
                speed = field.velocity(planar[wet])
                horizontal[index, wet], vertical[index, wet] = speed[:, 0], speed[:, 1]

        stations = self.stations
        if self.symmetric:
            # Mirror the stations so the spline is even in y; a station at y = 0 appears once
            mirror = np.arange(len(stations) - 1, -1 if stations[0] > 0.0 else 0, -1)
            rows = np.concatenate([mirror, np.arange(len(stations))])
            potential, horizontal, vertical = potential[rows], horizontal[rows], vertical[rows]
            stations = np.concatenate([-stations[mirror], stations])
        y = points[:, 1]

        result = np.full((count, 3), np.nan)
        for index in range(count):
            run = wetStretch(np.isfinite(potential[:, index]), stations, y[index])
            if run is None:
                continue
            span = stations[run]
            result[index, 0] = CubicSpline(span, horizontal[run, index])(y[index])
            result[index, 1] = CubicSpline(span, potential[run, index])(y[index], 1)
            result[index, 2] = CubicSpline(span, vertical[run, index])(y[index])

        return result

    def crestHeights(self, time: float) -> np.ndarray:

        '''

        Each slice's highest surface elevation above still water at `time` [m], from its crest
        track; NaN for a slice whose run ended before it.

        '''

        heights = []
        for wave in self.slices:
            track = wave.tracker
            if not track.times or time > track.times[-1] or time < track.times[0]:
                heights.append(np.nan)
            else:
                heights.append(float(np.interp(time, track.times, track.crestHeight)))

        return np.asarray(heights)
