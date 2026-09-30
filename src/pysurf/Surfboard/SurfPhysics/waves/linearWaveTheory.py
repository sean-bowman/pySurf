# -- Linear (Airy) Wave Theory -- #

'''

Linear wave theory implementation following Dean & Dalrymple.

Implements the WaveModel protocol using small-amplitude (Airy) wave theory.
Valid for H/L << 1 (small steepness) and H/d << 1 (small amplitude relative
to depth).

Key equations:
- Dispersion relation: omega^2 = g * k * tanh(k * d)
- Phase speed: c = omega / k
- Group velocity: cg = c/2 * (1 + 2kd / sinh(2kd))
- Surface elevation: eta = (H/2) * cos(kx - wt)
- Velocity: u, w from cosh/sinh profiles
- Energy: E = (1/8) * rho * g * H^2
- Breaking: H/d > 0.78 or H/L > 1/7

References:
-----------
Dean, R.G. & Dalrymple, R.A. -- Water Wave Mechanics for Engineers and Scientists
Airy wave theory: https://en.wikipedia.org/wiki/Airy_wave_theory
NTNU course notes: https://folk.ntnu.no/oivarn/hercules_ntnu/LWTcourse/lwt_new_2000_Part_A.pdf

Sean Bowman [02/03/2026]

'''

from __future__ import annotations

import math

from pysurf.Surfboard.SurfPhysics import constants as c
from pysurf.Surfboard.SurfPhysics.waves.waveConditions import WaveConditions


class LinearWaveTheory:

    '''

    Linear (Airy) wave theory model.

    Provides wave kinematics, dynamics, energy, and breaking criteria
    based on small-amplitude wave theory. Satisfies the WaveModel protocol.

    '''

    #--------------------------------------------------------------------#
    # -- Dispersion Relation -- #
    #--------------------------------------------------------------------#

    def solveDispersionRelation(self, omega: float, depth: float) -> float:

        '''

        Solve omega^2 = g * k * tanh(k * d) for wavenumber k.

        Uses Newton-Raphson iteration with deep-water initial guess.
        Converges to machine precision in 5-10 iterations.

        Parameters:
        -----------
        omega : float
            Angular frequency [rad/s]
        depth : float
            Water depth [m]

        Returns:
        --------
        float : Wavenumber k [rad/m]

        '''

        g = c.gravity

        # Deep-water initial guess: k0 = omega^2 / g
        k = omega * omega / g

        # Newton-Raphson iteration
        # f(k) = omega^2 - g*k*tanh(k*d)
        # f'(k) = -g*(tanh(k*d) + k*d*(1 - tanh^2(k*d)))
        for _ in range(50):
            tanhKd = math.tanh(k * depth)
            f = omega * omega - g * k * tanhKd
            fPrime = -g * (tanhKd + k * depth * (1.0 - tanhKd * tanhKd))

            if abs(fPrime) < 1e-30:
                break

            dk = -f / fPrime
            k += dk

            if abs(dk) < 1e-12 * abs(k):
                break

        return max(k, 1e-12)

    #--------------------------------------------------------------------#
    # -- Wave Properties -- #
    #--------------------------------------------------------------------#

    def waveLength(self, waveConditions: WaveConditions) -> float:

        '''

        Wavelength L = 2*pi/k [m].

        Parameters:
        -----------
        waveConditions : WaveConditions
            Wave state

        Returns:
        --------
        float : Wavelength in meters

        '''

        omega = waveConditions.angularFrequency
        k = self.solveDispersionRelation(omega, waveConditions.depth)
        return 2.0 * math.pi / k

    def waveSpeed(self, waveConditions: WaveConditions) -> float:

        '''

        Phase speed c = omega/k [m/s].

        Parameters:
        -----------
        waveConditions : WaveConditions
            Wave state

        Returns:
        --------
        float : Phase speed in m/s

        '''

        omega = waveConditions.angularFrequency
        k = self.solveDispersionRelation(omega, waveConditions.depth)
        return omega / k

    def groupSpeed(self, waveConditions: WaveConditions) -> float:

        '''

        Group velocity cg = c/2 * (1 + 2kd/sinh(2kd)) [m/s].

        In deep water: cg = c/2
        In shallow water: cg = c = sqrt(g*d)

        Parameters:
        -----------
        waveConditions : WaveConditions
            Wave state

        Returns:
        --------
        float : Group velocity in m/s

        '''

        omega = waveConditions.angularFrequency
        d = waveConditions.depth
        k = self.solveDispersionRelation(omega, d)
        phaseSpeed = omega / k

        # n = cg/c = 0.5 * (1 + 2kd/sinh(2kd))
        twoKd = 2.0 * k * d
        if twoKd > 50.0:
            # Deep water limit: sinh(2kd) -> inf, so n -> 0.5
            n = 0.5
        else:
            n = 0.5 * (1.0 + twoKd / math.sinh(twoKd))

        return n * phaseSpeed

    #--------------------------------------------------------------------#
    # -- Kinematics -- #
    #--------------------------------------------------------------------#

    def surfaceElevation(
        self, x: float, t: float, waveConditions: WaveConditions
    ) -> float:

        '''

        Surface elevation eta(x, t) = (H/2) * cos(kx - omega*t) [m].

        Parameters:
        -----------
        x : float
            Horizontal position [m]
        t : float
            Time [s]
        waveConditions : WaveConditions
            Wave state

        Returns:
        --------
        float : Surface elevation in meters

        '''

        omega = waveConditions.angularFrequency
        k = self.solveDispersionRelation(omega, waveConditions.depth)
        return waveConditions.amplitude * math.cos(k * x - omega * t)

    def velocityField(
        self, x: float, z: float, t: float, waveConditions: WaveConditions
    ) -> tuple[float, float]:

        '''

        Velocity components (u, w) at a point under the wave.

        u = (H/2)*omega * cosh(k*(z+d))/sinh(k*d) * cos(kx - wt)
        w = (H/2)*omega * sinh(k*(z+d))/sinh(k*d) * sin(kx - wt)

        Parameters:
        -----------
        x : float
            Horizontal position [m]
        z : float
            Vertical position [m] (negative below still water level)
        t : float
            Time [s]
        waveConditions : WaveConditions
            Wave state

        Returns:
        --------
        tuple[float, float] : (u, w) velocity components in m/s

        '''

        omega = waveConditions.angularFrequency
        d = waveConditions.depth
        k = self.solveDispersionRelation(omega, d)
        a = waveConditions.amplitude

        phase = k * x - omega * t
        sinhKd = math.sinh(k * d)

        if abs(sinhKd) < 1e-30:
            return (0.0, 0.0)

        # Clamp z to valid range [-d, 0]
        zClamped = max(-d, min(0.0, z))

        u = a * omega * math.cosh(k * (zClamped + d)) / sinhKd * math.cos(phase)
        w = a * omega * math.sinh(k * (zClamped + d)) / sinhKd * math.sin(phase)

        return (u, w)

    def accelerationField(
        self, x: float, z: float, t: float, waveConditions: WaveConditions
    ) -> tuple[float, float]:

        '''

        Acceleration components (du/dt, dw/dt) at a point under the wave.

        Parameters:
        -----------
        x : float
            Horizontal position [m]
        z : float
            Vertical position [m]
        t : float
            Time [s]
        waveConditions : WaveConditions
            Wave state

        Returns:
        --------
        tuple[float, float] : (ax, az) acceleration components in m/s^2

        '''

        omega = waveConditions.angularFrequency
        d = waveConditions.depth
        k = self.solveDispersionRelation(omega, d)
        a = waveConditions.amplitude

        phase = k * x - omega * t
        sinhKd = math.sinh(k * d)

        if abs(sinhKd) < 1e-30:
            return (0.0, 0.0)

        zClamped = max(-d, min(0.0, z))

        # Time derivative of velocity (multiply by omega, swap trig)
        ax = a * omega * omega * math.cosh(k * (zClamped + d)) / sinhKd * math.sin(phase)
        az = -a * omega * omega * math.sinh(k * (zClamped + d)) / sinhKd * math.cos(phase)

        return (ax, az)

    def pressure(
        self, x: float, z: float, t: float, waveConditions: WaveConditions
    ) -> float:

        '''

        Total pressure at a point under the wave.

        p = -rho*g*z + rho*g*(H/2) * cosh(k*(z+d))/cosh(k*d) * cos(kx - wt)

        The first term is hydrostatic, the second is dynamic wave pressure.

        Parameters:
        -----------
        x : float
            Horizontal position [m]
        z : float
            Vertical position [m] (negative below still water level)
        t : float
            Time [s]
        waveConditions : WaveConditions
            Wave state

        Returns:
        --------
        float : Pressure in Pa

        '''

        omega = waveConditions.angularFrequency
        d = waveConditions.depth
        k = self.solveDispersionRelation(omega, d)
        a = waveConditions.amplitude
        rho = c.seawaterDensity
        g = c.gravity

        zClamped = max(-d, min(0.0, z))
        phase = k * x - omega * t
        coshKd = math.cosh(k * d)

        if abs(coshKd) < 1e-30:
            return -rho * g * zClamped

        # Hydrostatic + dynamic
        pHydrostatic = -rho * g * zClamped
        pDynamic = rho * g * a * math.cosh(k * (zClamped + d)) / coshKd * math.cos(phase)

        return pHydrostatic + pDynamic

    #--------------------------------------------------------------------#
    # -- Energy -- #
    #--------------------------------------------------------------------#

    def energyDensity(self, waveConditions: WaveConditions) -> float:

        '''

        Wave energy per unit surface area: E = (1/8) * rho * g * H^2 [J/m^2].

        Parameters:
        -----------
        waveConditions : WaveConditions
            Wave state

        Returns:
        --------
        float : Energy density in J/m^2

        '''

        rho = c.seawaterDensity
        g = c.gravity
        h = waveConditions.height
        return (1.0 / 8.0) * rho * g * h * h

    def energyFlux(self, waveConditions: WaveConditions) -> float:

        '''

        Energy flux (power per unit crest width): P = E * cg [W/m].

        Parameters:
        -----------
        waveConditions : WaveConditions
            Wave state

        Returns:
        --------
        float : Energy flux in W/m

        '''

        return self.energyDensity(waveConditions) * self.groupSpeed(waveConditions)

    #--------------------------------------------------------------------#
    # -- Breaking Criteria -- #
    #--------------------------------------------------------------------#

    def isDepthLimitedBreaking(self, waveConditions: WaveConditions) -> bool:

        '''

        Check depth-limited breaking: H/d > 0.78 (McCowan 1894).

        Parameters:
        -----------
        waveConditions : WaveConditions
            Wave state

        Returns:
        --------
        bool : True if depth-limited breaking occurs

        '''

        return waveConditions.height / waveConditions.depth > c.breakingDepthRatio

    def isSteepnessLimitedBreaking(self, waveConditions: WaveConditions) -> bool:

        '''

        Check steepness-limited breaking: H/L > 1/7 (Miche 1944).

        Parameters:
        -----------
        waveConditions : WaveConditions
            Wave state

        Returns:
        --------
        bool : True if steepness-limited breaking occurs

        '''

        wavelength = self.waveLength(waveConditions)
        if wavelength <= 0.0:
            return False
        return waveConditions.height / wavelength > c.breakingSteepnessRatio

    def isBroken(self, waveConditions: WaveConditions) -> bool:

        '''

        Check if the wave has broken (either depth or steepness limited).

        Parameters:
        -----------
        waveConditions : WaveConditions
            Wave state

        Returns:
        --------
        bool : True if the wave has broken

        '''

        return (
            self.isDepthLimitedBreaking(waveConditions)
            or self.isSteepnessLimitedBreaking(waveConditions)
        )

    #--------------------------------------------------------------------#
    # -- Depth Classification -- #
    #--------------------------------------------------------------------#

    def depthClassification(self, waveConditions: WaveConditions) -> str:

        '''

        Classify the water depth relative to wavelength.

        Deep water: d/L > 0.5 (tanh(kd) ~ 1, waves don't feel the bottom)
        Shallow water: d/L < 0.05 (tanh(kd) ~ kd, non-dispersive)
        Intermediate: between deep and shallow

        Parameters:
        -----------
        waveConditions : WaveConditions
            Wave state

        Returns:
        --------
        str : 'deep', 'intermediate', or 'shallow'

        '''

        wavelength = self.waveLength(waveConditions)
        if wavelength <= 0.0:
            return 'deep'

        ratio = waveConditions.depth / wavelength

        if ratio > 0.5:
            return 'deep'
        elif ratio < 0.05:
            return 'shallow'
        else:
            return 'intermediate'

    #--------------------------------------------------------------------#
    # -- Surfing-Specific -- #
    #--------------------------------------------------------------------#

    def surfableWaveSpeed(self, waveConditions: WaveConditions) -> float:

        '''

        Speed a surfer must match to ride the wave.

        For a breaking wave, this is approximately the phase speed
        at the breaking depth. Uses the shallow water approximation
        c = sqrt(g*d) at the breaking depth.

        Parameters:
        -----------
        waveConditions : WaveConditions
            Wave state

        Returns:
        --------
        float : Surfable wave speed in m/s

        '''

        g = c.gravity

        # Breaking depth from H/d = 0.78
        breakingDepth = waveConditions.height / c.breakingDepthRatio

        # Shallow water phase speed at breaking
        return math.sqrt(g * breakingDepth)

    def minimumSurfingSpeed(
        self, waveConditions: WaveConditions, boardAngleDeg: float = 30.0
    ) -> float:

        '''

        Minimum paddling speed to catch the wave.

        The surfer must match the wave speed component along their
        direction of travel. For angled takeoffs, this is reduced
        by the cosine of the angle.

        Parameters:
        -----------
        waveConditions : WaveConditions
            Wave state
        boardAngleDeg : float
            Angle between board direction and wave propagation (degrees)

        Returns:
        --------
        float : Minimum speed in m/s

        '''

        waveSpeed = self.surfableWaveSpeed(waveConditions)
        return waveSpeed * math.cos(math.radians(boardAngleDeg))
