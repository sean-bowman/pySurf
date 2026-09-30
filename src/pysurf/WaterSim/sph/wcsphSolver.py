# -- Weakly Compressible SPH Solver -- #

'''

WCSPH solver for incompressible free-surface flows.

Implements the weakly compressible SPH method where pressure is
computed from density via the Tait equation of state, avoiding
the need to solve a Poisson equation. The speed of sound is set
artificially high (~10x max flow velocity) to keep density
variations below ~1%, approximating incompressibility.

All inner loops are vectorized using NumPy for performance.
The solver operates on arrays of neighbor pairs rather than
looping over individual particles.

Algorithm per time step:
    1. Build neighbor list (spatial hash grid)
    2. Compute density by SPH summation
    3. Compute pressure via Tait equation of state
    4. Compute accelerations (pressure gradient + artificial viscosity + gravity)
    5. Compute adaptive time step (CFL condition)
    6. Integrate (Symplectic Euler)
    7. Enforce boundary conditions
    8. (Periodically) Apply XSPH velocity correction
    9. (Periodically) Apply density re-initialization (Shepard filter)

References:
-----------
Monaghan (1994) -- Simulating free surface flows with SPH
Monaghan (1992) -- Smoothed Particle Hydrodynamics
Batchelor (1967) -- An introduction to fluid dynamics
Morris et al. (1997) -- Modeling low Reynolds number incompressible flows

Sean Bowman [02/05/2026]

'''

from __future__ import annotations

import math

import numpy as np

from pysurf.WaterSim import constants as const
from pysurf.WaterSim.sph.protocols import SimulationConfig, SimulationState
from pysurf.WaterSim.sph.kernels import SphKernel, CubicSplineKernel
from pysurf.WaterSim.sph.particles import ParticleSystem
from pysurf.WaterSim.sph.neighborSearch import SpatialHashGrid
from pysurf.WaterSim.sph.boundaryHandling import BoundaryHandler
from pysurf.WaterSim.sph.timeIntegration import SymplecticEuler

class WcsphSolver:

    '''

    Weakly Compressible SPH solver.

    Manages the full time-stepping loop: neighbor search, density
    computation, pressure from equation of state, force computation,
    time integration, and boundary enforcement.

    All particle-pair computations are vectorized using NumPy
    batch operations for acceptable performance in pure Python.

    Parameters:
    -----------
    config : SimulationConfig
        Simulation configuration (domain, resolution, timing)
    kernel : SphKernel | None
        Smoothing kernel (defaults to CubicSplineKernel)
    boundaryHandler : BoundaryHandler | None
        Boundary handler (defaults to None, uses clamping only)

    '''

    def __init__(
        self,
        config: SimulationConfig,
        kernel: SphKernel | None = None,
        boundaryHandler: BoundaryHandler | None = None,
    ) -> None:
        self._config = config
        self._kernel = kernel or CubicSplineKernel(dimensions=config.dimensions)
        self._boundaryHandler = boundaryHandler
        self._integrator = SymplecticEuler()
        self._neighborGrid = SpatialHashGrid(
            cellSize=config.supportRadius,
            dimensions=config.dimensions,
        )

        self._particles: ParticleSystem | None = None
        self._time: float = 0.0
        self._step: int = 0
        self._dt: float = config.maxTimeStep

        # Pre-compute equation of state constant B
        # B = rho_0 * c^2 / gamma
        # where c = speedOfSoundFactor * sqrt(g * domainHeight)
        # Vertical axis: y (index 1) in 2D, z (index 2) in 3D
        verticalAxis = 2 if config.dimensions == 3 else 1
        domainHeight = config.domainSize[verticalAxis]
        gravityMagnitude = abs(config.gravity[verticalAxis])
        self._speedOfSound = const.speedOfSoundFactor * math.sqrt(
            gravityMagnitude * domainHeight
        )
        self._eosB = (
            config.referenceDensity * self._speedOfSound ** 2 / const.gamma
        )

    #--------------------------------------------------------------------#
    # -- Initialization -- #
    #--------------------------------------------------------------------#

    def initialize(self, particles: ParticleSystem) -> None:

        '''

        Set up initial conditions and compute initial density.

        Parameters:
        -----------
        particles : ParticleSystem
            Initial particle system (fluid + boundary particles)

        '''
        self._particles = particles
        self._time = 0.0
        self._step = 0

        # Build initial neighbor list and compute density
        self._neighborGrid.build(particles.positions)
        self._neighborPairs = self._neighborGrid.queryPairs(self._config.supportRadius)
        self._computeDensity()
        self._computePressure()

    #--------------------------------------------------------------------#
    # -- Main Time Step -- #
    #--------------------------------------------------------------------#

    def step(self) -> SimulationState:

        '''

        Advance one time step using the WCSPH algorithm.

        Returns:
        --------
        SimulationState : Simulation state after the step

        '''
        p = self._particles

        # 1. Build neighbor list (reuse pairs for the whole step)
        self._neighborGrid.build(p.positions)
        self._neighborPairs = self._neighborGrid.queryPairs(self._config.supportRadius)

        # 2. Compute density (SPH summation)
        self._computeDensity()

        # 3. Compute pressure (Tait EOS)
        self._computePressure()

        # 4. Compute accelerations (pressure + viscosity + gravity)
        self._computeAccelerations()

        # 5. Compute adaptive time step
        self._dt = self._computeAdaptiveTimeStep()

        # 6. Integrate (Symplectic Euler)
        self._integrator.integrate(p, self._dt)

        # 7. Enforce boundary conditions
        if self._boundaryHandler is not None:
            self._boundaryHandler.enforceBoundary(p)

        # 8. XSPH velocity correction (every step for smooth trajectories)
        self._applyXsphCorrection()

        # 9. Density re-initialization (Shepard filter, periodically)
        if self._step % const.densityFilterInterval == 0 and self._step > 0:
            self._applyDensityFilter()

        self._time += self._dt
        self._step += 1

        return self.currentState

    #--------------------------------------------------------------------#
    # -- Density Computation (Vectorized) -- #
    #--------------------------------------------------------------------#

    def _computeDensity(self) -> None:

        '''

        Compute particle densities using SPH summation.

        rho_i = sum_j m_j * W(|r_i - r_j|, h)

        This includes self-contribution (j = i) via W(0, h).
        Both fluid and boundary particles contribute to density.

        Vectorized: kernel evaluation for all pairs at once,
        then scatter-add using np.add.at.

        '''
        p = self._particles
        h = self._config.smoothingLength
        kernel = self._kernel

        # Start with self-contribution: m_i * W(0, h)
        w0 = kernel.evaluate(0.0, h)
        p.densities[:] = p.masses * w0

        iIdx, jIdx = self._neighborPairs
        if len(iIdx) == 0:
            return

        # Compute distances for all pairs (vectorized)
        dr = p.positions[iIdx] - p.positions[jIdx]
        dist = np.linalg.norm(dr, axis=1)

        # Evaluate kernel for all pairs (vectorized)
        wij = kernel.evaluateBatch(dist, h)

        # Scatter-add density contributions (symmetric)
        np.add.at(p.densities, iIdx, p.masses[jIdx] * wij)
        np.add.at(p.densities, jIdx, p.masses[iIdx] * wij)

    #--------------------------------------------------------------------#
    # -- Pressure (Equation of State) -- #
    #--------------------------------------------------------------------#

    def _computePressure(self) -> None:

        '''

        Compute pressure from density using the Tait equation of state.

        p = B * ((rho / rho_0)^gamma - 1)

        where B = rho_0 * c^2 / gamma.

        The Tait EOS with gamma = 7 enforces near-incompressibility
        by producing large pressure changes for small density deviations.
        Negative pressures are clamped to zero to prevent tensile instability.

        '''
        p = self._particles
        rho0 = self._config.referenceDensity

        # Tait equation of state
        densityRatio = p.densities / rho0
        p.pressures = self._eosB * (densityRatio ** const.gamma - 1.0)

        # Clamp negative pressures to zero (no tensile forces)
        np.maximum(p.pressures, 0.0, out=p.pressures)

    #--------------------------------------------------------------------#
    # -- Acceleration Computation (Vectorized) -- #
    #--------------------------------------------------------------------#

    def _computeAccelerations(self) -> None:

        '''

        Compute particle accelerations from pressure, viscosity, and gravity.

        a_i = -(1/rho_i) * grad(p_i) + viscous_term + g

        Pressure gradient (symmetric form for momentum conservation):
            a_pressure = -sum_j m_j * (p_i/rho_i^2 + p_j/rho_j^2) * grad_W_ij

        Artificial viscosity (Monaghan 1992):
            Pi_ij = { (-alpha * c * mu_ij) / rho_avg,  if v_ij . r_ij < 0
                    { 0,                                 otherwise

            mu_ij = h * (v_ij . r_ij) / (|r_ij|^2 + 0.01 * h^2)

        Vectorized: all pair computations done with NumPy arrays,
        then scatter-added to particle accelerations.

        '''
        p = self._particles
        h = self._config.smoothingLength
        kernel = self._kernel
        gravity = self._config.gravity

        # Reset accelerations: gravity for fluid, zero for boundary
        p.accelerations[:] = 0.0
        p.accelerations[p.isFluid] = gravity

        iIdx, jIdx = self._neighborPairs
        if len(iIdx) == 0:
            return

        # Compute displacement, velocity difference, and distances
        dr = p.positions[iIdx] - p.positions[jIdx]
        dv = p.velocities[iIdx] - p.velocities[jIdx]
        dist = np.linalg.norm(dr, axis=1)

        # Skip zero-distance pairs
        validMask = dist > 1e-12
        if not np.any(validMask):
            return

        # Kernel gradient for all valid pairs (vectorized)
        gradW = kernel.gradientBatch(dr, dist, h)  # shape (nPairs, dim)

        # --- Pressure force (vectorized) --- #
        # -m_j * (p_i/rho_i^2 + p_j/rho_j^2) * grad_W
        pressureTermI = p.pressures[iIdx] / (p.densities[iIdx] ** 2)
        pressureTermJ = p.pressures[jIdx] / (p.densities[jIdx] ** 2)
        pressureCoeff = -(pressureTermI + pressureTermJ)  # shape (nPairs,)

        # --- Artificial viscosity (vectorized) --- #
        # v_ij . r_ij for each pair
        vDotR = np.sum(dv * dr, axis=1)  # shape (nPairs,)

        # eta^2 term to prevent singularity at r = 0
        etaSq = 0.01 * h * h
        distSq = dist * dist

        # mu_ij = h * (v.r) / (|r|^2 + eta^2)
        mu = h * vDotR / (distSq + etaSq)

        # rho_avg for each pair
        rhoAvg = 0.5 * (p.densities[iIdx] + p.densities[jIdx])

        # Pi_ij = -alpha * c * mu / rho_avg (only when v.r < 0)
        piij = np.where(
            vDotR < 0.0,
            (-const.alphaViscosity * self._speedOfSound * mu) / rhoAvg,
            0.0,
        )

        # Total acceleration coefficient per pair:
        # (pressure + viscosity) * m_j * grad_W
        # For particle i: coeff_i * grad_W
        # For particle j: -coeff_j * grad_W (Newton's third law)
        totalCoeff = (pressureCoeff - piij)  # shape (nPairs,)

        # Acceleration contribution for each pair: coeff * m_j * gradW
        # For i: m_j * totalCoeff * gradW
        # For j: -m_i * totalCoeff * gradW (symmetric, with m_i for j)
        accelContribI = (p.masses[jIdx] * totalCoeff)[:, np.newaxis] * gradW
        accelContribJ = (p.masses[iIdx] * totalCoeff)[:, np.newaxis] * gradW

        # Scatter-add to particle accelerations
        # Only add to fluid particles
        fluidI = p.isFluid[iIdx]
        fluidJ = p.isFluid[jIdx]

        # Use np.add.at for scatter accumulation
        if np.any(fluidI):
            np.add.at(p.accelerations, iIdx[fluidI], accelContribI[fluidI])
        if np.any(fluidJ):
            np.add.at(p.accelerations, jIdx[fluidJ], -accelContribJ[fluidJ])

    #--------------------------------------------------------------------#
    # -- Adaptive Time Step -- #
    #--------------------------------------------------------------------#

    def _computeAdaptiveTimeStep(self) -> float:

        '''

        Compute adaptive time step using CFL condition.

        dt = CFL * min(dt_cfl, dt_force)

        dt_cfl = h / (c + max|v|)
        dt_force = sqrt(h / max|a|)

        Returns:
        --------
        float : Adaptive time step [s]

        '''
        h = self._config.smoothingLength
        maxVel = self._particles.maxSpeed()

        # CFL: dt = CFL * h / (c + max|v|)
        dtCfl = const.cflNumber * h / (self._speedOfSound + maxVel)

        # Force-based: dt = CFL * sqrt(h / max|a|)
        fluidAccels = self._particles.accelerations[self._particles.isFluid]
        if len(fluidAccels) > 0:
            maxAccel = float(np.max(np.linalg.norm(fluidAccels, axis=1)))
            if maxAccel > 1e-12:
                dtForce = const.cflNumber * math.sqrt(h / maxAccel)
                dtCfl = min(dtCfl, dtForce)

        # Clamp to maximum allowed time step
        return min(dtCfl, self._config.maxTimeStep)

    #--------------------------------------------------------------------#
    # -- XSPH Velocity Correction (Vectorized) -- #
    #--------------------------------------------------------------------#

    def _applyXsphCorrection(self) -> None:

        '''

        Apply XSPH velocity correction for smoother particle trajectories.

        v_i += epsilon * sum_j (m_j / rho_avg) * (v_j - v_i) * W_ij

        Vectorized using NumPy batch operations.

        '''
        p = self._particles
        h = self._config.smoothingLength
        kernel = self._kernel
        eps = const.xsphEpsilon

        iIdx, jIdx = self._neighborPairs
        if len(iIdx) == 0:
            return

        # Distances and kernel values (vectorized)
        dr = p.positions[iIdx] - p.positions[jIdx]
        dist = np.linalg.norm(dr, axis=1)
        wij = kernel.evaluateBatch(dist, h)

        # rho_avg and weighting factor
        rhoAvg = 0.5 * (p.densities[iIdx] + p.densities[jIdx])
        safeDenom = np.where(rhoAvg > 1e-12, rhoAvg, 1.0)
        weight = (p.masses[jIdx] / safeDenom) * wij  # shape (nPairs,)

        # Velocity difference: v_j - v_i (direction toward neighbor)
        dvJMinusI = p.velocities[jIdx] - p.velocities[iIdx]

        # Contribution: weight * (v_j - v_i)
        contrib = weight[:, np.newaxis] * dvJMinusI  # shape (nPairs, dim)

        # Accumulate corrections
        correction = np.zeros_like(p.velocities)
        np.add.at(correction, iIdx, contrib)
        np.add.at(correction, jIdx, -contrib)

        # Apply only to fluid particles
        p.velocities[p.isFluid] += eps * correction[p.isFluid]

    #--------------------------------------------------------------------#
    # -- Density Re-initialization (Shepard Filter, Vectorized) -- #
    #--------------------------------------------------------------------#

    def _applyDensityFilter(self) -> None:

        '''

        Apply Shepard density re-initialization to correct drift.

        rho_i = sum_j m_j * W_ij / sum_j (m_j / rho_j) * W_ij

        Vectorized using NumPy batch operations.

        '''
        p = self._particles
        h = self._config.smoothingLength
        kernel = self._kernel

        w0 = kernel.evaluate(0.0, h)

        # Self-contributions
        numerator = p.masses * w0
        denominator = (p.masses / p.densities) * w0

        iIdx, jIdx = self._neighborPairs
        if len(iIdx) == 0:
            return

        dr = p.positions[iIdx] - p.positions[jIdx]
        dist = np.linalg.norm(dr, axis=1)
        wij = kernel.evaluateBatch(dist, h)

        # Scatter-add pair contributions
        np.add.at(numerator, iIdx, p.masses[jIdx] * wij)
        np.add.at(numerator, jIdx, p.masses[iIdx] * wij)

        np.add.at(denominator, iIdx, (p.masses[jIdx] / p.densities[jIdx]) * wij)
        np.add.at(denominator, jIdx, (p.masses[iIdx] / p.densities[iIdx]) * wij)

        # Update density only for fluid particles with valid denominator
        fluid = p.isFluid
        fluidIndices = np.where(fluid)[0]
        validDenom = denominator[fluidIndices] > 1e-12
        validIndices = fluidIndices[validDenom]
        p.densities[validIndices] = numerator[validIndices] / denominator[validIndices]

    #--------------------------------------------------------------------#
    # -- Properties -- #
    #--------------------------------------------------------------------#

    @property
    def currentState(self) -> SimulationState:

        '''Current simulation state snapshot.'''
        p = self._particles
        gravMag = abs(self._config.gravity[1])

        return SimulationState(
            time=self._time,
            step=self._step,
            dt=self._dt,
            kineticEnergy=p.kineticEnergy(),
            potentialEnergy=p.potentialEnergy(gravMag),
            maxVelocity=p.maxSpeed(),
            maxDensityError=p.maxDensityError(self._config.referenceDensity),
        )

    @property
    def particles(self) -> ParticleSystem:

        '''Access the particle system.'''
        return self._particles

    @property
    def time(self) -> float:

        '''Current simulation time [s].'''
        return self._time

    @property
    def speedOfSound(self) -> float:

        '''Artificial speed of sound [m/s].'''
        return self._speedOfSound
