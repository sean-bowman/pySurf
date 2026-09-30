# -- SPH Time Integration Schemes -- #

'''
Time integration methods for the SPH particle system.

Implements the Symplectic Euler (semi-implicit Euler) integrator,
which is first-order but symplectic -- meaning it conserves energy
over long time integrations, making it ideal for fluid simulation.

The key property of a symplectic integrator is that it preserves
the phase-space volume (Liouville's theorem), preventing artificial
energy drift that plagues standard explicit Euler methods.

References:
-----------
Monaghan (2005) -- Smoothed Particle Hydrodynamics
Hairer et al. (2003) -- Geometric Numerical Integration

Sean Bowman [02/05/2026]
'''

from __future__ import annotations

from typing import Protocol

import numpy as np

from pysurf.WaterSim.sph.particles import ParticleSystem


#--------------------------------------------------------------------#
# -- Time Integrator Protocol -- #
#--------------------------------------------------------------------#

class TimeIntegrator(Protocol):
    '''Protocol for time integration schemes.'''

    def integrate(self, particles: ParticleSystem, dt: float) -> None:
        '''
        Advance fluid particles by one time step.

        Only fluid particles are updated; boundary particles
        remain fixed.

        Parameters:
        -----------
        particles : ParticleSystem
            Particle system to advance
        dt : float
            Time step size [s]
        '''
        ...


#--------------------------------------------------------------------#
# -- Symplectic Euler Integrator -- #
#--------------------------------------------------------------------#

class SymplecticEuler:
    '''
    Symplectic (semi-implicit) Euler integrator.

    Update sequence:
        v(t+dt) = v(t) + a(t) * dt      (kick)
        x(t+dt) = x(t) + v(t+dt) * dt   (drift)

    Note the drift uses the *updated* velocity, which is what makes
    this symplectic. This is sometimes called "kick-drift" or
    "velocity Verlet first half".

    Only fluid particles are integrated; boundary particles
    have zero velocity and fixed positions.
    '''

    def integrate(self, particles: ParticleSystem, dt: float) -> None:
        '''
        Advance fluid particles by one time step.

        Parameters:
        -----------
        particles : ParticleSystem
            Particle system to advance
        dt : float
            Time step size [s]
        '''
        fluid = particles.isFluid

        # Kick: update velocities from accelerations
        particles.velocities[fluid] += particles.accelerations[fluid] * dt

        # Drift: update positions from (new) velocities
        particles.positions[fluid] += particles.velocities[fluid] * dt
