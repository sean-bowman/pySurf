# -- Physical Constants for SPH Water Simulation -- #

'''
Physical and numerical constants for SPH water simulation.
All values in SI units unless otherwise noted.

References:
-----------
Monaghan (1994) -- Simulating free surface flows with SPH
Batchelor (1967) -- An introduction to fluid dynamics

Sean Bowman [02/05/2026]
'''

#--------------------------------------------------------------------#
# -- Fluid Properties -- #
#--------------------------------------------------------------------#

# Reference fluid density (freshwater at 20C) [kg/m^3]
referenceDensity: float = 1000.0

# Gravitational acceleration [m/s^2]
gravity: float = 9.81

# Kinematic viscosity of freshwater at 20C [m^2/s]
kinematicViscosity: float = 1.0e-6

# Dynamic viscosity of freshwater [Pa*s]
dynamicViscosity: float = referenceDensity * kinematicViscosity

#--------------------------------------------------------------------#
# -- SPH Numerical Parameters -- #
#--------------------------------------------------------------------#

# Speed of sound multiplier: c = speedOfSoundFactor * sqrt(g * H)
# For WCSPH, c should be at least 10x max fluid velocity
# to keep density fluctuations below ~1%
speedOfSoundFactor: float = 10.0

# Tait equation of state exponent
# gamma = 7 is standard for water in WCSPH
gamma: float = 7.0

# CFL number for adaptive time step control
# dt = cflNumber * h / (c + max|v|)
cflNumber: float = 0.25

# Monaghan artificial viscosity coefficient (alpha)
# Controls dissipation for shock-like features and stability
# Typical range: 0.01 - 0.5
alphaViscosity: float = 0.1

# Density re-initialization interval (Shepard filter)
# Applied every N time steps to correct density drift
densityFilterInterval: int = 30

# XSPH velocity correction factor
# Smooths particle trajectories; range [0, 1], typically 0.5
xsphEpsilon: float = 0.5

# Smoothing length to particle spacing ratio
# h = smoothingLengthRatio * particleSpacing
# Typical range: 1.2 - 1.5
defaultSmoothingLengthRatio: float = 1.3

# Number of boundary particle layers
# More layers = better pressure at walls, but more particles
defaultBoundaryLayers: int = 3
