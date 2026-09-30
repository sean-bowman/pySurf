# -- Physical Constants for Surfboard Physics -- #

'''

Physical and environmental constants for surfboard hydrodynamic analysis.
Mirrors SurfboardGeometry/Utils/Constants.cs with additional wave physics constants.

All values in SI units unless otherwise noted.

Sean Bowman [02/03/2026]

'''

#--------------------------------------------------------------------#
# -- Water Properties -- #
#--------------------------------------------------------------------#

# Seawater density [kg/m^3]
# Used for buoyancy and hydrodynamic force calculations
seawaterDensity: float = 1025.0

# Gravitational acceleration [m/s^2]
gravity: float = 9.81

# Kinematic viscosity of seawater at 20C [m^2/s]
# Used in Reynolds number calculations for drag estimation
seawaterKinematicViscosity: float = 1.05e-6

# Dynamic viscosity of seawater [Pa*s]
seawaterDynamicViscosity: float = seawaterDensity * seawaterKinematicViscosity

# Surface tension of seawater [N/m]
surfaceTension: float = 0.072

#--------------------------------------------------------------------#
# -- Surfboard Material Properties -- #
#--------------------------------------------------------------------#

# PU (polyurethane) foam core density [kg/m^3]
# Traditional surfboard blank material
puFoamDensity: float = 35.0

# EPS (expanded polystyrene) foam core density [kg/m^3]
# Lighter alternative used in epoxy boards
epsFoamDensity: float = 20.0

# Fiberglass laminate density [kg/m^3]
# Resin-saturated glass cloth shell
fiberglassDensity: float = 1800.0

# Fiberglass shell thickness [mm and m]
# Typical 4+4oz bottom, 4oz deck glass schedule
glassShellThicknessMm: float = 1.5
glassShellThicknessM: float = 0.0015

#--------------------------------------------------------------------#
# -- Wave Physics Constants -- #
#--------------------------------------------------------------------#

# Depth-limited breaking ratio H/d (McCowan 1894)
breakingDepthRatio: float = 0.78

# Steepness-limited breaking ratio H/L (Miche 1944)
breakingSteepnessRatio: float = 1.0 / 7.0

#--------------------------------------------------------------------#
# -- Default Surfer Properties -- #
#--------------------------------------------------------------------#

# Default surfer mass [kg]
defaultSurferMassKg: float = 75.0
