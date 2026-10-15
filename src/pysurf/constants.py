
# -- pySurf: Physical Constants -- #

'''

Physical constants shared by every pySurf model: gravity and the properties of the water a board
rides on.

Gravity is the standard value, derived from the unit registry rather than typed, so it cannot
drift from its definition. The water properties are reference values at 20 degC; nothing in pySurf
resolves temperature or salinity, so they are constants rather than a property model.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

Gravity is a definition (9.80665 m/s^2, CGPM 1901) and tests/testUnits.py checks the registry.
Freshwater density 998.2 kg/m^3 and dynamic viscosity 1.0016 mPa s (kinematic 1.0034e-6 m^2/s)
are the NIST/IAPWS values for pure water at 20 degC and 0.1 MPa. Seawater density 1025 kg/m^3 is the conventional
oceanographic reference for surface water at about 35 g/kg salinity; it is a round value, not a
property evaluation.

Sean Bowman [10/11/2026]

'''

# Local imports
from pysurf.units import Quantity

# Standard gravity [m/s^2]
gravity: float = Quantity(1.0, 'standard_gravity').to('m / s ** 2').magnitude

# Pure water at 20 degC, 0.1 MPa (NIST)
freshwaterDensity: float = 998.2                  # [kg/m^3]
freshwaterKinematicViscosity: float = 1.0034e-6   # [m^2/s]

# Surface seawater, about 35 g/kg salinity
seawaterDensity: float = 1025.0                   # [kg/m^3]
