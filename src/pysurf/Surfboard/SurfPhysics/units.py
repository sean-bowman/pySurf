# -- Unit Conversion Helpers -- #

'''

Conversion functions between measurement systems.

The C# SurfboardGeometry code works in millimeters.
Python physics computations use SI (meters, kg, seconds).
These helpers ensure correct conversions at boundaries.

Sean Bowman [02/03/2026]

'''

import math

def mmToM(valueMm: float) -> float:

    '''Convert millimeters to meters.'''

    return valueMm * 1e-3

def mToMm(valueM: float) -> float:

    '''Convert meters to millimeters.'''

    return valueM * 1e3

def feetToM(valueFt: float) -> float:

    '''Convert feet to meters.'''

    return valueFt * 0.3048

def inchesToM(valueIn: float) -> float:

    '''Convert inches to meters.'''

    return valueIn * 0.0254

def cubicMmToLiters(valueMm3: float) -> float:

    '''Convert cubic millimeters to liters.'''

    return valueMm3 * 1e-6

def litersToM3(valueLiters: float) -> float:

    '''Convert liters to cubic meters.'''

    return valueLiters * 1e-3

def m3ToLiters(valueM3: float) -> float:

    '''Convert cubic meters to liters.'''

    return valueM3 * 1e3

def knotsToMs(knots: float) -> float:

    '''Convert knots to meters per second.'''

    return knots * 0.514444

def msToKnots(ms: float) -> float:

    '''Convert meters per second to knots.'''

    return ms / 0.514444

def degreesToRadians(deg: float) -> float:

    '''Convert degrees to radians.'''

    return deg * math.pi / 180.0

def radiansToDegrees(rad: float) -> float:

    '''Convert radians to degrees.'''

    return rad * 180.0 / math.pi
