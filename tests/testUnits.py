
# -- pySurf: Unit Tests -- #

'''

Checks on pySurf's conversion factors and its conversion boundary.

The factors are derived from a Pint registry rather than typed, so what is worth testing is
definition: that each constant is the exact ratio its two units define, and that the boundary
round-trips.

Sean Bowman [10/10/2026]

'''

import numpy as np
import pint
import pytest

from pysurf import units

# Each factor and the exact value its definition gives. These are definitions, not measurements,
# so they are checked to machine precision.
exactFactors = (
    ('metersPerInch',             units.metersPerInch,             0.0254),
    ('metersPerFoot',             units.metersPerFoot,             0.3048),
    ('inchesPerFoot',             units.inchesPerFoot,             12.0),
    ('metersPerMillimeter',       units.metersPerMillimeter,       1.0e-3),
    ('squareMetersPerSquareFoot', units.squareMetersPerSquareFoot, 0.09290304),
    ('cubicMetersPerLiter',       units.cubicMetersPerLiter,       1.0e-3),
    ('cubicMetersPerCubicInch',   units.cubicMetersPerCubicInch,   1.6387064e-5),
    ('cubicMetersPerCubicFoot',   units.cubicMetersPerCubicFoot,   0.028316846592),
    ('kilogramsPerPound',         units.kilogramsPerPound,         0.45359237),
    ('newtonsPerPoundForce',      units.newtonsPerPoundForce,      4.4482216152605),
    ('metersPerSecondPerKnot',    units.metersPerSecondPerKnot,    1852.0 / 3600.0),
    ('metersPerSecondPerMph',     units.metersPerSecondPerMph,     0.44704),
    ('kgPerCubicMeterPerPoundPerCubicFoot', units.kgPerCubicMeterPerPoundPerCubicFoot, 16.0184633739601),
)

@pytest.mark.parametrize('name, derived, defined', exactFactors)
def testFactorMatchesItsDefinition(name, derived, defined):

    '''Every factor is the exact ratio its two units define.'''

    assert derived == pytest.approx(defined, rel = 1e-12), name

def testBoardDimensionsConvert():

    '''A 6'0" x 20" board is 1.8288 m long and 1.6667 ft wide.'''

    assert units.convert(6.0, 'ft', 'm') == pytest.approx(1.8288, rel = 1e-12)
    assert units.convert(20.0, 'in', 'ft') == pytest.approx(20.0 / 12.0, rel = 1e-12)
    assert units.convert(30.0, 'L', 'in^3') == pytest.approx(30.0e-3 / 1.6387064e-5, rel = 1e-12)

def testTheNoOpIsExactAndDoesNotCopy():

    array = np.array([1.0, 2.0, 3.0])

    assert units.convert(array, 'ft', 'ft') is array

def testArraysConvertElementwise():

    assert units.convert(np.array([1.0, 2.0]), 'ft', 'in') == pytest.approx([12.0, 24.0])

@pytest.mark.parametrize('dimension', sorted(units.dimensions))
def testEveryListedUnitRoundTrips(dimension):

    '''Every unit a dimension lists converts to SI and back without loss.'''

    for unit in units.unitsFor(dimension):
        value = 3.7
        assert units.fromSI(units.toSI(value, dimension, unit), dimension, unit) == pytest.approx(value, rel = 1e-12), unit

def testTheSiUnitIsListedFirst():

    for dimension in units.dimensions:
        assert units.unitsFor(dimension)[0] == units.siUnit(dimension)

def testMismatchedDimensionsAreRefused():

    with pytest.raises(pint.DimensionalityError):
        units.convert(1.0, 'ft', 'kg')
