
# -- pySurf: Units -- #

'''

One unit registry, from which every conversion factor in pySurf is derived.

Board geometry is quoted in the units shapers use: length in feet and inches, volume in liters,
rider weight in pounds, speed in knots or miles per hour. The physics runs in mass-base SI. This
module is the boundary between the two, and conversion belongs here rather than as a bare 12 or
0.0254 inside a solver.

The factors below are not transcribed. Each is computed from `ureg` at import, so a factor cannot
drift from its definition or be typed wrong, and the unit each one converts between is written in
the expression rather than in a comment beside it.

----------------------------------------------------------------------
                        Quantities versus magnitudes
----------------------------------------------------------------------

Solvers take and return plain floats and arrays, not `pint.Quantity` objects. A board surface
runs to hundreds of thousands of point evaluations and an SPH step to millions of pair
interactions, and unit objects in those loops buy nothing: the arrays are in one unit system by
construction. Pint is for the edge, where a number arrives carrying a unit that is not the
solver's.

`convert(value, currentlyIs, shouldBe)` is the general form of that edge. `toSI` and `fromSI`
are the narrower form for a field whose dimension is known; both delegate to `convert`, so there
is one conversion path rather than two that can disagree.

----------------------------------------------------------------------
                        Validation status
----------------------------------------------------------------------

Nothing here is a model. Every factor is a definition (the international inch is exactly
0.0254 m, the avoirdupois pound exactly 0.45359237 kg, the knot exactly 1852 m per hour), and
tests/testUnits.py checks each derived factor against its defining value to 1e-12 relative.

Sean Bowman [10/10/2026]

'''

# Third-party imports
from pint import UnitRegistry

# One registry for the whole package. Pint quantities from different registries do not
# interoperate, so anything needing a Quantity should take this one rather than build its own.
ureg = UnitRegistry()
Quantity = ureg.Quantity

def _factor(fromUnit: str, toUnit: str) -> float:

    '''Magnitude of one `fromUnit` expressed in `toUnit`.'''

    return Quantity(1.0, fromUnit).to(toUnit).magnitude

#----------------------------------------------------------------------#
# -- Conversion Factors -- #
#----------------------------------------------------------------------#

# -- Length -- #
metersPerInch       = _factor('inch', 'm')
metersPerFoot       = _factor('foot', 'm')
inchesPerFoot       = _factor('foot', 'inch')
metersPerMillimeter = _factor('mm', 'm')

# -- Area and volume -- #
squareMetersPerSquareFoot = _factor('foot ** 2', 'm ** 2')
cubicMetersPerLiter       = _factor('liter', 'm ** 3')
cubicMetersPerCubicInch   = _factor('inch ** 3', 'm ** 3')
cubicMetersPerCubicFoot   = _factor('foot ** 3', 'm ** 3')

# -- Mass and force -- #
kilogramsPerPound    = _factor('pound', 'kg')
newtonsPerPoundForce = _factor('force_pound', 'N')

# -- Speed -- #
metersPerSecondPerKnot = _factor('knot', 'm / s')
metersPerSecondPerMph  = _factor('mile / hour', 'm / s')

# -- Density -- #
kgPerCubicMeterPerPoundPerCubicFoot = _factor('pound / foot ** 3', 'kg / m ** 3')

#----------------------------------------------------------------------#
# -- Dimensions and the Display Boundary -- #
#----------------------------------------------------------------------#

# What a field of each dimension is stored as, and what it may be given in. The SI slot is the
# unit the solvers expect and is listed first.
dimensions = {
    'length':   {'si': 'm',      'units': ('m', 'mm', 'cm', 'in', 'ft')},
    'area':     {'si': 'm^2',    'units': ('m^2', 'cm^2', 'in^2', 'ft^2')},
    'volume':   {'si': 'm^3',    'units': ('m^3', 'L', 'in^3', 'ft^3')},
    'mass':     {'si': 'kg',     'units': ('kg', 'g', 'lb')},
    'force':    {'si': 'N',      'units': ('N', 'kN', 'lbf')},
    'speed':    {'si': 'm/s',    'units': ('m/s', 'km/h', 'kn', 'mph', 'ft/s')},
    'density':  {'si': 'kg/m^3', 'units': ('kg/m^3', 'g/cm^3', 'lb/ft^3')},
    'pressure': {'si': 'Pa',     'units': ('Pa', 'kPa', 'bar', 'psi')},
    'angle':    {'si': 'deg',    'units': ('deg', 'rad')},
    'time':     {'si': 's',      'units': ('s', 'min', 'h')},
}

# Short unit names as the registry spells them. Only the entries that differ need to appear.
_registryName = {
    'in':     'inch',
    'ft':     'foot',
    'L':      'liter',
    'm^2':    'm ** 2',
    'cm^2':   'cm ** 2',
    'in^2':   'inch ** 2',
    'ft^2':   'foot ** 2',
    'm^3':    'm ** 3',
    'in^3':   'inch ** 3',
    'ft^3':   'foot ** 3',
    'lb':     'pound',
    'lbf':    'force_pound',
    'kn':     'knot',
    'mph':    'mile / hour',
    'ft/s':   'foot / s',
    'km/h':   'km / hour',
    'kg/m^3': 'kg / m ** 3',
    'g/cm^3': 'g / cm ** 3',
    'lb/ft^3': 'pound / foot ** 3',
    'deg':    'degree',
    'rad':    'radian',
    'h':      'hour',
}

def _asRegistryUnit(unit: str) -> str:

    '''The registry's name for a short unit name.'''

    return _registryName.get(unit, unit)

def convert(value, currentlyIs: str, shouldBe: str):

    '''

    Convert a value from one unit to another.

    Units are named the way the rest of pySurf names them ('in', 'ft', 'L', 'lb', 'kn', 'mph'),
    and anything the registry already spells the same way works too.

    Parameters:
    -----------
    value : float | numpy.ndarray
        The quantity to convert.
    currentlyIs : str
        The unit `value` is in.
    shouldBe : str
        The unit to return it in.

    Returns:
    --------
    float | numpy.ndarray : The same physical quantity, expressed in `shouldBe`.

    Raises:
    -------
    pint.DimensionalityError : If the two units measure different things.
    pint.UndefinedUnitError : If either name is not a unit.

    Examples:
    ---------
    >>> convert(20.0, 'in', 'ft')
    1.6666666666666667
    >>> convert(6.0, 'ft', 'm')
    1.8288

    '''

    # The no-op has to stay exact. Round-tripping through the registry would return a float that
    # is equal but reconstructed, and for an array it would copy.
    if currentlyIs == shouldBe:
        return value

    return Quantity(value, _asRegistryUnit(currentlyIs)).to(_asRegistryUnit(shouldBe)).magnitude

def unitsFor(dimension: str) -> list:

    '''

    Ordered list of unit names a dimension may be given in, SI unit first.

    '''

    return list(dimensions[dimension]['units'])

def siUnit(dimension: str) -> str:

    '''

    The unit a dimension is stored and handed to the solvers in.

    '''

    return dimensions[dimension]['si']

def toSI(value, dimension: str, unit: str):

    '''

    Convert a value given in `unit` to the SI unit of its dimension.

    Parameters:
    -----------
    value : float | numpy.ndarray
        The number as given.
    dimension : str
        A key of `dimensions`.
    unit : str
        A unit listed for that dimension.

    Returns:
    --------
    float | numpy.ndarray : The same physical quantity in the dimension's SI unit.

    '''

    return convert(value, unit, dimensions[dimension]['si'])

def fromSI(value, dimension: str, unit: str):

    '''

    Convert a stored SI value to a chosen unit. The inverse of `toSI`.

    '''

    return convert(value, dimensions[dimension]['si'], unit)
