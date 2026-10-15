
# -- pySurf: Input Validation -- #

'''

Checking an input set before a calculation spends time on it.

A validator answers a different question from a test. A test asks whether the code is right,
against fixtures the author chose. A validator asks whether *this* input set is usable, about
numbers the author never saw. Neither substitutes for the other.

A rule is data here, and there is one checker. Every check is some combination of a small
vocabulary: is it present, is it a number in a range, is it one of these strings, is it a
non-empty finite array, does it match another array's length. Written as branches, each check
costs about ten lines and drifts out of step with the code it guards; written as a rule, a table
can be read at a glance, compared against the fields it guards, and tested as data.

----------------------------------------------------------------------
                              Writing rules
----------------------------------------------------------------------

    numericRule('totalLength', 'Board length', units = 'ft', minimum = 0)
    integerRule('resolution', 'Axial stations', minimum = 2, exclusiveMinimum = False)
    choiceRule('noseShape', 'Nose shape', choices = ('point', 'round', 'fish'))
    arrayRule('rBoardOutline', 'Planform half-width', units = 'ft', finite = True)

Each returns a Rule. `applyRules(source, rules)` walks them in order and raises on the first
failure, so rules are ordered from the most basic to the most specific and a message always
names the earliest thing that is wrong.

A rule that only applies sometimes carries a `when` predicate:

    numericRule('swallowNotchDepth', 'Swallow notch depth', units = 'in', minimum = 0,
                when = lambda source: read(source, 'tailShape') == 'swallow')

`source` may be an object or a dictionary; `read` handles both, so the same table guards a
`Surfboard` and a configuration dictionary loaded from JSON.

----------------------------------------------------------------------
                            What counts as unset
----------------------------------------------------------------------

An input is unset in three different ways depending on where it came from: the attribute is
absent, it is None, or it is NaN (a JSON null read into a float field). All three mean the same
thing, and `specified` treats them the same.

Sean Bowman [10/10/2026]

'''

# Standard library imports
from dataclasses import dataclass
from typing import Any, Callable, Sequence

# Third-party imports
import numpy as np

# Local imports
from pysurf.errors import InvalidInputError

# The sentinel an input can carry to mean 'work it out for me' rather than 'not specified'. A rule
# that admits it skips its value checks when it sees it.
defaultSentinel = 'default'

def read(source, name: str, fallback = None):

    '''

    Read a field from an object or a dictionary.

    Parameters:
    -----------
    source : Any
        Object with attributes, or a mapping.
    name : str
        Field to read.
    fallback : Any
        Returned when the field is absent.

    Returns:
    --------
    Any : The field's value, or the fallback.

    '''

    if isinstance(source, dict):
        return source.get(name, fallback)

    return getattr(source, name, fallback)

class Overlay:

    '''

    A source that reads a few named values first and falls back to an object for the rest.

    A method that takes some of its inputs as arguments and the rest off the object it belongs to
    would otherwise need two rule tables. Layering the arguments over the object lets one table
    describe the whole call.

    Parameters:
    -----------
    base : Any
        Object or mapping the fallback comes from.
    overrides : Any
        Values that take precedence, by name.

    '''

    def __init__(self, base, **overrides) -> None:

        object.__setattr__(self, '_base', base)
        object.__setattr__(self, '_overrides', overrides)

    def __getattr__(self, name):

        overrides = object.__getattribute__(self, '_overrides')
        if name in overrides:
            return overrides[name]

        return read(object.__getattribute__(self, '_base'), name)

def specified(source, name: str) -> bool:

    '''

    True when a field carries a value rather than merely existing.

    Absent, None and NaN all mean the same thing: the input did not say. Anything that cannot be
    tested for NaN, a string or an array among them, is taken as specified.

    Parameters:
    -----------
    source : Any
        Object or mapping to read from.
    name : str
        Field to test.

    Returns:
    --------
    bool : True when the field was filled in.

    '''

    value = read(source, name)

    if value is None:
        return False

    try:
        return not bool(np.isnan(value))
    except (TypeError, ValueError):
        return True

@dataclass
class Rule:

    '''

    One thing that must be true of an input set.

    Attributes:
    -----------
    field : str
        Name of the field the rule guards.
    label : str
        Human name used in the message. Falls back to the field name.
    units : str
        Appended to the message where a value has them.
    kind : str
        'numeric', 'integer', 'text', 'boolean', 'choice', 'array' or 'present'.
    required : bool
        False lets the field be unset, and skips the value checks when it is.
    minimum, maximum : float
        Bounds. None means unbounded on that side.
    exclusiveMinimum, exclusiveMaximum : bool
        True excludes the bound itself, so minimum = 0 with exclusiveMinimum means strictly
        positive.
    choices : tuple
        Permitted values for a 'choice' rule.
    allowDefault : bool
        True admits the string 'default' and skips the value checks for it.
    sameLengthAs : str
        Name of another field this array must match in length.
    positive : bool
        True requires every element of an array to be greater than zero.
    finite : bool
        True requires every element of an array to be finite. On by default.
    minimumLength : int
        Shortest an array may be.
    when : Callable
        Predicate on the source. The rule is skipped when it returns False.
    note : str
        Appended to the range description, for anything the vocabulary cannot say.

    '''

    field:            str
    label:            str      = ''
    units:            str      = ''
    kind:             str      = 'numeric'
    required:         bool     = True
    minimum:          float    = None
    maximum:          float    = None
    exclusiveMinimum: bool     = True
    exclusiveMaximum: bool     = True
    choices:          Sequence = ()
    allowDefault:     bool     = False
    sameLengthAs:     str      = None
    positive:         bool     = False
    finite:           bool     = True
    minimumLength:    int      = 1
    when:             Callable = None
    note:             str      = ''

    @property
    def name(self) -> str:

        '''Human name for messages.'''

        return self.label or self.field

    def rangeText(self) -> str:

        '''

        The range the message quotes, built from the rule rather than written twice.

        '''

        units = f' [{self.units}]' if self.units else ''

        if self.kind == 'choice':
            text = 'One of ' + ', '.join(repr(choice) for choice in self.choices)
        elif self.kind == 'array':
            pieces = [f'Array of at least {self.minimumLength}']
            if self.finite:
                pieces.append('all finite')
            if self.positive:
                pieces.append('all greater than zero')
            if self.sameLengthAs:
                pieces.append(f'same length as {self.sameLengthAs}')
            text = ', '.join(pieces) + units
        elif self.kind == 'text':
            text = 'A non-empty string'
        elif self.kind == 'boolean':
            text = 'True or False'
        elif self.kind == 'present':
            text = 'Present'
        else:
            noun = 'Integer' if self.kind == 'integer' else 'Float'
            if self.minimum is None and self.maximum is None:
                text = noun + units
            elif self.maximum is None:
                text = f'{noun} {">" if self.exclusiveMinimum else ">="} {self.minimum}{units}'
            elif self.minimum is None:
                text = f'{noun} {"<" if self.exclusiveMaximum else "<="} {self.maximum}{units}'
            else:
                lower = '(' if self.exclusiveMinimum else '['
                upper = ')' if self.exclusiveMaximum else ']'
                text = f'{noun} in {lower}{self.minimum}, {self.maximum}{upper}{units}'

        if self.allowDefault:
            text += f' or \'{defaultSentinel}\''
        if self.note:
            text += f'. {self.note}'

        return text

def numericRule(fieldName: str, label: str = '', **options) -> Rule:

    '''A rule for a floating point field.'''

    return Rule(field = fieldName, label = label, kind = 'numeric', **options)

def integerRule(fieldName: str, label: str = '', **options) -> Rule:

    '''A rule for a whole number field.'''

    return Rule(field = fieldName, label = label, kind = 'integer', **options)

def choiceRule(fieldName: str, label: str = '', **options) -> Rule:

    '''A rule for a field that must be one of a fixed set.'''

    return Rule(field = fieldName, label = label, kind = 'choice', **options)

def arrayRule(fieldName: str, label: str = '', **options) -> Rule:

    '''A rule for an array field.'''

    return Rule(field = fieldName, label = label, kind = 'array', **options)

def textRule(fieldName: str, label: str = '', **options) -> Rule:

    '''A rule for a non-empty string field.'''

    return Rule(field = fieldName, label = label, kind = 'text', **options)

def booleanRule(fieldName: str, label: str = '', **options) -> Rule:

    '''A rule for a true or false field.'''

    return Rule(field = fieldName, label = label, kind = 'boolean', **options)

def presentRule(fieldName: str, label: str = '', **options) -> Rule:

    '''A rule that only requires a field to carry something.'''

    return Rule(field = fieldName, label = label, kind = 'present', **options)

def _reject(rule: Rule, message: str, value: Any) -> None:

    '''Raise the input error, with the rule's own range description.'''

    raise InvalidInputError(
        message       = message,
        parameterName = rule.field,
        value         = value,
        validRange    = rule.rangeText())

def _checkNumber(rule: Rule, value: Any) -> None:

    '''Type and bounds for a numeric or integer rule.'''

    if rule.kind == 'integer':
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            _reject(rule, f'{rule.name} must be a whole number', value)
    else:
        if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
            _reject(rule, f'{rule.name} must be a number', value)

    if not np.isfinite(value):
        _reject(rule, f'{rule.name} must be finite', value)

    units = f' {rule.units}' if rule.units else ''

    if rule.minimum is not None:
        tooSmall = value <= rule.minimum if rule.exclusiveMinimum else value < rule.minimum
        if tooSmall:
            comparison = 'greater than' if rule.exclusiveMinimum else 'at least'
            _reject(rule, f'{rule.name} must be {comparison} {rule.minimum}{units}', value)

    if rule.maximum is not None:
        tooLarge = value >= rule.maximum if rule.exclusiveMaximum else value > rule.maximum
        if tooLarge:
            comparison = 'less than' if rule.exclusiveMaximum else 'at most'
            _reject(rule, f'{rule.name} must be {comparison} {rule.maximum}{units}', value)

def _checkArray(rule: Rule, value: Any, source) -> None:

    '''Shape and contents for an array rule.'''

    try:
        asArray = np.asarray(value, dtype = float)
    except (TypeError, ValueError):
        _reject(rule, f'{rule.name} must be an array of numbers', type(value).__name__)

    if asArray.ndim == 0 or asArray.size < rule.minimumLength:
        _reject(rule, f'{rule.name} must hold at least {rule.minimumLength} '
                      f'{"value" if rule.minimumLength == 1 else "values"}',
                int(asArray.size))

    if rule.finite and not np.isfinite(asArray).all():
        _reject(rule, f'{rule.name} contains values that are not finite',
                int(np.count_nonzero(~np.isfinite(asArray))))

    if rule.positive and not (asArray > 0).all():
        _reject(rule, f'{rule.name} must be greater than zero everywhere',
                float(np.nanmin(asArray)))

    if rule.sameLengthAs:
        other = read(source, rule.sameLengthAs)
        if other is not None and len(np.atleast_1d(other)) != asArray.size:
            _reject(rule, f'{rule.name} and {rule.sameLengthAs} must be the same length',
                    (asArray.size, len(np.atleast_1d(other))))

def applyRules(source, rules: Sequence[Rule]) -> None:

    '''

    Check an input set against a rule table.

    Rules are walked in order and the first failure raises, so a table reads from the most basic
    requirement to the most specific and a message always names the earliest thing that is wrong.

    Parameters:
    -----------
    source : Any
        Object or mapping holding the inputs.
    rules : Sequence[Rule]
        The table to check against.

    Raises:
    -------
    InvalidInputError : On the first rule the inputs fail.

    '''

    for rule in rules:

        if rule.when is not None and not rule.when(source):
            continue

        present = specified(source, rule.field)

        if not present:
            if rule.required:
                _reject(rule, f'{rule.name} ({rule.field}) is not specified', None)
            continue

        value = read(source, rule.field)

        if rule.allowDefault and isinstance(value, str) and value == defaultSentinel:
            continue

        if rule.kind == 'present':
            continue

        if rule.kind == 'choice':
            if value not in rule.choices:
                _reject(rule, f'{rule.name} is not one of the values it may take', value)

        elif rule.kind == 'text':
            if not isinstance(value, str) or not value.strip():
                _reject(rule, f'{rule.name} must be a non-empty string', value)

        elif rule.kind == 'boolean':
            if not isinstance(value, (bool, np.bool_)):
                _reject(rule, f'{rule.name} must be true or false', value)

        elif rule.kind == 'array':
            _checkArray(rule, value, source)

        else:
            _checkNumber(rule, value)

def fieldsCovered(rules: Sequence[Rule]) -> set:

    '''

    The set of field names a rule table guards.

    Used by the tests that hold a table against the object it validates, so a field that appears
    in one and not the other is caught by a check rather than by a run.

    '''

    return {rule.field for rule in rules}
