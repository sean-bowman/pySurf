
# -- pySurf: Input Validation and Error Tests -- #

'''

Checks on the rule vocabulary in pysurf.validation, the exception family in pysurf.errors, and
the rule table `Surfboard` validates its inputs against.

Sean Bowman [10/10/2026]

'''

import math

import numpy as np
import pytest

from pysurf.errors import (ConvergenceFailureError, GeometryError, InvalidInputError, NumericalInstabilityError,
                           PySurfError, createErrorContext)
from pysurf.Surfboard import Surfboard, inputRules
from pysurf.validation import (Overlay, applyRules, arrayRule, booleanRule, choiceRule, fieldsCovered, integerRule,
                               numericRule, specified)

#----------------------------------------------------------------------#
# -- Errors -- #
#----------------------------------------------------------------------#

@pytest.mark.parametrize('errorClass', [InvalidInputError, GeometryError, ConvergenceFailureError,
                                        NumericalInstabilityError])
def testEveryErrorIsAPySurfError(errorClass):

    assert issubclass(errorClass, PySurfError)

def testAnInvalidInputIsAlsoAValueError():

    '''Code written to catch the built-in exception for a bad argument still catches it.'''

    with pytest.raises(ValueError):
        raise InvalidInputError('Board length must be greater than 0.0', parameterName = 'totalLength', value = -1.0)

def testTheContextReachesTheMessage():

    error = ConvergenceFailureError('Dispersion solve stalled', iterations = 50, residual = 1e-3)

    assert 'iterations: 50' in str(error)
    assert error.getContext()['residual'] == 1e-3

def testTheCallersContextIsNotMutated():

    context = {'stationIndex': 3}
    InvalidInputError('bad', context = context, parameterName = 'halfWidth')

    assert context == {'stationIndex': 3}

def testCreateErrorContextKeepsOnlyWhatWasGiven():

    assert createErrorContext(stationIndex = 42, halfWidth = 0.25) == {'stationIndex': 42, 'halfWidth': 0.25}

#----------------------------------------------------------------------#
# -- Rule Vocabulary -- #
#----------------------------------------------------------------------#

def testAbsentNoneAndNaNAreAllUnset():

    source = {'a': None, 'b': math.nan, 'c': 1.0, 'd': 'text', 'e': np.array([1.0, 2.0])}

    assert not specified(source, 'a')
    assert not specified(source, 'b')
    assert not specified(source, 'missing')
    assert specified(source, 'c') and specified(source, 'd') and specified(source, 'e')

@pytest.mark.parametrize('value, passes', [(1.0, True), (0.0, False), (-2.0, False), (math.inf, False), ('6', False)])
def testNumericBoundsAndType(value, passes):

    rules = [numericRule('totalLength', 'Board length', units = 'ft', minimum = 0.0)]

    if passes:
        applyRules({'totalLength': value}, rules)
    else:
        with pytest.raises(InvalidInputError):
            applyRules({'totalLength': value}, rules)

def testAnInclusiveBoundAdmitsTheBound():

    applyRules({'n': 2}, [integerRule('n', minimum = 2, exclusiveMinimum = False)])

    with pytest.raises(InvalidInputError):
        applyRules({'n': 2.0}, [integerRule('n', minimum = 2, exclusiveMinimum = False)])

def testAChoiceOutsideItsSetIsRefusedWithTheSetInTheMessage():

    with pytest.raises(InvalidInputError, match = 'squash'):
        applyRules({'tailShape': 'swallow'}, [choiceRule('tailShape', choices = ('squash', 'round'))])

def testARequiredFieldThatIsUnsetIsRefused():

    with pytest.raises(InvalidInputError, match = 'not specified'):
        applyRules({}, [numericRule('maxWidth')])

def testAnOptionalFieldThatIsUnsetIsSkipped():

    applyRules({}, [numericRule('maxWidth', required = False, minimum = 0.0)])

def testAConditionalRuleOnlyAppliesWhenItsPredicateHolds():

    rules = [numericRule('notchDepth', minimum = 0.0, when = lambda source: source.get('tailShape') == 'swallow')]

    applyRules({'tailShape': 'round', 'notchDepth': -1.0}, rules)
    with pytest.raises(InvalidInputError):
        applyRules({'tailShape': 'swallow', 'notchDepth': -1.0}, rules)

def testArraysAreCheckedForLengthFinitenessAndSign():

    source = {'x': np.linspace(0.0, 1.0, 5), 'r': np.array([0.1, 0.2, np.nan, 0.2, 0.1])}

    with pytest.raises(InvalidInputError, match = 'not finite'):
        applyRules(source, [arrayRule('r')])
    with pytest.raises(InvalidInputError, match = 'same length'):
        applyRules({'x': np.zeros(5), 'r': np.ones(4)}, [arrayRule('r', sameLengthAs = 'x')])
    with pytest.raises(InvalidInputError, match = 'greater than zero'):
        applyRules({'r': np.array([0.1, 0.0])}, [arrayRule('r', positive = True)])

def testBooleansAreNotNumbers():

    with pytest.raises(InvalidInputError):
        applyRules({'flag': 1}, [booleanRule('flag')])
    with pytest.raises(InvalidInputError):
        applyRules({'length': True}, [numericRule('length')])

def testAnOverlayReadsItsOverridesFirst():

    class Board:
        totalLength = 6.0
        maxWidth = 20.0

    overlay = Overlay(Board(), maxWidth = -1.0)

    assert overlay.totalLength == 6.0 and overlay.maxWidth == -1.0

#----------------------------------------------------------------------#
# -- Surfboard Inputs -- #
#----------------------------------------------------------------------#

def testTheDefaultBoardIsValid():

    Surfboard()._validateInputs()

@pytest.mark.parametrize('field, value', [
    ('totalLength', 0.0), ('maxWidth', -20.0), ('midpointThickness', 0.0),
    ('noseShape', 'pointed'), ('tailShape', 'swallow'), ('rockerProfile', 'extreme'), ('finConfig', 'five'),
])
def testEachInvalidInputIsRefusedByName(field, value):

    board = Surfboard()
    setattr(board, field, value)

    with pytest.raises(InvalidInputError) as raised:
        board._validateInputs()

    assert raised.value.getContext()['parameterName'] == field

def testTheRuleTableCoversEveryConstructorInput():

    assert fieldsCovered(inputRules) == {'totalLength', 'maxWidth', 'midpointThickness', 'noseShape', 'tailShape',
                                         'rockerProfile', 'finConfig'}
