
# -- pySurf: Deliberate Failures -- #

'''

Every failure pySurf raises on purpose is one of these, so a caller can catch the family it cares
about instead of guessing at exception types.

'PySurfError' is the common base. Its subclasses narrow it to why a calculation stopped: an input
was invalid before any calculation ran, a geometric constraint could not be met, an iterative
solve did not converge, or a simulation produced values that are not finite. Catching the base
class catches every one of them.

'InvalidInputError' is also a 'ValueError', so code written to catch the built-in exception for a
bad argument still catches it.

'createErrorContext' builds the dictionary a raise site attaches to the exception, so the state at
the point of failure survives into the traceback rather than being lost to the stack unwinding
past it.

Sean Bowman [10/10/2026]

'''

# Standard library imports
from typing import Any, Optional

class PySurfError(Exception):

    '''

    Base class for every exception pySurf raises deliberately.

    Attributes:
    -----------
    message : str
        Human-readable description of the failure
    context : dict
        State at the point of failure (station index, variable values, and so on)

    '''

    def __init__(self, message: str, context: Optional[dict[str, Any]] = None) -> None:

        self.message = message
        self.context = context if context is not None else {}

        # The context is written into the message itself, so it reaches a log or a traceback
        # without the caller having to know the attribute exists
        fullMessage = message
        if self.context:
            contextLines = '\n'.join(f'    {key}: {value}' for key, value in self.context.items())
            fullMessage += f'\n  Context ({type(self).__name__}):\n{contextLines}'

        super().__init__(fullMessage)

    def getContext(self) -> dict[str, Any]:

        '''

        The context dictionary attached at the raise site.

        '''

        return self.context

class InvalidInputError(PySurfError, ValueError):

    '''

    An input is missing, the wrong type, or outside the range it may take.

    Raised during input validation, before any geometry is generated or any simulation step is
    taken, so the message names the field at fault rather than a symptom further downstream.

    Attributes:
    -----------
    parameterName : str
        Name of the invalid parameter (stored in context)
    value : Any
        The value provided (stored in context)
    validRange : str
        Description of what the parameter may take (stored in context)

    '''

    def __init__(self, message: str, context: Optional[dict[str, Any]] = None,
                 parameterName: Optional[str] = None, value: Any = None,
                 validRange: Optional[str] = None) -> None:

        context = dict(context) if context is not None else {}

        if parameterName is not None:
            context['parameterName'] = parameterName
        if value is not None:
            context['value'] = value
        if validRange is not None:
            context['validRange'] = validRange

        super().__init__(message, context)

class GeometryError(PySurfError):

    '''

    A geometric constraint cannot be met: a deck below the bottom, a fillet radius too large for
    the curves it joins, a section with no area.

    '''

class ConvergenceFailureError(PySurfError):

    '''

    An iterative solve stopped without meeting its tolerance.

    Attributes:
    -----------
    iterations : int
        Iterations attempted (stored in context)
    tolerance : float
        Tolerance that was not met (stored in context)
    residual : float
        Final residual (stored in context)

    '''

    def __init__(self, message: str, context: Optional[dict[str, Any]] = None,
                 iterations: Optional[int] = None, tolerance: Optional[float] = None,
                 residual: Optional[float] = None) -> None:

        context = dict(context) if context is not None else {}

        if iterations is not None:
            context['iterations'] = iterations
        if tolerance is not None:
            context['tolerance'] = tolerance
        if residual is not None:
            context['residual'] = residual

        super().__init__(message, context)

class NumericalInstabilityError(PySurfError):

    '''

    A calculation produced values that are not finite (NaN or infinity), or a simulation state
    left the range its method is valid in.

    Attributes:
    -----------
    variableName : str
        Name of the variable holding the bad value (stored in context)
    value : Any
        The bad value (stored in context)
    operation : str
        The operation that produced it (stored in context)

    '''

    def __init__(self, message: str, context: Optional[dict[str, Any]] = None,
                 variableName: Optional[str] = None, value: Any = None,
                 operation: Optional[str] = None) -> None:

        context = dict(context) if context is not None else {}

        if variableName is not None:
            context['variableName'] = variableName
        if value is not None:
            context['value'] = value
        if operation is not None:
            context['operation'] = operation

        super().__init__(message, context)

def createErrorContext(stationIndex: Optional[int] = None,
                       iterationCount: Optional[int] = None,
                       **kwargs) -> dict[str, Any]:

    '''

    Build the context dictionary a raise site attaches to an exception.

    Parameters:
    -----------
    stationIndex : int, optional
        Longitudinal station index where the failure occurred
    iterationCount : int, optional
        Iteration number when the failure occurred
    **kwargs
        Any further state worth keeping

    Returns:
    --------
    dict : The context, with only the entries that were given

    Examples:
    ---------
    >>> createErrorContext(stationIndex = 42, halfWidth = 0.25)
    {'stationIndex': 42, 'halfWidth': 0.25}

    '''

    context = {}

    if stationIndex is not None:
        context['stationIndex'] = stationIndex

    if iterationCount is not None:
        context['iterationCount'] = iterationCount

    context.update(kwargs)

    return context
