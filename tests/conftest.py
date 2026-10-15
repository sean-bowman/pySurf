
# -- pySurf: Test Session Configuration -- #

'''

What the whole test session needs set before any test module is imported.

pytest imports this before it collects anything, which is the only moment early enough to decide
things fixed at import time.

**No figure may open a window.** Setting `PYSURF_HEADLESS` here reaches `figures.headlessPlots`,
which selects a non-interactive matplotlib backend before pyplot is imported and turns every
`showFigure` call into a no-op. Every figure is still built, so a test that checks a figure still
has something to check. It is set with `setdefault`, so exporting `PYSURF_HEADLESS=0` before a
run puts the windows back.

**The backend is forced as well as requested.** A test run started through an IDE can have
imported pyplot already, in which case the environment variable arrives too late to choose the
backend and only forcing it works.

Sean Bowman [10/10/2026]

'''

import os

os.environ.setdefault('PYSURF_HEADLESS', '1')

import matplotlib

matplotlib.use('Agg', force = True)

import matplotlib.pyplot as plt
import pytest

@pytest.fixture(autouse = True)
def closeFiguresAfterEveryTest():

    '''

    Discard whatever figures a test left behind, so they do not accumulate across the suite.

    '''

    yield

    plt.close('all')
