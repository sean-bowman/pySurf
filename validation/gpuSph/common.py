# -- Shared paths and helpers for the GPU SPH validation runs -- #

'''

Every case in this folder runs pySurf's δ-SPH scheme ('deltaSph', the GPU backend) with depth
h0 = 1 m, so lengths are in h0 and the time unit sqrt(h0/g) is `timeScale` seconds. Each script
writes its record to results/<name>.json and prints its verdicts against the criteria in its own
docstring, which were set before the runs. The environment variable PYSURF_VALIDATION_RESULTS
sends the records to another folder, for trial runs at low resolution.

Sean Bowman [11/19/2026]

'''

import json
import math
import os
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
repoRoot = here.parents[1]
sys.path.insert(0, str(repoRoot / 'src'))

from pysurf import constants

resultsDir = Path(os.environ.get('PYSURF_VALIDATION_RESULTS', here / 'results'))
referenceDir = here / 'reference'
timeScale = math.sqrt(1.0 / constants.gravity)          # [s], sqrt(h0 / g) with h0 = 1 m

def percent(value: float, reference: float) -> float:

    '''The relative difference of value from reference, in percent.'''

    return 100.0 * (value / reference - 1.0)

def verdict(passes: bool) -> str:

    '''The word a printed result line ends with.'''

    return 'passes' if passes else 'fails'

def writeResult(name: str, record: dict) -> Path:

    '''Write a run's record to results/<name>.json and return the path.'''

    resultsDir.mkdir(parents = True, exist_ok = True)
    path = resultsDir / f'{name}.json'
    path.write_text(json.dumps(record, indent = 1, default = _jsonDefault), encoding = 'utf-8', newline = '\n')
    print(f'wrote {path}', flush = True)
    return path

def readReference(name: str) -> dict:

    '''A reference data file from reference/.'''

    return json.loads((referenceDir / name).read_text(encoding = 'utf-8'))

def _jsonDefault(value):

    '''numpy scalars and arrays as plain JSON values.'''

    if hasattr(value, 'tolist'):
        return value.tolist()
    return str(value)
