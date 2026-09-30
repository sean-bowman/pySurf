
# -- Code Interface (Entry-Point) -- #

'''

Code entry point for testing the Python parametric surfboard geometry generation functionality.

Sean Bowman

'''

import os
from pysurf.parametricSurfboard.Surfboard import Surfboard

os.system('cls' if os.name == 'nt' else 'clear')

# Code here
testBoard = Surfboard(totalLength = 6.0, maxWidth = 20.0, midpointThickness = 2.75,
                      noseShape = 'point', tailShape = 'round', rockerProfile = 'aggressive')
testBoard.plotsEnabled = True
testBoard.generateGeometry()

debug = 1
