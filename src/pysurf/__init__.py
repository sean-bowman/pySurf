
# -- pySurf Package -- #

'''

Parametric surfboard design and water simulation.

The package is a set of facades over single-purpose modules. `Surfboard` carries a board's
parameters and `SurfaceBreakingWave` a wave on a beach; each delegates every stage of its work to a module:

    from pysurf import Surfboard
    board = Surfboard(totalLength = 6.0, maxWidth = 20.0, midpointThickness = 2.5)
    board.generateGeometry()

    from pysurf.SurfaceBreakingWave import SurfaceBreakingWave
    wave = SurfaceBreakingWave(depth = 1.0, waveHeight = 0.3, beachSlope = 1 / 35)
    wave.run()
    wave.writeAnimation('surfaceWave.gif')

Facades:

    Surfboard           The board: parameters, presets, profile and cross-section generation
    SurfaceBreakingWave A solitary wave shoaling and breaking on a plane beach, free surface (SGN)
    OverturningWave     A solitary wave curling into a plunging jet over any bed, potential flow (BEM)
    ReefBreak           Regular swell breaking over a reef in open water, free surface (SGN)
    ReefOverturn        Regular waves overturning on a reef, an SGN train handed to potential flow
    SplashUp            The overturn carried through the jet's impact, potential flow handed to SPH
    BreakingWave        The solitary wave on a beach simulated with particles (SPH)

Board and shared modules:

    geometryTools   Curve and surface math: Bezier construction, offsets, intersections, fillets
    figures         Every figure, described once and drawn by both matplotlib and plotly
    palette         The Surfy Pastels palette figures are drawn in, light by default
    exports         STL and delimited-text output
    units           One unit registry and every conversion factor
    constants       Gravity and water properties
    errors          The exception family pySurf raises on purpose
    validation      Rule tables that check inputs before a calculation runs

Water modules:

    waveTheory      Linear and solitary wave mechanics, wave makers, breaking criteria
    bathymetry      The sea bed as a 2D profile: plane beach or reef break
    sphKernels      SPH smoothing kernels
    sphNeighbors    Particle pairs within the kernel support
    sphParticles    The particle state and lattice seeding
    sphBoundaries   Container and bed boundaries made of fixed particles
    sphSolver       The weakly compressible SPH equations and time stepping
    sgnSolver       The Serre-Green-Naghdi equations, wave source, absorbing layers, breaking switch
    sgnHandoff      An SGN state handed to the boundary element model: walls, bores, potential
    bemSolver       Fully nonlinear potential flow by boundary elements, with a Lagrangian surface,
                    and the flow inside the water: velocity, and pressure from phi_t
    sphHandoff      The potential flow handed to SPH particles in a window, behind a piston
    waveMakers      Piston wave maker and the solitary wave initial condition
    waveBreaking    Free surface, front face lean, breaking point and runup
    WaterSim        Ready-to-run sloshing and wave tank scenarios: `from pysurf import WaterSim`

The water modules and the wave facades are imported on their own, so `import pysurf` stays light for
board work.

----------------------------------------------------------------------
                        Units
----------------------------------------------------------------------

`Surfboard` takes length in feet and width and thickness in inches, the units boards are quoted
in. The water modules run in SI. Conversion goes through `units.convert` rather than a bare factor.

Sean Bowman [10/11/2026]

'''

__version__ = '0.1.0'

# Import order runs from the modules with no siblings of their own outward to the facade, so a
# circular import surfaces here
from . import errors, units, validation, geometryTools, exports, palette, figures
from .Surfboard import Surfboard

__all__ = [
    # Facade
    'Surfboard',
    # Modules
    'errors', 'exports', 'figures', 'geometryTools', 'palette', 'units', 'validation',
]
