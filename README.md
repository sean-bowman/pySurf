# Computational Engineering Portfolio

**Generation and analysis of physical models through computer algorithms.**

This repository demonstrates computational engineering: using software to define, generate, and analyze physical objects and their behavior. Rather than drawing shapes manually in CAD software, geometry is defined programmatically through mathematical functions and parameters, enabling precise control, reproducibility, and design exploration.

Domains live under `src/pysurf/`. Each domain is a self-contained Python package with its own configs, documentation, and code. `WaterSim` is organized as a generic tool for cross-domain reuse.

---

## Domains

### [Surfboard →](src/pysurf/Surfboard/README.md)

Parametric surfboard design, hydrodynamic physics, a pure-Python surface mesh generator, reference validation, and animated visualizations, benchmarked against a real reference board.

```bash
python codeInterface.py   # Reverse-engineer reference STL and launch comparison viewer
```

### [WaterSim →](src/pysurf/WaterSim/README.md)

Weakly Compressible SPH (WCSPH) solver for free-surface water flows. Supports 2D/3D sloshing tanks and numerical wave tanks with breaking wave detection. Used for physics simulations and Manim animations.

```bash
python -m pysurf.WaterSim   # Run default sloshing tank
```

---

## Project Structure

```txt
repo_root/
├── codeInterface.py                 Python entry point (reverse-engineering demo)
├── pyproject.toml                   Package metadata (src layout, editable install)
├── README.md
│
└── src/
    └── pysurf/
        ├── Surfboard/                Surfboard design domain → see Surfboard/README.md
        │   ├── SurfPhysics/          Python physics, geometry, validation
        │   ├── SurfAnimations/       Manim physics animations
        │   ├── SurfViewer/           Three.js interactive 3D viewer
        │   ├── configs/              JSON configuration files
        │   └── documentation/        Technical writeups
        │
        └── WaterSim/                 SPH water simulation → see WaterSim/README.md
            ├── sph/                  WCSPH solver engine
            ├── scenarios/            Pre-configured simulations
            ├── export/               Frame data export
            ├── configs/              Water sim JSON configs
            └── documentation/        Water sim technical writeups
```

---

## Prerequisites & Setup

```bash
git clone https://github.com/sean-bowman/pySurf.git
cd pySurf

# Editable install (src layout, makes `pysurf` importable everywhere)
pip install -e .
```

**Requirements:**

- Python 3.x

---

## Technology

| Component | Language | Purpose |
| --- | --- | --- |
| Surfboard Geometry | Python / trimesh | Parametric surface mesh generation |
| Physics Simulations | Python / NumPy | Wave theory and hydrodynamics |
| Water Simulation | Python / NumPy | WCSPH sloshing and wave tanks |
| 3D Visualization | JavaScript / Three.js | Interactive web viewer |
| Physics Animations | Python / Manim | Education videos |
| Reverse Engineering | Python / SciPy | Reference-board profile fitting |

---

## Future Work

- [X] Document all work thoroughly with informational overviews of all modules
- [X] Prioritize API calls over CLI calls for code library interfacing
- [X] Update surfboard and fin shapes against reference STLs for maximum realism
- [X] Investigate PicoGK geometry generation methodologies
- [X] Parallel-develop surface-mesh geometry generation through Python
- [X] Consolidate references and documentation folders
- [ ] Extend WaterSim to 3D with wave-maker boundaries for breaking wave modeling
- [ ] Add new geometry domains (boat hull, etc.) as sibling packages

---

## References

- Paine, Frank. *The Science of Surfing*. Cambridge University Press.
- Dean & Dalrymple. *Water Wave Mechanics for Engineers and Scientists*.
- Finney, Ben & Houston, James. *Surfing: A History of the Ancient Hawaiian Sport*.

---

Sean Bowman
