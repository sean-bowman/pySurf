
# -- DualSPHysics Build Check: the 2D Dam Break -- #

'''

Checks a DualSPHysics build on the 2D dam break it ships with its experimental data
(examples/main/01_DamBreak, CaseDambreakVal2D: a water column 1 m wide and 2 m high released into a
tank 4 m long, dp = 0.01 m), run as shipped on the GPU and the CPU solvers.

The front is the case's own gauge, Swl_z003: the free surface along the line z = 0.03 m (3 dp above
the floor), written every 0.005 s to out/GaugesSWL_Swl_z003.csv. The leading fluid particle's x,
from the PartVTK frames every 0.01 s, is reported beside it. Two criteria, set before the runs:

    build       the GPU run's front within 1 percent of the CPU run's at every gauge time before the
                front reaches the end wall (front below 4 - 2 dp)
    published   the GPU run's front within 10 percent of Koshizuka and Oka's (1996) measured front
                at each measured time where that front lies inside the tank (x < 4 m); the
                record's last point, x = 4.132 m at t = 0.751 s, lies past the modeled end wall

The front was first defined, before the runs, as the largest x of the fluid within two spacings of
the floor. DualSPHysics's dynamic boundary particles hold the water off the floor: the leading
particles ride 2 to 6 dp above it, so that definition missed the tip and measured water behind it,
and both criteria failed on it with fronts that stalled and jumped. The definition was replaced
after those runs by the gauge the case defines for this comparison; the tolerances are unchanged.

The simulated front is interpolated linearly in time to each measured time. The first criterion
checks that the two executables compute the same flow; the second that the build reproduces the
behavior DualSPHysics publishes for this case. Neither is a check of pySurf.

Run with the two run directories, each holding the out/ folder of the run and the csv/ folder
PartVTK wrote, and the experimental file shipped with the case:

    python checkDamBreak.py <gpuRun> <cpuRun> <EXP_X-DamTipPosition file> [outputDir]

It prints both comparisons and writes damBreakCheck.json and damBreakCheck.png to outputDir
(default: the GPU run directory).

Sean Bowman [11/08/2026]

'''

# Standard library imports
from __future__ import annotations

import json
import sys
from pathlib import Path

# Third-party imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from dsphOutput import readFrames

spacing = 0.01          # [m], dp of CaseDambreakVal2D
wallX = 4.0             # [m], the tank's end wall
buildTolerance = 0.01
publishedTolerance = 0.10

def gaugeFront(runDir: Path) -> tuple[np.ndarray, np.ndarray]:

    '''Gauge times [s] and the front [m] from the case's Swl_z003 gauge.'''

    data = np.genfromtxt(runDir / 'out' / 'GaugesSWL_Swl_z003.csv', delimiter = ';', skip_header = 1)
    return data[:, 0], data[:, 1]

def leadingParticle(runDir: Path) -> tuple[np.ndarray, np.ndarray]:

    '''Output times [s] and the largest x of the fluid [m], from the PartVTK frames.'''

    frames = readFrames(runDir / 'csv')
    return np.array([frame.time for frame in frames]), np.array([float(frame.positions[:, 0].max()) for frame in frames])

def readExperiment(path: Path) -> tuple[np.ndarray, np.ndarray]:

    '''Koshizuka and Oka's measured times [s] and fronts [m], from the tab-separated file shipped with the case.'''

    rows = []
    for line in path.read_text(encoding = 'latin-1').splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    data = np.array(rows)
    return data[:, 0], data[:, 1]

def main(gpuDir: Path, cpuDir: Path, experimentPath: Path, outputDir: Path) -> bool:

    '''Run both comparisons; return whether both criteria pass.'''

    gpuTimes, gpuFronts = gaugeFront(gpuDir)
    cpuTimes, cpuFronts = gaugeFront(cpuDir)
    assert len(gpuTimes) == len(cpuTimes) and np.allclose(gpuTimes, cpuTimes, atol = 1e-3), \
        'The two runs wrote different gauge times.'
    gpuLeadTimes, gpuLead = leadingParticle(gpuDir)
    cpuLeadTimes, cpuLead = leadingParticle(cpuDir)

    # Build: GPU against CPU at every gauge time before the front reaches the end wall
    before = (gpuFronts < wallX - 2.0 * spacing) & (cpuFronts < wallX - 2.0 * spacing)
    buildError = np.abs(gpuFronts[before] - cpuFronts[before]) / cpuFronts[before]
    buildPass = bool(np.all(buildError <= buildTolerance))
    leadBefore = (gpuLead < wallX - 2.0 * spacing) & (cpuLead < wallX - 2.0 * spacing)
    leadError = np.abs(gpuLead[leadBefore] - cpuLead[leadBefore]) / cpuLead[leadBefore]

    # Published: GPU against the experiment at each measured time inside the tank
    measuredTimes, measuredFronts = readExperiment(experimentPath)
    inside = measuredFronts < wallX
    simulated = np.interp(measuredTimes[inside], gpuTimes, gpuFronts)
    leading = np.interp(measuredTimes[inside], gpuLeadTimes, gpuLead)
    publishedError = (simulated - measuredFronts[inside]) / measuredFronts[inside]
    leadingError = (leading - measuredFronts[inside]) / measuredFronts[inside]
    publishedPass = bool(np.all(np.abs(publishedError) <= publishedTolerance))

    print(f'Build: GPU against CPU at {int(before.sum())} gauge times before the wall, largest difference '
          f'{100 * buildError.max():.3f} percent (criterion 1): {"passes" if buildPass else "fails"}; '
          f'leading particle {100 * leadError.max():.3f} percent at {int(leadBefore.sum())} frames')
    print('Published: GPU against Koshizuka and Oka (1996)')
    print('   t [s]   measured [m]   gauge [m]   error [%]   leading particle [m]   error [%]')
    for t, measured, model, error, lead, leadErr in zip(measuredTimes[inside], measuredFronts[inside], simulated,
                                                        publishedError, leading, leadingError):
        print(f'  {t:6.3f}   {measured:10.3f}   {model:9.3f}   {100 * error:+8.2f}   {lead:18.3f}   {100 * leadErr:+8.2f}')
    print(f'  largest {100 * np.abs(publishedError).max():.2f} percent (criterion 10): {"passes" if publishedPass else "fails"}; '
          f'{int((~inside).sum())} measured point(s) past the end wall not compared')

    outputDir.mkdir(parents = True, exist_ok = True)
    result = {
        'front': 'Swl_z003 gauge', 'buildLargestDifference': float(buildError.max()), 'buildGaugeTimes': int(before.sum()),
        'buildPass': buildPass, 'leadingParticleBuildDifference': float(leadError.max()),
        'publishedTimes': [float(t) for t in measuredTimes[inside]], 'publishedErrors': [float(e) for e in publishedError],
        'leadingParticleErrors': [float(e) for e in leadingError],
        'publishedLargestError': float(np.abs(publishedError).max()), 'publishedPass': publishedPass,
    }
    (outputDir / 'damBreakCheck.json').write_text(json.dumps(result, indent = 2), encoding = 'utf-8')

    figure, axis = plt.subplots(figsize = (7, 4.5))
    axis.plot(cpuTimes, cpuFronts, color = '#888888', linewidth = 2.5, label = 'DualSPHysics CPU, gauge at z = 0.03 m')
    axis.plot(gpuTimes, gpuFronts, color = '#E0975A', linewidth = 1.2, label = 'DualSPHysics GPU, gauge at z = 0.03 m')
    axis.plot(gpuLeadTimes, gpuLead, '--', color = '#E0975A', linewidth = 0.8, label = 'DualSPHysics GPU, leading particle')
    axis.plot(measuredTimes, measuredFronts, 'o', color = '#2f6f8f', label = 'Koshizuka and Oka (1996)')
    axis.axhline(wallX, color = 'k', linewidth = 0.8, linestyle = ':')
    axis.set_xlim(0.0, 0.8)
    axis.set_xlabel('t [s]')
    axis.set_ylabel('front x [m]')
    axis.legend(frameon = False)
    figure.tight_layout()
    figure.savefig(outputDir / 'damBreakCheck.png', dpi = 150)
    return buildPass and publishedPass

if __name__ == '__main__':
    arguments = [Path(a) for a in sys.argv[1:]]
    if len(arguments) < 3:
        sys.exit('python checkDamBreak.py <gpuRun> <cpuRun> <experiment file> [outputDir]')
    sys.exit(0 if main(arguments[0], arguments[1], arguments[2], arguments[3] if len(arguments) > 3 else arguments[0]) else 1)
