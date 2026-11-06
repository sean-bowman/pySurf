# -- Validation: The GPU Backend Against the CPU Backends -- #

'''

The GPU backend (`pysurf.sphGpu`, `backend = 'gpu'`) on the validation cases of the CPU backends.
It runs the same equations with float32 pair arithmetic, so each case must pass its own published
criterion and agree with the CPU backend.

    case                                    reference                       agreement with the CPU
    first-mode sloshing period              linear theory, within 3 %       numpy, rerun: 1 %
    solitary wave celerity, 12 ppd          sqrt(g (d + H)), within 3 %     numpy, rerun: 1 %, and
                                                                            the end crest height 1 %
    energy lost in active breaking, 48 ppd  Mostert and Deike (2020), 30 %  numpy, recorded: share
                                                                            5.5 %, scaled rate 9 %
    peeling breaker, 30 degrees, 16 ppd     steady peel (consistency)       compiled, rerun: each
                                                                            age's median plane
                                                                            energy 1 %

The agreement criteria were set before the runs at 1 percent throughout. The breaking case failed
it, and the criterion for that case was revised after measuring why. The breaking flow amplifies
rounding: the same run on the numpy, compiled and GPU backends, with BLAS threading changed in the
boundary element solve that starts it, and on the GPU from starts jittered by 1e-9 of a spacing,
gives seven shares lost between 0.280 and 0.290 (one standard deviation 1.3 percent) and, of the
six recorded, scaled rates between 0.00206 and 0.00216 (2.1 percent). The numpy and compiled backends, the same
equations summed in different orders, are 2.9 and 5.0 percent apart. Two runs of this flow are
independent draws from that spread, so they differ with a standard deviation of sqrt(2) times it;
the revised criteria are three of those: 5.5 percent on the share and 9 percent on the rate, against
the numpy result of the current code with single-threaded BLAS (share 0.2816, rate 0.002058).

Results, RTX 2070 Super, against the CPU backend's:

    case                        GPU             CPU             difference
    sloshing period             0.6866 s        0.68616 s       +0.06 %     (0.978 of linear theory)
    celerity ratio              0.99211         0.99222         -0.01 %
    end crest height            0.27617 d       0.27650 d       -0.12 %
    share lost in breaking      0.2799          0.2816          -0.6 %      (five GPU runs 0.2799 to 0.2882)
    scaled dissipation rate     0.002057        0.002058        -0.05 %
    peel, median plane energy   1.01271         1.01271         +0.000 %    at 0.25 sqrt(h0 / g)
                                1.02321         1.02321         -0.000 %    at 0.5
                                1.03396         1.03396         +0.000 %    at 0.8

The peel's 16 ppd run takes 36 s on the GPU and 236 s on the compiled backend, both stepping from
the host.

Run with `pytest -m slow`; skipped without Warp.

Sean Bowman [11/05/2026]

'''

import numpy as np
import pytest

pytest.importorskip('warp')

from testValidationPeelingWave import ages, newestPlanes, peelingRun
from testValidationSloshing import measureSloshingPeriod
from testValidationSolitary import measureCelerity
from testValidationSplashUp import mostertBreaking, mostertShareLost

numpyShareLost = 0.2816             # [-] numpy backend, 48 particles per depth, single-threaded BLAS
numpyScaledRate = 0.002058          # [-] (epsilon / epsilon0) (db / d0), same run
shareSpread, rateSpread = 0.013, 0.021      # [-] one standard deviation of seven rounding-level variants

@pytest.mark.slow
def testTheTankSloshesAtTheFirstModePeriod():

    gpu, cpu = measureSloshingPeriod(backend = 'gpu'), measureSloshingPeriod()

    assert abs(gpu['ratio'] - 1.0) < 0.03, gpu
    assert abs(gpu['period'] / cpu['period'] - 1.0) < 0.01, (gpu, cpu)

@pytest.mark.slow
def testTheSolitaryWaveTravelsAtItsCelerity():

    gpu, cpu = measureCelerity(12, backend = 'gpu'), measureCelerity(12)

    assert abs(gpu['ratio'] - 1.0) < 0.03, gpu
    assert abs(gpu['ratio'] / cpu['ratio'] - 1.0) < 0.01, (gpu, cpu)
    assert abs(gpu['endHeight'] / cpu['endHeight'] - 1.0) < 0.01, (gpu, cpu)

@pytest.mark.slow
def testTheBreakerLosesTheEnergyTheCpuBackendDoes():

    result = mostertBreaking(backend = 'gpu')

    assert result['activeEndReached'], result
    assert abs(result['energyShareLost'] / mostertShareLost - 1.0) < 0.30, result
    assert abs(result['energyShareLost'] / numpyShareLost - 1.0) < 3.0 * np.sqrt(2.0) * shareSpread, result
    assert abs(result['scaledDissipation'] / numpyScaledRate - 1.0) < 3.0 * np.sqrt(2.0) * rateSpread, result

@pytest.mark.slow
def testThePeelFollowsTheCompiledPeel():

    gpu, cpu = peelingRun(1.0, backend = 'gpu'), peelingRun(1.0)
    for age in ages:
        gpuMedian, cpuMedian = np.median(newestPlanes(gpu, age)), np.median(newestPlanes(cpu, age))
        assert abs(gpuMedian / cpuMedian - 1.0) < 0.01, (age, gpuMedian, cpuMedian)
