
# -- pySurf: SPH Kernel Tests -- #

'''

Checks of pysurf.sphKernels against the properties every SPH kernel must have: unit integral over
its support in 2D and 3D, zero at and beyond 2h, non-negative inside it, batch forms equal to the
pair forms, and an analytic gradient equal to a central difference of the kernel.

Sean Bowman [10/11/2026]

'''

import math

import numpy as np
import pytest
from scipy.integrate import quad

from pysurf.sphKernels import CubicSplineKernel, WendlandC2Kernel, createKernel

h = 0.013
kernelClasses = [CubicSplineKernel, WendlandC2Kernel]

@pytest.mark.parametrize('kernelClass', kernelClasses)
@pytest.mark.parametrize('dimensions', [2, 3])
def testTheKernelIntegratesToOne(kernelClass, dimensions):

    kernel = kernelClass(dimensions)
    shell = (lambda r: 2.0 * math.pi * r) if dimensions == 2 else (lambda r: 4.0 * math.pi * r**2)
    integral, _ = quad(lambda r: kernel.evaluate(r, h) * shell(r), 0.0, 2.0 * h, points = [h], epsabs = 1e-12)

    assert integral == pytest.approx(1.0, abs = 1e-6)

@pytest.mark.parametrize('kernelClass', kernelClasses)
def testTheSupportEndsAtTwoH(kernelClass):

    kernel = kernelClass(2)
    r = np.linspace(0.0, 1.999 * h, 200)

    assert kernel.evaluate(2.0 * h, h) == pytest.approx(0.0, abs = 1e-12)
    assert kernel.evaluate(2.5 * h, h) == 0.0
    assert kernel.gradientMagnitude(2.5 * h, h) == 0.0
    assert np.all(kernel.evaluateBatch(r, h) > 0.0)

@pytest.mark.parametrize('kernelClass', kernelClasses)
def testBatchFormsMatchPairForms(kernelClass):

    kernel = kernelClass(2)
    rng = np.random.default_rng(3)
    separations = rng.uniform(-2.2 * h, 2.2 * h, size = (50, 2))
    distances = np.linalg.norm(separations, axis = 1)

    assert kernel.evaluateBatch(distances, h) == pytest.approx([kernel.evaluate(r, h) for r in distances])
    assert kernel.gradientMagnitudeBatch(distances, h) == pytest.approx([kernel.gradientMagnitude(r, h) for r in distances])
    assert kernel.gradientBatch(separations, distances, h) \
        == pytest.approx(np.array([kernel.gradient(v, r, h) for v, r in zip(separations, distances)]))

@pytest.mark.parametrize('kernelClass', kernelClasses)
@pytest.mark.parametrize('dimensions', [2, 3])
def testTheGradientIsTheDerivativeOfTheKernel(kernelClass, dimensions):

    kernel = kernelClass(dimensions)
    step = 1e-7 * h
    for r in np.array([0.1, 0.5, 0.9, 1.1, 1.5, 1.9]) * h:
        derivative = (kernel.evaluate(r + step, h) - kernel.evaluate(r - step, h)) / (2.0 * step)
        assert kernel.gradientMagnitude(r, h) == pytest.approx(derivative, rel = 1e-5)

@pytest.mark.parametrize('kernelClass', kernelClasses)
def testTheGradientPointsAlongTheSeparation(kernelClass):

    '''W falls with distance, so grad W with respect to particle i points from i toward j.'''

    kernel = kernelClass(2)
    separation = np.array([0.6 * h, 0.8 * h])
    gradient = kernel.gradient(separation, h, h)

    assert gradient[0] * separation[1] - gradient[1] * separation[0] == pytest.approx(0.0, abs = 1e-9)
    assert np.dot(gradient, separation) < 0.0
    assert kernel.gradientBatch(np.zeros((1, 2)), np.zeros(1), h) == pytest.approx(np.zeros((1, 2)))

def testTheFactoryBuildsByName():

    assert isinstance(createKernel('cubicSpline', 3), CubicSplineKernel)
    assert isinstance(createKernel('wendlandC2'), WendlandC2Kernel)
    assert createKernel('cubicSpline', 3).dimensions == 3
    with pytest.raises(ValueError):
        createKernel('gaussian')
