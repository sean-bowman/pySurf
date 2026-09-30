# -- SPH Smoothing Kernels -- #

'''

Smoothing kernel functions for SPH interpolation.

Implements the cubic spline (M4) kernel in 2D with its gradient.
The kernel provides the weighting function W(r, h) used to
interpolate field quantities from neighboring particles.

Key properties of a valid SPH kernel:
- Normalization: integral of W over the domain = 1
- Compact support: W = 0 for r > support radius
- Positivity: W >= 0 within support
- Delta function limit: W -> delta(r) as h -> 0

References:
-----------
Monaghan (1992) -- Smoothed Particle Hydrodynamics
Monaghan & Lattanzio (1985) -- A refined particle method for
    astrophysical problems

Sean Bowman [02/05/2026]

'''

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

#--------------------------------------------------------------------#
# -- Kernel Protocol -- #
#--------------------------------------------------------------------#

class SphKernel(Protocol):

    '''Protocol for SPH smoothing kernel functions.'''

    def evaluate(self, r: float, h: float) -> float:

        '''

        Evaluate kernel W(r, h).

        Parameters:
        -----------
        r : float
            Distance between particles [m]
        h : float
            Smoothing length [m]

        Returns:
        --------
        float : Kernel value [1/m^dim]

        '''
        ...

    def gradient(self, rVec: np.ndarray, r: float, h: float) -> np.ndarray:

        '''

        Evaluate kernel gradient nabla_W(r, h).

        The gradient points from particle j toward particle i
        (in the direction of rVec = r_i - r_j).

        Parameters:
        -----------
        rVec : np.ndarray
            Vector from particle j to particle i [m]
        r : float
            Distance |rVec| [m]
        h : float
            Smoothing length [m]

        Returns:
        --------
        np.ndarray : Gradient vector [1/m^(dim+1)]

        '''
        ...

    @property
    def dimensions(self) -> int:

        '''Number of spatial dimensions (2 or 3).'''
        ...


#--------------------------------------------------------------------#
# -- Cubic Spline Kernel (M4) -- #
#--------------------------------------------------------------------#

class CubicSplineKernel:

    '''

    Cubic spline (M4) smoothing kernel.

    The most widely used SPH kernel. Piecewise cubic polynomial
    with compact support at q = 2 (where q = r/h).

    W(q) = sigma * {
        1 - (3/2)*q^2 + (3/4)*q^3    for 0 <= q < 1
        (1/4)*(2 - q)^3               for 1 <= q < 2
        0                              for q >= 2
    }

    Normalization constants (sigma):
        2D: sigma = 10 / (7 * pi * h^2)
        3D: sigma = 1 / (pi * h^3)

    Parameters:
    -----------
    dimensions : int
        Number of spatial dimensions (2 or 3)

    '''

    def __init__(self, dimensions: int = 2) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:

        '''Number of spatial dimensions.'''
        return self._dimensions

    def _normalization(self, h: float) -> float:

        '''

        Compute normalization constant sigma for given h.

        Parameters:
        -----------
        h : float
            Smoothing length [m]

        Returns:
        --------
        float : Normalization constant sigma

        '''
        if self._dimensions == 2:
            return 10.0 / (7.0 * math.pi * h * h)
        else:
            return 1.0 / (math.pi * h * h * h)

    def evaluate(self, r: float, h: float) -> float:

        '''

        Evaluate cubic spline kernel W(r, h).

        Parameters:
        -----------
        r : float
            Distance between particles [m]
        h : float
            Smoothing length [m]

        Returns:
        --------
        float : Kernel value [1/m^dim]

        '''
        q = r / h
        sigma = self._normalization(h)

        if q < 1.0:
            # Inner region: 1 - (3/2)*q^2 + (3/4)*q^3
            return sigma * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
        elif q < 2.0:
            # Outer region: (1/4)*(2 - q)^3
            twoMinusQ = 2.0 - q
            return sigma * 0.25 * twoMinusQ * twoMinusQ * twoMinusQ
        else:
            # Beyond support
            return 0.0

    def gradientMagnitude(self, r: float, h: float) -> float:

        '''

        Compute the scalar part of the kernel gradient: dW/dr.

        The full gradient is: grad_W = (dW/dr) * (rVec / r)

        Parameters:
        -----------
        r : float
            Distance between particles [m]
        h : float
            Smoothing length [m]

        Returns:
        --------
        float : dW/dr [1/m^(dim+1)]

        '''
        q = r / h
        sigma = self._normalization(h)

        if q < 1e-12:
            # At r = 0, gradient is zero by symmetry
            return 0.0
        elif q < 1.0:
            # dW/dq = -3*q + (9/4)*q^2
            # dW/dr = dW/dq * (1/h)
            dwdq = -3.0 * q + 2.25 * q * q
            return sigma * dwdq / h
        elif q < 2.0:
            # dW/dq = -(3/4)*(2-q)^2
            twoMinusQ = 2.0 - q
            dwdq = -0.75 * twoMinusQ * twoMinusQ
            return sigma * dwdq / h
        else:
            return 0.0

    def gradient(self, rVec: np.ndarray, r: float, h: float) -> np.ndarray:

        '''

        Evaluate kernel gradient vector nabla_W.

        grad_W = (dW/dr) * (rVec / |rVec|)

        Parameters:
        -----------
        rVec : np.ndarray
            Vector from particle j to particle i (r_i - r_j) [m]
        r : float
            Distance |rVec| [m]
        h : float
            Smoothing length [m]

        Returns:
        --------
        np.ndarray : Gradient vector [1/m^(dim+1)]

        '''
        if r < 1e-12:
            return np.zeros_like(rVec)

        dwdr = self.gradientMagnitude(r, h)
        return dwdr * rVec / r

    #--------------------------------------------------------------------#
    # -- Vectorized (Batch) Operations -- #
    #--------------------------------------------------------------------#

    def evaluateBatch(self, distances: np.ndarray, h: float) -> np.ndarray:

        '''

        Evaluate kernel W(r, h) for an array of distances.

        Fully vectorized using NumPy -- no Python loops.

        Parameters:
        -----------
        distances : np.ndarray
            Array of distances [m], shape (N,)
        h : float
            Smoothing length [m]

        Returns:
        --------
        np.ndarray : Kernel values, shape (N,)

        '''
        q = distances / h
        sigma = self._normalization(h)

        result = np.zeros_like(q)

        # Inner region: q < 1
        inner = q < 1.0
        qInner = q[inner]
        result[inner] = sigma * (1.0 - 1.5 * qInner ** 2 + 0.75 * qInner ** 3)

        # Outer region: 1 <= q < 2
        outer = (q >= 1.0) & (q < 2.0)
        twoMinusQ = 2.0 - q[outer]
        result[outer] = sigma * 0.25 * twoMinusQ ** 3

        return result

    def gradientMagnitudeBatch(self, distances: np.ndarray, h: float) -> np.ndarray:

        '''

        Compute dW/dr for an array of distances.

        Fully vectorized using NumPy.

        Parameters:
        -----------
        distances : np.ndarray
            Array of distances [m], shape (N,)
        h : float
            Smoothing length [m]

        Returns:
        --------
        np.ndarray : dW/dr values, shape (N,)

        '''
        q = distances / h
        sigma = self._normalization(h)

        result = np.zeros_like(q)

        # Inner region: q < 1 (but q > 0)
        inner = (q > 1e-12) & (q < 1.0)
        qInner = q[inner]
        result[inner] = sigma * (-3.0 * qInner + 2.25 * qInner ** 2) / h

        # Outer region: 1 <= q < 2
        outer = (q >= 1.0) & (q < 2.0)
        twoMinusQ = 2.0 - q[outer]
        result[outer] = sigma * (-0.75 * twoMinusQ ** 2) / h

        return result

    def gradientBatch(
        self, drVecs: np.ndarray, distances: np.ndarray, h: float
    ) -> np.ndarray:

        '''

        Evaluate kernel gradient vectors for an array of particle pairs.

        grad_W_k = (dW/dr)_k * (dr_k / |dr_k|)

        Fully vectorized using NumPy.

        Parameters:
        -----------
        drVecs : np.ndarray
            Displacement vectors r_i - r_j, shape (N, dim)
        distances : np.ndarray
            Distances |dr|, shape (N,)
        h : float
            Smoothing length [m]

        Returns:
        --------
        np.ndarray : Gradient vectors, shape (N, dim)

        '''
        dwdr = self.gradientMagnitudeBatch(distances, h)

        # Avoid division by zero
        safeDistances = np.where(distances > 1e-12, distances, 1.0)
        # (dW/dr) * (rVec / |rVec|) for each pair
        gradients = (dwdr / safeDistances)[:, np.newaxis] * drVecs

        # Zero out where distance was zero
        zeroMask = distances < 1e-12
        gradients[zeroMask] = 0.0

        return gradients

#--------------------------------------------------------------------#
# -- Wendland C2 (Quintic) Kernel -- #
#--------------------------------------------------------------------#

class WendlandC2Kernel:

    '''

    Wendland C2 (quintic) smoothing kernel.

    A compactly supported radial basis function with C2 continuity.
    Recommended for 3D simulations due to better stability properties.

    W(q) = sigma * (1 - q/2)^4 * (2*q + 1)  for 0 <= q < 2

    Advantages over cubic spline:
    - Strictly positive second derivative (no tensile instability)
    - Smoother pressure field
    - Better stability for free-surface flows

    Normalization constants (sigma):
        2D: sigma = 7 / (4 * pi * h^2)
        3D: sigma = 21 / (16 * pi * h^3)

    Parameters:
    -----------
    dimensions : int
        Number of spatial dimensions (2 or 3)

    '''

    def __init__(self, dimensions: int = 2) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:

        '''Number of spatial dimensions.'''
        return self._dimensions

    def _normalization(self, h: float) -> float:

        '''

        Compute normalization constant sigma for given h.

        Parameters:
        -----------
        h : float
            Smoothing length [m]

        Returns:
        --------
        float : Normalization constant sigma

        '''
        if self._dimensions == 2:
            return 7.0 / (4.0 * math.pi * h * h)
        else:
            return 21.0 / (16.0 * math.pi * h * h * h)

    def evaluate(self, r: float, h: float) -> float:

        '''

        Evaluate Wendland C2 kernel W(r, h).

        W(q) = sigma * (1 - q/2)^4 * (2*q + 1)  for q < 2

        Parameters:
        -----------
        r : float
            Distance between particles [m]
        h : float
            Smoothing length [m]

        Returns:
        --------
        float : Kernel value [1/m^dim]

        '''
        q = r / h
        if q >= 2.0:
            return 0.0

        sigma = self._normalization(h)
        oneMinusHalfQ = 1.0 - 0.5 * q
        return sigma * (oneMinusHalfQ ** 4) * (2.0 * q + 1.0)

    def gradientMagnitude(self, r: float, h: float) -> float:

        '''

        Compute the scalar part of the kernel gradient: dW/dr.

        dW/dq = sigma * [-2*(1-q/2)^3*(2*q+1) + 2*(1-q/2)^4]
              = sigma * (1-q/2)^3 * [-2*(2*q+1) + 2*(1-q/2)]
              = sigma * (1-q/2)^3 * [-4*q - 2 + 2 - q]
              = sigma * (1-q/2)^3 * (-5*q)
              = -5*sigma*q * (1-q/2)^3

        Parameters:
        -----------
        r : float
            Distance between particles [m]
        h : float
            Smoothing length [m]

        Returns:
        --------
        float : dW/dr [1/m^(dim+1)]

        '''
        q = r / h
        if q < 1e-12 or q >= 2.0:
            return 0.0

        sigma = self._normalization(h)
        oneMinusHalfQ = 1.0 - 0.5 * q
        dwdq = -5.0 * sigma * q * (oneMinusHalfQ ** 3)
        return dwdq / h

    def gradient(self, rVec: np.ndarray, r: float, h: float) -> np.ndarray:

        '''

        Evaluate kernel gradient vector nabla_W.

        grad_W = (dW/dr) * (rVec / |rVec|)

        Parameters:
        -----------
        rVec : np.ndarray
            Vector from particle j to particle i (r_i - r_j) [m]
        r : float
            Distance |rVec| [m]
        h : float
            Smoothing length [m]

        Returns:
        --------
        np.ndarray : Gradient vector [1/m^(dim+1)]

        '''
        if r < 1e-12:
            return np.zeros_like(rVec)

        dwdr = self.gradientMagnitude(r, h)
        return dwdr * rVec / r

    #--------------------------------------------------------------------#
    # -- Vectorized (Batch) Operations -- #
    #--------------------------------------------------------------------#

    def evaluateBatch(self, distances: np.ndarray, h: float) -> np.ndarray:

        '''

        Evaluate kernel W(r, h) for an array of distances.

        Fully vectorized using NumPy -- no Python loops.

        Parameters:
        -----------
        distances : np.ndarray
            Array of distances [m], shape (N,)
        h : float
            Smoothing length [m]

        Returns:
        --------
        np.ndarray : Kernel values, shape (N,)

        '''
        q = distances / h
        sigma = self._normalization(h)

        result = np.zeros_like(q)

        # Active region: q < 2
        active = q < 2.0
        qActive = q[active]
        oneMinusHalfQ = 1.0 - 0.5 * qActive
        result[active] = sigma * (oneMinusHalfQ ** 4) * (2.0 * qActive + 1.0)

        return result

    def gradientMagnitudeBatch(self, distances: np.ndarray, h: float) -> np.ndarray:

        '''

        Compute dW/dr for an array of distances.

        Fully vectorized using NumPy.

        Parameters:
        -----------
        distances : np.ndarray
            Array of distances [m], shape (N,)
        h : float
            Smoothing length [m]

        Returns:
        --------
        np.ndarray : dW/dr values, shape (N,)

        '''
        q = distances / h
        sigma = self._normalization(h)

        result = np.zeros_like(q)

        # Active region: 0 < q < 2
        active = (q > 1e-12) & (q < 2.0)
        qActive = q[active]
        oneMinusHalfQ = 1.0 - 0.5 * qActive
        dwdq = -5.0 * sigma * qActive * (oneMinusHalfQ ** 3)
        result[active] = dwdq / h

        return result

    def gradientBatch(
        self, drVecs: np.ndarray, distances: np.ndarray, h: float
    ) -> np.ndarray:

        '''

        Evaluate kernel gradient vectors for an array of particle pairs.

        grad_W_k = (dW/dr)_k * (dr_k / |dr_k|)

        Fully vectorized using NumPy.

        Parameters:
        -----------
        drVecs : np.ndarray
            Displacement vectors r_i - r_j, shape (N, dim)
        distances : np.ndarray
            Distances |dr|, shape (N,)
        h : float
            Smoothing length [m]

        Returns:
        --------
        np.ndarray : Gradient vectors, shape (N, dim)

        '''
        dwdr = self.gradientMagnitudeBatch(distances, h)

        # Avoid division by zero
        safeDistances = np.where(distances > 1e-12, distances, 1.0)
        # (dW/dr) * (rVec / |rVec|) for each pair
        gradients = (dwdr / safeDistances)[:, np.newaxis] * drVecs

        # Zero out where distance was zero
        zeroMask = distances < 1e-12
        gradients[zeroMask] = 0.0

        return gradients

#--------------------------------------------------------------------#
# -- Kernel Factory -- #
#--------------------------------------------------------------------#

def createKernel(kernelType: str, dimensions: int = 2) -> SphKernel:

    '''

    Create a kernel instance by type name.

    Parameters:
    -----------
    kernelType : str
        Kernel type: 'cubicSpline' or 'wendlandC2'
    dimensions : int
        Number of spatial dimensions (2 or 3)

    Returns:
    --------
    SphKernel : Kernel instance

    Raises:
    -------
    ValueError : If kernel type is unknown

    '''
    if kernelType == 'cubicSpline':
        return CubicSplineKernel(dimensions)
    elif kernelType == 'wendlandC2':
        return WendlandC2Kernel(dimensions)
    else:
        raise ValueError(f'Unknown kernel type: {kernelType}')
