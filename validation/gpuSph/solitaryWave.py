# -- Gate 1: the exact solitary wave down 30 depths of flat bed -- #

'''

The exact potential flow solitary wave of H/h0 = 0.3 (Clamond and Dutykh 2013), laid with its
velocity and pressure in bed-fitted columns, travelling 30 h0 down a flat channel between walls of
six boundary layers, on the 'deltaSph' scheme's settings (Wendland C2 at h = 3 spacings,
hydrostatic δ-SPH density diffusion, the kernel gradient correction, wall pressure extrapolated
from the water, alpha 0.01 with the limited artificial viscosity, no XSPH, density floor 0.9 rho0).
Every quarter sqrt(h0/g) the crest height and position come from H sech²(k (x - x0)) fitted to the
smoothed surface within 3 h0 of the crest.

Criterion: the fitted crest height lost between 1 h0 and 30 h0 of travel under 2 percent. The
crest's speed against the exact wave's is reported without a rule.

Recorded: the crest loses 0.42 to 0.50 percent at 32 particles per depth and 0.76 at 64: passes.

    python solitaryWave.py <ppd>

Writes results/solitaryWave_<ppd>.json.

Sean Bowman [11/19/2026]

'''

import argparse
import math
import time

import numpy as np
from scipy.optimize import curve_fit

import common
from common import verdict, writeResult

from pysurf import constants, waveTheory
from pysurf.bathymetry import Bathymetry
from pysurf.sphBoundaries import BedBoundary
from pysurf.sphParticles import ParticleSystem
from pysurf.sphSolver import SimulationConfig, WcsphSolver, taitDensity
from pysurf.waveBreaking import fluidSurface, freeSurfaceProfile, profileCrest
from pysurf.waveMakers import exactSolitaryBoundaryPressure, exactSolitaryWaveFluid

scheme = dict(densityFilterInterval = 0, densityDiffusion = 'hydrostatic', xsphEpsilon = 0.0, alphaViscosity = 0.01,
              viscosityLimiter = True, wallPressure = 'extrapolated', kernelType = 'wendlandC2',
              continuityCorrection = True, densityFloor = 0.9, smoothingLengthRatio = 3.0)
wallLayers, travel, height, depth = 6, 30.0, 0.3, 1.0

def fittedCrest(positions: np.ndarray, spacing: float, guess: float) -> tuple:

    '''The crest height and position of H sech²(k (x - x0)) fitted to the smoothed surface within 3 h0 of the crest.'''

    lines = fluidSurface(positions, spacing, np.array([guess - 4.0 * depth, 0.5 * depth]),
                         np.array([guess + 4.0 * depth, depth + 1.6 * height]))
    surface = max(lines, key = len)
    eta = surface[:, 1] - depth
    near = (np.abs(surface[:, 0] - guess) < 3.0 * depth) & (eta > 0.05 * height)
    model = lambda x, amplitude, center, decay: amplitude / np.cosh(decay * (x - center))**2
    parameters, _ = curve_fit(model, surface[near, 0], eta[near], p0 = [eta[near].max(), guess, 0.5])
    return float(parameters[0]), float(parameters[1])

def main(ppd: int) -> dict:

    '''Run the channel at `ppd` particles per depth and report the crest's loss against the criterion.'''

    rho0, g = constants.freshwaterDensity, constants.gravity
    spacing = depth / ppd
    crestX = waveTheory.solitaryWaveHalfLength(height, depth, 0.01)
    celerity = waveTheory.exactSolitaryWaveSpeed(height, depth)
    channel = Bathymetry([0.0, crestX + travel + 2.0 * crestX + 2.0], [0.0, 0.0], stillWaterLevel = depth)
    wallTop = depth + 3.0 * height
    boundary = BedBoundary(channel, spacing, nLayers = wallLayers, wallTop = wallTop)
    heights = np.arange(-(wallLayers - 0.5) * spacing, wallTop, spacing)
    farWall = np.vstack([np.column_stack([np.full_like(heights, channel.endX + (layer + 0.5) * spacing), heights])
                         for layer in range(wallLayers)])
    fluid, velocities, pressures = exactSolitaryWaveFluid(channel, height, crestX, spacing, bedFitted = True)
    walls = np.vstack([boundary.positions(), farWall])
    particles = ParticleSystem.assemble(fluid, walls, spacing, rho0, velocities)
    config = SimulationConfig(domainMin = np.array([0.0, -(wallLayers + 1.0) * spacing]),
                              domainMax = np.array([channel.endX + (wallLayers + 1.0) * spacing, wallTop]),
                              particleSpacing = spacing, referenceDensity = rho0, maxTimeStep = 1.0,
                              speedOfSound = 10.0 * celerity, backend = 'gpu', **scheme)
    solver = WcsphSolver(config, boundary = boundary)
    solver.initialize(particles)
    # The start carries the exact wave's pressure, in the fluid and in the walls
    isFluid = particles.isFluid
    particles.densities[isFluid] = taitDensity(rho0 * pressures, rho0, solver.speedOfSound)
    wallPressure = exactSolitaryBoundaryPressure(channel, walls, height, crestX, spacing, g)
    particles.densities[~isFluid] = taitDensity(rho0 * wallPressure, rho0, solver.speedOfSound)

    scale = math.sqrt(depth / g)
    samples = []
    started = time.perf_counter()
    target = 0.0
    while True:
        positions = particles.positions[particles.isFluid]
        centers, elevation = freeSurfaceProfile(positions, channel.startX, channel.endX, spacing)
        guess = profileCrest(centers, np.nan_to_num(elevation - depth))
        fitted, center = fittedCrest(positions, spacing, guess)
        samples.append((solver.time / scale, center, fitted))
        if center - crestX >= travel or center > channel.endX - 2.0 * crestX:
            break
        target += 0.25 * scale
        solver.advance(target, maxSteps = 20000)
        fastest = float(np.max(np.linalg.norm(particles.velocities[particles.isFluid], axis = 1))) / math.sqrt(g * depth)
        if solver.time < target - 1e-9 or not math.isfinite(fastest) or fastest > 5.0:
            raise RuntimeError(f'The run diverged at {solver.time / scale:.2f} sqrt(h0/g) (fastest {fastest:.2f} sqrt(g h0)).')
    samples = np.array(samples)
    travelled = samples[:, 1] - crestX
    first = int(np.argmin(np.abs(travelled - 1.0)))
    heightAt = lambda distance: float(np.interp(distance, travelled, samples[:, 2]))
    loss = 100.0 * (1.0 - heightAt(travel) / heightAt(1.0))
    speedRatio = float(np.polyfit(samples[first:, 0], samples[first:, 1], 1)[0]) / waveTheory.exactSolitaryWave(height).froudeNumber
    passes = loss < 2.0
    print(f'{ppd} ppd: crest height lost from 1 to {travel:.0f} h0 of travel {loss:.2f} percent (limit 2): {verdict(passes)}; '
          f'speed {speedRatio:.4f} of the exact wave\'s; {particles.nFluid} particles, {time.perf_counter() - started:.0f} s',
          flush = True)
    record = {'case': 'solitaryWave', 'particlesPerDepth': ppd, 'fluidParticles': int(particles.nFluid),
              'samples': samples.tolist(), 'verdict': {'heightLostPercent': loss, 'speedRatio': speedRatio, 'passes': passes}}
    writeResult(f'solitaryWave_{ppd}', record)
    return record

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Gate 1: the exact solitary wave down 30 depths of flat bed.')
    parser.add_argument('ppd', type = int, help = 'particles per depth')
    main(parser.parse_args().ppd)
