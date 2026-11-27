# -- pySurf: Spray, Foam and Bubbles -- #

'''

Diffuse particles after Ihmsen et al. (2012), run over saved SPH frames as a post-process, so the
fluid is untouched, and drawn over the water surface of `waterSurface`.

    potentials  per fluid particle, over the neighbors within h_w = 2 spacings, each weighted by
                its volume over s^3 so a daughter counts an eighth (Warp hash grid):
                trapped air   v_diff = sum |v_ij| (1 - v_ij . x_ij / |v_ij| |x_ij|) (1 - r/h_w)
                wave crest    curvature sum (1 - n_i . n_j)(1 - r/h_w) over surface neighbors
                              below the particle's tangent plane, counted only where the
                              particle moves along its outward normal (v . n >= 0.6 |v|)
                kinetic       1/2 |v|^2 [J/kg]
                each mapped to 0..1 by Phi(I, tau_min, tau_max) = (min(I, tau_max) - min(I,
                tau_min)) / (tau_max - tau_min).
    normals     -grad of the fluid's volume fraction at the run's own resolution: the volumes
                deposited trilinearly on a grid of one spacing, blurred by a Gaussian whose standard
                deviation matches the run's kernel (Wendland C2 at support 6 s: sigma^2 = R^2 / 15,
                1.55 s), the gradient by central differences sampled trilinearly at each particle.
                A particle is on the surface where the gradient exceeds half its value at a flat
                cubic lattice's top layer, read through the same steps. Within a few spacings of a
                running free surface the particles pack about a fifth denser at the 2 s scale, in
                clusters, so a colour field at h_w flags much of the interior as surface.
    spawning    n_d = I_k (k_ta I_ta + k_wc I_wc) dt per fluid particle, rounded stochastically,
                placed uniformly in the cylinder of the particle's volume radius along v dt and
                launched at v plus the radial offset, as Ihmsen et al.'s section 3.1 states
    classes     by the volume-weighted fluid neighbor count within h_w: spray under 6 (gravity and
                quadratic air drag), foam from 6 to 20 (carried at the fluid's Wendland-averaged
                velocity, its lifetime spent), bubbles over 20 (buoyancy against gravity and
                relaxation to the fluid's velocity)
    drawing     each particle splatted by its projected area into a front layer (spray and foam in
                front of the surface or within a tolerance behind it) or a behind layer (bubbles
                under the surface, faded by Beer's law over their depth), composited as white with
                opacity 1 - exp(-coverage)

One step per saved frame uses only that frame's fluid: splitting and merging change the particle
set between frames, so no fluid particle is followed from one frame to the next.

Ihmsen et al. give tau_ta 5 to 20, tau_wc 2 to 8 and tau_k 5 to 50 for all their scenes at 50
frames per second, and a full neighborhood of about 30 by their neighbor thresholds of 6 and 20;
h_w = 2 spacings holds about 34. pySurf runs at h0 = 1 m in SI, as their scenes do, so tau_ta and
tau_wc apply as published; their particle masses are not given, so tau_k is read per kilogram.
The rates, lifetimes, buoyancy, drag and radii are a look, not physics: a drawing with them keeps
the run's "Not validated" label and says the whitewater is a look.

References:
-----------
Ihmsen, M., Akinci, N., Akinci, G. & Teschner, M. (2012) -- Unified spray, foam and air bubbles for
    particle-based fluids, The Visual Computer 28, 669-677.
Lague, S. -- Fluid-Sim, https://github.com/SebLague/Fluid-Sim (main branch, 2025), spray drag and
    bubble relaxation.
Wendland, H. (1995) -- Piecewise polynomial, positive definite and compactly supported radial
    functions of minimal degree, Advances in Computational Mathematics 4, 389-396.

Sean Bowman [11/26/2026]

'''

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp
from scipy.ndimage import gaussian_filter, map_coordinates

gravity = 9.80665          # [m/s^2]
spray, foam, bubble = 0, 1, 2

#----------------------------------------------------------------------#
# -- Settings -- #
#----------------------------------------------------------------------#

@dataclass
class WhitewaterSettings:

    '''

    Ihmsen et al.'s (2012) constants, scaled to pySurf's spacing.

    Parameters:
    -----------
    spacing : float
        Particle spacing s of the run [m]
    influence : float
        Neighbor radius h_w in spacings
    normalSmoothing : float
        Standard deviation of the Gaussian the normals' volume fraction is blurred by, in
        spacings; the run's kernel (Wendland C2, support 6 s) has 6 / sqrt(15) = 1.55
    trappedAir, waveCrest : tuple
        (tau_min, tau_max) of the trapped-air [m/s] and wave-crest potentials
    kinetic : tuple
        (tau_min, tau_max) of the specific kinetic energy [J/kg]
    trappedAirRate, waveCrestRate : float
        k_ta and k_wc, the most diffuse particles one fluid particle makes per second [1/s]
    alignment : float
        Least v . n / |v| for a wave crest
    sprayBelow, bubblesAbove : float
        Neighbor counts that bound foam
    sprayDrag : float
        Quadratic air drag of spray [1/m]
    buoyancy : float
        Bubbles' upward acceleration over g
    relaxation : float
        Time a bubble takes to follow the fluid's velocity [s]
    lifetime : tuple
        Foam lifetime from the weakest to the strongest generating potential [s]
    radii : tuple
        Drawn radii of spray, foam and bubbles in spacings
    maxParticles : int
        The newest particles kept beyond this count are dropped
    seed : int
        Seed of the sampling

    '''

    spacing: float
    influence: float = 2.0
    normalSmoothing: float = 6.0 / math.sqrt(15.0)
    trappedAir: tuple = (5.0, 20.0)
    waveCrest: tuple = (2.0, 8.0)
    kinetic: tuple = (5.0, 50.0)
    trappedAirRate: float = 1500.0
    waveCrestRate: float = 2500.0
    alignment: float = 0.6
    sprayBelow: float = 6.0
    bubblesAbove: float = 20.0
    sprayDrag: float = 0.04
    buoyancy: float = 0.5
    relaxation: float = 0.05
    lifetime: tuple = (1.0, 4.0)
    radii: tuple = (1.0 / 16.0, 0.1, 0.1)
    maxParticles: int = 8_000_000
    seed: int = 1

def clampPotential(value: np.ndarray, bounds: tuple) -> np.ndarray:
    '''Ihmsen et al.'s eq. 1: Phi(I, tau_min, tau_max), the value mapped to 0..1.'''
    low, high = bounds
    return (np.minimum(value, high) - np.minimum(value, low)) / (high - low)

def volumeFractionGradient(points: np.ndarray, volumes: np.ndarray, spacing: float, smoothing: float) -> np.ndarray:

    '''

    grad of the volume fraction at each point (N, 3) [1/m]: the volumes deposited trilinearly on a
    grid of one spacing around the points, blurred by a Gaussian of standard deviation `smoothing`
    [m], differenced centrally and sampled trilinearly back at the points.

    '''

    pad = 4.0 * smoothing + 2.0 * spacing
    origin = points.min(axis = 0) - pad
    shape = tuple(np.ceil((points.max(axis = 0) + pad - origin) / spacing).astype(int) + 1)
    u = (points - origin) / spacing
    base = np.floor(u).astype(int)
    frac = u - base
    fraction = np.zeros(int(np.prod(shape)))
    for corner in range(8):
        offset = np.array([(corner >> axis) & 1 for axis in range(3)])
        weight = np.prod(np.where(offset, frac, 1.0 - frac), axis = 1) * volumes / spacing**3
        index = np.ravel_multi_index((base + offset).T, shape)
        fraction += np.bincount(index, weights = weight, minlength = fraction.size)
    fraction = gaussian_filter(fraction.reshape(shape), smoothing / spacing, mode = 'constant', truncate = 4.0)
    gradient = np.stack(np.gradient(fraction, spacing), axis = 0)

    return np.column_stack([map_coordinates(gradient[axis], u.T, order = 1, mode = 'nearest') for axis in range(3)])

#----------------------------------------------------------------------#
# -- Kernels -- #
#----------------------------------------------------------------------#

@wp.func
def wendlandC2(r: float, support: float):
    q = r / support
    if q >= 1.0:
        return 0.0
    a = 1.0 - q
    return 21.0 / (2.0 * 3.14159265358979 * support * support * support) * a * a * a * a * (1.0 + 4.0 * q)

@wp.kernel
def fluidPotentials(grid: wp.uint64, points: wp.array(dtype = wp.vec3), velocities: wp.array(dtype = wp.vec3),
                    volumes: wp.array(dtype = float), gradients: wp.array(dtype = wp.vec3), threshold: float,
                    cellVolume: float, influence: float, alignment: float, trappedAir: wp.array(dtype = float),
                    curvature: wp.array(dtype = float), aligned: wp.array(dtype = int),
                    surface: wp.array(dtype = int)):

    i = wp.tid()
    x = points[i]
    v = velocities[i]
    own = gradients[i]
    ownLength = wp.length(own)
    onSurface = ownLength > threshold
    normal = wp.vec3(0.0, 0.0, 0.0)
    if onSurface:
        normal = -own / ownLength
    difference = float(0.0)
    bend = float(0.0)
    query = wp.hash_grid_query(grid, x, influence)
    n = int(0)
    while wp.hash_grid_query_next(query, n):
        offset = x - points[n]
        r = wp.length(offset)
        if r > 1.0e-9 and r < influence:
            weight = volumes[n] / cellVolume * (1.0 - r / influence)
            relative = v - velocities[n]
            speed = wp.length(relative)
            if speed > 1.0e-9:
                # Zero for particles moving apart, two for particles moving toward each other
                difference += weight * speed * (1.0 - wp.dot(relative / speed, offset / r))
            if onSurface:
                other = gradients[n]
                otherLength = wp.length(other)
                # Neighbors below the tangent plane only: x_ji . n_i < 0, a convex surface
                if otherLength > threshold and wp.dot(-offset / r, normal) < 0.0:
                    bend += weight * (1.0 - wp.dot(normal, -other / otherLength))
    trappedAir[i] = difference
    curvature[i] = bend
    surface[i] = int(onSurface)
    aligned[i] = 0
    speed = wp.length(v)
    if onSurface and speed > 1.0e-9:
        if wp.dot(v / speed, normal) >= alignment:
            aligned[i] = 1

@wp.kernel
def diffuseStep(grid: wp.uint64, points: wp.array(dtype = wp.vec3), velocities: wp.array(dtype = wp.vec3),
                volumes: wp.array(dtype = float), cellVolume: float, influence: float, dt: float, gravityValue: float,
                sprayBelow: float, bubblesAbove: float, sprayDrag: float, buoyancy: float, relaxation: float,
                positions: wp.array(dtype = wp.vec3), motion: wp.array(dtype = wp.vec3), lifetime: wp.array(dtype = float),
                kinds: wp.array(dtype = int), counts: wp.array(dtype = float)):

    i = wp.tid()
    x = positions[i]
    u = motion[i]
    count = float(0.0)
    carried = wp.vec3(0.0, 0.0, 0.0)
    weights = float(0.0)
    query = wp.hash_grid_query(grid, x, influence)
    n = int(0)
    while wp.hash_grid_query_next(query, n):
        r = wp.length(x - points[n])
        if r < influence:
            count += volumes[n] / cellVolume
            w = volumes[n] * wendlandC2(r, influence)
            carried += w * velocities[n]
            weights += w
    down = wp.vec3(0.0, 0.0, -gravityValue)
    kind = int(1)
    if count < sprayBelow or weights <= 0.0:
        kind = 0
    elif count > bubblesAbove:
        kind = 2
    if kind == 0:
        # Euler-Cromer under gravity and quadratic drag
        u = u + dt * (down - sprayDrag * wp.length(u) * u)
        x = x + dt * u
    elif kind == 1:
        u = carried / weights
        x = x + dt * u
        lifetime[i] = lifetime[i] - dt
    else:
        u = u + dt * (-buoyancy * down + (carried / weights - u) / relaxation)
        x = x + dt * u
    positions[i] = x
    motion[i] = u
    kinds[i] = kind
    counts[i] = count

@wp.kernel
def splatDiffuse(positions: wp.array(dtype = wp.vec3), kinds: wp.array(dtype = int), radii: wp.vec3, eye: wp.vec3,
                 right: wp.vec3, up: wp.vec3, forward: wp.vec3, focal: float, cx: float, cy: float,
                 surfaceDistance: wp.array2d(dtype = float), tolerance: float, extinction: float,
                 front: wp.array2d(dtype = float), behind: wp.array2d(dtype = float)):

    i = wp.tid()
    offset = positions[i] - eye
    depth = wp.dot(offset, forward)
    if depth < 1.0e-6:
        return
    # Continuous pixel coordinates, pixel (c, r) centred at (c + 0.5, r + 0.5) as in the march
    fx = cx + wp.dot(offset, right) / depth * focal - 0.5
    fy = cy - wp.dot(offset, up) / depth * focal - 0.5
    kind = kinds[i]
    radius = radii[kind] * focal / depth
    spread = wp.max(radius, 0.75)
    reach = wp.min(int(wp.ceil(2.0 * spread)), 12)
    distance = wp.length(offset)
    rows = surfaceDistance.shape[0]
    cols = surfaceDistance.shape[1]
    centreX = int(wp.floor(fx + 0.5))
    centreY = int(wp.floor(fy + 0.5))
    total = float(0.0)
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            ex = float(centreX + dx) - fx
            ey = float(centreY + dy) - fy
            total += wp.exp(-(ex * ex + ey * ey) / (spread * spread))
    # The particle's projected area in pixels, spread over its footprint
    area = 3.14159265358979 * radius * radius
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            px = centreX + dx
            py = centreY + dy
            if px >= 0 and px < cols and py >= 0 and py < rows:
                ex = float(px) - fx
                ey = float(py) - fy
                share = area * wp.exp(-(ex * ex + ey * ey) / (spread * spread)) / total
                ahead = surfaceDistance[py, px]
                if kind == 2 and distance > ahead:
                    wp.atomic_add(behind, py, px, share * wp.exp(-extinction * (distance - ahead)))
                elif distance < ahead + tolerance:
                    wp.atomic_add(front, py, px, share)

#----------------------------------------------------------------------#
# -- Whitewater -- #
#----------------------------------------------------------------------#

class Whitewater:

    '''

    Diffuse particles carried from one saved frame to the next inside a box.

    Parameters:
    -----------
    settings : WhitewaterSettings
        Constants
    box : tuple
        (x0, x1, y0, y1, z0, z1) [m]; fluid inside spawns, diffuse particles leaving it are dropped
    device : str | None
        Warp device; the GPU when there is one

    '''

    def __init__(self, settings: WhitewaterSettings, box: tuple, device: str | None = None) -> None:
        wp.init()
        self.settings = settings
        self.device = device or ('cuda' if wp.get_cuda_device_count() > 0 else 'cpu')
        self.low = np.array(box[0::2], dtype = float)
        self.high = np.array(box[1::2], dtype = float)
        self.influence = settings.influence * settings.spacing                          # [m]
        self.smoothing = settings.normalSmoothing * settings.spacing                    # [m]
        # Fluid within this of the box enters the sums, so the particles inside see a whole neighborhood
        self.margin = max(2.0 * self.influence, 4.0 * self.smoothing) + settings.spacing  # [m]
        self.cellVolume = settings.spacing**3                                            # [m^3]
        self.grid = wp.HashGrid(128, 128, 128, device = self.device)
        self.threshold = 0.5 * self.flatTopGradient()                                   # [1/m]
        self.rng = np.random.default_rng(settings.seed)
        self.positions = np.zeros((0, 3), np.float32)
        self.velocities = np.zeros((0, 3), np.float32)
        self.lifetime = np.zeros(0, np.float32)
        self.kinds = np.zeros(0, np.int32)

    def flatTopGradient(self) -> float:

        '''

        The volume fraction's gradient at the top layer of a flat cubic lattice of the run's
        spacing, read through the same steps as the potentials, clear of the lattice's sides [1/m].

        '''

        s = self.settings.spacing
        count = int(np.ceil(2.0 * self.margin / s)) + 8
        x, y, z = np.meshgrid((np.arange(count) - 0.5 * (count - 1)) * s, (np.arange(count) - 0.5 * (count - 1)) * s,
                              -np.arange(count // 2) * s, indexing = 'ij')
        lattice = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        gradients = volumeFractionGradient(lattice, np.full(len(lattice), s**3), s, self.smoothing)
        middle = (np.abs(lattice[:, 0]) < 2 * s) & (np.abs(lattice[:, 1]) < 2 * s) & (lattice[:, 2] > -0.5 * s)

        return float(np.median(np.linalg.norm(gradients[middle], axis = 1)))

    def _fluid(self, positions: np.ndarray, velocities: np.ndarray, volumes: np.ndarray) -> tuple:

        '''

        The fluid within the margin of the box on the device, the hash grid built over it, which of
        it lies inside, and its volume fraction's gradient.

        '''

        margin = self.margin
        positions = np.asarray(positions, dtype = np.float32)
        near = np.all((positions >= self.low - margin) & (positions <= self.high + margin), axis = 1)
        points = positions[near]
        inside = np.all((points >= self.low) & (points <= self.high), axis = 1)
        sizes = np.asarray(volumes, dtype = float)[near]
        device = self.device
        fluid = (wp.array(points, dtype = wp.vec3, device = device),
                 wp.array(np.asarray(velocities, dtype = np.float32)[near], dtype = wp.vec3, device = device),
                 wp.array(sizes.astype(np.float32), dtype = float, device = device))
        self.grid.build(fluid[0], self.influence)
        gradients = volumeFractionGradient(points.astype(float), sizes, self.settings.spacing, self.smoothing)

        return fluid, inside, gradients

    def potentials(self, positions: np.ndarray, velocities: np.ndarray, volumes: np.ndarray, dt: float) -> dict:

        '''

        Each fluid particle's raw measures, potentials and spawn count n_d over `dt`, for the
        particles within the margin of the box ('inside' marks those in it).

        '''

        settings, device = self.settings, self.device
        fluid, inside, fractionGradient = self._fluid(positions, velocities, volumes)
        self.built = (fluid, inside)
        points, rates, sizes = fluid
        count = points.shape[0]
        gradients = wp.array(fractionGradient.astype(np.float32), dtype = wp.vec3, device = device)
        trappedAir = wp.zeros(count, dtype = float, device = device)
        curvature = wp.zeros(count, dtype = float, device = device)
        aligned = wp.zeros(count, dtype = int, device = device)
        surface = wp.zeros(count, dtype = int, device = device)
        wp.launch(fluidPotentials, dim = count,
                  inputs = [self.grid.id, points, rates, sizes, gradients, self.threshold, self.cellVolume, self.influence,
                            settings.alignment, trappedAir, curvature, aligned, surface], device = device)
        v = rates.numpy().astype(float)
        result = {'positions': points.numpy().astype(float), 'velocities': v, 'volumes': sizes.numpy().astype(float),
                  'inside': inside, 'trappedAir': trappedAir.numpy().astype(float),
                  'curvature': curvature.numpy().astype(float), 'aligned': aligned.numpy().astype(bool),
                  'surface': surface.numpy().astype(bool), 'energy': 0.5 * np.sum(v * v, axis = 1),
                  'gradient': fractionGradient}
        result['airPotential'] = clampPotential(result['trappedAir'], settings.trappedAir)
        result['crestPotential'] = clampPotential(result['curvature'] * result['aligned'], settings.waveCrest)
        result['energyPotential'] = clampPotential(result['energy'], settings.kinetic)
        rate = result['energyPotential'] * (settings.trappedAirRate * result['airPotential']
                                            + settings.waveCrestRate * result['crestPotential'])        # [1/s]
        result['spawnCount'] = np.where(inside, rate * dt, 0.0)
        result['generation'] = rate / (settings.trappedAirRate + settings.waveCrestRate)

        return result

    def spawn(self, measures: dict, dt: float) -> np.ndarray:

        '''Diffuse particles from a `potentials` result, appended; returns their positions.'''

        settings, rng = self.settings, self.rng
        expected = measures['spawnCount']
        counts = np.floor(expected).astype(int)
        counts += rng.random(len(expected)) < (expected - counts)
        source = np.repeat(np.arange(len(expected)), counts)
        if len(source) == 0:
            return np.zeros((0, 3))
        x = measures['positions'][source]
        v = measures['velocities'][source]
        speed = np.linalg.norm(v, axis = 1)
        axis = np.where(speed[:, None] > 1e-9, v / np.maximum(speed, 1e-12)[:, None], np.array([0.0, 0.0, 1.0]))
        # An orthonormal pair across each velocity
        helper = np.where(np.abs(axis[:, 2:3]) < 0.9, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
        first = np.cross(axis, helper)
        first /= np.linalg.norm(first, axis = 1, keepdims = True)
        second = np.cross(axis, first)
        volumeRadius = 0.5 * np.cbrt(measures['volumes'][source])                                 # [m]
        radius = volumeRadius * np.sqrt(rng.random(len(source)))
        angle = 2.0 * math.pi * rng.random(len(source))
        along = rng.random(len(source)) * speed * dt
        across = radius[:, None] * (np.cos(angle)[:, None] * first + np.sin(angle)[:, None] * second)
        positions = x + across + along[:, None] * axis
        # Ihmsen et al.'s launch: the fluid's velocity plus the radial offset taken as a velocity
        velocities = v + across
        low, high = settings.lifetime
        lifetime = low + (high - low) * measures['generation'][source]
        self.positions = np.vstack([self.positions, positions.astype(np.float32)])
        self.velocities = np.vstack([self.velocities, velocities.astype(np.float32)])
        self.lifetime = np.concatenate([self.lifetime, lifetime.astype(np.float32)])
        self.kinds = np.concatenate([self.kinds, np.full(len(source), foam, np.int32)])
        if len(self.lifetime) > settings.maxParticles:
            drop = len(self.lifetime) - settings.maxParticles
            self.positions, self.velocities = self.positions[drop:], self.velocities[drop:]
            self.lifetime, self.kinds = self.lifetime[drop:], self.kinds[drop:]

        return positions

    def advance(self, positions: np.ndarray, velocities: np.ndarray, volumes: np.ndarray, dt: float,
                built: tuple | None = None) -> np.ndarray:

        '''Classify and move the diffuse particles over `dt` through a frame's fluid; returns their neighbor counts.'''

        settings, device = self.settings, self.device
        if len(self.lifetime) == 0:
            return np.zeros(0)
        points, rates, sizes = built[0] if built is not None else self._fluid(positions, velocities, volumes)[0]
        x = wp.array(self.positions, dtype = wp.vec3, device = device)
        u = wp.array(self.velocities, dtype = wp.vec3, device = device)
        life = wp.array(self.lifetime, dtype = float, device = device)
        kinds = wp.array(self.kinds, dtype = int, device = device)
        counts = wp.zeros(len(self.lifetime), dtype = float, device = device)
        wp.launch(diffuseStep, dim = len(self.lifetime),
                  inputs = [self.grid.id, points, rates, sizes, self.cellVolume, self.influence, dt, gravity,
                            settings.sprayBelow, settings.bubblesAbove, settings.sprayDrag, settings.buoyancy,
                            settings.relaxation, x, u, life, kinds, counts], device = device)
        self.positions, self.velocities = x.numpy(), u.numpy()
        self.lifetime, self.kinds = life.numpy(), kinds.numpy()
        counts = counts.numpy()
        keep = (self.lifetime > 0.0) & np.all((self.positions >= self.low) & (self.positions <= self.high), axis = 1)
        self.positions, self.velocities = self.positions[keep], self.velocities[keep]
        self.lifetime, self.kinds = self.lifetime[keep], self.kinds[keep]

        return counts[keep]

    def step(self, positions: np.ndarray, velocities: np.ndarray, volumes: np.ndarray, dt: float) -> dict:

        '''

        One frame interval: the diffuse particles moved through this frame's fluid, then the new
        ones spawned from it. Returns the potentials and the spawned positions.

        '''

        measures = self.potentials(positions, velocities, volumes, dt)
        # The hash grid still holds this frame's fluid, built by `potentials`
        self.advance(None, None, None, dt, built = self.built)
        spawned = self.spawn(measures, dt)

        return {'measures': measures, 'spawned': spawned}

    def splat(self, camera, surfaceDistance: np.ndarray, tolerance: float, extinction: float) -> tuple:

        '''Coverage of the front and behind layers (H, W) from one camera, against the surface's distance per pixel.'''

        device = self.device
        front = wp.zeros(surfaceDistance.shape, dtype = float, device = device)
        behind = wp.zeros(surfaceDistance.shape, dtype = float, device = device)
        if len(self.lifetime) > 0:
            vec = lambda values: wp.vec3(*[float(v) for v in values])
            radii = vec(np.array(self.settings.radii) * self.settings.spacing)
            wp.launch(splatDiffuse, dim = len(self.lifetime),
                      inputs = [wp.array(self.positions, dtype = wp.vec3, device = device),
                                wp.array(self.kinds, dtype = int, device = device), radii, vec(camera.eye),
                                vec(camera.right), vec(camera.upward), vec(camera.forward), camera.focal,
                                camera.centre[0], camera.centre[1],
                                wp.array(surfaceDistance.astype(np.float32), dtype = float, device = device), tolerance,
                                extinction, front, behind], device = device)

        return front.numpy(), behind.numpy()

def composite(image: np.ndarray, front: np.ndarray, behind: np.ndarray, colour: tuple = (0.92, 0.95, 0.98),
              bubbleColour: tuple = (0.78, 0.88, 0.95), gain: float = 1.0) -> np.ndarray:

    '''The surface image with bubbles under it and spray and foam over it, opacity 1 - exp(-gain coverage).'''

    under = (1.0 - np.exp(-gain * behind))[..., None]
    over = (1.0 - np.exp(-gain * front))[..., None]
    image = image * (1.0 - under) + np.array(bubbleColour) * under

    return np.clip(image * (1.0 - over) + np.array(colour) * over, 0.0, 1.0)
