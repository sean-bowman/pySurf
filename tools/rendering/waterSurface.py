# -- pySurf: Water Surface Rendering -- #

'''

Saved SPH frames drawn as a water surface: the particles' volumes spread over a grid by a smooth
kernel, and the grid ray marched on the GPU, as the Fluid-Sim project's main branch marches its
density volume (NVIDIA Warp; Warp's CPU device stands in where there is no GPU).

    field       each grid point gathers V_j W(|x - x_j|) over the particles within the kernel's
                support R, through Warp's hash grid, W the Wendland C2 kernel. Inside the water the
                field is 1; over a flat surface laid as a lattice it falls through 0.5 at the top
                layer plus half a spacing, the middle of the band the free surface lies in, so the
                surface is the field's 0.5 level. A second grid carries the particles' speed
                averaged under the same kernel, and with `normalRadius` a third the same volumes
                under a wider kernel, for the normals only.
    march       each pixel's ray through the grid to the first crossing of 0.5, refined by
                bisection; the normal from the gradient of the wider field (of the surface's own
                field without one); the thickness as the field integrated along the ray beyond the
                surface. A ray entering the box inside the water draws the box's cut face.

Particles laid in bed-fitted columns carry the free surface as a staircase of one spacing s. A
field reaching R = 3 s sees at most one step on a gentle face, so it draws flat treads and risers
tilted 20 to 30 degrees, which show as stripes along the crest. The surface keeps R = 3 s so thin
sheets keep their place; the normals from a field at R_n = 6 s spread each riser over more of
its tread.
    shade       Schlick's Fresnel for water (n 1.33) weighting a sky gradient's reflection against
                the water's colour, set by the speed and faded by Beer's law through the thickness;
                wrapped diffuse light from a sun

A lone particle of one spacing's volume peaks at 21 / (54 pi) = 0.124 under R = 3 spacings, below
the surface level: isolated drops and sheets thinner than about the kernel's reach do not draw, and
belong to a spray layer. The renderer draws; it does not measure, and a rendered run keeps its own
validation label.

Checks (tests/testWaterSurfaceRender.py, 2026-11-26): one particle's level sphere within 0.1 s of
its closed-form radius; a flat slab drawn at its surface within 0.25 s, normals within 5 degrees.
On the exact solitary wave in bed-fitted columns at 32 per depth, R = 3 s, R_n = 6 s:

    H/h0 = 0.3 (calibration: R_n chosen on it)   offset -0.44 to +0.45 s, median +0.04 s;
                                                 normals 9.6 deg at the 95th percentile, 13.4 max
    H/h0 = 0.5 (held out, second camera)         offset -0.31 to +0.34 s, median +0.03 s;
                                                 normals 5.5 deg at the 95th percentile, 9.2 max

against bounds of 1 s, 0.25 s median, 10 deg at the 95th percentile and 15 deg at all. Without the
wider field the held-out normals read 18.1 and 24.2 deg.

References:
-----------
Zhu, Y. & Bridson, R. (2005) -- Animating sand as a fluid, ACM Transactions on Graphics 24(3),
    965-972 (the particle field and its level set).
Wendland, H. (1995) -- Piecewise polynomial, positive definite and compactly supported radial
    functions of minimal degree, Advances in Computational Mathematics 4, 389-396.
Green, S. (2010) -- Screen Space Fluid Rendering for Games, Game Developers Conference, NVIDIA
    (Fresnel and Beer's law shading).
Lague, S. -- Fluid-Sim, https://github.com/SebLague/Fluid-Sim (main branch, 2025), density volume
    and ray marching.

Sean Bowman [11/26/2026]

'''

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import warp as wp

#----------------------------------------------------------------------#
# -- Camera -- #
#----------------------------------------------------------------------#

@dataclass
class Camera:

    '''

    A pinhole camera.

    Parameters:
    -----------
    eye : array_like
        Camera position (3,)
    target : array_like
        Point the camera looks at (3,)
    width, height : int
        Image size [px]
    fieldOfView : float
        Vertical field of view [deg]
    up : array_like
        World direction that appears up (3,)

    '''

    eye: np.ndarray
    target: np.ndarray
    width: int
    height: int
    fieldOfView: float = 30.0
    up: np.ndarray = field(default_factory = lambda: np.array([0.0, 0.0, 1.0]))

    def __post_init__(self) -> None:
        self.eye = np.asarray(self.eye, dtype = float)
        self.target = np.asarray(self.target, dtype = float)
        forward = self.target - self.eye
        self.forward = forward / np.linalg.norm(forward)
        right = np.cross(self.forward, np.asarray(self.up, dtype = float))
        self.right = right / np.linalg.norm(right)
        self.upward = np.cross(self.right, self.forward)
        self.focal = 0.5 * self.height / math.tan(math.radians(0.5 * self.fieldOfView))      # [px]
        self.centre = (0.5 * self.width, 0.5 * self.height)

    @classmethod
    def orbit(cls, target, distance: float, elevation: float, azimuth: float, width: int, height: int,
              fieldOfView: float = 30.0) -> Camera:

        '''A camera `distance` from `target`, `elevation` degrees above the horizontal, at `azimuth` degrees from +x.'''

        e, a = math.radians(elevation), math.radians(azimuth)
        offset = distance * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])

        return cls(eye = np.asarray(target, dtype = float) + offset, target = target, width = width, height = height,
                   fieldOfView = fieldOfView)

#----------------------------------------------------------------------#
# -- Kernels -- #
#----------------------------------------------------------------------#

@wp.func
def wendland(r: float, support: float):
    '''The 3D Wendland C2 kernel of support radius R: 21 / (2 pi R^3) (1 - q)^4 (1 + 4 q), q = r / R.'''
    q = r / support
    if q >= 1.0:
        return 0.0
    a = 1.0 - q
    return 21.0 / (2.0 * 3.14159265358979 * support * support * support) * a * a * a * a * (1.0 + 4.0 * q)

@wp.kernel
def gatherField(grid: wp.uint64, points: wp.array(dtype = wp.vec3), volumes: wp.array(dtype = float),
                speeds: wp.array(dtype = float), origin: wp.vec3, cell: float, support: float,
                volumeField: wp.array3d(dtype = float), speedField: wp.array3d(dtype = float)):

    i, j, k = wp.tid()
    x = origin + wp.vec3(float(i), float(j), float(k)) * cell
    total = float(0.0)
    weighted = float(0.0)
    query = wp.hash_grid_query(grid, x, support)
    n = int(0)
    while wp.hash_grid_query_next(query, n):
        w = volumes[n] * wendland(wp.length(x - points[n]), support)
        total += w
        weighted += w * speeds[n]
    volumeField[i, j, k] = total
    if total > 1.0e-6:
        speedField[i, j, k] = weighted / total
    else:
        speedField[i, j, k] = 0.0

@wp.kernel
def gatherVolume(grid: wp.uint64, points: wp.array(dtype = wp.vec3), volumes: wp.array(dtype = float), origin: wp.vec3,
                 cell: float, support: float, volumeField: wp.array3d(dtype = float)):

    i, j, k = wp.tid()
    x = origin + wp.vec3(float(i), float(j), float(k)) * cell
    total = float(0.0)
    query = wp.hash_grid_query(grid, x, support)
    n = int(0)
    while wp.hash_grid_query_next(query, n):
        total += volumes[n] * wendland(wp.length(x - points[n]), support)
    volumeField[i, j, k] = total

@wp.func
def sampleField(values: wp.array3d(dtype = float), origin: wp.vec3, cell: float, p: wp.vec3):
    '''Trilinear sample of a grid; zero outside it.'''
    u = (p - origin) / cell
    i0 = int(wp.floor(u[0]))
    j0 = int(wp.floor(u[1]))
    k0 = int(wp.floor(u[2]))
    if i0 < 0 or j0 < 0 or k0 < 0 or i0 >= values.shape[0] - 1 or j0 >= values.shape[1] - 1 or k0 >= values.shape[2] - 1:
        return 0.0
    fx = u[0] - float(i0)
    fy = u[1] - float(j0)
    fz = u[2] - float(k0)
    c00 = values[i0, j0, k0] * (1.0 - fx) + values[i0 + 1, j0, k0] * fx
    c10 = values[i0, j0 + 1, k0] * (1.0 - fx) + values[i0 + 1, j0 + 1, k0] * fx
    c01 = values[i0, j0, k0 + 1] * (1.0 - fx) + values[i0 + 1, j0, k0 + 1] * fx
    c11 = values[i0, j0 + 1, k0 + 1] * (1.0 - fx) + values[i0 + 1, j0 + 1, k0 + 1] * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz

@wp.kernel
def marchRays(volumeField: wp.array3d(dtype = float), normalField: wp.array3d(dtype = float),
              speedField: wp.array3d(dtype = float), origin: wp.vec3,
              cell: float, boxLow: wp.vec3, boxHigh: wp.vec3, eye: wp.vec3, right: wp.vec3, up: wp.vec3,
              forward: wp.vec3, focal: float, cx: float, cy: float, step: float, level: float,
              hits: wp.array2d(dtype = wp.vec3), normals: wp.array2d(dtype = wp.vec3),
              thickness: wp.array2d(dtype = float), speed: wp.array2d(dtype = float),
              found: wp.array2d(dtype = int)):

    py, px = wp.tid()
    found[py, px] = 0
    thickness[py, px] = 0.0
    direction = wp.normalize(forward + right * ((float(px) + 0.5 - cx) / focal) - up * ((float(py) + 0.5 - cy) / focal))
    # Where the ray meets the grid's box, and the face it enters through
    tNear = float(0.0)
    tFar = float(1.0e9)
    entryAxis = int(-1)
    for axis in range(3):
        if wp.abs(direction[axis]) < 1.0e-12:
            if eye[axis] < boxLow[axis] or eye[axis] > boxHigh[axis]:
                return
        else:
            t1 = (boxLow[axis] - eye[axis]) / direction[axis]
            t2 = (boxHigh[axis] - eye[axis]) / direction[axis]
            if wp.min(t1, t2) > tNear:
                tNear = wp.min(t1, t2)
                entryAxis = axis
            tFar = wp.min(tFar, wp.max(t1, t2))
    if tFar <= tNear:
        return
    # A box that crops the water: a ray entering inside it draws the cut face
    tHit = float(-1.0)
    normal = wp.vec3(0.0, 0.0, 0.0)
    t = tNear
    before = sampleField(volumeField, origin, cell, eye + direction * t)
    if before >= level and entryAxis >= 0:
        tHit = tNear
        if entryAxis == 0:
            normal = wp.vec3(-wp.sign(direction[0]), 0.0, 0.0)
        elif entryAxis == 1:
            normal = wp.vec3(0.0, -wp.sign(direction[1]), 0.0)
        else:
            normal = wp.vec3(0.0, 0.0, -wp.sign(direction[2]))
    while tHit < 0.0 and t < tFar:
        tNext = wp.min(t + step, tFar)
        after = sampleField(volumeField, origin, cell, eye + direction * tNext)
        if before < level and after >= level:
            low = t
            high = tNext
            for _ in range(12):
                middle = 0.5 * (low + high)
                if sampleField(volumeField, origin, cell, eye + direction * middle) < level:
                    low = middle
                else:
                    high = middle
            tHit = 0.5 * (low + high)
            p = eye + direction * tHit
            gradient = wp.vec3(
                sampleField(normalField, origin, cell, p + wp.vec3(cell, 0.0, 0.0))
                - sampleField(normalField, origin, cell, p - wp.vec3(cell, 0.0, 0.0)),
                sampleField(normalField, origin, cell, p + wp.vec3(0.0, cell, 0.0))
                - sampleField(normalField, origin, cell, p - wp.vec3(0.0, cell, 0.0)),
                sampleField(normalField, origin, cell, p + wp.vec3(0.0, 0.0, cell))
                - sampleField(normalField, origin, cell, p - wp.vec3(0.0, 0.0, cell)))
            normal = -wp.normalize(gradient)
        before = after
        t = tNext
    if tHit < 0.0:
        return
    hit = eye + direction * tHit
    hits[py, px] = hit
    normals[py, px] = normal
    speed[py, px] = sampleField(speedField, origin, cell, hit)
    found[py, px] = 1
    # The water the ray crosses beyond the surface, the field integrated to the box's far side
    path = float(0.0)
    s = tHit
    while s < tFar:
        path += wp.min(1.0, sampleField(volumeField, origin, cell, eye + direction * s)) * step
        s += step
    thickness[py, px] = path

#----------------------------------------------------------------------#
# -- Renderer -- #
#----------------------------------------------------------------------#

@dataclass
class Look:

    '''

    The renderer's settings.

    Parameters:
    -----------
    supportRadius : float
        The kernel's support R [m]; three particle spacings
    normalRadius : float | None
        Support R_n of the wider field the normals come from [m]; six spacings, calibrated on the
        exact solitary wave of H/h0 = 0.3. None takes the normals from the surface's own field
    cellSize : float
        Grid spacing [m]; half a particle spacing
    marchStep : float
        Ray step as a share of the cell
    level : float
        The field's surface level
    speedScale : float
        Speed that draws white [m/s]
    extinction : tuple
        Beer's-law coefficients per colour channel [1/m]
    sun : tuple
        Direction toward the sun
    sunSpecular, shininess : float
        Strength and Blinn-Phong exponent of the sun's highlight

    '''

    supportRadius: float
    cellSize: float
    normalRadius: float | None = None
    marchStep: float = 0.5
    level: float = 0.5
    speedScale: float = 1.0
    extinction: tuple = (6.0, 3.0, 1.5)
    deepColour: tuple = (0.03, 0.10, 0.22)
    skyHorizon: tuple = (0.55, 0.65, 0.78)
    skyZenith: tuple = (0.10, 0.20, 0.38)
    background: tuple = (0.055, 0.082, 0.133)
    sun: tuple = (0.3, -0.4, 0.87)
    sunSpecular: float = 0.45
    shininess: float = 160.0
    speedColours: tuple = ((0.114, 0.310, 0.569), (0.180, 0.482, 0.776), (0.498, 0.722, 0.910), (1.0, 1.0, 1.0))

class WaterSurfaceRenderer:

    '''

    Draws particles inside a box as a water surface from one camera.

    Parameters:
    -----------
    camera : Camera
        The camera
    look : Look
        Field and shading settings
    box : tuple
        (x0, x1, y0, y1, z0, z1) of the grid [m]; particles within the kernel's reach of it count
    device : str | None
        Warp device; the GPU when there is one

    '''

    def __init__(self, camera: Camera, look: Look, box: tuple, device: str | None = None) -> None:
        wp.init()
        self.camera, self.look = camera, look
        self.device = device or ('cuda' if wp.get_cuda_device_count() > 0 else 'cpu')
        self.low = np.array(box[0::2], dtype = float)
        self.high = np.array(box[1::2], dtype = float)
        self.shape = tuple(int(math.floor((self.high[a] - self.low[a]) / look.cellSize)) + 1 for a in range(3))
        self.high = self.low + (np.array(self.shape) - 1) * look.cellSize
        cells = max(1, int(round(max(self.high - self.low) / look.supportRadius)))
        self.grid = wp.HashGrid(min(cells, 256), min(cells, 256), min(cells, 256), device = self.device)
        self.volumeField = wp.zeros(self.shape, dtype = float, device = self.device)
        self.speedField = wp.zeros(self.shape, dtype = float, device = self.device)
        self.normalField = (self.volumeField if look.normalRadius is None
                            else wp.zeros(self.shape, dtype = float, device = self.device))
        image = (camera.height, camera.width)
        self.hits = wp.zeros(image, dtype = wp.vec3, device = self.device)
        self.normals = wp.zeros(image, dtype = wp.vec3, device = self.device)
        self.thickness = wp.zeros(image, dtype = float, device = self.device)
        self.speed = wp.zeros(image, dtype = float, device = self.device)
        self.found = wp.zeros(image, dtype = int, device = self.device)

    def field(self, positions: np.ndarray, volumes: np.ndarray, speeds: np.ndarray | None = None) -> None:

        '''Gather the volume, speed and normal fields from the particles within the kernels' reach of the box.'''

        look, device = self.look, self.device
        reach = max(look.supportRadius, look.normalRadius or 0.0)
        positions = np.asarray(positions, dtype = float)
        keep = np.all((positions >= self.low - reach) & (positions <= self.high + reach), axis = 1)
        points = wp.array(positions[keep].astype(np.float32), dtype = wp.vec3, device = device)
        sizes = wp.array(np.asarray(volumes, dtype = np.float32)[keep], dtype = float, device = device)
        values = np.zeros(int(keep.sum()), np.float32) if speeds is None else np.asarray(speeds, dtype = np.float32)[keep]
        rates = wp.array(values, dtype = float, device = device)
        self.grid.build(points, look.supportRadius)
        wp.launch(gatherField, dim = self.shape, inputs = [self.grid.id, points, sizes, rates, wp.vec3(*self.low),
                                                           look.cellSize, look.supportRadius, self.volumeField,
                                                           self.speedField], device = device)
        if look.normalRadius is not None:
            self.grid.build(points, look.normalRadius)
            wp.launch(gatherVolume, dim = self.shape, inputs = [self.grid.id, points, sizes, wp.vec3(*self.low),
                                                                look.cellSize, look.normalRadius, self.normalField],
                      device = device)

    def march(self) -> dict:

        '''March every pixel's ray to the surface: hit points, normals, thickness and speed there.'''

        camera, look, device = self.camera, self.look, self.device
        vec = lambda values: wp.vec3(*[float(v) for v in values])
        wp.launch(marchRays, dim = (camera.height, camera.width),
                  inputs = [self.volumeField, self.normalField, self.speedField, vec(self.low), look.cellSize, vec(self.low), vec(self.high),
                            vec(camera.eye), vec(camera.right), vec(camera.upward), vec(camera.forward), camera.focal,
                            camera.centre[0], camera.centre[1], look.marchStep * look.cellSize, look.level,
                            self.hits, self.normals, self.thickness, self.speed, self.found], device = device)
        found = self.found.numpy().astype(bool)
        hits = self.hits.numpy()
        hits[~found] = np.nan

        return {'found': found, 'hits': hits, 'normals': self.normals.numpy(), 'thickness': self.thickness.numpy(),
                'speed': self.speed.numpy()}

    def shade(self, surface: dict) -> np.ndarray:

        '''The image (H, W, 3) in 0 to 1 from a march's surface.'''

        look, camera = self.look, self.camera
        found, normals = surface['found'], surface['normals']
        rows, cols = np.mgrid[0:camera.height, 0:camera.width]
        view = (camera.forward + ((cols + 0.5 - camera.centre[0]) / camera.focal)[..., None] * camera.right
                - ((rows + 0.5 - camera.centre[1]) / camera.focal)[..., None] * camera.upward)
        view /= np.linalg.norm(view, axis = -1, keepdims = True)
        cosine = np.clip(-np.sum(view * normals, axis = -1), 0.0, 1.0)
        fresnel = 0.02 + 0.98 * (1.0 - cosine)**5
        reflected = view - 2.0 * np.sum(view * normals, axis = -1, keepdims = True) * normals
        lift = np.clip(reflected[..., 2], 0.0, 1.0)[..., None]
        sky = np.array(look.skyHorizon) * (1.0 - lift) + np.array(look.skyZenith) * lift
        stops = np.array(look.speedColours)
        share = np.clip(surface['speed'] / look.speedScale, 0.0, 1.0) * (len(stops) - 1)
        lower = np.minimum(np.floor(share).astype(int), len(stops) - 2)
        blend = (share - lower)[..., None]
        tint = stops[lower] * (1.0 - blend) + stops[lower + 1] * blend
        transmission = np.exp(-surface['thickness'][..., None] * np.array(look.extinction))
        body = tint * transmission + np.array(look.deepColour) * (1.0 - transmission)
        sun = np.asarray(look.sun, dtype = float) / np.linalg.norm(look.sun)
        diffuse = (0.35 + 0.65 * (0.5 * np.sum(normals * sun, axis = -1) + 0.5))[..., None]
        halfway = sun - view
        halfway /= np.maximum(np.linalg.norm(halfway, axis = -1, keepdims = True), 1e-12)
        highlight = look.sunSpecular * np.clip(np.sum(normals * halfway, axis = -1), 0.0, 1.0)**look.shininess
        image = body * diffuse * (1.0 - fresnel[..., None]) + sky * fresnel[..., None] + highlight[..., None]
        image[~found] = look.background

        return np.clip(image, 0.0, 1.0)

    def surfaceDistance(self, surface: dict) -> np.ndarray:

        '''Distance from the eye to the surface along each pixel's ray [m]; infinite where the ray finds none.'''

        distance = np.linalg.norm(np.nan_to_num(surface['hits']) - self.camera.eye, axis = -1)

        return np.where(surface['found'], distance, np.inf)

    def render(self, positions: np.ndarray, volumes: np.ndarray, speeds: np.ndarray | None = None) -> dict:

        '''Field, march and shade one frame: the march's arrays and the 'image'.'''

        self.field(positions, volumes, speeds)
        surface = self.march()
        surface['image'] = self.shade(surface)

        return surface
