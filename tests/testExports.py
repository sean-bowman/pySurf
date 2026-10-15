
# -- pySurf: Export Tests -- #

'''

Every file pysurf.exports writes is read back by an independent reader.

STL output goes through trimesh, and is also checked byte for byte against a facet-by-facet
reference writer that follows the format specification directly, so the vectorized writer cannot
reorder facets, flip a winding, or change a byte without a test noticing. Text output goes
through the csv module.

Sean Bowman [10/10/2026]

'''

import csv
import struct

import numpy as np
import pytest
import trimesh

from pysurf.exports import py2cad, writeFile, writeStl

def referenceStlBytes(x, y, z) -> bytes:

    '''

    A binary STL written one facet at a time, straight from the format: an 80-byte header, a
    uint32 facet count, and per facet a normal, three vertices and a uint16 attribute, all
    little-endian float32. Cell (i, j) gives (i, j) -> (i, j+1) -> (i+1, j+1) then
    (i+1, j+1) -> (i+1, j) -> (i, j).

    '''

    header = b'Created by py2cad.py [Sean Bowman]'
    out = bytearray(header + b' ' * (80 - len(header)))
    facets = []
    for i in range(x.shape[0] - 1):
        for j in range(x.shape[1] - 1):
            corners = [(i, j), (i, j + 1), (i + 1, j + 1)]
            facets.append([np.array([x[c], y[c], z[c]]) for c in corners])
            corners = [(i + 1, j + 1), (i + 1, j), (i, j)]
            facets.append([np.array([x[c], y[c], z[c]]) for c in corners])
    out += struct.pack('<I', len(facets))
    for a, b, c in facets:
        normal = np.cross(b - a, c - a)
        magnitude = np.linalg.norm(normal)
        normal = normal / magnitude if magnitude > 0.0 else np.zeros(3)
        out += struct.pack('<12fH', *np.float32(normal), *np.float32(a), *np.float32(b), *np.float32(c), 0)
    return bytes(out)

def cylinderGrid(nAxial = 9, nAround = 17):

    '''A closed-around open cylinder of radius 2 and length 5 as a structured grid.'''

    theta = np.linspace(0.0, 2 * np.pi, nAround)
    axial = np.linspace(0.0, 5.0, nAxial)
    angle, length = np.meshgrid(theta, axial)

    return length, 2.0 * np.cos(angle), 2.0 * np.sin(angle)

class TestPy2cad:

    '''Binary STL from a structured grid.'''

    def testTheFileMatchesTheFacetByFacetReference(self, tmp_path):

        x, y, z = cylinderGrid()
        path = tmp_path / 'cylinder.stl'

        py2cad(str(path), x, y, z)

        assert path.read_bytes() == referenceStlBytes(x, y, z)

    def testTrimeshReadsTheFacetsBack(self, tmp_path):

        x, y, z = cylinderGrid()
        path = tmp_path / 'cylinder.stl'
        py2cad(str(path), x, y, z)

        mesh = trimesh.load(str(path), process = False)

        assert len(mesh.faces) == 2 * (x.shape[0] - 1) * (x.shape[1] - 1)
        assert mesh.bounds == pytest.approx(np.array([[0.0, -2.0, -2.0], [5.0, 2.0, 2.0]]), abs = 1e-6)

    def testTheExtensionIsAppended(self, tmp_path):

        x, y, z = cylinderGrid(3, 5)
        py2cad(str(tmp_path / 'noExtension'), x, y, z)

        assert (tmp_path / 'noExtension.stl').exists()

    def testADegenerateFacetHasAZeroNormalRatherThanNaN(self, tmp_path):

        '''A grid row collapsed to a point, as at a board tip, gives facets with no area.'''

        x, y, z = cylinderGrid(4, 7)
        y[0, :] = 0.0
        z[0, :] = 0.0
        path = tmp_path / 'tip.stl'
        py2cad(str(path), x, y, z)

        raw = path.read_bytes()
        count = struct.unpack('<I', raw[80:84])[0]
        normals = np.array([struct.unpack('<3f', raw[84 + 50 * k: 84 + 50 * k + 12]) for k in range(count)])

        assert np.all(np.isfinite(normals))
        assert np.any(np.all(normals == 0.0, axis = 1))

    def testListsAreAccepted(self, tmp_path):

        x, y, z = cylinderGrid(3, 5)
        path = tmp_path / 'fromLists.stl'
        py2cad(str(path), x.tolist(), y.tolist(), z.tolist())

        assert path.read_bytes() == referenceStlBytes(x, y, z)

class TestWriteStl:

    '''Binary STL from a triangle list.'''

    # A unit tetrahedron, each face counterclockwise as seen from outside
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])

    def testAClosedSurfaceReadsBackWatertightAndOutward(self, tmp_path):

        path = tmp_path / 'tetrahedron.stl'
        writeStl(str(path), self.vertices, self.faces)
        mesh = trimesh.load(str(path))

        assert len(mesh.faces) == 4 and mesh.is_watertight and mesh.is_winding_consistent
        assert mesh.volume == pytest.approx(1.0 / 6.0)

    def testTheFacetsKeepTheirCornerOrder(self, tmp_path):

        path = tmp_path / 'tetrahedron.stl'
        writeStl(str(path), self.vertices.tolist(), self.faces.tolist())
        raw = path.read_bytes()
        corners = [np.array(struct.unpack('<9f', raw[84 + 50 * k + 12: 84 + 50 * k + 48])).reshape(3, 3)
                   for k in range(4)]

        assert np.allclose(corners, self.vertices[self.faces])

class TestWriteFile:

    '''Delimited text output.'''

    data = np.array([[1.0, 2.0, 3.0], [4.5, 5.5, 6.5]])

    def testCsvWithHeadersHasOneHeaderRow(self, tmp_path):

        path = tmp_path / 'points.csv'
        writeFile(str(path), self.data, headers = True)

        with open(path, newline = '') as csvFile:
            rows = list(csv.reader(csvFile))

        assert rows[0] == ['X', 'Y', 'Z']
        assert [[float(v) for v in row] for row in rows[1:]] == self.data.tolist()

    def testTxtIsTabDelimited(self, tmp_path):

        path = tmp_path / 'points.txt'
        writeFile(str(path), self.data.tolist())

        lines = path.read_text().splitlines()

        assert [[float(v) for v in line.split('\t')] for line in lines] == self.data.tolist()

    @pytest.mark.parametrize('name', ['points', 'points.dat'])
    def testAMissingOrUnsupportedExtensionIsRefused(self, tmp_path, name):

        with pytest.raises(ValueError):
            writeFile(str(tmp_path / name), self.data)
