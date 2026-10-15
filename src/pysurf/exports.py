
# -- pySurf: Data Export -- #

'''

Writing geometry and data out in the forms other tools read.

    py2cad      Binary STL from a structured surface grid, for CAD and print preparation.
    writeStl    Binary STL from a list of triangles, for closed surfaces a single grid cannot form.
    writeFile   Delimited text (.csv or .txt) from a 2D array, for spreadsheets and CAD sketches.

----------------------------------------------------------------------
                            Validation status
----------------------------------------------------------------------

Nothing here is a model. Export can lose a field, change its units or write a file the consumer
cannot read; it cannot get physics wrong, because it computes nothing beyond facet normals.
tests/testExports.py reads every written file back: STL output through trimesh, checking facet
count, vertex positions and normals, and text output through the csv module.

Sean Bowman [10/10/2026]

'''

# Standard library imports
import csv

# Third-party imports
import numpy as np

# Binary STL record: facet normal, three vertices, and a two-byte attribute count, all
# little-endian as the format specifies
_stlRecord = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])

def py2cad(filename: str, xData: np.ndarray | list, yData: np.ndarray | list, zData: np.ndarray | list) -> None:

    '''

    Export a surface mesh to a binary .stl file from raw coordinate arrays.

    Takes in arrays containing spatial coordinates of a structured surface mesh and exports the
    surface as a .stl file. Useful for exporting geometry out of Python CAD design tools and into
    CAD packages such as NX.

    Each grid cell (i, j) becomes two facets, (i, j) -> (i, j+1) -> (i+1, j+1) and
    (i+1, j+1) -> (i+1, j) -> (i, j), written in row-major cell order. A facet with two coincident
    corners has no normal, so its normal is written as zero (see `_writeFacets`).

    Parameters:
    -----------
    filename : str
        Output file path. The .stl extension is appended if not provided.
    xData : np.ndarray | list
        X coordinates of the surface mesh, shape (m, n).
    yData : np.ndarray | list
        Y coordinates of the surface mesh, shape (m, n).
    zData : np.ndarray | list
        Z coordinates of the surface mesh, shape (m, n).

    Returns:
    --------
    None : A .stl file is created and saved to the specified filename location.

    '''

    # Append file extension if the given name does not contain it
    if '.' not in filename:
        filename += '.stl'

    # Stack the grid into one (m, n, 3) point array
    points = np.stack([np.asarray(xData, dtype = float),
                       np.asarray(yData, dtype = float),
                       np.asarray(zData, dtype = float)], axis = -1)

    # The two triangles of every cell, interleaved so the facet order is cell by cell with the
    # upper triangle first: shape (m - 1, n - 1, 2, 3 corners, 3 coordinates)
    upper = np.stack([points[:-1, :-1], points[:-1, 1:], points[1:, 1:]], axis = -2)
    lower = np.stack([points[1:, 1:], points[1:, :-1], points[:-1, :-1]], axis = -2)
    facets = np.stack([upper, lower], axis = 2).reshape(-1, 3, 3)

    _writeFacets(filename, facets, b'Created by py2cad.py [Sean Bowman]')

def writeStl(filename: str, vertices: np.ndarray | list, faces: np.ndarray | list) -> None:

    '''

    Export a triangulated surface to a binary .stl file.

    Each face lists three vertex indices; the corner order sets the facet normal by the right-hand
    rule, so faces listed counterclockwise as seen from outside give outward normals. A closed
    surface whose faces share edges reads back watertight in tools that merge coincident vertices.

    Parameters:
    -----------
    filename : str
        Output file path. The .stl extension is appended if not provided.
    vertices : np.ndarray | list
        Vertex coordinates, shape (n, 3).
    faces : np.ndarray | list
        Vertex indices of each triangle, shape (m, 3).

    Returns:
    --------
    None : A .stl file is created and saved to the specified filename location.

    '''

    if '.' not in filename:
        filename += '.stl'

    facets = np.asarray(vertices, dtype = float)[np.asarray(faces, dtype = int)]
    _writeFacets(filename, facets, b'Created by pySurf [Sean Bowman]')

def _writeFacets(filename: str, facets: np.ndarray, metadataString: bytes) -> None:

    '''

    Write facets, shape (m, 3 corners, 3 coordinates), as binary STL records behind an 80-byte
    header. A facet with two coincident corners has no normal, which the division would write as
    NaN, so its normal is written as zero instead: the format allows it and readers recompute
    normals from the vertices.

    '''

    # Facet normals from the right-hand rule on the corner order
    crossProduct = np.cross(facets[:, 1] - facets[:, 0], facets[:, 2] - facets[:, 0])
    magnitude = np.linalg.norm(crossProduct, axis = 1, keepdims = True)
    normals = np.divide(crossProduct, magnitude, out = np.zeros_like(crossProduct), where = magnitude > 0.0)

    records = np.zeros(len(facets), dtype = _stlRecord)
    records['normal'] = normals
    records['vertices'] = facets

    # .stl files start with 80 bytes of metadata, padded with spaces, then the facet count
    metadataTitle = metadataString + b' ' * (80 - len(metadataString))

    with open(filename, 'wb') as fileID:
        fileID.write(metadataTitle)
        fileID.write(np.array([len(records)], dtype = '<i4').tobytes())
        fileID.write(records.tobytes())

def writeFile(filename: str, data: np.ndarray | list, headers: bool = False) -> None:

    '''

    Write array data to a .csv or .txt file.

    Parameters:
    -----------
    filename : str
        Output file path including extension (.csv or .txt).
    data : np.ndarray | list
        Data matrix of shape (n, m) to write.
    headers : bool, optional
        If True, write an X/Y/Z header row for CSV files (default False).

    Returns:
    --------
    None : A file is created at the specified path.

    Raises:
    -------
    ValueError : If the filename has no extension, or the extension is not .csv or .txt.

    '''

    if '.' not in filename:
        raise ValueError('Filename must include a file extension (.csv or .txt).')

    if isinstance(data, list):
        data = np.array(data)

    whichType = filename[-4:]

    match whichType:

        case '.csv':

            with open(filename, 'w', newline = '') as csvFile:
                writer = csv.writer(csvFile)
                if headers:
                    writer.writerow(['X', 'Y', 'Z'])
                writer.writerows(data)

        case '.txt':

            # One tab-delimited line per row of the data
            with open(filename, 'w') as txtFile:
                for i, _ in enumerate(data[:, 0]):
                    txtFile.write('\t'.join([str(val) for val in list(data[i, :])]) + '\n')

        case _:

            raise ValueError(f'Unsupported file extension {whichType!r}: use .csv or .txt.')
