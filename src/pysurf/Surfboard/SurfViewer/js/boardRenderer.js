// -- Board Renderer -- //

// Handles surfboard mesh loading (STL) and parametric surface construction.
// Supports solid, wireframe, and transparent render modes.
// The parametric mode builds a BufferGeometry from the sampled surface grid
// exported by Python's meshSampler.

import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import * as theme from './theme.js';


// Render mode constants
export const RENDER_SOLID = 'solid';
export const RENDER_WIREFRAME = 'wireframe';
export const RENDER_TRANSPARENT = 'transparent';

// Geometry source constants
export const SOURCE_PARAMETRIC = 'parametric';
export const SOURCE_STL = 'stl';


export class BoardRenderer {
    constructor(scene) {
        this._scene = scene;
        this._mesh = null;
        this._boardGroup = new THREE.Group();
        this._scene.add(this._boardGroup);

        this._currentColor = theme.BLUE;
        this._currentRenderMode = RENDER_SOLID;
        this._currentSource = SOURCE_PARAMETRIC;
        this._boardData = null;

        // Cached STL geometry (to avoid reloading)
        this._stlGeometry = null;
    }

    get boardGroup() { return this._boardGroup; }
    get boardData() { return this._boardData; }

    // ---------------------------------------------------------
    // Public API
    // ---------------------------------------------------------

    async loadBoardData(boardData, source = SOURCE_PARAMETRIC) {
        this._boardData = boardData;
        this._currentSource = source;

        this.clear();

        if (source === SOURCE_STL) {
            await this._loadStlMesh(boardData);
        } else {
            this._buildParametricMesh(boardData);
        }
    }

    setRenderMode(mode) {
        this._currentRenderMode = mode;
        if (this._mesh) {
            const material = this._createMaterial(this._currentColor, mode);
            this._mesh.material.dispose();
            this._mesh.material = material;
        }
    }

    setColor(color) {
        this._currentColor = color;
        if (this._mesh) {
            this._mesh.material.color.setHex(color);
        }
    }

    setOpacity(opacity) {
        if (this._mesh) {
            this._mesh.material.opacity = opacity;
            this._mesh.material.transparent = opacity < 1.0;
        }
    }

    setPosition(x, y, z) {
        this._boardGroup.position.set(x, y, z);
    }

    clear() {
        while (this._boardGroup.children.length > 0) {
            const child = this._boardGroup.children[0];
            this._boardGroup.remove(child);
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
        }
        this._mesh = null;
    }

    getBounds() {
        if (!this._boardData) return null;
        const params = this._boardData.parameters;
        return {
            lengthMm: params.lengthMm,
            widthMm: params.maxWidthMm,
            thicknessMm: params.maxThicknessMm,
        };
    }

    dispose() {
        this.clear();
        this._scene.remove(this._boardGroup);
        if (this._stlGeometry) {
            this._stlGeometry.dispose();
            this._stlGeometry = null;
        }
    }

    // ---------------------------------------------------------
    // STL Loading
    // ---------------------------------------------------------

    async _loadStlMesh(boardData) {
        let geometry;

        // Check for embedded base64 STL data
        if (boardData.stlEmbedded && boardData.stlBase64) {
            geometry = this._parseBase64Stl(boardData.stlBase64);
        } else if (boardData.stlFile) {
            // Load from file path (requires HTTP server)
            const stlPath = '../SurfPhysics/output/' + boardData.stlFile;
            geometry = await this._loadStlFile(stlPath);
        } else {
            console.warn('No STL data available, falling back to parametric');
            this._buildParametricMesh(boardData);
            return;
        }

        if (!geometry) {
            console.warn('Failed to load STL, falling back to parametric');
            this._buildParametricMesh(boardData);
            return;
        }

        this._stlGeometry = geometry;
        geometry.computeVertexNormals();

        const material = this._createMaterial(this._currentColor, this._currentRenderMode);
        this._mesh = new THREE.Mesh(geometry, material);
        this._boardGroup.add(this._mesh);
    }

    _parseBase64Stl(base64Data) {
        try {
            const binaryStr = atob(base64Data);
            const bytes = new Uint8Array(binaryStr.length);
            for (let i = 0; i < binaryStr.length; i++) {
                bytes[i] = binaryStr.charCodeAt(i);
            }
            const loader = new STLLoader();
            return loader.parse(bytes.buffer);
        } catch (e) {
            console.error('Failed to parse base64 STL:', e);
            return null;
        }
    }

    _loadStlFile(url) {
        return new Promise((resolve, reject) => {
            const loader = new STLLoader();
            loader.load(
                url,
                (geometry) => resolve(geometry),
                undefined,
                (error) => {
                    console.error('STL load error:', error);
                    resolve(null);
                }
            );
        });
    }

    // ---------------------------------------------------------
    // Parametric Surface Construction
    // ---------------------------------------------------------

    _buildParametricMesh(boardData) {
        const surface = boardData.parametricSurface;
        if (!surface) {
            console.error('No parametricSurface data in boardData');
            return;
        }

        const geometry = this._buildSurfaceGeometry(surface, boardData.parameters);
        geometry.computeVertexNormals();

        const material = this._createMaterial(this._currentColor, this._currentRenderMode);
        this._mesh = new THREE.Mesh(geometry, material);
        this._boardGroup.add(this._mesh);
    }

    _buildSurfaceGeometry(surface, params) {
        const nLong = surface.nLongitudinal;
        const nLat = surface.nLateral;
        const tValues = surface.tValues;
        const lateralFractions = surface.lateralFractions;
        const halfWidths = surface.outline.halfWidthsMm;
        const rockerHeights = surface.rocker.heightsMm;
        const deckHeights = surface.deckSurface.heightsMm;
        const bottomHeights = surface.bottomSurface.heightsMm;
        const length = params.lengthMm;

        // Build vertex positions for one half (positive Y), then mirror.
        // Total vertices: deck + bottom, each side has nLong * nLat points.
        // Full board: 2 sides (pos/neg Y) x 2 surfaces (deck/bottom) = 4 patches.
        // We stitch them into one closed mesh.

        const positions = [];
        const indices = [];

        // Helper: push a vertex and return its index
        let vertexCount = 0;
        const pushVertex = (x, y, z) => {
            positions.push(x, y, z);
            return vertexCount++;
        };

        // Build vertex grid for positive-Y deck surface
        // indexMap[surface][longIdx][latIdx] = vertex index
        const deckPosY = [];
        const deckNegY = [];
        const bottomPosY = [];
        const bottomNegY = [];

        for (let i = 0; i < nLong; i++) {
            const t = tValues[i];
            const x = t * length;
            const hw = halfWidths[i];
            const rz = rockerHeights[i];

            const dPosRow = [];
            const dNegRow = [];
            const bPosRow = [];
            const bNegRow = [];

            for (let j = 0; j < nLat; j++) {
                const lf = lateralFractions[j];
                const y = lf * hw;
                const deckZ = rz + deckHeights[i][j];
                const bottomZ = rz + bottomHeights[i][j];

                // Positive Y side
                dPosRow.push(pushVertex(x, y, deckZ));
                bPosRow.push(pushVertex(x, y, bottomZ));

                // Negative Y side (mirror, skip centerline j=0 to avoid duplicate)
                if (j > 0) {
                    dNegRow.push(pushVertex(x, -y, deckZ));
                    bNegRow.push(pushVertex(x, -y, bottomZ));
                }
            }

            deckPosY.push(dPosRow);
            bottomPosY.push(bPosRow);
            deckNegY.push(dNegRow);
            bottomNegY.push(bNegRow);
        }

        // Helper: triangulate a quad grid patch
        const triangulatePatch = (grid, nRows, nCols, flipWinding) => {
            for (let i = 0; i < nRows - 1; i++) {
                for (let j = 0; j < nCols - 1; j++) {
                    const a = grid[i][j];
                    const b = grid[i][j + 1];
                    const c = grid[i + 1][j + 1];
                    const d = grid[i + 1][j];

                    if (flipWinding) {
                        indices.push(a, c, b);
                        indices.push(a, d, c);
                    } else {
                        indices.push(a, b, c);
                        indices.push(a, c, d);
                    }
                }
            }
        };

        // Triangulate deck surfaces (normals face outward = up)
        triangulatePatch(deckPosY, nLong, nLat, false);
        triangulatePatch(bottomPosY, nLong, nLat, true); // flip for outward normals

        // Negative Y side: deck faces outward (same winding as posY but mirrored)
        // Build combined grid for negY by prepending centerline vertices
        if (deckNegY[0].length > 0) {
            // Combine: negY reversed + centerline (from posY j=0)
            const deckFullNegY = [];
            const bottomFullNegY = [];

            for (let i = 0; i < nLong; i++) {
                // Reverse negY so it goes from -maxWidth to centerline
                const dRow = [...deckNegY[i]].reverse();
                dRow.push(deckPosY[i][0]); // centerline vertex
                deckFullNegY.push(dRow);

                const bRow = [...bottomNegY[i]].reverse();
                bRow.push(bottomPosY[i][0]);
                bottomFullNegY.push(bRow);
            }

            const negCols = deckFullNegY[0].length;
            triangulatePatch(deckFullNegY, nLong, negCols, true); // flipped winding for mirror
            triangulatePatch(bottomFullNegY, nLong, negCols, false);
        }

        // Close the rail edges: connect deck to bottom at the outer rail (j = nLat-1)
        for (let i = 0; i < nLong - 1; i++) {
            const dTop = deckPosY[i][nLat - 1];
            const dBot = bottomPosY[i][nLat - 1];
            const dTopNext = deckPosY[i + 1][nLat - 1];
            const dBotNext = bottomPosY[i + 1][nLat - 1];

            indices.push(dTop, dTopNext, dBotNext);
            indices.push(dTop, dBotNext, dBot);

            // Negative Y rail
            if (deckNegY[i].length > 0) {
                const nTop = deckNegY[i][nLat - 2]; // last negY index
                const nBot = bottomNegY[i][nLat - 2];
                const nTopNext = deckNegY[i + 1][nLat - 2];
                const nBotNext = bottomNegY[i + 1][nLat - 2];

                indices.push(nTop, nBotNext, nTopNext);
                indices.push(nTop, nBot, nBotNext);
            }
        }

        // Build BufferGeometry
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute(
            'position',
            new THREE.Float32BufferAttribute(positions, 3)
        );
        geometry.setIndex(indices);

        return geometry;
    }

    // ---------------------------------------------------------
    // Materials
    // ---------------------------------------------------------

    _createMaterial(color, renderMode) {
        switch (renderMode) {
            case RENDER_WIREFRAME:
                return new THREE.MeshBasicMaterial({
                    color: color,
                    wireframe: true,
                    opacity: 0.8,
                    transparent: true,
                });

            case RENDER_TRANSPARENT:
                return new THREE.MeshPhysicalMaterial({
                    color: color,
                    roughness: 0.35,
                    metalness: 0.0,
                    clearcoat: 0.5,
                    clearcoatRoughness: 0.3,
                    opacity: 0.5,
                    transparent: true,
                    side: THREE.DoubleSide,
                    depthWrite: false,
                });

            case RENDER_SOLID:
            default:
                return new THREE.MeshPhysicalMaterial({
                    color: color,
                    roughness: 0.35,
                    metalness: 0.0,
                    clearcoat: 0.8,
                    clearcoatRoughness: 0.2,
                    side: THREE.DoubleSide,
                });
        }
    }
}
