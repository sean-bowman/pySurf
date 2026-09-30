// -- Deviation Heatmap -- //

// Renders deviation heatmap visualization on surfboard meshes.
// Shows signed distance differences between reference and generated geometry
// using a diverging colorscale (blue = generated smaller, red = generated larger).

import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import * as theme from './theme.js';


// Colorscale constants
const COLORSCALE_RDYLBU = 'RdYlBu';
const COLORSCALE_VIRIDIS = 'viridis';

// Default deviation range (mm)
const DEFAULT_MIN_MM = -30;
const DEFAULT_MAX_MM = 30;


export class DeviationHeatmap {
    constructor(scene) {
        this._scene = scene;
        this._heatmapGroup = new THREE.Group();
        this._heatmapGroup.visible = false;
        this._scene.add(this._heatmapGroup);

        this._legendGroup = new THREE.Group();
        this._legendGroup.visible = false;
        this._scene.add(this._legendGroup);

        this._deviationData = null;
        this._mesh = null;

        // Colorscale settings
        this._colorscale = COLORSCALE_RDYLBU;
        this._minMm = DEFAULT_MIN_MM;
        this._maxMm = DEFAULT_MAX_MM;
        this._autoRange = true;
    }

    // ---------------------------------------------------------
    // Public API
    // ---------------------------------------------------------

    async loadDeviationData(dataPath) {
        try {
            const response = await fetch(dataPath);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            this._deviationData = await response.json();
            console.log('Deviation data loaded:', {
                nVertices: this._deviationData.deviations?.vertexDistances?.length || 0,
                minMm: this._deviationData.deviations?.minMm,
                maxMm: this._deviationData.deviations?.maxMm,
            });

            if (this._autoRange && this._deviationData.deviations) {
                // Use symmetric range centered on zero
                const absMax = Math.max(
                    Math.abs(this._deviationData.deviations.minMm || DEFAULT_MIN_MM),
                    Math.abs(this._deviationData.deviations.maxMm || DEFAULT_MAX_MM)
                );
                this._minMm = -absMax;
                this._maxMm = absMax;
            }

            this._rebuildHeatmap();
            return true;
        } catch (error) {
            console.error('Failed to load deviation data:', error);
            return false;
        }
    }

    loadDeviationDataFromObject(data) {
        this._deviationData = data;

        if (this._autoRange && data.deviations) {
            const absMax = Math.max(
                Math.abs(data.deviations.minMm || DEFAULT_MIN_MM),
                Math.abs(data.deviations.maxMm || DEFAULT_MAX_MM)
            );
            this._minMm = -absMax;
            this._maxMm = absMax;
        }

        this._rebuildHeatmap();
    }

    setVisible(visible) {
        this._heatmapGroup.visible = visible;
        this._legendGroup.visible = visible;
    }

    isVisible() {
        return this._heatmapGroup.visible;
    }

    setColorscale(scale) {
        this._colorscale = scale;
        this._rebuildHeatmap();
    }

    setRange(minMm, maxMm) {
        this._minMm = minMm;
        this._maxMm = maxMm;
        this._autoRange = false;
        this._rebuildHeatmap();
    }

    setAutoRange(enabled) {
        this._autoRange = enabled;
        if (enabled && this._deviationData?.deviations) {
            const absMax = Math.max(
                Math.abs(this._deviationData.deviations.minMm || DEFAULT_MIN_MM),
                Math.abs(this._deviationData.deviations.maxMm || DEFAULT_MAX_MM)
            );
            this._minMm = -absMax;
            this._maxMm = absMax;
            this._rebuildHeatmap();
        }
    }

    getStats() {
        if (!this._deviationData?.deviations) return null;
        return {
            minMm: this._deviationData.deviations.minMm,
            maxMm: this._deviationData.deviations.maxMm,
            rmsMm: this._deviationData.deviations.rmsMm,
            meanMm: this._deviationData.deviations.meanMm,
        };
    }

    clear() {
        this._clearGroup(this._heatmapGroup);
        this._clearGroup(this._legendGroup);
        this._mesh = null;
    }

    dispose() {
        this.clear();
        this._scene.remove(this._heatmapGroup);
        this._scene.remove(this._legendGroup);
    }

    // ---------------------------------------------------------
    // Heatmap Construction
    // ---------------------------------------------------------

    _rebuildHeatmap() {
        this.clear();

        if (!this._deviationData) return;

        // Build heatmap mesh
        if (this._deviationData.surfaceMesh && this._deviationData.deviations) {
            this._buildHeatmapMesh();
        }

        // Build legend
        this._buildLegend();
    }

    _buildHeatmapMesh() {
        const surfaceMesh = this._deviationData.surfaceMesh;
        const deviations = this._deviationData.deviations;

        if (!surfaceMesh.vertices || !surfaceMesh.faces) {
            console.warn('Invalid surface mesh data');
            return;
        }

        const vertices = surfaceMesh.vertices;
        const faces = surfaceMesh.faces;
        const distances = deviations.vertexDistances;

        if (vertices.length === 0 || faces.length === 0) {
            console.warn('Empty mesh data');
            return;
        }

        // Create BufferGeometry
        const geometry = new THREE.BufferGeometry();

        // Build position array
        const positions = new Float32Array(vertices.length * 3);
        for (let i = 0; i < vertices.length; i++) {
            positions[i * 3] = vertices[i][0];
            positions[i * 3 + 1] = vertices[i][1];
            positions[i * 3 + 2] = vertices[i][2];
        }
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        // Build index array
        const indices = new Uint32Array(faces.length * 3);
        for (let i = 0; i < faces.length; i++) {
            indices[i * 3] = faces[i][0];
            indices[i * 3 + 1] = faces[i][1];
            indices[i * 3 + 2] = faces[i][2];
        }
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));

        // Build vertex colors from deviations
        const colors = new Float32Array(vertices.length * 3);
        for (let i = 0; i < vertices.length; i++) {
            const deviation = distances[i] || 0;
            const color = this._deviationToColor(deviation);
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        }
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        geometry.computeVertexNormals();

        // Create material with vertex colors
        const material = new THREE.MeshPhongMaterial({
            vertexColors: true,
            side: THREE.DoubleSide,
            shininess: 30,
        });

        this._mesh = new THREE.Mesh(geometry, material);
        this._heatmapGroup.add(this._mesh);
    }

    _deviationToColor(deviationMm) {
        // Normalize to [0, 1]
        const t = (deviationMm - this._minMm) / (this._maxMm - this._minMm);
        const clamped = Math.max(0, Math.min(1, t));

        if (this._colorscale === COLORSCALE_VIRIDIS) {
            return this._viridis(clamped);
        } else {
            // RdYlBu diverging (reversed: blue=negative, red=positive)
            return this._rdYlBu(clamped);
        }
    }

    _rdYlBu(t) {
        // RdYlBu diverging colorscale (blue at t=0, yellow at t=0.5, red at t=1)
        // Actually reversed: blue = small deviation (good), red = large deviation (bad)
        // For signed distances: blue = generated smaller, red = generated larger

        const r = new THREE.Color();

        if (t < 0.25) {
            // Blue to light blue
            r.setRGB(
                0.192 + t * 4 * (0.421 - 0.192),
                0.211 + t * 4 * (0.682 - 0.211),
                0.584 + t * 4 * (0.839 - 0.584)
            );
        } else if (t < 0.5) {
            // Light blue to yellow
            const s = (t - 0.25) * 4;
            r.setRGB(
                0.421 + s * (0.996 - 0.421),
                0.682 + s * (0.878 - 0.682),
                0.839 + s * (0.565 - 0.839)
            );
        } else if (t < 0.75) {
            // Yellow to orange
            const s = (t - 0.5) * 4;
            r.setRGB(
                0.996 + s * (0.961 - 0.996),
                0.878 + s * (0.488 - 0.878),
                0.565 + s * (0.314 - 0.565)
            );
        } else {
            // Orange to red
            const s = (t - 0.75) * 4;
            r.setRGB(
                0.961 + s * (0.647 - 0.961),
                0.488 + s * (0.059 - 0.488),
                0.314 + s * (0.145 - 0.314)
            );
        }

        return r;
    }

    _viridis(t) {
        // Viridis sequential colorscale
        const r = new THREE.Color();

        if (t < 0.25) {
            r.setRGB(
                0.267 + t * 4 * (0.282 - 0.267),
                0.004 + t * 4 * (0.141 - 0.004),
                0.329 + t * 4 * (0.458 - 0.329)
            );
        } else if (t < 0.5) {
            const s = (t - 0.25) * 4;
            r.setRGB(
                0.282 + s * (0.128 - 0.282),
                0.141 + s * (0.566 - 0.141),
                0.458 + s * (0.551 - 0.458)
            );
        } else if (t < 0.75) {
            const s = (t - 0.5) * 4;
            r.setRGB(
                0.128 + s * (0.478 - 0.128),
                0.566 + s * (0.821 - 0.566),
                0.551 + s * (0.318 - 0.551)
            );
        } else {
            const s = (t - 0.75) * 4;
            r.setRGB(
                0.478 + s * (0.993 - 0.478),
                0.821 + s * (0.906 - 0.821),
                0.318 + s * (0.144 - 0.318)
            );
        }

        return r;
    }

    // ---------------------------------------------------------
    // Legend
    // ---------------------------------------------------------

    _buildLegend() {
        // Create a 2D legend element positioned at corner of viewport
        const legendDiv = document.createElement('div');
        legendDiv.id = 'deviation-legend';
        legendDiv.style.cssText = `
            position: fixed;
            bottom: 80px;
            right: 20px;
            width: 30px;
            height: 200px;
            background: linear-gradient(to bottom,
                rgb(165, 15, 37),
                rgb(245, 124, 80),
                rgb(254, 224, 144),
                rgb(171, 217, 233),
                rgb(49, 54, 149)
            );
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 4px;
            pointer-events: none;
        `;

        // Labels
        const labelsDiv = document.createElement('div');
        labelsDiv.style.cssText = `
            position: fixed;
            bottom: 80px;
            right: 55px;
            height: 200px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            font-family: 'Consolas', monospace;
            font-size: 11px;
            color: ${theme.CSS.textPrimary};
            pointer-events: none;
        `;

        const maxLabel = document.createElement('span');
        maxLabel.textContent = `+${this._maxMm.toFixed(0)} mm`;
        maxLabel.style.textAlign = 'right';

        const midLabel = document.createElement('span');
        midLabel.textContent = '0 mm';
        midLabel.style.textAlign = 'right';

        const minLabel = document.createElement('span');
        minLabel.textContent = `${this._minMm.toFixed(0)} mm`;
        minLabel.style.textAlign = 'right';

        labelsDiv.appendChild(maxLabel);
        labelsDiv.appendChild(midLabel);
        labelsDiv.appendChild(minLabel);

        // Title
        const titleDiv = document.createElement('div');
        titleDiv.style.cssText = `
            position: fixed;
            bottom: 285px;
            right: 20px;
            font-family: 'Consolas', monospace;
            font-size: 12px;
            font-weight: bold;
            color: ${theme.CSS.textPrimary};
            pointer-events: none;
        `;
        titleDiv.textContent = 'Deviation';

        // Store references for cleanup
        this._legendElements = [legendDiv, labelsDiv, titleDiv];

        // Add to document
        document.body.appendChild(legendDiv);
        document.body.appendChild(labelsDiv);
        document.body.appendChild(titleDiv);
    }

    // ---------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------

    _clearGroup(group) {
        while (group.children.length > 0) {
            const child = group.children[0];
            group.remove(child);

            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();

            if (child.isCSS2DObject && child.element && child.element.parentNode) {
                child.element.parentNode.removeChild(child.element);
            }
        }

        // Clean up legend DOM elements
        if (this._legendElements) {
            for (const el of this._legendElements) {
                if (el && el.parentNode) {
                    el.parentNode.removeChild(el);
                }
            }
            this._legendElements = null;
        }
    }
}

// Export colorscale constants
export { COLORSCALE_RDYLBU, COLORSCALE_VIRIDIS };
