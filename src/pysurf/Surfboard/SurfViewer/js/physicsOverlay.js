// -- Physics Overlay -- //

// Renders physics visualization overlays on the 3D surfboard:
// - Waterline plane (semi-transparent at draft height)
// - Force vectors (arrows for weight, buoyancy, drag, lift)
// - Draft indicator (vertical line from board bottom to waterplane)

import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import * as theme from './theme.js';


export class PhysicsOverlay {
    constructor(scene) {
        this._scene = scene;
        this._overlayGroup = new THREE.Group();
        this._scene.add(this._overlayGroup);

        // Overlay sub-groups (for independent toggling)
        this._waterlineGroup = new THREE.Group();
        this._forceGroup = new THREE.Group();
        this._draftGroup = new THREE.Group();

        this._waterlineGroup.visible = false;
        this._forceGroup.visible = false;
        this._draftGroup.visible = false;

        this._overlayGroup.add(this._waterlineGroup);
        this._overlayGroup.add(this._forceGroup);
        this._overlayGroup.add(this._draftGroup);

        this._boardData = null;
    }

    // ---------------------------------------------------------
    // Public API
    // ---------------------------------------------------------

    updateFromBoardData(boardData) {
        this._boardData = boardData;
        this._rebuildAll();
    }

    setVisible(overlayId, visible) {
        switch (overlayId) {
            case 'waterline':
                this._waterlineGroup.visible = visible;
                break;
            case 'forceVectors':
                this._forceGroup.visible = visible;
                break;
            case 'draft':
                this._draftGroup.visible = visible;
                break;
        }
    }

    clear() {
        this._clearGroup(this._waterlineGroup);
        this._clearGroup(this._forceGroup);
        this._clearGroup(this._draftGroup);
    }

    dispose() {
        this.clear();
        this._scene.remove(this._overlayGroup);
    }

    // ---------------------------------------------------------
    // Rebuild all overlays from current board data
    // ---------------------------------------------------------

    _rebuildAll() {
        this.clear();
        if (!this._boardData || !this._boardData.physics) return;

        this._buildWaterline();
        this._buildForceVectors();
        this._buildDraftIndicator();
    }

    // ---------------------------------------------------------
    // Waterline plane
    // ---------------------------------------------------------

    _buildWaterline() {
        const physics = this._boardData.physics;
        const params = this._boardData.parameters;

        const waterplaneZ = physics.waterline.waterplaneZMm;
        const planeWidth = params.maxWidthMm * 3;
        const planeLength = params.lengthMm * 1.4;

        const geometry = new THREE.PlaneGeometry(planeLength, planeWidth);
        const material = new THREE.MeshBasicMaterial({
            color: theme.BLUE,
            transparent: true,
            opacity: 0.12,
            side: THREE.DoubleSide,
            depthWrite: false,
        });

        const plane = new THREE.Mesh(geometry, material);
        // PlaneGeometry lies in XY by default; rotate to XY plane at waterlineZ
        plane.position.set(params.lengthMm * 0.5, 0, waterplaneZ);
        // Already in XY orientation (board is X-length, Y-width)
        // No rotation needed since PlaneGeometry is in XY

        this._waterlineGroup.add(plane);

        // Wireframe border for visibility
        const edgeGeom = new THREE.EdgesGeometry(geometry);
        const edgeMat = new THREE.LineBasicMaterial({
            color: theme.BLUE,
            transparent: true,
            opacity: 0.35,
        });
        const edges = new THREE.LineSegments(edgeGeom, edgeMat);
        edges.position.copy(plane.position);
        this._waterlineGroup.add(edges);

        // Label
        this._addLabel(
            `Waterline: ${waterplaneZ.toFixed(1)} mm`,
            params.lengthMm * 0.5,
            params.maxWidthMm * 1.2,
            waterplaneZ,
            theme.CSS.blue
        );
    }

    // ---------------------------------------------------------
    // Force vectors
    // ---------------------------------------------------------

    _buildForceVectors() {
        const forceVectors = this._boardData.physics.forceVectors;
        if (!forceVectors) return;

        // Scale factor: arrow length per Newton of force
        // Calibrated so typical forces produce visible arrows
        const scaleFactor = 0.3; // mm per Newton

        const forceConfigs = [
            { key: 'weight', color: theme.FORCE_COLORS.weight },
            { key: 'buoyancy', color: theme.FORCE_COLORS.buoyancy },
            { key: 'drag', color: theme.FORCE_COLORS.drag },
            { key: 'lift', color: theme.FORCE_COLORS.lift },
        ];

        for (const fc of forceConfigs) {
            const fv = forceVectors[fc.key];
            if (!fv || fv.magnitudeN === 0) continue;

            const origin = new THREE.Vector3(...fv.origin);
            const dir = new THREE.Vector3(...fv.direction).normalize();
            const length = Math.min(fv.magnitudeN * scaleFactor, 500); // cap at 500mm
            const headLength = length * 0.2;
            const headWidth = headLength * 0.5;

            const arrow = new THREE.ArrowHelper(
                dir, origin, length, fc.color, headLength, headWidth
            );
            this._forceGroup.add(arrow);

            // Label at arrow tip
            const tipPos = origin.clone().add(dir.clone().multiplyScalar(length + 20));
            this._addLabel(
                `${fv.label}: ${fv.magnitudeN.toFixed(0)} N`,
                tipPos.x, tipPos.y, tipPos.z,
                '#' + fc.color.toString(16).padStart(6, '0'),
                this._forceGroup
            );
        }
    }

    // ---------------------------------------------------------
    // Draft indicator
    // ---------------------------------------------------------

    _buildDraftIndicator() {
        const physics = this._boardData.physics;
        const params = this._boardData.parameters;

        const waterplaneZ = physics.waterline.waterplaneZMm;
        const draftCm = physics.buoyancy.riderDraftCm;

        // Vertical line at the thickest point (40% from nose)
        const lineX = params.lengthMm * 0.4;
        const lineY = params.maxWidthMm * 0.6;
        const bottomZ = -10; // approximate bottom at this station

        const points = [
            new THREE.Vector3(lineX, lineY, bottomZ),
            new THREE.Vector3(lineX, lineY, waterplaneZ),
        ];

        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineDashedMaterial({
            color: theme.CYAN,
            dashSize: 8,
            gapSize: 4,
            transparent: true,
            opacity: 0.7,
        });

        const line = new THREE.Line(geometry, material);
        line.computeLineDistances();
        this._draftGroup.add(line);

        // Tick marks at top and bottom
        const tickWidth = 15;
        for (const z of [bottomZ, waterplaneZ]) {
            const tickPts = [
                new THREE.Vector3(lineX, lineY - tickWidth, z),
                new THREE.Vector3(lineX, lineY + tickWidth, z),
            ];
            const tickGeom = new THREE.BufferGeometry().setFromPoints(tickPts);
            const tickLine = new THREE.Line(tickGeom, new THREE.LineBasicMaterial({
                color: theme.CYAN,
                transparent: true,
                opacity: 0.7,
            }));
            this._draftGroup.add(tickLine);
        }

        // Draft label
        const midZ = (bottomZ + waterplaneZ) / 2;
        this._addLabel(
            `Draft: ${draftCm.toFixed(1)} cm`,
            lineX, lineY + 30, midZ,
            theme.CSS.cyan,
            this._draftGroup
        );
    }

    // ---------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------

    _addLabel(text, x, y, z, color, parent = null) {
        const div = document.createElement('div');
        div.textContent = text;
        div.style.cssText = `
            font-size: 11px; font-family: 'Consolas', monospace;
            color: ${color}; background: rgba(0, 0, 0, 0.6);
            padding: 2px 6px; border-radius: 3px; white-space: nowrap;
            pointer-events: none;
        `;

        const label = new CSS2DObject(div);
        label.position.set(x, y, z);
        (parent || this._overlayGroup).add(label);
    }

    _clearGroup(group) {
        while (group.children.length > 0) {
            const child = group.children[0];
            group.remove(child);

            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();

            // Handle CSS2DObject cleanup
            if (child.isCSS2DObject && child.element && child.element.parentNode) {
                child.element.parentNode.removeChild(child.element);
            }

            // Handle ArrowHelper (has children)
            if (child.children) {
                child.children.forEach(c => {
                    if (c.geometry) c.geometry.dispose();
                    if (c.material) c.material.dispose();
                });
            }
        }
    }
}
