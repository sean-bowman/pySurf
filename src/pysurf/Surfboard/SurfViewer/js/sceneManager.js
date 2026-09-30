// -- Scene Manager -- //

// Three.js scene setup: renderer, camera, lighting, controls, and animation loop.
// Provides the rendering foundation that all other modules attach to.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import * as theme from './theme.js';


export class SceneManager {
    constructor(container) {
        this._container = container;

        // Core Three.js objects
        this._scene = new THREE.Scene();
        this._scene.background = new THREE.Color(theme.BACKGROUND);

        // Camera
        this._camera = new THREE.PerspectiveCamera(
            45,
            container.clientWidth / container.clientHeight,
            0.1,
            50000
        );

        // Renderer
        this._renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: false,
        });
        this._renderer.setSize(container.clientWidth, container.clientHeight);
        this._renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this._renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this._renderer.toneMappingExposure = 1.0;
        container.appendChild(this._renderer.domElement);

        // CSS2D renderer for text labels
        this._labelRenderer = new CSS2DRenderer();
        this._labelRenderer.setSize(container.clientWidth, container.clientHeight);
        this._labelRenderer.domElement.style.position = 'absolute';
        this._labelRenderer.domElement.style.top = '0';
        this._labelRenderer.domElement.style.left = '0';
        this._labelRenderer.domElement.style.pointerEvents = 'none';
        container.appendChild(this._labelRenderer.domElement);

        // Controls
        this._controls = new OrbitControls(this._camera, this._renderer.domElement);
        this._controls.enableDamping = true;
        this._controls.dampingFactor = 0.08;
        this._controls.rotateSpeed = 0.8;
        this._controls.panSpeed = 0.8;
        this._controls.zoomSpeed = 1.2;

        // Lighting
        this._setupLighting();

        // Grid helper (subtle reference plane)
        this._gridHelper = null;

        // Animation loop
        this._animationId = null;
        this._onAnimateCallbacks = [];

        // Handle resize
        this._boundResize = this._onResize.bind(this);
        window.addEventListener('resize', this._boundResize);
    }

    get scene() { return this._scene; }
    get camera() { return this._camera; }
    get renderer() { return this._renderer; }
    get controls() { return this._controls; }

    // ---------------------------------------------------------
    // Lighting
    // ---------------------------------------------------------

    _setupLighting() {
        // Warm key light from upper right
        const keyLight = new THREE.DirectionalLight(0xfff5e6, 1.4);
        keyLight.position.set(2000, 1500, 1200);
        this._scene.add(keyLight);

        // Cool fill light from lower left
        const fillLight = new THREE.DirectionalLight(0x42A5F5, 0.35);
        fillLight.position.set(-1000, -800, 400);
        this._scene.add(fillLight);

        // Rim light from behind for edge definition
        const rimLight = new THREE.DirectionalLight(0xE0E0E0, 0.5);
        rimLight.position.set(-1500, 0, 300);
        this._scene.add(rimLight);

        // Soft ambient
        const ambient = new THREE.AmbientLight(0x333344, 0.4);
        this._scene.add(ambient);

        // Hemisphere for subtle sky/ground variation
        const hemi = new THREE.HemisphereLight(0x444466, 0x222233, 0.2);
        this._scene.add(hemi);
    }

    // ---------------------------------------------------------
    // Camera positioning
    // ---------------------------------------------------------

    frameBoardBounds(lengthMm, widthMm, thicknessMm) {
        // Position camera above and to the right, looking at center of board
        const centerX = lengthMm * 0.5;
        const distance = lengthMm * 1.2;

        this._camera.position.set(
            centerX * 0.3,
            widthMm * 2.5,
            distance * 0.5
        );
        this._controls.target.set(centerX, 0, 0);
        this._controls.update();
    }

    frameMultipleBoards(boards) {
        // Find bounding box of all boards and position camera to see them all
        if (boards.length === 0) return;

        let maxLength = 0;
        let maxWidth = 0;
        let maxOffsetY = 0;

        for (const board of boards) {
            maxLength = Math.max(maxLength, board.lengthMm || 2000);
            maxWidth = Math.max(maxWidth, board.widthMm || 500);
            maxOffsetY = Math.max(maxOffsetY, Math.abs(board.offsetY || 0));
        }

        const totalSpan = maxOffsetY * 2 + maxWidth;
        const centerX = maxLength * 0.5;
        const distance = Math.max(maxLength, totalSpan) * 1.4;

        this._camera.position.set(
            centerX * 0.3,
            totalSpan * 1.2,
            distance * 0.5
        );
        this._controls.target.set(centerX, 0, 0);
        this._controls.update();
    }

    // ---------------------------------------------------------
    // Grid
    // ---------------------------------------------------------

    showGrid(lengthMm) {
        this.hideGrid();

        const size = Math.ceil(lengthMm * 1.5 / 100) * 100;
        const divisions = size / 100; // 100mm grid spacing
        this._gridHelper = new THREE.GridHelper(size, divisions, theme.GRID_COLOR, theme.GRID_COLOR);
        this._gridHelper.rotation.x = Math.PI / 2; // align to XY plane
        this._gridHelper.position.set(lengthMm * 0.5, 0, -50);
        this._gridHelper.material.opacity = 0.15;
        this._gridHelper.material.transparent = true;
        this._scene.add(this._gridHelper);
    }

    hideGrid() {
        if (this._gridHelper) {
            this._scene.remove(this._gridHelper);
            this._gridHelper.geometry.dispose();
            this._gridHelper.material.dispose();
            this._gridHelper = null;
        }
    }

    // ---------------------------------------------------------
    // Animation loop
    // ---------------------------------------------------------

    start() {
        if (this._animationId !== null) return;

        const animate = () => {
            this._animationId = requestAnimationFrame(animate);
            this._controls.update();

            for (const cb of this._onAnimateCallbacks) {
                cb();
            }

            this._renderer.render(this._scene, this._camera);
            this._labelRenderer.render(this._scene, this._camera);
        };
        animate();
    }

    stop() {
        if (this._animationId !== null) {
            cancelAnimationFrame(this._animationId);
            this._animationId = null;
        }
    }

    onAnimate(callback) {
        this._onAnimateCallbacks.push(callback);
    }

    // ---------------------------------------------------------
    // Resize handling
    // ---------------------------------------------------------

    _onResize() {
        const width = this._container.clientWidth;
        const height = this._container.clientHeight;

        this._camera.aspect = width / height;
        this._camera.updateProjectionMatrix();
        this._renderer.setSize(width, height);
        this._labelRenderer.setSize(width, height);
    }

    // ---------------------------------------------------------
    // Cleanup
    // ---------------------------------------------------------

    dispose() {
        this.stop();
        window.removeEventListener('resize', this._boundResize);
        this.hideGrid();
        this._renderer.dispose();
        this._controls.dispose();
    }
}
