// -- Board Comparison -- //

// Manages multiple BoardRenderer instances for side-by-side or overlay
// comparison of different surfboard types.

import * as THREE from 'three';
import { BoardRenderer, SOURCE_PARAMETRIC, RENDER_TRANSPARENT } from './boardRenderer.js';
import * as theme from './theme.js';


export class BoardComparison {
    constructor(scene, sceneManager) {
        this._scene = scene;
        this._sceneManager = sceneManager;

        // Map of boardType -> { renderer, boardData, active }
        this._boards = new Map();

        // Comparison mode: 'side-by-side' or 'overlay'
        this._mode = 'side-by-side';

        // Current geometry source for comparison boards
        this._source = SOURCE_PARAMETRIC;
    }

    get mode() { return this._mode; }
    get activeBoards() {
        return [...this._boards.entries()]
            .filter(([_, entry]) => entry.active)
            .map(([type, _]) => type);
    }

    // ---------------------------------------------------------
    // Public API
    // ---------------------------------------------------------

    setMode(mode) {
        this._mode = mode;
        this._repositionBoards();
    }

    setSource(source) {
        this._source = source;
    }

    async addBoard(boardType, boardData) {
        // If already exists, just update its data
        if (this._boards.has(boardType)) {
            const entry = this._boards.get(boardType);
            entry.boardData = boardData;
            entry.active = true;
            await entry.renderer.loadBoardData(boardData, this._source);

            // Apply comparison color
            const color = theme.BOARD_COLORS[boardType] || theme.BLUE;
            entry.renderer.setColor(color);

            if (this._mode === 'overlay') {
                entry.renderer.setOpacity(0.5);
            }
        } else {
            const renderer = new BoardRenderer(this._scene);
            const color = theme.BOARD_COLORS[boardType] || theme.BLUE;
            renderer.setColor(color);

            await renderer.loadBoardData(boardData, this._source);

            if (this._mode === 'overlay') {
                renderer.setOpacity(0.5);
            }

            this._boards.set(boardType, {
                renderer,
                boardData,
                active: true,
                offsetY: 0,
            });
        }

        this._repositionBoards();
    }

    removeBoard(boardType) {
        const entry = this._boards.get(boardType);
        if (entry) {
            entry.active = false;
            entry.renderer.clear();
            this._repositionBoards();
        }
    }

    toggleBoard(boardType, enabled, boardData) {
        if (enabled && boardData) {
            return this.addBoard(boardType, boardData);
        } else {
            this.removeBoard(boardType);
        }
    }

    clear() {
        for (const [_, entry] of this._boards) {
            entry.renderer.dispose();
        }
        this._boards.clear();
    }

    // ---------------------------------------------------------
    // Board positioning
    // ---------------------------------------------------------

    _repositionBoards() {
        const activeEntries = [...this._boards.entries()]
            .filter(([_, e]) => e.active);

        if (activeEntries.length === 0) return;

        if (this._mode === 'side-by-side') {
            this._layoutSideBySide(activeEntries);
        } else {
            this._layoutOverlay(activeEntries);
        }

        // Reframe camera to show all boards
        this._frameCameraForComparison(activeEntries);
    }

    _layoutSideBySide(entries) {
        // Find the max width for spacing
        let maxWidth = 0;
        for (const [_, entry] of entries) {
            const params = entry.boardData.parameters;
            maxWidth = Math.max(maxWidth, params.maxWidthMm);
        }

        const spacing = maxWidth * 2.0;
        const totalSpan = (entries.length - 1) * spacing;
        const startY = -totalSpan / 2;

        entries.forEach(([type, entry], idx) => {
            const y = startY + idx * spacing;
            entry.renderer.setPosition(0, y, 0);
            entry.renderer.setOpacity(1.0);
            entry.offsetY = y;
        });
    }

    _layoutOverlay(entries) {
        // All boards at origin with transparency
        for (const [_, entry] of entries) {
            entry.renderer.setPosition(0, 0, 0);
            entry.renderer.setOpacity(0.45);
            entry.offsetY = 0;
        }
    }

    _frameCameraForComparison(entries) {
        const boardInfos = entries.map(([type, entry]) => ({
            lengthMm: entry.boardData.parameters.lengthMm,
            widthMm: entry.boardData.parameters.maxWidthMm,
            offsetY: entry.offsetY,
        }));

        this._sceneManager.frameMultipleBoards(boardInfos);
    }

    // ---------------------------------------------------------
    // Comparison table data
    // ---------------------------------------------------------

    getComparisonData() {
        const data = [];
        for (const [boardType, entry] of this._boards) {
            if (!entry.active) continue;
            const p = entry.boardData.parameters;
            const ph = entry.boardData.physics;
            data.push({
                type: boardType,
                lengthMm: p.lengthMm,
                widthMm: p.maxWidthMm,
                thicknessMm: p.maxThicknessMm,
                volumeL: p.volumeLiters,
                finConfig: p.finConfiguration,
                boardMassKg: ph ? ph.boardMassKg : null,
                buoyancyRatio: ph ? ph.buoyancy.buoyancyRatio : null,
            });
        }
        return data;
    }

    dispose() {
        this.clear();
    }
}
