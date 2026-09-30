// -- Application Bootstrap -- //

// Main entry point for the Three.js surfboard viewer.
// Initializes all modules and wires UI events to scene updates.

import { SceneManager } from './sceneManager.js';
import { BoardRenderer, SOURCE_PARAMETRIC, SOURCE_STL } from './boardRenderer.js';
import { PhysicsOverlay } from './physicsOverlay.js';
import { BoardComparison } from './boardComparison.js';
import { DeviationHeatmap } from './deviationHeatmap.js';
import { UIPanel } from './uiPanel.js';
import * as theme from './theme.js';


class SurfViewerApp {
    constructor() {
        this._sceneManager = null;
        this._primaryRenderer = null;
        this._physicsOverlay = null;
        this._comparison = null;
        this._deviationHeatmap = null;
        this._uiPanel = null;

        // Currently loaded board data (keyed by board type)
        this._loadedData = new Map();

        // Current state
        this._currentBoardType = 'shortboard';
        this._currentSource = SOURCE_PARAMETRIC;
        this._comparisonActive = false;
        this._deviationActive = false;
    }

    async init() {
        // Scene
        const viewport = document.getElementById('viewport');
        this._sceneManager = new SceneManager(viewport);

        // Board renderer (primary, single-board view)
        this._primaryRenderer = new BoardRenderer(this._sceneManager.scene);

        // Physics overlay
        this._physicsOverlay = new PhysicsOverlay(this._sceneManager.scene);

        // Board comparison
        this._comparison = new BoardComparison(
            this._sceneManager.scene, this._sceneManager
        );

        // Deviation heatmap
        this._deviationHeatmap = new DeviationHeatmap(this._sceneManager.scene);

        // UI panel
        const panel = document.getElementById('panel');
        this._uiPanel = new UIPanel(panel);

        // Wire events
        this._wireEvents();

        // Start render loop
        this._sceneManager.start();

        // Load default board
        await this._loadBoard('shortboard');
    }

    // ---------------------------------------------------------
    // Event wiring
    // ---------------------------------------------------------

    _wireEvents() {
        // Board selection
        this._uiPanel.on('boardChange', async (e) => {
            const { boardType } = e.detail;
            this._currentBoardType = boardType;
            await this._loadBoard(boardType);
        });

        // Geometry source
        this._uiPanel.on('sourceChange', async (e) => {
            const { source } = e.detail;
            this._currentSource = source;
            if (this._comparisonActive) {
                this._comparison.setSource(source);
                // Reload all active comparison boards
                for (const type of this._comparison.activeBoards) {
                    const data = this._loadedData.get(type);
                    if (data) await this._comparison.addBoard(type, data);
                }
            } else {
                await this._reloadPrimary();
            }
        });

        // Render mode
        this._uiPanel.on('renderModeChange', (e) => {
            const { mode } = e.detail;
            this._primaryRenderer.setRenderMode(mode);
        });

        // Physics overlays
        this._uiPanel.on('overlayToggle', (e) => {
            const { overlay, enabled } = e.detail;
            this._physicsOverlay.setVisible(overlay, enabled);
        });

        // Comparison mode
        this._uiPanel.on('comparisonModeChange', (e) => {
            const { mode } = e.detail;
            this._comparison.setMode(mode);
        });

        // Comparison board toggle
        this._uiPanel.on('comparisonBoardToggle', async (e) => {
            const { boardType, enabled } = e.detail;

            if (enabled) {
                // Load data if needed
                let data = this._loadedData.get(boardType);
                if (!data) {
                    data = await this._fetchBoardData(boardType);
                    if (data) this._loadedData.set(boardType, data);
                }

                if (data) {
                    // Hide primary renderer, switch to comparison mode
                    if (!this._comparisonActive) {
                        this._comparisonActive = true;
                        this._primaryRenderer.clear();
                        // Add current primary board to comparison
                        const primaryData = this._loadedData.get(this._currentBoardType);
                        if (primaryData) {
                            await this._comparison.addBoard(this._currentBoardType, primaryData);
                        }
                    }
                    await this._comparison.addBoard(boardType, data);
                }
            } else {
                this._comparison.removeBoard(boardType);

                // If no comparison boards remain, exit comparison mode
                if (this._comparison.activeBoards.length <= 1) {
                    this._comparisonActive = false;
                    this._comparison.clear();
                    await this._reloadPrimary();
                }
            }
        });

        // Deviation heatmap toggle
        this._uiPanel.on('deviationToggle', (e) => {
            const { enabled } = e.detail;
            this._deviationActive = enabled;
            this._deviationHeatmap.setVisible(enabled);

            // If enabling and no data loaded, try loading default
            if (enabled && !this._deviationHeatmap.getStats()) {
                this._loadDeviationData('data/deviationData.json');
            }

            // Hide primary mesh when showing heatmap for clarity
            if (enabled) {
                this._primaryRenderer.clear();
            } else {
                this._reloadPrimary();
            }
        });

        // Deviation data load
        this._uiPanel.on('deviationLoad', async (e) => {
            const { path } = e.detail;
            await this._loadDeviationData(path);
        });
    }

    async _loadDeviationData(path) {
        const success = await this._deviationHeatmap.loadDeviationData(path);
        if (success) {
            const stats = this._deviationHeatmap.getStats();
            this._uiPanel.updateDeviationStats(stats);

            if (this._deviationActive) {
                this._deviationHeatmap.setVisible(true);
            }

            console.log('Deviation data loaded:', stats);
        } else {
            console.warn('Failed to load deviation data from:', path);
            this._uiPanel.updateDeviationStats(null);
        }
    }

    // ---------------------------------------------------------
    // Board loading
    // ---------------------------------------------------------

    async _loadBoard(boardType) {
        let boardData = this._loadedData.get(boardType);
        if (!boardData) {
            boardData = await this._fetchBoardData(boardType);
            if (!boardData) {
                console.error(`Failed to load board data for: ${boardType}`);
                return;
            }
            this._loadedData.set(boardType, boardData);
        }

        // If in comparison mode, update comparison instead
        if (this._comparisonActive) {
            await this._comparison.addBoard(boardType, boardData);
        } else {
            // Apply board color
            const color = theme.BOARD_COLORS[boardType] || theme.BLUE;
            this._primaryRenderer.setColor(color);

            await this._primaryRenderer.loadBoardData(boardData, this._currentSource);

            // Update physics overlay
            this._physicsOverlay.updateFromBoardData(boardData);

            // Frame camera
            const bounds = this._primaryRenderer.getBounds();
            if (bounds) {
                this._sceneManager.frameBoardBounds(
                    bounds.lengthMm, bounds.widthMm, bounds.thicknessMm
                );
                this._sceneManager.showGrid(bounds.lengthMm);
            }
        }

        // Update info panel
        this._uiPanel.updateBoardInfo(boardData);
    }

    async _reloadPrimary() {
        const boardData = this._loadedData.get(this._currentBoardType);
        if (boardData) {
            await this._primaryRenderer.loadBoardData(boardData, this._currentSource);

            const bounds = this._primaryRenderer.getBounds();
            if (bounds) {
                this._sceneManager.frameBoardBounds(
                    bounds.lengthMm, bounds.widthMm, bounds.thicknessMm
                );
            }
        }
    }

    async _fetchBoardData(boardType) {
        const url = `data/boardData_${boardType}.json`;
        try {
            const response = await fetch(url);
            if (!response.ok) {
                console.warn(`Board data not found: ${url} (${response.status})`);
                return null;
            }
            return await response.json();
        } catch (e) {
            console.error(`Error fetching board data: ${url}`, e);
            return null;
        }
    }
}


// ---------------------------------------------------------
// Initialize on DOM ready
// ---------------------------------------------------------

document.addEventListener('DOMContentLoaded', async () => {
    const app = new SurfViewerApp();
    try {
        await app.init();
    } catch (e) {
        console.error('Failed to initialize SurfViewer:', e);
        const viewport = document.getElementById('viewport');
        if (viewport) {
            const msg = document.createElement('div');
            msg.style.cssText = `
                position: absolute; top: 50%; left: 50%;
                transform: translate(-50%, -50%);
                color: ${theme.CSS.red}; font-family: monospace;
                font-size: 14px; text-align: center; max-width: 500px;
                padding: 24px; background: rgba(0,0,0,0.7);
                border-radius: 8px;
            `;
            msg.innerHTML = `
                <div style="font-size: 18px; margin-bottom: 12px;">Failed to load viewer</div>
                <div>${e.message}</div>
                <div style="margin-top: 16px; color: ${theme.CSS.textSecondary}; font-size: 12px;">
                    Ensure you have exported board data by running the analysis pipeline,<br>
                    and that you are serving this page via HTTP (not file://).<br><br>
                    <code>python -m http.server 8080 --directory SurfViewer</code>
                </div>
            `;
            viewport.appendChild(msg);
        }
    }
});
