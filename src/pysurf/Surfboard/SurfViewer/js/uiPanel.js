// -- UI Panel -- //

// Control panel for the 3D viewer: board selector, render modes,
// physics overlay toggles, and comparison controls.
// Dispatches custom events so other modules can react to changes.

import { CSS } from './theme.js';
import {
    RENDER_SOLID, RENDER_WIREFRAME, RENDER_TRANSPARENT,
    SOURCE_PARAMETRIC, SOURCE_STL,
} from './boardRenderer.js';


export class UIPanel {
    constructor(panelElement) {
        this._panel = panelElement;
        this._eventTarget = new EventTarget();
        this._buildPanel();
    }

    // ---------------------------------------------------------
    // Event system
    // ---------------------------------------------------------

    on(eventName, callback) {
        this._eventTarget.addEventListener(eventName, callback);
    }

    _emit(eventName, detail) {
        this._eventTarget.dispatchEvent(
            new CustomEvent(eventName, { detail })
        );
    }

    // ---------------------------------------------------------
    // Panel construction
    // ---------------------------------------------------------

    _buildPanel() {
        this._panel.innerHTML = '';

        // Title
        const title = document.createElement('h2');
        title.textContent = 'SurfViewer';
        title.style.cssText = `
            margin: 0 0 16px 0; font-size: 18px; font-weight: 600;
            color: ${CSS.blue}; letter-spacing: 1px;
        `;
        this._panel.appendChild(title);

        // Board selector
        this._buildSection('Board', this._buildBoardSelector.bind(this));

        // Geometry source toggle
        this._buildSection('Geometry', this._buildGeometryToggle.bind(this));

        // Render mode
        this._buildSection('Render Mode', this._buildRenderModeToggle.bind(this));

        // Physics overlays
        this._buildSection('Physics Overlays', this._buildPhysicsToggles.bind(this));

        // Comparison
        this._buildSection('Board Comparison', this._buildComparisonControls.bind(this));

        // Deviation Heatmap
        this._buildSection('Deviation Analysis', this._buildDeviationControls.bind(this));

        // Info panel
        this._infoSection = this._buildSection('Board Info', () => {
            const container = document.createElement('div');
            container.id = 'board-info-content';
            container.style.cssText = `
                font-size: 11px; line-height: 1.6;
                color: ${CSS.textSecondary}; font-family: monospace;
            `;
            container.textContent = 'Load a board to see parameters.';
            return container;
        });
    }

    _buildSection(label, contentBuilder) {
        const section = document.createElement('div');
        section.style.cssText = `
            margin-bottom: 16px; padding-bottom: 12px;
            border-bottom: 1px solid ${CSS.sectionDivider};
        `;

        const header = document.createElement('div');
        header.textContent = label;
        header.style.cssText = `
            font-size: 11px; font-weight: 600; text-transform: uppercase;
            letter-spacing: 1.5px; color: ${CSS.textSecondary};
            margin-bottom: 8px;
        `;
        section.appendChild(header);

        const content = contentBuilder();
        section.appendChild(content);

        this._panel.appendChild(section);
        return section;
    }

    // ---------------------------------------------------------
    // Board selector
    // ---------------------------------------------------------

    _buildBoardSelector() {
        const container = document.createElement('div');

        const select = document.createElement('select');
        select.id = 'board-select';
        select.style.cssText = this._selectStyle();

        const boards = [
            { value: 'shortboard', label: 'Shortboard (6\'0")' },
            { value: 'longboard', label: 'Longboard (9\'0")' },
            { value: 'fish', label: 'Fish / RNF (5\'6")' },
        ];

        for (const b of boards) {
            const opt = document.createElement('option');
            opt.value = b.value;
            opt.textContent = b.label;
            select.appendChild(opt);
        }

        select.addEventListener('change', () => {
            this._emit('boardChange', { boardType: select.value });
        });

        container.appendChild(select);

        // Load button
        const loadBtn = document.createElement('button');
        loadBtn.textContent = 'Load Board';
        loadBtn.style.cssText = this._buttonStyle();
        loadBtn.addEventListener('click', () => {
            this._emit('boardChange', { boardType: select.value });
        });
        container.appendChild(loadBtn);

        return container;
    }

    // ---------------------------------------------------------
    // Geometry source toggle
    // ---------------------------------------------------------

    _buildGeometryToggle() {
        const container = document.createElement('div');
        container.style.cssText = 'display: flex; gap: 8px;';

        const sources = [
            { value: SOURCE_PARAMETRIC, label: 'Parametric' },
            { value: SOURCE_STL, label: 'STL Mesh' },
        ];

        for (const src of sources) {
            const btn = document.createElement('button');
            btn.textContent = src.label;
            btn.dataset.source = src.value;
            btn.style.cssText = this._toggleButtonStyle(src.value === SOURCE_PARAMETRIC);
            btn.addEventListener('click', () => {
                // Update button states
                container.querySelectorAll('button').forEach(b => {
                    b.style.cssText = this._toggleButtonStyle(b.dataset.source === src.value);
                });
                this._emit('sourceChange', { source: src.value });
            });
            container.appendChild(btn);
        }

        return container;
    }

    // ---------------------------------------------------------
    // Render mode toggle
    // ---------------------------------------------------------

    _buildRenderModeToggle() {
        const container = document.createElement('div');
        container.style.cssText = 'display: flex; gap: 6px;';

        const modes = [
            { value: RENDER_SOLID, label: 'Solid' },
            { value: RENDER_WIREFRAME, label: 'Wire' },
            { value: RENDER_TRANSPARENT, label: 'Glass' },
        ];

        for (const mode of modes) {
            const btn = document.createElement('button');
            btn.textContent = mode.label;
            btn.dataset.mode = mode.value;
            btn.style.cssText = this._toggleButtonStyle(mode.value === RENDER_SOLID);
            btn.addEventListener('click', () => {
                container.querySelectorAll('button').forEach(b => {
                    b.style.cssText = this._toggleButtonStyle(b.dataset.mode === mode.value);
                });
                this._emit('renderModeChange', { mode: mode.value });
            });
            container.appendChild(btn);
        }

        return container;
    }

    // ---------------------------------------------------------
    // Physics overlay toggles
    // ---------------------------------------------------------

    _buildPhysicsToggles() {
        const container = document.createElement('div');

        const overlays = [
            { id: 'waterline', label: 'Waterline', checked: false },
            { id: 'forceVectors', label: 'Force Vectors', checked: false },
            { id: 'draft', label: 'Draft Indicator', checked: false },
        ];

        for (const overlay of overlays) {
            const row = document.createElement('label');
            row.style.cssText = `
                display: flex; align-items: center; gap: 8px;
                margin-bottom: 6px; cursor: pointer; font-size: 13px;
                color: ${CSS.textPrimary};
            `;

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = `overlay-${overlay.id}`;
            checkbox.checked = overlay.checked;
            checkbox.style.cssText = 'accent-color: ' + CSS.blue + ';';
            checkbox.addEventListener('change', () => {
                this._emit('overlayToggle', {
                    overlay: overlay.id,
                    enabled: checkbox.checked,
                });
            });

            row.appendChild(checkbox);
            row.appendChild(document.createTextNode(overlay.label));
            container.appendChild(row);
        }

        return container;
    }

    // ---------------------------------------------------------
    // Comparison controls
    // ---------------------------------------------------------

    _buildComparisonControls() {
        const container = document.createElement('div');

        // Mode toggle
        const modeRow = document.createElement('div');
        modeRow.style.cssText = 'display: flex; gap: 6px; margin-bottom: 10px;';

        const compModes = [
            { value: 'side-by-side', label: 'Side by Side' },
            { value: 'overlay', label: 'Overlay' },
        ];

        for (const mode of compModes) {
            const btn = document.createElement('button');
            btn.textContent = mode.label;
            btn.dataset.compMode = mode.value;
            btn.style.cssText = this._toggleButtonStyle(mode.value === 'side-by-side');
            btn.addEventListener('click', () => {
                modeRow.querySelectorAll('button').forEach(b => {
                    b.style.cssText = this._toggleButtonStyle(b.dataset.compMode === mode.value);
                });
                this._emit('comparisonModeChange', { mode: mode.value });
            });
            modeRow.appendChild(btn);
        }
        container.appendChild(modeRow);

        // Board checkboxes for comparison
        const boards = [
            { id: 'shortboard', label: 'Shortboard', color: CSS.blue },
            { id: 'longboard', label: 'Longboard', color: CSS.green },
            { id: 'fish', label: 'Fish', color: CSS.orange },
        ];

        for (const board of boards) {
            const row = document.createElement('label');
            row.style.cssText = `
                display: flex; align-items: center; gap: 8px;
                margin-bottom: 4px; cursor: pointer; font-size: 13px;
                color: ${CSS.textPrimary};
            `;

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = `compare-${board.id}`;
            checkbox.style.cssText = `accent-color: ${board.color};`;
            checkbox.addEventListener('change', () => {
                this._emit('comparisonBoardToggle', {
                    boardType: board.id,
                    enabled: checkbox.checked,
                });
            });

            const colorDot = document.createElement('span');
            colorDot.style.cssText = `
                width: 8px; height: 8px; border-radius: 50%;
                background: ${board.color}; display: inline-block;
            `;

            row.appendChild(checkbox);
            row.appendChild(colorDot);
            row.appendChild(document.createTextNode(board.label));
            container.appendChild(row);
        }

        return container;
    }

    // ---------------------------------------------------------
    // Deviation analysis controls
    // ---------------------------------------------------------

    _buildDeviationControls() {
        const container = document.createElement('div');

        // Enable toggle
        const enableRow = document.createElement('label');
        enableRow.style.cssText = `
            display: flex; align-items: center; gap: 8px;
            margin-bottom: 10px; cursor: pointer; font-size: 13px;
            color: ${CSS.textPrimary};
        `;

        const enableCheckbox = document.createElement('input');
        enableCheckbox.type = 'checkbox';
        enableCheckbox.id = 'deviation-enable';
        enableCheckbox.style.cssText = 'accent-color: ' + CSS.purple + ';';
        enableCheckbox.addEventListener('change', () => {
            this._emit('deviationToggle', { enabled: enableCheckbox.checked });
        });

        enableRow.appendChild(enableCheckbox);
        enableRow.appendChild(document.createTextNode('Show Deviation Heatmap'));
        container.appendChild(enableRow);

        // Reference file input
        const fileRow = document.createElement('div');
        fileRow.style.cssText = 'margin-bottom: 10px;';

        const fileLabel = document.createElement('div');
        fileLabel.textContent = 'Deviation Data:';
        fileLabel.style.cssText = `
            font-size: 11px; color: ${CSS.textSecondary};
            margin-bottom: 4px;
        `;
        fileRow.appendChild(fileLabel);

        const fileInput = document.createElement('input');
        fileInput.type = 'text';
        fileInput.id = 'deviation-file';
        fileInput.value = 'data/deviationData.json';
        fileInput.style.cssText = `
            width: 100%; box-sizing: border-box;
            padding: 6px 8px; background: ${CSS.inputBackground};
            color: ${CSS.textPrimary}; border: 1px solid ${CSS.inputBorder};
            border-radius: 4px; font-size: 11px; font-family: monospace;
        `;
        fileRow.appendChild(fileInput);

        const loadBtn = document.createElement('button');
        loadBtn.textContent = 'Load Data';
        loadBtn.style.cssText = this._buttonStyle();
        loadBtn.style.marginTop = '6px';
        loadBtn.addEventListener('click', () => {
            this._emit('deviationLoad', { path: fileInput.value });
        });
        fileRow.appendChild(loadBtn);

        container.appendChild(fileRow);

        // Stats display
        const statsDiv = document.createElement('div');
        statsDiv.id = 'deviation-stats';
        statsDiv.style.cssText = `
            font-size: 11px; line-height: 1.5; margin-top: 8px;
            color: ${CSS.textSecondary}; font-family: monospace;
            padding: 8px; background: ${CSS.inputBackground};
            border-radius: 4px; display: none;
        `;
        container.appendChild(statsDiv);

        return container;
    }

    updateDeviationStats(stats) {
        const statsDiv = document.getElementById('deviation-stats');
        if (!statsDiv || !stats) {
            if (statsDiv) statsDiv.style.display = 'none';
            return;
        }

        statsDiv.style.display = 'block';
        statsDiv.innerHTML = `
<strong style="color:${CSS.purple}">Deviation Statistics</strong>
RMS:  ${stats.rmsMm?.toFixed(2) || ': '} mm
Mean: ${stats.meanMm?.toFixed(2) || ': '} mm
Min:  ${stats.minMm?.toFixed(2) || ': '} mm
Max:  ${stats.maxMm?.toFixed(2) || ': '} mm
        `.trim();
    }

    // ---------------------------------------------------------
    // Info panel update
    // ---------------------------------------------------------

    updateBoardInfo(boardData) {
        const content = document.getElementById('board-info-content');
        if (!content || !boardData) return;

        const p = boardData.parameters;
        const ph = boardData.physics;
        const b = ph ? ph.buoyancy : {};

        const mmToFtIn = (mm) => {
            const totalIn = mm / 25.4;
            const ft = Math.floor(totalIn / 12);
            const inches = totalIn - ft * 12;
            return `${ft}'${inches.toFixed(1)}"`;
        };

        content.innerHTML = `
<strong style="color:${CSS.blue}">${boardData.meta.boardType.toUpperCase()}</strong>
${mmToFtIn(p.lengthMm)} x ${(p.maxWidthMm / 25.4).toFixed(1)}" x ${(p.maxThicknessMm / 25.4).toFixed(2)}"

Volume:    ${p.volumeLiters.toFixed(1)} L
Fins:      ${p.finConfiguration}
Tail:      ${p.tailShape}
Foam:      ${p.foamType.toUpperCase()}

<strong style="color:${CSS.textPrimary}">Physics</strong>
Board Mass:  ${ph ? ph.boardMassKg.toFixed(2) : ': '} kg
Total Wt:   ${b.totalWeightN ? b.totalWeightN.toFixed(0) : ': '} N
Buoyancy:   ${b.maxBuoyancyN ? b.maxBuoyancyN.toFixed(0) : ': '} N
Ratio:      ${b.buoyancyRatio ? b.buoyancyRatio.toFixed(2) : ': '}x
Draft:      ${b.riderDraftCm ? b.riderDraftCm.toFixed(1) : ': '} cm
Floats:     ${b.floats ? 'Yes' : 'No'}
        `.trim();
    }

    // ---------------------------------------------------------
    // Style helpers
    // ---------------------------------------------------------

    _selectStyle() {
        return `
            width: 100%; padding: 8px 10px; margin-bottom: 8px;
            background: ${CSS.inputBackground}; color: ${CSS.textPrimary};
            border: 1px solid ${CSS.inputBorder}; border-radius: 4px;
            font-size: 13px; outline: none; cursor: pointer;
        `;
    }

    _buttonStyle() {
        return `
            width: 100%; padding: 8px; margin-top: 4px;
            background: ${CSS.buttonBackground}; color: ${CSS.blue};
            border: 1px solid ${CSS.inputBorder}; border-radius: 4px;
            font-size: 13px; font-weight: 600; cursor: pointer;
            transition: background 0.15s;
        `;
    }

    _toggleButtonStyle(active) {
        const bg = active ? CSS.buttonHover : CSS.inputBackground;
        const color = active ? CSS.blue : CSS.textSecondary;
        const border = active ? CSS.blue : CSS.inputBorder;
        return `
            flex: 1; padding: 6px 8px;
            background: ${bg}; color: ${color};
            border: 1px solid ${border}; border-radius: 4px;
            font-size: 12px; font-weight: 500; cursor: pointer;
            transition: all 0.15s;
        `;
    }
}
