// -- Viewer Theme -- //

// Centralized dark-mode theme for the Three.js surfboard viewer.
// Mirrors SurfPhysics/visualization/theme.py color palette.

// Primary color palette (Material Design: visible on dark backgrounds)
export const BLUE = 0x42A5F5;
export const RED = 0xEF5350;
export const GREEN = 0x66BB6A;
export const ORANGE = 0xFFA726;
export const PURPLE = 0xAB47BC;
export const BROWN = 0xA1887F;
export const CYAN = 0x26C6DA;

// Neutrals
export const WHITE = 0xE0E0E0;
export const REFERENCE_LINE = 0x888888;

// Scene
export const BACKGROUND = 0x1a1a2e;
export const GRID_COLOR = 0x333355;

// Ordered palette for multi-board comparison
export const PALETTE = [BLUE, RED, GREEN, ORANGE, PURPLE, BROWN, CYAN];

// CSS color strings (for UI panel and labels)
export const CSS = {
    blue: '#42A5F5',
    red: '#EF5350',
    green: '#66BB6A',
    orange: '#FFA726',
    purple: '#AB47BC',
    brown: '#A1887F',
    cyan: '#26C6DA',
    white: '#E0E0E0',
    background: '#1a1a2e',
    panelBackground: 'rgba(22, 22, 42, 0.92)',
    panelBorder: 'rgba(255, 255, 255, 0.08)',
    textPrimary: '#E0E0E0',
    textSecondary: '#9E9E9E',
    inputBackground: 'rgba(255, 255, 255, 0.06)',
    inputBorder: 'rgba(255, 255, 255, 0.12)',
    buttonBackground: 'rgba(66, 165, 245, 0.15)',
    buttonHover: 'rgba(66, 165, 245, 0.30)',
    sectionDivider: 'rgba(255, 255, 255, 0.06)',
};

// Force vector colors (for physics overlay)
export const FORCE_COLORS = {
    weight: RED,
    buoyancy: GREEN,
    drag: ORANGE,
    lift: BLUE,
};

// Board type display colors (for comparison mode)
export const BOARD_COLORS = {
    shortboard: BLUE,
    longboard: GREEN,
    fish: ORANGE,
    custom: PURPLE,
};
