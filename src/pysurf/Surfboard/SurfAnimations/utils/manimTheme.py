# -- Manim Animation Theme -- #

'''

Centralized color theme for all SurfAnimations Manim scenes.

Mirrors the SurfPhysics visualization theme (theme.py) for visual
consistency across Plotly dashboards, Three.js viewer, and Manim animations.

Sean Bowman [02/04/2026]

'''

from manim import ManimColor

#--------------------------------------------------------------------#
# -- Primary Color Palette (Material Design) -- #
#--------------------------------------------------------------------#

BLUE = ManimColor('#42A5F5')
RED = ManimColor('#EF5350')
GREEN = ManimColor('#66BB6A')
ORANGE = ManimColor('#FFA726')
PURPLE = ManimColor('#AB47BC')
BROWN = ManimColor('#A1887F')
CYAN = ManimColor('#26C6DA')

# Neutrals
WHITE = ManimColor('#E0E0E0')
REFERENCE_LINE = ManimColor('#888888')

# Ordered palette for multi-series
PALETTE = [BLUE, RED, GREEN, ORANGE, PURPLE, BROWN, CYAN]

#--------------------------------------------------------------------#
# -- Semantic Force Colors -- #
#--------------------------------------------------------------------#

WEIGHT_COLOR = RED
BUOYANCY_COLOR = GREEN
DRAG_COLOR = ORANGE
LIFT_COLOR = BLUE

#--------------------------------------------------------------------#
# -- Scene Colors -- #
#--------------------------------------------------------------------#

WAVE_COLOR = BLUE
BOARD_COLOR = WHITE
WATER_FILL = ManimColor('#1a3a5c')
BG_COLOR = ManimColor('#1a1a2e')

#--------------------------------------------------------------------#
# -- Board-Specific Colors -- #
#--------------------------------------------------------------------#

SHORTBOARD_COLOR = BLUE
LONGBOARD_COLOR = GREEN
FISH_COLOR = ORANGE

BOARD_COLORS = {
    'shortboard': SHORTBOARD_COLOR,
    'longboard': LONGBOARD_COLOR,
    'fish': FISH_COLOR,
}
