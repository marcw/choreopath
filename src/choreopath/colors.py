from typing import Tuple

DEFAULT_PALETTE = {
    'face': '#f87171',
    'left_arm': '#fb923c',
    'right_arm': '#facc15',
    'hips': '#71717a',
    'left_leg': '#06b6d4',
    'right_leg': '#3b82f6'
}

WHITE_AND_GOLD_PALETTE = {
    'face': '#ffffff',
    'left_arm': '#ffd700',
    'right_arm': '#ffffff',
    'hips': '#ffffff',
    'left_leg': '#ffffff',
    'right_leg': '#ffffff'
}

class Palette:
    def __init__(self, palette: str = 'default'):
        self.body_colors = DEFAULT_PALETTE
        if palette == 'white_and_gold':
            self.body_colors = WHITE_AND_GOLD_PALETTE

    def get_body_region_color(self, body_region: str) -> str:
        """Get color for a specific body region."""
        return self.body_colors[body_region]

    def get_landmark_color(self, landmark_id: int) -> str:
        """Get color for a specific landmark based on body region."""
        if landmark_id <= 10:
            return self.body_colors['face']
        elif landmark_id in [11, 13, 15, 17, 19, 21]:
            return self.body_colors['left_arm']
        elif landmark_id in [12, 14, 16, 18, 20, 22]:
            return self.body_colors['right_arm']
        elif landmark_id in [23, 24]:
            return self.body_colors['hips']
        elif landmark_id in [25, 27, 29, 31]:
            return self.body_colors['left_leg']
        elif landmark_id in [26, 28, 30, 32]:
            return self.body_colors['right_leg']
        else:
            return '#888888'  # Gray fallback

    def get_landmark_color_bgr(self, landmark_id: int) -> Tuple[int, int, int]:
        """Get BGR color tuple for landmark based on body region."""
        return self.hex_to_bgr(self.get_landmark_color(landmark_id))

    def hex_to_bgr(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color (#RRGGBB) to BGR tuple for OpenCV."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (b, g, r)
