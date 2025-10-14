import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple
from .colors import Palette
from .video import Video

class VideoOverlayRenderer:
    """Renders progressive pose path overlays on video frames."""

    def __init__(self,
                 line_thickness: int = 1,
                 show_current_point: bool = True,
                 min_visibility: float = 0.5,
                 paths_only: bool = False):
        """
        Initialize renderer with visual parameters.

        Args:
            line_thickness: Thickness of path lines
            show_current_point: Draw marker at current position
            min_visibility: Minimum visibility threshold for drawing
            paths_only: Render only paths on black background (no original video)
        """
        self.line_thickness = line_thickness
        self.show_current_point = show_current_point
        self.min_visibility = min_visibility
        self.paths_only = paths_only

        # Reuse SVGGenerator color scheme for consistency
        self.palette = Palette()

    def _render_paths_on_frame(self, frame: np.ndarray, landmark_paths: Dict[int, List[Tuple[int, int]]]) -> None:
        """
        Draw accumulated paths on frame (in-place modification).

        Args:
            frame: Video frame (BGR format, HxWx3)
            landmark_paths: Dict mapping landmark_id to list of (x, y) pixel points
        """
        for landmark_id, path in landmark_paths.items():
            if len(path) < 2:
                continue

            color_bgr = self.palette.get_landmark_color_bgr(landmark_id)

            # Draw accumulated path
            points = np.array(path, dtype=np.int32)
            cv2.polylines(
                frame,
                [points],
                False,
                color_bgr,
                self.line_thickness,
                cv2.LINE_AA
            )

            # Optionally highlight current position
            if self.show_current_point and len(path) > 0:
                current = path[-1]
                cv2.circle(
                    frame,
                    current,
                    5,
                    color_bgr,
                    -1,
                    cv2.LINE_AA
                )

    def render_overlay(self, video: Video, output_path: str,) -> bool:
        """
        Track poses and render progressive path overlays in one pass.

        Args:
            video_path: Path to input video file
            output_path: Path to output video file

        Returns:
            True if successful
        """
        # Get video properties
        width = video.width()
        height = video.height()
        fps = video.fps()

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Initialize path accumulator
        landmark_paths: Dict[int, List[Tuple[int, int]]] = {}

        while True:
            ok, frame, tracking_data = video.next_pose()
            if not ok:
                break

            # Extract and accumulate landmarks
            for tracking_datum in tracking_data:
                if tracking_datum['visibility'] >= self.min_visibility:
                    x_px = int(tracking_datum['x'] * width)
                    y_px = int(tracking_datum['y'] * height)

                    if tracking_datum['landmark_id'] not in landmark_paths:
                        landmark_paths[tracking_datum['landmark_id']] = []

                    landmark_paths[tracking_datum['landmark_id']].append((x_px, y_px))

            # Create black background if paths_only mode, otherwise use original frame
            if self.paths_only:
                frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Render accumulated paths on current frame
            self._render_paths_on_frame(frame, landmark_paths)

            # Write frame to output video
            writer.write(frame)

        return True
