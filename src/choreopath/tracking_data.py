import pandas as pd
import numpy as np
import mediapipe as mp
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class TrackingData:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None

        # MediaPipe pose connections for drawing skeleton
        self.pose_connections = [
            # Face
            (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
            (0, 9), (0, 10), (9, 10),
            # Upper body
            (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
            (11, 23), (12, 24), (23, 24),
            # Lower body
            (23, 25), (25, 27), (27, 29), (27, 31),
            (24, 26), (26, 28), (28, 30), (28, 32)
        ]

    def load(self) -> bool:
        """
        Load a CSV file containing tracking data.
        """
        self.df = pd.read_csv(self.csv_path)

        required_columns = ['frame', 'timestamp', 'landmark_name', 'landmark_id', 'x', 'y', 'z', 'visibility']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def reject_invisible(self, threshold: float = 0.5):
        """
        Reject rows where the visibility is less than the threshold.
        """
        self.df = self.df[self.df['visibility'] >= threshold].copy()

    def sort_by_frame_and_landmark_id(self):
        """ 
        Sort the underlying dataframe by frame and landmark_id.
        """
        self.df = self.df.sort_values(['frame', 'landmark_id']).copy()

    def total_frames(self) -> int:
        """
        Return the total number of frames in the dataframe.
        """
        return self.df['frame'].nunique()

    def __total_possible_landmarks(self) -> int:
        """
        Return the total number of landmarks exposed by MediaPipe.
        """
        return len(mp.solutions.pose.PoseLandmark)

    def audit(self) -> Dict:
        """
        Audit the underlying dataframe for data quality and consistency.
        """
        audit_results = {
            'total_rows': len(self.df),
            'unique_frames': self.total_frames(),
            'frame_range': (self.df['frame'].min(), self.df['frame'].max()),
            'timestamp_range': (self.df['timestamp'].min(), self.df['timestamp'].max()),
            'missing_values': {},
            'coordinate_ranges': {},
            'visibility_stats': {},
            'landmarks_per_frame': {},
            'anomalies': []
        }

        # Check for missing values in coordinates and visibility columns
        for col in ['x', 'y', 'z', 'visibility']:
            missing = self.df[col].isnull().sum()
            audit_results['missing_values'][col] = missing
            if missing > 0:
                audit_results['anomalies'].append(f"Missing {missing} values in column '{col}'")

        # Check coordinate ranges (should be 0-1 for x,y and normalized for z)
        for coord in ['x', 'y', 'z']:
            coord_data = self.df[coord].dropna()
            audit_results['coordinate_ranges'][coord] = {
                'min': float(coord_data.min()),
                'max': float(coord_data.max()),
                'mean': float(coord_data.mean()),
                'std': float(coord_data.std())
            }
            
            if coord in ['x', 'y']:
                out_of_range = ((coord_data < 0) | (coord_data > 1)).sum()
                if out_of_range > 0:
                    audit_results['anomalies'].append(f"{out_of_range} {coord} coordinates outside 0-1 range")

        # Visibility statistics
        vis_data = self.df['visibility'].dropna()
        audit_results['visibility_stats'] = {
            'mean': float(vis_data.mean()),
            'min': float(vis_data.min()),
            'max': float(vis_data.max()),
            'low_visibility_count': int((vis_data < 0.5).sum())
        }

        landmarks_per_frame = self.df.groupby('frame')['landmark_id'].nunique()
        audit_results['landmarks_per_frame'] = {
            'expected': self.__total_possible_landmarks(),
            'actual_range': (int(landmarks_per_frame.min()), int(landmarks_per_frame.max())),
            'frames_with_missing_landmarks': int((landmarks_per_frame < self.__total_possible_landmarks()).sum())
        }

        # Check timestamp continuity
        unique_timestamps = self.df['timestamp'].unique()
        if len(unique_timestamps) > 1:
            timestamp_diffs = np.diff(sorted(unique_timestamps))
            expected_fps = 1.0 / np.median(timestamp_diffs)
            audit_results['estimated_fps'] = float(expected_fps)
            
            # Check for irregular timing
            irregular_timing = np.std(timestamp_diffs) / np.mean(timestamp_diffs)
            if irregular_timing > 0.1:  # 10% variation threshold
                audit_results['anomalies'].append(f"Irregular frame timing detected (CV: {irregular_timing:.3f})")

        return audit_results

    def to_animation(self, output_path: str = "tracking_animation.mp4", fps: int = 24) -> bool:
        """
        Use matplotlib to create an animation of the tracking data.
        """

        # Prepare data
        frames = sorted(self.df['frame'].unique())
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Invert Y-axis to match image coordinates
        ax.set_title('Pose Tracking Animation')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # Initialize empty plots
        points = ax.scatter([], [], c=[], s=50, cmap='viridis', vmin=0, vmax=1)
        lines = []
        for _ in self.pose_connections:
            line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7)
            lines.append(line)
        
        frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add colorbar for visibility
        cbar = plt.colorbar(points, ax=ax)
        cbar.set_label('Visibility Score')
        
        def animate(frame_idx):
            frame_num = frames[frame_idx]
            frame_data = self.df[self.df['frame'] == frame_num]
            
            if len(frame_data) == 0:
                return points, *lines, frame_text
            
            # Sort by landmark_id to ensure consistent ordering
            frame_data = frame_data.sort_values('landmark_id')
            
            # Update points
            x_coords = frame_data['x'].values
            y_coords = frame_data['y'].values
            visibility = frame_data['visibility'].values
            
            points.set_offsets(np.column_stack([x_coords, y_coords]))
            points.set_array(visibility)
            
            # Update skeleton lines
            for i, (start_idx, end_idx) in enumerate(self.pose_connections):
                if start_idx < len(x_coords) and end_idx < len(x_coords):
                    # Only draw line if both points have reasonable visibility
                    if visibility[start_idx] > 0.3 and visibility[end_idx] > 0.3:
                        lines[i].set_data([x_coords[start_idx], x_coords[end_idx]], 
                                        [y_coords[start_idx], y_coords[end_idx]])
                        lines[i].set_alpha(min(visibility[start_idx], visibility[end_idx]))
                    else:
                        lines[i].set_data([], [])
                else:
                    lines[i].set_data([], [])
            
            # Update frame info
            timestamp = frame_data['timestamp'].iloc[0] if len(frame_data) > 0 else 0
            frame_text.set_text(f'Frame: {frame_num}\nTime: {timestamp:.2f}s')
            
            return points, *lines, frame_text
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(frames), 
                                        interval=1000/fps, blit=True, repeat=True)
        
        # Save animation
        anim.save(output_path, writer='ffmpeg', fps=fps)
        plt.close()
