from pydoc import doc
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple

class Video:
    def __init__(self, path: str, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        """
        Initialize the video object and open the video file for reading.
        """
        self.path = path
        self.cap = cv2.VideoCapture(self.path)
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {self.path}")

        self.pose = mp.solutions.pose.Pose(min_detection_confidence=self.min_detection_confidence, min_tracking_confidence=self.min_tracking_confidence)
        self.pose_frame_count = 0

    def next_pose(self) -> Tuple[bool, np.ndarray, Dict]:
        if not self.cap.isOpened():
            return False, None, []

        ret, frame = self.cap.read()
        if not ret:
            return False, None, []

        frame_tracking_data = []
        timestamp = self.pose_frame_count / self.fps()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmark_name = mp.solutions.pose.PoseLandmark(idx).name
                frame_tracking_data.append({
                    'frame': self.pose_frame_count,
                    'timestamp': timestamp,
                    'landmark_name': landmark_name,
                    'landmark_id': idx,
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            
            self.pose_frame_count += 1
        
        return True, frame, frame_tracking_data

    def close(self):
        self.pose.close()
        self.cap.release()

    def track_poses(self) -> List[Dict]:
        """
        Track poses in the video and return tracking data.
        """
        tracking_data = []

        while True:
            ok, frame, frame_tracking_data = self.next_pose()
            if not ok:
                break

            for frame in frame_tracking_data:
                tracking_data.append(frame)
        
        return tracking_data


    def width(self) -> int:
        """
        Returns the width of the video.
        """
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def height(self) -> int:
        """
        Returns the height of the video.
        """
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def fps(self) -> float:
        """ 
        Returns the frames per second of the video.
        """
        return self.cap.get(cv2.CAP_PROP_FPS)

    def total_frames(self) -> int:
        """
        Returns the total number of frames in the video.
        """
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
