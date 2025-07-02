import cv2
import mediapipe as mp
from typing import List, Dict

class Video:
    def __init__(self, path: str):
        """
        Initialize the video object and open the video file for reading.
        """
        self.path = path
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {self.path}")

    def track_poses(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5) -> List[Dict]:
        """
        Track poses in the video and return tracking data.
        """
        pose = mp.solutions.pose.Pose(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)
        _frame_count = 0
        _fps = self.fps()
        _total_frames = self.total_frames()

        tracking_data = []

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = pose.process(rgb_frame)

            _timestamp = _frame_count / _fps

            # Extract landmarks
            if results.pose_landmarks:
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    landmark_name = mp.solutions.pose.PoseLandmark(idx).name
                    tracking_data.append({
                        'frame': _frame_count,
                        'timestamp': _timestamp,
                        'landmark_name': landmark_name,
                        'landmark_id': idx,
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
            
            _frame_count += 1
        
        self.cap.release()

        return tracking_data

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
