"""
Video processing module for extracting frames and track features
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Process onboard racing video for analysis"""
    
    def __init__(self, video_path: str, target_fps: int = 30):
        """
        Initialize video processor
        
        Args:
            video_path: Path to the video file
            target_fps: Target frames per second for processing
        """
        self.video_path = Path(video_path)
        self.target_fps = target_fps
        self.cap = None
        self.original_fps = None
        self.frame_skip = 1
        
    def __enter__(self):
        """Context manager entry"""
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {self.video_path}")
        
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_skip = max(1, int(self.original_fps / self.target_fps))
        
        logger.info(f"Video opened: {self.video_path}")
        logger.info(f"Original FPS: {self.original_fps}, Target FPS: {self.target_fps}")
        logger.info(f"Frame skip: {self.frame_skip}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.cap:
            self.cap.release()
    
    def get_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generate frames from video
        
        Yields:
            Tuple of (frame_number, frame_image)
        """
        frame_count = 0
        processed_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_count % self.frame_skip == 0:
                yield processed_count, frame
                processed_count += 1
            
            frame_count += 1
    
    def extract_track_features(self, frame: np.ndarray) -> dict:
        """
        Extract track features from a frame
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary containing track features
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Edge detection for track boundaries
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find track boundaries using edge detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                                minLineLength=100, maxLineGap=10)
        
        # Extract road region (assuming lower 2/3 of frame)
        h, w = frame.shape[:2]
        roi = frame[int(h/3):, :]
        
        return {
            'edges': edges,
            'lines': lines if lines is not None else [],
            'hsv': hsv,
            'roi': roi,
            'frame_shape': frame.shape
        }
    
    def preprocess_frame(self, frame: np.ndarray, 
                        target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """
        Preprocess frame for model input
        
        Args:
            frame: Input frame
            target_size: Target size for model input
            
        Returns:
            Preprocessed frame
        """
        # Resize
        resized = cv2.resize(frame, target_size)
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
        
        return rgb
    
    def draw_racing_line(self, frame: np.ndarray, 
                        points: List[Tuple[int, int]], 
                        color: Tuple[int, int, int] = (0, 255, 0),
                        thickness: int = 3) -> np.ndarray:
        """
        Draw racing line on frame
        
        Args:
            frame: Input frame
            points: List of (x, y) points forming the racing line
            color: Line color (BGR)
            thickness: Line thickness
            
        Returns:
            Frame with racing line drawn
        """
        result = frame.copy()
        
        if len(points) > 1:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(result, [pts], False, color, thickness)
            
            # Draw optimal point markers
            for i, pt in enumerate(points):
                if i % 5 == 0:  # Draw every 5th point
                    cv2.circle(result, pt, 5, color, -1)
        
        return result
    
    @staticmethod
    def overlay_telemetry(frame: np.ndarray, 
                         telemetry: dict,
                         position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """
        Overlay telemetry data on frame
        
        Args:
            frame: Input frame
            telemetry: Dictionary with telemetry data
            position: Starting position for text
            
        Returns:
            Frame with telemetry overlay
        """
        result = frame.copy()
        x, y = position
        line_height = 30
        
        # Create semi-transparent background
        overlay = result.copy()
        cv2.rectangle(overlay, (x-5, y-25), (x+300, y+line_height*len(telemetry)+10), 
                     (0, 0, 0), -1)
        result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
        
        # Draw telemetry text
        for i, (key, value) in enumerate(telemetry.items()):
            text = f"{key}: {value}"
            cv2.putText(result, text, (x, y + i*line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result


class VideoWriter:
    """Write processed video with overlays"""
    
    def __init__(self, output_path: str, fps: int, frame_size: Tuple[int, int]):
        """
        Initialize video writer
        
        Args:
            output_path: Output video path
            fps: Frames per second
            frame_size: Frame size (width, height)
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.frame_size = frame_size
        self.writer = None
        
    def __enter__(self):
        """Context manager entry"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(str(self.output_path), fourcc, 
                                     self.fps, self.frame_size)
        
        if not self.writer.isOpened():
            raise ValueError(f"Failed to create video writer: {self.output_path}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.writer:
            self.writer.release()
    
    def write_frame(self, frame: np.ndarray):
        """Write a frame to video"""
        self.writer.write(frame)
