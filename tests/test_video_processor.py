"""
Unit tests for video processing module
"""
import unittest
import numpy as np
import cv2
from pathlib import Path
import tempfile

from src.video_processor import VideoProcessor, VideoWriter


class TestVideoProcessor(unittest.TestCase):
    """Test VideoProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a simple test video
        self.temp_dir = tempfile.mkdtemp()
        self.test_video_path = Path(self.temp_dir) / "test_video.mp4"
        
        # Create test video with 10 frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.test_video_path), fourcc, 30, (640, 480))
        
        for i in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :, i % 3] = 255  # Different color each frame
            out.write(frame)
        
        out.release()
    
    def test_video_opening(self):
        """Test video file opening"""
        with VideoProcessor(str(self.test_video_path), target_fps=30) as vp:
            self.assertIsNotNone(vp.cap)
            self.assertTrue(vp.cap.isOpened())
    
    def test_frame_extraction(self):
        """Test frame extraction"""
        with VideoProcessor(str(self.test_video_path), target_fps=30) as vp:
            frames = list(vp.get_frames())
            self.assertGreater(len(frames), 0)
            
            for frame_num, frame in frames:
                self.assertEqual(frame.shape[:2], (480, 640))
    
    def test_preprocess_frame(self):
        """Test frame preprocessing"""
        with VideoProcessor(str(self.test_video_path)) as vp:
            frame_num, frame = next(vp.get_frames())
            preprocessed = vp.preprocess_frame(frame, target_size=(320, 320))
            
            self.assertEqual(preprocessed.shape, (320, 320, 3))
            self.assertTrue(preprocessed.max() <= 1.0)
            self.assertTrue(preprocessed.min() >= 0.0)


if __name__ == '__main__':
    unittest.main()
