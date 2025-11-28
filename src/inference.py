"""
Real-time inference engine for racing line analysis
"""
import torch
import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
from queue import Queue
from threading import Thread
import time
import logging

from .models import RacingLineOptimizer
from .video_processor import VideoProcessor
from .telemetry_processor import TelemetryProcessor, TelemetryPoint
from .config import config

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Real-time inference for racing line prediction"""
    
    def __init__(self, model_path: str):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model weights
        """
        self.device = config.get_device()
        self.model = RacingLineOptimizer()
        
        # Load model weights
        if model_path:
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}. Using untrained model.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Enable CPU optimizations
        if self.device.type == 'cpu':
            # Use optimized CPU inference
            torch.set_num_threads(16)
            logger.info("Running on CPU with 16 threads (consider GPU for 30-100x speedup)")
        
        # Racing line buffer for visualization (stores recent points)
        self.racing_line_buffer = []
        self.max_line_points = 30  # Show last 30 points (~1 second at 30fps)
        
        # Telemetry buffer for sequence processing
        self.telemetry_buffer: List[TelemetryPoint] = []
        self.buffer_size = 100  # Number of telemetry points to keep
        
    def preprocess_image(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for model input
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Preprocessed tensor
        """
        # Resize to smaller resolution for faster CPU processing
        resized = cv2.resize(frame, (320, 320))
        
        # Convert BGR to RGB and normalize
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def preprocess_telemetry(self, telemetry_points: List[TelemetryPoint]) -> torch.Tensor:
        """
        Preprocess telemetry sequence for model input
        
        Args:
            telemetry_points: List of telemetry points
            
        Returns:
            Preprocessed tensor [1, T, features]
        """
        if not telemetry_points:
            # Return dummy tensor if no telemetry
            return torch.zeros(1, self.buffer_size, 7).to(self.device)
        
        # Extract features
        features = []
        for point in telemetry_points[-self.buffer_size:]:
            features.append([
                point.speed / 300.0,  # Normalize to ~0-1
                point.throttle / 100.0,
                point.brake / 100.0,
                point.steering / 100.0,
                point.gear / 8.0,
                point.rpm / 10000.0,
                point.timestamp / 1000.0  # Normalize time
            ])
        
        # Pad if necessary
        while len(features) < self.buffer_size:
            features.insert(0, [0] * 7)
        
        tensor = torch.FloatTensor(features).unsqueeze(0)
        return tensor.to(self.device)
    
    @torch.no_grad()
    def predict(self, frame: np.ndarray, 
                telemetry: Optional[TelemetryPoint] = None) -> Dict:
        """
        Predict racing line for current frame
        
        Args:
            frame: Input frame
            telemetry: Current telemetry point
            
        Returns:
            Dictionary with predictions
        """
        # Add telemetry to buffer
        if telemetry:
            self.telemetry_buffer.append(telemetry)
            if len(self.telemetry_buffer) > self.buffer_size:
                self.telemetry_buffer.pop(0)
        
        # Preprocess inputs
        image_tensor = self.preprocess_image(frame)
        telemetry_tensor = self.preprocess_telemetry(self.telemetry_buffer)
        
        # Inference
        start_time = time.time()
        optimal_line, seg_map, confidence = self.model(image_tensor, telemetry_tensor)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Post-process results
        seg_map_np = seg_map[0].cpu().numpy()
        seg_map_np = np.argmax(seg_map_np, axis=0)
        
        confidence_np = confidence[0, 0].cpu().numpy()
        optimal_line_np = optimal_line[0].cpu().numpy()
        
        return {
            'optimal_line': optimal_line_np,
            'segmentation': seg_map_np,
            'confidence': confidence_np,
            'inference_time_ms': inference_time
        }
    
    def visualize_prediction(self, frame: np.ndarray, 
                           prediction: Dict) -> np.ndarray:
        """
        Visualize prediction on frame
        
        Args:
            frame: Input frame
            prediction: Prediction results
            
        Returns:
            Frame with visualization overlay
        """
        result = frame.copy()
        h, w = frame.shape[:2]
        
        # Resize segmentation to frame size
        seg_map = cv2.resize(
            prediction['segmentation'].astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Detect track edges/boundaries
        # Find contours for track boundaries (class 0 = track surface)
        track_mask = (seg_map == 0).astype(np.uint8) * 255
        edges = cv2.Canny(track_mask, 50, 150)
        
        # Dilate edges to make them more visible
        kernel = np.ones((3, 3), np.uint8)
        edges_thick = cv2.dilate(edges, kernel, iterations=2)
        
        # Create colored overlay
        overlay = np.zeros_like(frame)
        
        # Highlight track boundaries in bright cyan
        overlay[edges_thick > 0] = [255, 255, 0]  # Cyan for track edges
        
        # Show racing line area in semi-transparent green
        overlay[seg_map == 1] = [0, 255, 0]  # Green for optimal racing line zone
        
        # Show off-track areas in red
        overlay[seg_map == 2] = [0, 0, 255]  # Red for off-track
        
        # Blend overlay with original frame
        result = cv2.addWeighted(result, 0.75, overlay, 0.25, 0)
        
        # Add current optimal point to buffer
        opt_x, opt_y = prediction['optimal_line']
        point_x = int((opt_x + 1) / 2 * w)  # Convert from [-1, 1] to [0, w]
        point_y = int((opt_y + 1) / 2 * h)
        self.racing_line_buffer.append((point_x, point_y))
        
        # Keep buffer size limited
        if len(self.racing_line_buffer) > self.max_line_points:
            self.racing_line_buffer.pop(0)
        
        # Draw racing line path that driver should follow
        if len(self.racing_line_buffer) >= 2:
            # Draw thick background line first (for better visibility)
            for i in range(len(self.racing_line_buffer) - 1):
                pt1 = self.racing_line_buffer[i]
                pt2 = self.racing_line_buffer[i + 1]
                
                # Black outline for contrast
                cv2.line(result, pt1, pt2, (0, 0, 0), 12)
            
            # Draw colored racing line on top
            for i in range(len(self.racing_line_buffer) - 1):
                pt1 = self.racing_line_buffer[i]
                pt2 = self.racing_line_buffer[i + 1]
                
                # Fade effect: newer points are brighter
                alpha = (i + 1) / len(self.racing_line_buffer)
                thickness = int(4 + alpha * 4)  # 4-8 pixels thick
                
                # Bright purple/magenta line for the racing line to follow
                cv2.line(result, pt1, pt2, (255, 0, 255), thickness)
            
            # Draw a larger circle at the current target point with outline
            cv2.circle(result, self.racing_line_buffer[-1], 12, (0, 0, 0), -1)  # Black outline
            cv2.circle(result, self.racing_line_buffer[-1], 10, (255, 255, 0), -1)  # Yellow center
        
        # Add confidence indicator
        confidence = prediction['confidence'].mean()
        cv2.putText(result, f"Confidence: {confidence:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add inference time
        cv2.putText(result, f"Inference: {prediction['inference_time_ms']:.1f}ms", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return result


class RealtimeProcessor:
    """Process video and telemetry in real-time"""
    
    def __init__(self, model_path: str):
        """
        Initialize real-time processor
        
        Args:
            model_path: Path to model weights
        """
        self.engine = InferenceEngine(model_path)
        self.frame_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=10)
        self.telemetry_queue = Queue(maxsize=100)
        self.running = False
        
    def start(self):
        """Start processing threads"""
        self.running = True
        
        # Start inference thread
        self.inference_thread = Thread(target=self._inference_loop)
        self.inference_thread.start()
        
        logger.info("Real-time processor started")
    
    def stop(self):
        """Stop processing threads"""
        self.running = False
        if hasattr(self, 'inference_thread'):
            self.inference_thread.join()
        logger.info("Real-time processor stopped")
    
    def add_frame(self, frame: np.ndarray):
        """Add frame to processing queue"""
        if not self.frame_queue.full():
            self.frame_queue.put(frame)
    
    def add_telemetry(self, telemetry: TelemetryPoint):
        """Add telemetry point to queue"""
        if not self.telemetry_queue.full():
            self.telemetry_queue.put(telemetry)
    
    def get_result(self, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Get processed result
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (frame, prediction) or None
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except:
            return None
    
    def _inference_loop(self):
        """Inference loop running in separate thread"""
        while self.running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=0.1)
                
                # Get latest telemetry
                telemetry = None
                while not self.telemetry_queue.empty():
                    telemetry = self.telemetry_queue.get()
                
                # Run inference
                prediction = self.engine.predict(frame, telemetry)
                
                # Visualize
                result_frame = self.engine.visualize_prediction(frame, prediction)
                
                # Put result in output queue
                if not self.result_queue.full():
                    self.result_queue.put((result_frame, prediction))
                
            except Exception as e:
                if self.running:
                    logger.error(f"Error in inference loop: {e}")
                continue


class BatchProcessor:
    """Process entire video files in batch mode"""
    
    def __init__(self, model_path: str):
        """
        Initialize batch processor
        
        Args:
            model_path: Path to model weights
        """
        self.engine = InferenceEngine(model_path)
    
    def process_video(self, video_path: str, output_path: str,
                     telemetry_path: Optional[str] = None) -> Dict:
        """
        Process entire video file
        
        Args:
            video_path: Input video path
            output_path: Output video path
            telemetry_path: Optional telemetry CSV path
            
        Returns:
            Processing statistics
        """
        logger.info(f"Processing video: {video_path}")
        
        # Load telemetry if provided
        telemetry_data = []
        if telemetry_path:
            tel_processor = TelemetryProcessor()
            df = tel_processor.load_from_csv(telemetry_path)
            # Convert to TelemetryPoint objects
            for _, row in df.iterrows():
                telemetry_data.append(TelemetryPoint(
                    timestamp=row.get('timestamp', 0),
                    speed=row.get('speed', 0),
                    throttle=row.get('throttle', 0),
                    brake=row.get('brake', 0),
                    steering=row.get('steering', 0),
                    gear=row.get('gear', 1),
                    rpm=row.get('rpm', 0)
                ))
        
        # Process video
        stats = {
            'total_frames': 0,
            'avg_inference_time': 0,
            'predictions': []
        }
        
        with VideoProcessor(video_path) as vp:
            # Get frame size
            ret, first_frame = vp.cap.read()
            vp.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            h, w = first_frame.shape[:2]
            
            # Create video writer
            from .video_processor import VideoWriter
            with VideoWriter(output_path, vp.target_fps, (w, h)) as writer:
                inference_times = []
                
                for frame_num, frame in vp.get_frames():
                    # Get corresponding telemetry
                    telemetry = None
                    if telemetry_data and frame_num < len(telemetry_data):
                        telemetry = telemetry_data[frame_num]
                    
                    # Predict
                    prediction = self.engine.predict(frame, telemetry)
                    inference_times.append(prediction['inference_time_ms'])
                    
                    # Visualize and write
                    result_frame = self.engine.visualize_prediction(frame, prediction)
                    writer.write_frame(result_frame)
                    
                    stats['total_frames'] += 1
                    stats['predictions'].append(prediction)
                    
                    if frame_num % 30 == 0:
                        logger.info(f"Processed frame {frame_num}")
        
        stats['avg_inference_time'] = np.mean(inference_times)
        logger.info(f"Processing complete. Average inference time: {stats['avg_inference_time']:.2f}ms")
        
        return stats
