"""
Real-time inference engine for road surface detection
"""
import torch
import cv2
import numpy as np
from typing import Dict, Optional
import time
import logging

from .models import RacingLineOptimizer
from .config import config

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Simplified inference for road surface detection only"""
    
    def __init__(self, model_path: str, device: str = None):
        """Initialize inference engine"""
        if device is None:
            self.device = config.get_device()
        else:
            self.device = torch.device(device)
        self.model = RacingLineOptimizer()
        
        # Load model weights
        if model_path:
            try:
                logger.info(f"📦 Loading model from: {model_path}")
                logger.info(f"🖥️  Using device: {self.device}")
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state = checkpoint['state_dict']
                    else:
                        state = checkpoint
                else:
                    state = checkpoint

                try:
                    self.model.load_state_dict(state)
                except RuntimeError:
                    fixed_state = {}
                    for k, v in state.items():
                        new_k = k.replace('module.', '') if k.startswith('module.') else k
                        fixed_state[new_k] = v
                    self.model.load_state_dict(fixed_state)
                logger.info(f"✓ Model loaded successfully")
            except FileNotFoundError:
                logger.warning(f"⚠️  Model file not found: {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
        
        # GPU compatibility check
        try:
            self.model.to(self.device)
            self.model.eval()
            
            if self.device.type == 'cuda':
                gpu_name = torch.cuda.get_device_name(0)
                if "RTX 50" in gpu_name or "RTX50" in gpu_name:
                    logger.warning(f"⚠️  {gpu_name} detected but not yet supported")
                    self.device = torch.device("cpu")
                    self.model.to(self.device)
                    self.model.eval()
                else:
                    test_img = torch.randn(1, 3, 320, 320).to(self.device)
                    test_tel = torch.randn(1, 100, 5).to(self.device)
                    with torch.no_grad():
                        _ = self.model(test_img, test_tel)
                    del test_img, test_tel
                    logger.info(f"✓ Running on GPU: {gpu_name}")
        except RuntimeError as e:
            if "no kernel image is available" in str(e) or "CUDA" in str(e):
                logger.warning(f"⚠️  GPU incompatible, falling back to CPU")
                self.device = torch.device("cpu")
                self.model.to(self.device)
                self.model.eval()
            else:
                raise
        
        if self.device.type == 'cpu':
            torch.set_num_threads(16)
            logger.info("✓ Running on CPU with 16-thread optimization")
        
        # Minimal telemetry buffer
        self.telemetry_buffer = []
        self.buffer_size = 100
        
        # Temporal smoothing for track masks (reduce flickering)
        self.prev_track_mask = None
        self.mask_history = []  # Store last N masks
        self.mask_history_size = 5

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
    
    def preprocess_telemetry(self, telemetry_points) -> torch.Tensor:
        """Preprocess telemetry sequence"""
        if not telemetry_points:
            return torch.zeros(1, self.buffer_size, 7).to(self.device)
        
        features = []
        for point in telemetry_points[-self.buffer_size:]:
            features.append([
                point.speed / 300.0,
                point.throttle / 100.0,
                point.brake / 100.0,
                point.steering / 100.0,
                point.gear / 8.0,
                point.rpm / 10000.0,
                point.timestamp / 1000.0
            ])
        
        while len(features) < self.buffer_size:
            features.insert(0, [0] * 7)
        
        tensor = torch.FloatTensor(features).unsqueeze(0)
        return tensor.to(self.device)
    
    @torch.no_grad()
    def predict(self, frame: np.ndarray, telemetry=None) -> Dict:
        """Predict road surface for current frame"""
        if telemetry:
            self.telemetry_buffer.append(telemetry)
            if len(self.telemetry_buffer) > self.buffer_size:
                self.telemetry_buffer.pop(0)
        
        # Preprocess
        image_tensor = self.preprocess_image(frame)
        telemetry_tensor = self.preprocess_telemetry(self.telemetry_buffer)
        
        # Inference
        start_time = time.time()
        optimal_line, seg_map, confidence = self.model(image_tensor, telemetry_tensor)
        inference_time = (time.time() - start_time) * 1000
        
        # Post-process
        seg_map_np = seg_map[0].cpu().numpy()
        seg_map_np = np.argmax(seg_map_np, axis=0)
        seg_map_np = cv2.medianBlur(seg_map_np.astype(np.uint8), 5)
        
        # Temporal smoothing to reduce flickering
        self.mask_history.append(seg_map_np.copy())
        if len(self.mask_history) > self.mask_history_size:
            self.mask_history.pop(0)
        
        # Average recent masks using voting
        if len(self.mask_history) >= 3:
            stacked = np.stack(self.mask_history, axis=0)
            # Use mode (most common value) for each pixel
            from scipy import stats
            seg_map_np, _ = stats.mode(stacked, axis=0, keepdims=False)
            seg_map_np = seg_map_np.astype(np.uint8)
        
        confidence_np = confidence[0, 0].cpu().numpy()
        optimal_line_np = optimal_line[0].cpu().numpy()
        
        return {
            'optimal_line': optimal_line_np,
            'segmentation': seg_map_np,
            'confidence': confidence_np,
            'inference_time_ms': inference_time
        }
    
    def visualize_prediction(self, frame: np.ndarray, prediction: Dict) -> np.ndarray:
        """Clean visualization with track surface, racing line, and speed coloring"""
        result = frame.copy()
        h, w = frame.shape[:2]
        
        # Resize segmentation to frame size
        seg_map = cv2.resize(
            prediction['segmentation'].astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )
        
        # IMPROVED: Multi-class track detection with connected component analysis
        # Try both class 0 and non-zero classes to find track surface
        track_mask_class0 = (seg_map == 0).astype(np.uint8)
        track_mask_nonzero = (seg_map != 0).astype(np.uint8)
        
        # Use whichever has less coverage (track is usually smaller than background)
        if np.sum(track_mask_class0) < np.sum(track_mask_nonzero):
            track_mask = track_mask_class0
        else:
            track_mask = track_mask_nonzero
        
        # IMPROVED: More aggressive morphological cleanup to handle fragmentation
        # Close small gaps first
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        # Remove small noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_OPEN, kernel_open)
        
        # IMPROVED: Connected component analysis to select main track region
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(track_mask, connectivity=8)
        
        if num_labels > 1:  # More than just background
            # Filter components by size and position
            valid_components = []
            for i in range(1, num_labels):  # Skip background (0)
                area = stats[i, cv2.CC_STAT_AREA]
                x, y = int(centroids[i][0]), int(centroids[i][1])
                
                # Require minimum area (remove tiny fragments)
                if area > 5000:
                    # Prefer components in lower 2/3 of image (where track typically is)
                    distance_score = 1.0 - (y / h) * 0.5  # Lower is better
                    valid_components.append((i, area * distance_score))
            
            # Select largest valid component
            if valid_components:
                valid_components.sort(key=lambda x: x[1], reverse=True)
                main_component_id = valid_components[0][0]
                track_mask = (labels == main_component_id).astype(np.uint8)
            else:
                # Fallback: use largest component regardless
                sizes = stats[1:, cv2.CC_STAT_AREA]
                if len(sizes) > 0:
                    largest_id = np.argmax(sizes) + 1
                    track_mask = (labels == largest_id).astype(np.uint8)
        
        # Final smoothing
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_CLOSE, kernel_smooth)
        
        # Show track surface with semi-transparent green
        track_overlay = np.zeros_like(frame)
        track_overlay[track_mask > 0] = [90, 200, 90]  # Green for track
        result = cv2.addWeighted(result, 0.7, track_overlay, 0.3, 0)
        
        # Draw white track boundaries
        edges = cv2.Canny(track_mask * 255, 50, 150)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        result[edges > 0] = [255, 255, 255]  # White edges
        
        # IMPROVED: Extract and draw racing line with corner-aware positioning
        try:
            # Find contours of track
            contours, _ = cv2.findContours(track_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Get largest contour (main track)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Only process if contour is large enough
                if cv2.contourArea(largest_contour) > 5000:
                    # IMPROVED: Compute racing line with corner detection and proper positioning
                    racing_line_pts = []
                    left_edges = []
                    right_edges = []
                    
                    for y in range(0, h, 4):  # Sample every 4 pixels for more detail
                        row_mask = track_mask[y, :]
                        track_xs = np.where(row_mask > 0)[0]
                        if len(track_xs) > 5:  # Valid track row
                            left_x = track_xs[0]
                            right_x = track_xs[-1]
                            left_edges.append([left_x, y])
                            right_edges.append([right_x, y])
                    
                    if len(left_edges) > 15 and len(right_edges) > 15:
                        left_edges = np.array(left_edges, dtype=np.float32)
                        right_edges = np.array(right_edges, dtype=np.float32)
                        
                        # Smooth edges first
                        from scipy.ndimage import gaussian_filter1d
                        left_edges[:, 0] = gaussian_filter1d(left_edges[:, 0], sigma=5)
                        right_edges[:, 0] = gaussian_filter1d(right_edges[:, 0], sigma=5)
                        
                        # IMPROVED: Detect corner direction by analyzing edge curvature
                        # Calculate track width at each point
                        track_widths = right_edges[:, 0] - left_edges[:, 0]
                        
                        # Calculate lateral movement of edges
                        window_size = 10
                        racing_line_pts = []
                        
                        for i in range(len(left_edges)):
                            # Look ahead to detect corner direction
                            start_idx = max(0, i - window_size)
                            end_idx = min(len(left_edges), i + window_size)
                            
                            # Analyze edge movement
                            if end_idx - start_idx > 5:
                                # Calculate edge displacement trends
                                left_trend = left_edges[end_idx-1, 0] - left_edges[start_idx, 0]
                                right_trend = right_edges[end_idx-1, 0] - right_edges[start_idx, 0]
                                
                                # Determine corner type:
                                # Left turn: right edge moves left more (negative), stay RIGHT
                                # Right turn: left edge moves right more (positive), stay LEFT
                                # Straight: balanced movement, use center
                                
                                corner_indicator = right_trend - left_trend
                                track_width = track_widths[i]
                                
                                # Bias factor: -1 (left) to +1 (right)
                                # Approaching left turn -> bias right (+1)
                                # Approaching right turn -> bias left (-1)
                                if abs(corner_indicator) > track_width * 0.15:  # Significant corner
                                    if corner_indicator < 0:  # Left turn approaching
                                        bias = 0.4  # Position 40% toward right (outside)
                                    else:  # Right turn approaching
                                        bias = -0.4  # Position 40% toward left (outside)
                                else:  # Straight or gentle curve
                                    bias = 0.0  # Center
                                
                                # Calculate racing line position
                                center_x = (left_edges[i, 0] + right_edges[i, 0]) / 2
                                offset = (track_width / 2) * bias
                                racing_x = center_x + offset
                                
                                racing_line_pts.append([racing_x, left_edges[i, 1]])
                            else:
                                # Fallback to center for edge cases
                                center_x = (left_edges[i, 0] + right_edges[i, 0]) / 2
                                racing_line_pts.append([center_x, left_edges[i, 1]])
                        
                        racing_line_pts = np.array(racing_line_pts, dtype=np.float32)
                        
                        # Multi-pass smoothing for stability
                        from scipy.ndimage import gaussian_filter1d
                        racing_line_pts[:, 0] = gaussian_filter1d(racing_line_pts[:, 0], sigma=5)
                        racing_line_pts[:, 1] = gaussian_filter1d(racing_line_pts[:, 1], sigma=2)
                        racing_line_pts = racing_line_pts.astype(np.int32)
                    
                        # Compute speed profile based on curvature
                        # More curved = slower (red), straighter = faster (green)
                        speeds = []
                        window_size = 8  # Look ahead/behind for curvature
                        for i in range(len(racing_line_pts)):
                            # Get window of points
                            start = max(0, i - window_size)
                            end = min(len(racing_line_pts), i + window_size + 1)
                            window = racing_line_pts[start:end]
                            
                            if len(window) < 3:
                                speeds.append(0.5)
                                continue
                            
                            # Calculate curvature from window variance
                            dx = np.diff(window[:, 0])
                            dy = np.diff(window[:, 1])
                            angles = np.arctan2(dy, dx)
                            
                            # Measure angle variation (higher = more curved)
                            if len(angles) > 1:
                                angle_std = np.std(angles)
                                curvature = min(angle_std * 5, 1.0)
                            else:
                                curvature = 0.0
                            
                            # Convert to speed (low curvature = high speed)
                            speed = 1.0 - curvature
                            speeds.append(speed)
                        
                        speeds = np.array(speeds)
                        # Smooth speeds for visual stability
                        speeds = gaussian_filter1d(speeds, sigma=3)
                        
                        # Draw racing line with speed coloring - thicker and with outline
                        for i in range(len(racing_line_pts) - 1):
                            pt1 = tuple(racing_line_pts[i])
                            pt2 = tuple(racing_line_pts[i + 1])
                            
                            # Color from red (slow) to yellow (medium) to green (fast)
                            speed = speeds[i]
                            if speed < 0.5:
                                # Red to Yellow
                                r = 255
                                g = int(255 * (speed * 2))
                                b = 0
                            else:
                                # Yellow to Green
                                r = int(255 * (2 - speed * 2))
                                g = 255
                                b = 0
                            
                            color = (b, g, r)  # BGR format
                            
                            # Draw black outline first for visibility
                            cv2.line(result, pt1, pt2, (0, 0, 0), 8, cv2.LINE_AA)
                            cv2.line(result, pt1, pt2, color, 5, cv2.LINE_AA)
                else:
                    # Contour too small
                    pass
        except Exception as e:
            # If racing line extraction fails, just continue without it
            logger.debug(f"Could not draw racing line: {e}")
        
        # Add minimal info overlay
        confidence = float(np.mean(prediction['confidence']))
        cv2.putText(result, f"Confidence: {confidence:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result, f"Inference: {prediction['inference_time_ms']:.1f}ms", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result


class BatchProcessor:
    """Batch video processor for GUI compatibility"""
    
    def __init__(self, model_path: str, device: str = None):
        """Initialize batch processor"""
        self.engine = InferenceEngine(model_path, device)
    
    def process_video(self, video_path: str, output_path: str, 
                     progress_callback=None, stop_callback=None) -> Dict:
        """Process video file with inference"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        stats = {
            'total_frames': 0,
            'avg_inference_time': 0,
            'predictions': []
        }
        
        inference_times = []
        frame_num = 0
        start_time = time.time()
        
        try:
            while True:
                # Check stop callback
                if stop_callback and stop_callback():
                    logger.info("Processing stopped by user")
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run inference
                prediction = self.engine.predict(frame)
                inference_times.append(prediction['inference_time_ms'])
                
                # Visualize
                result_frame = self.engine.visualize_prediction(frame, prediction)
                
                # Write frame
                out.write(result_frame)
                
                frame_num += 1
                stats['total_frames'] = frame_num
                stats['predictions'].append(prediction)
                
                # Progress callback
                if progress_callback and frame_num % 5 == 0:
                    elapsed = time.time() - start_time
                    processing_fps = frame_num / elapsed if elapsed > 0 else 0
                    progress_callback(frame_num, total_frames, processing_fps, 
                                    prediction['inference_time_ms'])
                
                if frame_num % 30 == 0:
                    elapsed = time.time() - start_time
                    processing_fps = frame_num / elapsed if elapsed > 0 else 0
                    logger.info(f"Processed {frame_num}/{total_frames} ({processing_fps:.1f} FPS)")
        
        finally:
            cap.release()
            out.release()
        
        if inference_times:
            stats['avg_inference_time'] = np.mean(inference_times)
        
        elapsed = time.time() - start_time
        logger.info(f"Processing complete: {frame_num} frames in {elapsed:.1f}s")
        logger.info(f"Average inference: {stats['avg_inference_time']:.2f}ms")
        
        return stats
