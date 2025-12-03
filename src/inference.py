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
            interpolation=cv2.INTER_LINEAR  # Smoother interpolation
        )
        
        # Apply bilateral filter to preserve edges while smoothing
        seg_map = cv2.bilateralFilter(seg_map, 5, 50, 50)
        
        # IMPROVED: Multi-class track detection with connected component analysis
        # Try both class 0 and non-zero classes to find track surface
        track_mask_class0 = (seg_map == 0).astype(np.uint8)
        track_mask_nonzero = (seg_map != 0).astype(np.uint8)
        
        # Use whichever has less coverage (track is usually smaller than background)
        if np.sum(track_mask_class0) < np.sum(track_mask_nonzero):
            track_mask = track_mask_class0
        else:
            track_mask = track_mask_nonzero
        
        # Apply morphological gradient to enhance boundaries
        kernel_enhance = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(track_mask, cv2.MORPH_GRADIENT, kernel_enhance)
        track_mask = cv2.add(track_mask, gradient)
        
        # IMPROVED: Multi-stage morphological operations for wider, cleaner coverage
        # Stage 1: Remove small noise first
        kernel_denoise = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_OPEN, kernel_denoise)
        
        # Stage 2: Close small gaps before expansion
        kernel_close_small = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_CLOSE, kernel_close_small, iterations=2)
        
        # Stage 3: Expand track area significantly with rectangular kernel
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
        track_mask = cv2.dilate(track_mask, kernel_dilate, iterations=3)
        
        # Stage 4: Final closing to fill any remaining gaps
        kernel_close_large = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_CLOSE, kernel_close_large, iterations=2)
        
        # IMPROVED: Connected component analysis to select main track region
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(track_mask, connectivity=8)
        
        if num_labels > 1:  # More than just background
            # Filter components by size, position, and aspect ratio
            valid_components = []
            for i in range(1, num_labels):  # Skip background (0)
                area = stats[i, cv2.CC_STAT_AREA]
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                cx, cy = int(centroids[i][0]), int(centroids[i][1])
                
                # Require substantial area (remove fragments)
                min_area = (w * h) * 0.05  # At least 5% of frame
                if area > min_area:
                    # Multi-factor scoring:
                    # 1. Area (larger is better)
                    area_score = area / (w * h)
                    # 2. Position (lower in frame is better for track)
                    position_score = (cy / h) ** 0.5
                    # 3. Width (wider components more likely to be track)
                    width_score = (width / w) ** 0.5
                    # 4. Horizontal centering (track usually centered)
                    center_score = 1.0 - abs((cx / w) - 0.5)
                    
                    total_score = area_score * 2.0 + position_score * 1.5 + width_score * 1.0 + center_score * 0.5
                    valid_components.append((i, total_score))
            
            # Select best component
            if valid_components:
                valid_components.sort(key=lambda x: x[1], reverse=True)
                main_component_id = valid_components[0][0]
                track_mask = (labels == main_component_id).astype(np.uint8)
            else:
                # Fallback: use largest component
                sizes = stats[1:, cv2.CC_STAT_AREA]
                if len(sizes) > 0:
                    largest_id = np.argmax(sizes) + 1
                    track_mask = (labels == largest_id).astype(np.uint8)
        
        # Final refinement with rectangular kernel for sharp, clean edges
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_CLOSE, kernel_smooth)
        
        # Optional: Apply slight gaussian blur to track edges for anti-aliasing
        track_mask_float = track_mask.astype(np.float32)
        track_mask_float = cv2.GaussianBlur(track_mask_float, (5, 5), 0)
        track_mask = (track_mask_float > 0.5).astype(np.uint8)
        
        # Enhanced visualization with gradient overlay for depth
        track_overlay = np.zeros_like(frame)
        # Create distance transform for gradient effect (brighter at center)
        dist_transform = cv2.distanceTransform(track_mask, cv2.DIST_L2, 5)
        dist_normalized = cv2.normalize(dist_transform, None, 0, 1, cv2.NORM_MINMAX)
        
        # Apply green color with intensity based on distance from edge
        for c in range(3):
            track_overlay[:, :, c] = track_mask * (90 if c == 0 else 200 if c == 1 else 90)
            # Add gradient intensity (brighter toward center)
            track_overlay[:, :, c] = np.minimum(255, track_overlay[:, :, c] * (0.7 + 0.3 * dist_normalized))
        
        result = cv2.addWeighted(result, 0.65, track_overlay.astype(np.uint8), 0.35, 0)
        
        # Draw clean white track boundaries with anti-aliasing
        edges = cv2.Canny(track_mask * 255, 30, 100)
        # Dilate edges slightly for visibility
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges = cv2.dilate(edges, kernel_edge, iterations=1)
        # Draw with slight blur for anti-aliasing
        edge_mask = edges > 0
        result[edge_mask] = [255, 255, 255]
        
        # Racing line visualization removed - focusing on track detection only
        
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
