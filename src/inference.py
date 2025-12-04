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
        
        # Use openpilot-inspired architecture with edge regression
        use_edge_head = bool(getattr(config, 'USE_EDGE_HEAD', True))
        self.model = RacingLineOptimizer(use_unet=True, use_edge_head=use_edge_head)
        logger.info(f"Using U-Net + Edge Regression (edge_head={use_edge_head})")
        
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
        
        # Edge coordinate smoothing (EMA)
        self.prev_edges = None
        self.edge_alpha = 0.7  # Higher = more current frame weight

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
        optimal_line, seg_map, confidence, edge_outputs = self.model(image_tensor, telemetry_tensor)
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
        
        # Extract and smooth edge coordinates
        edges_np = None
        edge_probs_np = None
        if edge_outputs is not None:
            edges_np = edge_outputs['edges'][0].cpu().numpy()  # [2, 33, 2]
            edge_probs_np = edge_outputs['edge_probs'][0].cpu().numpy()  # [2]
            
            # Temporal smoothing with EMA
            if self.prev_edges is not None:
                edges_np = self.edge_alpha * edges_np + (1 - self.edge_alpha) * self.prev_edges
            self.prev_edges = edges_np.copy()
        
        return {
            'optimal_line': optimal_line_np,
            'segmentation': seg_map_np,
            'confidence': confidence_np,
            'edges': edges_np,  # [2, 33, 2] or None
            'edge_probs': edge_probs_np,  # [2] or None
            'inference_time_ms': inference_time
        }
    
    def visualize_prediction(self, frame: np.ndarray, prediction: Dict) -> np.ndarray:
        """openpilot-style visualization with edge regression"""
        from .edge_constants import EdgeConstants
        
        result = frame.copy()
        h, w = frame.shape[:2]
        
        # Simple track overlay from segmentation
        seg_map = cv2.resize(
            prediction['segmentation'].astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Simple track mask (class 1 = track, not 0)
        track_mask = (seg_map == 1).astype(np.uint8)
        
        # Basic cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_CLOSE, kernel)
        
        # Green overlay
        track_overlay = np.zeros_like(frame)
        track_overlay[:, :, 1] = track_mask * 180  # Green channel
        result = cv2.addWeighted(result, 0.7, track_overlay, 0.3, 0)
        
        # Draw edge regression predictions (openpilot-style)
        show_edges = bool(getattr(config, 'SHOW_EDGE_LINES', True))
        if show_edges and prediction.get('edges') is not None:
            edges = prediction['edges']  # [2, 33, 2] (left/right, points, y_offset/height)
            edge_probs = prediction.get('edge_probs', np.array([1.0, 1.0]))  # [2]
            
            conf_threshold = float(getattr(config, 'EDGE_CONF_THRESHOLD', 0.3))
            center_x = w // 2
            
            for edge_idx in range(2):  # left=0, right=1
                if edge_probs[edge_idx] < conf_threshold:
                    continue  # Skip low-confidence edges
                
                edge_points = edges[edge_idx]  # [33, 2]
                poly_pts = []
                
                for point_idx in range(len(edge_points)):
                    lateral_norm, height_norm = edge_points[point_idx]
                    
                    # Convert normalized coords to pixel coords
                    lateral_pixels = int(lateral_norm * (w / 2.0))
                    x = center_x + lateral_pixels
                    y = int(h * height_norm)  # height_norm=0 -> top, height_norm=1 -> bottom
                    
                    # Bounds check
                    if 0 <= x < w and 0 <= y < h:
                        poly_pts.append((x, y))
                
                # Draw white polyline for track edge
                if len(poly_pts) > 5:
                    pts_array = np.array(poly_pts, dtype=np.int32)
                    cv2.polylines(result, [pts_array], False, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        
        # Add info overlay
        confidence = float(np.mean(prediction['confidence']))
        cv2.putText(result, f"Confidence: {confidence:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result, f"Inference: {prediction['inference_time_ms']:.1f}ms", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show edge confidence if available
        if prediction.get('edge_probs') is not None:
            edge_probs = prediction['edge_probs']
            cv2.putText(result, f"Edge L/R: {edge_probs[0]:.2f} / {edge_probs[1]:.2f}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
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
