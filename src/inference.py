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
import os
import glob
import matplotlib.pyplot as plt

from .models import RacingLineOptimizer
from .video_processor import VideoProcessor
from .telemetry_processor import TelemetryProcessor, TelemetryPoint
from .config import config
from .racing_line import RacingLineEstimator, plot_debug_track
from .geometry import apply_homography, invert_homography

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Real-time inference for racing line prediction"""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model weights
            device: Device to use ('cuda' or 'cpu'). If None, uses config.get_device()
        """
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
                # Load checkpoint robustly: support full checkpoint dicts, plain state_dicts,
                # and checkpoints saved from DataParallel (with 'module.' prefixes).
                checkpoint = torch.load(model_path, map_location=self.device)
                # Determine if checkpoint is a full dict with named keys
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state = checkpoint['state_dict']
                    else:
                        # Might already be a state_dict
                        state = checkpoint
                else:
                    # checkpoint is likely a state_dict already
                    state = checkpoint

                # Try loading state dict; if keys are prefixed with 'module.' (DataParallel), strip it.
                try:
                    self.model.load_state_dict(state)
                except RuntimeError:
                    fixed_state = {}
                    for k, v in state.items():
                        new_k = k.replace('module.', '') if k.startswith('module.') else k
                        fixed_state[new_k] = v
                    self.model.load_state_dict(fixed_state)
                logger.info(f"✓ Model loaded successfully from {model_path}")
            except FileNotFoundError:
                logger.warning(
                    f"⚠️  Model file not found: {model_path}\n"
                    "   Using untrained model (predictions will be random).\n"
                    "   To train the model:\n"
                    "   1. Open the 'Train Model' tab in the GUI\n"
                    "   2. Click 'Generate Training Data' and select a racing video\n"
                    "   3. Click 'Start Training' (takes 30-60 minutes)\n"
                )
            except Exception as e:
                logger.warning(f"Could not load model: {e}. Using untrained model.")
        
        # Try to use GPU, fallback to CPU if incompatible
        try:
            self.model.to(self.device)
            self.model.eval()
            
            # Test if GPU actually works with a simple forward pass
            if self.device.type == 'cuda':
                gpu_name = torch.cuda.get_device_name(0)
                # Check for RTX 50 series - skip test as we know it won't work
                if "RTX 50" in gpu_name or "RTX50" in gpu_name:
                    logger.warning(f"⚠️  {gpu_name} detected but not yet supported by PyTorch")
                    logger.warning("   Falling back to CPU mode (GPU will be 10-20x faster once supported)")
                    self.device = torch.device("cpu")
                    self.model.to(self.device)
                    self.model.eval()
                else:
                    # Test GPU with forward pass
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
        
        # Enable CPU optimizations
        if self.device.type == 'cpu':
            # Use optimized CPU inference with all available threads
            torch.set_num_threads(16)
            logger.info("✓ Running on CPU with 16-thread optimization")
        
        # Racing line buffer for visualization (stores recent points)
        self.racing_line_buffer = []
        self.max_line_points = 60  # Show last 60 points (~2 seconds at 30fps for better lookahead)
        
        # Telemetry buffer for sequence processing
        self.telemetry_buffer: List[TelemetryPoint] = []
        self.buffer_size = 100  # Number of telemetry points to keep

        # Optional homography for BEV/world mapping (image->world)
        self.H = None
        self.H_inv = None
        if getattr(config, 'HOMOGRAPHY_PATH', ''):
            try:
                if os.path.isfile(config.HOMOGRAPHY_PATH):
                    self.H = np.load(config.HOMOGRAPHY_PATH)
                    self.H_inv = np.linalg.inv(self.H)
                    logger.info(f"Loaded homography for live BEV: {config.HOMOGRAPHY_PATH}")
            except Exception as e:
                logger.warning(f"Failed loading homography at {config.HOMOGRAPHY_PATH}: {e}")

        # Live optimized spline cache (pixel coords) updated every N frames
        self._live_pts_px = None
        self._live_speed = None
        self._frame_idx = 0
        self._opt_every = max(3, int(getattr(config, 'LIVE_OPTIMIZE_EVERY_N', 8)))
        
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
        

        # Optionally save segmentation debug maps (controlled via config to avoid uncontrolled growth)
        segmentation_dir = os.path.join(config.LOGS_DIR, "segmentation_maps")
        # Use a per-engine counter to control save frequency
        if not hasattr(self, '_predict_count'):
            self._predict_count = 0
        self._predict_count += 1

        if getattr(config, 'SAVE_SEGMENTATION_MAPS', False):
            if self._predict_count % max(1, config.SEGMENTATION_SAVE_FREQ) == 0:
                os.makedirs(segmentation_dir, exist_ok=True)
                seg_map_debug_path = os.path.join(segmentation_dir, f"segmentation_map_debug_{int(time.time()*1000)}.png")
                cv2.imwrite(seg_map_debug_path, seg_map_np.astype(np.uint8) * 85)
                logger.info(f"Saved raw segmentation map for validation: {seg_map_debug_path}")

                # Prune old segmentation files if we exceed the configured maximum
                try:
                    max_files = int(getattr(config, 'SEGMENTATION_MAX_FILES', 0))
                    if max_files > 0:
                        files = sorted(glob.glob(os.path.join(segmentation_dir, '*.png')), key=os.path.getmtime)
                        while len(files) > max_files:
                            try:
                                os.remove(files[0])
                                logger.debug(f"Pruned old segmentation map: {files[0]}")
                                files.pop(0)
                            except Exception:
                                # If a file was removed by another process or locked, skip it
                                files = sorted(glob.glob(os.path.join(segmentation_dir, '*.png')), key=os.path.getmtime)
                except Exception as e:
                    logger.warning(f"Failed to prune segmentation debug files: {e}")
        
        # Add debugging code to log class mapping
        logger.info("Class mapping validation:")
        logger.info("Class 0: Track surface")
        logger.info("Class 1: Racing line")
        logger.info("Class 2: Off-track")
        
        # Add post-processing step to smooth segmentation map
        seg_map_np = cv2.medianBlur(seg_map_np.astype(np.uint8), 5)  # Apply median blur for noise reduction
        logger.info("Applied median blur to segmentation map for improved confidence.")
        
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
        # Refine visualization logic to clear previous paths dynamically
        result = frame.copy()  # Reset visualization frame before drawing
        logger.info("Cleared previous paths dynamically for racing line visualization.")
        
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

        # Derive a centerline from segmentation and overlay it for alignment
        try:
            rle = RacingLineEstimator()
            # seg_map is already resized to frame size (w, h)
            centerline_pts = rle.extract_centerline_from_segmentation(seg_map.astype(np.int32))
            if centerline_pts is not None and centerline_pts.shape[0] >= 8:
                # Smooth with a light spline fit and render
                try:
                    from scipy.interpolate import splev
                    tck, _ = rle.fit_spline(centerline_pts, smoothing=2.0)
                    x_s, y_s = splev(np.linspace(0, 1, min(800, centerline_pts.shape[0] * 4)), tck)
                    pts_px = np.vstack([x_s, y_s]).T
                except Exception:
                    pts_px = centerline_pts

                # Keep in-bounds points only
                h, w = result.shape[:2]
                pts_px = np.asarray(pts_px, dtype=np.float32)
                inside = ((pts_px[:, 0] >= 0) & (pts_px[:, 0] < w) & (pts_px[:, 1] >= 0) & (pts_px[:, 1] < h))
                pts_px = pts_px[inside]
                if pts_px.shape[0] > 1:
                    cv2.polylines(result, [np.round(pts_px).astype(np.int32).reshape((-1, 1, 2))], isClosed=False, color=(0, 255, 255), thickness=2)

                # Optionally compute/update a lap-time-optimized line (BEV/world if homography exists)
                self._frame_idx += 1
                if self._frame_idx % self._opt_every == 0:
                    try:
                        cl_for_opt = centerline_pts
                        if self.H is not None:
                            # Map pixels->world
                            world_pts = apply_homography(self.H, cl_for_opt)
                            cl_space = world_pts
                        else:
                            cl_space = cl_for_opt  # pixel space fallback

                        # Optimize with modest iteration budget for live
                        try:
                            spline_opt, curvature_opt, arc_opt, speed_opt, _ = rle.optimize_racing_line(cl_space, n_control=10, maxiter=80)
                            # Map back to pixels if world space used
                            opt_xy = np.vstack([spline_opt[0], spline_opt[1]]).T
                            if self.H is not None:
                                px = apply_homography(self.H_inv, opt_xy)
                            else:
                                px = opt_xy
                            self._live_pts_px = px.astype(np.float32)
                            self._live_speed = speed_opt
                        except Exception:
                            # If optimization fails, cache smoothed centerline
                            self._live_pts_px = pts_px.astype(np.float32)
                            self._live_speed = np.linspace(1.0, 1.0, self._live_pts_px.shape[0])
                    except Exception:
                        pass

                # Overlay cached optimized line if present
                try:
                    if self._live_pts_px is not None and self._live_pts_px.shape[0] > 1:
                        px = self._live_pts_px
                        # In-bounds filter
                        h, w = result.shape[:2]
                        inside2 = ((px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h))
                        px = px[inside2]
                        if px.shape[0] > 1:
                            # color by speed if available
                            spd = self._live_speed if self._live_speed is not None else np.linspace(1.0, 1.0, px.shape[0])
                            result = rle.overlay_racing_line(result, (px[:, 0], px[:, 1]), spd)
                except Exception:
                    pass
        except Exception:
            pass
        
        # Get optimal point from model prediction
        opt_x, opt_y = prediction['optimal_line']
        
        # Handle NaN values
        if np.isnan(opt_x) or np.isnan(opt_y):
            opt_x, opt_y = 0.0, 0.0
        
        # (Removed debug random jitter. Use raw model outputs for smoother, consistent predictions.)
        
        # Clamp to [-1, 1] range
        opt_x = max(-1.0, min(1.0, opt_x))
        opt_y = max(-1.0, min(1.0, opt_y))
        
        # Convert from [-1, 1] to [0, w] and [0, h]
        point_x = int((opt_x + 1) / 2 * w)  # Convert from [-1, 1] to [0, w]
        point_y = int((opt_y + 1) / 2 * h)  # Convert from [-1, 1] to [0, h]
        
        # Clamp to frame boundaries
        point_x = max(0, min(w - 1, point_x))
        point_y = max(0, min(h - 1, point_y))
        
        # Only add valid, on-track points with high confidence to buffer
        confidence_threshold = 0.5  # Lowered threshold for debugging
        track_mask = (prediction['segmentation'] == 0)
        conf_val = float(prediction['confidence']) if np.isscalar(prediction['confidence']) else float(np.mean(prediction['confidence']))
        mask_h, mask_w = track_mask.shape
        scaled_x = int(point_x * mask_w / w)
        scaled_y = int(point_y * mask_h / h)
        scaled_x = max(0, min(mask_w - 1, scaled_x))
        scaled_y = max(0, min(mask_h - 1, scaled_y))
        mask_value = int(track_mask[scaled_y, scaled_x])
        logger.info(f"Debug: Confidence={conf_val:.2f}, Mask Value={mask_value}, Scaled Point=({scaled_x},{scaled_y}), Frame Point=({point_x},{point_y})")
        if conf_val > confidence_threshold and mask_value == 1:
            # Only append if sufficiently far from last point to avoid tight clustering
            min_pixel_dist = 12
            if not self.racing_line_buffer:
                self.racing_line_buffer.append((float(point_x), float(point_y)))
            else:
                last = self.racing_line_buffer[-1]
                dist = np.hypot(point_x - last[0], point_y - last[1])
                if dist >= min_pixel_dist:
                    self.racing_line_buffer.append((float(point_x), float(point_y)))
                else:
                    # slightly nudge last point towards the new predicted point (small smoothing)
                    lr = 0.5
                    newx = lr * point_x + (1 - lr) * last[0]
                    newy = lr * point_y + (1 - lr) * last[1]
                    self.racing_line_buffer[-1] = (float(newx), float(newy))

        # Limit buffer size for smoother visualization
        max_buffer_size = 20
        if len(self.racing_line_buffer) > max_buffer_size:
            self.racing_line_buffer = self.racing_line_buffer[-max_buffer_size:]

        # Apply exponential smoothing to reduce jitter while preserving responsiveness
        smoothed_buffer = []
        alpha = 0.4
        for i, pt in enumerate(self.racing_line_buffer):
            if i == 0:
                smoothed_buffer.append((float(pt[0]), float(pt[1])))
            else:
                prev = smoothed_buffer[-1]
                sx = alpha * float(pt[0]) + (1 - alpha) * prev[0]
                sy = alpha * float(pt[1]) + (1 - alpha) * prev[1]
                smoothed_buffer.append((sx, sy))

        # Spline interpolation for even smoother curve (buffer-based fallback)
        try:
            from scipy.interpolate import CubicSpline
            if len(smoothed_buffer) > 3:
                xs = [pt[0] for pt in smoothed_buffer]
                ys = [pt[1] for pt in smoothed_buffer]
                t = np.arange(len(smoothed_buffer))
                cs_x = CubicSpline(t, xs)
                cs_y = CubicSpline(t, ys)
                t_new = np.linspace(0, len(smoothed_buffer)-1, 50)
                interp_pts = np.array([(int(cs_x(ti)), int(cs_y(ti))) for ti in t_new], np.int32).reshape((-1, 1, 2))
                cv2.polylines(result, [interp_pts], isClosed=False, color=(0, 255, 0), thickness=4)
            else:
                # Fallback to polyline if not enough points
                if len(smoothed_buffer) > 1:
                    pts = np.array(smoothed_buffer, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(result, [pts], isClosed=False, color=(0, 255, 0), thickness=3)
        except ImportError:
            # Fallback to polyline if scipy not available
            if len(smoothed_buffer) > 1:
                pts = np.array(smoothed_buffer, np.int32).reshape((-1, 1, 2))
                cv2.polylines(result, [pts], isClosed=False, color=(0, 255, 0), thickness=3)
        
        # Draw direction indicator at the most recent point
        if len(self.racing_line_buffer) > 0:
            last_opt = self.racing_line_buffer[-1]
            last_x = int(last_opt[0])
            last_y = int(last_opt[1])
            last_x = max(0, min(w - 1, last_x))
            last_y = max(0, min(h - 1, last_y))
            cv2.circle(result, (last_x, last_y), 25, (255, 255, 255), -1)  # White outline
            cv2.circle(result, (last_x, last_y), 20, (0, 0, 0), -1)  # Black fill
            cv2.circle(result, (last_x, last_y), 10, (255, 0, 255), -1)  # Purple center
        
        # Add confidence indicator
        confidence = prediction['confidence'].mean()
        cv2.putText(result, f"Confidence: {confidence:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add inference time
        cv2.putText(result, f"Inference: {prediction['inference_time_ms']:.1f}ms", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Debug: Show predicted coordinates and buffer size
        cv2.putText(result, f"Line Point: ({opt_x:.2f}, {opt_y:.2f}) -> ({point_x}, {point_y})", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(result, f"Buffer: {len(self.racing_line_buffer)} points", 
                   (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return result


class RealtimeProcessor:
    """Process video and telemetry in real-time

    Optional quick-fix: pass a precomputed spline (pixel coordinates) to overlay
    the ideal racing line in live preview. This lets users display the full
    lap-optimized line even when the segmentation model does not produce a
    visible racing-line class.
    """

    def __init__(self, model_path: str, precomputed_spline_px: Optional[np.ndarray] = None):
        """
        Initialize real-time processor

        Args:
            model_path: Path to model weights
            precomputed_spline_px: Optional Nx2 array of pixel coordinates to overlay each frame
        """
        self.engine = InferenceEngine(model_path)
        self.precomputed_spline_px = precomputed_spline_px
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

                # Overlay precomputed spline if provided (quick live overlay option)
                if self.precomputed_spline_px is not None:
                    try:
                        pts_arr = np.asarray(self.precomputed_spline_px)
                        if pts_arr.ndim == 2 and pts_arr.shape[1] == 2 and pts_arr.shape[0] > 1:
                            # Create a simple speed array placeholder so the overlay function can color by speed
                            speed_placeholder = np.linspace(1.0, 1.0, pts_arr.shape[0])
                            rle = RacingLineEstimator()
                            result_frame = rle.overlay_racing_line(result_frame, (pts_arr[:, 0], pts_arr[:, 1]), speed_placeholder)
                    except Exception as e:
                        logger.warning(f"Could not overlay precomputed spline: {e}")
                
                # Put result in output queue
                if not self.result_queue.full():
                    self.result_queue.put((result_frame, prediction))
                
            except Exception as e:
                if self.running:
                    logger.error(f"Error in inference loop: {e}")
                continue


class BatchProcessor:
    """Process entire video files in batch mode"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize batch processor
        
        Args:
            model_path: Path to model weights
            device: Device to use ('auto', 'cuda', or 'cpu')
        """
        # Determine actual device
        if device == 'auto':
            import torch
            actual_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            actual_device = device
            
        self.device = actual_device
        self.engine = InferenceEngine(model_path, device=actual_device)
    
    def process_video(self, video_path: str, output_path: str,
                     telemetry_path: Optional[str] = None,
                     centerline_path: Optional[str] = None,
                     homography_path: Optional[str] = None,
                     save_racing_csv: Optional[str] = None,
                     save_fig_prefix: Optional[str] = None,
                     optimize_racing_line: bool = False,
                     optimize_maxiter: int = 200,
                     stop_callback: Optional[callable] = None,
                     progress_callback: Optional[callable] = None) -> Dict:
        """
        Process entire video file
        
        Args:
            video_path: Input video path
            output_path: Output video path
            telemetry_path: Optional telemetry CSV path
            stop_callback: Optional callback function that returns True to stop processing
            progress_callback: Optional callback(frame_num, total_frames, fps, inference_time)
            
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
            # Get frame size and total frames
            ret, first_frame = vp.cap.read()
            vp.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            h, w = first_frame.shape[:2]
            total_frames = int(vp.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create video writer
            from .video_processor import VideoWriter
            import time as time_module
            start_time = time_module.time()
            
            with VideoWriter(output_path, vp.target_fps, (w, h)) as writer:
                inference_times = []
                # Prepare optional racing line computation if a centerline CSV was provided
                rle = None
                pts_px = None
                speed = None
                was_optimized = False
                if centerline_path:
                    # load centerline CSV (supports header or raw two-column CSV)
                    try:
                        centerline = np.loadtxt(centerline_path, delimiter=',', skiprows=1)
                    except Exception:
                        centerline = np.loadtxt(centerline_path, delimiter=',')

                    # Attempt to load homography (image->world) if provided
                    H = None
                    if homography_path:
                        try:
                            H = np.load(homography_path)
                            logger.info(f"Loaded homography from {homography_path}")
                        except Exception as e:
                            logger.warning(f"Could not load homography: {e}")

                    # Compute spline, curvature and speed
                    rle = RacingLineEstimator()
                    tck, _ = rle.fit_spline(centerline)
                    spline_world, curvature, arc = rle.compute_curvature_and_arc_length(tck)
                    speed = rle.compute_speed_profile(curvature, arc)

                    # Optionally optimize the racing line (lap-time driven) from the centerline
                    if optimize_racing_line:
                        try:
                            logger.info("Optimizing racing line for lap time (this may take a while)")
                            spline_opt, curvature_opt, arc_opt, speed_opt, opt_res = rle.optimize_racing_line(centerline, n_control=12, maxiter=optimize_maxiter)
                            # Replace world spline + speed with optimized one for overlay & export
                            spline_world = spline_opt
                            curvature = curvature_opt
                            arc = arc_opt
                            speed = speed_opt
                            # Optionally save optimization result object
                            logger.info(f"Racing line optimization finished: success={opt_res.success}, lap_time={opt_res.fun:.3f}")
                            # remap optimized world spline to image pixels if homography available
                            if H is not None:
                                try:
                                    pts_world = np.vstack([spline_world[0], spline_world[1]]).T
                                    pts = np.hstack([pts_world, np.ones((len(pts_world), 1))])
                                    pts_img = (np.linalg.inv(H) @ pts.T).T
                                    pts_px = pts_img[:, :2] / pts_img[:, 2:3]
                                except Exception as e:
                                    logger.warning(f"Could not remap optimized world spline to pixels: {e}")
                            else:
                                pts_px = np.vstack([spline_world[0], spline_world[1]]).T
                            was_optimized = True
                        except Exception as e:
                            logger.warning(f"Racing line optimization failed: {e}")

                    # Map world->image using H inverse if homography is provided
                    if H is not None:
                        try:
                            H_inv = np.linalg.inv(H)
                            pts_world = np.vstack([spline_world[0], spline_world[1]]).T
                            pts = np.hstack([pts_world, np.ones((len(pts_world), 1))])
                            pts_img = (H_inv @ pts.T).T
                            pts_px = pts_img[:, :2] / pts_img[:, 2:3]
                            logger.info(f"Mapped {len(pts_px)} spline points world->image; x range {pts_px[:,0].min():.1f}..{pts_px[:,0].max():.1f}")
                        except Exception as e:
                            logger.warning(f"Error mapping world->image: {e}")
                            pts_px = np.vstack([spline_world[0], spline_world[1]]).T
                    else:
                        # assume the centerline is already in pixel coordinates
                        pts_px = np.vstack([spline_world[0], spline_world[1]]).T
                        logger.info(f"Loaded centerline into pixel coords; {len(pts_px)} points; x range {pts_px[:,0].min():.1f}..{pts_px[:,0].max():.1f}")

                    # Save CSV and plots if requested
                    if save_racing_csv:
                        try:
                            header = 's_m,x_m,y_m,curvature,speed_mps'
                            data = np.vstack([arc, spline_world[0], spline_world[1], curvature, speed]).T
                            np.savetxt(save_racing_csv, data, delimiter=',', header=header, comments='')
                            logger.info(f"Saved racing line samples to {save_racing_csv}")
                        except Exception as e:
                            logger.warning(f"Could not save racing CSV: {e}")

                    if save_fig_prefix:
                        try:
                            # create and save static debug plot (avoid interactive display)
                            fig, axs = plt.subplots(3, 1, figsize=(8, 14))
                            axs[0].plot(spline_world[0], spline_world[1])
                            axs[0].set_title('Racing Line Spline')
                            axs[1].plot(curvature)
                            axs[1].set_title('Curvature')
                            axs[2].plot(speed)
                            axs[2].set_title('Speed Profile')
                            plt.tight_layout()
                            fig_path = f"{save_fig_prefix}_line_speed.png"
                            fig.savefig(fig_path, dpi=200)
                            plt.close(fig)
                            logger.info(f"Saved racing line debug plot to {fig_path}")
                        except Exception as e:
                            logger.warning(f"Could not produce debug plots: {e}")
                # If user did not supply a centerline CSV, attempt automatic extraction
                if centerline_path is None and first_frame is not None:
                    try:
                        logger.info("Attempting automatic centerline extraction from model segmentation (first frame)")
                        pred0 = self.engine.predict(first_frame, None)
                        seg_small = pred0['segmentation']
                        seg_resized = cv2.resize(seg_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                        rle = RacingLineEstimator()
                        pts_auto_px = rle.extract_centerline_from_segmentation(seg_resized)
                        if pts_auto_px is not None and len(pts_auto_px) > 3:
                            pts_px = pts_auto_px
                            logger.info(f"Auto-extracted centerline with {len(pts_px)} points from segmentation")
                            try:
                                logger.info(f"Auto-extract x range: {pts_px[:,0].min():.1f}..{pts_px[:,0].max():.1f}")
                            except Exception:
                                pass

                            # If a homography (image->world) was provided, try converting pixels -> world coords
                            if homography_path:
                                try:
                                    H_img_to_world = np.load(homography_path)
                                    pts = np.hstack([pts_px, np.ones((len(pts_px), 1))])
                                    world_pts = (H_img_to_world @ pts.T).T
                                    world_pts = world_pts[:, :2] / world_pts[:, 2:3]
                                    centerline = world_pts
                                    tck, _ = rle.fit_spline(centerline)
                                    spline_world, curvature, arc = rle.compute_curvature_and_arc_length(tck)
                                    speed = rle.compute_speed_profile(curvature, arc)
                                    logger.info("Mapped extracted centerline to world coordinates and computed speed profile.")
                                except Exception as e:
                                    logger.warning(f"Failed mapping extracted centerline to world coords: {e}")
                            else:
                                # compute spline/speed using pixel-space centerline (units will be pixels)
                                try:
                                    tck, _ = rle.fit_spline(pts_px)
                                    spline_world, curvature, arc = rle.compute_curvature_and_arc_length(tck)
                                    speed = rle.compute_speed_profile(curvature, arc)
                                    logger.info("Computed speed profile on extracted pixel-space centerline (units in px)")

                                    # optionally optimize the pixel-space centerline as well
                                    if optimize_racing_line:
                                        try:
                                            logger.info("Optimizing extracted pixel-space centerline (pixel units)")
                                            spline_opt, curvature_opt, arc_opt, speed_opt, opt_res = rle.optimize_racing_line(pts_px, n_control=12, maxiter=optimize_maxiter)
                                            spline_world = spline_opt
                                            curvature = curvature_opt
                                            arc = arc_opt
                                            speed = speed_opt
                                            logger.info(f"Pixel-space optimization finished: success={opt_res.success}, lap_time={opt_res.fun:.3f}")
                                            # Update pts_px for overlay
                                            pts_px = np.vstack([spline_world[0], spline_world[1]]).T
                                            was_optimized = True
                                        except Exception as e:
                                            logger.warning(f"Pixel-space optimization failed: {e}")
                                except Exception as e:
                                    logger.warning(f"Failed computing spline/speed on extracted pixels: {e}")
                        else:
                            logger.warning("Automatic centerline extraction returned insufficient points; skipping overlay")
                    except Exception as e:
                        logger.warning(f"Automatic centerline extraction from segmentation failed: {e}")
                
                for frame_num, frame in vp.get_frames():
                    # Check if stop requested
                    if stop_callback and stop_callback():
                        logger.info(f"Processing stopped by user at frame {frame_num}")
                        break
                    
                    # Get corresponding telemetry
                    telemetry = None
                    if telemetry_data and frame_num < len(telemetry_data):
                        telemetry = telemetry_data[frame_num]
                    
                    # Predict
                    prediction = self.engine.predict(frame, telemetry)
                    inference_times.append(prediction['inference_time_ms'])
                    
                    # Visualize and write
                    result_frame = self.engine.visualize_prediction(frame, prediction)

                    # Overlay computed racing line (if available)
                    if rle is not None and pts_px is not None:
                        if speed is None:
                            logger.debug("RacingLine: speed not available; skipping overlay")
                        else:
                            try:
                                # Ensure enough points are inside the image bounds before drawing
                                pts_arr = np.asarray(pts_px)
                                inside = ((pts_arr[:, 0] >= 0) & (pts_arr[:, 0] < w) & (pts_arr[:, 1] >= 0) & (pts_arr[:, 1] < h))
                                inside_ratio = inside.sum() / max(1, len(pts_arr))
                                if inside_ratio < 0.1:
                                    logger.warning(f"RacingLine: too few centerline points visible (ratio={inside_ratio:.2f}), skipping overlay")
                                else:
                                    spline_px = (pts_px[:, 0], pts_px[:, 1])
                                    result_frame = rle.overlay_racing_line(result_frame, spline_px, speed)
                                    # If optimized, draw a strong purple polyline for the optimized path (better visual cue)
                                    if was_optimized:
                                        try:
                                            px = np.round(pts_px).astype(np.int32)
                                            # only keep points inside frame
                                            inside = ((px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h))
                                            px = px[inside]
                                            if px.shape[0] > 1:
                                                cv2.polylines(result_frame, [px.reshape((-1, 1, 2))], isClosed=False, color=(255, 0, 255), thickness=4)
                                        except Exception as e:
                                            logger.warning(f"Failed drawing optimized purple line: {e}")
                            except Exception as e:
                                logger.warning(f"Could not overlay racing line on frame {frame_num}: {e}")

                    writer.write_frame(result_frame)
                    
                    stats['total_frames'] += 1
                    stats['predictions'].append(prediction)
                    
                    # Calculate processing FPS
                    elapsed = time_module.time() - start_time
                    processing_fps = stats['total_frames'] / elapsed if elapsed > 0 else 0
                    
                    # Call progress callback
                    if progress_callback and frame_num % 5 == 0:  # Update every 5 frames
                        progress_callback(frame_num, total_frames, processing_fps, 
                                        prediction['inference_time_ms'])
                    
                    if frame_num % 30 == 0:
                        logger.info(f"Processed frame {frame_num}/{total_frames} ({processing_fps:.1f} FPS)")
        
        stats['avg_inference_time'] = np.mean(inference_times)
        logger.info(f"Processing complete. Average inference time: {stats['avg_inference_time']:.2f}ms")
        
        return stats
