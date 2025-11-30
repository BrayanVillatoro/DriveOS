"""
Interactive annotation tool for creating training data
Allows users to draw racing lines and track boundaries on video frames
"""
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Optional
import logging
from PIL import Image, ImageTk

logger = logging.getLogger(__name__)


class AnnotationTool:
    """Interactive tool for annotating racing videos"""
    
    def __init__(self, video_path: str, output_dir: str = 'data/user_annotations'):
        """
        Initialize annotation tool
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save annotations
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'masks').mkdir(exist_ok=True)
        (self.output_dir / 'annotations').mkdir(exist_ok=True)
        
        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Annotation state
        self.current_frame_idx = 0
        self.current_frame = None
        self.display_frame = None
        self.annotation_mode = 'racing_line'  # 'racing_line', 'left_boundary', 'right_boundary'
        
        # Drawing state
        self.racing_line_points = []
        self.left_boundary_points = []
        self.right_boundary_points = []
        self.is_drawing = False
        
        # Colors
        self.colors = {
            'racing_line': (0, 255, 255),  # Yellow
            'left_boundary': (0, 0, 255),  # Red
            'right_boundary': (255, 0, 0),  # Blue
        }
        
        # Saved frames
        self.saved_count = 0
        
        # Display scale factor
        self.display_scale = 1.0
        
        # Update flag to prevent recursion
        self.is_updating = False
        
        logger.info(f"Loaded video: {video_path}")
        logger.info(f"Total frames: {self.total_frames}, FPS: {self.fps}")
    
    def on_mouse_down(self, event):
        """Handle mouse button press"""
        self.is_drawing = True
        self._add_point(event.x, event.y)
    
    def on_mouse_move(self, event):
        """Handle mouse movement"""
        if self.is_drawing:
            self._add_point(event.x, event.y)
    
    def on_mouse_up(self, event):
        """Handle mouse button release"""
        self.is_drawing = False
    
    def on_right_click(self, event):
        """Handle right click to undo"""
        self._remove_last_point()
    
    def _add_point(self, x: int, y: int):
        """Add point to current annotation mode"""
        # Convert canvas coordinates to image coordinates
        if hasattr(self, 'display_scale') and self.display_scale > 0:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Get image dimensions
            img_h, img_w = self.current_frame.shape[:2]
            scaled_w = int(img_w * self.display_scale)
            scaled_h = int(img_h * self.display_scale)
            
            # Calculate offset (image is centered)
            offset_x = (canvas_width - scaled_w) // 2
            offset_y = (canvas_height - scaled_h) // 2
            
            # Convert to original image coordinates
            img_x = int((x - offset_x) / self.display_scale)
            img_y = int((y - offset_y) / self.display_scale)
            
            # Clamp to image bounds
            img_x = max(0, min(img_x, img_w - 1))
            img_y = max(0, min(img_y, img_h - 1))
            
            point = (img_x, img_y)
        else:
            point = (x, y)
        
        if self.annotation_mode == 'racing_line':
            self.racing_line_points.append(point)
        elif self.annotation_mode == 'left_boundary':
            self.left_boundary_points.append(point)
        elif self.annotation_mode == 'right_boundary':
            self.right_boundary_points.append(point)
        
        # Only update if not already updating
        if hasattr(self, 'window') and not self.is_updating:
            self._update_display()
    
    def _remove_last_point(self):
        """Remove last point from current annotation mode"""
        if self.annotation_mode == 'racing_line' and self.racing_line_points:
            self.racing_line_points.pop()
        elif self.annotation_mode == 'left_boundary' and self.left_boundary_points:
            self.left_boundary_points.pop()
        elif self.annotation_mode == 'right_boundary' and self.right_boundary_points:
            self.right_boundary_points.pop()
        
        # Only update if not already updating
        if hasattr(self, 'window') and not self.is_updating:
            self._update_display()
    
    def _update_display(self):
        """Update display with current annotations"""
        if not hasattr(self, 'canvas'):
            # Canvas not ready yet
            return
        
        # Prevent recursive updates
        if self.is_updating:
            return
        
        self.is_updating = True
        
        try:
            self.display_frame = self.current_frame.copy()
            
            # Draw racing line
            if len(self.racing_line_points) > 1:
                points = np.array(self.racing_line_points, dtype=np.int32)
                cv2.polylines(self.display_frame, [points], False, 
                             self.colors['racing_line'], 3, cv2.LINE_AA)
                # Draw starting point marker
                cv2.circle(self.display_frame, self.racing_line_points[0], 8, (0, 255, 0), -1)
                cv2.circle(self.display_frame, self.racing_line_points[0], 10, (255, 255, 255), 2)
            elif len(self.racing_line_points) == 1:
                # Show single point
                cv2.circle(self.display_frame, self.racing_line_points[0], 8, (0, 255, 0), -1)
                cv2.circle(self.display_frame, self.racing_line_points[0], 10, (255, 255, 255), 2)
            
            # Draw left boundary
            if len(self.left_boundary_points) > 1:
                points = np.array(self.left_boundary_points, dtype=np.int32)
                cv2.polylines(self.display_frame, [points], False, 
                             self.colors['left_boundary'], 2, cv2.LINE_AA)
                # Draw starting point marker
                cv2.circle(self.display_frame, self.left_boundary_points[0], 6, (255, 255, 0), -1)
            elif len(self.left_boundary_points) == 1:
                cv2.circle(self.display_frame, self.left_boundary_points[0], 6, (255, 255, 0), -1)
            
            # Draw right boundary
            if len(self.right_boundary_points) > 1:
                points = np.array(self.right_boundary_points, dtype=np.int32)
                cv2.polylines(self.display_frame, [points], False, 
                             self.colors['right_boundary'], 2, cv2.LINE_AA)
                # Draw starting point marker
                cv2.circle(self.display_frame, self.right_boundary_points[0], 6, (255, 255, 0), -1)
            elif len(self.right_boundary_points) == 1:
                cv2.circle(self.display_frame, self.right_boundary_points[0], 6, (255, 255, 0), -1)
            
            # Convert to PIL Image for tkinter
            frame_rgb = cv2.cvtColor(self.display_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Resize to fit canvas
            self.canvas.update()  # Force update to get real dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 10 and canvas_height > 10:
                # Calculate scaling to fit
                img_width, img_height = pil_image.size
                scale = min(canvas_width / img_width, canvas_height / img_height) * 0.95
                
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Store scale factor for mouse coordinate conversion
                self.display_scale = scale
            
                # Convert to PhotoImage and display
                self.photo = ImageTk.PhotoImage(pil_image)
                self.canvas.delete('all')
                self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.photo, anchor='center')
                
                # Update stats
                if hasattr(self, 'stats_label'):
                    self.stats_label.config(text=f"Saved: {self.saved_count}\nFrame: {self.current_frame_idx}/{self.total_frames}")
        
        finally:
            self.is_updating = False
    
    def load_frame(self, frame_idx: int) -> bool:
        """Load specific frame from video"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if ret:
            self.current_frame_idx = frame_idx
            self.current_frame = frame
            self.display_frame = frame.copy()
            return True
        return False
    
    def save_annotation(self):
        """Save current frame with annotations"""
        if not self.racing_line_points:
            logger.warning("No racing line annotated, skipping save")
            return
        
        # Generate filename
        filename = f"frame_{self.saved_count:06d}"
        
        # Save original image
        image_path = self.output_dir / 'images' / f"{filename}.jpg"
        cv2.imwrite(str(image_path), self.current_frame)
        
        # Create segmentation mask
        mask = self._create_mask()
        mask_path = self.output_dir / 'masks' / f"{filename}.png"
        cv2.imwrite(str(mask_path), mask)
        
        # Save annotation data (JSON)
        annotation = {
            'frame_index': self.current_frame_idx,
            'racing_line': self.racing_line_points,
            'left_boundary': self.left_boundary_points,
            'right_boundary': self.right_boundary_points,
        }
        
        json_path = self.output_dir / 'annotations' / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        self.saved_count += 1
        logger.info(f"Saved annotation {self.saved_count}: {filename}")
        
        # Clear annotations for next frame
        self.clear_annotations()
        
        # Auto-advance to next frame
        self.load_frame(self.current_frame_idx + 10)
        self._update_display()
    
    def _create_mask(self) -> np.ndarray:
        """
        Create segmentation mask from annotations
        
        Mask classes:
        0: Background
        1: Racing line
        2: Track boundaries
        3: Track surface (area between boundaries)
        """
        h, w = self.current_frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Draw track surface (area between boundaries)
        if len(self.left_boundary_points) > 2 and len(self.right_boundary_points) > 2:
            # Create polygon from both boundaries
            left_pts = np.array(self.left_boundary_points, dtype=np.int32)
            right_pts = np.array(self.right_boundary_points[::-1], dtype=np.int32)  # Reverse
            track_polygon = np.vstack([left_pts, right_pts])
            
            cv2.fillPoly(mask, [track_polygon], 3)
        
        # Draw boundaries (thicker)
        if len(self.left_boundary_points) > 1:
            points = np.array(self.left_boundary_points, dtype=np.int32)
            cv2.polylines(mask, [points], False, 2, thickness=8)
        
        if len(self.right_boundary_points) > 1:
            points = np.array(self.right_boundary_points, dtype=np.int32)
            cv2.polylines(mask, [points], False, 2, thickness=8)
        
        # Draw racing line (on top)
        if len(self.racing_line_points) > 1:
            points = np.array(self.racing_line_points, dtype=np.int32)
            cv2.polylines(mask, [points], False, 1, thickness=10)
        
        return mask
    
    def clear_annotations(self):
        """Clear all annotations for current frame"""
        self.racing_line_points = []
        self.left_boundary_points = []
        self.right_boundary_points = []
        if hasattr(self, 'window') and not self.is_updating:
            self._update_display()
    
    def run(self, parent_window=None):
        """Run interactive annotation tool with tkinter GUI"""
        # Create tkinter window
        if parent_window:
            self.window = tk.Toplevel(parent_window)
        else:
            self.window = tk.Tk()
        
        self.window.title("DriveOS Annotation Tool")
        self.window.geometry("1400x900")
        self.window.configure(bg='#1e1e1e')
        
        # Create main frame
        main_frame = tk.Frame(self.window, bg='#1e1e1e')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Canvas for image display
        canvas_frame = tk.Frame(main_frame, bg='#2d2d30', relief='solid', bd=2)
        canvas_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        self.canvas = tk.Canvas(canvas_frame, bg='#1e1e1e', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_move)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.canvas.bind('<Button-3>', self.on_right_click)
        
        # Control panel
        control_frame = tk.Frame(main_frame, bg='#2d2d30', width=300, relief='solid', bd=2)
        control_frame.pack(side='right', fill='y')
        control_frame.pack_propagate(False)
        
        # Title
        tk.Label(control_frame, text="Annotation Controls", font=('Segoe UI', 14, 'bold'),
                bg='#2d2d30', fg='#ffffff').pack(pady=15)
        
        # Mode selection
        mode_frame = tk.LabelFrame(control_frame, text="Draw Mode", bg='#2d2d30', fg='#ffffff',
                                   font=('Segoe UI', 10, 'bold'), padx=10, pady=10)
        mode_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(mode_frame, text="1 - Racing Line", bg='#007acc', fg='white',
                 font=('Segoe UI', 10), command=lambda: self.set_mode('racing_line')).pack(fill='x', pady=3)
        tk.Button(mode_frame, text="2 - Left Boundary", bg='#dc3545', fg='white',
                 font=('Segoe UI', 10), command=lambda: self.set_mode('left_boundary')).pack(fill='x', pady=3)
        tk.Button(mode_frame, text="3 - Right Boundary", bg='#0d6efd', fg='white',
                 font=('Segoe UI', 10), command=lambda: self.set_mode('right_boundary')).pack(fill='x', pady=3)
        
        # Current mode display
        self.mode_label = tk.Label(control_frame, text="Mode: Racing Line", font=('Segoe UI', 11),
                                   bg='#2d2d30', fg='#00ff00')
        self.mode_label.pack(pady=5)
        
        # Action buttons
        action_frame = tk.LabelFrame(control_frame, text="Actions", bg='#2d2d30', fg='#ffffff',
                                     font=('Segoe UI', 10, 'bold'), padx=10, pady=10)
        action_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(action_frame, text="Save Frame (SPACE)", bg='#28a745', fg='white',
                 font=('Segoe UI', 10, 'bold'), command=self.save_annotation).pack(fill='x', pady=3)
        tk.Button(action_frame, text="Undo Last Point (U)", bg='#ff9800', fg='white',
                 font=('Segoe UI', 10), command=self._remove_last_point).pack(fill='x', pady=3)
        tk.Button(action_frame, text="Clear All (C)", bg='#ffc107', fg='black',
                 font=('Segoe UI', 10), command=self.clear_annotations).pack(fill='x', pady=3)
        tk.Button(action_frame, text="Next Frame (N)", bg='#6c757d', fg='white',
                 font=('Segoe UI', 10), command=self.next_frame).pack(fill='x', pady=3)
        tk.Button(action_frame, text="Previous Frame (B)", bg='#6c757d', fg='white',
                 font=('Segoe UI', 10), command=self.prev_frame).pack(fill='x', pady=3)
        tk.Button(action_frame, text="Quit (Q)", bg='#dc3545', fg='white',
                 font=('Segoe UI', 10, 'bold'), command=self.quit_tool).pack(fill='x', pady=3)
        
        # Stats
        stats_frame = tk.LabelFrame(control_frame, text="Statistics", bg='#2d2d30', fg='#ffffff',
                                   font=('Segoe UI', 10, 'bold'), padx=10, pady=10)
        stats_frame.pack(fill='x', padx=10, pady=10)
        
        self.stats_label = tk.Label(stats_frame, text=f"Saved: 0\nFrame: 0/{self.total_frames}",
                                    font=('Segoe UI', 10), bg='#2d2d30', fg='#ffffff', justify='left')
        self.stats_label.pack(anchor='w')
        
        # Instructions
        instructions = tk.Label(control_frame, text="Left click: Draw\nRight click: Undo last point\nMouse drag: Continuous draw\nU key: Undo last point\n\nGreen circle = Start point",
                               font=('Segoe UI', 9), bg='#2d2d30', fg='#cccccc', justify='left')
        instructions.pack(pady=10)
        
        # Keyboard bindings
        self.window.bind('1', lambda e: self.set_mode('racing_line'))
        self.window.bind('2', lambda e: self.set_mode('left_boundary'))
        self.window.bind('3', lambda e: self.set_mode('right_boundary'))
        self.window.bind('<space>', lambda e: self.save_annotation())
        self.window.bind('u', lambda e: self._remove_last_point())
        self.window.bind('c', lambda e: self.clear_annotations())
        self.window.bind('n', lambda e: self.next_frame())
        self.window.bind('b', lambda e: self.prev_frame())
        self.window.bind('q', lambda e: self.quit_tool())
        
        # Load first frame
        self.load_frame(0)
        
        logger.info("Annotation tool started")
        logger.info("Use mouse to draw, keyboard for controls")
        
        # Schedule initial display update after window is ready
        self.window.after(100, self._update_display)
        
        # Start GUI loop
        self.window.mainloop()
        
        # Cleanup
        self.cap.release()
        
        logger.info(f"Annotation complete! Saved {self.saved_count} frames to {self.output_dir}")
        logger.info(f"You can now use this data for training with: python -m src.train --data-dir {self.output_dir}")
    
    def set_mode(self, mode):
        """Set annotation mode"""
        self.annotation_mode = mode
        mode_names = {
            'racing_line': 'Racing Line',
            'left_boundary': 'Left Boundary',
            'right_boundary': 'Right Boundary'
        }
        self.mode_label.config(text=f"Mode: {mode_names[mode]}")
        if not self.is_updating:
            self._update_display()
    
    def next_frame(self):
        """Load next frame"""
        next_frame_idx = min(self.current_frame_idx + 10, self.total_frames - 1)
        self.load_frame(next_frame_idx)
        if not self.is_updating:
            self._update_display()
    
    def prev_frame(self):
        """Load previous frame"""
        prev_frame_idx = max(self.current_frame_idx - 10, 0)
        self.load_frame(prev_frame_idx)
        if not self.is_updating:
            self._update_display()
    
    def quit_tool(self):
        """Quit annotation tool"""
        if messagebox.askyesno("Quit", f"Are you sure? You have saved {self.saved_count} frames."):
            self.window.destroy()


def annotate_video(video_path: str, output_dir: str = 'data/user_annotations', parent_window=None):
    """
    Launch interactive annotation tool
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save annotations
        parent_window: Parent tkinter window (optional)
    """
    tool = AnnotationTool(video_path, output_dir)
    tool.run(parent_window)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Annotate racing videos for training')
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--output', '-o', default='data/user_annotations',
                       help='Output directory for annotations')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    annotate_video(args.video, args.output)
