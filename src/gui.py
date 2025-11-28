"""
DriveOS GUI Application
Modern interface for racing line analysis with live preview, file upload, and training controls
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import cv2
from PIL import Image, ImageTk
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from .inference import BatchProcessor
from .train import train_model


class DriveOSGUI:
    """Main GUI application for DriveOS"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("DriveOS - AI Racing Line Analyzer")
        self.root.geometry("1200x800")
        
        # Set window icon if available
        try:
            self.root.iconbitmap(default='icon.ico')
        except:
            pass
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Custom colors - professional dark theme
        self.colors = {
            'bg': '#f0f0f0',
            'fg': '#2d2d2d',
            'accent': '#0078d4',
            'success': '#107c10',
            'warning': '#ff8c00',
            'error': '#e81123',
            'card': '#ffffff'
        }
        
        # Configure root background
        self.root.configure(bg=self.colors['bg'])
        
        # Configure styles
        self.setup_styles()
        
        # State variables
        self.current_video_path = None
        self.is_processing = False
        self.processor = None
        self.frame_queue = queue.Queue(maxsize=1)
        self.stop_processing = False
        
        # Create main layout
        self.create_layout()
        
    def setup_styles(self):
        """Configure custom styles"""
        self.style.configure('Title.TLabel', 
                           font=('Segoe UI', 28, 'bold'),
                           foreground=self.colors['accent'],
                           background=self.colors['bg'])
        self.style.configure('Subtitle.TLabel',
                           font=('Segoe UI', 11),
                           foreground='#666666',
                           background=self.colors['bg'])
        self.style.configure('Heading.TLabel',
                           font=('Segoe UI', 14, 'bold'),
                           foreground=self.colors['fg'],
                           background=self.colors['card'])
        self.style.configure('Info.TLabel',
                           font=('Segoe UI', 10),
                           foreground='#666666',
                           background=self.colors['card'])
        self.style.configure('Success.TLabel',
                           font=('Segoe UI', 10, 'bold'),
                           foreground=self.colors['success'],
                           background=self.colors['card'])
        self.style.configure('Card.TFrame',
                           background=self.colors['card'],
                           relief='flat')
        self.style.configure('Big.TButton',
                           font=('Segoe UI', 14, 'bold'),
                           padding=15)
        self.style.configure('Action.TButton',
                           font=('Segoe UI', 12, 'bold'),
                           padding=10)
        
    def create_layout(self):
        """Create main application layout"""
        # Header
        header = tk.Frame(self.root, bg=self.colors['bg'], height=120)
        header.pack(fill='x', padx=30, pady=20)
        header.pack_propagate(False)
        
        title = ttk.Label(header, text="🏁 DriveOS Racing Line Analyzer",
                         style='Title.TLabel')
        title.pack(anchor='w')
        
        subtitle = ttk.Label(header, 
                            text="AI-powered racing line analysis • Identify the fastest path around any track",
                            style='Subtitle.TLabel')
        subtitle.pack(anchor='w', pady=(5, 0))
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        # Tab 1: Analyze Video (Main feature)
        self.upload_tab = ttk.Frame(self.notebook, style='Card.TFrame')
        self.notebook.add(self.upload_tab, text='  📹 Analyze Video  ')
        self.create_upload_view()
        
        # Tab 2: Live View
        self.live_tab = ttk.Frame(self.notebook, style='Card.TFrame')
        self.notebook.add(self.live_tab, text='  ▶️ Live Preview  ')
        self.create_live_view()
        
        # Tab 3: Training
        self.training_tab = ttk.Frame(self.notebook, style='Card.TFrame')
        self.notebook.add(self.training_tab, text='  🎯 Train Model  ')
        self.create_training_view()
        
    def create_live_view(self):
        """Create live view interface"""
        # Header
        header = ttk.Frame(self.live_tab)
        header.pack(fill='x', padx=20, pady=10)
        
        title = ttk.Label(header, text="Live Racing Line Analysis",
                         style='Title.TLabel')
        title.pack(side='left')
        
        # Source selection frame
        source_frame = ttk.LabelFrame(self.live_tab, text="Video Source", padding=15)
        source_frame.pack(fill='x', padx=20, pady=10)
        
        # Source type selector
        source_type_frame = ttk.Frame(source_frame)
        source_type_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(source_type_frame, text="Source Type:", style='Info.TLabel').pack(side='left', padx=(0, 10))
        
        self.source_type_var = tk.StringVar(value="video")
        
        ttk.Radiobutton(source_type_frame, text="📹 Video File", 
                       variable=self.source_type_var, value="video",
                       command=self.update_source_controls).pack(side='left', padx=10)
        
        ttk.Radiobutton(source_type_frame, text="📷 Webcam/Camera", 
                       variable=self.source_type_var, value="camera",
                       command=self.update_source_controls).pack(side='left', padx=10)
        
        ttk.Radiobutton(source_type_frame, text="🖥️ Screen Capture", 
                       variable=self.source_type_var, value="screen",
                       command=self.update_source_controls).pack(side='left', padx=10)
        
        # Source-specific controls
        self.source_controls_frame = ttk.Frame(source_frame)
        self.source_controls_frame.pack(fill='x')
        
        # Video file controls
        self.video_controls = ttk.Frame(self.source_controls_frame)
        self.live_video_path_var = tk.StringVar(value="No video selected")
        ttk.Label(self.video_controls, textvariable=self.live_video_path_var, 
                 style='Info.TLabel').pack(side='left', padx=(0, 10))
        ttk.Button(self.video_controls, text="📁 Browse", 
                  command=self.select_live_video).pack(side='left', padx=5)
        
        # Camera controls
        self.camera_controls = ttk.Frame(self.source_controls_frame)
        ttk.Label(self.camera_controls, text="Camera Index:", 
                 style='Info.TLabel').pack(side='left', padx=(0, 10))
        self.camera_index_var = tk.IntVar(value=0)
        ttk.Spinbox(self.camera_controls, from_=0, to=10, 
                   textvariable=self.camera_index_var, width=10).pack(side='left', padx=5)
        ttk.Button(self.camera_controls, text="🔍 Test Camera",
                  command=self.test_camera).pack(side='left', padx=5)
        
        # Screen capture controls
        self.screen_controls = ttk.Frame(self.source_controls_frame)
        ttk.Label(self.screen_controls, text="Capture Mode:", 
                 style='Info.TLabel').pack(side='left', padx=(0, 10))
        self.screen_mode_var = tk.StringVar(value="fullscreen")
        screen_mode_combo = ttk.Combobox(self.screen_controls, 
                                        textvariable=self.screen_mode_var,
                                        values=['fullscreen', 'window'], 
                                        state='readonly', width=15)
        screen_mode_combo.pack(side='left', padx=5)
        ttk.Label(self.screen_controls, text="(Window selection will appear on start)", 
                 style='Info.TLabel', foreground='#999999').pack(side='left', padx=10)
        
        # Show initial controls
        self.video_controls.pack(fill='x')
        
        # Video display area
        video_frame = ttk.LabelFrame(self.live_tab, text="Video Feed", padding=10)
        video_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Canvas for video
        self.live_canvas = tk.Canvas(video_frame, bg='#000000', 
                                     width=1280, height=720)
        self.live_canvas.pack()
        
        # Status label
        self.live_status = ttk.Label(video_frame, text="Ready - Select a source to start",
                                    style='Info.TLabel')
        self.live_status.pack(pady=5)
        
        # Controls
        controls = ttk.Frame(self.live_tab)
        controls.pack(fill='x', padx=20, pady=10)
        
        self.play_btn = ttk.Button(controls, text="▶ Start Processing",
                                   command=self.start_live_processing,
                                   state='disabled')
        self.play_btn.pack(side='left', padx=5)
        
        self.stop_btn = ttk.Button(controls, text="⬛ Stop",
                                   command=self.stop_live_processing,
                                   state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        # Info panel
        info_frame = ttk.LabelFrame(self.live_tab, text="Real-time Statistics", padding=10)
        info_frame.pack(fill='x', padx=20, pady=10)
        
        self.fps_label = ttk.Label(info_frame, text="FPS: --", style='Info.TLabel')
        self.fps_label.pack(side='left', padx=20)
        
        self.frame_label = ttk.Label(info_frame, text="Frame: --", style='Info.TLabel')
        self.frame_label.pack(side='left', padx=20)
        
        self.confidence_label = ttk.Label(info_frame, text="Confidence: --", style='Info.TLabel')
        self.confidence_label.pack(side='left', padx=20)
        
    def create_upload_view(self):
        """Create upload and processing interface"""
        # Instructions card
        instruction_frame = tk.Frame(self.upload_tab, bg='#e8f4fd', relief='solid', borderwidth=1)
        instruction_frame.pack(fill='x', padx=30, pady=20)
        
        instruction_text = tk.Label(instruction_frame,
                                   text="📖 How to use:\n"
                                        "1. Click 'Select Video' to choose your racing video\n"
                                        "2. Choose where to save the analyzed video\n"
                                        "3. Click 'Analyze Video' to start processing\n"
                                        "4. Wait for the analysis to complete - a purple line will show the optimal racing line!",
                                   font=('Segoe UI', 10),
                                   bg='#e8f4fd',
                                   fg='#004578',
                                   justify='left',
                                   padx=20,
                                   pady=15)
        instruction_text.pack(anchor='w')
        
        # File selection card
        file_frame = ttk.LabelFrame(self.upload_tab, text="Step 1: Select Racing Video", padding=20)
        file_frame.pack(fill='x', padx=30, pady=10)
        
        self.file_path_var = tk.StringVar(value="No video selected - click 'Select Video' button")
        
        path_display = tk.Label(file_frame, textvariable=self.file_path_var,
                               font=('Segoe UI', 10),
                               fg='#666666',
                               bg=self.colors['card'],
                               anchor='w',
                               wraplength=800)
        path_display.pack(fill='x', pady=(0, 10))
        
        ttk.Button(file_frame, text="📁 Select Video",
                  command=self.browse_video,
                  style='Action.TButton').pack()
        
        # Output settings
        output_frame = ttk.LabelFrame(self.upload_tab, text="Step 2: Choose Output Location", padding=20)
        output_frame.pack(fill='x', padx=30, pady=10)
        
        self.output_dir_var = tk.StringVar(value=str(Path.home() / "Videos"))
        
        output_display = tk.Label(output_frame,
                                 text=f"Analyzed videos will be saved to: {self.output_dir_var.get()}",
                                 font=('Segoe UI', 10),
                                 fg='#666666',
                                 bg=self.colors['card'],
                                 anchor='w')
        output_display.pack(fill='x', pady=(0, 10))
        
        ttk.Button(output_frame, text="📂 Change Output Folder",
                  command=self.browse_output,
                  style='Action.TButton').pack()
        
        # Process button - LARGE and prominent
        process_frame = tk.Frame(self.upload_tab, bg=self.colors['card'])
        process_frame.pack(fill='x', padx=30, pady=30)
        
        self.process_btn = tk.Button(process_frame,
                                     text="▶️  ANALYZE VIDEO WITH AI  ▶️",
                                     command=self.start_batch_processing,
                                     font=('Segoe UI', 16, 'bold'),
                                     bg=self.colors['accent'],
                                     fg='white',
                                     activebackground='#005a9e',
                                     activeforeground='white',
                                     relief='flat',
                                     padx=40,
                                     pady=20,
                                     cursor='hand2',
                                     state='disabled')
        self.process_btn.pack()
        
        # Progress
        progress_frame = ttk.LabelFrame(self.upload_tab, text="Processing Status", padding=20)
        progress_frame.pack(fill='both', expand=True, padx=30, pady=(0, 20))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                           maximum=100)
        self.progress_bar.pack(fill='x', pady=10)
        
        self.progress_label = ttk.Label(progress_frame, text="Ready - Select a video to begin",
                                       style='Info.TLabel')
        self.progress_label.pack()

        
    def create_training_view(self):
        """Create training interface with adjustable parameters"""
        # Header
        header = ttk.Frame(self.training_tab)
        header.pack(fill='x', padx=20, pady=10)
        
        title = ttk.Label(header, text="Model Training",
                         style='Title.TLabel')
        title.pack(side='left')
        
        # Training data
        data_frame = ttk.LabelFrame(self.training_tab, text="Training Data", padding=20)
        data_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Label(data_frame, text="Training Data Directory:",
                 style='Info.TLabel').grid(row=0, column=0, sticky='w', pady=5)
        
        self.training_data_var = tk.StringVar(value="data/training")
        ttk.Entry(data_frame, textvariable=self.training_data_var,
                 width=50).grid(row=0, column=1, padx=10)
        
        ttk.Button(data_frame, text="Browse...",
                  command=self.browse_training_data).grid(row=0, column=2)
        
        ttk.Button(data_frame, text="Generate Training Data",
                  command=self.generate_training_data).grid(row=1, column=1, pady=10)
        
        # Training parameters
        params_frame = ttk.LabelFrame(self.training_tab, text="Training Parameters", padding=20)
        params_frame.pack(fill='x', padx=20, pady=10)
        
        # Epochs
        ttk.Label(params_frame, text="Epochs:",
                 style='Info.TLabel').grid(row=0, column=0, sticky='w', pady=5)
        self.epochs_var = tk.IntVar(value=20)
        ttk.Scale(params_frame, from_=5, to=100, variable=self.epochs_var,
                 orient='horizontal', length=300).grid(row=0, column=1, padx=10)
        ttk.Label(params_frame, textvariable=self.epochs_var).grid(row=0, column=2)
        
        # Batch size
        ttk.Label(params_frame, text="Batch Size:",
                 style='Info.TLabel').grid(row=1, column=0, sticky='w', pady=5)
        self.batch_size_var = tk.IntVar(value=2)
        ttk.Spinbox(params_frame, from_=1, to=16, textvariable=self.batch_size_var,
                   width=10).grid(row=1, column=1, sticky='w', padx=10)
        
        # Learning rate
        ttk.Label(params_frame, text="Learning Rate:",
                 style='Info.TLabel').grid(row=2, column=0, sticky='w', pady=5)
        self.lr_var = tk.DoubleVar(value=0.001)
        ttk.Entry(params_frame, textvariable=self.lr_var,
                 width=15).grid(row=2, column=1, sticky='w', padx=10)
        
        # Device selection
        ttk.Label(params_frame, text="Device:",
                 style='Info.TLabel').grid(row=3, column=0, sticky='w', pady=5)
        self.device_var = tk.StringVar(value="auto")
        device_combo = ttk.Combobox(params_frame, textvariable=self.device_var,
                                    values=['auto', 'cpu', 'cuda'], state='readonly',
                                    width=12)
        device_combo.grid(row=3, column=1, sticky='w', padx=10)
        
        # Training progress
        training_progress_frame = ttk.LabelFrame(self.training_tab, 
                                                text="Training Progress", padding=20)
        training_progress_frame.pack(fill='x', padx=20, pady=10)
        
        self.training_progress_var = tk.DoubleVar()
        self.training_progress_bar = ttk.Progressbar(training_progress_frame,
                                                     variable=self.training_progress_var,
                                                     maximum=100, length=800)
        self.training_progress_bar.pack(fill='x', pady=10)
        
        self.training_status_label = ttk.Label(training_progress_frame,
                                              text="Ready to train",
                                              style='Info.TLabel')
        self.training_status_label.pack()
        
        # Train button
        self.train_btn = ttk.Button(self.training_tab, text="🎯 Start Training",
                                    command=self.start_training,
                                    style='Accent.TButton')
        self.train_btn.pack(pady=20)
        
        # Training metrics
        metrics_frame = ttk.LabelFrame(self.training_tab, text="Training Metrics", padding=10)
        metrics_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.metrics_text = tk.Text(metrics_frame, height=10, bg='#2d2d2d', fg='#ffffff',
                                   font=('Consolas', 9))
        self.metrics_text.pack(fill='both', expand=True)
        
    # Live View Methods
    def update_source_controls(self):
        """Update source controls based on selected type"""
        # Hide all controls
        self.video_controls.pack_forget()
        self.camera_controls.pack_forget()
        self.screen_controls.pack_forget()
        
        # Show relevant controls
        source_type = self.source_type_var.get()
        if source_type == "video":
            self.video_controls.pack(fill='x')
            self.live_status.config(text="Select a video file to begin")
    def start_live_processing(self):
        """Start live video processing"""
        source_type = self.source_type_var.get()
        
        # Validate source
        if source_type == "video" and not hasattr(self, 'current_video_path'):
            messagebox.showwarning("No Source", "Please select a video file first")
            return
        
        self.is_processing = True
        self.stop_processing = False
        self.play_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        
        # Start processing thread
        thread = threading.Thread(target=self.live_processing_thread, daemon=True)
        thread.start()
        
        # Start display update
        self.update_live_display()
        
        if file_path:
            self.current_video_path = file_path
            self.live_video_path_var.set(Path(file_path).name)
            self.live_status.config(text=f"Selected: {Path(file_path).name}")
            self.play_btn.config(state='normal')
    
    def test_camera(self):
        """Test camera connection"""
        camera_index = self.camera_index_var.get()
        try:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    messagebox.showinfo("Camera Test", f"✓ Camera {camera_index} is working!")
                else:
                    messagebox.showerror("Camera Test", f"Camera {camera_index} opened but failed to read frame")
            else:
                messagebox.showerror("Camera Test", f"Cannot open camera {camera_index}")
        except Exception as e:
            messagebox.showerror("Camera Test", f"Error testing camera: {str(e)}")
            
    def start_live_processing(self):
        """Start live video processing"""
        if not self.current_video_path:
            return
        
        self.is_processing = True
        self.stop_processing = False
        self.play_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        
        # Start processing thread
        thread = threading.Thread(target=self.live_processing_thread, daemon=True)
        thread.start()
        
        # Start display update
        self.update_live_display()
        
    def stop_live_processing(self):
        """Stop live processing"""
        self.stop_processing = True
        self.is_processing = False
    def live_processing_thread(self):
        """Background thread for live processing"""
        try:
            processor = BatchProcessor('models/racing_line_model.pth')
            
            # Determine capture source
            source_type = self.source_type_var.get()
            
            if source_type == "video":
                cap = cv2.VideoCapture(self.current_video_path)
            elif source_type == "camera":
                camera_index = self.camera_index_var.get()
                cap = cv2.VideoCapture(camera_index)
                if not cap.isOpened():
                    self.live_status.config(text=f"Error: Cannot open camera {camera_index}")
                    self.stop_processing = True
                    return
            elif source_type == "screen":
                # Import screen capture libraries
                try:
                    import mss
                    import mss.tools
                    
                    sct = mss.mss()
                    
                    # If window mode, get the monitor (full screen for now)
                    monitor = sct.monitors[1]  # Primary monitor
                    
                    frame_count = 0
                    import time
                    
                    while not self.stop_processing:
                        # Capture screen
                        sct_img = sct.grab(monitor)
                        frame = np.array(sct_img)
                        
                        # Convert from BGRA to BGR
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                        
                        # Process frame
                        prediction = processor.engine.predict(frame, None)
                        result_frame = processor.engine.visualize_prediction(frame, prediction)
                        
                        # Convert to RGB for display
                        result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                        
                        # Put in queue (non-blocking)
                        try:
                            self.frame_queue.put_nowait((result_frame, frame_count, prediction))
                        except queue.Full:
                            pass
                        
                        frame_count += 1
                        time.sleep(0.033)  # ~30 FPS
                        
                    sct.close()
                    self.stop_processing = True
                    return
                    
                except ImportError:
                    self.live_status.config(text="Error: mss library required for screen capture. Install with: pip install mss")
                    self.stop_processing = True
                    return
            
            # Standard video/camera processing loop
            frame_count = 0
            while not self.stop_processing and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    if source_type == "video":
                        break  # End of video
                    else:
                        continue  # Camera error, try again
                
                # Process frame
                prediction = processor.engine.predict(frame, None)
                result_frame = processor.engine.visualize_prediction(frame, prediction)
                
                # Convert to RGB for display
                result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                
                # Put in queue (non-blocking)
                try:
                    self.frame_queue.put_nowait((result_frame, frame_count, prediction))
                except queue.Full:
                    pass
                
                frame_count += 1
                
            cap.release()
            self.stop_processing = True
            
        except Exception as e:
            self.live_status.config(text=f"Error: {str(e)}")
            self.stop_processing = True
            
        except Exception as e:
            self.live_status.config(text=f"Error: {str(e)}")
            
    def update_live_display(self):
        """Update live video display"""
        if not self.is_processing:
            return
        
        try:
            frame, frame_num, prediction = self.frame_queue.get_nowait()
            
            # Resize for display
            h, w = frame.shape[:2]
            max_w, max_h = 1280, 720
            scale = min(max_w/w, max_h/h)
            new_w, new_h = int(w*scale), int(h*scale)
            frame = cv2.resize(frame, (new_w, new_h))
            
            # Convert to PhotoImage
            img = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=img)
            
            # Update canvas
            self.live_canvas.create_image(max_w//2, max_h//2, image=photo)
            self.live_canvas.image = photo  # Keep reference
            
            # Update stats
            self.frame_label.config(text=f"Frame: {frame_num}")
            confidence = prediction['confidence'].mean()
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}")
            
        except queue.Empty:
            pass
        
        # Schedule next update
        if self.is_processing:
            self.root.after(30, self.update_live_display)
        
    # Upload Methods
    def browse_video(self):
        """Browse for video file"""
        file_path = filedialog.askopenfilename(
            title="Select Racing Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.file_path_var.set(f"✓ Selected: {file_path}")
            self.process_btn.config(state='normal', bg=self.colors['success'])
            self.progress_label.config(text="Ready to analyze! Click the big green button above.")
            
    def browse_output(self):
        """Browse for output directory"""
        dir_path = filedialog.askdirectory(title="Select Output Directory")
        if dir_path:
            self.output_dir_var.set(dir_path)
            
    def start_batch_processing(self):
        """Start batch video processing"""
        video_path = self.file_path_var.get()
        if "No video selected" in video_path:
            messagebox.showwarning("No Video Selected", "Please select a racing video file first.")
            return
        
        # Extract actual path
        if "✓ Selected: " in video_path:
            video_path = video_path.replace("✓ Selected: ", "")
        
        # Generate output filename
        input_name = Path(video_path).stem
        output_path = Path(self.output_dir_var.get()) / f"analyzed_{input_name}.mp4"
        
        self.process_btn.config(state='disabled')
        self.progress_label.config(text="🔄 Processing video with AI... This may take a few minutes.")
        self.progress_var.set(0)
        
        # Start processing thread
        thread = threading.Thread(
            target=self.batch_processing_thread,
            args=(video_path, str(output_path)),
            daemon=True
        )
        thread.start()
        
    def batch_processing_thread(self, video_path, output_path):
        """Background thread for batch processing"""
        try:
            processor = BatchProcessor('models/racing_line_model.pth')
            
            # Get total frames
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            self.progress_label.config(text=f"🔄 Analyzing {total_frames} frames with AI...")
            self.progress_var.set(25)
            
            # Process with progress updates
            stats = processor.process_video(video_path, output_path)
            
            self.progress_var.set(100)
            self.progress_label.config(text="✅ Analysis complete!")
            
            # Show success dialog with output location
            result = messagebox.showinfo(
                "Analysis Complete! 🏁",
                f"Your racing video has been analyzed successfully!\n\n"
                f"The analyzed video with the optimal racing line (purple) has been saved to:\n\n"
                f"{output_path}\n\n"
                f"Average processing time: {stats['avg_inference_time']:.1f}ms per frame\n"
                f"Total frames analyzed: {stats['total_frames']}"
            )
            
            # Ask if user wants to open the folder
            if messagebox.askyesno("Open Folder?", "Would you like to open the folder containing your analyzed video?"):
                import subprocess
                subprocess.run(['explorer', '/select,', str(output_path)])
            
        except Exception as e:
            self.progress_label.config(text=f"❌ Error: {str(e)}")
            messagebox.showerror("Processing Error", 
                               f"Sorry, the video could not be processed.\n\n"
                               f"Error: {str(e)}\n\n"
                               f"Make sure the video file is valid and not corrupted.")
        finally:
            self.process_btn.config(state='normal', bg=self.colors['success'])
            self.progress_var.set(0)
        
    # Training Methods
    def browse_training_data(self):
        """Browse for training data directory"""
        dir_path = filedialog.askdirectory(title="Select Training Data Directory")
        if dir_path:
            self.training_data_var.set(dir_path)
            
    def generate_training_data(self):
        """Open dialog to generate training data"""
        video_path = filedialog.askopenfilename(
            title="Select Video for Training Data",
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        
        if not video_path:
            return
        
        output_dir = self.training_data_var.get()
        
        response = messagebox.askyesno(
            "Generate Training Data",
            f"Generate training data from:\n{video_path}\n\nOutput to:\n{output_dir}\n\nContinue?"
        )
        
        if response:
            self.log_training("Generating training data...")
            # Run auto labeling in thread
            thread = threading.Thread(
                target=self.run_auto_labeling,
                args=(video_path, output_dir),
                daemon=True
            )
            thread.start()
            
    def run_auto_labeling(self, video_path, output_dir):
        """Run automatic labeling"""
        try:
            from auto_label_track import AutoTrackLabeler
            labeler = AutoTrackLabeler()
            labeler.process_video(video_path, output_dir, frame_interval=30, preview=False)
            self.log_training("✓ Training data generated successfully!")
            messagebox.showinfo("Success", "Training data generated!")
        except Exception as e:
            self.log_training(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate data: {str(e)}")
            
    def start_training(self):
        """Start model training"""
        data_dir = self.training_data_var.get()
        
        if not Path(data_dir).exists():
            messagebox.showerror("Error", "Training data directory not found")
            return
        
        self.train_btn.config(state='disabled')
        self.log_training("Starting training...")
        
        # Start training thread
        thread = threading.Thread(target=self.training_thread, daemon=True)
        thread.start()
        
    def training_thread(self):
        """Background thread for training"""
        try:
            train_model(
                data_dir=self.training_data_var.get(),
                output_dir='models',
                epochs=self.epochs_var.get(),
                batch_size=self.batch_size_var.get(),
                learning_rate=self.lr_var.get(),
                device=self.device_var.get()
            )
            
            self.log_training("✓ Training complete!")
            messagebox.showinfo("Success", "Model training completed!")
            
        except Exception as e:
            self.log_training(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Training failed: {str(e)}")
        finally:
            self.train_btn.config(state='normal')
            
    def log_training(self, message):
        """Add message to training log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.metrics_text.insert('end', f"[{timestamp}] {message}\n")
        self.metrics_text.see('end')


def launch_gui():
    """Launch the GUI application"""
    root = tk.Tk()
    app = DriveOSGUI(root)
    root.mainloop()


if __name__ == '__main__':
    launch_gui()
