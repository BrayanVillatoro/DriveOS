"""
DriveOS GUI Application
Modern interface for racing line analysis with live preview, file upload, and training controls
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import cv2
import os
from PIL import Image, ImageTk
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import time

from .inference import BatchProcessor
from .train import train_model


class DriveOSGUI:
    """Main GUI application for DriveOS"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("DriveOS - AI Racing Line Analyzer")
        self.root.geometry("1200x800")
        # Start maximized
        self.root.state('zoomed')
        
        # Set window icon with absolute path (for both window and taskbar)
        try:
            from pathlib import Path
            icon_path = Path(__file__).parent.parent / 'launchers' / 'DriveOS.ico'
            if icon_path.exists():
                self.root.iconbitmap(str(icon_path))
                # For Windows taskbar grouping
                import ctypes
                myappid = 'BrayanVillatoro.DriveOS.RacingLineAnalyzer.1.0'
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception as e:
            print(f"Could not set icon: {e}")
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # VS Code Dark Theme Colors
        self.colors = {
            'bg': '#1e1e1e',           # VS Code editor background
            'bg_light': '#252526',     # VS Code sidebar
            'card': '#2d2d30',         # Card/panel background
            'card_hover': '#3e3e42',   # Hover state
            'fg': '#cccccc',           # Main text color
            'fg_dim': '#858585',       # Dimmed text
            'accent': '#007acc',       # VS Code blue
            'accent_light': '#1c97ea', # Light blue
            'success': '#4ec9b0',      # Teal/cyan
            'warning': '#ce9178',      # Orange
            'error': '#f48771',        # Light red
            'purple': '#c586c0',       # Purple
            'gradient_start': '#007acc',
            'gradient_end': '#4ec9b0',
            'info_bg': '#1a3a52'       # Info banner background (dark blue)
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
        # Precomputed optimized spline (pixel coords Nx2)
        self.precomputed_pts_px = None
        self.available_cameras = []
        
        # Create main layout
        self.create_layout()
        
    def setup_styles(self):
        """Configure modern custom styles"""
        # Notebook/Tab styling
        self.style.configure('TNotebook', background=self.colors['bg'], borderwidth=0, tabmargins=[0, 0, 0, 0])
        self.style.configure('TNotebook.Tab', 
                           background=self.colors['card'],
                           foreground=self.colors['fg'],
                           padding=[20, 10],
                           font=('Segoe UI', 10),
                           borderwidth=0,
                           focuscolor='none')
        self.style.map('TNotebook.Tab',
                      background=[('selected', self.colors['accent']), ('!selected', self.colors['card'])],
                      foreground=[('selected', '#ffffff'), ('!selected', self.colors['fg'])],
                      padding=[('selected', [20, 10]), ('!selected', [20, 10])],
                      borderwidth=[('selected', 0), ('!selected', 0)],
                      expand=[('selected', [0, 0, 0, 0]), ('!selected', [0, 0, 0, 0])])
        
        # Title and text styles
        self.style.configure('Title.TLabel', 
                           font=('Segoe UI', 28, 'bold'),
                           foreground=self.colors['fg'],
                           background=self.colors['bg'])
        self.style.configure('Subtitle.TLabel',
                           font=('Segoe UI', 11),
                           foreground=self.colors['fg_dim'],
                           background=self.colors['bg'])
        self.style.configure('Heading.TLabel',
                           font=('Segoe UI', 14, 'bold'),
                           foreground=self.colors['fg'],
                           background=self.colors['card'])
        self.style.configure('Info.TLabel',
                           font=('Segoe UI', 10),
                           foreground=self.colors['fg_dim'],
                           background=self.colors['card'])
        self.style.configure('Success.TLabel',
                           font=('Segoe UI', 10, 'bold'),
                           foreground=self.colors['success'],
                           background=self.colors['card'])
        
        # Frame styles
        self.style.configure('Card.TFrame',
                           background=self.colors['card'])
        self.style.configure('Dark.TFrame',
                           background=self.colors['bg_light'])
        
        # Button styles
        self.style.configure('Accent.TButton',
                           font=('Segoe UI', 10),
                           padding=10,
                           background=self.colors['accent'],
                           foreground='white')
        self.style.map('Accent.TButton',
                      background=[('active', self.colors['accent_light'])])
        
        self.style.configure('Action.TButton',
                           font=('Segoe UI', 11, 'bold'),
                           padding=10)
        
        # Progressbar styling
        self.style.configure('TProgressbar',
                           background=self.colors['success'],
                           troughcolor=self.colors['bg_light'],
                           borderwidth=0,
                           thickness=12)
        
        # Custom progress bar for analyze video
        self.style.configure('Analyze.Horizontal.TProgressbar',
                           background=self.colors['accent'],
                           troughcolor=self.colors['card'],
                           borderwidth=1,
                           bordercolor=self.colors['bg_light'],
                           thickness=16)
        
        # LabelFrame styling
        self.style.configure('TLabelframe',
                           background=self.colors['card'],
                           foreground=self.colors['fg'],
                           borderwidth=2,
                           relief='flat')
        self.style.configure('TLabelframe.Label',
                           font=('Segoe UI', 11, 'bold'),
                           foreground=self.colors['accent'],
                           background=self.colors['card'])
        
    def create_layout(self):
        """Create main application layout"""
        # Header with gradient effect
        header = tk.Frame(self.root, bg=self.colors['bg'], height=100)
        header.pack(fill='x', padx=0, pady=0)
        header.pack_propagate(False)
        
        # Title with icon
        title_frame = tk.Frame(header, bg=self.colors['bg'])
        title_frame.pack(pady=15, padx=30)
        
        title = ttk.Label(title_frame, text="🏁 DriveOS",
                         style='Title.TLabel')
        title.pack(side='left')
        
        subtitle = ttk.Label(title_frame, 
                            text="  AI Racing Line Analyzer",
                            font=('Segoe UI', 18),
                            foreground=self.colors['fg'],
                            background=self.colors['bg'])
        subtitle.pack(side='left', padx=(10, 0))
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        self.notebook.bind('<<NotebookTabChanged>>', self.on_tab_changed)
        
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
        
        # Tab 4: Annotation Tool
        self.annotation_tab = ttk.Frame(self.notebook, style='Card.TFrame')
        self.notebook.add(self.annotation_tab, text='  ✏️ Create Training Data  ')
        self.create_annotation_view()
    
    def on_tab_changed(self, event):
        """Handle tab change events"""
        current_tab = self.notebook.index(self.notebook.select())
        if current_tab == 1:  # Live Preview tab
            # Auto-detect cameras on first visit
            if not hasattr(self, '_cameras_detected'):
                self._cameras_detected = True
                self.root.after(500, self.detect_cameras)  # Delay to let UI render
        elif current_tab == 2:  # Training tab
            self.update_data_stats()
        
    def create_live_view(self):
        """Create live view interface"""
        # Source selection frame
        source_frame = ttk.LabelFrame(self.live_tab, text="Video Source", padding=25)
        source_frame.pack(fill='x', padx=30, pady=(20, 5))
        
        # Source type selector
        source_type_frame = tk.Frame(source_frame, bg=self.colors['card'])
        source_type_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(source_type_frame, text="Source Type:", 
                fg=self.colors['fg'], bg=self.colors['card'],
                font=('Segoe UI', 10)).pack(side='left', padx=(0, 10))
        
        self.source_type_var = tk.StringVar(value="camera")
        
        ttk.Radiobutton(source_type_frame, text="📷 Webcam/Camera", 
                       variable=self.source_type_var, value="camera",
                       command=self.update_source_controls).pack(side='left', padx=10)
        
        ttk.Radiobutton(source_type_frame, text="🖥️ Screen Capture", 
                       variable=self.source_type_var, value="screen",
                       command=self.update_source_controls).pack(side='left', padx=10)
        
        # Source-specific controls
        self.source_controls_frame = tk.Frame(source_frame, bg=self.colors['card'])
        self.source_controls_frame.pack(fill='x')
        
        # Camera controls
        self.camera_controls = tk.Frame(self.source_controls_frame, bg=self.colors['card'])
        tk.Label(self.camera_controls, text="Select Camera:", 
                fg=self.colors['fg'], bg=self.colors['card'],
                font=('Segoe UI', 10)).pack(side='left', padx=(0, 10))
        self.camera_var = tk.StringVar(value="No cameras detected")
        self.camera_combo = ttk.Combobox(self.camera_controls, 
                                         textvariable=self.camera_var,
                                         state='readonly', width=30)
        self.camera_combo.pack(side='left', padx=5)
        ttk.Button(self.camera_controls, text="🔄 Detect Cameras",
                  command=self.detect_cameras).pack(side='left', padx=5)
        
        # Screen capture controls
        self.screen_controls = tk.Frame(self.source_controls_frame, bg=self.colors['card'])
        tk.Label(self.screen_controls, text="Capture Mode:", 
                fg=self.colors['fg'], bg=self.colors['card'],
                font=('Segoe UI', 10)).pack(side='left', padx=(0, 10))
        self.screen_mode_var = tk.StringVar(value="fullscreen")
        screen_mode_combo = ttk.Combobox(self.screen_controls, 
                                        textvariable=self.screen_mode_var,
                                        values=['fullscreen', 'window'], 
                                        state='readonly', width=15)
        screen_mode_combo.pack(side='left', padx=5)
        tk.Label(self.screen_controls, text="(Window selection will appear on start)", 
                fg=self.colors['fg_dim'], bg=self.colors['card'],
                font=('Segoe UI', 9)).pack(side='left', padx=10)
        
        # Show initial controls
        self.camera_controls.pack(fill='x')
        
        # Video display area
        video_frame = ttk.LabelFrame(self.live_tab, text="Video Feed", padding=10)
        video_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Canvas for video
        self.live_canvas = tk.Canvas(video_frame, bg='#000000', 
                                     width=1280, height=720)
        self.live_canvas.pack()
        
        # Status label
        self.live_status = tk.Label(video_frame, text="Ready - Select a source to start",
                                   fg=self.colors['fg'], bg=self.colors['card'],
                                   font=('Segoe UI', 11))
        self.live_status.pack(pady=5)
        
        # Controls
        controls = tk.Frame(self.live_tab, bg=self.colors['bg'])
        controls.pack(fill='x', padx=20, pady=10)
        
        self.play_btn = ttk.Button(controls, text="▶ Start Processing",
                                   command=self.start_live_processing,
                                   state='normal')
        self.play_btn.pack(side='left', padx=5)
        
        self.stop_btn = ttk.Button(controls, text="⬛ Stop",
                                   command=self.stop_live_processing,
                                   state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        # Info panel
        info_frame = ttk.LabelFrame(self.live_tab, text="Real-time Statistics", padding=10)
        info_frame.pack(fill='x', padx=20, pady=10)
        
        self.fps_label = tk.Label(info_frame, text="FPS: --", 
                                 fg=self.colors['fg'], bg=self.colors['card'],
                                 font=('Segoe UI', 10))
        self.fps_label.pack(side='left', padx=20)
        
        self.frame_label = tk.Label(info_frame, text="Frame: --", 
                                    fg=self.colors['fg'], bg=self.colors['card'],
                                    font=('Segoe UI', 10))
        self.frame_label.pack(side='left', padx=20)
        
        self.confidence_label = ttk.Label(info_frame, text="Confidence: --", style='Info.TLabel')
        self.confidence_label.pack(side='left', padx=20)
        
    def create_upload_view(self):
        """Create upload and processing interface"""
        # Instructions card with modern styling
        instruction_frame = tk.Frame(self.upload_tab, bg=self.colors['bg_light'], 
                                    relief='flat', borderwidth=0)
        instruction_frame.pack(fill='x', padx=30, pady=20)
        
        # Icon header
        icon_label = tk.Label(instruction_frame,
                             text="📖",
                             font=('Segoe UI', 24),
                             bg=self.colors['bg_light'],
                             fg=self.colors['success'])
        icon_label.pack(pady=(15, 5))
        
        instruction_text = tk.Label(instruction_frame,
                                   text="Quick Start Guide\n\n"
                                        "① Select your racing video\n"
                                        "② Choose output location\n"
                                        "③ Choose CPU or GPU processing\n"
                                        "④ Click Analyze to get your video with the optimal racing line!",
                                   font=('Segoe UI', 11),
                                   bg=self.colors['bg_light'],
                                   fg=self.colors['fg'],
                                   justify='center')
        instruction_text.pack(padx=20, pady=(0, 15))
        
        # Top row container for Steps 1 and 2
        top_row = tk.Frame(self.upload_tab, bg=self.colors['card'])
        top_row.pack(fill='x', padx=30, pady=10)
        
        # File selection card (LEFT)
        file_frame = ttk.LabelFrame(top_row, text="Step 1: Select Racing Video", padding=15)
        file_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        self.file_path_var = tk.StringVar(value="No video selected")
        
        path_display = tk.Label(file_frame, textvariable=self.file_path_var,
                               font=('Segoe UI', 9),
                               fg=self.colors['fg'],
                               bg=self.colors['card'],
                               anchor='w',
                               wraplength=400)
        path_display.pack(fill='x', pady=(0, 10))
        
        ttk.Button(file_frame, text="📁 Select Video",
                  command=self.browse_video,
                  style='Action.TButton').pack()
        
        # Output settings (RIGHT)
        output_frame = ttk.LabelFrame(top_row, text="Step 2: Output Location", padding=15)
        output_frame.pack(side='left', fill='both', expand=True, padx=(5, 0))
        
        self.output_dir_var = tk.StringVar(value=str(Path.home() / "Videos"))
        
        output_display = tk.Label(output_frame,
                                 text=f"Save to:\n{self.output_dir_var.get()}",
                                 font=('Segoe UI', 9),
                                 fg=self.colors['fg'],
                                 bg=self.colors['card'],
                                 anchor='w',
                                 wraplength=400)
        output_display.pack(fill='x', pady=(0, 10))
        
        ttk.Button(output_frame, text="📂 Change Folder",
                  command=self.browse_output,
                  style='Action.TButton').pack()
        
        # Device selection
        device_frame = ttk.LabelFrame(self.upload_tab, text="Step 3: Choose Processing Device", padding=25)
        device_frame.pack(fill='x', padx=30, pady=10)
        
        tk.Label(device_frame, text="Select which hardware to use for AI processing:",
                font=('Segoe UI', 10),
                fg=self.colors['fg'],
                bg=self.colors['card'],
                anchor='w').pack(fill='x', pady=(0, 15))
        
        self.analyze_device_var = tk.StringVar(value="auto")
        
        device_radio_frame = tk.Frame(device_frame, bg=self.colors['card'])
        device_radio_frame.pack(anchor='w')
        
        ttk.Radiobutton(device_radio_frame, text="Auto (Recommended - GPU if available, otherwise CPU)", 
                       variable=self.analyze_device_var, value="auto").pack(anchor='w', pady=3)
        ttk.Radiobutton(device_radio_frame, text="CPU Only (Slower but works on all computers)", 
                       variable=self.analyze_device_var, value="cpu").pack(anchor='w', pady=3)
        ttk.Radiobutton(device_radio_frame, text="GPU - CUDA (10-20x faster, requires NVIDIA GPU)", 
                       variable=self.analyze_device_var, value="cuda").pack(anchor='w', pady=3)
        
        # Initialize stop flag for batch processing
        self.stop_batch_analysis = False
        
        # Process button - LARGE and prominent
        process_frame = tk.Frame(self.upload_tab, bg=self.colors['card'])
        process_frame.pack(fill='x', padx=30, pady=30)
        
        button_container = tk.Frame(process_frame, bg=self.colors['card'])
        button_container.pack()
        
        self.process_btn = tk.Button(button_container,
                                     text="▶  ANALYZE VIDEO",
                                     command=self.start_batch_processing,
                                     font=('Segoe UI', 18, 'bold'),
                                     bg=self.colors['accent'],
                                     fg='white',
                                     activebackground=self.colors['accent_light'],
                                     activeforeground='white',
                                     relief='flat',
                                     borderwidth=0,
                                     padx=50,
                                     pady=25,
                                     cursor='hand2',
                                     state='disabled')
        self.process_btn.pack(side='left', padx=5)
        
        self.analyze_stop_btn = tk.Button(button_container,
                                          text="⏹  STOP",
                                          command=self.stop_batch_processing,
                                          font=('Segoe UI', 18, 'bold'),
                                          bg='#d13438',
                                          fg='white',
                                          activebackground='#a02528',
                                          activeforeground='white',
                                          relief='flat',
                                          borderwidth=0,
                                          padx=50,
                                          pady=25,
                                          cursor='hand2',
                                          state='disabled')
        self.analyze_stop_btn.pack(side='left', padx=5)
        
        # Progress - Enhanced with more details
        progress_frame = ttk.LabelFrame(self.upload_tab, text="Processing Status", padding=20)
        progress_frame.pack(fill='both', expand=True, padx=30, pady=(0, 20))
        
        # Main progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                           maximum=100, style='Analyze.Horizontal.TProgressbar')
        self.progress_bar.pack(fill='x', pady=(10, 5))
        
        # Status message
        self.progress_label = tk.Label(progress_frame, 
                                      text="Ready - Select a video to begin",
                                      font=('Segoe UI', 11, 'bold'),
                                      fg=self.colors['fg'],
                                      bg=self.colors['card'])
        self.progress_label.pack(pady=5)
        
        # Detailed status grid
        status_grid = tk.Frame(progress_frame, bg=self.colors['card'])
        status_grid.pack(fill='x', pady=10)
        
        # Left column
        left_col = tk.Frame(status_grid, bg=self.colors['card'])
        left_col.pack(side='left', fill='both', expand=True, padx=10)
        
        self.frame_status_label = tk.Label(left_col,
                                          text="📊 Frames: --/--",
                                          font=('Segoe UI', 10),
                                          fg=self.colors['fg'],
                                          bg=self.colors['card'],
                                          anchor='w')
        self.frame_status_label.pack(fill='x', pady=2)
        
        self.fps_status_label = tk.Label(left_col,
                                         text="⚡ Processing Speed: -- FPS",
                                         font=('Segoe UI', 10),
                                         fg=self.colors['fg'],
                                         bg=self.colors['card'],
                                         anchor='w')
        self.fps_status_label.pack(fill='x', pady=2)
        
        self.time_status_label = tk.Label(left_col,
                                          text="⏱️ Time: --:-- / Est. --:--",
                                          font=('Segoe UI', 10),
                                          fg=self.colors['fg'],
                                          bg=self.colors['card'],
                                          anchor='w')
        self.time_status_label.pack(fill='x', pady=2)
        
        # Right column
        right_col = tk.Frame(status_grid, bg=self.colors['card'])
        right_col.pack(side='left', fill='both', expand=True, padx=10)
        
        self.model_info_label = tk.Label(right_col,
                                        text="📦 Model: models/racing_line_model.pth",
                                        font=('Segoe UI', 10),
                                        fg=self.colors['accent'],
                                        bg=self.colors['card'],
                                        anchor='w')
        self.model_info_label.pack(fill='x', pady=2)
        
        self.device_status_label = tk.Label(right_col,
                                           text="🖥️ Device: Not started",
                                           font=('Segoe UI', 10),
                                           fg=self.colors['accent'],
                                           bg=self.colors['card'],
                                           anchor='w')
        self.device_status_label.pack(fill='x', pady=2)
        
        self.inference_status_label = tk.Label(right_col,
                                              text="🧠 Inference: -- ms/frame",
                                              font=('Segoe UI', 10),
                                              fg=self.colors['fg'],
                                              bg=self.colors['card'],
                                              anchor='w')
        self.inference_status_label.pack(fill='x', pady=2)

        
    def create_training_view(self):
        """Create training interface with adjustable parameters"""
        # Initialize training data directory variable
        self.training_data_var = tk.StringVar(value="data/training")
        
        # Workflow explanation banner
        info_frame = tk.Frame(self.training_tab, bg=self.colors['info_bg'], relief='solid', bd=1)
        info_frame.pack(fill='x', padx=30, pady=(15, 10))
        
        tk.Label(info_frame, 
                text="💡 How Training Works",
                font=('Segoe UI', 12, 'bold'),
                fg=self.colors['accent'],
                bg=self.colors['info_bg']).pack(anchor='w', padx=15, pady=(10, 5))
        
        workflow_text = (
            "1. CREATE DATA: Use 'Create Training Data' tab to annotate racing lines on video frames\n"
            "2. TRAIN MODEL: Use the data to train the AI (this tab - takes 30-60 minutes)\n"
            "3. USE MODEL: The trained model (racing_line_model.pth) will be used to analyze new videos\n\n"
            "🎯 Current Status: The model in 'models/racing_line_model.pth' is what the Analyze Video tab uses.\n"
            "Training will REPLACE this model with a new one based on your custom data!"
        )
        
        tk.Label(info_frame,
                text=workflow_text,
                font=('Segoe UI', 10),
                fg=self.colors['fg'],
                bg=self.colors['info_bg'],
                justify='left',
                anchor='w').pack(anchor='w', padx=15, pady=(0, 10))
        
        # STEP 1: Generate Training Data
        step1_frame = ttk.LabelFrame(self.training_tab, text="① STEP 1: Prepare Training Data", padding=25)
        step1_frame.pack(fill='x', padx=30, pady=(10, 5))
        
        tk.Label(step1_frame, text="Your annotated data from 'Create Training Data' tab should be in the folder below.\nOr use 'Generate Training Data' to extract racing lines automatically from a video.",
                font=('Segoe UI', 10), fg=self.colors['fg'], bg=self.colors['card'],
                anchor='w', justify='left').pack(fill='x', pady=(0, 15))
        
        # Action buttons
        button_frame1 = tk.Frame(step1_frame, bg=self.colors['card'])
        button_frame1.pack(fill='x')
        
        ttk.Button(button_frame1, text="🎬 Generate Training Data",
                  command=self.generate_training_data,
                  style='Accent.TButton').pack(side='left', padx=5)
        
        ttk.Button(button_frame1, text="🧹 Refine/Clean Data",
                  command=self.refine_training_data).pack(side='left', padx=5)
        
        self.data_stats_label = ttk.Label(step1_frame, text="No training data found",
                                          style='Info.TLabel', foreground=self.colors['warning'])
        self.data_stats_label.pack(pady=(10, 0), anchor='w')
        
        # Create horizontal container for steps 2 and 3
        steps_container = tk.Frame(self.training_tab, bg=self.colors['card'])
        steps_container.pack(fill='x', padx=30, pady=5)
        
        # STEP 2: Configure Training Parameters (LEFT SIDE)
        step2_frame = ttk.LabelFrame(steps_container, text="② STEP 2: Parameters", padding=15)
        step2_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Training parameters grid
        params_frame = tk.Frame(step2_frame, bg=self.colors['card'])
        params_frame.pack(fill='x')
        
        # Epochs
        epochs_label_frame = tk.Frame(params_frame, bg=self.colors['card'])
        epochs_label_frame.grid(row=0, column=0, sticky='w', pady=10, padx=(0, 20))
        tk.Label(epochs_label_frame, text="Epochs:",
                fg=self.colors['fg'],
                bg=self.colors['card'],
                font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        tk.Label(epochs_label_frame, text="(Training iterations)",
                fg=self.colors['fg_dim'],
                bg=self.colors['card'],
                font=('Segoe UI', 8)).pack(anchor='w')
        
        self.epochs_var = tk.IntVar(value=20)
        ttk.Scale(params_frame, from_=5, to=100, variable=self.epochs_var,
                 orient='horizontal', length=250).grid(row=0, column=1, padx=15)
        self.epochs_label = tk.Label(params_frame, text="20", width=6,
                fg=self.colors['accent'],
                bg=self.colors['card'],
                font=('Segoe UI', 12, 'bold'), anchor='center')
        self.epochs_label.grid(row=0, column=2, padx=15)
        tk.Label(params_frame, text="More = Better quality (slower)\nLess = Faster (lower quality)",
                fg=self.colors['fg_dim'],
                bg=self.colors['card'],
                font=('Segoe UI', 8), justify='left').grid(row=0, column=3, sticky='w', padx=5)
        
        # Update label when slider moves
        def update_epochs_label(*args):
            self.epochs_label.config(text=str(self.epochs_var.get()))
        self.epochs_var.trace_add('write', update_epochs_label)
        
        # Batch size
        batch_label_frame = tk.Frame(params_frame, bg=self.colors['card'])
        batch_label_frame.grid(row=1, column=0, sticky='w', pady=10, padx=(0, 20))
        tk.Label(batch_label_frame, text="Batch Size:",
                fg=self.colors['fg'],
                bg=self.colors['card'],
                font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        tk.Label(batch_label_frame, text="(Images per step)",
                fg=self.colors['fg_dim'],
                bg=self.colors['card'],
                font=('Segoe UI', 8)).pack(anchor='w')
        
        self.batch_size_var = tk.IntVar(value=2)
        ttk.Spinbox(params_frame, from_=2, to=16, textvariable=self.batch_size_var,
                   width=20).grid(row=1, column=1, padx=15, sticky='w')
        tk.Label(params_frame, text="",
                fg=self.colors['fg'],
                bg=self.colors['card'],
                font=('Segoe UI', 12, 'bold')).grid(row=1, column=2, padx=15)
        tk.Label(params_frame, text="Higher = Uses more memory\nLower = Safer for CPU training",
                fg=self.colors['fg_dim'],
                bg=self.colors['card'],
                font=('Segoe UI', 8), justify='left').grid(row=1, column=3, sticky='w', padx=5)
        
        # Learning rate
        lr_label_frame = tk.Frame(params_frame, bg=self.colors['card'])
        lr_label_frame.grid(row=2, column=0, sticky='w', pady=10, padx=(0, 20))
        tk.Label(lr_label_frame, text="Learning Rate:",
                fg=self.colors['fg'],
                bg=self.colors['card'],
                font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        tk.Label(lr_label_frame, text="(Step size)",
                fg=self.colors['fg_dim'],
                bg=self.colors['card'],
                font=('Segoe UI', 8)).pack(anchor='w')
        
        self.lr_var = tk.DoubleVar(value=0.001)
        ttk.Entry(params_frame, textvariable=self.lr_var,
                 width=20).grid(row=2, column=1, padx=15, sticky='w')
        tk.Label(params_frame, text="",
                fg=self.colors['fg'],
                bg=self.colors['card'],
                font=('Segoe UI', 12, 'bold')).grid(row=2, column=2, padx=15)
        tk.Label(params_frame, text="0.001 recommended\n(Don't change unless experienced)",
                fg=self.colors['fg_dim'],
                bg=self.colors['card'],
                font=('Segoe UI', 8), justify='left').grid(row=2, column=3, sticky='w', padx=5)
        
        # STEP 3: Hardware Configuration (RIGHT SIDE)
        step3_frame = ttk.LabelFrame(steps_container, text="③ STEP 3: Hardware", padding=15)
        step3_frame.pack(side='left', fill='both', expand=True, padx=(5, 0))
        
        # Device selection
        device_container = tk.Frame(step3_frame, bg=self.colors['card'])
        device_container.pack(fill='x', pady=(0, 10))
        
        tk.Label(device_container, text="Device:",
                fg=self.colors['fg'],
                bg=self.colors['card'],
                font=('Segoe UI', 10)).pack(anchor='w', pady=(0, 5))
        
        self.device_var = tk.StringVar(value="cpu")
        device_btns = tk.Frame(device_container, bg=self.colors['card'])
        device_btns.pack(anchor='w', padx=10)
        ttk.Radiobutton(device_btns, text="💻 CPU", variable=self.device_var, 
                       value="cpu", command=self.update_hardware_config).pack(side='left', padx=5)
        ttk.Radiobutton(device_btns, text="🎮 GPU (CUDA)", variable=self.device_var, 
                       value="cuda", command=self.update_hardware_config).pack(side='left', padx=5)
        
        # CPU cores selection
        cores_container = tk.Frame(step3_frame, bg=self.colors['card'])
        cores_container.pack(fill='x')
        
        self.max_cpu_cores = os.cpu_count() or 8
        
        cpu_label_frame = tk.Frame(cores_container, bg=self.colors['card'])
        cpu_label_frame.pack(anchor='w', pady=(0, 5))
        
        self.cpu_cores_label = tk.Label(cpu_label_frame, 
                                        text=f"CPU Threads (Your system: {self.max_cpu_cores}):",
                                        fg=self.colors['fg'],
                                        bg=self.colors['card'],
                                        font=('Segoe UI', 10, 'bold'))
        self.cpu_cores_label.pack(anchor='w')
        
        tk.Label(cpu_label_frame, 
                text="Higher = Faster training (uses more CPU)",
                fg=self.colors['fg_dim'],
                bg=self.colors['card'],
                font=('Segoe UI', 8)).pack(anchor='w')
        
        self.cpu_cores_var = tk.IntVar(value=min(self.max_cpu_cores, 8))
        
        cores_control_frame = tk.Frame(cores_container, bg=self.colors['card'])
        cores_control_frame.pack(anchor='w', padx=10)
        
        self.cpu_cores_scale = ttk.Scale(cores_control_frame, from_=1, to=self.max_cpu_cores, 
                                         variable=self.cpu_cores_var,
                                         orient='horizontal', length=200)
        self.cpu_cores_scale.pack(side='left', padx=5)
        self.cpu_cores_value_label = tk.Label(cores_control_frame, text=str(self.cpu_cores_var.get()), width=6,
                                              fg=self.colors['accent'],
                                              bg=self.colors['card'],
                                              font=('Segoe UI', 12, 'bold'), anchor='center')
        self.cpu_cores_value_label.pack(side='left', padx=5)
        
        # Update label when slider moves
        def update_cpu_cores_label(*args):
            self.cpu_cores_value_label.config(text=str(self.cpu_cores_var.get()))
        self.cpu_cores_var.trace_add('write', update_cpu_cores_label)
        
        # STEP 4: Start Training
        step4_frame = ttk.LabelFrame(self.training_tab, text="④ STEP 4: Train the Model", padding=25)
        step4_frame.pack(fill='x', padx=30, pady=5)
        
        tk.Label(step4_frame, text="Verify your training data directory below (should contain 'images' and 'masks' folders), then click START TRAINING.\nTraining takes 30-60 minutes and creates a NEW racing_line_model.pth file.",
                font=('Segoe UI', 10), fg=self.colors['fg'], bg=self.colors['card'],
                anchor='w', justify='left').pack(fill='x', pady=(0, 15))
        
        # Training data directory selection
        dir_frame = tk.Frame(step4_frame, bg=self.colors['card'])
        dir_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(dir_frame, text="Training Data:",
                fg=self.colors['fg'],
                bg=self.colors['card'],
                font=('Segoe UI', 10, 'bold')).pack(side='left', padx=(0, 15))
        
        data_entry = tk.Entry(dir_frame, textvariable=self.training_data_var, width=50,
                             bg='#3c3c3c', fg='#ffffff', font=('Segoe UI', 9),
                             relief='solid', bd=1, insertbackground='#ffffff',
                             selectbackground=self.colors['accent'], selectforeground='#ffffff')
        data_entry.pack(side='left', padx=5, fill='x', expand=True)
        
        ttk.Button(dir_frame, text="Browse...",
                  command=self.browse_training_data).pack(side='left', padx=5)
        
        ttk.Button(dir_frame, text="🔄 Refresh",
                  command=self.update_data_stats).pack(side='left', padx=5)
        
        # Initialize training stop flag
        self.stop_training = False
        
        # Train buttons container
        train_btn_container = tk.Frame(step4_frame, bg=self.colors['card'])
        train_btn_container.pack(pady=(0, 10))
        
        self.train_btn = ttk.Button(train_btn_container, text="🚀 Start Training",
                                    command=self.start_training,
                                    style='Accent.TButton')
        self.train_btn.pack(side='left', padx=5)
        
        self.train_stop_btn = tk.Button(train_btn_container,
                                        text="⏹ Stop Training",
                                        command=self.stop_training_process,
                                        font=('Segoe UI', 10, 'bold'),
                                        bg='#d13438',
                                        fg='white',
                                        activebackground='#a02528',
                                        activeforeground='white',
                                        relief='flat',
                                        padx=20,
                                        pady=10,
                                        cursor='hand2',
                                        state='disabled')
        self.train_stop_btn.pack(side='left', padx=5)
        
        # Progress section
        self.training_progress_var = tk.DoubleVar()
        self.training_progress_bar = ttk.Progressbar(step4_frame,
                                                     variable=self.training_progress_var,
                                                     maximum=100, length=800)
        self.training_progress_bar.pack(fill='x', pady=5)
        
        self.training_status_label = ttk.Label(step4_frame,
                                              text="Not started - Complete steps above first",
                                              style='Info.TLabel')
        self.training_status_label.pack()
        
        # Training metrics
        metrics_frame = ttk.LabelFrame(self.training_tab, text="Training Output & Metrics", padding=10)
        metrics_frame.pack(fill='both', expand=True, padx=20, pady=(5, 10))
        
        # Create text widget with scrollbar
        text_container = tk.Frame(metrics_frame, bg=self.colors['bg_light'])
        text_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(text_container)
        scrollbar.pack(side='right', fill='y')
        
        self.metrics_text = tk.Text(text_container, height=15, 
                                   bg=self.colors['bg_light'], 
                                   fg=self.colors['success'],
                                   font=('Consolas', 9),
                                   borderwidth=0,
                                   relief='flat',
                                   insertbackground=self.colors['success'],
                                   yscrollcommand=scrollbar.set,
                                   wrap='word')
        self.metrics_text.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.metrics_text.yview)
        
        # Check for existing training data on startup
        self.root.after(500, self.update_data_stats)
        
    def update_hardware_config(self):
        """Update hardware configuration controls based on device selection"""
        if self.device_var.get() == "cpu":
            # Show CPU cores controls
            self.cpu_cores_label.pack(anchor='w', pady=(0, 5))
            self.cpu_cores_scale.pack(side='left', padx=5)
            self.cpu_cores_value_label.pack(side='left', padx=5)
        else:
            # Hide CPU cores controls for GPU
            self.cpu_cores_label.pack_forget()
            self.cpu_cores_scale.pack_forget()
            self.cpu_cores_value_label.pack_forget()
    
    def create_annotation_view(self):
        """Create annotation tool interface"""
        # Introduction section
        intro_frame = ttk.LabelFrame(self.annotation_tab, text="📝 Create Training Data Interactively", padding=25)
        intro_frame.pack(fill='x', padx=30, pady=(20, 10))
        
        intro_text = """This tool allows you to manually annotate racing videos to create high-quality training data.

You can draw:
• The ideal racing line through corners
• Track boundaries (left and right edges)
• Track surface areas

The tool will automatically generate labeled images and masks for training."""
        
        tk.Label(intro_frame, text=intro_text,
                font=('Segoe UI', 10), fg=self.colors['fg'], bg=self.colors['card'],
                justify='left', anchor='w').pack(fill='x')
        
        # Video selection
        video_frame = ttk.LabelFrame(self.annotation_tab, text="Select Video to Annotate", padding=25)
        video_frame.pack(fill='x', padx=30, pady=10)
        
        self.annotation_video_var = tk.StringVar(value="No video selected")
        
        file_frame = tk.Frame(video_frame, bg=self.colors['card'])
        file_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(file_frame, text="Video File:",
                fg=self.colors['fg'],
                bg=self.colors['card'],
                font=('Segoe UI', 10, 'bold')).pack(side='left', padx=(0, 15))
        
        video_entry = tk.Entry(file_frame, textvariable=self.annotation_video_var, width=50,
                              bg='#3c3c3c', fg='#ffffff', font=('Segoe UI', 9),
                              relief='solid', bd=1, insertbackground='#ffffff',
                              selectbackground=self.colors['accent'], selectforeground='#ffffff')
        video_entry.pack(side='left', padx=5, fill='x', expand=True)
        
        ttk.Button(file_frame, text="Browse Video...",
                  command=self.browse_annotation_video).pack(side='left', padx=5)
        
        # Output directory
        output_frame = tk.Frame(video_frame, bg=self.colors['card'])
        output_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(output_frame, text="Save To:",
                fg=self.colors['fg'],
                bg=self.colors['card'],
                font=('Segoe UI', 10, 'bold')).pack(side='left', padx=(0, 15))
        
        self.annotation_output_var = tk.StringVar(value="data/user_annotations")
        
        output_entry = tk.Entry(output_frame, textvariable=self.annotation_output_var, width=50,
                               bg='#3c3c3c', fg='#ffffff', font=('Segoe UI', 9),
                               relief='solid', bd=1, insertbackground='#ffffff',
                               selectbackground=self.colors['accent'], selectforeground='#ffffff')
        output_entry.pack(side='left', padx=5, fill='x', expand=True)
        
        ttk.Button(output_frame, text="Browse...",
                  command=self.browse_annotation_output).pack(side='left', padx=5)
        
        # Launch button
        self.launch_annotation_btn = ttk.Button(video_frame, text="🎨 Launch Annotation Tool",
                                               command=self.launch_annotation_tool,
                                               style='Accent.TButton',
                                               state='disabled')
        self.launch_annotation_btn.pack(pady=(10, 0))
        
        # Instructions
        instructions_frame = ttk.LabelFrame(self.annotation_tab, text="How to Use", padding=25)
        instructions_frame.pack(fill='both', expand=True, padx=30, pady=10)
        
        instructions = """⌨️ Keyboard Controls:
1, 2, 3    - Switch between Racing Line, Left Boundary, Right Boundary modes
SPACE      - Save current frame and advance to next
C          - Clear current annotations
N          - Next frame (skip forward 10 frames)
B          - Back frame (skip backward 10 frames)
Q          - Quit annotation tool

🖱️ Mouse Controls:
Left Click + Drag  - Draw the line for current mode
Right Click        - Undo last point

💡 Tips:
• Start with the racing line (yellow) - this is the most important
• Then mark left boundary (red) and right boundary (blue)
• The tool will automatically create segmentation masks
• Annotate frames with different track conditions and corner types
• Aim for at least 50-100 annotated frames for good results"""
        
        tk.Label(instructions_frame, text=instructions,
                font=('Consolas', 9), fg=self.colors['fg'], bg=self.colors['card'],
                justify='left', anchor='w').pack(fill='both', expand=True)
    
    # Live View Methods
    def update_source_controls(self):
        """Update source controls based on selected type"""
        # Hide all controls
        self.camera_controls.pack_forget()
        self.screen_controls.pack_forget()
        
        # Show relevant controls
        source_type = self.source_type_var.get()
        if source_type == "camera":
            self.camera_controls.pack(fill='x')
            self.live_status.config(text="Ready to start camera capture")
            self.play_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
        elif source_type == "screen":
            self.screen_controls.pack(fill='x')
            self.live_status.config(text="Ready to start screen capture")
            self.play_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
    
    def detect_cameras(self):
        """Detect available cameras with improved reliability"""
        # Run detection in background thread to prevent UI freezing
        thread = threading.Thread(target=self._detect_cameras_thread, daemon=True)
        thread.start()
    
    def _detect_cameras_thread(self):
        """Background thread for camera detection"""
        self.live_status.config(text="🔍 Detecting cameras...")
        self.root.update()
        
        self.available_cameras = []
        camera_names = []
        
        # Use faster backends - DirectShow can hang
        import platform
        backends = []
        if platform.system() == 'Windows':
            backends = [
                (cv2.CAP_MSMF, "Media Foundation"),  # Faster, more reliable
                (cv2.CAP_DSHOW, "DirectShow"),  # Slower but compatible
            ]
        else:
            backends = [
                (cv2.CAP_V4L2, "Video4Linux"),  # Linux
                (cv2.CAP_AVFOUNDATION, "AVFoundation"),  # macOS
                (cv2.CAP_ANY, "Auto")  # Fallback
            ]
        
        detected = set()  # Avoid duplicates
        
        # Try each backend with timeout protection
        for backend, backend_name in backends:
            self.live_status.config(text=f"Trying {backend_name}...")
            self.root.update()
            
            # Test camera indices 0-3 (most systems have 0-1 cameras)
            for i in range(4):
                if i in detected:
                    continue
                
                # Skip if we already found 2 cameras (most common case)
                if len(detected) >= 2 and backend != backends[0][0]:
                    break
                    
                cap = None
                try:
                    import time
                    start_time = time.time()
                    
                    # Try to open with specific backend (with timeout)
                    cap = cv2.VideoCapture(i, backend)
                    
                    # Set shorter timeout
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 1500)  # 1.5 second timeout
                    
                    # Check if open timed out
                    if time.time() - start_time > 2:
                        if cap:
                            cap.release()
                        continue
                    
                    if cap.isOpened():
                        # Single quick read attempt
                        ret = False
                        frame = None
                        
                        read_start = time.time()
                        ret, frame = cap.read()
                        
                        # Timeout after 1 second
                        if time.time() - read_start > 1:
                            cap.release()
                            continue
                        
                        if ret and frame is not None and frame.size > 0:
                            # Get camera properties
                            try:
                                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                fps = int(cap.get(cv2.CAP_PROP_FPS))
                                
                                # Try to get camera name (Windows only)
                                camera_name = f"Camera {i}"
                                if backend == cv2.CAP_DSHOW:
                                    try:
                                        # Windows can sometimes get device name
                                        backend_name_prop = cap.getBackendName()
                                        if backend_name_prop:
                                            camera_name = f"{backend_name_prop} {i}"
                                    except:
                                        pass
                                
                                # Format display name
                                display_name = f"{camera_name} ({width}x{height}"
                                if fps > 0:
                                    display_name += f" @ {fps}fps"
                                display_name += ")"
                                
                                self.available_cameras.append(i)
                                camera_names.append(display_name)
                                detected.add(i)
                                
                                self.live_status.config(text=f"✓ {display_name}")
                                self.root.update()
                            except Exception:
                                # Fallback if properties fail
                                self.available_cameras.append(i)
                                camera_names.append(f"Camera {i}")
                                detected.add(i)
                        
                        cap.release()
                        
                except Exception:
                    # Silently continue on error
                    pass
                finally:
                    if cap is not None:
                        try:
                            cap.release()
                        except:
                            pass
            
            # If we found cameras, don't try other backends
            if len(detected) > 0:
                break
        
        # Update UI with results
        if self.available_cameras:
            self.camera_combo['values'] = camera_names
            self.camera_combo.current(0)
            self.live_status.config(text=f"✓ Found {len(self.available_cameras)} camera(s)")
            
            camera_list = "\n".join([f"  • {name}" for name in camera_names])
            messagebox.showinfo(
                "Camera Detection Complete", 
                f"Successfully detected {len(self.available_cameras)} camera(s):\n\n{camera_list}\n\n"
                f"Select a camera and click 'Start Processing' to begin."
            )
        else:
            self.camera_combo['values'] = ["No cameras detected"]
            self.camera_var.set("No cameras detected")
            self.live_status.config(text="❌ No cameras found")
            
            messagebox.showwarning(
                "No Cameras Detected",
                "No cameras were detected on your system.\n\n"
                "Troubleshooting steps:\n"
                "1. Make sure your camera is connected\n"
                "2. Check if camera works in other apps (Camera app, Zoom, etc.)\n"
                "3. Grant camera permissions to Python/this app\n"
                "4. Try unplugging and replugging the camera\n"
                "5. Restart the application\n\n"
                "On Windows: Check Device Manager for camera issues\n"
                "On macOS: Check System Preferences > Security & Privacy > Camera\n"
                "On Linux: Check camera permissions and v4l2 drivers"
            )
    
    def start_live_processing(self):
        """Start live video processing"""
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
        self.play_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.live_status.config(text="Stopped")
        
        # Clear canvas
        self.live_canvas.delete('all')
        
        # Release capture if exists
        if hasattr(self, 'capture'):
            if self.capture is not None:
                self.capture.release()
                self.capture = None
    def live_processing_thread(self):
        """Background thread for live processing"""
        cap = None
        try:
            # Initialize inference engine (uses improved detection from inference.py)
            from .inference import InferenceEngine
            engine = InferenceEngine('models/racing_line_model.pth')
            
            source_type = self.source_type_var.get()
            
            if source_type == "camera":
                # Get selected camera index
                if not self.available_cameras:
                    self.live_status.config(text="❌ No cameras detected. Click 'Detect Cameras' first.")
                    self.stop_processing = True
                    self.play_btn.config(state='normal')
                    self.stop_btn.config(state='disabled')
                    return
                
                selected_idx = self.camera_combo.current()
                if selected_idx < 0:
                    self.live_status.config(text="❌ Please select a camera")
                    self.stop_processing = True
                    self.play_btn.config(state='normal')
                    self.stop_btn.config(state='disabled')
                    return
                
                camera_index = self.available_cameras[selected_idx]
                
                # Use Media Foundation for faster camera access on Windows
                import platform
                if platform.system() == 'Windows':
                    cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
                else:
                    cap = cv2.VideoCapture(camera_index)
                
                self.capture = cap
                if not cap.isOpened():
                    self.live_status.config(text=f"❌ Cannot open camera {camera_index}")
                    self.stop_processing = True
                    self.play_btn.config(state='normal')
                    self.stop_btn.config(state='disabled')
                    return
                
                # Set higher resolution and FPS for better quality
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                cap.set(cv2.CAP_PROP_FPS, 60)
                
                # Get actual settings (camera may not support requested values)
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                self.live_status.config(text=f"▶️ Processing camera {camera_index} ({actual_width}x{actual_height} @ {actual_fps}fps)...")
                
            elif source_type == "screen":
                # Screen capture mode
                try:
                    import mss
                    sct = mss.mss()
                    monitor = sct.monitors[1]  # Primary monitor
                    
                    self.live_status.config(text="▶️ Processing screen capture...")
                    frame_count = 0
                    import time
                    
                    while not self.stop_processing:
                        # Capture screen
                        sct_img = sct.grab(monitor)
                        frame = np.array(sct_img)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                        
                        # Process with improved inference engine
                        prediction = engine.predict(frame, None)
                        result_frame = engine.visualize_prediction(frame, prediction)
                        result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                        
                        # Queue frame
                        try:
                            self.frame_queue.put_nowait((result_frame, frame_count, prediction))
                        except queue.Full:
                            pass
                        
                        frame_count += 1
                        time.sleep(0.033)  # ~30 FPS
                    
                    sct.close()
                    return
                    
                except ImportError:
                    self.live_status.config(text="❌ Install 'mss' library: pip install mss")
                    self.stop_processing = True
                    return
            
            # Camera processing loop - uses improved inference with corner-aware racing line
            frame_count = 0
            import time
            fps_start_time = time.time()
            fps_frame_count = 0
            
            while not self.stop_processing and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    continue  # Camera error, try again
                
                # Process frame with improved inference engine
                # (includes corner detection, track cleanup, temporal smoothing)
                prediction = engine.predict(frame, None)
                result_frame = engine.visualize_prediction(frame, prediction)
                
                # Convert to RGB for display
                result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                
                # Queue frame for display
                try:
                    self.frame_queue.put_nowait((result_frame, frame_count, prediction))
                except queue.Full:
                    pass
                
                frame_count += 1
                fps_frame_count += 1
                
                # Update FPS every second
                if time.time() - fps_start_time >= 1.0:
                    fps = fps_frame_count / (time.time() - fps_start_time)
                    self.fps_label.config(text=f"FPS: {fps:.1f}")
                    fps_start_time = time.time()
                    fps_frame_count = 0
                
            if cap is not None:
                cap.release()
            self.capture = None
            self.stop_processing = True
            self.play_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            
        except Exception as e:
            self.live_status.config(text=f"Error: {str(e)}")
            if cap is not None:
                cap.release()
            self.capture = None
            self.stop_processing = True
            self.play_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            
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

    def browse_video(self):
        """Browse and select a racing video file for analysis"""
        file_path = filedialog.askopenfilename(
            title="Select Racing Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(f"✓ Selected: {file_path}")
            try:
                self.process_btn.config(state='normal', bg=self.colors['accent'])
            except Exception:
                pass
            try:
                self.progress_label.config(text="✓ Ready to analyze! Click the button above.")
            except Exception:
                pass

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
        
        self.stop_batch_analysis = False
        self.process_btn.config(state='disabled')
        self.analyze_stop_btn.config(state='normal', bg='#d13438')
        self.progress_label.config(text="🔄 Processing video with AI... This may take a few minutes.")
        self.progress_var.set(0)
        
        # Get device selection
        device = self.analyze_device_var.get()
        
        # Start processing thread
        thread = threading.Thread(
            target=self.batch_processing_thread,
            args=(video_path, str(output_path), device),
            daemon=True
        )
        thread.start()
        
    def stop_batch_processing(self):
        """Stop batch processing"""
        self.stop_batch_analysis = True
        self.progress_label.config(text="⏹ Stopping analysis...")
        self.analyze_stop_btn.config(state='disabled')
    
    def batch_processing_thread(self, video_path, output_path, device='auto'):
        """Background thread for batch processing"""
        try:
            # Set device
            import torch
            if device == 'auto':
                actual_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            elif device == 'cuda':
                # Validate CUDA is actually available
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "GPU/CUDA is not available. This could mean:\n\n"
                        "1. PyTorch was installed without CUDA support\n"
                        "2. No NVIDIA GPU is present\n"
                        "3. CUDA drivers are not installed\n\n"
                        "Please select 'CPU Only' or 'Auto' mode instead.\n\n"
                        "To enable GPU support, you may need to reinstall PyTorch with CUDA."
                    )
                actual_device = 'cuda'
            else:
                actual_device = device
            
            model_path = 'models/racing_line_model.pth'
            self.progress_label.config(text=f"📦 Loading model: {model_path}")
            self.root.update()
            
            processor = BatchProcessor(model_path, device=actual_device)
            
            # Get actual device being used (may differ from requested due to compatibility)
            actual_device_used = str(processor.engine.device).replace('cuda:0', 'cuda')
            
            self.model_info_label.config(text=f"📦 Model: {model_path} | 🖥️ Device: {actual_device_used.upper()}")
            self.progress_label.config(text=f"🚀 Processing with {actual_device_used.upper()}...")
            self.root.update()
            
            # Get total frames
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            self.progress_label.config(text=f"🔄 Analyzing {total_frames} frames with AI...")
            self.progress_var.set(0)
            
            # Process with progress updates
            def update_progress(frame_num, total, fps, inference_ms):
                """Update GUI with processing progress"""
                try:
                    progress = (frame_num / total * 100) if total > 0 else 0
                    self.progress_var.set(progress)
                    
                    # Update status labels
                    self.frame_status_label.config(text=f"📊 Frames: {frame_num}/{total}")
                    self.fps_status_label.config(text=f"⚡ Processing Speed: {fps:.1f} FPS")
                    self.inference_status_label.config(text=f"🧠 Inference: {inference_ms:.1f} ms/frame")
                    
                    # Calculate time remaining
                    if fps > 0:
                        remaining_frames = total - frame_num
                        remaining_seconds = remaining_frames / fps
                        remaining_mins = int(remaining_seconds // 60)
                        remaining_secs = int(remaining_seconds % 60)
                        
                        elapsed_seconds = frame_num / fps
                        elapsed_mins = int(elapsed_seconds // 60)
                        elapsed_secs = int(elapsed_seconds % 60)
                        
                        self.time_status_label.config(
                            text=f"⏱️ Time: {elapsed_mins:02d}:{elapsed_secs:02d} / Est. {remaining_mins:02d}:{remaining_secs:02d} remaining"
                        )
                    
                    self.root.update_idletasks()
                except:
                    pass  # Ignore errors during GUI updates
            
            stats = processor.process_video(video_path, output_path, 
                                           stop_callback=lambda: self.stop_batch_analysis,
                                           progress_callback=update_progress)
            
            # Check if stopped or completed
            if self.stop_batch_analysis:
                self.progress_var.set(0)
                self.progress_label.config(text="⏹ Analysis stopped by user")
                messagebox.showinfo(
                    "Analysis Stopped",
                    f"Video analysis was stopped.\n\n"
                    f"Frames processed: {stats['total_frames']}\n\n"
                    f"Note: Partial output may be incomplete."
                )
                return
            
            self.progress_var.set(100)
            self.progress_label.config(text="✅ Analysis complete!")
            
            # Show success dialog with output location
            result = messagebox.showinfo(
                "Analysis Complete! 🏁",
                f"Your racing video has been analyzed successfully!\n\n"
                f"The analyzed video with the optimal racing line (purple) has been saved to:\n\n"
                f"{output_path}\n\n"
                f"Model used: {model_path}\n"
                f"Device: {actual_device.upper()}\n"
                f"Average processing time: {stats['avg_inference_time']:.1f}ms per frame\n"
                f"Total frames analyzed: {stats['total_frames']}"
            )
            
            # Ask if user wants to open the folder
            if messagebox.askyesno("Open Folder?", "Would you like to open the folder containing your analyzed video?"):
                import subprocess
                subprocess.run(['explorer', '/select,', str(output_path)])
            
        except Exception as e:
            self.progress_label.config(text=f"❌ Error: {str(e)}")
            self.progress_var.set(0)
            messagebox.showerror("Processing Error", 
                               f"Sorry, the video could not be processed.\n\n"
                               f"Error: {str(e)}\n\n"
                               f"Make sure the video file is valid and not corrupted.")
        finally:
            self.process_btn.config(state='normal', bg=self.colors['accent'])
            self.analyze_stop_btn.config(state='disabled')
            self.stop_batch_analysis = False
            self.progress_var.set(0)
        
    # Training Methods
    def browse_training_data(self):
        """Browse for training data directory"""
        dir_path = filedialog.askdirectory(title="Select Training Data Directory")
        if dir_path:
            self.training_data_var.set(dir_path)
            
    def generate_training_data(self):
        """Open dialog to configure and generate training data"""
        # Create dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title("Generate Training Data")
        dialog.geometry("750x650")
        dialog.configure(bg=self.colors['bg'])
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(True, True)
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (750 // 2)
        y = (dialog.winfo_screenheight() // 2) - (650 // 2)
        dialog.geometry(f"750x650+{x}+{y}")
        
        # Title
        title_label = tk.Label(dialog, text="Training Data Generator - Configure All Settings",
                              font=('Segoe UI', 16, 'bold'),
                              bg=self.colors['bg'], fg=self.colors['fg'])
        title_label.pack(pady=20)
        
        # Settings frame
        settings_frame = tk.Frame(dialog, bg=self.colors['bg_light'])
        settings_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        settings_frame.grid_rowconfigure(0, minsize=60)
        settings_frame.grid_rowconfigure(1, minsize=60)
        settings_frame.grid_rowconfigure(2, minsize=60)
        settings_frame.grid_rowconfigure(3, minsize=60)
        settings_frame.grid_rowconfigure(4, minsize=15)  # Spacer
        settings_frame.grid_rowconfigure(5, minsize=50)
        settings_frame.grid_rowconfigure(6, minsize=50)
        settings_frame.grid_rowconfigure(7, minsize=50)
        
        # Video file selection
        tk.Label(settings_frame, text="Source Video:", 
                bg=self.colors['bg_light'], fg=self.colors['fg'],
                font=('Segoe UI', 11, 'bold')).grid(row=0, column=0, sticky='nw', pady=(15, 5), padx=(20, 10))
        video_var = tk.StringVar(value="No video selected")
        video_entry = tk.Entry(settings_frame, textvariable=video_var,
                bg='#3c3c3c', fg='#ffffff',
                font=('Segoe UI', 10),
                relief='solid', bd=2,
                insertbackground='#ffffff',
                selectbackground=self.colors['accent'], selectforeground='#ffffff')
        video_entry.grid(row=0, column=1, padx=10, pady=(15, 5), sticky='ew', ipady=5)
        
        def select_video():
            path = filedialog.askopenfilename(
                title="Select Video for Training Data",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
            )
            if path:
                video_var.set(path)
        
        tk.Button(settings_frame, text="Browse...", command=select_video,
                 bg=self.colors['accent'], fg='white', relief='raised',
                 font=('Segoe UI', 10, 'bold'),
                 padx=20, pady=10, cursor='hand2').grid(row=0, column=2, padx=(5, 20), pady=(15, 5), sticky='ew')
        
        # Output directory
        tk.Label(settings_frame, text="Output Directory:",
                bg=self.colors['bg_light'], fg=self.colors['fg'],
                font=('Segoe UI', 11, 'bold')).grid(row=1, column=0, sticky='nw', pady=(15, 5), padx=(20, 10))
        output_var = tk.StringVar(value=self.training_data_var.get())
        output_entry = tk.Entry(settings_frame, textvariable=output_var,
                bg='#3c3c3c', fg='#ffffff',
                font=('Segoe UI', 10),
                relief='solid', bd=2,
                insertbackground='#ffffff',
                selectbackground=self.colors['accent'], selectforeground='#ffffff')
        output_entry.grid(row=1, column=1, padx=10, pady=(15, 5), sticky='ew', ipady=5)
        
        def select_output():
            path = filedialog.askdirectory(title="Select Output Directory")
            if path:
                output_var.set(path)
        
        tk.Button(settings_frame, text="Browse...", command=select_output,
                 bg=self.colors['accent'], fg='white', relief='raised',
                 font=('Segoe UI', 10, 'bold'),
                 padx=20, pady=10, cursor='hand2').grid(row=1, column=2, padx=(5, 20), pady=(15, 5), sticky='ew')
        
        # Configure grid column weights so entry fields expand
        settings_frame.grid_columnconfigure(1, weight=3)
        settings_frame.grid_columnconfigure(2, weight=1)
        
        # Frame interval
        tk.Label(settings_frame, text="Frame Interval:",
                bg=self.colors['bg_light'], fg=self.colors['fg'],
                font=('Segoe UI', 11, 'bold')).grid(row=2, column=0, sticky='nw', pady=(15, 5), padx=(20, 10))
        interval_var = tk.IntVar(value=30)
        tk.Spinbox(settings_frame, from_=10, to=120, textvariable=interval_var,
                  width=15, bg='#3c3c3c', fg='#ffffff',
                  font=('Segoe UI', 10), relief='solid', bd=2,
                  buttonbackground=self.colors['accent'],
                  selectbackground=self.colors['accent'], selectforeground='#ffffff').grid(row=2, column=1, sticky='w', padx=10, pady=(15, 5), ipady=3)
        tk.Label(settings_frame, text="Extract 1 frame every N frames",
                bg=self.colors['bg_light'], fg=self.colors['fg_dim'],
                font=('Segoe UI', 9)).grid(row=2, column=2, sticky='w', padx=(5, 20), pady=(15, 5))
        
        # Preview option
        tk.Label(settings_frame, text="Show Preview:",
                bg=self.colors['bg_light'], fg=self.colors['fg'],
                font=('Segoe UI', 11, 'bold')).grid(row=3, column=0, sticky='nw', pady=(15, 5), padx=(20, 10))
        preview_var = tk.BooleanVar(value=False)
        tk.Checkbutton(settings_frame, variable=preview_var,
                      bg=self.colors['bg_light'], fg=self.colors['fg'],
                      font=('Segoe UI', 10),
                      selectcolor=self.colors['card']).grid(row=3, column=1, sticky='w', padx=10, pady=(15, 5))
        
        # Spacer
        tk.Label(settings_frame, text="─" * 80,
                bg=self.colors['bg_light'], fg=self.colors['fg_dim']).grid(row=4, column=0, columnspan=3, pady=10)
        
        # Training Parameters Section Header
        tk.Label(settings_frame, text="Training Parameters (Optional - can adjust later)",
                bg=self.colors['bg_light'], fg=self.colors['accent'],
                font=('Segoe UI', 11, 'bold')).grid(row=5, column=0, columnspan=3, sticky='w', padx=(20, 10), pady=(5, 10))
        
        # Epochs
        tk.Label(settings_frame, text="Epochs:",
                bg=self.colors['bg_light'], fg=self.colors['fg'],
                font=('Segoe UI', 10)).grid(row=6, column=0, sticky='w', padx=(20, 10), pady=5)
        epochs_dialog_var = tk.IntVar(value=self.epochs_var.get())
        tk.Scale(settings_frame, from_=5, to=100, variable=epochs_dialog_var,
                orient='horizontal', length=150,
                bg=self.colors['bg_light'], fg=self.colors['fg'],
                highlightthickness=0, troughcolor=self.colors['card']).grid(row=6, column=1, sticky='w', padx=10, pady=5)
        tk.Label(settings_frame, textvariable=epochs_dialog_var,
                bg=self.colors['bg_light'], fg=self.colors['fg'],
                width=3, font=('Segoe UI', 10, 'bold')).grid(row=6, column=2, sticky='w', padx=5)
        
        # Batch Size
        tk.Label(settings_frame, text="Batch Size:",
                bg=self.colors['bg_light'], fg=self.colors['fg'],
                font=('Segoe UI', 10)).grid(row=7, column=0, sticky='w', padx=(20, 10), pady=5)
        batch_dialog_var = tk.IntVar(value=self.batch_size_var.get())
        tk.Spinbox(settings_frame, from_=2, to=16, textvariable=batch_dialog_var,
                  width=10, bg='#3c3c3c', fg='#ffffff',
                  font=('Segoe UI', 10), relief='solid', bd=2,
                  selectbackground=self.colors['accent'], selectforeground='#ffffff').grid(row=7, column=1, sticky='w', padx=10, pady=5)
        tk.Label(settings_frame, text="(Min: 2)",
                bg=self.colors['bg_light'], fg=self.colors['fg_dim'],
                font=('Segoe UI', 9)).grid(row=7, column=2, sticky='w', padx=5)
        
        # Buttons
        button_frame = tk.Frame(dialog, bg=self.colors['bg'])
        button_frame.pack(pady=10)
        
        def start_generation():
            video_path = video_var.get()
            if video_path == "No video selected" or not Path(video_path).exists():
                messagebox.showerror("Error", "Please select a valid video file")
                return
            
            output_dir = output_var.get()
            interval = interval_var.get()
            preview = preview_var.get()
            
            # Save training parameters from dialog
            self.epochs_var.set(epochs_dialog_var.get())
            self.batch_size_var.set(batch_dialog_var.get())
            
            dialog.destroy()
            
            # Update training data directory to match output
            self.training_data_var.set(output_dir)
            
            self.log_training("Starting training data generation...")
            self.log_training(f"Video: {Path(video_path).name}")
            self.log_training(f"Interval: {interval} frames")
            self.log_training(f"Output: {output_dir}")
            self.log_training(f"Training will use: {self.epochs_var.get()} epochs, batch size {self.batch_size_var.get()}")
            
            # Run auto labeling in thread
            thread = threading.Thread(
                target=self.run_auto_labeling,
                args=(video_path, output_dir, interval, preview),
                daemon=True
            )
            thread.start()
        
        tk.Button(button_frame, text="Generate", command=start_generation,
                 bg=self.colors['success'], fg='white', relief='flat',
                 padx=20, pady=8, font=('Segoe UI', 10, 'bold')).pack(side='left', padx=5)
        tk.Button(button_frame, text="Cancel", command=dialog.destroy,
                 bg=self.colors['card'], fg=self.colors['fg'], relief='flat',
                 padx=20, pady=8).pack(side='left', padx=5)
    
    def refine_training_data(self):
        """Clean and refine training data"""
        data_dir = Path(self.training_data_var.get())
        
        if not data_dir.exists():
            messagebox.showerror("Error", "Training data directory not found")
            return
        
        # Count current data
        images_dir = data_dir / 'images'
        masks_dir = data_dir / 'masks'
        
        if not images_dir.exists() or not masks_dir.exists():
            messagebox.showerror("Error", "Training data structure invalid")
            return
        
        # Check for both .jpg and .png files
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        mask_files = list(masks_dir.glob('*.jpg')) + list(masks_dir.glob('*.png'))
        
        response = messagebox.askyesno(
            "Refine Training Data",
            f"Current data:\n"
            f"  Images: {len(image_files)}\n"
            f"  Masks: {len(mask_files)}\n\n"
            f"This will:\n"
            f"  • Remove corrupted files\n"
            f"  • Remove mismatched pairs\n"
            f"  • Validate all masks\n"
            f"  • Create backup of original data\n\n"
            f"Continue?"
        )
        
        if response:
            self.log_training("Refining training data...")
            thread = threading.Thread(
                target=self.run_data_refinement,
                args=(str(data_dir),),
                daemon=True
            )
            thread.start()
            
    def run_auto_labeling(self, video_path, output_dir, frame_interval=30, preview=False):
        """Run automatic labeling with progress tracking"""
        try:
            import cv2
            import sys
            from pathlib import Path
            
            # Add scripts directory to path
            scripts_dir = Path(__file__).parent.parent / 'scripts'
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            
            from auto_label_track import AutoTrackLabeler # type: ignore
            
            # Get total frames for progress calculation
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            estimated_samples = total_frames // frame_interval
            self.log_training(f"Processing video: {total_frames} frames")
            self.log_training(f"Estimated samples: ~{estimated_samples}")
            
            # Update progress
            self.training_status_label.config(text="Generating training data...")
            self.training_progress_var.set(0)
            
            # Create labeler and process
            labeler = AutoTrackLabeler()
            
            # Process video (without progress callback since it's not supported)
            labeler.process_video(video_path, output_dir, 
                                frame_interval=frame_interval, 
                                preview=preview)
            
            self.training_progress_var.set(100)
            self.training_status_label.config(text="Data generation complete!")
            self.log_training("✓ Training data generated successfully!")
            
            # Update training data directory and stats (ensure on main thread)
            self.training_data_var.set(output_dir)
            self.root.after(100, self.update_data_stats)
            
            # Count actual results
            images_dir = Path(output_dir) / 'images'
            if images_dir.exists():
                sample_count = len(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))
                self.log_training(f"✓ Created {sample_count} training samples")
            
            messagebox.showinfo("Success", 
                              f"Training data generated successfully!\n\n"
                              f"Location: {output_dir}\n\n"
                              f"You can now proceed to Step 2 to configure training parameters.")
        except Exception as e:
            self.log_training(f"✗ Error: {str(e)}")
            self.training_status_label.config(text="Generation failed")
            messagebox.showerror("Error", f"Failed to generate data: {str(e)}")
        finally:
            self.training_progress_var.set(0)
    
    def run_data_refinement(self, data_dir):
        """Refine and clean training data"""
        try:
            import shutil
            data_path = Path(data_dir)
            images_dir = data_path / 'images'
            masks_dir = data_path / 'masks'
            
            # Create backup
            self.log_training("Creating backup...")
            backup_dir = data_path.parent / f"{data_path.name}_backup_{int(time.time())}"
            shutil.copytree(data_path, backup_dir)
            self.log_training(f"✓ Backup created: {backup_dir.name}")
            
            # Find all image files (check both .jpg and .png)
            image_files = {f.stem: f for f in list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))}
            mask_files = {f.stem: f for f in list(masks_dir.glob('*.jpg')) + list(masks_dir.glob('*.png'))}
            
            removed_count = 0
            corrupted_count = 0
            
            # Check each pair
            for stem in list(image_files.keys()):
                image_path = image_files[stem]
                mask_path = mask_files.get(stem)
                
                should_remove = False
                reason = ""
                
                # Check if pair exists
                if not mask_path:
                    should_remove = True
                    reason = "missing mask"
                else:
                    # Validate image
                    try:
                        img = cv2.imread(str(image_path))
                        if img is None or img.size == 0:
                            should_remove = True
                            reason = "corrupted image"
                            corrupted_count += 1
                    except:
                        should_remove = True
                        reason = "unreadable image"
                        corrupted_count += 1
                    
                    # Validate mask
                    if not should_remove:
                        try:
                            mask = cv2.imread(str(mask_path))
                            if mask is None or mask.size == 0:
                                should_remove = True
                                reason = "corrupted mask"
                                corrupted_count += 1
                        except:
                            should_remove = True
                            reason = "unreadable mask"
                            corrupted_count += 1
                
                # Remove if needed
                if should_remove:
                    self.log_training(f"Removing {stem}: {reason}")
                    image_path.unlink(missing_ok=True)
                    if mask_path:
                        mask_path.unlink(missing_ok=True)
                    removed_count += 1
            
            # Check for orphaned masks
            for stem in mask_files.keys():
                if stem not in image_files:
                    self.log_training(f"Removing orphaned mask: {stem}")
                    mask_files[stem].unlink(missing_ok=True)
                    removed_count += 1
            
            # Summary
            remaining = len(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))
            self.log_training(f"\n✓ Refinement complete!")
            self.log_training(f"  Removed: {removed_count} files")
            self.log_training(f"  Corrupted: {corrupted_count} files")
            self.log_training(f"  Remaining: {remaining} valid pairs")
            self.log_training(f"  Backup: {backup_dir.name}")
            
            self.update_data_stats()
            
            messagebox.showinfo(
                "Refinement Complete",
                f"Data refinement finished!\n\n"
                f"Removed: {removed_count} files\n"
                f"Corrupted: {corrupted_count} files\n"
                f"Remaining: {remaining} valid pairs\n\n"
                f"Original data backed up to:\n{backup_dir.name}"
            )
            
        except Exception as e:
            self.log_training(f"\n✗ Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to refine data: {str(e)}")
    
    def browse_training_data(self):
        """Browse for training data directory"""
        dir_path = filedialog.askdirectory(
            title="Select Training Data Directory",
            initialdir=self.training_data_var.get()
        )
        if dir_path:
            self.training_data_var.set(dir_path)
            self.update_data_stats()
            self.log_training(f"Training data directory: {dir_path}")
    
    def update_data_stats(self):
        """Update training data statistics display"""
        try:
            data_dir = Path(self.training_data_var.get())
            if data_dir.exists():
                images_dir = data_dir / 'images'
                masks_dir = data_dir / 'masks'
                
                if images_dir.exists() and masks_dir.exists():
                    # Check for both .jpg and .png files
                    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
                    mask_files = list(masks_dir.glob('*.jpg')) + list(masks_dir.glob('*.png'))
                    image_count = len(image_files)
                    mask_count = len(mask_files)
                    
                    if image_count == mask_count and image_count > 0:
                        self.data_stats_label.config(
                            text=f"✓ Ready! {image_count} valid training pairs found",
                            foreground=self.colors['success']
                        )
                        self.training_status_label.config(text="Ready to train!")
                    elif image_count != mask_count:
                        self.data_stats_label.config(
                            text=f"⚠ Data mismatch: {image_count} images, {mask_count} masks - Click 'Refine/Clean Data'",
                            foreground=self.colors['warning']
                        )
                    else:
                        self.data_stats_label.config(
                            text="No training data found - Generate data first",
                            foreground=self.colors['warning']
                        )
                    return
            
            self.data_stats_label.config(
                text="No training data found - Click 'Generate Training Data'",
                foreground=self.colors['warning']
            )
        except:
            pass
            
    def start_training(self):
        """Start model training"""
        # Create models directory first to prevent access errors
        from pathlib import Path
        models_dir = Path('models')
        models_dir.mkdir(parents=True, exist_ok=True)
        
        data_dir = self.training_data_var.get()
        
        if not Path(data_dir).exists():
            messagebox.showerror(
                "Training Data Not Found", 
                f"Training data directory not found:\n{data_dir}\n\n"
                f"Please:\n"
                f"1. Generate training data in Step 1, OR\n"
                f"2. Use the 'Browse...' button to select existing training data"
            )
            return
        
        # Check if data exists
        images_dir = Path(data_dir) / 'images'
        masks_dir = Path(data_dir) / 'masks'
        
        if not images_dir.exists() or not masks_dir.exists():
            messagebox.showerror(
                "Invalid Training Data Structure",
                f"Training data directory must contain 'images' and 'masks' folders.\n\n"
                f"Current directory: {data_dir}\n\n"
                f"Please generate training data or select a valid directory."
            )
            return
        
        # Check for both .jpg and .png files
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        mask_files = list(masks_dir.glob('*.jpg')) + list(masks_dir.glob('*.png'))
        image_count = len(image_files)
        mask_count = len(mask_files)
        
        if image_count == 0:
            response = messagebox.askyesno(
                "No Training Data",
                f"No training images found in:\n{images_dir}\n\n"
                f"Would you like to generate training data now?"
            )
            if response:
                self.generate_training_data()
            return
        
        if image_count != mask_count:
            response = messagebox.askyesno(
                "Data Mismatch Warning",
                f"Found {image_count} images but {mask_count} masks.\n\n"
                f"This may cause training issues.\n\n"
                f"Would you like to clean/refine the data first?"
            )
            if response:
                self.refine_training_data()
                return
        
        self.stop_training = False
        self.train_btn.config(state='disabled')
        self.train_stop_btn.config(state='normal', bg='#d13438')
        self.training_status_label.config(text="Preparing to train...")
        self.log_training("="*50)
        self.log_training("Starting model training...")
        self.log_training("="*50)
        
        # Start training thread
        thread = threading.Thread(target=self.training_thread, daemon=True)
        thread.start()
        
    def stop_training_process(self):
        """Stop training process"""
        self.stop_training = True
        self.log_training("\n⏹ Stop requested - training will stop after current epoch...")
        self.training_status_label.config(text="Stopping training...")
        self.train_stop_btn.config(state='disabled')
    
    def training_thread(self):
        """Background thread for training"""
        try:
            # Redirect logging to GUI
            import sys
            from io import StringIO
            
            class GUILogger:
                def __init__(self, callback, progress_callback):
                    self.callback = callback
                    self.progress_callback = progress_callback
                    self.buffer = ""
                
                def write(self, text):
                    self.buffer += text
                    if '\n' in self.buffer:
                        lines = self.buffer.split('\n')
                        for line in lines[:-1]:
                            if line.strip():
                                self.callback(line.strip())
                                # Extract progress from epoch info
                                if 'Epoch' in line and '/' in line:
                                    try:
                                        # Parse "Epoch X/Y" pattern
                                        parts = line.split('Epoch')[1].split('/')[0].strip()
                                        current_epoch = int(parts.split()[0])
                                        total_epochs = self.progress_callback.total_epochs
                                        progress = (current_epoch / total_epochs) * 100
                                        self.progress_callback.update(progress, f"Epoch {current_epoch}/{total_epochs}")
                                    except:
                                        pass
                        self.buffer = lines[-1]
                
                def flush(self):
                    if self.buffer.strip():
                        self.callback(self.buffer.strip())
                        self.buffer = ""
            
            class ProgressTracker:
                def __init__(self, gui_obj, total_epochs):
                    self.gui = gui_obj
                    self.total_epochs = total_epochs
                
                def update(self, progress, status):
                    self.gui.training_progress_var.set(progress)
                    self.gui.training_status_label.config(text=status)
            
            # Create progress tracker
            progress_tracker = ProgressTracker(self, self.epochs_var.get())
            
            # Capture stdout/stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = GUILogger(self.log_training, progress_tracker)
            sys.stderr = GUILogger(self.log_training, progress_tracker)
            
            try:
                self.training_status_label.config(text="Training in progress...")
                
                # Create models directory if it doesn't exist
                from pathlib import Path
                models_dir = Path('models')
                models_dir.mkdir(parents=True, exist_ok=True)
                self.log_training(f"Output directory: {models_dir.absolute()}")
                
                # Set CPU threads if using CPU
                if self.device_var.get() == 'cpu':
                    import torch
                    num_threads = self.cpu_cores_var.get()
                    torch.set_num_threads(num_threads)
                    self.log_training(f"Using {num_threads} CPU threads for training")
                
                train_model(
                    data_dir=self.training_data_var.get(),
                    output_dir='models',
                    epochs=self.epochs_var.get(),
                    batch_size=self.batch_size_var.get(),
                    learning_rate=self.lr_var.get(),
                    device=self.device_var.get()
                )
                
                self.log_training("\n" + "="*50)
                self.log_training("✓ TRAINING COMPLETE!")
                self.log_training("="*50)
                self.log_training("Model saved to: models/racing_line_model.pth")
                self.training_status_label.config(text="Training complete!")
                self.training_progress_var.set(100)
                
                messagebox.showinfo(
                    "Training Complete! 🎉",
                    "Model training finished successfully!\n\n"
                    "The NEW trained model has been saved to:\n"
                    "models/racing_line_model.pth\n\n"
                    "This model will now be used automatically when you:\n"
                    "• Analyze videos in the 'Analyze Video' tab\n"
                    "• Use live preview mode\n\n"
                    "Your custom training data has improved the AI!"
                )
                
            finally:
                # Restore stdout/stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            
        except KeyboardInterrupt:
            self.log_training("\n✗ Training cancelled by user")
            self.training_status_label.config(text="Training cancelled")
            messagebox.showwarning("Cancelled", "Training was cancelled")
            
        except Exception as e:
            import traceback
            self.log_training(f"\n✗ ERROR: {str(e)}")
            self.log_training(traceback.format_exc())
            self.training_status_label.config(text="Training failed")
            messagebox.showerror(
                "Training Failed",
                f"An error occurred during training:\n\n{str(e)}\n\n"
                f"Check the training log for details."
            )
            
        finally:
            self.train_btn.config(state='normal')
            self.train_stop_btn.config(state='disabled')
            self.training_progress_var.set(0)
            self.stop_training = False
    
    # Annotation Methods
    def browse_annotation_video(self):
        """Browse for video to annotate"""
        file_path = filedialog.askopenfilename(
            title="Select Racing Video to Annotate",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.annotation_video_var.set(file_path)
            self.launch_annotation_btn.config(state='normal')
    
    def browse_annotation_output(self):
        """Browse for annotation output directory"""
        dir_path = filedialog.askdirectory(title="Select Output Directory for Annotations")
        if dir_path:
            self.annotation_output_var.set(dir_path)
    
    def launch_annotation_tool(self):
        """Launch the interactive annotation tool"""
        video_path = self.annotation_video_var.get()
        output_dir = self.annotation_output_var.get()
        
        if video_path == "No video selected" or not video_path:
            messagebox.showwarning("No Video", "Please select a video file first.")
            return
        
        if not Path(video_path).exists():
            messagebox.showerror("File Not Found", f"Video file not found:\n{video_path}")
            return
        
        # Confirm launch
        response = messagebox.askyesno(
            "Launch Annotation Tool",
            "The annotation tool will open in a new window.\n\n"
            "Use your mouse to draw:\n"
            "• Racing line (yellow)\n"
            "• Track boundaries (red/blue)\n\n"
            "Press SPACE to save each frame.\n"
            "Press Q to quit when done.\n\n"
            "The tool will save annotated data to:\n"
            f"{output_dir}\n\n"
            "Ready to launch?"
        )
        
        if not response:
            return
        
        # Import and run annotation tool
        try:
            # Test if PIL is available
            from PIL import Image, ImageTk
            
            from .annotate import annotate_video
            
            # Run annotation tool as child window (don't hide main window)
            try:
                annotate_video(video_path, output_dir, parent_window=self.root)
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                messagebox.showerror("Annotation Error", 
                                   f"Error during annotation:\n{str(e)}\n\n"
                                   f"Please check that:\n"
                                   f"1. PIL/Pillow is installed\n"
                                   f"2. Video file is valid\n"
                                   f"3. Output directory is writable\n\n"
                                   f"Details:\n{error_details[:200]}")
            
        except ImportError as e:
            messagebox.showerror("Missing Dependency", 
                               f"PIL/Pillow is not installed.\n\n"
                               f"Please run: pip install Pillow\n\n"
                               f"Error: {str(e)}")
        except Exception as e:
            import traceback
            messagebox.showerror("Error", f"Failed to launch annotation tool:\n{str(e)}\n\n{traceback.format_exc()[:200]}")
            
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
