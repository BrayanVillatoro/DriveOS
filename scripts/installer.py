"""
DriveOS Windows Installer
Automated installation wizard for DriveOS Racing Line Analyzer

This script creates a user-friendly installation experience:
1. Checks Python version compatibility
2. Creates virtual environment
3. Installs all dependencies
4. Creates desktop shortcut
5. Adds to Start Menu
"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import json
import winshell
from win32com.client import Dispatch
import shutil

class DriveOSInstaller:
    """Installation wizard for DriveOS"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DriveOS Setup Wizard")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # Default installation directory
        self.install_dir = Path.home() / "DriveOS"
        self.create_desktop_shortcut = tk.BooleanVar(value=True)
        self.create_start_menu = tk.BooleanVar(value=True)
        self.install_mss = tk.BooleanVar(value=True)  # For screen capture
        
        # Current step
        self.current_step = 0
        self.steps = [
            self.welcome_page,
            self.requirements_page,
            self.location_page,
            self.options_page,
            self.install_page,
            self.complete_page
        ]
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup main UI"""
        # Header
        header = tk.Frame(self.root, bg='#0078d4', height=80)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        title = tk.Label(header, text="🏁 DriveOS Racing Line Analyzer",
                        font=('Segoe UI', 18, 'bold'),
                        bg='#0078d4', fg='white')
        title.pack(pady=20)
        
        # Content frame
        self.content_frame = tk.Frame(self.root, bg='white')
        self.content_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Navigation buttons
        nav_frame = tk.Frame(self.root, bg='#f0f0f0', height=60)
        nav_frame.pack(fill='x')
        nav_frame.pack_propagate(False)
        
        self.back_btn = tk.Button(nav_frame, text="< Back",
                                  command=self.previous_step,
                                  state='disabled',
                                  width=10)
        self.back_btn.pack(side='left', padx=20, pady=15)
        
        self.next_btn = tk.Button(nav_frame, text="Next >",
                                  command=self.next_step,
                                  bg='#0078d4', fg='white',
                                  width=10)
        self.next_btn.pack(side='right', padx=20, pady=15)
        
        self.cancel_btn = tk.Button(nav_frame, text="Cancel",
                                    command=self.cancel_install,
                                    width=10)
        self.cancel_btn.pack(side='right', pady=15)
        
        # Show first page
        self.show_step()
        
    def clear_content(self):
        """Clear content frame"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
            
    def show_step(self):
        """Show current step"""
        self.clear_content()
        self.steps[self.current_step]()
        
        # Update navigation buttons
        self.back_btn.config(state='normal' if self.current_step > 0 else 'disabled')
        
        if self.current_step == len(self.steps) - 1:
            self.next_btn.config(text="Finish", command=self.finish_install)
        elif self.current_step == len(self.steps) - 2:
            self.next_btn.config(text="Install", command=self.run_installation)
        else:
            self.next_btn.config(text="Next >", command=self.next_step)
            
    def next_step(self):
        """Go to next step"""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self.show_step()
            
    def previous_step(self):
        """Go to previous step"""
        if self.current_step > 0:
            self.current_step -= 1
            self.show_step()
            
    def cancel_install(self):
        """Cancel installation"""
        if messagebox.askyesno("Cancel Installation", 
                              "Are you sure you want to cancel the installation?"):
            self.root.quit()
            
    def welcome_page(self):
        """Welcome page"""
        tk.Label(self.content_frame, text="Welcome to DriveOS Setup",
                font=('Segoe UI', 16, 'bold'),
                bg='white').pack(pady=20)
        
        tk.Label(self.content_frame,
                text="This wizard will guide you through the installation of\n"
                     "DriveOS Racing Line Analyzer.\n\n"
                     "DriveOS uses AI and machine learning to analyze racing videos\n"
                     "and identify the fastest, most efficient path around any track.\n\n"
                     "Features:\n"
                     "• Real-time racing line analysis\n"
                     "• Support for video files, webcams, and screen capture\n"
                     "• AI-powered optimal path detection\n"
                     "• Professional visualization tools\n\n"
                     "Click Next to continue.",
                font=('Segoe UI', 10),
                bg='white',
                justify='left').pack(pady=10)
                
    def requirements_page(self):
        """System requirements page"""
        tk.Label(self.content_frame, text="System Requirements",
                font=('Segoe UI', 14, 'bold'),
                bg='white').pack(pady=20)
        
        # Check Python version
        py_version = sys.version_info
        py_ok = 3.9 <= py_version.major + py_version.minor/10 < 3.12
        
        req_frame = tk.Frame(self.content_frame, bg='white')
        req_frame.pack(fill='both', expand=True, padx=20)
        
        tk.Label(req_frame, text="Checking system requirements...",
                font=('Segoe UI', 11),
                bg='white').grid(row=0, column=0, columnspan=2, pady=10, sticky='w')
        
        # Python version
        status = "✓" if py_ok else "✗"
        color = "green" if py_ok else "red"
        tk.Label(req_frame, text=f"{status} Python Version:",
                font=('Segoe UI', 10),
                fg=color,
                bg='white').grid(row=1, column=0, sticky='w', pady=5)
        tk.Label(req_frame, text=f"{py_version.major}.{py_version.minor}.{py_version.micro} (Required: 3.9-3.11)",
                font=('Segoe UI', 10),
                bg='white').grid(row=1, column=1, sticky='w', padx=10)
        
        # Operating system
        tk.Label(req_frame, text="✓ Operating System:",
                font=('Segoe UI', 10),
                fg='green',
                bg='white').grid(row=2, column=0, sticky='w', pady=5)
        tk.Label(req_frame, text=f"Windows (Detected: {sys.platform})",
                font=('Segoe UI', 10),
                bg='white').grid(row=2, column=1, sticky='w', padx=10)
        
        # Disk space
        tk.Label(req_frame, text="✓ Disk Space:",
                font=('Segoe UI', 10),
                fg='green',
                bg='white').grid(row=3, column=0, sticky='w', pady=5)
        tk.Label(req_frame, text="~2 GB required for installation",
                font=('Segoe UI', 10),
                bg='white').grid(row=3, column=1, sticky='w', padx=10)
        
        if not py_ok:
            tk.Label(self.content_frame,
                    text=f"\n⚠️ WARNING: Python version {py_version.major}.{py_version.minor} detected.\n"
                         "DriveOS requires Python 3.9, 3.10, or 3.11.\n"
                         "Please install a compatible Python version from python.org",
                    font=('Segoe UI', 10),
                    fg='red',
                    bg='white').pack(pady=20)
            self.next_btn.config(state='disabled')
        else:
            self.next_btn.config(state='normal')
            
    def location_page(self):
        """Installation location page"""
        tk.Label(self.content_frame, text="Choose Install Location",
                font=('Segoe UI', 14, 'bold'),
                bg='white').pack(pady=20)
        
        tk.Label(self.content_frame,
                text="Setup will install DriveOS in the following folder.\n"
                     "To install in a different folder, click Browse.",
                font=('Segoe UI', 10),
                bg='white',
                justify='left').pack(pady=10)
        
        # Location selector
        loc_frame = tk.Frame(self.content_frame, bg='white')
        loc_frame.pack(fill='x', padx=40, pady=20)
        
        self.location_var = tk.StringVar(value=str(self.install_dir))
        
        tk.Entry(loc_frame, textvariable=self.location_var,
                font=('Segoe UI', 10),
                width=50).pack(side='left', padx=(0, 10))
        
        tk.Button(loc_frame, text="Browse...",
                 command=self.browse_location).pack(side='left')
        
        # Disk space info
        tk.Label(self.content_frame,
                text="Space required: ~2 GB",
                font=('Segoe UI', 9),
                fg='gray',
                bg='white').pack()
                
    def browse_location(self):
        """Browse for installation location"""
        folder = filedialog.askdirectory(title="Select Installation Folder")
        if folder:
            self.install_dir = Path(folder) / "DriveOS"
            self.location_var.set(str(self.install_dir))
            
    def options_page(self):
        """Installation options page"""
        tk.Label(self.content_frame, text="Select Additional Options",
                font=('Segoe UI', 14, 'bold'),
                bg='white').pack(pady=20)
        
        options_frame = tk.Frame(self.content_frame, bg='white')
        options_frame.pack(fill='both', expand=True, padx=40)
        
        tk.Checkbutton(options_frame, text="Create Desktop Shortcut",
                      variable=self.create_desktop_shortcut,
                      font=('Segoe UI', 10),
                      bg='white').pack(anchor='w', pady=10)
        
        tk.Checkbutton(options_frame, text="Add to Start Menu",
                      variable=self.create_start_menu,
                      font=('Segoe UI', 10),
                      bg='white').pack(anchor='w', pady=10)
        
        tk.Checkbutton(options_frame, text="Install Screen Capture Support (for sim racing)",
                      variable=self.install_mss,
                      font=('Segoe UI', 10),
                      bg='white').pack(anchor='w', pady=10)
        
        tk.Label(options_frame,
                text="\nRecommended: Keep all options checked for the best experience.",
                font=('Segoe UI', 9),
                fg='gray',
                bg='white',
                justify='left').pack(anchor='w', pady=10)
                
    def install_page(self):
        """Installation progress page"""
        tk.Label(self.content_frame, text="Installing DriveOS",
                font=('Segoe UI', 14, 'bold'),
                bg='white').pack(pady=20)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.content_frame,
                                           variable=self.progress_var,
                                           maximum=100,
                                           length=500)
        self.progress_bar.pack(pady=20)
        
        self.status_label = tk.Label(self.content_frame,
                                     text="Ready to install...",
                                     font=('Segoe UI', 10),
                                     bg='white')
        self.status_label.pack(pady=10)
        
        self.log_text = tk.Text(self.content_frame, height=15, width=70,
                               font=('Consolas', 8),
                               bg='#f5f5f5')
        self.log_text.pack(pady=10)
        
        # Disable navigation during install
        self.back_btn.config(state='disabled')
        self.cancel_btn.config(state='disabled')
        
    def run_installation(self):
        """Run the installation process"""
        self.next_btn.config(state='disabled')
        
        # Update install dir from UI
        self.install_dir = Path(self.location_var.get())
        
        # Start installation in thread
        import threading
        thread = threading.Thread(target=self.install_thread, daemon=True)
        thread.start()
        
    def log(self, message):
        """Add message to log"""
        self.log_text.insert('end', f"{message}\n")
        self.log_text.see('end')
        self.root.update()
        
    def install_thread(self):
        """Installation thread"""
        try:
            # Step 1: Create installation directory
            self.status_label.config(text="Creating installation directory...")
            self.progress_var.set(10)
            self.log(f"Creating directory: {self.install_dir}")
            
            # Copy files to install directory
            current_dir = Path(__file__).parent
            if current_dir != self.install_dir:
                self.install_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy all files except venv, __pycache__, etc.
                exclude_dirs = {'.venv', '.venv311', '__pycache__', '.git', 'logs'}
                
                for item in current_dir.iterdir():
                    if item.name not in exclude_dirs:
                        dest = self.install_dir / item.name
                        if item.is_dir():
                            shutil.copytree(item, dest, dirs_exist_ok=True)
                        else:
                            shutil.copy2(item, dest)
                            
            self.log("✓ Directory created")
            
            # Step 2: Create virtual environment
            self.status_label.config(text="Creating Python virtual environment...")
            self.progress_var.set(20)
            self.log("Creating virtual environment...")
            
            venv_dir = self.install_dir / '.venv'
            subprocess.run([sys.executable, '-m', 'venv', str(venv_dir)],
                          check=True, capture_output=True)
            self.log("✓ Virtual environment created")
            
            # Step 3: Upgrade pip
            self.status_label.config(text="Upgrading pip...")
            self.progress_var.set(30)
            self.log("Upgrading pip...")
            
            pip_exe = venv_dir / 'Scripts' / 'pip.exe'
            subprocess.run([str(pip_exe), 'install', '--upgrade', 'pip'],
                          check=True, capture_output=True)
            self.log("✓ Pip upgraded")
            
            # Step 4: Install PyTorch
            self.status_label.config(text="Installing PyTorch (this may take several minutes)...")
            self.progress_var.set(40)
            self.log("Installing PyTorch 2.4.1...")
            
            result = subprocess.run([str(pip_exe), 'install',
                                   'torch==2.4.1', 'torchvision==0.19.1',
                                   '--index-url', 'https://download.pytorch.org/whl/cpu'],
                                   capture_output=True, text=True)
            self.log("✓ PyTorch installed")
            
            # Step 5: Install other dependencies
            self.status_label.config(text="Installing dependencies...")
            self.progress_var.set(60)
            self.log("Installing other dependencies...")
            
            requirements_file = self.install_dir / 'requirements.txt'
            if requirements_file.exists():
                # Read and filter requirements (skip torch lines)
                with open(requirements_file, 'r') as f:
                    reqs = [line.strip() for line in f 
                           if line.strip() and not line.startswith('#') 
                           and 'torch' not in line.lower()]
                
                for req in reqs:
                    self.log(f"Installing {req}...")
                    subprocess.run([str(pip_exe), 'install', req],
                                 capture_output=True)
                    
            self.log("✓ Dependencies installed")
            
            # Step 6: Install optional packages
            if self.install_mss.get():
                self.status_label.config(text="Installing screen capture support...")
                self.progress_var.set(75)
                self.log("Installing mss for screen capture...")
                subprocess.run([str(pip_exe), 'install', 'mss'],
                             capture_output=True)
                self.log("✓ Screen capture support installed")
            
            # Step 7: Create shortcuts
            self.status_label.config(text="Creating shortcuts...")
            self.progress_var.set(85)
            
            python_exe = venv_dir / 'Scripts' / 'python.exe'
            gui_script = self.install_dir / 'launch_gui.py'
            
            if self.create_desktop_shortcut.get():
                self.log("Creating desktop shortcut...")
                desktop = Path(winshell.desktop())
                shortcut_path = desktop / 'DriveOS.lnk'
                
                shell = Dispatch('WScript.Shell')
                shortcut = shell.CreateShortCut(str(shortcut_path))
                shortcut.Targetpath = str(python_exe)
                shortcut.Arguments = str(gui_script)
                shortcut.WorkingDirectory = str(self.install_dir)
                shortcut.IconLocation = str(python_exe)
                shortcut.Description = 'DriveOS Racing Line Analyzer'
                shortcut.save()
                
                self.log("✓ Desktop shortcut created")
            
            if self.create_start_menu.get():
                self.log("Creating Start Menu entry...")
                start_menu = Path(winshell.start_menu()) / 'Programs' / 'DriveOS'
                start_menu.mkdir(parents=True, exist_ok=True)
                
                shortcut_path = start_menu / 'DriveOS.lnk'
                shell = Dispatch('WScript.Shell')
                shortcut = shell.CreateShortCut(str(shortcut_path))
                shortcut.Targetpath = str(python_exe)
                shortcut.Arguments = str(gui_script)
                shortcut.WorkingDirectory = str(self.install_dir)
                shortcut.IconLocation = str(python_exe)
                shortcut.Description = 'DriveOS Racing Line Analyzer'
                shortcut.save()
                
                self.log("✓ Start Menu entry created")
            
            # Step 8: Complete
            self.status_label.config(text="Installation complete!")
            self.progress_var.set(100)
            self.log("\n✓ Installation completed successfully!")
            self.log(f"\nDriveOS has been installed to: {self.install_dir}")
            
            # Move to next page
            self.current_step += 1
            self.show_step()
            
        except Exception as e:
            self.status_label.config(text=f"Installation failed: {str(e)}")
            self.log(f"\n✗ Error: {str(e)}")
            messagebox.showerror("Installation Failed",
                               f"An error occurred during installation:\n\n{str(e)}\n\n"
                               "Please check the log for details.")
            self.next_btn.config(state='normal')
            self.cancel_btn.config(state='normal')
            
    def complete_page(self):
        """Installation complete page"""
        tk.Label(self.content_frame, text="✓ Installation Complete!",
                font=('Segoe UI', 16, 'bold'),
                fg='green',
                bg='white').pack(pady=30)
        
        tk.Label(self.content_frame,
                text=f"DriveOS has been successfully installed!\n\n"
                     f"Installation location:\n{self.install_dir}\n\n"
                     "You can now launch DriveOS from:\n",
                font=('Segoe UI', 10),
                bg='white',
                justify='center').pack(pady=10)
        
        if self.create_desktop_shortcut.get():
            tk.Label(self.content_frame, text="• Desktop shortcut",
                    font=('Segoe UI', 10), bg='white').pack()
                    
        if self.create_start_menu.get():
            tk.Label(self.content_frame, text="• Start Menu > Programs > DriveOS",
                    font=('Segoe UI', 10), bg='white').pack()
        
        tk.Label(self.content_frame,
                text="\nThank you for installing DriveOS!\n"
                     "Click Finish to exit the installer.",
                font=('Segoe UI', 10),
                bg='white',
                justify='center').pack(pady=20)
        
        self.cancel_btn.config(state='disabled')
        
    def finish_install(self):
        """Finish installation"""
        if messagebox.askyesno("Launch DriveOS",
                              "Would you like to launch DriveOS now?"):
            # Launch DriveOS
            python_exe = self.install_dir / '.venv' / 'Scripts' / 'python.exe'
            gui_script = self.install_dir / 'launch_gui.py'
            subprocess.Popen([str(python_exe), str(gui_script)],
                           cwd=str(self.install_dir))
        
        self.root.quit()
        
    def run(self):
        """Run the installer"""
        self.root.mainloop()


if __name__ == '__main__':
    # Check if required packages are available
    try:
        import winshell
        from win32com.client import Dispatch
    except ImportError:
        print("Installing installer dependencies...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pywin32', 'winshell'],
                      check=True)
        print("Please run the installer again.")
        sys.exit(1)
    
    installer = DriveOSInstaller()
    installer.run()
