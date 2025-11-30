"""
DriveOS GUI Launcher
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import src
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from src.gui import launch_gui

if __name__ == '__main__':
    launch_gui()
