"""
DriveOS GUI Launcher
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import src
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    # Import the GUI entrypoint from the project source
    from src.gui import launch_gui
except ModuleNotFoundError as e:
    # Helpful, actionable message for missing dependencies (e.g. cv2 / OpenCV)
    missing = e.name if hasattr(e, 'name') else str(e)
    print('\nERROR: Missing dependency:', missing)
    print('DriveOS requires a set of Python packages to run. Install them using:')
    print('\n    python -m pip install -r config/requirements.txt\n')
    print('Or to install just the missing package, for example:')
    print('\n    python -m pip install opencv-python\n')
    print('After installing dependencies, re-run this launcher script.')
    # exit with a non-zero status so shell callers know something failed
    sys.exit(1)


if __name__ == '__main__':
    launch_gui()
