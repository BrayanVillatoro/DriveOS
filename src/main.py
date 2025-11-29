"""
DriveOS - Main entry point
This file maintains CLI compatibility but the primary interface is the GUI.
Launch the GUI with: python launch_gui.py or use DriveOS.bat
"""
import argparse
import logging
import sys
from pathlib import Path

from .config import config
from .inference import BatchProcessor

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point - Launch GUI by default"""
    parser = argparse.ArgumentParser(
        description='DriveOS - AI Racing Line Analyzer',
        epilog='For the full experience, use the GUI: python launch_gui.py or run DriveOS.bat'
    )
    
    parser.add_argument('--gui', action='store_true', default=True,
                       help='Launch GUI (default)')
    parser.add_argument('--analyze', type=str, metavar='VIDEO',
                       help='Analyze video file (command-line mode)')
    parser.add_argument('--output', type=str,
                       help='Output file path (for --analyze)')
    parser.add_argument('--model', type=str, default=config.MODEL_PATH,
                       help='Model weights path')
    
    args = parser.parse_args()
    
    # If video analysis requested via CLI
    if args.analyze:
        analyze_video_cli(args.analyze, args.output, args.model)
    else:
        # Launch GUI
        logger.info("Launching DriveOS GUI...")
        from .gui import launch_gui
        launch_gui()


def analyze_video_cli(video_path, output_path=None, model_path=None):
    """Simplified CLI video analysis"""
    logger.info(f"Analyzing video: {video_path}")
    
    # Default output to Videos folder if not specified
    if not output_path:
        videos_folder = Path.home() / "Videos"
        output_path = str(videos_folder / f"analyzed_{Path(video_path).name}")
    
    try:
        processor = BatchProcessor(model_path or config.MODEL_PATH)
        stats = processor.process_video(video_path, output_path)
        
        logger.info(f"✓ Analysis complete!")
        logger.info(f"  Processed {stats['total_frames']} frames")
        logger.info(f"  Average: {stats['avg_inference_time']:.2f}ms per frame")
        logger.info(f"  Saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"✗ Analysis failed: {str(e)}")
        logger.info("Tip: Use the GUI for a better experience - run DriveOS.bat or python launch_gui.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
