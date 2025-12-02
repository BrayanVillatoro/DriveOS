"""
Simple road surface detection preview script
Run on a video to test track surface detection
"""
import cv2
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import InferenceEngine
from src.config import config

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Preview road surface detection')
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--output', default='road_preview.mp4', help='Output video path')
    parser.add_argument('--model', default=None, help='Model path (uses config default if not specified)')
    args = parser.parse_args()
    
    # Initialize engine
    model_path = args.model or config.MODEL_PATH
    print(f"Initializing with model: {model_path}")
    engine = InferenceEngine(model_path)
    
    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
    
    print(f"Processing {total_frames} frames at {fps} FPS...")
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            prediction = engine.predict(frame)
            result = engine.visualize_prediction(frame, prediction)
            
            # Write frame
            out.write(result)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
    
    finally:
        cap.release()
        out.release()
    
    print(f"✓ Output saved to: {args.output}")

if __name__ == '__main__':
    main()
