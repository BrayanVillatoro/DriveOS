"""
Quick script to prepare sample training data from pre-labeled images
Use this when you have images with visible racing lines (like game screenshots)
"""
import cv2
import numpy as np
from pathlib import Path
import argparse


def extract_yellow_line_mask(image_path: str, output_dir: str):
    """
    Extract yellow racing line from game footage and create training mask
    
    Args:
        image_path: Path to image with yellow racing line
        output_dir: Directory to save processed data
    """
    output_path = Path(output_dir)
    images_dir = output_path / 'images'
    masks_dir = output_path / 'masks'
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return
    
    # Convert to HSV for better yellow detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define yellow color range
    # Yellow in HSV: Hue ~20-40, high Saturation and Value
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    
    # Create mask for yellow pixels
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Clean up mask with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    
    # Dilate to make line thicker
    kernel_dilate = np.ones((5, 5), np.uint8)
    yellow_mask = cv2.dilate(yellow_mask, kernel_dilate, iterations=2)
    
    # Convert mask: 0=background, 1=racing line
    mask = (yellow_mask > 0).astype(np.uint8)
    
    # Save files
    filename = Path(image_path).stem
    cv2.imwrite(str(images_dir / f"{filename}.jpg"), img)
    cv2.imwrite(str(masks_dir / f"{filename}.png"), mask)
    
    print(f"✓ Processed: {filename}")
    print(f"  Saved to: {output_dir}")
    
    # Show preview
    preview = img.copy()
    preview[yellow_mask > 0] = [0, 255, 255]  # Highlight yellow line in cyan
    
    # Resize for display
    h, w = preview.shape[:2]
    if w > 1280:
        scale = 1280 / w
        preview = cv2.resize(preview, (1280, int(h * scale)))
    
    cv2.imshow('Preview (press any key to continue)', preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_directory(input_dir: str, output_dir: str):
    """
    Process all images in a directory
    
    Args:
        input_dir: Directory containing images with yellow racing lines
        output_dir: Directory to save processed training data
    """
    input_path = Path(input_dir)
    
    # Find all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(input_path.glob(ext))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Processing...")
    
    for img_file in image_files:
        extract_yellow_line_mask(str(img_file), output_dir)
    
    print(f"\n✓ Processed {len(image_files)} images")
    print(f"Training data saved to: {output_dir}")
    print(f"\nYou can now train with:")
    print(f"  python -m src.train --data-dir {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create training data from images with visible yellow racing lines'
    )
    parser.add_argument('input', help='Input image file or directory')
    parser.add_argument('--output', '-o', default='data/sample_training',
                       help='Output directory for training data')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        extract_yellow_line_mask(args.input, args.output)
    elif input_path.is_dir():
        process_directory(args.input, args.output)
    else:
        print(f"Error: {args.input} is not a valid file or directory")
