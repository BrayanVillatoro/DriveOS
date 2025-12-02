"""
Training Data Augmentation for DriveOS

This script applies augmentations to expand your training dataset:
- Brightness/contrast variations (simulate different lighting)
- Horizontal flips (mirror tracks for left/right balance)
- Small rotations and perspective shifts
- Noise and blur (simulate camera issues)

Usage:
    python augment_training_data.py --input-dir data/user_annotations --output-dir data/augmented --multiplier 5
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import random
from typing import Tuple
import albumentations as A
from tqdm import tqdm


def create_augmentation_pipeline():
    """
    Create augmentation pipeline that preserves masks
    """
    return A.Compose([
        # Color augmentations (only affect image, not mask)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        ], p=0.8),
        
        # Geometric augmentations (affect both image and mask)
        A.OneOf([
            A.ShiftScaleRotate(
                shift_limit=0.05, 
                scale_limit=0.1, 
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.7
            ),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
        ], p=0.5),
        
        # Noise and blur (only affect image)
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=0.5),
            A.MotionBlur(blur_limit=5, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.3),
        ], p=0.3),
        
        # Horizontal flip (for left/right track balance)
        A.HorizontalFlip(p=0.5),
    ])


def augment_dataset(input_dir: str, output_dir: str, multiplier: int = 5, 
                   preserve_originals: bool = True):
    """
    Augment training dataset
    
    Args:
        input_dir: Directory containing images/ and masks/
        output_dir: Directory to save augmented data
        multiplier: How many augmented versions per original (1 = no augmentation)
        preserve_originals: If True, copy originals to output as well
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    image_dir = input_path / 'images'
    mask_dir = input_path / 'masks'
    
    output_image_dir = output_path / 'images'
    output_mask_dir = output_path / 'masks'
    
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = sorted(list(image_dir.glob('*.jpg')))
    
    if len(image_files) == 0:
        print(f"Error: No images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Generating {multiplier}x augmentations per image")
    print(f"Total output: {len(image_files) * (multiplier + (1 if preserve_originals else 0))} images")
    
    # Create augmentation pipeline
    augment = create_augmentation_pipeline()
    
    output_idx = 0
    
    for img_path in tqdm(image_files, desc="Augmenting dataset"):
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not load {img_path}")
            continue
        
        # Load mask
        mask_path = mask_dir / img_path.name.replace('.jpg', '.png')
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"Warning: Could not load mask for {img_path}")
            continue
        
        # Save original if requested
        if preserve_originals:
            out_img_path = output_image_dir / f'frame_{output_idx:06d}.jpg'
            out_mask_path = output_mask_dir / f'frame_{output_idx:06d}.png'
            cv2.imwrite(str(out_img_path), image)
            cv2.imwrite(str(out_mask_path), mask)
            output_idx += 1
        
        # Generate augmented versions
        for aug_num in range(multiplier):
            try:
                # Apply augmentation
                augmented = augment(image=image, mask=mask)
                aug_image = augmented['image']
                aug_mask = augmented['mask']
                
                # Ensure mask values are still valid (0-4)
                aug_mask = np.clip(aug_mask, 0, 4)
                
                # Save augmented version
                out_img_path = output_image_dir / f'frame_{output_idx:06d}.jpg'
                out_mask_path = output_mask_dir / f'frame_{output_idx:06d}.png'
                
                cv2.imwrite(str(out_img_path), aug_image)
                cv2.imwrite(str(out_mask_path), aug_mask)
                
                output_idx += 1
            
            except Exception as e:
                print(f"Warning: Augmentation failed for {img_path}: {e}")
                continue
    
    print(f"\n✓ Augmentation complete!")
    print(f"Generated {output_idx} total training samples")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Augment training data for racing line detection'
    )
    parser.add_argument('--input-dir', type=str, default='data/user_annotations',
                       help='Input directory containing images/ and masks/')
    parser.add_argument('--output-dir', type=str, default='data/augmented',
                       help='Output directory for augmented data')
    parser.add_argument('--multiplier', type=int, default=5,
                       help='Number of augmented versions per original (default: 5)')
    parser.add_argument('--no-preserve-originals', action='store_true',
                       help='Do not copy original images to output')
    
    args = parser.parse_args()
    
    try:
        import albumentations
        augment_dataset(
            args.input_dir,
            args.output_dir,
            args.multiplier,
            preserve_originals=not args.no_preserve_originals
        )
    except ImportError:
        print("Error: albumentations library not found")
        print("Install with: pip install albumentations")
        print("\nAlternatively, manually duplicate and vary your training images")


if __name__ == '__main__':
    main()
