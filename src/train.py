"""
Training script for DriveOS racing line detection model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from typing import Tuple, List
import json

from .models import RacingLineOptimizer, ModelTrainer
from .config import config

logger = logging.getLogger(__name__)


class RacingLineDataset(Dataset):
    """Dataset for racing line detection training"""
    
    def __init__(self, data_dir: str, split: str = 'train'):
        """
        Initialize dataset
        
        Args:
            data_dir: Directory containing images/ and masks/ folders
            split: 'train' or 'val'
        """
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / 'images'
        self.mask_dir = self.data_dir / 'masks'
        
        # Get all image files
        self.image_files = sorted(list(self.image_dir.glob('*.jpg')))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
        
        # Split into train/val (80/20)
        split_idx = int(len(self.image_files) * 0.8)
        if split == 'train':
            self.image_files = self.image_files[:split_idx]
        else:
            self.image_files = self.image_files[split_idx:]
        
        logger.info(f"Loaded {len(self.image_files)} {split} samples")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get training sample
        
        Returns:
            image: Input image tensor [3, H, W]
            mask: Segmentation mask tensor [H, W]
            racing_line: Racing line coordinates [2] (x, y normalized to [-1, 1])
        """
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.mask_dir / img_path.name.replace('.jpg', '.png')
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # Create empty mask if file is missing/corrupted
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Resize
        target_size = (320, 320)
        image = cv2.resize(image, target_size)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        
        # Extract racing line point from mask (class 1)
        racing_line_pixels = np.where(mask == 1)
        if len(racing_line_pixels[0]) > 0:
            # Use center of racing line pixels
            y_center = int(np.mean(racing_line_pixels[0]))
            x_center = int(np.mean(racing_line_pixels[1]))
            
            # Normalize to [-1, 1]
            racing_line_x = (x_center / target_size[0]) * 2 - 1
            racing_line_y = (y_center / target_size[1]) * 2 - 1
        else:
            # Default to center if no racing line detected
            racing_line_x, racing_line_y = 0.0, 0.0
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).long()
        racing_line_tensor = torch.tensor([racing_line_x, racing_line_y], dtype=torch.float32)
        
        return image_tensor, mask_tensor, racing_line_tensor


def train_model(data_dir: str, 
                output_dir: str = 'models',
                epochs: int = 50,
                batch_size: int = 4,
                learning_rate: float = 0.001,
                device: str = 'auto'):
    """
    Train racing line detection model
    
    Args:
        data_dir: Directory containing training data
        output_dir: Directory to save model checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use ('auto', 'cuda', or 'cpu')
    """
    # Setup device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    logger.info(f"Training on device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = RacingLineDataset(data_dir, split='train')
    val_dataset = RacingLineDataset(data_dir, split='val')
    
    # Check minimum dataset size
    if len(train_dataset) < batch_size * 2:
        logger.warning(f"Training set has only {len(train_dataset)} samples, need at least {batch_size * 2} for stable training")
        logger.warning("Reducing batch size to fit available data")
        batch_size = max(2, len(train_dataset) // 2)  # Use at least 2 for BatchNorm
    
    if len(train_dataset) < 2:
        raise ValueError(f"Need at least 2 training samples, found {len(train_dataset)}. Generate more training data first.")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Use 0 for Windows compatibility
        drop_last=True  # Always drop incomplete batches to avoid BatchNorm issues
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1,  # Use batch size of 1 for validation to avoid issues
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    logger.info("Creating model...")
    model = RacingLineOptimizer()
    trainer = ModelTrainer(model, device)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Loss functions
    criterion_seg = nn.CrossEntropyLoss()
    criterion_line = nn.MSELoss()
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_seg_loss = 0
        train_line_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, masks, racing_lines in pbar:
            images = images.to(device)
            masks = masks.to(device)
            racing_lines = racing_lines.to(device)
            
            # Create dummy telemetry (all zeros for now)
            batch_size = images.size(0)
            telemetry = torch.zeros(batch_size, 10, 7).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions, seg_maps, confidence = model(images, telemetry)
            
            # Calculate losses
            seg_loss = criterion_seg(seg_maps, masks)
            line_loss = criterion_line(predictions, racing_lines)
            
            # Combined loss
            loss = seg_loss + line_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            train_seg_loss += seg_loss.item()
            train_line_loss += line_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'seg': f'{seg_loss.item():.4f}',
                'line': f'{line_loss.item():.4f}'
            })
        
        # Average training losses
        train_loss /= len(train_loader)
        train_seg_loss /= len(train_loader)
        train_line_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_seg_loss = 0
        val_line_loss = 0
        
        with torch.no_grad():
            for images, masks, racing_lines in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                racing_lines = racing_lines.to(device)
                
                batch_size = images.size(0)
                telemetry = torch.zeros(batch_size, 10, 7).to(device)
                
                predictions, seg_maps, confidence = model(images, telemetry)
                
                seg_loss = criterion_seg(seg_maps, masks)
                line_loss = criterion_line(predictions, racing_lines)
                loss = seg_loss + line_loss
                
                val_loss += loss.item()
                val_seg_loss += seg_loss.item()
                val_line_loss += line_loss.item()
        
        val_loss /= len(val_loader)
        val_seg_loss /= len(val_loader)
        val_line_loss /= len(val_loader)
        
        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f} (seg: {train_seg_loss:.4f}, line: {train_line_loss:.4f}) - "
            f"Val Loss: {val_loss:.4f} (seg: {val_seg_loss:.4f}, line: {val_line_loss:.4f})"
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = output_path / f'checkpoint_epoch_{epoch+1}.pth'
            trainer.save_checkpoint(str(checkpoint_path), epoch, optimizer)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = output_path / 'racing_line_model.pth'
            trainer.save_checkpoint(str(best_model_path), epoch, optimizer)
            logger.info(f"✓ New best model saved! Val loss: {val_loss:.4f}")
    
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best model saved to: {output_path / 'racing_line_model.pth'}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train racing line detection model')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device
    )
