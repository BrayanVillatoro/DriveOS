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
    """Dataset for racing line detection training with edge annotations"""
    
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
        
        # Load edge annotations if available
        edge_json_path = self.data_dir / 'edge_annotations.json'
        self.edge_annotations = {}
        if edge_json_path.exists():
            try:
                with open(edge_json_path, 'r') as f:
                    self.edge_annotations = json.load(f)
                logger.info(f"Loaded edge annotations: {len(self.edge_annotations)} frames")
            except Exception as e:
                logger.warning(f"Could not load edge annotations: {e}")
        
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
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get training sample
        
        Returns:
            image: Input image tensor [3, H, W]
            mask: Segmentation mask tensor [H, W]
            racing_line: Racing line coordinates [2] (x, y normalized to [-1, 1])
            edges: Edge coordinates [2, 33, 2] or zeros if not available
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
        
        # IMPROVED: Better mask preprocessing for training
        
        # Merge edge class (3) into track (0) if configured to reduce fragmentation
        if getattr(config, 'MERGE_EDGE_INTO_TRACK', True):
            mask[mask == 3] = 0
        # Keep only classes 0,1,2 (track, racing_line, off_track). Curbs (4) -> track.
        mask[mask == 4] = 0
        
        # Dilation of racing line to increase pixel footprint (improves class balance)
        if (mask == 1).sum() > 0:
            k = int(getattr(config, 'RACING_LINE_DILATE_KERNEL', 7))
            if k > 1:
                try:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                    rl = (mask == 1).astype(np.uint8)
                    rl_dil = cv2.dilate(rl, kernel, iterations=1)
                    mask[(rl_dil > 0)] = 1
                except Exception:
                    pass
        
        # IMPROVED: Additional track cleanup to reduce fragmentation
        # Close small gaps in track surface
        track_mask = (mask == 0).astype(np.uint8)
        if track_mask.sum() > 0:
            try:
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_CLOSE, kernel_close)
                
                # Select largest connected component to remove small fragments
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(track_mask, connectivity=8)
                if num_labels > 1:
                    # Find largest component (excluding background)
                    sizes = stats[1:, cv2.CC_STAT_AREA]
                    largest_id = np.argmax(sizes) + 1
                    track_mask = (labels == largest_id).astype(np.uint8)
                
                # Update mask with cleaned track
                mask[track_mask == 0] = 2  # Non-track -> off-track
                mask[track_mask == 1] = 0  # Track remains track
                
                # Restore racing line on top of track
                if (rl_dil > 0).sum() > 0:
                    mask[rl_dil > 0] = 1
            except Exception:
                pass
        
        # Clamp remaining values to 0..2 range
        mask = np.clip(mask, 0, 2)
        
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
        
        # Load edge annotations if available
        frame_id = img_path.stem
        if frame_id in self.edge_annotations:
            edge_data = self.edge_annotations[frame_id]
            left_edge = np.array(edge_data['left_edge'], dtype=np.float32)
            right_edge = np.array(edge_data['right_edge'], dtype=np.float32)
            edges = np.stack([left_edge, right_edge], axis=0)  # [2, 33, 2]
        else:
            # No edge annotations - create zeros placeholder
            from .edge_constants import EdgeConstants
            edges = np.zeros((EdgeConstants.NUM_ROAD_EDGES, EdgeConstants.IDX_N, EdgeConstants.EDGE_WIDTH), dtype=np.float32)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).long()
        racing_line_tensor = torch.tensor([racing_line_x, racing_line_y], dtype=torch.float32)
        edges_tensor = torch.from_numpy(edges).float()
        
        return image_tensor, mask_tensor, racing_line_tensor, edges_tensor


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
    # Use U-Net architecture with edge regression head
    use_edge_head = bool(getattr(config, 'USE_EDGE_HEAD', True))
    model = RacingLineOptimizer(use_unet=True, use_edge_head=use_edge_head)
    logger.info(f"Training with U-Net architecture (edge_head={use_edge_head})")
    trainer = ModelTrainer(model, device)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Loss functions
    # Class weights (track, racing_line, off_track) to compensate imbalance
    try:
        weights = [float(x) for x in getattr(config, 'CLASS_WEIGHTS', '0.2,5.0,1.0').split(',')]
        if len(weights) != 3:
            raise ValueError
        weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion_seg = nn.CrossEntropyLoss(weight=weight_tensor)
    except Exception:
        criterion_seg = nn.CrossEntropyLoss()
    criterion_line = nn.MSELoss()
    
    # Edge coordinate loss
    use_edge_loss = use_edge_head and len(train_dataset.edge_annotations) > 0
    edge_weight = float(getattr(config, 'EDGE_LOSS_WEIGHT', 1.0))
    edge_conf_weight = float(getattr(config, 'EDGE_CONF_WEIGHT', 0.5))
    
    def edge_coordinate_loss(pred_edges, gt_edges, edge_probs=None):
        """L1 loss on edge coordinates with optional confidence weighting"""
        if not use_edge_loss:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        # Check if GT edges are non-zero (valid)
        valid_mask = (gt_edges.abs().sum(dim=-1) > 0).float()  # [B, 2, 33]
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        # Coordinate L1 loss
        coord_diff = torch.abs(pred_edges - gt_edges)
        masked_diff = coord_diff * valid_mask.unsqueeze(-1)
        coord_loss = masked_diff.sum() / (valid_mask.sum() + 1e-6)
        
        # Edge confidence loss (BCE with valid edges as positive)
        conf_loss = torch.tensor(0.0, device=device)
        if edge_probs is not None:
            valid_edges = valid_mask.max(dim=-1)[0]  # [B, 2]
            conf_loss = nn.functional.binary_cross_entropy(edge_probs, valid_edges)
        
        return coord_loss, conf_loss

    # Optional: boundary-aware loss to sharpen edges
    use_boundary_loss = bool(getattr(config, 'USE_BOUNDARY_LOSS', False))
    boundary_weight = float(getattr(config, 'BOUNDARY_LOSS_WEIGHT', 0.2))
    horizon_ratio = float(getattr(config, 'HORIZON_ROW_RATIO', 0.40))
    
    logger.info(f"Loss configuration: edge={use_edge_loss}, boundary={use_boundary_loss}")

    def boundary_loss_fn(seg_logits, masks):
        # seg_logits: [B, C, H, W], masks: [B, H, W]
        if not use_boundary_loss:
            return torch.tensor(0.0, device=device)
        with torch.no_grad():
            # Focus on road vs background for boundary extraction
            road_pred = torch.softmax(seg_logits, dim=1)[:, 0]  # class 0 assumed road
            road_gt = (masks == 0).float()
            # Clamp to lower ROI
            B, H, W = road_gt.shape
            horizon = int(H * horizon_ratio)
            road_pred[:, :horizon, :] = 0
            road_gt[:, :horizon, :] = 0
            # Sobel edges
            def sobel(x):
                kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=device).view(1,1,3,3)
                ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=device).view(1,1,3,3)
                x = x.unsqueeze(1)
                gx = torch.nn.functional.conv2d(x, kx, padding=1)
                gy = torch.nn.functional.conv2d(x, ky, padding=1)
                return torch.sqrt(gx**2 + gy**2 + 1e-6).squeeze(1)
            pred_edges = sobel(road_pred)
            gt_edges = sobel(road_gt)
        # L1 between edge maps
        return torch.mean(torch.abs(pred_edges - gt_edges))
    
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
        for batch_data in pbar:
            images, masks, racing_lines, gt_edges = batch_data
            images = images.to(device)
            masks = masks.to(device)
            racing_lines = racing_lines.to(device)
            gt_edges = gt_edges.to(device)
            
            # Create dummy telemetry (all zeros for now)
            batch_size_iter = images.size(0)
            telemetry = torch.zeros(batch_size_iter, 10, 7).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions, seg_maps, confidence, edge_outputs = model(images, telemetry)
            
            # Calculate losses
            seg_loss = criterion_seg(seg_maps, masks)
            line_loss = criterion_line(predictions, racing_lines)
            
            # Edge coordinate loss
            edge_coord_loss = torch.tensor(0.0, device=device)
            edge_conf_loss = torch.tensor(0.0, device=device)
            if edge_outputs is not None:
                edge_coord_loss, edge_conf_loss = edge_coordinate_loss(
                    edge_outputs['edges'], gt_edges, edge_outputs['edge_probs']
                )
            
            # Boundary-aware loss
            b_loss = boundary_loss_fn(seg_maps, masks)
            
            # Combined loss
            loss = (seg_loss + line_loss + 
                   boundary_weight * b_loss + 
                   edge_weight * edge_coord_loss + 
                   edge_conf_weight * edge_conf_loss)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            train_seg_loss += seg_loss.item()
            train_line_loss += line_loss.item()
            
            metrics = {
                'loss': f'{loss.item():.4f}',
                'seg': f'{seg_loss.item():.4f}',
                'line': f'{line_loss.item():.4f}'
            }
            if use_boundary_loss:
                metrics['bdry'] = f'{b_loss.item():.4f}'
            if use_edge_loss and edge_outputs is not None:
                metrics['edge'] = f'{edge_coord_loss.item():.4f}'
            pbar.set_postfix(metrics)
        
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
            for batch_data in val_loader:
                images, masks, racing_lines, gt_edges = batch_data
                images = images.to(device)
                masks = masks.to(device)
                racing_lines = racing_lines.to(device)
                gt_edges = gt_edges.to(device)
                
                batch_size_val = images.size(0)
                telemetry = torch.zeros(batch_size_val, 10, 7).to(device)
                
                predictions, seg_maps, confidence, edge_outputs = model(images, telemetry)
                
                seg_loss = criterion_seg(seg_maps, masks)
                line_loss = criterion_line(predictions, racing_lines)
                
                edge_coord_loss_val = torch.tensor(0.0, device=device)
                edge_conf_loss_val = torch.tensor(0.0, device=device)
                if edge_outputs is not None:
                    edge_coord_loss_val, edge_conf_loss_val = edge_coordinate_loss(
                        edge_outputs['edges'], gt_edges, edge_outputs['edge_probs']
                    )
                
                b_loss = boundary_loss_fn(seg_maps, masks)
                loss = (seg_loss + line_loss + 
                       boundary_weight * b_loss + 
                       edge_weight * edge_coord_loss_val + 
                       edge_conf_weight * edge_conf_loss_val)
                
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
