"""
Machine Learning models for racing line prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation
    Better for track detection with skip connections
    """
    
    def __init__(self, in_channels: int = 3, num_classes: int = 3, base_filters: int = 64):
        """
        Initialize U-Net
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes
            base_filters: Base number of filters (will be multiplied in deeper layers)
        """
        super(UNet, self).__init__()
        
        # Encoder (downsampling path)
        self.enc1 = self._make_encoder_block(in_channels, base_filters)
        self.enc2 = self._make_encoder_block(base_filters, base_filters * 2)
        self.enc3 = self._make_encoder_block(base_filters * 2, base_filters * 4)
        self.enc4 = self._make_encoder_block(base_filters * 4, base_filters * 8)
        
        # Bottleneck
        self.bottleneck = self._make_encoder_block(base_filters * 8, base_filters * 16)
        
        # Decoder (upsampling path with skip connections)
        self.upconv4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.dec4 = self._make_decoder_block(base_filters * 16, base_filters * 8)
        
        self.upconv3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = self._make_decoder_block(base_filters * 8, base_filters * 4)
        
        self.upconv2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = self._make_decoder_block(base_filters * 4, base_filters * 2)
        
        self.upconv1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = self._make_decoder_block(base_filters * 2, base_filters)
        
        # Final output layer
        self.out_conv = nn.Conv2d(base_filters, num_classes, kernel_size=1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def _make_encoder_block(self, in_channels: int, out_channels: int):
        """Create encoder block with two convolutions"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_block(self, in_channels: int, out_channels: int):
        """Create decoder block with two convolutions"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connections
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Segmentation map [B, num_classes, H, W]
        """
        # Encoder path with skip connections
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder path with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)  # Skip connection
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)  # Skip connection
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)  # Skip connection
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)  # Skip connection
        dec1 = self.dec1(dec1)
        
        # Final output
        out = self.out_conv(dec1)
        
        return out


class TelemetryLSTM(nn.Module):
    """
    LSTM model for telemetry sequence prediction
    Predicts optimal throttle/brake/steering based on track position
    """
    
    def __init__(self, input_size: int = 7, hidden_size: int = 128, 
                 num_layers: int = 2, output_size: int = 3):
        """
        Initialize telemetry LSTM
        
        Args:
            input_size: Number of input features (speed, position, etc.)
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            output_size: Number of outputs (throttle, brake, steering)
        """
        super(TelemetryLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, T, input_size]
            hidden: Optional hidden state
            
        Returns:
            Output predictions and hidden state
        """
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
        
        # Use last timestep output
        last_output = lstm_out[:, -1, :]
        predictions = self.fc(last_output)
        
        return predictions, hidden


class RacingLineOptimizer(nn.Module):
    """
    U-Net based model for track segmentation
    Simplified architecture focused on track detection only
    """
    
    def __init__(self, use_unet: bool = True):
        """
        Initialize racing line optimizer with U-Net
        
        Args:
            use_unet: Compatibility parameter (always uses U-Net now)
        """
        super(RacingLineOptimizer, self).__init__()
        
        # Use U-Net for track segmentation
        self.vision_model = UNet(in_channels=3, num_classes=3, base_filters=64)
        
        # Confidence estimation from segmentation output
        self.confidence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )
        
        self.telemetry_model = TelemetryLSTM()
        
        # Fusion layer for combined predictions
        self.fusion = nn.Sequential(
            nn.Linear(4, 64),  # 1 from vision + 3 from telemetry
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Output: (optimal_x, optimal_y) on track
        )
        
    def forward(self, image: torch.Tensor, telemetry_seq: torch.Tensor):
        """
        Forward pass
        
        Args:
            image: Input image [B, 3, H, W]
            telemetry_seq: Telemetry sequence [B, T, features]
            
        Returns:
            Tuple of (optimal_line, segmentation_map, confidence)
        """
        # U-Net segmentation
        seg_map = self.vision_model(image)
        
        # Confidence estimation
        confidence_scalar = self.confidence_head(seg_map)
        confidence = confidence_scalar.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
        vision_features = confidence.view(image.size(0), -1)
        
        # Telemetry branch
        telemetry_pred, _ = self.telemetry_model(telemetry_seq)
        
        # Combine features
        combined = torch.cat([vision_features, telemetry_pred], dim=1)
        optimal_line = self.fusion(combined)
        
        return optimal_line, seg_map, confidence


class ModelTrainer:
    """Training utilities for racing line models"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            device: Computing device
        """
        self.model = model.to(device)
        self.device = device
        
    def train_step(self, images: torch.Tensor, telemetry: torch.Tensor,
                   targets: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """
        Single training step
        
        Args:
            images: Input images
            telemetry: Telemetry data
            targets: Target racing lines
            optimizer: Optimizer
            
        Returns:
            Loss value
        """
        self.model.train()
        optimizer.zero_grad()
        
        # Forward pass
        predictions, seg_map, confidence = self.model(
            images.to(self.device), 
            telemetry.to(self.device)
        )
        
        # Calculate loss
        mse_loss = F.mse_loss(predictions, targets.to(self.device))
        
        # Backpropagation
        mse_loss.backward()
        optimizer.step()
        
        return mse_loss.item()
    
    def validate(self, val_loader) -> float:
        """
        Validation pass
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, telemetry, targets in val_loader:
                predictions, _, _ = self.model(
                    images.to(self.device),
                    telemetry.to(self.device)
                )
                
                loss = F.mse_loss(predictions, targets.to(self.device))
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def save_checkpoint(self, path: str, epoch: int, optimizer: torch.optim.Optimizer):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        logger.info(f"Checkpoint loaded from {path} (epoch {epoch})")
        return epoch
