"""
Machine Learning models for racing line prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RacingLineDetector(nn.Module):
    """
    Deep learning model for detecting optimal racing line from video
    Uses a combination of semantic segmentation and regression
    """
    
    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        """
        Initialize racing line detector
        
        Args:
            num_classes: Number of output classes (track, racing_line, off_track)
            pretrained: Use pretrained backbone
        """
        super(RacingLineDetector, self).__init__()
        
        # Use ResNet50 as backbone with DeepLabV3
        self.backbone = models.segmentation.deeplabv3_resnet50(
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # Additional regression head for line confidence
        self.confidence_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Tuple of (segmentation_map, confidence_map)
        """
        # Segmentation output
        seg_output = self.backbone(x)['out']
        
        # Extract features for confidence head
        features = self.backbone.backbone.layer3(
            self.backbone.backbone.layer2(
                self.backbone.backbone.layer1(
                    self.backbone.backbone.maxpool(
                        self.backbone.backbone.relu(
                            self.backbone.backbone.bn1(
                                self.backbone.backbone.conv1(x)
                            )
                        )
                    )
                )
            )
        )
        
        confidence = self.confidence_head(features)
        
        return seg_output, confidence


class TrackBoundaryDetector(nn.Module):
    """
    Detect track boundaries and drivable area
    """
    
    def __init__(self, pretrained: bool = True):
        super(TrackBoundaryDetector, self).__init__()
        
        # UNet-like architecture
        self.encoder = models.resnet34(pretrained=pretrained)
        
        # Decoder
        self.decoder = nn.ModuleList([
            self._make_decoder_block(512, 256),
            self._make_decoder_block(256, 128),
            self._make_decoder_block(128, 64),
            self._make_decoder_block(64, 32),
        ])
        
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        
    def _make_decoder_block(self, in_channels: int, out_channels: int):
        """Create decoder block"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Encoder
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        
        # Decoder
        for decoder_block in self.decoder:
            x = decoder_block(x)
        
        x = self.final_conv(x)
        return torch.sigmoid(x)


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
    Combined model that integrates vision and telemetry
    """
    
    def __init__(self):
        super(RacingLineOptimizer, self).__init__()
        
        self.vision_model = RacingLineDetector()
        self.telemetry_model = TelemetryLSTM()
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(640 * 640 + 3, 256),  # Vision features + telemetry
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Output: (optimal_x, optimal_y) on track
        )
        
    def forward(self, image: torch.Tensor, telemetry_seq: torch.Tensor):
        """
        Forward pass combining vision and telemetry
        
        Args:
            image: Input image [B, 3, H, W]
            telemetry_seq: Telemetry sequence [B, T, features]
            
        Returns:
            Optimal racing line coordinates
        """
        # Vision branch
        seg_map, confidence = self.vision_model(image)
        vision_features = F.adaptive_avg_pool2d(confidence, (1, 1)).view(image.size(0), -1)
        
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
