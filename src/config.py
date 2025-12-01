"""
Configuration management for DriveOS
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create necessary directories
LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


class Config:
    """Application configuration"""
    
    # Model Settings
    MODEL_PATH: str = os.getenv("MODEL_PATH", str(MODELS_DIR / "racing_line_model.pth"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
    NMS_THRESHOLD: float = float(os.getenv("NMS_THRESHOLD", "0.45"))
    
    # Video Processing
    VIDEO_FPS: int = int(os.getenv("VIDEO_FPS", "30"))
    FRAME_WIDTH: int = int(os.getenv("FRAME_WIDTH", "1920"))
    FRAME_HEIGHT: int = int(os.getenv("FRAME_HEIGHT", "1080"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "8"))
    
    # Telemetry Settings
    TELEMETRY_SAMPLE_RATE: int = int(os.getenv("TELEMETRY_SAMPLE_RATE", "100"))
    GPS_ACCURACY_THRESHOLD: float = float(os.getenv("GPS_ACCURACY_THRESHOLD", "5.0"))
    
    # Redis Configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    
    # API Settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    WEBSOCKET_PORT: int = int(os.getenv("WEBSOCKET_PORT", "8001"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", str(LOGS_DIR / "driveos.log"))
    
    # Performance
    USE_GPU: bool = os.getenv("USE_GPU", "true").lower() == "true"
    NUM_WORKERS: int = int(os.getenv("NUM_WORKERS", "4"))
    
    # Directory Settings
    LOGS_DIR: Path = LOGS_DIR
    # Segmentation map debug saving (disabled by default to avoid large logs)
    SAVE_SEGMENTATION_MAPS: bool = os.getenv("SAVE_SEGMENTATION_MAPS", "false").lower() == "true"
    # Save every N predictions when enabled
    SEGMENTATION_SAVE_FREQ: int = int(os.getenv("SEGMENTATION_SAVE_FREQ", "30"))
    # Maximum number of segmentation map files to keep (older files will be removed). 0 = unlimited
    SEGMENTATION_MAX_FILES: int = int(os.getenv("SEGMENTATION_MAX_FILES", "200"))
    
    @classmethod
    def get_device(cls):
        """Get the computing device (GPU or CPU) - uses CPU for maximum compatibility"""
        import torch
        import logging
        logger = logging.getLogger(__name__)
        
        if not cls.USE_GPU:
            return torch.device("cpu")
        
        # DirectML has compatibility issues with LSTM operations
        # Use CUDA if available (for compatible GPUs)
        if torch.cuda.is_available():
            try:
                # Check if RTX 50 series (sm_120)
                gpu_name = torch.cuda.get_device_name(0)
                if "RTX 50" in gpu_name or "RTX50" in gpu_name:
                    logger.warning(f"GPU detected but unsupported: {gpu_name} (CUDA capability 12.0)")
                    logger.warning("RTX 50 series support requires a newer PyTorch build; falling back to CPU for now.")
                    return torch.device("cpu")
                
                # Test if GPU actually works with kernel execution
                test = torch.zeros(10, 10).cuda()
                result = test @ test  # Test matrix multiplication (kernel execution)
                del test, result
                return torch.device("cuda")
            except Exception as e:
                # GPU detected but not compatible
                logger.warning(f"GPU detected but not compatible: {e}")
                logger.warning("Falling back to CPU mode")
                pass
        
        # Use CPU for maximum compatibility (optimized with 16 threads)
        return torch.device("cpu")


config = Config()
