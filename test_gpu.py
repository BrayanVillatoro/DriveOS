"""Test RTX 5070 Ti GPU support with PyTorch nightly"""
import torch
from src.models import RacingLineOptimizer

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device name:", torch.cuda.get_device_name(0))
print()

try:
    # Test model on GPU
    print("Loading model on GPU...")
    model = RacingLineOptimizer().cuda()
    model.eval()
    
    print("Creating test inputs...")
    img = torch.randn(1, 3, 320, 320).cuda()
    tel = torch.randn(1, 100, 5).cuda()
    
    print("Running inference...")
    with torch.no_grad():
        line, seg, conf = model(img, tel)
    
    print("✓ SUCCESS! Full model inference works on RTX 5070 Ti!")
    print(f"Output shapes: line={line.shape}, seg={seg.shape}, conf={conf.shape}")
    print(f"Output device: {line.device}")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
