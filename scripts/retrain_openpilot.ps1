# DriveOS Retrain Script - openpilot Architecture
# Generates edge labels and retrains model with edge regression

Write-Host "===== DriveOS Retrain (openpilot-inspired) =====" -ForegroundColor Cyan

# Step 1: Install dependencies
Write-Host "`n[1/4] Checking dependencies..." -ForegroundColor Yellow
pip install opencv-python numpy torch tqdm scipy

# Step 2: Generate edge labels from masks
Write-Host "`n[2/4] Generating edge annotations from masks..." -ForegroundColor Yellow
python .\scripts\generate_edge_labels.py --data-dir .\data\user_annotations

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error generating edge labels. Check that masks exist in data\user_annotations\masks\" -ForegroundColor Red
    exit 1
}

# Step 3: Configure training
Write-Host "`n[3/4] Configuring training..." -ForegroundColor Yellow
$env:USE_EDGE_HEAD = "true"
$env:USE_BOUNDARY_LOSS = "true"
$env:EDGE_LOSS_WEIGHT = "1.0"
$env:EDGE_CONF_WEIGHT = "0.5"
$env:BOUNDARY_LOSS_WEIGHT = "0.2"
$env:HORIZON_ROW_RATIO = "0.40"

Write-Host "  - Edge regression: ENABLED" -ForegroundColor Green
Write-Host "  - Boundary loss: ENABLED" -ForegroundColor Green
Write-Host "  - Edge loss weight: 1.0" -ForegroundColor Green

# Step 4: Train model
Write-Host "`n[4/4] Training model with edge regression..." -ForegroundColor Yellow
python .\src\train.py --data-dir .\data\user_annotations --output-dir .\models --epochs 50 --batch-size 8 --lr 0.001 --device auto

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n===== Training Complete! =====" -ForegroundColor Green
    Write-Host "Best model saved to: .\models\racing_line_model.pth" -ForegroundColor Green
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "  1. Process a video to see improved edge detection" -ForegroundColor White
    Write-Host "  2. Run: python .\launchers\launch_gui.pyw" -ForegroundColor White
} else {
    Write-Host "`n===== Training Failed =====" -ForegroundColor Red
    Write-Host "Check logs above for errors" -ForegroundColor Red
    exit 1
}
