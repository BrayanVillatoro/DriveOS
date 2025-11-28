"""
FastAPI application for DriveOS
"""
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import json
import logging
from pathlib import Path
import tempfile
import shutil

from .inference import InferenceEngine, BatchProcessor, RealtimeProcessor
from .telemetry_processor import TelemetryProcessor, TelemetryPoint
from .config import config

logger = logging.getLogger(__name__)

app = FastAPI(title="DriveOS API", version="0.1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global processor instance
realtime_processor: Optional[RealtimeProcessor] = None


class TelemetryData(BaseModel):
    """Telemetry data model"""
    timestamp: float
    speed: float
    throttle: float
    brake: float
    steering: float
    gear: int
    rpm: int
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class AnalysisRequest(BaseModel):
    """Video analysis request"""
    video_url: Optional[str] = None
    telemetry_url: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize application"""
    logger.info("Starting DriveOS API")
    global realtime_processor
    realtime_processor = RealtimeProcessor(config.MODEL_PATH)
    realtime_processor.start()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down DriveOS API")
    if realtime_processor:
        realtime_processor.stop()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "DriveOS API",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/analyze/video")
async def analyze_video(
    video: UploadFile = File(...),
    telemetry: Optional[UploadFile] = File(None)
):
    """
    Analyze racing video with optional telemetry
    
    Args:
        video: Video file
        telemetry: Optional telemetry CSV file
        
    Returns:
        Analysis results
    """
    try:
        # Save uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / video.filename
            output_path = Path(temp_dir) / f"analyzed_{video.filename}"
            
            with open(video_path, "wb") as f:
                shutil.copyfileobj(video.file, f)
            
            telemetry_path = None
            if telemetry:
                telemetry_path = Path(temp_dir) / telemetry.filename
                with open(telemetry_path, "wb") as f:
                    shutil.copyfileobj(telemetry.file, f)
            
            # Process video
            processor = BatchProcessor(config.MODEL_PATH)
            stats = processor.process_video(
                str(video_path),
                str(output_path),
                str(telemetry_path) if telemetry_path else None
            )
            
            # Generate insights
            if telemetry_path:
                tel_processor = TelemetryProcessor()
                df = tel_processor.load_from_csv(str(telemetry_path))
                metrics = tel_processor.calculate_racing_metrics(df)
                corners = tel_processor.detect_corners(df)
                corner_analysis = tel_processor.analyze_corner_performance(df, corners)
                insights = tel_processor.generate_insights(metrics, corner_analysis)
                
                return {
                    "status": "success",
                    "stats": stats,
                    "metrics": metrics,
                    "corners": corner_analysis,
                    "insights": insights,
                    "output_video": str(output_path)
                }
            
            return {
                "status": "success",
                "stats": stats,
                "output_video": str(output_path)
            }
            
    except Exception as e:
        logger.error(f"Error analyzing video: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.post("/telemetry/analyze")
async def analyze_telemetry(telemetry: UploadFile = File(...)):
    """
    Analyze telemetry data
    
    Args:
        telemetry: Telemetry CSV file
        
    Returns:
        Telemetry analysis
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            shutil.copyfileobj(telemetry.file, temp_file)
            temp_path = temp_file.name
        
        processor = TelemetryProcessor()
        df = processor.load_from_csv(temp_path)
        
        # Calculate metrics
        metrics = processor.calculate_racing_metrics(df)
        corners = processor.detect_corners(df)
        corner_analysis = processor.analyze_corner_performance(df, corners)
        insights = processor.generate_insights(metrics, corner_analysis)
        
        return {
            "status": "success",
            "metrics": metrics,
            "corners": corner_analysis,
            "insights": insights
        }
        
    except Exception as e:
        logger.error(f"Error analyzing telemetry: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time analysis
    
    Clients can send frames and telemetry, receive predictions
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_json()
            
            if data.get("type") == "telemetry":
                # Add telemetry point
                telemetry = TelemetryPoint(
                    timestamp=data["timestamp"],
                    speed=data["speed"],
                    throttle=data["throttle"],
                    brake=data["brake"],
                    steering=data["steering"],
                    gear=data["gear"],
                    rpm=data["rpm"]
                )
                realtime_processor.add_telemetry(telemetry)
            
            # Check for results
            result = realtime_processor.get_result(timeout=0.01)
            if result:
                frame, prediction = result
                
                # Send prediction back to client
                await websocket.send_json({
                    "type": "prediction",
                    "optimal_line": prediction["optimal_line"].tolist(),
                    "confidence": float(prediction["confidence"].mean()),
                    "inference_time_ms": prediction["inference_time_ms"]
                })
            
            await asyncio.sleep(0.01)
            
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


@app.get("/insights/compare")
async def compare_laps(lap1_file: UploadFile = File(...), 
                       lap2_file: UploadFile = File(...)):
    """
    Compare two laps
    
    Args:
        lap1_file: First lap telemetry
        lap2_file: Second lap telemetry
        
    Returns:
        Comparison results
    """
    try:
        from .telemetry_processor import TelemetryComparator
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp1:
            shutil.copyfileobj(lap1_file.file, temp1)
            lap1_path = temp1.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp2:
            shutil.copyfileobj(lap2_file.file, temp2)
            lap2_path = temp2.name
        
        processor = TelemetryProcessor()
        lap1_df = processor.load_from_csv(lap1_path)
        lap2_df = processor.load_from_csv(lap2_path)
        
        comparator = TelemetryComparator()
        comparison = comparator.compare_laps(lap1_df, lap2_df)
        time_diff = comparator.find_time_differences(lap1_df, lap2_df)
        
        return {
            "status": "success",
            "comparison": comparison,
            "time_differences": time_diff.tolist()
        }
        
    except Exception as e:
        logger.error(f"Error comparing laps: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
