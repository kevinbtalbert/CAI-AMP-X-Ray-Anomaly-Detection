"""
X-Ray Anomaly Detection Application
FastAPI-based web application that calls a deployed Cloudera ML Model

This is the UI application that:
1. Provides a web interface for browsing X-ray images
2. Calls the deployed CML Model for analysis
3. Displays results with visual overlays
"""
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import base64
import io

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from PIL import Image
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="X-Ray Anomaly Detection",
    description="Medical X-ray analysis with visual anomaly localization",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration for CML Model endpoint
def get_cml_model_endpoint():
    """Auto-detect CML Model endpoint from environment."""
    endpoint = os.getenv("CML_MODEL_ENDPOINT")
    
    # If not set, try to construct from CDSW_DOMAIN
    if not endpoint:
        cdsw_domain = os.getenv("CDSW_DOMAIN")
        if cdsw_domain:
            endpoint = f"https://{cdsw_domain}/api/altus-ds-1/models/call-model"
            logger.info(f"Auto-detected CML Model endpoint: {endpoint}")
    
    return endpoint

CML_MODEL_ENDPOINT = get_cml_model_endpoint()
CML_MODEL_ACCESS_KEY = os.getenv("CML_MODEL_ACCESS_KEY")
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "true").lower() == "true"

# Global model variable (for local testing)
local_model = None

def load_local_model():
    """Load the local predict.py model for testing"""
    global local_model
    if local_model is None:
        logger.info("Loading local model from predict.py...")
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from predict import predict
        local_model = predict
        logger.info("Local model loaded successfully")
    return local_model

def call_cml_model(image_base64: str) -> dict:
    """
    Call the deployed CML Model endpoint.
    
    Args:
        image_base64: Base64 encoded image
        
    Returns:
        Model prediction response
    """
    if not CML_MODEL_ENDPOINT or not CML_MODEL_ACCESS_KEY:
        raise Exception("CML Model endpoint not configured. Set CML_MODEL_ENDPOINT and CML_MODEL_ACCESS_KEY in .env")
    
    payload = {
        "accessKey": CML_MODEL_ACCESS_KEY,
        "request": {
            "image_base64": image_base64
        }
    }
    
    response = requests.post(CML_MODEL_ENDPOINT, json=payload, timeout=60)
    
    if response.status_code == 200:
        return response.json()["response"]
    else:
        raise Exception(f"CML Model call failed: {response.status_code} - {response.text}")

# Pydantic models
class AnalysisResult(BaseModel):
    file_path: str
    anomaly_score: float
    risk_level: str
    top_findings: List[Dict[str, Any]]
    bounding_boxes: List[Dict[str, int]]
    heatmap_base64: str
    predictions: Dict[str, float]
    image_size: Dict[str, int]
    timestamp: str
    processing_time: float

class FileInfo(BaseModel):
    name: str
    path: str
    type: str  # 'normal' or 'pneumonia'
    size: int

# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application page"""
    html_path = Path(__file__).parent / "static" / "index.html"
    if not html_path.exists():
        return """
        <html>
            <head><title>X-Ray Anomaly Detection</title></head>
            <body>
                <h1>X-Ray Anomaly Detection API</h1>
                <p>API is running. Use the following endpoints:</p>
                <ul>
                    <li>GET /api/health - Check API health</li>
                    <li>GET /api/images - List available X-ray images</li>
                    <li>POST /api/analyze - Analyze an X-ray image</li>
                    <li>GET /docs - Interactive API documentation</li>
                </ul>
            </body>
        </html>
        """
    
    with open(html_path, 'r') as f:
        return f.read()

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check data directories
        normal_dir = Path("data/NORMAL")
        pneumonia_dir = Path("data/PNEUMONIA")
        sample_dir = Path("data/sample_xrays")
        
        normal_count = len(list(normal_dir.glob("*.jpeg"))) if normal_dir.exists() else 0
        pneumonia_count = len(list(pneumonia_dir.glob("*.jpeg"))) if pneumonia_dir.exists() else 0
        sample_count = len(list(sample_dir.glob("*.jpeg"))) if sample_dir.exists() else 0
        
        # Check model status
        if USE_LOCAL_MODEL:
            model_status = "local"
            try:
                load_local_model()
                model_ready = True
            except:
                model_ready = False
        else:
            model_status = "cml_endpoint"
            model_ready = bool(CML_MODEL_ENDPOINT and CML_MODEL_ACCESS_KEY)
        
        return {
            "status": "healthy" if model_ready else "degraded",
            "model": {
                "type": model_status,
                "ready": model_ready,
                "endpoint": CML_MODEL_ENDPOINT if not USE_LOCAL_MODEL else None
            },
            "data": {
                "normal_images": normal_count,
                "pneumonia_images": pneumonia_count,
                "sample_images": sample_count,
                "total": normal_count + pneumonia_count
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/images")
async def list_images(category: Optional[str] = None, limit: int = 50):
    """
    List available X-ray images
    
    Args:
        category: Filter by 'normal', 'pneumonia', or 'sample' (default: all)
        limit: Maximum number of images to return per category
    """
    try:
        images = []
        
        # Sample images (curated set)
        if category in [None, 'sample']:
            sample_dir = Path("data/sample_xrays")
            if sample_dir.exists():
                for img_path in list(sample_dir.glob("*.jpeg"))[:limit]:
                    img_type = "normal" if "normal" in img_path.name else "pneumonia"
                    images.append({
                        "name": img_path.name,
                        "path": str(img_path),
                        "type": img_type,
                        "category": "sample",
                        "size": img_path.stat().st_size
                    })
        
        # Normal X-rays
        if category in [None, 'normal']:
            normal_dir = Path("data/NORMAL")
            if normal_dir.exists():
                for img_path in list(normal_dir.glob("*.jpeg"))[:limit]:
                    images.append({
                        "name": img_path.name,
                        "path": str(img_path),
                        "type": "normal",
                        "category": "full_dataset",
                        "size": img_path.stat().st_size
                    })
        
        # Pneumonia X-rays
        if category in [None, 'pneumonia']:
            pneumonia_dir = Path("data/PNEUMONIA")
            if pneumonia_dir.exists():
                for img_path in list(pneumonia_dir.glob("*.jpeg"))[:limit]:
                    img_type = "bacteria" if "bacteria" in img_path.name else "virus"
                    images.append({
                        "name": img_path.name,
                        "path": str(img_path),
                        "type": f"pneumonia_{img_type}",
                        "category": "full_dataset",
                        "size": img_path.stat().st_size
                    })
        
        return {
            "images": images,
            "count": len(images),
            "category": category or "all"
        }
    
    except Exception as e:
        logger.error(f"Error listing images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/image/{file_path:path}")
async def get_image(file_path: str):
    """
    Get an X-ray image file
    """
    try:
        img_path = Path(file_path)
        if not img_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        
        return FileResponse(img_path, media_type="image/jpeg")
    
    except Exception as e:
        logger.error(f"Error retrieving image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze")
async def analyze_xray(file_path: str):
    """
    Analyze an X-ray image for anomalies
    
    Args:
        file_path: Path to the X-ray image file
    
    Returns:
        AnalysisResult with anomaly score, findings, heatmap, and bounding boxes
    """
    try:
        start_time = datetime.now()
        
        # Verify file exists
        img_path = Path(file_path)
        if not img_path.exists():
            raise HTTPException(status_code=404, detail=f"Image not found: {file_path}")
        
        logger.info(f"Analyzing X-ray: {file_path}")
        
        # Read and encode image
        with open(img_path, 'rb') as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Call model (CML or local)
        if USE_LOCAL_MODEL:
            logger.info("Using local model")
            model = load_local_model()
            result = model({"image": str(img_path)})
        else:
            logger.info("Calling CML Model endpoint")
            result = call_cml_model(image_b64)
        
        # Determine risk level
        anomaly_score = result['anomaly_score']
        if anomaly_score >= 0.7:
            risk_level = "HIGH RISK"
        elif anomaly_score >= 0.4:
            risk_level = "SUSPICIOUS"
        else:
            risk_level = "NORMAL"
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Build response
        analysis_result = AnalysisResult(
            file_path=file_path,
            anomaly_score=anomaly_score,
            risk_level=risk_level,
            top_findings=result['top_findings'],
            bounding_boxes=result['bounding_boxes'],
            heatmap_base64=result['heatmap_base64'],
            predictions=result['predictions'],
            image_size=result['image_size'],
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
        
        logger.info(f"Analysis completed in {processing_time:.2f}s - Risk: {risk_level}")
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Error analyzing X-ray: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/upload")
async def analyze_uploaded(file: UploadFile = File(...)):
    """
    Analyze an uploaded X-ray image
    """
    try:
        start_time = datetime.now()
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        image_b64 = base64.b64encode(contents).decode('utf-8')
        
        logger.info(f"Analyzing uploaded X-ray: {file.filename}")
        
        # Call model (CML or local)
        if USE_LOCAL_MODEL:
            logger.info("Using local model")
            model = load_local_model()
            image = Image.open(io.BytesIO(contents))
            result = model({"image": image})
        else:
            logger.info("Calling CML Model endpoint")
            result = call_cml_model(image_b64)
        
        # Determine risk level
        anomaly_score = result['anomaly_score']
        if anomaly_score >= 0.7:
            risk_level = "HIGH RISK"
        elif anomaly_score >= 0.4:
            risk_level = "SUSPICIOUS"
        else:
            risk_level = "NORMAL"
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Build response
        analysis_result = AnalysisResult(
            file_path=file.filename,
            anomaly_score=anomaly_score,
            risk_level=risk_level,
            top_findings=result['top_findings'],
            bounding_boxes=result['bounding_boxes'],
            heatmap_base64=result['heatmap_base64'],
            predictions=result['predictions'],
            image_size=result['image_size'],
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
        
        logger.info(f"Analysis completed in {processing_time:.2f}s - Risk: {risk_level}")
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Error analyzing uploaded X-ray: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_statistics():
    """
    Get dataset statistics
    """
    try:
        stats = {}
        
        # Read statistics file if it exists
        stats_file = Path("data/sample_xrays/dataset_statistics.txt")
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                content = f.read()
                # Parse the statistics
                lines = content.split('\n')
                for line in lines:
                    if "Normal X-rays:" in line:
                        stats['normal_count'] = int(line.split(':')[1].strip())
                    elif "Pneumonia X-rays:" in line:
                        stats['pneumonia_count'] = int(line.split(':')[1].strip())
                    elif "Bacterial:" in line:
                        stats['bacterial_count'] = int(line.split(':')[1].strip())
                    elif "Viral:" in line:
                        stats['viral_count'] = int(line.split(':')[1].strip())
                    elif "Total:" in line:
                        stats['total_count'] = int(line.split(':')[1].strip())
        
        return stats
    
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files if directory exists
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("data/sample_xrays", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

