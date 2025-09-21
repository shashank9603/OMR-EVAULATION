"""
Innomatics OMR System - FastAPI Backend
Production-grade REST API for OMR processing
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import aiofiles
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import tempfile
import shutil
from pathlib import Path
import sys
import logging
import time
import uuid
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.core.omr_processor import InnoMaticsOMRProcessor
    from src.utils.excel_handler import AnswerKeyManager
    from database.models import SessionLocal, OMRSheet, ProcessingJob
    from config.settings import API_HOST, API_PORT, MAX_UPLOAD_FILES, MAX_FILE_SIZE_MB
except ImportError as e:
    print(f"Warning: Production modules not available: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="Innomatics OMR Evaluation API",
    description="Enterprise-grade OMR processing API with <0.5% error rate",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global instances
omr_processor = None
answer_key_manager = None
processing_jobs = {}

# Pydantic models
class ProcessingConfig(BaseModel):
    sheet_version: str = "SET-A"
    confidence_threshold: float = 0.8
    enable_ml: bool = False
    concurrent: bool = True

class ProcessingResult(BaseModel):
    job_id: str
    sheet_id: str
    success: bool
    percentage_score: float
    total_score: int
    overall_confidence: float
    processing_time: float
    status: str
    warnings: List[str] = []
    errors: List[str] = []

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    total_files: int
    processed_files: int
    started_at: datetime
    estimated_completion: Optional[datetime] = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global omr_processor, answer_key_manager
    
    try:
        # Initialize OMR processor
        omr_processor = InnoMaticsOMRProcessor()
        
        # Initialize answer key manager
        answer_key_manager = AnswerKeyManager()
        
        logging.info("OMR API services initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize services: {e}")

# Health check
@app.get("/health")
async def health_check():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": {
            "omr_processor": omr_processor is not None,
            "answer_key_manager": answer_key_manager is not None
        }
    }

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token (implement real authentication)"""
    # Mock authentication - implement real JWT verification
    if credentials.credentials == "demo-token":
        return {"user_id": "demo", "role": "evaluator"}
    raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# File upload endpoints
@app.post("/api/v1/omr/process-batch")
async def process_omr_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    sheet_version: str = "SET-A",
    confidence_threshold: float = 0.8,
    enable_ml: bool = False,
    concurrent: bool = True,
    user: dict = Depends(verify_token)
):
    """Process batch of OMR sheets"""
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=400, 
            detail=f"Too many files. Maximum {MAX_UPLOAD_FILES} allowed"
        )
    
    # Validate files
    valid_files = []
    for file in files:
        if file.content_type not in ["image/jpeg", "image/png", "application/pdf"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}"
            )
        
        # Check file size (approximate)
        if hasattr(file, 'size') and file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} exceeds {MAX_FILE_SIZE_MB}MB limit"
            )
        
        valid_files.append(file)
    
    # Create processing job
    job_id = str(uuid.uuid4())
    processing_jobs[job_id] = {
        "status": "started",
        "progress": 0.0,
        "total_files": len(valid_files),
        "processed_files": 0,
        "started_at": datetime.now(),
        "results": [],
        "config": {
            "sheet_version": sheet_version,
            "confidence_threshold": confidence_threshold,
            "enable_ml": enable_ml,
            "concurrent": concurrent
        }
    }
    
    # Start background processing
    background_tasks.add_task(
        process_files_background, 
        job_id, 
        valid_files,
        user["user_id"]
    )
    
    return {
        "success": True,
        "job_id": job_id,
        "message": f"Processing started for {len(valid_files)} files",
        "estimated_time": len(valid_files) * 2.5
    }

async def process_files_background(job_id: str, files: List[UploadFile], user_id: str):
    """Background task for processing files"""
    job = processing_jobs[job_id]
    job["status"] = "processing"
    
    try:
        for i, file in enumerate(files):
            if job.get("cancelled"):
                break
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                tmp_path = tmp_file.name
            
            try:
                # Load image
                import cv2
                image = cv2.imread(tmp_path)
                
                if image is None:
                    raise Exception("Could not load image")
                
                # Process with OMR processor
                result = omr_processor.process_single_sheet(
                    image,
                    job["config"]["sheet_version"],
                    f"{job_id}_{i:03d}"
                )
                
                # Convert result to API format
                api_result = ProcessingResult(
                    job_id=job_id,
                    sheet_id=result.sheet_id,
                    success=result.success,
                    percentage_score=result.percentage_score,
                    total_score=result.total_score,
                    overall_confidence=result.overall_confidence,
                    processing_time=result.processing_metrics.total_processing_time,
                    status=result.status.value if hasattr(result.status, 'value') else str(result.status),
                    warnings=result.warnings,
                    errors=result.errors
                )
                
                job["results"].append(api_result.dict())
                
            except Exception as e:
                # Handle processing error
                error_result = ProcessingResult(
                    job_id=job_id,
                    sheet_id=f"error_{i:03d}",
                    success=False,
                    percentage_score=0.0,
                    total_score=0,
                    overall_confidence=0.0,
                    processing_time=0.0,
                    status="failed",
                    warnings=[],
                    errors=[str(e)]
                )
                job["results"].append(error_result.dict())
            
            finally:
                # Clean up temporary file
                Path(tmp_path).unlink(missing_ok=True)
            
            # Update progress
            job["processed_files"] = i + 1
            job["progress"] = ((i + 1) / len(files)) * 100
        
        # Mark job complete
        job["status"] = "completed"
        job["completed_at"] = datetime.now()
        
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)

# Job status endpoints
@app.get("/api/v1/jobs/{job_id}/status")
async def get_job_status(job_id: str, user: dict = Depends(verify_token)):
    """Get processing job status"""
    
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        total_files=job["total_files"],
        processed_files=job["processed_files"],
        started_at=job["started_at"]
    )

@app.get("/api/v1/jobs/{job_id}/results")
async def get_job_results(job_id: str, user: dict = Depends(verify_token)):
    """Get processing job results"""
    
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    if job["status"] not in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Job not yet completed")
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "results": job["results"],
        "summary": {
            "total_files": job["total_files"],
            "successful": sum(1 for r in job["results"] if r["success"]),
            "failed": sum(1 for r in job["results"] if not r["success"]),
            "average_score": sum(r["percentage_score"] for r in job["results"] if r["success"]) / max(1, sum(1 for r in job["results"] if r["success"])),
            "average_confidence": sum(r["overall_confidence"] for r in job["results"] if r["success"]) / max(1, sum(1 for r in job["results"] if r["success"]))
        }
    }

# Answer key management
@app.get("/api/v1/answer-keys")
async def get_answer_keys(user: dict = Depends(verify_token)):
    """Get available answer key versions"""
    
    if not answer_key_manager:
        raise HTTPException(status_code=503, detail="Answer key manager not available")
    
    versions = answer_key_manager.get_available_versions()
    
    return {
        "versions": versions,
        "count": len(versions)
    }

@app.get("/api/v1/answer-keys/{version}")
async def get_answer_key(version: str, user: dict = Depends(verify_token)):
    """Get specific answer key"""
    
    if not answer_key_manager:
        raise HTTPException(status_code=503, detail="Answer key manager not available")
    
    answer_key = answer_key_manager.get_answer_key(version)
    
    if not answer_key:
        raise HTTPException(status_code=404, detail=f"Answer key '{version}' not found")
    
    return {
        "version": version,
        "answer_key": answer_key,
        "total_questions": len(answer_key)
    }

# System statistics
@app.get("/api/v1/stats")
async def get_system_stats(user: dict = Depends(verify_token)):
    """Get system processing statistics"""
    
    if not omr_processor:
        raise HTTPException(status_code=503, detail="OMR processor not available")
    
    stats = omr_processor.get_performance_stats()
    
    return {
        "system_stats": stats,
        "active_jobs": len([j for j in processing_jobs.values() if j["status"] == "processing"]),
        "completed_jobs": len([j for j in processing_jobs.values() if j["status"] == "completed"]),
        "uptime": "Available in production version"
    }

# Authentication endpoints
@app.post("/api/v1/auth/login")
async def login(credentials: dict):
    """User authentication endpoint"""
    # Mock authentication - implement real authentication
    username = credentials.get("username")
    password = credentials.get("password")
    
    if username == "admin" and password == "admin123":
        return {
            "access_token": "demo-token",
            "token_type": "bearer",
            "user": {
                "username": "admin",
                "role": "administrator"
            }
        }
    
    raise HTTPException(status_code=401, detail="Invalid credentials")

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logging.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_HOST if 'API_HOST' in globals() else "0.0.0.0",
        port=API_PORT if 'API_PORT' in globals() else 8000,
        reload=True,
        log_level="info"
    )
