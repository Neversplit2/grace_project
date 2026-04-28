"""
GRACE Downscaling Engine - FastAPI Backend
Main entry point for the REST API server

Run with: uvicorn main:app --reload --port 5000
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
from datetime import datetime
from contextlib import asynccontextmanager

# Configure matplotlib to run headlessly in backend threads
import matplotlib
matplotlib.use('Agg')

# Add code directory to path so we can import Python modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

# Import route modules
from routes import setup, data_processing, training, maps, analysis, utils

# Import session manager
from utils.session_manager import SessionManager

# Initialize session manager
session_manager = SessionManager()

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 GRACE API Server starting...")
    yield
    # Shutdown
    print("🛑 GRACE API Server shutting down...")
    session_manager.cleanup_expired_sessions()

# Create FastAPI app with lifespan
app = FastAPI(
    title="GRACE Downscaling Engine API",
    description="REST API for GRACE Downscaling Engine",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware (allow React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# UTILITY ENDPOINTS (Health, Config, Session Management)
# ============================================================================

@app.get("/api/health")
def health_check():
    """Check if backend is running"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "GRACE Downscaling Engine API"
    }

@app.get("/api/config")
def get_config():
    """Get system configuration and available options"""
    return {
        "status": "success",
        "data": {
            "models": {
                "training": ["XGBoost", "Random Forest"],
                "rfe": ["XGBoost", "Random Forest"]
            },
            "variables": {
                "era5": [
                    "2m_temperature",
                    "total_precipitation",
                    "total_evaporation",
                    "potential_evaporation",
                    "sub_surface_runoff",
                    "surface_runoff",
                    "evaporation_from_bare_soil",
                    "volumetric_soil_water_layer_1",
                    "volumetric_soil_water_layer_2",
                    "volumetric_soil_water_layer_3",
                    "volumetric_soil_water_layer_4",
                    "leaf_area_index_high_vegetation",
                    "leaf_area_index_low_vegetation"
                ]
            },
            "bounds": {
                "latitude": {"min": -90, "max": 90},
                "longitude": {"min": -180, "max": 180}
            },
            "years": {"min": 2002, "max": 2025},
            "rfe_features": {"min": 1, "max": 13}
        }
    }

@app.post("/api/session/create")
def create_session():
    """Create a new session"""
    session_id = session_manager.create_session()
    return {
        "status": "success",
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    """Get session info"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    return {
        "status": "success",
        "data": {
            "session_id": session_id,
            "created_at": session.get("created_at"),
            "last_accessed": session.get("last_accessed"),
            "data_keys": list(session.get("data", {}).keys())
        }
    }

@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    """Delete a session"""
    success = session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    return {
        "status": "success",
        "message": f"Session {session_id} deleted"
    }

# ============================================================================
# INCLUDE ROUTE MODULES
# ============================================================================

# Pass session_manager to route modules
setup.session_manager = session_manager
data_processing.session_manager = session_manager
training.session_manager = session_manager
maps.session_manager = session_manager
analysis.session_manager = session_manager

# Include route routers
app.include_router(setup.router)
app.include_router(data_processing.router)
app.include_router(training.router)
app.include_router(maps.router)
app.include_router(analysis.router)

# ============================================================================
# ERROR HANDLING
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error_code": "HTTP_ERROR",
            "message": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch all exceptions"""
    print(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
def read_root():
    """Welcome endpoint with API documentation link"""
    return {
        "message": "🎯 GRACE Downscaling Engine API",
        "version": "1.0.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "health": "/api/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
