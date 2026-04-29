"""
Standalone EXECUTABLE entry point for GRACE Downscaling Engine.
Combines FastAPI backend + built React frontend into a single server.
"""

import os
import sys
import threading
import webbrowser
import time
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime

# Configure matplotlib headlessly
import matplotlib
matplotlib.use('Agg')

# Ensure we're in the right directory when frozen
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app 
    # path into variable _MEIPASS
    BASE_DIR = sys._MEIPASS
    os.environ["IS_FROZEN"] = "true"
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FRONTEND_DIST_DIR = os.path.join(BASE_DIR, "frontend_react", "dist")
BACKEND_DIR = os.path.join(BASE_DIR, "backend_api")

# Add backend and code to path so imports work correctly
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "code"))

try:
    from routes import setup, data_processing, training, maps, analysis
    from utils.session_manager import SessionManager
except ImportError as e:
    print(f"Error importing backend modules: {e}")
    sys.exit(1)

session_manager = SessionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    session_manager.cleanup_expired_sessions()

app = FastAPI(title="GRACE Engine EXE", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5321", "http://127.0.0.1:5321"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pass session_manager to route modules
setup.session_manager = session_manager
data_processing.session_manager = session_manager
training.session_manager = session_manager
maps.session_manager = session_manager
analysis.session_manager = session_manager

app.include_router(setup.router)
app.include_router(data_processing.router)
app.include_router(training.router)
app.include_router(maps.router)
app.include_router(analysis.router)

# ============================================================================
# UTILITY ENDPOINTS (Health, Config, Session Management)
# ============================================================================

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "GRACE Downscaling Engine EXE"
    }

@app.get("/api/config")
def get_config():
    return {
        "status": "success",
        "data": {
            "models": {
                "training": ["XGBoost", "Random Forest"],
                "rfe": ["XGBoost", "Random Forest"]
            },
            "variables": {
                "era5": [
                    "2m_temperature", "total_precipitation", "total_evaporation",
                    "potential_evaporation", "sub_surface_runoff", "surface_runoff",
                    "evaporation_from_bare_soil", "volumetric_soil_water_layer_1",
                    "volumetric_soil_water_layer_2", "volumetric_soil_water_layer_3",
                    "volumetric_soil_water_layer_4", "leaf_area_index_high_vegetation",
                    "leaf_area_index_low_vegetation"
                ]
            },
            "bounds": {"latitude": {"min": -90, "max": 90}, "longitude": {"min": -180, "max": 180}},
            "years": {"min": 2002, "max": 2025},
            "rfe_features": {"min": 1, "max": 13}
        }
    }

@app.post("/api/session/create")
def create_session():
    session_id = session_manager.create_session()
    return {
        "status": "success",
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/session/{session_id}")
def get_session(session_id: str):
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
    success = session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return {"status": "success", "message": f"Session {session_id} deleted"}

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})

# Mount React App
if os.path.exists(FRONTEND_DIST_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST_DIR, "assets")), name="assets")
    
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Prevent api calls from falling back to index.html
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
            
        file_path = os.path.join(FRONTEND_DIST_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(FRONTEND_DIST_DIR, "index.html"))
else:
    print(f"⚠️ WARNING: Frontend build folder not found at {FRONTEND_DIST_DIR}")

def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:5321")

if __name__ == "__main__":
    # Fix for multiprocessing in PyInstaller
    import multiprocessing
    multiprocessing.freeze_support()
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    print("\n" + "="*50)
    print("🌍 Starting GRACE Downscaling Desktop Engine...")
    print("Please do not close this console window.")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="127.0.0.1", port=5321, log_level="warning")
