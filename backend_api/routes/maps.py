"""
Tab 4: Maps & Geospatial Visualization
Endpoints for generating ERA5 and GRACE comparison maps
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
import sys
import os
import uuid
import base64
import io

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'code'))

# Import Python functions
imports_successful = False

try:
    import vis_4_app
    import data_processing as dpr
    import configuration_settings as cs
    imports_successful = True
    print("✅ Successfully imported Python backend functions (maps.py)")
except ImportError as e:
    print(f"❌ ERROR: Could not import backend functions: {e}")
    import traceback
    traceback.print_exc()

# Global session_manager (will be set by main.py)
session_manager = None

router = APIRouter(prefix="/api/maps", tags=["Tab 4: Maps"])

# In-memory task tracking and file storage
map_tasks = {}
generated_maps = {}

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ERA5MapRequest(BaseModel):
    session_id: str
    year: int
    month: int  # 1-12
    variable: str  # "tp", "sro", etc.

class GRACEMapRequest(BaseModel):
    session_id: str
    year: int
    month: int

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return f"data:image/png;base64,{img_base64}"

# ============================================================================
# BACKGROUND TASKS
# ============================================================================

def generate_era5_map_task(session_id: str, task_id: str, year: int, month: int, variable: str):
    """Background task for generating ERA5 map"""
    try:
        map_tasks[task_id] = {
            "status": "in_progress",
            "progress": 10,
            "message": "Loading ERA5 data..."
        }
        
        # Get data from session
        ds_ERA_sliced = session_manager.get_data(session_id, "ds_ERA_sliced")
        bounds = session_manager.get_data(session_id, "bounds")
        
        if ds_ERA_sliced is None:
            raise ValueError("No ERA5 data found. Please run Tab 1 first.")
        
        print(f"🔄 Generating ERA5 map: year={year}, month={month}, variable={variable}")
        map_tasks[task_id]["progress"] = 30
        map_tasks[task_id]["message"] = "Generating map..."
        
        # Create basin name from bounds
        basin_name = f"Region ({bounds['lat_min']}°N, {bounds['lon_min']}°E)"
        
        # Generate the map
        fig = vis_4_app.ERA_plot(ds_ERA_sliced, year, month, variable, basin_name)
        
        map_tasks[task_id]["progress"] = 70
        map_tasks[task_id]["message"] = "Converting to image..."
        
        # Convert to base64
        img_base64 = fig_to_base64(fig)
        
        # Close figure to free memory
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        # Store the map
        map_id = str(uuid.uuid4())
        generated_maps[map_id] = {
            "type": "era5",
            "year": year,
            "month": month,
            "variable": variable,
            "image": img_base64,
            "session_id": session_id
        }
        
        print(f"✅ ERA5 map generated: {map_id}")
        
        map_tasks[task_id] = {
            "status": "complete",
            "progress": 100,
            "message": "Map generated successfully",
            "result": {
                "map_id": map_id,
                "type": "era5",
                "year": year,
                "month": month,
                "variable": variable,
                "image": img_base64
            }
        }
    
    except Exception as e:
        print(f"❌ ERA5 map generation error: {e}")
        import traceback
        traceback.print_exc()
        map_tasks[task_id] = {
            "status": "error",
            "progress": 0,
            "message": str(e)
        }

def generate_grace_map_task(session_id: str, task_id: str, year: int, month: int):
    """Background task for generating GRACE comparison map"""
    try:
        map_tasks[task_id] = {
            "status": "in_progress",
            "progress": 10,
            "message": "Loading data..."
        }
        
        # Get data from session
        model_path = session_manager.get_data(session_id, "model_path")
        ds_ERA_sliced = session_manager.get_data(session_id, "ds_ERA_sliced")
        ds_CSR_sliced = session_manager.get_data(session_id, "ds_CSR_sliced")
        df_CSR_on_ERA_grid = session_manager.get_data(session_id, "df_CSR_on_ERA_grid")
        df_ERA = session_manager.get_data(session_id, "df_ERA")
        bounds = session_manager.get_data(session_id, "bounds")
        
        if model_path is None:
            raise ValueError("No trained model found. Please run Tab 3 first.")
        if ds_ERA_sliced is None:
            raise ValueError("No ERA5 data found. Please run Tab 1 first.")
        
        print(f"🔄 Generating GRACE comparison map: year={year}, month={month}")
        map_tasks[task_id]["progress"] = 30
        map_tasks[task_id]["message"] = "Generating comparison map..."
        
        # Create basin name from bounds
        basin_name = f"Region ({bounds['lat_min']}°N, {bounds['lon_min']}°E)"
        
        # Generate the comparison map
        fig = vis_4_app.CSR_plot(
            model_path, year, month,
            ds_CSR_sliced, df_CSR_on_ERA_grid, df_ERA,
            "lwe_thickness", basin_name
        )
        
        map_tasks[task_id]["progress"] = 70
        map_tasks[task_id]["message"] = "Converting to image..."
        
        # Convert to base64
        img_base64 = fig_to_base64(fig)
        
        # Close figure to free memory
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        # Store the map
        map_id = str(uuid.uuid4())
        generated_maps[map_id] = {
            "type": "grace_comparison",
            "year": year,
            "month": month,
            "image": img_base64,
            "session_id": session_id
        }
        
        print(f"✅ GRACE comparison map generated: {map_id}")
        
        map_tasks[task_id] = {
            "status": "complete",
            "progress": 100,
            "message": "Map generated successfully",
            "result": {
                "map_id": map_id,
                "type": "grace_comparison",
                "year": year,
                "month": month,
                "image": img_base64
            }
        }
    
    except Exception as e:
        print(f"❌ GRACE map generation error: {e}")
        import traceback
        traceback.print_exc()
        map_tasks[task_id] = {
            "status": "error",
            "progress": 0,
            "message": str(e)
        }

# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/era5")
async def generate_era5_map(request: ERA5MapRequest, background_tasks: BackgroundTasks):
    """
    Generate ERA5 variable map for specified year/month
    
    Returns task_id for polling status
    """
    try:
        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Validate inputs
        if not (2002 <= request.year <= 2024):
            raise ValueError("Year must be between 2002 and 2024")
        if not (1 <= request.month <= 12):
            raise ValueError("Month must be between 1 and 12")
        
        # Check if ERA5 data exists
        ds_ERA_sliced = session_manager.get_data(request.session_id, "ds_ERA_sliced")
        if ds_ERA_sliced is None:
            return {
                "status": "error",
                "error_code": "NO_DATA",
                "message": "No ERA5 data found. Please run Tab 1 first.",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Create task
        task_id = str(uuid.uuid4())
        map_tasks[task_id] = {
            "status": "starting",
            "progress": 0,
            "message": "Initializing map generation..."
        }
        
        # Start background task
        background_tasks.add_task(
            generate_era5_map_task,
            request.session_id,
            task_id,
            request.year,
            request.month,
            request.variable
        )
        
        return {
            "status": "success",
            "data": {
                "task_id": task_id,
                "status": "in_progress",
                "message": "Map generation started"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ ERA5 map start error: {e}")
        return {
            "status": "error",
            "error_code": "MAP_START_ERROR",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.post("/grace-comparison")
async def generate_grace_map(request: GRACEMapRequest, background_tasks: BackgroundTasks):
    """
    Generate GRACE comparison map (observed vs predicted)
    
    Returns task_id for polling status
    """
    try:
        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Validate inputs
        if not (2002 <= request.year <= 2024):
            raise ValueError("Year must be between 2002 and 2024")
        if not (1 <= request.month <= 12):
            raise ValueError("Month must be between 1 and 12")
        
        # Check if model exists
        model_path = session_manager.get_data(request.session_id, "model_path")
        if model_path is None:
            return {
                "status": "error",
                "error_code": "NO_MODEL",
                "message": "No trained model found. Please run Tab 3 first.",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Create task
        task_id = str(uuid.uuid4())
        map_tasks[task_id] = {
            "status": "starting",
            "progress": 0,
            "message": "Initializing map generation..."
        }
        
        # Start background task
        background_tasks.add_task(
            generate_grace_map_task,
            request.session_id,
            task_id,
            request.year,
            request.month
        )
        
        return {
            "status": "success",
            "data": {
                "task_id": task_id,
                "status": "in_progress",
                "message": "Map generation started"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ GRACE map start error: {e}")
        return {
            "status": "error",
            "error_code": "MAP_START_ERROR",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.get("/status/{task_id}")
async def get_map_status(task_id: str):
    """Get status of a map generation task"""
    if task_id not in map_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_status = map_tasks[task_id]
    
    return {
        "status": "success",
        "data": {
            "task_id": task_id,
            "status": task_status.get("status"),
            "progress": task_status.get("progress", 0),
            "message": task_status.get("message", ""),
            "result": task_status.get("result", None)
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/download/{map_id}")
async def download_map(map_id: str):
    """
    Get a generated map by ID
    
    Returns the map image (base64 encoded in JSON)
    """
    if map_id not in generated_maps:
        raise HTTPException(status_code=404, detail="Map not found")
    
    map_data = generated_maps[map_id]
    
    return {
        "status": "success",
        "data": map_data,
        "timestamp": datetime.utcnow().isoformat()
    }
