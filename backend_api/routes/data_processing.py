"""
Tab 2: Data Processing & RFE Feature Selection
Endpoints for data preparation and feature selection
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
import sys
import os
import uuid
import time

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'code'))

# Import Python functions
main_4_app = None
imports_successful = False

try:
    import main_4_app
    import training as tr
    import vis_4_app as vis
    imports_successful = True
    print("✅ Successfully imported Python backend functions (data_processing.py)")
except ImportError as e:
    print(f"❌ ERROR: Could not import backend functions: {e}")
    import traceback
    traceback.print_exc()

# Global session_manager (will be set by main.py)
session_manager = None

router = APIRouter(prefix="/api/data-processing", tags=["Tab 2: Data Processing"])

# In-memory task tracking
active_tasks = {}

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class DataPrepRequest(BaseModel):
    session_id: str

class RFERequest(BaseModel):
    session_id: str
    model_type: str  # "XGBoost" or "Random Forest"
    n_features: int  # 1-13

# ============================================================================
# BACKGROUND TASKS
# ============================================================================

def run_rfe_task(session_id: str, task_id: str, model_type: str, n_features: int):
    """Background task for RFE feature selection"""
    try:
        if not imports_successful or main_4_app is None:
            raise RuntimeError("Backend Python modules not loaded. Check server logs for import errors.")
        
        active_tasks[task_id] = {
            "status": "in_progress",
            "progress": 5,
            "message": "Starting RFE..."
        }
        
        # Get merged dataframe from session
        merged = session_manager.get_data(session_id, "merged")
        if merged is None:
            raise ValueError("No merged data found. Please run Tab 1 first.")
        
        print(f"🔄 Running RFE with model={model_type}, n_features={n_features}")
        active_tasks[task_id]["progress"] = 20
        active_tasks[task_id]["message"] = "Running RFE algorithm..."
        
        # Call actual RFE function
        rfe, selected_features, x = main_4_app.pipe_RFE(merged, model_type, n_features)
        
        print(f"✅ RFE complete! Selected features: {selected_features}")
        
        # Save results to session
        session_manager.save_data(session_id, "rfe", rfe)
        session_manager.save_data(session_id, "selected_features", selected_features)
        session_manager.save_data(session_id, "x", x)
        session_manager.save_data(session_id, "model_type", model_type)
        session_manager.save_data(session_id, "n_features", n_features)
        
        # Generate RFE plot
        import io
        import base64
        import matplotlib.pyplot as plt
        
        fig = vis.rfe_plot(rfe, x)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor='#0b0f19')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        
        active_tasks[task_id] = {
            "status": "complete",
            "progress": 100,
            "message": "RFE complete",
            "result": {
                "selected_features": list(selected_features),
                "n_features_selected": len(selected_features),
                "n_features_total": merged.shape[1],
                "model_type": model_type,
                "rfe_plot_base64": img_str
            }
        }
    
    except Exception as e:
        print(f"❌ RFE task error: {e}")
        import traceback
        traceback.print_exc()
        active_tasks[task_id] = {
            "status": "error",
            "progress": 0,
            "message": str(e)
        }

# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/prep")
async def data_prep(request: DataPrepRequest):
    """
    Data preparation endpoint
    
    Note: In the current implementation, data prep is done in Tab 1 (load-data).
    This endpoint is kept for compatibility but returns immediately.
    """
    try:
        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Check if data is already loaded
        merged = session_manager.get_data(request.session_id, "merged")
        if merged is None:
            return {
                "status": "error",
                "error_code": "NO_DATA",
                "message": "No data found. Please load data in Tab 1 first.",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Get ERA5 dataframe to extract available features
        df_ERA = session_manager.get_data(request.session_id, "df_ERA")
        available_features = list(df_ERA.columns) if df_ERA is not None else []
        
        return {
            "status": "success",
            "data": {
                "message": "Data already prepared in Tab 1",
                "available_features": len(available_features),
                "n_samples": merged.shape[0],
                "merged_shape": list(merged.shape),
                "feature_names": available_features,
                "ready_for_rfe": True
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Data prep error: {e}")
        return {
            "status": "error",
            "error_code": "PREP_ERROR",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.post("/rfe")
async def start_rfe(request: RFERequest, background_tasks: BackgroundTasks):
    """
    Start RFE feature selection (runs in background)
    
    Returns task_id for polling status
    """
    try:
        if not imports_successful or main_4_app is None:
            return {
                "status": "error",
                "error_code": "IMPORT_ERROR",
                "message": "Backend Python modules not loaded. Check server logs for import errors.",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Validate inputs
        if request.model_type not in ["XGBoost", "Random Forest", "RF"]:
            raise ValueError("model_type must be 'XGBoost' or 'Random Forest'")
        if not (1 <= request.n_features <= 13):
            raise ValueError("n_features must be between 1 and 13")
        
        # Check if merged data exists
        merged = session_manager.get_data(request.session_id, "merged")
        if merged is None:
            return {
                "status": "error",
                "error_code": "NO_DATA",
                "message": "No data found. Please load data in Tab 1 first.",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Create task
        task_id = str(uuid.uuid4())
        active_tasks[task_id] = {
            "status": "starting",
            "progress": 0,
            "message": "Initializing RFE..."
        }
        
        # Start background task
        background_tasks.add_task(
            run_rfe_task,
            request.session_id,
            task_id,
            request.model_type,
            request.n_features
        )
        
        return {
            "status": "success",
            "data": {
                "task_id": task_id,
                "status": "in_progress",
                "message": "RFE started"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ RFE start error: {e}")
        return {
            "status": "error",
            "error_code": "RFE_START_ERROR",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a background task"""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_status = active_tasks[task_id]
    
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

@router.get("/result/{task_id}")
async def get_task_result(task_id: str):
    """Get final result of a completed task"""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_status = active_tasks[task_id]
    
    if task_status["status"] != "complete":
        return {
            "status": "error",
            "error_code": "TASK_NOT_COMPLETE",
            "message": f"Task status is: {task_status['status']}",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    return {
        "status": "success",
        "data": task_status.get("result", {}),
        "timestamp": datetime.utcnow().isoformat()
    }
