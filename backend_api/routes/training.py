"""
Tab 3: Model Training
Endpoints for training ML models with progress tracking
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
import sys
import os
import uuid
import time
import joblib

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'code'))

# Import Python functions
try:
    import main_4_app
    import training as tr
    import configuration_settings as cs
    print("✅ Successfully imported Python backend functions (training.py)")
except ImportError as e:
    print(f"⚠️ Warning: Could not import backend functions: {e}")

# Global session_manager (will be set by main.py)
session_manager = None

router = APIRouter(prefix="/api/training", tags=["Tab 3: Model Training"])

# In-memory task tracking
training_tasks = {}

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class TrainingRequest(BaseModel):
    session_id: str
    model: str  # "XGBoost" or "RF"
    train_type: str  # "Hyper" or "Quick"

class UploadModelRequest(BaseModel):
    session_id: str

# ============================================================================
# BACKGROUND TASKS
# ============================================================================

def run_training_task(session_id: str, task_id: str, model: str, train_type: str):
    """Background task for model training"""
    try:
        training_tasks[task_id] = {
            "status": "in_progress",
            "progress": 5,
            "message": "Preparing training data..."
        }
        
        # Get data from session
        selected_features = session_manager.get_data(session_id, "selected_features")
        x = session_manager.get_data(session_id, "x")
        merged = session_manager.get_data(session_id, "merged")
        
        if selected_features is None or x is None or merged is None:
            raise ValueError("Missing RFE data. Please run Tab 2 first.")
        
        print(f"🔄 Starting model training: {model} ({train_type})")
        training_tasks[task_id]["progress"] = 10
        training_tasks[task_id]["message"] = "Splitting data..."
        
        # Call actual training pipeline
        X_train, X_test, y_train, y_test, best_model = main_4_app.pipe_model_train(
            selected_features, x, merged, model, train_type
        )
        
        training_tasks[task_id]["progress"] = 60
        training_tasks[task_id]["message"] = "Training complete, saving model..."
        
        # Save model to file
        model_filename = f"{model}_{train_type}_{session_id[:8]}.pkl"
        model_path = cs.MODEL_DIR / model_filename
        
        # Ensure model directory exists
        cs.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(best_model, model_path)
        print(f"✅ Model saved to: {model_path}")
        
        training_tasks[task_id]["progress"] = 80
        training_tasks[task_id]["message"] = "Generating learning curves..."
        
        # Generate learning curves
        try:
            if model == "XGBoost":
                learning_curve_fig = tr.XGBoost_curves(X_train, X_test, y_train, y_test, train_type)
            elif model == "RF":
                learning_curve_fig = tr.RF_curves(X_train, X_test, y_train, y_test, train_type)
            else:
                learning_curve_fig = None
        except Exception as e:
            print(f"⚠️ Warning: Could not generate learning curves: {e}")
            learning_curve_fig = None
        
        # Store results in session
        session_manager.save_data(session_id, "trained_model", best_model)
        session_manager.save_data(session_id, "model_path", str(model_path))
        session_manager.save_data(session_id, "X_train", X_train)
        session_manager.save_data(session_id, "X_test", X_test)
        session_manager.save_data(session_id, "y_train", y_train)
        session_manager.save_data(session_id, "y_test", y_test)
        session_manager.save_data(session_id, "model_name", model)
        session_manager.save_data(session_id, "train_type", train_type)
        
        training_tasks[task_id] = {
            "status": "complete",
            "progress": 100,
            "message": "Training complete!",
            "result": {
                "model": model,
                "train_type": train_type,
                "model_path": str(model_path),
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "features": list(selected_features)
            }
        }
        print(f"✅ Training task {task_id} complete!")
    
    except Exception as e:
        print(f"❌ Training task error: {e}")
        import traceback
        traceback.print_exc()
        training_tasks[task_id] = {
            "status": "error",
            "progress": 0,
            "message": str(e)
        }

# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Start model training (runs in background)
    
    Returns task_id for polling status
    """
    try:
        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Validate inputs
        if request.model not in ["XGBoost", "RF"]:
            raise ValueError("model must be 'XGBoost' or 'RF'")
        if request.train_type not in ["Hyper", "Quick"]:
            raise ValueError("train_type must be 'Hyper' or 'Quick'")
        
        # Check if RFE data exists
        selected_features = session_manager.get_data(request.session_id, "selected_features")
        if selected_features is None:
            return {
                "status": "error",
                "error_code": "NO_RFE_DATA",
                "message": "No feature selection data found. Please run Tab 2 (RFE) first.",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Create task
        task_id = str(uuid.uuid4())
        training_tasks[task_id] = {
            "status": "starting",
            "progress": 0,
            "message": "Initializing training..."
        }
        
        # Start background task
        background_tasks.add_task(
            run_training_task,
            request.session_id,
            task_id,
            request.model,
            request.train_type
        )
        
        return {
            "status": "success",
            "data": {
                "task_id": task_id,
                "status": "in_progress",
                "message": "Training started"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Training start error: {e}")
        return {
            "status": "error",
            "error_code": "TRAINING_START_ERROR",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.get("/status/{task_id}")
async def get_training_status(task_id: str):
    """Get status of a training task"""
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_status = training_tasks[task_id]
    
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
async def get_training_result(task_id: str):
    """Get final result of a completed training task"""
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_status = training_tasks[task_id]
    
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

@router.post("/upload-model")
async def upload_model(
    session_id: str,
    file: UploadFile = File(...)
):
    """
    Upload a pre-trained model file
    
    Accepts .pkl or .joblib files
    """
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Validate file type
        if not (file.filename.endswith('.pkl') or file.filename.endswith('.joblib')):
            raise ValueError("File must be .pkl or .joblib format")
        
        # Read file content
        content = await file.read()
        
        # Save to models directory
        model_filename = f"uploaded_{session_id[:8]}_{file.filename}"
        model_path = cs.MODEL_DIR / model_filename
        
        # Ensure model directory exists
        cs.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        # Write file
        with open(model_path, 'wb') as f:
            f.write(content)
        
        print(f"✅ Model uploaded to: {model_path}")
        
        # Try to load the model to validate it
        try:
            loaded_model = joblib.load(model_path)
            print(f"✅ Model validated successfully")
        except Exception as e:
            # Clean up invalid file
            model_path.unlink()
            raise ValueError(f"Invalid model file: {str(e)}")
        
        # Store in session
        session_manager.save_data(session_id, "trained_model", loaded_model)
        session_manager.save_data(session_id, "model_path", str(model_path))
        session_manager.save_data(session_id, "model_name", "Uploaded")
        
        return {
            "status": "success",
            "data": {
                "message": "Model uploaded successfully",
                "filename": file.filename,
                "model_path": str(model_path),
                "size_bytes": len(content)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Model upload error: {e}")
        return {
            "status": "error",
            "error_code": "UPLOAD_ERROR",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
