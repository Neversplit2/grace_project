"""
Tab 5: Statistical Analysis
Endpoints for model evaluation and feature importance analysis
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
import sys
import os
import base64
import io

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'code'))

# Import Python functions
try:
    import vis_4_app
    import main_4_app
    import data_processing as dpr
    print("✅ Successfully imported Python backend functions (analysis.py)")
except ImportError as e:
    print(f"⚠️ Warning: Could not import backend functions: {e}")

# Global session_manager (will be set by main.py)
session_manager = None

router = APIRouter(prefix="/api/analysis", tags=["Tab 5: Statistical Analysis"])

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class EvaluateRequest(BaseModel):
    session_id: str
    latitude: float
    longitude: float
    start_year: int
    end_year: int

class FeatureImportanceRequest(BaseModel):
    session_id: str

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
# ENDPOINTS
# ============================================================================

@router.post("/evaluate")
async def evaluate_model(request: EvaluateRequest):
    """
    Evaluate model performance at a specific location
    
    Returns time series plot and scatter plot with R-score
    """
    try:
        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Validate inputs
        if not (-90 <= request.latitude <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        if not (-180 <= request.longitude <= 180):
            raise ValueError("Longitude must be between -180 and 180")
        if not (2002 <= request.start_year <= 2024):
            raise ValueError("Start year must be between 2002 and 2024")
        if not (2002 <= request.end_year <= 2024):
            raise ValueError("End year must be between 2002 and 2024")
        if request.start_year >= request.end_year:
            raise ValueError("Start year must be less than end year")
        
        # Get data from session
        model_path = session_manager.get_data(request.session_id, "model_path")
        df_ERA = session_manager.get_data(request.session_id, "df_ERA")
        df_CSR_on_ERA_grid = session_manager.get_data(request.session_id, "df_CSR_on_ERA_grid")
        
        if model_path is None:
            return {
                "status": "error",
                "error_code": "NO_MODEL",
                "message": "No trained model found. Please run Tab 3 first.",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        if df_ERA is None or df_CSR_on_ERA_grid is None:
            return {
                "status": "error",
                "error_code": "NO_DATA",
                "message": "No data found. Please run Tab 1 first.",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        print(f"🔄 Evaluating model at lat={request.latitude}, lon={request.longitude}")
        
        # Call the statistics pipeline
        merged_stats = main_4_app.pipe_stats(
            df_ERA,
            model_path,
            request.start_year,
            request.end_year,
            df_CSR_on_ERA_grid,
            request.latitude,
            request.longitude
        )
        
        print(f"✅ Statistics calculated, generating plots...")
        
        # Generate evaluation plot
        eval_fig = vis_4_app.model_eval_plot(merged_stats)
        
        # Convert to base64
        eval_img_base64 = fig_to_base64(eval_fig)
        
        # Close figure to free memory
        import matplotlib.pyplot as plt
        plt.close(eval_fig)
        
        # Calculate statistics
        r_score, p_value = dpr.corr_pearson(merged_stats)
        rmse = dpr.stats_lwe(merged_stats, merged_stats)  # May need adjustment based on function
        
        print(f"✅ Evaluation complete! R-score: {r_score:.4f}")
        
        # Store results in session
        session_manager.save_data(request.session_id, "eval_stats", merged_stats)
        session_manager.save_data(request.session_id, "eval_location", {
            "latitude": request.latitude,
            "longitude": request.longitude,
            "start_year": request.start_year,
            "end_year": request.end_year
        })
        
        return {
            "status": "success",
            "data": {
                "location": {
                    "latitude": request.latitude,
                    "longitude": request.longitude
                },
                "time_period": {
                    "start_year": request.start_year,
                    "end_year": request.end_year
                },
                "statistics": {
                    "r_score": float(r_score),
                    "p_value": float(p_value) if p_value is not None else None,
                    "n_samples": len(merged_stats)
                },
                "plot": eval_img_base64,
                "message": "Model evaluation complete"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error_code": "EVALUATION_ERROR",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.post("/feature-importance")
async def get_feature_importance(request: FeatureImportanceRequest):
    """
    Get feature importance as a pie chart
    
    Returns pie chart showing which features are most important
    """
    try:
        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get data from session
        model_path = session_manager.get_data(request.session_id, "model_path")
        X_train = session_manager.get_data(request.session_id, "X_train")
        
        if model_path is None:
            return {
                "status": "error",
                "error_code": "NO_MODEL",
                "message": "No trained model found. Please run Tab 3 first.",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        if X_train is None:
            return {
                "status": "error",
                "error_code": "NO_TRAINING_DATA",
                "message": "No training data found. Please run Tab 3 first.",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        print(f"🔄 Generating feature importance plot...")
        
        # Generate feature importance pie chart
        importance_fig = vis_4_app.feature_importance_pie(model_path, X_train)
        
        # Convert to base64
        importance_img_base64 = fig_to_base64(importance_fig)
        
        # Close figure to free memory
        import matplotlib.pyplot as plt
        plt.close(importance_fig)
        
        # Get feature importances as data
        import joblib
        model = joblib.load(model_path)
        importances = model.feature_importances_
        feature_names = X_train.columns
        
        # Create sorted list
        features_data = [
            {
                "feature": str(name),
                "importance": float(imp),
                "percentage": float(imp * 100 / importances.sum())
            }
            for name, imp in zip(feature_names, importances)
        ]
        features_data.sort(key=lambda x: x["importance"], reverse=True)
        
        print(f"✅ Feature importance generated!")
        
        return {
            "status": "success",
            "data": {
                "features": features_data,
                "plot": importance_img_base64,
                "total_features": len(features_data),
                "message": "Feature importance analysis complete"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Feature importance error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error_code": "FEATURE_IMPORTANCE_ERROR",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
