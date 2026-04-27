"""
Tab 1: Setup & Geographic Bounds Selection
Endpoints for geographic bounds validation and data loading
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any
import sys
import os
import requests

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'code'))

# Import Python functions
try:
    import data_processing as dpr
    import main_4_app
    print("✅ Successfully imported Python backend functions (setup.py)")
except ImportError as e:
    print(f"⚠️ Warning: Could not import backend functions: {e}")

# Global session_manager (will be set by main.py)
session_manager = None

router = APIRouter(prefix="/api/setup", tags=["Tab 1: Setup"])

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ValidateBoundsRequest(BaseModel):
    session_id: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

class LoadDataRequest(BaseModel):
    session_id: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    grace_dataset: str = "CSR"

# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/validate-bounds")
async def validate_bounds(request: ValidateBoundsRequest):
    """Validate geographic bounds"""
    try:
        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        errors = []
        
        # Validate latitude
        if not (-90 <= request.lat_min <= 90):
            errors.append(f"lat_min {request.lat_min} out of range [-90, 90]")
        if not (-90 <= request.lat_max <= 90):
            errors.append(f"lat_max {request.lat_max} out of range [-90, 90]")
        if request.lat_min >= request.lat_max:
            errors.append("lat_min must be < lat_max")
        
        # Validate longitude
        if not (-180 <= request.lon_min <= 180):
            errors.append(f"lon_min {request.lon_min} out of range [-180, 180]")
        if not (-180 <= request.lon_max <= 180):
            errors.append(f"lon_max {request.lon_max} out of range [-180, 180]")
        if request.lon_min >= request.lon_max:
            errors.append("lon_min must be < lon_max")
        
        if errors:
            return {
                "status": "error",
                "error_code": "INVALID_BOUNDS",
                "message": "; ".join(errors),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Save bounds to session
        bounds = {
            "lat_min": request.lat_min,
            "lat_max": request.lat_max,
            "lon_min": request.lon_min,
            "lon_max": request.lon_max
        }
        session_manager.save_data(request.session_id, "bounds", bounds)
        
        return {
            "status": "success",
            "data": {
                "valid": True,
                "bounds": bounds,
                "area": {
                    "lat_range": request.lat_max - request.lat_min,
                    "lon_range": request.lon_max - request.lon_min
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        return {
            "status": "error",
            "error_code": "VALIDATION_ERROR",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.post("/load-data")
async def load_data(request: LoadDataRequest):
    """Load GRACE and ERA5 data for specified bounds"""
    try:
        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Validate bounds
        if not (-90 <= request.lat_min <= 90 and -90 <= request.lat_max <= 90):
            raise ValueError("Invalid latitude bounds")
        if not (-180 <= request.lon_min <= 180 and -180 <= request.lon_max <= 180):
            raise ValueError("Invalid longitude bounds")
        
        # Call the actual Python function from main_4_app
        print(f"🔄 Loading data for bounds: lat=[{request.lat_min}, {request.lat_max}], lon=[{request.lon_min}, {request.lon_max}]")
        
        df_ERA, df_CSR, ds_ERA_sliced, ds_CSR_sliced, merged, df_CSR_on_ERA_grid = main_4_app.pipe_data_prp(
            request.grace_dataset,
            request.lat_min, request.lat_max,
            request.lon_min, request.lon_max
        )
        
        print(f"✅ Data loaded successfully!")
        
        # Store data in session
        session_manager.save_data(request.session_id, "df_ERA", df_ERA)
        session_manager.save_data(request.session_id, "df_CSR", df_CSR)
        session_manager.save_data(request.session_id, "ds_ERA_sliced", ds_ERA_sliced)
        session_manager.save_data(request.session_id, "ds_CSR_sliced", ds_CSR_sliced)
        session_manager.save_data(request.session_id, "merged", merged)
        session_manager.save_data(request.session_id, "df_CSR_on_ERA_grid", df_CSR_on_ERA_grid)
        session_manager.save_data(request.session_id, "grace_dataset", request.grace_dataset)
        session_manager.save_data(request.session_id, "bounds", {
            "lat_min": request.lat_min,
            "lat_max": request.lat_max,
            "lon_min": request.lon_min,
            "lon_max": request.lon_max
        })
        
        return {
            "status": "success",
            "data": {
                "grace_dataset": request.grace_dataset,
                "era5_shape": list(df_ERA.shape),
                "csr_shape": list(df_CSR.shape),
                "merged_shape": list(merged.shape),
                "era5_variables": list(df_ERA.columns),
                "date_range": {
                    "start": str(df_ERA.index.min()),
                    "end": str(df_ERA.index.max())
                },
                "bounds": {
                    "lat_min": request.lat_min,
                    "lat_max": request.lat_max,
                    "lon_min": request.lon_min,
                    "lon_max": request.lon_max
                },
                "message": "Data loaded and stored in session"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error_code": "DATA_LOAD_ERROR",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.get("/coastlines")
async def get_coastlines():
    """Get coastline GeoJSON data for 3D globe visualization"""
    try:
        # Try Natural Earth CDN first (fast)
        url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.geojson"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            return {
                "status": "success",
                "data": response.json(),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Fallback to GitHub
        fallback_url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
        response = requests.get(fallback_url, timeout=10)
        
        if response.status_code == 200:
            return {
                "status": "success",
                "data": response.json(),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        raise ValueError("Could not fetch coastlines from any source")
    
    except Exception as e:
        print(f"❌ Error fetching coastlines: {e}")
        return {
            "status": "error",
            "error_code": "COASTLINES_ERROR",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
