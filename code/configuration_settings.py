#Import libraries
from pathlib import Path
import os

import sys

# Determine the project root dynamically
if getattr(sys, 'frozen', False):
    # If running as a compiled PyInstaller executable, the root is where the .exe sits
    PROJECT_ROOT = Path(sys.executable).parent
else:
    # If running as a standard python script, it's the parent of the code/ folder
    CODE_DIR = Path(__file__).parent.absolute()
    PROJECT_ROOT = CODE_DIR.parent

# File structure settings - use absolute paths
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = PROJECT_ROOT / "models"
MAPS_DIR = PROJECT_ROOT / "Maps"

ERA5_FILE = "ERA5_data.nc"
GRACE_CSR_FILE = "CSR_Mascon_Grace.nc"
GRACE_JPL_FILE = "JPL_MASCON_GRACE.nc"

# Ensure that user has the wanted folder structure
# If not create it
for d in [DATA_DIR, RESULTS_DIR, MODEL_DIR, MAPS_DIR]:
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"⚠️ Warning: Could not create directory {d}: {e}")