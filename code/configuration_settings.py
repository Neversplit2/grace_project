#Import libraries
from pathlib import Path
#File structure settings
DATA_DIR = Path("../data")
RESULTS_DIR = Path("../results")
MODELS_DIR =Path("../models")
MAPS_DIR = Path("../maps")

ERA5_FILE = "ERA5_data.nc"
GRACE_CSR_FILE = "CSR_Mascon_Grace.nc"
# Ensure that user has the wanted folder structure
# If not create it
# Creating a loop to check 
for d in [DATA_DIR, RESULTS_DIR, MODELS_DIR, MAPS_DIR]:
    d.mkdir(parents=True, exist_ok=True) #Parents= true: Create the folder, #exist_ok= true: If it already exists do nothing