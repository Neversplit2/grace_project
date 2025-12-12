# Backend Notebook - Setup Guide

This document provides setup instructions for the `backend_notebook.ipynb` Jupyter notebook.

## Required Libraries

Install the following Python libraries before running the notebook:

```bash
pip install xarray pandas matplotlib cartopy numpy scikit-learn xgboost joblib
```

### Library Breakdown:
- **xarray**: For reading and processing NetCDF (.nc) files
- **pandas**: For data manipulation and analysis
- **matplotlib**: For creating visualizations and plots
- **cartopy**: For geographical maps and coordinate transformations
- **numpy**: For numerical operations
- **scikit-learn**: For machine learning algorithms (Random Forest, feature selection)
- **xgboost**: For XGBoost machine learning model
- **joblib**: For saving and loading trained models

### Optional - Using conda:

```bash
conda install -c conda-forge xarray pandas matplotlib cartopy numpy scikit-learn xgboost joblib
```

## Folder Structure

The notebook expects the following folder structure:

```
<root_folder>/
├── code/
│   ├── backend_notebook.ipynb    # The main notebook
│   └── README_notebook.md         # This file
│
├── data/
│   ├── CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc
│   ├── data_stream-moda_stepType-avgad.nc
│   └── data_stream-moda_stepType-avgua.nc
│
├── results/                      # Generated models and plots
│   ├── model.pkl
│   └── random_forest_training_curve_fast.png
│
└── maps/                         # Generated prediction maps
    └── Grace_Actual_vs_Predicted.jpg
```

### Notes:
- `<root_folder>` is your project's root directory 
- The notebook uses **relative paths** (`../data/`) to access data files
- Ensure all three NetCDF data files are placed in the `data/` folder
- The notebook should be run from the `code/` directory

## Running the Notebook

1. Ensure you're in the `code/` directory
2. Start Jupyter:
   ```bash
   jupyter notebook
   ```
3. Open `backend_notebook.ipynb`
4. Run cells sequentially, starting from cell #1

## Data Files

The notebook requires three NetCDF data files:

1. **CSR GRACE/GRACE-FO**: `CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc`
   - Contains liquid water equivalent thickness measurements
   
2. **ERA5 Dataset 1**: `data_stream-moda_stepType-avgad.nc`
   - Contains averaged atmospheric data (precipitation, evaporation, runoff, etc.)
   
3. **ERA5 Dataset 2**: `data_stream-moda_stepType-avgua.nc`
   - Contains temperature, soil moisture, and vegetation data

## Workflow Overview

The notebook performs the following steps:

1. **Import libraries** (Cell #1)
2. **Load datasets** (Cell #2) - Uses relative paths
3. **Read NetCDF files** (Cell #3)
4. **Spatial slicing** (Cell #4) - Select region of interest
5. **Data preprocessing** (Cells #6-11) - Time alignment, merging, regridding
6. **Feature selection** (Cell #14) - Using RFE (Recursive Feature Elimination)
7. **Model training** (Cell after #14) - Hyperparameter tuning and final model training
8. **Training Curves** (Cell #14 - second instance) - Visualizes model learning performance
9. **Map Generation** (Last Cell) - Creates side-by-side comparison maps of Actual vs Predicted data

## Output Files

The notebook generates the following outputs in the `../results/` and `../maps/` directories (created automatically if missing):

- **Trained Model**: `../results/<model_name>.pkl`
- **Training Curve**: `../results/random_forest_training_curve_fast.png`
- **Prediction Map**: `../maps/Grace_Actual_vs_Predicted.jpg`

## Troubleshooting

- **Import errors**: Ensure all libraries are installed
- **File not found**: Verify folder structure and that data files exist in `../data/`
- **Path issues**: Check that you're running the notebook from the `code/` directory
- **Map Generation Error**: If you see a "non-unique MultiIndex" error, ensure the last cell uses `groupby(...).mean()` instead of `set_index(...)` to handle duplicate coordinates.
