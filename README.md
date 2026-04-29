# 🌍 GRACE Downscaling Engine

**A full-stack geographic application for downscaling GRACE satellite data using machine learning**

![Status](https://img.shields.io/badge/Status-Stable-brightgreen)
![Backend](https://img.shields.io/badge/Backend-FastAPI-green)
![Frontend](https://img.shields.io/badge/Frontend-React-blue)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20Random%20Forest-orange)

---

## 📋 Overview

The GRACE Downscaling Engine is a  tool that combines GRACE (Gravity Recovery and Climate Experiment) satellite data with ERA5 reanalysis data to create high-resolution groundwater storage estimates using machine learning techniques.

### Key Features

- 🗺️ **Interactive Geographic Selection** - 3D globe interface for selecting study areas
- 🧠 **Automated Feature Selection** - RFE (Recursive Feature Elimination) for optimal features
- 🦾 **ML Model Training** - Random Forest & XGBoost with hyperparameter tuning
- 📊 **Visualization** - ERA5 variable maps and GRACE comparison plots
- 📈 **Statistical Analysis** - Model evaluation and feature importance analysis

---

## 🏗️ Architecture & Project Structure

This project has evolved into a fully decoupled web application architecture.

```
grace_project/
├── backend_api/          # FastAPI REST API
│   ├── routes/           # API endpoints (setup, training, visualization)
│   ├── main.py           # FastAPI application
│   └── session_manager.py
├── frontend_react/       # React + Vite dynamic frontend
│   └── src/
│       ├── components/   # React components (5 workflow tabs)
│       └── services/     # API service layer
├── code/                 # Core Python ML Data Pipeline
│   ├── main_4_app.py     # Data processing pipeline entry point
│   ├── training.py       # Random Forest & XGBoost models
│   ├── vis_4_app.py      # Plotly & Seaborn map generation
│   └── data_processing.py
└── data/                 # Raw data files (GRACE, ERA5) (Requires user download)
```

---

## 🚀 How to Run Locally (Developer Mode)

If you are on the `main` branch, you can run the project as a standard web application. You will need to start both the Python backend server and the Node.js frontend server.

### 1. Prerequisites & Required Data
- Python 3.11+
- Node.js 18+
- A `data/` folder placed in the root of the project containing your datasets. **Your files must be named exactly as follows (case-sensitive):**
  - `ERA5_data.nc` (Your ERA5 climate variables dataset)
  - `CSR_Mascon_Grace.nc` (GRACE CSR dataset)
  - `JPL_MASCON_GRACE.nc` (GRACE JPL dataset)

### 2. Start the Backend (FastAPI)
Open a terminal, activate your virtual environment, and install dependencies:
```bash
# Create and activate venv using uv (Ultra-fast Python package installer)
uv venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# Install all required Python packages
uv pip install -r grace_requirements.txt
uv pip install -r backend_api/requirements.txt
uv pip install fastapi uvicorn python-multipart

# Start the server
cd backend_api
uvicorn main:app --reload --port 5321
```
*The backend API will run on `http://127.0.0.1:5321`*

### 3A. Start the Modern Frontend (React)
Open a **second** terminal, navigate to the React folder, and start the development server:
```bash
cd frontend_react
npm install
npm run dev
```
*The React UI will automatically open in your browser, typically at `http://localhost:5173`*

### 3B. Start the Legacy Frontend (Streamlit)
If you prefer the original Streamlit interface, you can start it instead of (or alongside) the React UI. Open a new terminal and run:
```bash
streamlit run code/app.py
```
*The Streamlit UI will open at `http://localhost:8501`*

---

## ⚡ Quick Start Scripts (Mac/Linux)
If you are on a Unix-based system, you can use the provided bash scripts to launch the services without manually typing the commands:
```bash
./QUICK_START.sh
```
This script will display a menu of helper commands (like `./start-streamlit.sh` or `./start-react.sh`) to automatically spin up the interfaces.

---

## 💻 Standalone Desktop Application (.EXE)

We have successfully packaged the entire Web UI, FastAPI backend, and Machine Learning pipeline into a single, portable Windows Executable!

If you want to use the `.exe` version or build it yourself, **switch to the `exe` branch**.

```bash
git fetch origin
git checkout exe
```

### How the Standalone Version Works:
On the `exe` branch, the React frontend is pre-compiled into static HTML/JS files, and the FastAPI backend serves them directly. A master Python script (`main_exe.py`) binds them together, meaning **you do not need Node.js or a web server to run it!**

### Building the Executable Yourself
If you are on the `exe` branch and want to recompile the desktop app using PyInstaller:
```bash
# Ensure frontend is built
cd frontend_react
npm run build
cd ..

# Install PyInstaller
uv pip install pyinstaller

# Package the entire stack into an executable
uv run pyinstaller grace_app.spec -y --clean --distpath dist_final
```
The final application will be generated inside `dist_final/GraceDownscalingEngine/`. 

### Running the Executable
**CRITICAL DATA SETUP:** The executable does not package the massive 10GB+ dataset files. To run it successfully:
1. You MUST create a folder named `data` exactly next to your new `GraceDownscalingEngine.exe` file.
2. Place your explicitly named `.nc` files (`ERA5_data.nc`, `CSR_Mascon_Grace.nc`, etc.) inside that `data` folder.
3. Double-click the `.exe`. The app will launch a backend console and automatically open your default browser to the UI!

---

## 📚 Workflow Guide

The application follows a strict 5-step workflow:

1. **Geographic Setup**: Use the interactive 3D globe to select your bounding box. The backend automatically slices the massive global `.nc` files to your targeted coordinates.
2. **Feature Selection**: Run Recursive Feature Elimination (RFE) to determine which ERA5 climate variables (Temperature, Soil Moisture, etc.) correlate best with GRACE anomalies in your region.
3. **Model Training**: Train an XGBoost or Random Forest model (with automatic hyperparameter tuning) to learn the relationship between the ERA5 data and GRACE.
4. **Visualization**: Generate highly detailed, publication-ready geospatial maps comparing the observed GRACE data vs the predicted AI model outputs.
5. **Statistical Analysis**: Select a specific coordinate within your bounds to view time-series breakdowns, R-scores, and feature importance pie charts.

---

## 🤝 Support & Acknowledgments

- **GRACE Mission:** NASA/DLR
- **ERA5 Data:** Copernicus Climate Data Store
- **Authors:** Alexandros Karachles & Anastasia I. Triantafyllou (ANASTRIA-LAB)

For issues or questions regarding the codebase, please open an issue on the GitHub repository.

🌍 **Building the future of groundwater monitoring with AI** 🚀
