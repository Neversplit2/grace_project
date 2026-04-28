# 🌍 GRACE Downscaling Engine

**A full-stack web application for downscaling GRACE satellite data using machine learning**

![Status](https://img.shields.io/badge/Status-In%20Development-yellow)
![Backend](https://img.shields.io/badge/Backend-FastAPI-green)
![Frontend](https://img.shields.io/badge/Frontend-React-blue)
![Python](https://img.shields.io/badge/Python-3.13-blue)

---

## 📋 Overview

The GRACE Downscaling Engine is a sophisticated tool that combines GRACE (Gravity Recovery and Climate Experiment) satellite data with ERA5 reanalysis data to create high-resolution groundwater storage estimates using machine learning techniques.

### Key Features

- 🗺️ **Interactive Geographic Selection** - 3D globe interface for selecting study areas
- 🧠 **Automated Feature Selection** - RFE (Recursive Feature Elimination) for optimal features
- 🦾 **ML Model Training** - Random Forest & XGBoost with hyperparameter tuning
- 📊 **Visualization** - ERA5 variable maps and GRACE comparison plots
- 📈 **Statistical Analysis** - Model evaluation and feature importance analysis

---

## 🏗️ Architecture

```
grace_project/
├── backend_api/          # FastAPI REST API (22 endpoints)
│   ├── routes/           # API route handlers
│   ├── main.py          # FastAPI application
│   └── session_manager.py
├── frontend_react/       # React + Vite frontend
│   └── src/
│       ├── components/   # React components (5 tabs)
│       └── services/     # API service layer
├── code/                 # Python ML pipeline
│   ├── main_4_app.py    # Data processing pipeline
│   ├── training.py      # Model training
│   ├── vis_4_app.py     # Visualization
│   └── data_processing.py
└── data/                 # Data files (GRACE, ERA5)
```

### Technology Stack

**Backend:**
- FastAPI 1.0.0
- Python 3.13.7
- Uvicorn ASGI server
- Session-based state management

**Frontend:**
- React 18.2.0
- Vite 5.4.21
- Plotly.js for 3D visualizations
- Fetch API for backend communication

**ML Pipeline:**
- scikit-learn (RFE, Random Forest)
- XGBoost
- matplotlib for plotting
- xarray for NetCDF handling

---

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- Node.js 18+
- ~20-30 GB disk space for data
- CDS API credentials (for data download)

### 1. Clone Repository

```bash
git clone https://github.com/Neversplit2/grace_project.git
cd grace_project
```

### 2. Backend Setup

```bash
# Create Python virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install fastapi uvicorn pydantic python-multipart joblib

# Start backend server
cd backend_api
uvicorn main:app --reload --port 5321
```

Backend will be available at: http://localhost:5321

**API Documentation:** http://localhost:5321/docs

### 3. Frontend Setup

```bash
# Install dependencies
cd frontend_react
npm install

# Configure API URL (copy .env.example to .env)
cp .env.example .env

# Start development server
npm run dev
```

Frontend will be available at: http://localhost:3001

**Note:** Make sure to create `.env` from `.env.example` before running. The default configuration connects to `http://localhost:5321`.

---

## 📖 Usage Guide

### Workflow Overview

The application follows a 5-tab workflow:

```
Tab 1: Setup & Area Selection
  ↓ (Define geographic bounds, load GRACE + ERA5 data)
Tab 2: Data Processing
  ↓ (Run RFE for feature selection)
Tab 3: Model Training
  ↓ (Train Random Forest or XGBoost model)
Tab 4: Maps & Visualization
  ↓ (Generate ERA5 and GRACE comparison maps)
Tab 5: Statistical Analysis
  ↓ (Evaluate model, analyze feature importance)
```

### Tab 1: Geographic Setup

1. **Select Study Area**
   - Use 3D globe to visualize region
   - Set latitude bounds (-90 to 90)
   - Set longitude bounds (-180 to 180)

2. **Choose GRACE Dataset**
   - CSR (Center for Space Research)
   - JPL (Jet Propulsion Laboratory)

3. **Load Data**
   - Validates bounds
   - Downloads/loads GRACE mascon data
   - Downloads/loads ERA5 climate variables
   - Processing time: 1-3 minutes

### Tab 2: Data Processing & Feature Selection

1. **Prepare Data**
   - Validates loaded data
   - Shows available features

2. **Run RFE**
   - Choose model type (RF or XGBoost)
   - Select number of features (5-15 recommended)
   - Processing time: 2-5 minutes
   - Results: List of selected features

### Tab 3: Model Training

1. **Configure Training**
   - Model: Random Forest or XGBoost
   - Type: Quick (fast) or Hyper (optimized)

2. **Train Model**
   - Background processing
   - Real-time progress updates
   - Processing time: 5-15 minutes
   - Results: Trained model + metrics

3. **Upload Pre-trained Model** (optional)
   - Upload .pkl file
   - Skip training step

### Tab 4: Maps & Visualization

1. **ERA5 Variable Maps**
   - Choose variable (temp, precipitation, etc.)
   - Select year and month
   - Generate high-resolution map

2. **GRACE Comparison**
   - Observed vs Predicted
   - Select year and month
   - Side-by-side visualization

### Tab 5: Statistical Analysis

1. **Model Evaluation**
   - Select location (lat/lon)
   - Choose date range
   - View time series plots
   - Get R-score and p-value

2. **Feature Importance**
   - Pie chart visualization
   - Ranked feature list
   - Percentage contributions

---

## 🧪 Testing

### Backend API Tests

```bash
cd backend_api

# Run validation tests (no data required)
python test_validation.py
# Expected: 15/15 tests PASS

# Run end-to-end tests (requires real data)
python test_end_to_end.py
```

### Manual API Testing

```bash
# Health check
curl http://localhost:5321/api/health

# Create session
curl -X POST http://localhost:5321/api/session/create

# Validate bounds
curl -X POST http://localhost:5321/api/setup/validate-bounds \
  -H "Content-Type: application/json" \
  -d '{"session_id":"YOUR_SESSION","lat_min":-20,"lat_max":5,"lon_min":-80,"lon_max":-45}'
```

---

## 📚 Documentation

| Document | Description | Location |
|----------|-------------|----------|
| **API Quick Reference** | All 22 endpoints with examples | [`backend_api/QUICK_REFERENCE.md`](backend_api/QUICK_REFERENCE.md) |
| **Test Summary** | Testing results and coverage | [`backend_api/TEST_SUMMARY.md`](backend_api/TEST_SUMMARY.md) |
| **Integration Status** | Frontend-backend integration | [`frontend_react/INTEGRATION_STATUS.md`](frontend_react/INTEGRATION_STATUS.md) |
| **Backend README** | Backend setup details | [`backend_api/README.md`](backend_api/README.md) |
| **Frontend README** | Frontend setup details | [`frontend_react/README.md`](frontend_react/README.md) |

---

## 🔧 Configuration

### Backend Configuration

**Environment Variables:**
- `PORT`: Server port (default: 5321)
- `HOST`: Server host (default: 127.0.0.1)

**Files:**
- `backend_api/main.py`: CORS settings, middleware
- `code/configuration_settings.py`: Data paths, model settings

### Frontend Configuration

**Environment Variables:**
```bash
# frontend_react/.env
VITE_API_URL=http://localhost:5321
```

---

## 📊 Project Status

### ✅ Completed

- [x] Backend API implementation (22 endpoints)
- [x] Session management system
- [x] Background task processing
- [x] API service layer (frontend)
- [x] Tab 1 integration (Geographic Setup)
- [x] Validation testing (100% pass)
- [x] API documentation

### ⏳ In Progress

- [ ] Tab 2-5 frontend integration
- [ ] End-to-end testing with real data
- [ ] Error boundaries and retry logic
- [ ] Progress persistence

### 📅 Planned

- [ ] Deployment configuration
- [ ] Docker containerization
- [ ] Standalone EXE packaging
- [ ] User authentication
- [ ] Result export (CSV, plots)
- [ ] Multi-user support

---

## 🐛 Known Issues

### Issue 1: Data Loading Requires CDS API
**Solution:** Set up CDS API credentials or create mock data for testing

### Issue 2: Large Memory Requirements
**Solution:** Ensure ~16 GB RAM for full workflow

### Issue 3: Processing Time
**Solution:** Background tasks implemented, but still requires patience

---

## 🤝 Contributing

This is currently a research project. For questions or collaboration:

**Repository:** https://github.com/Neversplit2/grace_project

---

## 📄 License

[Specify your license here]

---

## 🙏 Acknowledgments

- **GRACE Mission:** NASA/DLR
- **ERA5 Data:** Copernicus Climate Data Store
- **Natural Earth:** Map data

---

## 📞 Support

For issues, questions, or feature requests:

1. Check documentation in `backend_api/` and `frontend_react/`
2. Review `backend_api/TEST_SUMMARY.md` for testing info
3. See `backend_api/QUICK_REFERENCE.md` for API details

---

**Last Updated:** April 27, 2026  
**Version:** 1.0.0-beta  
**Status:** 🟡 Development (Tab 1 integration complete)

🌍 **Building the future of groundwater monitoring with AI** 🚀
