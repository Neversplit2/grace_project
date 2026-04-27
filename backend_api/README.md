# GRACE Downscaling Engine - REST API Backend (backend_api)

FastAPI backend for the GRACE Downscaling Engine React frontend.

## Quick Start

### Prerequisites
- Uses existing `uv` venv in `/home/triana04/work/testing/experiment/.venv`
- Python 3.13.7 (via uv)

### 1. Install Dependencies (using uv)

```bash
# From project root
cd /home/triana04/work/testing/experiment

# Install/reinstall packages
uv pip install fastapi uvicorn pydantic python-multipart
```

### 2. Run the Server

```bash
# Activate venv
source /home/triana04/work/testing/experiment/.venv/bin/activate

# Navigate to backend_api
cd grace_project/backend_api

# Run with uvicorn (default: http://localhost:8000)
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

### 3. View Interactive API Documentation

Visit: **http://localhost:5000/docs**

You can test all endpoints directly from the browser!

## API Endpoints

### Health & Config
- `GET /api/health` - Check if backend is running
- `GET /api/config` - Get system configuration

### Session Management
- `POST /api/session/create` - Create new session
- `GET /api/session/{session_id}` - Get session info
- `DELETE /api/session/{session_id}` - Delete session

### Tab 1: Setup & Bounds
- `POST /api/setup/validate-bounds` - Validate geographic bounds
- `POST /api/setup/load-data` - Load GRACE & ERA5 data
- `GET /api/setup/coastlines` - Get coastlines GeoJSON

### Tab 2: Data Processing
- `POST /api/data-processing/data-prep` - Start data preparation
- `POST /api/data-processing/rfe` - Start RFE feature selection
- `GET /api/data-processing/status/{task_id}` - Get task status
- `GET /api/data-processing/result/{task_id}` - Get task results

### Tab 3: Model Training
- `POST /api/training/start` - Start model training
- `GET /api/training/status/{task_id}` - Get training progress
- `GET /api/training/result/{task_id}` - Get training results
- `POST /api/training/upload-model` - Upload pre-trained model

### Tab 4: Maps & Visualization
- `POST /api/maps/era5-maps` - Generate ERA5 map
- `POST /api/maps/grace-comparison` - Generate GRACE comparison map
- `GET /api/maps/download/{map_type}` - Download map

### Tab 5: Statistical Analysis
- `POST /api/analysis/evaluate` - Evaluate model at location
- `POST /api/analysis/feature-importance` - Get feature importance

## Project Structure

```
backend_exe/
├── main.py                     # Entry point
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── routes/                     # API endpoint modules
│   ├── setup.py               # Tab 1 endpoints
│   ├── data_processing.py     # Tab 2 endpoints
│   ├── training.py            # Tab 3 endpoints
│   ├── maps.py                # Tab 4 endpoints
│   ├── analysis.py            # Tab 5 endpoints
│   └── utils.py               # Utility endpoints
└── utils/
    ├── session_manager.py     # Session handling
    └── __init__.py
```

## Response Format

All endpoints return JSON with this structure:

### Success Response
```json
{
  "status": "success",
  "data": { /* response data */ },
  "timestamp": "2024-04-27T10:30:00Z"
}
```

### Error Response
```json
{
  "status": "error",
  "error_code": "ERROR_CODE",
  "message": "Error description",
  "timestamp": "2024-04-27T10:30:00Z"
}
```

## Session Management

Every request requires a `session_id`:

1. Create session: `POST /api/session/create`
2. Use returned `session_id` in all subsequent requests
3. Data is stored per-session and cleaned up after 24 hours

## Background Tasks

Long-running operations use background tasks:

1. Start task (e.g., `POST /api/training/start`) → returns `task_id`
2. Poll status: `GET /api/training/status/{task_id}`
3. Get results: `GET /api/training/result/{task_id}`

Example polling:
```javascript
const interval = setInterval(async () => {
  const res = await fetch(`/api/training/status/${taskId}`);
  const data = await res.json();
  
  if (data.data.status === 'complete') {
    clearInterval(interval);
    // Fetch full results
    const results = await fetch(`/api/training/result/${taskId}`);
  }
}, 500);
```

## Image Data (Base64)

Maps and plots are returned as base64-encoded images:

```json
{
  "data": {
    "image": "data:image/png;base64,iVBORw0KGg..."
  }
}
```

Use directly in HTML/React:
```html
<img src={imageBase64} alt="Map" />
```

## Integration with React Frontend

React frontend (running on `http://localhost:3001`) communicates with this API via:

```javascript
fetch('/api/setup/validate-bounds', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    session_id: sessionId,
    lat_min: -17,
    lat_max: 5,
    lon_min: -80,
    lon_max: -50
  })
})
.then(r => r.json())
.then(data => console.log(data))
```

## Development Notes

### Adding New Endpoints

1. Create function in appropriate route file (e.g., `routes/setup.py`)
2. Add Pydantic model for request validation
3. Decorate with `@router.post()` or `@router.get()`
4. Return standard JSON format
5. Endpoint automatically added to `/docs`

### Testing

Use the built-in Swagger UI at `http://localhost:5000/docs`:
- Click on endpoint
- Click "Try it out"
- Fill in parameters
- Click "Execute"
- See response

### Environment Variables

Create `.env` file:
```
API_HOST=0.0.0.0
API_PORT=5000
SESSION_TIMEOUT_HOURS=24
```

Load with:
```python
from dotenv import load_dotenv
load_dotenv()
```

## Deployment

### Production Server

Use Gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 "main:app" --worker-class uvicorn.workers.UvicornWorker
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
```

Build and run:
```bash
docker build -t grace-api .
docker run -p 5000:5000 grace-api
```

### Standalone EXE (PyInstaller)

```bash
pip install pyinstaller
pyinstaller --onefile --add-data "session_data:session_data" main.py
```

## Troubleshooting

### Port Already in Use
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Or use different port
uvicorn main:app --port 8000
```

### CORS Issues
Check that React frontend can reach API:
```bash
curl -X OPTIONS http://localhost:5000/api/health -v
```

### Sessions Not Persisting
Check `./session_data/` directory exists and is writable

## Next Steps

- [ ] Integrate actual Python functions from `../code/`
- [ ] Add database for persistent session storage
- [ ] Implement WebSocket for real-time progress
- [ ] Add authentication/authorization
- [ ] Write unit tests
- [ ] Deploy to production server
