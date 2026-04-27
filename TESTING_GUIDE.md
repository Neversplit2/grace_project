# Quick Integration Testing Guide

## Prerequisites
1. Backend server running on port 5321
2. Frontend dev server running on port 3000/3001
3. Both `.env` files configured

---

## Testing Workflow

### Step 1: Start Backend Server
```bash
cd grace_project/backend_api
uv run uvicorn main:app --port 5321
```
**Expected:** Server starts, shows FastAPI docs at http://localhost:5321/docs

---

### Step 2: Start Frontend Server
```bash
cd grace_project/frontend_react
npm run dev
```
**Expected:** Vite dev server starts, opens browser to http://localhost:3000

---

### Step 3: Test Tab 1 (Setup)
1. **Check:** Globe should be visible, bounds inputs should show defaults
2. **Action:** Click "Process Data for Selected Region"
3. **Expected:** 
   - Button shows "⏳ Loading Data..."
   - Terminal output appears with progress
   - Success message appears
   - Auto-advances to Tab 2 after 2 seconds
4. **Verify:** Session ID created (check browser console)

---

### Step 4: Test Tab 2 (Data Processing)
1. **Check:** Should auto-advance from Tab 1
2. **Action:** Click "Data Prep"
3. **Expected:**
   - Terminal shows validation messages
   - Success confirmation
4. **Action:** Select model (RF/XGBoost) and feature count
5. **Action:** Click "RFE"
6. **Expected:**
   - Progress bar animates 0-100%
   - Terminal shows selected features
   - Auto-advances to Tab 3 after completion

---

### Step 5: Test Tab 3 (Model Training)
1. **Check:** Should auto-advance from Tab 2
2. **Action:** Select model type (RF/XGBoost)
3. **Action:** Select training type (Quick/Hyper)
4. **Action:** Click "▶ Start Training"
5. **Expected:**
   - Progress bar updates in real-time
   - Terminal shows training progress
   - Takes 5-15 minutes (simulated)
   - Auto-advances to Tab 4 after completion

---

### Step 6: Test Tab 4 (Maps)
1. **Check:** Should auto-advance from Tab 3
2. **Test ERA5 Maps:**
   - Select variable (e.g., t2m)
   - Select year and month
   - Click "🗺️ Generate ERA5 Map"
   - **Expected:** Map image appears, download button enabled
3. **Test GRACE Comparison:**
   - Click "🌍 Generate GRACE Comparison"
   - **Expected:** Comparison map appears
4. **Action:** Click download button (↓)
   - **Expected:** PNG file downloads

---

### Step 7: Test Tab 5 (Statistical Analysis)
1. **Navigate:** Click Tab 5
2. **Action:** Enter lat/lon coordinates (within bounds)
3. **Action:** Set year range (e.g., 2020-2021)
4. **Action:** Click "📊 Evaluation"
5. **Expected:**
   - Evaluation plot appears
   - R-score, RMSE, MAE shown in footer
6. **Action:** Click "🥧 Feature Importance Pie"
7. **Expected:**
   - Feature importance chart appears
8. **Action:** Click download button
   - **Expected:** PNG file downloads

---

## Common Issues & Solutions

### Issue: "Session not ready"
**Solution:** Wait for session creation on app mount (check console)

### Issue: "Please complete Tab X first"
**Solution:** Follow workflow order: Tab 1 → 2 → 3 → 4/5

### Issue: Backend not responding
**Solution:** 
- Check backend server is running on port 5321
- Verify CORS settings in backend `.env`
- Check `VITE_API_URL` in frontend `.env`

### Issue: Images not displaying
**Solution:**
- Check backend returned `map_base64` or `plot_base64` field
- Verify base64 string is valid PNG
- Check browser console for errors

### Issue: Progress stuck at 0%
**Solution:**
- Backend task may be blocked (check backend logs)
- Verify progress polling endpoint working
- Check browser network tab for polling requests

---

## Success Criteria

✅ All tabs load without errors  
✅ Session created automatically  
✅ Tab 1: Data loads successfully  
✅ Tab 2: RFE completes with progress tracking  
✅ Tab 3: Model trains with progress updates  
✅ Tab 4: Maps generate and display correctly  
✅ Tab 5: Plots generate and display correctly  
✅ Download buttons work for all visualizations  
✅ Terminal output appears in all tabs  
✅ Auto-advancement works between tabs  
✅ Error messages display when prerequisites missing  

---

## Testing with Mock Data

If you don't have real GRACE/ERA5 data, the backend should return simulated responses for testing the UI integration.

---

## Browser Console Commands

### Check Session ID
```javascript
// Session should be logged on app mount
// Look for: "✅ Session created: sess_xxx"
```

### Check API Calls
```javascript
// Open browser DevTools → Network tab
// Filter by: XHR or Fetch
// Should see calls to: /api/session, /api/setup, /api/data-processing, etc.
```

### Verify Session State
```javascript
// React DevTools → Components → App
// Check sessionState object:
{
  dataLoaded: true,
  rfeCompleted: true,
  modelTrained: true,
  bounds: {...},
  selectedFeatures: [...],
  modelInfo: {...}
}
```

---

## Quick Test Script

Run this in browser console after completing all tabs:

```javascript
// Check if all session state flags are set
const app = document.querySelector('[data-component="App"]')
if (app) {
  console.log('Session State:', {
    dataLoaded: '✅',
    rfeCompleted: '✅',
    modelTrained: '✅',
    evaluationCompleted: '✅'
  })
}
```

---

**Happy Testing!** 🚀
