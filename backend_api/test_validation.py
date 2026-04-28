#!/usr/bin/env python3
"""
Validation Testing Script - Tests API structure without requiring real data
Tests all endpoints for proper validation logic and error handling
"""

import requests
import json
import sys

BASE_URL = "http://localhost:5321"
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def log(message, color=RESET):
    print(f"{color}{message}{RESET}")

def test_endpoint(name, method, url, payload=None, expected_error=None):
    """Test an endpoint and check for expected behavior"""
    try:
        if method == "GET":
            response = requests.get(url, params=payload)
        elif method == "POST":
            response = requests.post(url, json=payload)
        elif method == "DELETE":
            response = requests.delete(url)
        
        data = response.json()
        
        # Check if we got the expected error
        if expected_error:
            status = data.get('status', '')
            message = str(data.get('message', '') or data.get('detail', '')).lower()
            
            if status == 'error' or 'detail' in data:
                if any(err.lower() in message for err in expected_error) if isinstance(expected_error, list) else expected_error.lower() in message:
                    log(f"  ✅ {name}: Validation working correctly", GREEN)
                    log(f"     Error: {data.get('message') or data.get('detail', '')[:100]}", BLUE)
                    return True
            log(f"  ❌ {name}: Expected error, got: {data.get('message') or data.get('detail', '')}", RED)
            return False
        else:
            if data.get('status') == 'success' or data.get('status') == 'ok':
                log(f"  ✅ {name}: Success", GREEN)
                return True
            else:
                log(f"  ⚠️  {name}: {data.get('message', 'Unknown error')}", YELLOW)
                return False
    except Exception as e:
        log(f"  ❌ {name}: Exception - {str(e)}", RED)
        return False

def main():
    log("\n" + "="*80, BLUE)
    log("🧪 API VALIDATION TESTING (No Real Data Required)", BLUE)
    log("="*80, BLUE)
    
    results = []
    
    # Test 1: Health Check
    log("\n📍 Testing: Server Health", BLUE)
    results.append(test_endpoint(
        "Health Check",
        "GET",
        f"{BASE_URL}/api/health"
    ))
    
    # Test 2: Create Session
    log("\n📍 Testing: Session Management", BLUE)
    response = requests.post(f"{BASE_URL}/api/session/create")
    session_data = response.json()
    session_id = session_data.get('session_id')
    
    if session_id:
        log(f"  ✅ Session created: {session_id}", GREEN)
        results.append(True)
    else:
        log(f"  ❌ Session creation failed", RED)
        results.append(False)
        return
    
    # Test 3: Invalid session
    log("\n📍 Testing: Invalid Session Handling", BLUE)
    results.append(test_endpoint(
        "Invalid Session",
        "POST",
        f"{BASE_URL}/api/setup/validate-bounds",
        {"session_id": "invalid_session_id", "lat_min": 0, "lat_max": 10, "lon_min": 0, "lon_max": 10},
        expected_error=["session not found", "session"]
    ))
    
    # Test 4: Tab 1 - Validate Bounds (Valid)
    log("\n📍 Testing: Tab 1 - Geographic Setup", BLUE)
    results.append(test_endpoint(
        "Valid Bounds",
        "POST",
        f"{BASE_URL}/api/setup/validate-bounds",
        {"session_id": session_id, "lat_min": -20, "lat_max": 5, "lon_min": -80, "lon_max": -45}
    ))
    
    # Test 5: Tab 1 - Invalid Bounds (lat > 90)
    results.append(test_endpoint(
        "Invalid Latitude",
        "POST",
        f"{BASE_URL}/api/setup/validate-bounds",
        {"session_id": session_id, "lat_min": -20, "lat_max": 95, "lon_min": -80, "lon_max": -45},
        expected_error=["invalid", "out of range"]
    ))
    
    # Test 6: Tab 2 - Missing Data
    log("\n📍 Testing: Tab 2 - Data Processing", BLUE)
    results.append(test_endpoint(
        "Prep without Data",
        "POST",
        f"{BASE_URL}/api/data-processing/prep",
        {"session_id": session_id},
        expected_error=["no data", "please load data"]
    ))
    
    # Test 7: Tab 2 - RFE without Data
    results.append(test_endpoint(
        "RFE without Data",
        "POST",
        f"{BASE_URL}/api/data-processing/rfe",
        {"session_id": session_id, "model_type": "RF", "n_features": 5},
        expected_error=["no data", "please load data"]
    ))
    
    # Test 8: Tab 3 - Training without RFE
    log("\n📍 Testing: Tab 3 - Model Training", BLUE)
    results.append(test_endpoint(
        "Training without RFE",
        "POST",
        f"{BASE_URL}/api/training/start",
        {"session_id": session_id, "model": "RF", "train_type": "Quick"},
        expected_error=["not found", "please run", "rfe", "selected_features"]
    ))
    
    # Test 9: Tab 4 - Maps without Data
    log("\n📍 Testing: Tab 4 - Maps & Visualization", BLUE)
    results.append(test_endpoint(
        "ERA5 Map without Data",
        "POST",
        f"{BASE_URL}/api/maps/era5",
        {"session_id": session_id, "variable": "tp", "year": 2015, "month": 6},
        expected_error=["no era5", "no data", "please run"]
    ))
    
    # Test 10: Tab 5 - Analysis without Model
    log("\n📍 Testing: Tab 5 - Statistical Analysis", BLUE)
    results.append(test_endpoint(
        "Evaluate without Model",
        "POST",
        f"{BASE_URL}/api/analysis/evaluate",
        {"session_id": session_id, "latitude": 0, "longitude": -65, "start_year": 2005, "end_year": 2015},
        expected_error=["no model", "no trained", "please run"]
    ))
    
    results.append(test_endpoint(
        "Feature Importance without Model",
        "POST",
        f"{BASE_URL}/api/analysis/feature-importance",
        {"session_id": session_id},
        expected_error=["no model", "no trained", "please run"]
    ))
    
    # Test 11: Coastlines (should work without session)
    log("\n📍 Testing: Utility Endpoints", BLUE)
    results.append(test_endpoint(
        "Coastlines GeoJSON",
        "GET",
        f"{BASE_URL}/api/setup/coastlines"
    ))
    
    # Test 12: Config
    results.append(test_endpoint(
        "Configuration",
        "GET",
        f"{BASE_URL}/api/config"
    ))
    
    # Test 13: Session Info
    results.append(test_endpoint(
        "Session Info",
        "GET",
        f"{BASE_URL}/api/session/{session_id}"
    ))
    
    # Cleanup
    log("\n📍 Testing: Session Cleanup", BLUE)
    results.append(test_endpoint(
        "Delete Session",
        "DELETE",
        f"{BASE_URL}/api/session/{session_id}"
    ))
    
    # Summary
    log("\n" + "="*80, BLUE)
    log("📊 TEST SUMMARY", BLUE)
    log("="*80, BLUE)
    
    passed = sum(results)
    total = len(results)
    percentage = (passed / total * 100) if total > 0 else 0
    
    log(f"\nTests Passed: {passed}/{total} ({percentage:.1f}%)", 
        GREEN if passed == total else YELLOW)
    
    if passed == total:
        log("\n🎉 ALL VALIDATION TESTS PASSED!", GREEN)
        log("✅ API structure is working correctly", GREEN)
        log("✅ All validation logic is functioning", GREEN)
        log("✅ Error handling is comprehensive", GREEN)
        log("\n⏳ Next: Test with real data for full end-to-end workflow", YELLOW)
        return 0
    else:
        log(f"\n⚠️  {total - passed} test(s) failed", RED)
        return 1

if __name__ == "__main__":
    sys.exit(main())
