#!/usr/bin/env python3
"""
End-to-End Testing Script for GRACE Downscaling Engine API
Tests complete workflow: Tab 1 → Tab 2 → Tab 3 → Tab 4 → Tab 5
"""

import requests
import time
import json
import sys
from typing import Dict, Any, Optional

# Configuration
BASE_URL = "http://localhost:5321"
POLL_INTERVAL = 2  # seconds
MAX_WAIT = 300  # 5 minutes max wait for background tasks

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

class APITester:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session_id: Optional[str] = None
        self.test_results = []
        
    def log(self, message: str, color: str = RESET):
        """Print colored log message"""
        print(f"{color}{message}{RESET}")
        
    def test_health(self) -> bool:
        """Test health endpoint"""
        self.log("\n" + "="*80, BLUE)
        self.log("TESTING: Health Check", BLUE)
        self.log("="*80, BLUE)
        
        try:
            response = requests.get(f"{self.base_url}/api/health")
            if response.status_code == 200:
                data = response.json()
                self.log(f"✅ Health check passed: {data.get('status')}", GREEN)
                self.log(f"   Version: {data.get('version')}", GREEN)
                return True
            else:
                self.log(f"❌ Health check failed: {response.status_code}", RED)
                return False
        except Exception as e:
            self.log(f"❌ Health check error: {str(e)}", RED)
            return False
    
    def create_session(self) -> bool:
        """Create a new session"""
        self.log("\n" + "="*80, BLUE)
        self.log("TESTING: Session Creation", BLUE)
        self.log("="*80, BLUE)
        
        try:
            response = requests.post(f"{self.base_url}/api/session/create")
            if response.status_code == 200:
                data = response.json()
                self.session_id = data['session_id']
                self.log(f"✅ Session created: {self.session_id}", GREEN)
                return True
            else:
                self.log(f"❌ Session creation failed: {response.status_code}", RED)
                return False
        except Exception as e:
            self.log(f"❌ Session creation error: {str(e)}", RED)
            return False
    
    def tab1_validate_bounds(self) -> bool:
        """Tab 1: Validate geographic bounds"""
        self.log("\n" + "="*80, BLUE)
        self.log("TAB 1: Geographic Setup - Validate Bounds", BLUE)
        self.log("="*80, BLUE)
        
        # Test with Amazon basin bounds
        bounds = {
            "session_id": self.session_id,
            "lat_min": -20.0,
            "lat_max": 5.0,
            "lon_min": -80.0,
            "lon_max": -45.0
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/setup/validate-bounds",
                json=bounds
            )
            if response.status_code == 200:
                data = response.json()
                self.log(f"✅ Bounds validated", GREEN)
                self.log(f"   Bounds: {data['data']['bounds']}", GREEN)
                return True
            else:
                self.log(f"❌ Bounds validation failed: {response.text}", RED)
                return False
        except Exception as e:
            self.log(f"❌ Bounds validation error: {str(e)}", RED)
            return False
    
    def tab1_load_data(self) -> bool:
        """Tab 1: Load GRACE and ERA5 data"""
        self.log("\n" + "="*80, BLUE)
        self.log("TAB 1: Geographic Setup - Load Data", BLUE)
        self.log("="*80, BLUE)
        
        payload = {
            "session_id": self.session_id,
            "lat_min": -20.0,
            "lat_max": 5.0,
            "lon_min": -80.0,
            "lon_max": -45.0
        }
        
        try:
            self.log("⏳ Loading data (this may take a while)...", YELLOW)
            response = requests.post(
                f"{self.base_url}/api/setup/load-data",
                json=payload,
                timeout=120  # 2 minutes timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                self.log(f"✅ Data loaded successfully", GREEN)
                
                # Debug: print the response structure
                if 'data' not in data:
                    self.log(f"⚠️  Response structure: {list(data.keys())}", YELLOW)
                    self.log(f"   Full response: {data}", YELLOW)
                    return False
                
                info = data['data']
                self.log(f"   ERA5 shape: {info['era5_shape']}", GREEN)
                self.log(f"   GRACE shape: {info.get('csr_shape', info.get('grace_shape'))}", GREEN)
                self.log(f"   Date range: {info['date_range']['start']} to {info['date_range']['end']}", GREEN)
                self.log(f"   Variables: {len(info['era5_variables'])} features", GREEN)
                return True
            else:
                self.log(f"❌ Data loading failed: {response.text}", RED)
                return False
        except requests.Timeout:
            self.log(f"❌ Data loading timed out (>120s)", RED)
            return False
        except Exception as e:
            self.log(f"❌ Data loading error: {str(e)}", RED)
            return False
    
    def tab2_prep_data(self) -> bool:
        """Tab 2: Prepare data for RFE"""
        self.log("\n" + "="*80, BLUE)
        self.log("TAB 2: Data Processing - Prep Data", BLUE)
        self.log("="*80, BLUE)
        
        try:
            response = requests.post(
                f"{self.base_url}/api/data-processing/prep",
                json={"session_id": self.session_id}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.log(f"✅ Data prep successful", GREEN)
                self.log(f"   Available features: {data['data']['available_features']}", GREEN)
                return True
            else:
                self.log(f"❌ Data prep failed: {response.text}", RED)
                return False
        except Exception as e:
            self.log(f"❌ Data prep error: {str(e)}", RED)
            return False
    
    def tab2_run_rfe(self) -> bool:
        """Tab 2: Run RFE (background task)"""
        self.log("\n" + "="*80, BLUE)
        self.log("TAB 2: Data Processing - Run RFE", BLUE)
        self.log("="*80, BLUE)
        
        payload = {
            "session_id": self.session_id,
            "model_type": "RF",  # Random Forest
            "n_features": 5
        }
        
        try:
            # Start RFE
            response = requests.post(
                f"{self.base_url}/api/data-processing/rfe",
                json=payload
            )
            
            if response.status_code != 200:
                self.log(f"❌ RFE start failed: {response.text}", RED)
                return False
            
            task_data = response.json()
            task_id = task_data['data']['task_id']
            self.log(f"⏳ RFE started (task_id: {task_id})", YELLOW)
            
            # Poll for completion
            start_time = time.time()
            while time.time() - start_time < MAX_WAIT:
                time.sleep(POLL_INTERVAL)
                
                status_response = requests.get(
                    f"{self.base_url}/api/data-processing/status/{task_id}",
                    params={"session_id": self.session_id}
                )
                
                if status_response.status_code != 200:
                    self.log(f"❌ Status check failed", RED)
                    return False
                
                status_data = status_response.json()['data']
                status = status_data['status']
                progress = status_data.get('progress', 0)
                
                self.log(f"   Progress: {progress}% - {status}", YELLOW)
                
                if status == 'completed':
                    # Get results
                    result_response = requests.get(
                        f"{self.base_url}/api/data-processing/result/{task_id}",
                        params={"session_id": self.session_id}
                    )
                    
                    if result_response.status_code == 200:
                        result_data = result_response.json()['data']
                        self.log(f"✅ RFE completed successfully", GREEN)
                        self.log(f"   Selected features: {result_data['selected_features']}", GREEN)
                        self.log(f"   Number of features: {result_data['n_features_selected']}", GREEN)
                        return True
                    else:
                        self.log(f"❌ Failed to get RFE results", RED)
                        return False
                
                elif status == 'failed':
                    error_msg = status_data.get('error', 'Unknown error')
                    self.log(f"❌ RFE failed: {error_msg}", RED)
                    return False
            
            self.log(f"❌ RFE timed out after {MAX_WAIT}s", RED)
            return False
            
        except Exception as e:
            self.log(f"❌ RFE error: {str(e)}", RED)
            return False
    
    def tab3_train_model(self) -> bool:
        """Tab 3: Train model (background task)"""
        self.log("\n" + "="*80, BLUE)
        self.log("TAB 3: Model Training - Train Model", BLUE)
        self.log("="*80, BLUE)
        
        payload = {
            "session_id": self.session_id,
            "model_type": "RF",
            "train_type": "grid"
        }
        
        try:
            # Start training
            response = requests.post(
                f"{self.base_url}/api/training/start",
                json=payload
            )
            
            if response.status_code != 200:
                self.log(f"❌ Training start failed: {response.text}", RED)
                return False
            
            task_data = response.json()
            task_id = task_data['data']['task_id']
            self.log(f"⏳ Training started (task_id: {task_id})", YELLOW)
            
            # Poll for completion
            start_time = time.time()
            while time.time() - start_time < MAX_WAIT:
                time.sleep(POLL_INTERVAL)
                
                status_response = requests.get(
                    f"{self.base_url}/api/training/status/{task_id}",
                    params={"session_id": self.session_id}
                )
                
                if status_response.status_code != 200:
                    self.log(f"❌ Status check failed", RED)
                    return False
                
                status_data = status_response.json()['data']
                status = status_data['status']
                progress = status_data.get('progress', 0)
                
                self.log(f"   Progress: {progress}% - {status}", YELLOW)
                
                if status == 'completed':
                    # Get results
                    result_response = requests.get(
                        f"{self.base_url}/api/training/result/{task_id}",
                        params={"session_id": self.session_id}
                    )
                    
                    if result_response.status_code == 200:
                        result_data = result_response.json()['data']
                        self.log(f"✅ Training completed successfully", GREEN)
                        self.log(f"   Model: {result_data['model_name']}", GREEN)
                        self.log(f"   Training samples: {result_data['training_info']['n_samples']}", GREEN)
                        self.log(f"   Features used: {result_data['training_info']['n_features']}", GREEN)
                        return True
                    else:
                        self.log(f"❌ Failed to get training results", RED)
                        return False
                
                elif status == 'failed':
                    error_msg = status_data.get('error', 'Unknown error')
                    self.log(f"❌ Training failed: {error_msg}", RED)
                    return False
            
            self.log(f"❌ Training timed out after {MAX_WAIT}s", RED)
            return False
            
        except Exception as e:
            self.log(f"❌ Training error: {str(e)}", RED)
            return False
    
    def tab4_generate_era5_map(self) -> bool:
        """Tab 4: Generate ERA5 map"""
        self.log("\n" + "="*80, BLUE)
        self.log("TAB 4: Maps - Generate ERA5 Map", BLUE)
        self.log("="*80, BLUE)
        
        payload = {
            "session_id": self.session_id,
            "variable": "tp",  # Total precipitation
            "year": 2015,
            "month": 6
        }
        
        try:
            # Start map generation
            response = requests.post(
                f"{self.base_url}/api/maps/era5",
                json=payload
            )
            
            if response.status_code != 200:
                self.log(f"❌ ERA5 map generation start failed: {response.text}", RED)
                return False
            
            task_data = response.json()
            task_id = task_data['data']['task_id']
            self.log(f"⏳ ERA5 map generation started (task_id: {task_id})", YELLOW)
            
            # Poll for completion
            start_time = time.time()
            while time.time() - start_time < MAX_WAIT:
                time.sleep(POLL_INTERVAL)
                
                status_response = requests.get(
                    f"{self.base_url}/api/maps/status/{task_id}",
                    params={"session_id": self.session_id}
                )
                
                if status_response.status_code != 200:
                    self.log(f"❌ Status check failed", RED)
                    return False
                
                status_data = status_response.json()['data']
                status = status_data['status']
                
                self.log(f"   Status: {status}", YELLOW)
                
                if status == 'completed':
                    map_id = status_data.get('map_id')
                    self.log(f"✅ ERA5 map generated successfully", GREEN)
                    self.log(f"   Map ID: {map_id}", GREEN)
                    
                    # Try to download the map
                    download_response = requests.get(
                        f"{self.base_url}/api/maps/download/{map_id}",
                        params={"session_id": self.session_id}
                    )
                    
                    if download_response.status_code == 200:
                        result = download_response.json()
                        plot_data = result['data'].get('plot', '')
                        if plot_data.startswith('data:image/png;base64,'):
                            self.log(f"   Plot size: {len(plot_data)} chars", GREEN)
                        return True
                    else:
                        self.log(f"⚠️  Map download failed but generation succeeded", YELLOW)
                        return True
                
                elif status == 'failed':
                    error_msg = status_data.get('error', 'Unknown error')
                    self.log(f"❌ ERA5 map generation failed: {error_msg}", RED)
                    return False
            
            self.log(f"❌ ERA5 map generation timed out after {MAX_WAIT}s", RED)
            return False
            
        except Exception as e:
            self.log(f"❌ ERA5 map generation error: {str(e)}", RED)
            return False
    
    def tab5_evaluate_model(self) -> bool:
        """Tab 5: Evaluate model at specific location"""
        self.log("\n" + "="*80, BLUE)
        self.log("TAB 5: Statistical Analysis - Evaluate Model", BLUE)
        self.log("="*80, BLUE)
        
        payload = {
            "session_id": self.session_id,
            "latitude": -10.0,
            "longitude": -60.0,
            "start_year": 2005,
            "end_year": 2015
        }
        
        try:
            self.log(f"⏳ Evaluating model at location ({payload['latitude']}, {payload['longitude']})...", YELLOW)
            response = requests.post(
                f"{self.base_url}/api/analysis/evaluate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()['data']
                self.log(f"✅ Model evaluation completed", GREEN)
                self.log(f"   Location: ({data['location']['latitude']}, {data['location']['longitude']})", GREEN)
                self.log(f"   Time period: {data['time_period']['start_year']}-{data['time_period']['end_year']}", GREEN)
                
                stats = data['statistics']
                self.log(f"   R-score: {stats['r_score']:.4f}", GREEN)
                self.log(f"   P-value: {stats['p_value']:.6f}", GREEN)
                self.log(f"   Samples: {stats['n_samples']}", GREEN)
                
                plot_data = data.get('plot', '')
                if plot_data.startswith('data:image/png;base64,'):
                    self.log(f"   Plot generated: {len(plot_data)} chars", GREEN)
                
                return True
            else:
                self.log(f"❌ Model evaluation failed: {response.text}", RED)
                return False
        except Exception as e:
            self.log(f"❌ Model evaluation error: {str(e)}", RED)
            return False
    
    def tab5_feature_importance(self) -> bool:
        """Tab 5: Get feature importance"""
        self.log("\n" + "="*80, BLUE)
        self.log("TAB 5: Statistical Analysis - Feature Importance", BLUE)
        self.log("="*80, BLUE)
        
        try:
            response = requests.post(
                f"{self.base_url}/api/analysis/feature-importance",
                json={"session_id": self.session_id},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()['data']
                self.log(f"✅ Feature importance generated", GREEN)
                self.log(f"   Total features: {data['total_features']}", GREEN)
                
                # Show top features
                features = data['features'][:5]  # Top 5
                self.log(f"   Top 5 features:", GREEN)
                for feat in features:
                    self.log(f"      {feat['feature']}: {feat['percentage']:.2f}%", GREEN)
                
                plot_data = data.get('plot', '')
                if plot_data.startswith('data:image/png;base64,'):
                    self.log(f"   Pie chart generated: {len(plot_data)} chars", GREEN)
                
                return True
            else:
                self.log(f"❌ Feature importance failed: {response.text}", RED)
                return False
        except Exception as e:
            self.log(f"❌ Feature importance error: {str(e)}", RED)
            return False
    
    def cleanup_session(self) -> bool:
        """Delete test session"""
        self.log("\n" + "="*80, BLUE)
        self.log("CLEANUP: Delete Session", BLUE)
        self.log("="*80, BLUE)
        
        if not self.session_id:
            self.log("⚠️  No session to cleanup", YELLOW)
            return True
        
        try:
            response = requests.delete(
                f"{self.base_url}/api/session/{self.session_id}"
            )
            if response.status_code == 200:
                self.log(f"✅ Session deleted: {self.session_id}", GREEN)
                return True
            else:
                self.log(f"⚠️  Session cleanup failed: {response.status_code}", YELLOW)
                return False
        except Exception as e:
            self.log(f"⚠️  Session cleanup error: {str(e)}", YELLOW)
            return False
    
    def run_full_test(self):
        """Run complete end-to-end test"""
        self.log("\n" + "="*80, BLUE)
        self.log("🚀 STARTING END-TO-END TEST", BLUE)
        self.log("="*80, BLUE)
        
        start_time = time.time()
        tests = [
            ("Health Check", self.test_health),
            ("Session Creation", self.create_session),
            ("Tab 1: Validate Bounds", self.tab1_validate_bounds),
            ("Tab 1: Load Data", self.tab1_load_data),
            ("Tab 2: Prep Data", self.tab2_prep_data),
            ("Tab 2: Run RFE", self.tab2_run_rfe),
            ("Tab 3: Train Model", self.tab3_train_model),
            ("Tab 4: Generate ERA5 Map", self.tab4_generate_era5_map),
            ("Tab 5: Evaluate Model", self.tab5_evaluate_model),
            ("Tab 5: Feature Importance", self.tab5_feature_importance),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                success = test_func()
                results.append((test_name, success))
                
                if not success:
                    self.log(f"\n⚠️  Test '{test_name}' failed. Stopping test suite.", RED)
                    break
                    
            except Exception as e:
                self.log(f"\n❌ Unexpected error in '{test_name}': {str(e)}", RED)
                results.append((test_name, False))
                break
        
        # Cleanup
        self.cleanup_session()
        
        # Print summary
        elapsed = time.time() - start_time
        self.log("\n" + "="*80, BLUE)
        self.log("📊 TEST SUMMARY", BLUE)
        self.log("="*80, BLUE)
        
        passed = sum(1 for _, success in results if success)
        total = len(results)
        
        for test_name, success in results:
            status = f"{GREEN}✅ PASS{RESET}" if success else f"{RED}❌ FAIL{RESET}"
            self.log(f"  {status} - {test_name}")
        
        self.log(f"\n{BLUE}Results: {passed}/{total} tests passed{RESET}")
        self.log(f"{BLUE}Time elapsed: {elapsed:.2f} seconds{RESET}")
        
        if passed == total:
            self.log(f"\n{GREEN}🎉 ALL TESTS PASSED!{RESET}", GREEN)
            return 0
        else:
            self.log(f"\n{RED}❌ SOME TESTS FAILED{RESET}", RED)
            return 1


def main():
    """Main entry point"""
    tester = APITester(BASE_URL)
    exit_code = tester.run_full_test()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
