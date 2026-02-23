#!/usr/bin/env python3
"""
DRDO Air Defence API - Endpoint Tester
Run this to test all endpoints and identify failures
"""

import requests
import json
import sys
from time import sleep

BASE_URL = "http://localhost:8000"

def test_endpoint(method, path, data=None, description=""):
    """Test a single endpoint and report result"""
    url = f"{BASE_URL}{path}"
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        if response.status_code == 200:
            print(f"✅ {method:4} {path:50} OK")
            return True
        else:
            print(f"❌ {method:4} {path:50} ERROR {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ {method:4} {path:50} CONNECTION FAILED")
        print(f"   Is the server running at {BASE_URL}?")
        return False
    except Exception as e:
        print(f"❌ {method:4} {path:50} EXCEPTION: {e}")
        return False

def main():
    print("=" * 80)
    print("DRDO AIR DEFENCE API - ENDPOINT TESTING")
    print("=" * 80)
    print(f"Testing against: {BASE_URL}")
    print(f"Make sure server is running with: uvicorn api.main:app --reload")
    print("=" * 80)
    
    sleep(0.5)
    
    tests = [
        # (method, path, data, description)
        ("GET",  "/",                                   None,        "Health Check"),
        ("GET",  "/api/stats/overview",                 None,        "Dashboard Stats"),
        ("GET",  "/api/models/info",                    None,        "Model Info"),
        ("GET",  "/api/countries",                      None,        "All Countries"),
        ("GET",  "/api/countries/names?q=in",           None,        "Country Names Autocomplete"),
        ("GET",  "/api/countries?risk_zone=Red",        None,        "Red Zone Countries"),
        ("GET",  "/api/countries/India",                None,        "India Profile"),
        ("GET",  "/api/systems",                        None,        "All Systems"),
        ("GET",  "/api/systems?system_name=Rafale",     None,        "Systems Search by Name"),
        ("GET",  "/api/systems/names?q=ra",             None,        "System Names Autocomplete"),
        ("GET",  "/api/systems/by-name/Rafale%20(IAF)", None,        "System by Name"),
        ("GET",  "/api/compare?country1=India&country2=China", None, "Country Comparison"),
        ("GET",  "/api/map/zones",                      None,        "Map Zones"),
        ("GET",  "/api/country/China/insights",         None,        "China Insights"),
        ("GET",  "/api/country/Pakistan/insights",      None,        "Pakistan Insights"),
        ("GET",  "/api/predict/war?attacker_country=China&defender_country=India",
                 None,
                 "War Prediction (China vs India)"),
        ("GET",  "/api/predict/war?attacker_country=Pakistan&defender_country=India",
                 None,
                 "War Prediction (Pakistan vs India)"),
        ("POST", "/api/predict/classify-system",       
                 {"system_name": "Rafale (IAF)"}, 
                 "Classify System by Name"),
        ("POST", "/api/predict/classify-system",       
                 {"system_name": "HQ-9B"},
                 "Classify Another System by Name"),
    ]
    
    passed = 0
    failed = 0
    
    for i, (method, path, data, desc) in enumerate(tests, 1):
        print(f"\nTest {i}/{len(tests)}: {desc}")
        if test_endpoint(method, path, data, desc):
            passed += 1
        else:
            failed += 1
        sleep(0.2)  # Small delay between tests
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED - API is healthy!")
        print("You can share the API with your frontend team.")
        return 0
    else:
        print(f"\n⚠️  {failed} endpoint(s) failing")
        print("\nTroubleshooting steps:")
        print("1. Check server logs in the terminal where uvicorn is running")
        print("2. Look for Python tracebacks in the server output")
        print("3. Verify data files exist in data/ directory")
        print("4. Verify model files exist in models/ directory")
        print("5. See TROUBLESHOOTING.md for common fixes")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user")
        sys.exit(130)
