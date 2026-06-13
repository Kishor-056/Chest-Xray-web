import requests
import sys

# Test the backend connection
backend_url = "https://monocyclic-shara-unrotative.ngrok-free.dev"

print(f"Testing connection to: {backend_url}")
print("=" * 60)

try:
    # Test health endpoint
    print("\n1. Testing health endpoint (/)...")
    response = requests.get(f"{backend_url}/", headers={'ngrok-skip-browser-warning': 'true'}, timeout=10)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   Response: {response.json()}")
    else:
        print(f"   Error: {response.text[:200]}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

try:
    # Test models endpoint
    print("\n2. Testing models endpoint (/models)...")
    response = requests.get(f"{backend_url}/models", headers={'ngrok-skip-browser-warning': 'true'}, timeout=10)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Models loaded: {data.get('models_loaded', 'N/A')}")
    else:
        print(f"   Error: {response.text[:200]}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

try:
    # Test validate endpoint
    print("\n3. Testing validate endpoint (/validate/xray)...")
    response = requests.post(f"{backend_url}/validate/xray", headers={'ngrok-skip-browser-warning': 'true'}, timeout=10)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.text[:200]}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

print("\n" + "=" * 60)
print("Connection test complete!")
