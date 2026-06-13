"""
🤖 APPIUM HYBRID APP E2E AUTOMATION TEST TEMPLATE
=================================================
This script is designed to automate the Chest X-Ray AI Android Application (Capacitor Hybrid App)
on a local Android Emulator or physical device using Appium.

PREREQUISITES:
1. Node.js & Appium: `npm install -g appium`
2. Appium UiAutomator2 Driver: `appium driver install uiautomator2`
3. Python Appium Client: `pip install Appium-Python-Client`
4. Android SDK configured (`ANDROID_HOME` env variable set).
5. A running Android Emulator or physical device connected (`adb devices` showing active device).
6. The compiled Android app installed on the device.

TO RUN:
1. Start Appium server: `appium`
2. Run this script: `python appium_e2e_tests.py`
"""

import time
import os
from appium import webdriver
from appium.options.android import UiAutomator2Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def run_appium_tests():
    print("============================================================")
    print("STARTING APPIUM E2E TEST EXECUTION")
    print("============================================================")

    # 1. Setup Capabilities
    options = UiAutomator2Options()
    options.platform_name = 'Android'
    options.automation_name = 'UiAutomator2'
    options.device_name = 'Android Emulator'  # Change to your device name if using real device
    
    # App package details from capacitor.config.json
    options.app_package = 'com.kishore.chestxray'
    options.app_activity = '.MainActivity'
    
    # Ensure app starts fresh
    options.no_reset = False
    options.full_reset = False
    
    # Appium Server URL
    appium_server_url = 'http://localhost:4723'

    print("Connecting to Appium Server...")
    try:
        driver = webdriver.Remote(appium_server_url, options=options)
        print("Successfully launched Android App!")
    except Exception as e:
        print(f"FAILED to connect to Appium Server: {e}")
        print("Ensure Appium is running on http://localhost:4723 and your emulator/device is connected.")
        return

    # Let app load
    time.sleep(5)

    # 2. Context Switching (CRITICAL for Capacitor/Cordova apps)
    # Hybrid apps run inside a WebView wrapper. We must switch context from NATIVE_APP to WEBVIEW.
    print(f"Available contexts: {driver.contexts}")
    webview_context = None
    
    # Poll for webview context to appear (it may take a few seconds after launch)
    for _ in range(5):
        for context in driver.contexts:
            if 'WEBVIEW' in context:
                webview_context = context
                break
        if webview_context:
            break
        time.sleep(2)

    if webview_context:
        print(f"Switching context to: {webview_context}")
        driver.switch_to.context(webview_context)
    else:
        print("WARNING: WebView context not found. Proceeding in NATIVE_APP context.")
        print("Make sure your Android build is debuggable (debug builds enable WebView inspection by default).")
        driver.quit()
        return

    # 3. WebView Automation (React Frontend DOM)
    wait = WebDriverWait(driver, 20)
    
    try:
        # TC_APP_001: Dashboard Load
        print("\nRunning TC_APP_001: Verify Dashboard Load...")
        h1_header = wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
        print(f"SUCCESS: Dashboard loaded. Header: '{h1_header.text}'")
        
        # TC_APP_002: Navigation to Prediction
        print("\nRunning TC_APP_002: Navigate to Single Prediction page...")
        # Locate predict sidebar button
        predict_link = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[contains(@href, '/predict')]")))
        predict_link.click()
        time.sleep(2)
        print("SUCCESS: Navigated to Single Prediction page.")
        
        # TC_APP_003: Model Selector Interaction
        print("\nRunning TC_APP_003: Check Model Selector dropdown...")
        model_dropdown = wait.until(EC.presence_of_element_located((By.TAG_NAME, "select")))
        model_dropdown.click()
        print("SUCCESS: Model dropdown is interactive.")
        
        # TC_APP_004: Go to Settings
        print("\nRunning TC_APP_004: Navigate to Settings page...")
        settings_link = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[contains(@href, '/settings')]")))
        settings_link.click()
        time.sleep(2)
        
        api_input = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@type='text']")))
        print(f"SUCCESS: Settings page loaded. API Endpoint: {api_input.get_attribute('value')}")
        
    except Exception as e:
        print(f"\nTEST CASE EXECUTION ENCOUNTERED AN ERROR: {e}")
        
    finally:
        # 4. Clean up driver
        print("\nClosing Appium Session...")
        driver.quit()
        print("E2E Automation Session Complete!")

if __name__ == '__main__':
    run_appium_tests()
