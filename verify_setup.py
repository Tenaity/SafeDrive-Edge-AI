import sys
import os
import time

print("Checking environment...")
try:
    import cv2
    import snap7
    import ultralytics
    import mediapipe
    import requests
    from dotenv import load_dotenv
    print("Imports passed.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

print("Checking APILogger...")
try:
    from utils.api_logger import APILogger
    # Temporarily disable endpoint to test init only, or mock it?
    # We just want to ensure it doesn't crash on init.
    logger = APILogger()
    print(f"APILogger initialized. Enabled: {logger.enabled}")
    
    # Send a test log (will likely fail siliently if no server, but verify no crash)
    logger.log_alert("TEST_ALERT", {"is_lifting": True}, {"drowsy": False})
    print("Test alert sent (check logs if endpoint active).")

except Exception as e:
    print(f"APILogger failed: {e}")
    sys.exit(1)

print("Verification Successful.")
