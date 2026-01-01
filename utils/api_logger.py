import requests
import threading
import json
import logging
import os
from datetime import datetime

class APILogger:
    def __init__(self):
        self.endpoint = os.getenv("API_ENDPOINT", "")
        self.logger = logging.getLogger("APILogger")
        self.enabled = bool(self.endpoint)

        if not self.enabled:
            self.logger.warning("API_ENDPOINT not set. Logging to API disabled.")

    def _send_payload(self, payload):
        if not self.enabled:
            return

        try:
            headers = {'Content-Type': 'application/json'}
            response = requests.post(self.endpoint, json=payload, headers=headers, timeout=5)
            if response.status_code == 200:
                self.logger.info("Alert sent to API successfully.")
            else:
                self.logger.warning(f"Failed to send alert to API. Status: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Error sending alert to API: {e}")

    def log_alert(self, alert_level, crane_status, driver_state, image_path=None):
        """
        Logs an alert to the API asynchronously.
        """
        if not self.enabled:
            return

        payload = {
            "timestamp": datetime.now().isoformat(),
            "alert_level": alert_level,
            "crane_status": crane_status,
            "driver_state": {k: str(v) for k, v in driver_state.items()}, # Ensure serializable
            "image_path": image_path
            # Note: For actual image upload, we'd need multipart/form-data.
            # Here we send the path, assuming the dashboard reads from shared storage
            # or we implement image upload later.
        }

        # Run in a separate thread to avoid blocking the video processing loop
        t = threading.Thread(target=self._send_payload, args=(payload,))
        t.start()
