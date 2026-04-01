import os
import logging
import threading
from datetime import datetime

import requests


class APILogger:
    def __init__(self):
        self.endpoint = os.getenv("API_ENDPOINT", "").strip()
        self.logger = logging.getLogger("APILogger")
        self.enabled = bool(self.endpoint)

        if not self.enabled:
            self.logger.warning("API_ENDPOINT not set. Logging to API disabled.")

    @staticmethod
    def _safe_dict(data):
        if not isinstance(data, dict):
            return {}

        safe = {}
        for k, v in data.items():
            try:
                if isinstance(v, (str, int, float, bool)) or v is None:
                    safe[k] = v
                elif isinstance(v, dict):
                    safe[k] = {kk: str(vv) for kk, vv in v.items()}
                elif isinstance(v, (list, tuple)):
                    safe[k] = [str(x) for x in v]
                else:
                    safe[k] = str(v)
            except Exception:
                safe[k] = str(v)
        return safe

    def _send_payload(self, payload):
        if not self.enabled:
            return

        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5
            )

            if response.status_code == 200:
                self.logger.info("[API LOGGER] Alert sent successfully.")
            else:
                self.logger.warning(f"[API LOGGER] Failed to send alert. Status: {response.status_code}")
        except Exception as e:
            self.logger.error(f"[API LOGGER] Error sending alert: {e}")

    def log_alert(self, alert_level, crane_status, driver_state, image_path=None):
        """
        Logs an alert to the API asynchronously.
        """
        if not self.enabled:
            return

        if hasattr(alert_level, "name"):
            alert_level_value = alert_level.name
        else:
            alert_level_value = str(alert_level)

        payload = {
            "timestamp": datetime.now().isoformat(),
            "alert_level": alert_level_value,
            "crane_status": self._safe_dict(crane_status),
            "driver_state": self._safe_dict(driver_state),
            "image_path": image_path,
        }

        t = threading.Thread(target=self._send_payload, args=(payload,), daemon=True)
        t.start()