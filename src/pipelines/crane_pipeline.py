import logging
import os
import time
from dotenv import load_dotenv

# Try importing snap7, handle failure gracefully (e.g. if not installed yet)
try:
    import snap7
    from snap7.util import get_bool
    SNAP7_AVAILABLE = True
except ImportError:
    SNAP7_AVAILABLE = False

load_dotenv()

class CranePipeline:
    def __init__(self):
        self.logger = logging.getLogger("CranePipeline")
        
        # Load config
        self.ip = os.getenv("PLC_IP", "127.0.0.1")
        self.rack = int(os.getenv("PLC_RACK", "0"))
        self.slot = int(os.getenv("PLC_SLOT", "1"))
        self.db_number = int(os.getenv("PLC_DB_NUMBER", "1"))
        self.start_byte = int(os.getenv("PLC_START_BYTE", "0"))
        self.bit_index = int(os.getenv("PLC_BIT_INDEX", "0"))
        
        self.mock_mode = os.getenv("MOCK_PLC", "false").lower() == "true"
        
        self.client = None
        self.connected = False
        self.last_connect_attempt = 0
        self.reconnect_interval = 5.0  # seconds

        if not SNAP7_AVAILABLE:
            self.logger.warning("python-snap7 not installed. Forcing MOCK_PLC=true")
            self.mock_mode = True

        if not self.mock_mode:
            self.connect()

    def connect(self):
        if self.mock_mode:
            return

        try:
            self.client = snap7.client.Client()
            self.client.connect(self.ip, self.rack, self.slot)
            self.connected = self.client.get_connected()
            if self.connected:
                self.logger.info(f"Connected to PLC at {self.ip}")
            else:
                self.logger.error(f"Failed to connect to PLC at {self.ip}")
        except Exception as e:
            self.logger.error(f"PLC Connection Error: {e}")
            self.connected = False

    def check_connection(self):
        """Ensure connected before reading."""
        if self.mock_mode:
            return True
            
        if self.client and self.client.get_connected():
            return True
            
        # Try to reconnect if enough time passed
        now = time.time()
        if now - self.last_connect_attempt > self.reconnect_interval:
            self.last_connect_attempt = now
            self.logger.info("Attempting to reconnect to PLC...")
            self.connect()
            
        return self.connected

    def run(self, *args, **kwargs):
        """
        Returns a dict: {"is_lifting": bool}
        """
        if self.mock_mode:
            # In mock mode, check for a local file flag, or toggle periodically?
            # For simplicity, let's look for a file named "MOCK_LIFTING" in current dir.
            is_lifting = os.path.exists("MOCK_LIFTING")
            return {"is_lifting": is_lifting}

        if not self.check_connection():
            # If not connected, default to False (safe state? or True to be safe?)
            # Assuming "Free" is safe state to NOT annoy driver.
            return {"is_lifting": False, "error": "PLC Disconnected"}

        try:
            # Read 1 byte from DB
            # snap7.client.Client.db_read(db_number, start, size)
            data = self.client.db_read(self.db_number, self.start_byte, 1)
            is_lifting = get_bool(data, 0, self.bit_index)
            return {"is_lifting": is_lifting}
        except Exception as e:
            self.logger.error(f"PLC Read Error: {e}")
            self.connected = False # Mark as disconnected to trigger reconnect
            return {"is_lifting": False, "error": str(e)}

    def close(self):
        if self.client:
            self.client.disconnect()
            self.client.destroy()
