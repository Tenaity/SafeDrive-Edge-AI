FORCE_LIFT = True  # mock lifting for testing policy
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

signal_map = {
    'DB17.DBX7.0': {'db_number': 17, 'start_byte': 7, 'bit_index': 0, 'file_name': './voice/1.mp4', 'desc': 'Cẩu di chuyển qua hố cáp'},
    'DB17.DBX7.1': {'db_number': 17, 'start_byte': 7, 'bit_index': 1, 'file_name': './voice/2.mp4', 'desc': 'Khung chụp đang nâng'},
    'DB17.DBX7.2': {'db_number': 17, 'start_byte': 7, 'bit_index': 2, 'file_name': './voice/3.mp4', 'desc': 'Khung chụp đang hạ'},
    'DB17.DBX7.3': {'db_number': 17, 'start_byte': 7, 'bit_index': 3, 'file_name': './voice/4.mp4', 'desc': 'Đang nâng tải nặng'},
    'DB17.DBX7.4': {'db_number': 17, 'start_byte': 7, 'bit_index': 4, 'file_name': './voice/5.mp4', 'desc': 'Vượt quá giới hạn hành trình xe rùa phía trước'},
    'DB17.DBX7.5': {'db_number': 17, 'start_byte': 7, 'bit_index': 5, 'file_name': './voice/6.mp4', 'desc': 'Vượt quá giới hạn hành trình xe rùa phía sau'},
    'DB17.DBX7.6': {'db_number': 17, 'start_byte': 7, 'bit_index': 6, 'file_name': './voice/7.mp4', 'desc': 'Tải nâng vượt quá trọng tải cho phép'},
    'DB17.DBX7.7': {'db_number': 17, 'start_byte': 7, 'bit_index': 7, 'file_name': './voice/8.mp4', 'desc': 'Tốc độ di chuyển vượt mức cho phép'},
    'DB17.DBX8.0': {'db_number': 17, 'start_byte': 8, 'bit_index': 0, 'file_name': './voice/10.mp4', 'desc': 'Khung chụp đang khóa gù'},
    'DB17.DBX8.1': {'db_number': 17, 'start_byte': 8, 'bit_index': 1, 'file_name': './voice/11.mp4', 'desc': 'Cửa cabin đang mở'},
    'DB17.DBX8.2': {'db_number': 17, 'start_byte': 8, 'bit_index': 2, 'file_name': './voice/12.mp4', 'desc': 'Đang bị lệch tải'},
    'DB17.DBX8.3': {'db_number': 17, 'start_byte': 8, 'bit_index': 3, 'file_name': './voice/13.mp4', 'desc': 'Quá giới hạn chiều cao khung chụp'},
    'DB17.DBX8.4': {'db_number': 17, 'start_byte': 8, 'bit_index': 4, 'file_name': './voice/14.mp4', 'desc': 'Cửa buồng điện đang mở'},
    'DB17.DBX8.5': {'db_number': 17, 'start_byte': 8, 'bit_index': 5, 'file_name': './voice/15.mp4', 'desc': 'Cẩu chuẩn bị di chuyển, chú ý quan sát xung quanh'},
    'DB17.DBX8.6': {'db_number': 17, 'start_byte': 8, 'bit_index': 6, 'file_name': './voice/16.mp4', 'desc': 'Quá tốc độ di chuyển xe rùa'},
    'DB17.DBX8.7': {'db_number': 17, 'start_byte': 8, 'bit_index': 7, 'file_name': './voice/17.mp4', 'desc': 'Cẩu đang bị lệch tải'},
    'DB17.DBX9.0': {'db_number': 17, 'start_byte': 9, 'bit_index': 0, 'file_name': './voice/18.mp4', 'desc': 'Quá giới hạn di chuyển dài'},
    'DB17.DBX9.1': {'db_number': 17, 'start_byte': 9, 'bit_index': 1, 'file_name': './voice/19.mp4', 'desc': 'Quá nhiệt buồng điện'},
    'DB17.DBX9.2': {'db_number': 17, 'start_byte': 9, 'bit_index': 2, 'file_name': './voice/20.mp4', 'desc': 'Cảnh báo va chạm container khi di chuyển xe rùa'},
    'DB17.DBX9.3': {'db_number': 17, 'start_byte': 9, 'bit_index': 3, 'file_name': './voice/21.mp4', 'desc': 'Tốc độ gió mạnh'},
}

class CranePipeline:
    def __init__(self):
        self.logger = logging.getLogger("CranePipeline")
        
        # Load config
        self.ip = os.getenv("PLC_IP", "127.0.0.1")
        self.rack = int(os.getenv("PLC_RACK", "0"))
        self.slot = int(os.getenv("PLC_SLOT", "1"))
        
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
            return signal_map['DB17.DBX7.0']

        if not self.check_connection():
            return {"error": "PLC Disconnected"}

        try:
            active_signals = {}
            byte_cache = {}

            for signal_name, cfg in signal_map.items():
                db_number = cfg["db_number"]
                start_byte = cfg["start_byte"]
                bit_index = cfg["bit_index"]
                cache_key = (db_number, start_byte)

                if cache_key not in byte_cache:
                    # Read exactly 1 byte for the requested DB and byte index.
                    byte_cache[cache_key] = self.client.db_read(db_number, start_byte, 1)

                data = byte_cache[cache_key]
                bit_value = get_bool(data, 0, bit_index)

                if bit_value:
                    active_signals[signal_name] = signal_map[signal_name]

            return active_signals
        except Exception as e:
            self.logger.error(f"PLC Read Error: {e}")
            self.connected = False # Mark as disconnected to trigger reconnect
            return {"error": str(e)}

    def close(self):
        if self.client:
            self.client.disconnect()
            self.client.destroy()
