import logging
import os
import time
import subprocess
import platform
from typing import Dict, TypedDict

from dotenv import load_dotenv
import pygame

try:
    import snap7
    from snap7.util import get_bool
    SNAP7_AVAILABLE = True
except ImportError:
    SNAP7_AVAILABLE = False

load_dotenv()


class SignalConfig(TypedDict):
    db_number: int
    start_byte: int
    bit_index: int
    file_name: str
    desc: str


signal_map: Dict[str, SignalConfig] = {
    "DB17.DBX7.0": {"db_number": 17, "start_byte": 9, "bit_index": 0, "file_name": "./voices/1.mp3", "desc": "Cẩu di chuyển qua hố cáp"},
    "DB17.DBX7.1": {"db_number": 17, "start_byte": 14, "bit_index": 1, "file_name": "./voices/2.mp3", "desc": "Khung chụp đang nâng"},
    "DB17.DBX7.2": {"db_number": 17, "start_byte": 14, "bit_index": 2, "file_name": "./voices/3.mp3", "desc": "Khung chụp đang hạ"},
    "DB17.DBX7.3": {"db_number": 17, "start_byte": 14, "bit_index": 3, "file_name": "./voices/4.mp3", "desc": "Đang nâng tải nặng"},
    "DB17.DBX7.4": {"db_number": 17, "start_byte": 14, "bit_index": 4, "file_name": "./voices/5.mp3", "desc": "Vượt quá giới hạn hành trình xe rùa phía trước"},
    "DB17.DBX7.5": {"db_number": 17, "start_byte": 14, "bit_index": 5, "file_name": "./voices/6.mp3", "desc": "Vượt quá giới hạn hành trình xe rùa phía sau"},
    "DB17.DBX7.6": {"db_number": 17, "start_byte": 14, "bit_index": 6, "file_name": "./voices/7.mp3", "desc": "Tải nâng vượt quá trọng tải cho phép"},
    "DB17.DBX7.7": {"db_number": 17, "start_byte": 14, "bit_index": 7, "file_name": "./voices/8.mp3", "desc": "Tốc độ di chuyển vượt mức cho phép"},
    "DB17.DBX8.0": {"db_number": 17, "start_byte": 15, "bit_index": 0, "file_name": "./voices/10.mp3", "desc": "Khung chụp đang khóa gù"},
    "DB17.DBX8.1": {"db_number": 17, "start_byte": 15, "bit_index": 1, "file_name": "./voices/11.mp3", "desc": "Cửa cabin đang mở"},
    "DB17.DBX8.2": {"db_number": 17, "start_byte": 15, "bit_index": 2, "file_name": "./voices/12.mp3", "desc": "Đang bị lệch tải"},
    "DB17.DBX8.3": {"db_number": 17, "start_byte": 15, "bit_index": 3, "file_name": "./voices/13.mp3", "desc": "Quá giới hạn chiều cao khung chụp"},
    "DB17.DBX8.4": {"db_number": 17, "start_byte": 15, "bit_index": 4, "file_name": "./voices/14.mp3", "desc": "Cửa buồng điện đang mở"},
    "DB17.DBX8.5": {"db_number": 17, "start_byte": 15, "bit_index": 5, "file_name": "./voices/15.mp3", "desc": "Cẩu chuẩn bị di chuyển, chú ý quan sát xung quanh"},
    "DB17.DBX8.6": {"db_number": 17, "start_byte": 15, "bit_index": 6, "file_name": "./voices/16.mp3", "desc": "Quá tốc độ di chuyển xe rùa"},
    "DB17.DBX8.7": {"db_number": 17, "start_byte": 15, "bit_index": 7, "file_name": "./voices/17.mp3", "desc": "Cẩu đang bị lệch tải"},
    "DB17.DBX9.0": {"db_number": 17, "start_byte": 16, "bit_index": 0, "file_name": "./voices/18.mp3", "desc": "Quá giới hạn di chuyển dài"},
    "DB17.DBX9.1": {"db_number": 17, "start_byte": 16, "bit_index": 1, "file_name": "./voices/19.mp3", "desc": "Quá nhiệt buồng điện"},
    "DB17.DBX9.2": {"db_number": 17, "start_byte": 16, "bit_index": 2, "file_name": "./voices/20.mp3", "desc": "Cảnh báo va chạm container khi di chuyển xe rùa"},
    "DB17.DBX9.3": {"db_number": 17, "start_byte": 16, "bit_index": 3, "file_name": "./voices/23.mp3", "desc": "Tốc độ gió mạnh"},
}


class CranePipeline:
    def __init__(self):
        self.logger = logging.getLogger("CranePipeline")

        self.ip = os.getenv("PLC_IP", "192.168.150.103")
        self.rack = int(os.getenv("PLC_RACK", "0"))
        self.slot = int(os.getenv("PLC_SLOT", "1"))
        self.mock_mode = os.getenv("MOCK_PLC", "false").lower() == "true"

        self.client = None
        self.connected = False
        self.last_connect_attempt = 0.0
        self.reconnect_interval = 5.0
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.log_path = os.path.join(self.base_dir, "logs", "plc_voice.log")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        self._ensure_audio()

        if not SNAP7_AVAILABLE:
            self.logger.warning("python-snap7 not installed. Forcing MOCK_PLC=true")
            self.mock_mode = True

        if not self.mock_mode:
            self.connect()

    def _ensure_audio(self):
        if not hasattr(pygame, "mixer"):
            self._log_plc_event("pygame mixer module unavailable")
            self.plc_channel = None
            return

        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.pre_init(44100, -16, 2, 2048)
                pygame.mixer.init()
            self.plc_channel = pygame.mixer.find_channel(True)
            if self.plc_channel is not None:
                self.plc_channel.set_volume(1.0)
            self._log_plc_event(f"PLC audio channel set to {self.plc_channel}")
        except Exception as e:
            self.logger.error(f"Failed to initialize pygame mixer: {e}")
            self._log_plc_event(f"Failed to initialize pygame mixer: {e}")
            self.plc_channel = None

    def _play_with_system_audio(self, file_path: str):
        if platform.system().lower() != "windows":
            self._log_plc_event(f"System audio fallback unsupported on {platform.system()}")
            return

        try:
            escaped_path = file_path.replace("'", "''")
            ps_cmd = (
                "$player = New-Object -ComObject WMPlayer.OCX.7; "
                f"$player.URL = '{escaped_path}'; "
                "$player.controls.play(); "
                "while ($player.playState -eq 3) { Start-Sleep -Milliseconds 100 }"
            )
            subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True, timeout=60)
            self._log_plc_event(f"Played via PowerShell fallback: {file_path}")
            return
        except Exception as e:
            self._log_plc_event(f"PowerShell audio fallback failed: {e}")

        try:
            os.startfile(file_path)
            self._log_plc_event(f"Opened file with default application: {file_path}")
        except Exception as e:
            self._log_plc_event(f"os.startfile fallback failed: {e}")

    def _cleanup_client(self):
        if self.client:
            try:
                self.client.disconnect()
            except Exception:
                pass
            try:
                self.client.destroy()
            except Exception:
                pass
        self.client = None
        self.connected = False

    def _log_plc_event(self, message: str):
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
        except Exception:
            pass

    def connect(self):
        if self.mock_mode:
            return

        try:
            self._cleanup_client()

            self.client = snap7.client.Client()
            self.logger.info(f"Connecting PLC: ip={self.ip}, rack={self.rack}, slot={self.slot}")
            print(f"[PLC DEBUG] ip={self.ip}, rack={self.rack}, slot={self.slot}, mock={self.mock_mode}")

            self.client.connect(self.ip, self.rack, self.slot)
            self.connected = self.client.get_connected()
            print(f"[PLC DEBUG] connected={self.connected}")

            if self.connected:
                self.logger.info(f"Connected to PLC at {self.ip}")
            else:
                self.logger.error(f"Failed to connect to PLC at {self.ip}")
        except Exception as e:
            print(f"[PLC DEBUG] connect error={e}")
            self.logger.error(f"PLC Connection Error: {e}")
            self.connected = False

    def check_connection(self) -> bool:
        if self.mock_mode:
            return True

        try:
            if self.client and self.client.get_connected():
                self.connected = True
                return True
        except Exception:
            self.connected = False

        now = time.time()
        if now - self.last_connect_attempt > self.reconnect_interval:
            self.last_connect_attempt = now
            self.logger.info("Attempting to reconnect to PLC...")
            self.connect()

        return self.connected

    def run(self, *args, **kwargs):
        if self.mock_mode:
            return {
                "signals": {
                    "DB17.DBX7.0": signal_map["DB17.DBX7.0"]
                }
            }

        if not self.check_connection():
            return {
                "signals": {},
                "error": "PLC Disconnected"
            }

        try:
            active_signals = {}
            byte_cache = {}

            for signal_name, cfg in signal_map.items():
                db_number = cfg["db_number"]
                start_byte = cfg["start_byte"]
                bit_index = cfg["bit_index"]
                cache_key = (db_number, start_byte)

                if cache_key not in byte_cache:
                    byte_cache[cache_key] = self.client.db_read(db_number, start_byte, 1)  # type: ignore

                data = byte_cache[cache_key]
                bit_value = get_bool(data, 0, bit_index)

                if bit_value:
                    active_signals[signal_name] = cfg

            return {
                "signals": active_signals
            }

        except Exception as e:
            self.logger.error(f"PLC Read Error: {e}")
            self.connected = False
            return {
                "signals": {},
                "error": str(e)
            }

    def flatten_signals(self, signals: Dict[str, SignalConfig]):
        items = []

        for key, value in signals.items():
            if isinstance(value, str):
                file_name = value
                desc = key
            elif isinstance(value, dict):
                file_name = value.get("file_name", "")
                desc = value.get("desc", key)
            else:
                self.logger.warning(f"Skip invalid signal format: key={key}, value={value}")
                continue

            if not file_name:
                self.logger.warning(f"Missing file_name for signal: key={key}, value={value}")
                continue

            items.append({
                "file_name": file_name,
                "desc": desc
            })

        return items

    def play_signal(self, signal):
        self._ensure_audio()

        file_name = signal.get("file_name", "")
        desc = signal.get("desc", "")

        if not file_name:
            return

        file_path = file_name
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.base_dir, file_path)

        if not os.path.exists(file_path):
            self.logger.error(f"PLC audio file not found: {file_path}")
            self._log_plc_event(f"PLC audio file not found: {file_path}")
            return

        try:
            self.logger.info(f"Playing alert: {desc} from {file_path}")
            self._log_plc_event(f"Playing alert: {desc} from {file_path}")

            if self.plc_channel is None:
                self._log_plc_event("PLC audio channel unavailable, using system fallback")
                self._play_with_system_audio(file_path)
                return

            snd = pygame.mixer.Sound(file_path)
            snd.set_volume(1.0)
            self.plc_channel.play(snd)

            while self.plc_channel.get_busy():
                time.sleep(0.05)

        except Exception as e:
            self.logger.error(f"Failed to play audio {file_name}: {e}")
            self._log_plc_event(f"Failed to play audio {file_name}: {e}")
            self._play_with_system_audio(file_path)

    def voice_alert(self, signals: Dict[str, SignalConfig]):
        items = self.flatten_signals(signals)
        for sig in items:
            self.play_signal(sig)

    def close(self):
        self._cleanup_client()