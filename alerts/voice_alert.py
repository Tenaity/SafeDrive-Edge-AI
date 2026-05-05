import os
import time
import logging
import subprocess
import platform
import pygame


class VoiceAlert:
    """
    VoiceAlert:
    - Play AI alert audio on dedicated pygame channel
    - Cooldown per alert type
    - Safe path handling
    - Safe cleanup / stop / reset
    """

    def __init__(self, cooldown_sec=8):
        self.logger = logging.getLogger("VoiceAlert")
        self.cooldown_sec = cooldown_sec
        self.last_spoken_at = {}

        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.voice_dir = os.path.join(self.base_dir, "voices")

        self.ai_channel = None
        self.log_path = os.path.join(self.base_dir, "logs", "voice.log")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._ensure_audio()

    def _ensure_audio(self):
        if not hasattr(pygame, "mixer"):
            print("[VOICE] pygame mixer module unavailable")
            self.ai_channel = None
            return

        try:
            if pygame.mixer.get_init() is None:
                print("[VOICE] Pre-initializing pygame mixer")
                pygame.mixer.pre_init(44100, -16, 2, 2048)
                pygame.mixer.init()
                print("[VOICE] Pygame mixer initialized")
            self.ai_channel = pygame.mixer.find_channel(True)
            if self.ai_channel is not None:
                self.ai_channel.set_volume(1.0)
            print(f"[VOICE] AI channel set to {self.ai_channel}")
        except Exception as e:
            print(f"[VOICE] Failed to initialize pygame mixer: {e}")
            self.ai_channel = None

    def _play_with_system_audio(self, file_path: str):
        if platform.system().lower() != "windows":
            print(f"[VOICE] System audio fallback not supported on {platform.system()}")
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
            print(f"[VOICE] Played via PowerShell fallback: {file_path}")
            return
        except Exception as e:
            print(f"[VOICE] PowerShell audio fallback failed: {e}")

        try:
            os.startfile(file_path)
            print(f"[VOICE] Opened file with default application: {file_path}")
        except Exception as e:
            print(f"[VOICE] os.startfile fallback failed: {e}")

    def _log_voice_event(self, message: str):
        print(message)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
        except Exception:
            pass

    def _can_speak(self, key: str) -> bool:
        now = time.time()
        last = self.last_spoken_at.get(key, 0.0)

        cooldown = self.cooldown_sec
        if key == "medium":
            cooldown = 30.0

        if now - last < cooldown:
            return False

        self.last_spoken_at[key] = now
        return True

    def _voice_path(self, file_name: str):
        return os.path.join(self.voice_dir, file_name)

    def _play_file(self, file_name: str):
        self._ensure_audio()

        file_path = self._voice_path(file_name)
        if not os.path.exists(file_path):
            print(f"[VOICE] File not found: {file_path}")
            return

        if self.ai_channel is None:
            print(f"[VOICE] ai_channel is None, falling back to system audio")
            self._play_with_system_audio(file_path)
            return

        try:
            self._log_voice_event(f"[VOICE] Loading sound: {file_path}")
            snd = pygame.mixer.Sound(file_path)
            snd.set_volume(1.0)
            self._log_voice_event(f"[VOICE] Playing on channel {self.ai_channel}")
            self.ai_channel.play(snd)

            while self.ai_channel.get_busy():
                time.sleep(0.05)
            self._log_voice_event(f"[VOICE] Finished playing {file_name}")

        except Exception as e:
            print(f"[VOICE] Error playing {file_name}: {e}")
            self._play_with_system_audio(file_path)

    def speak(self, alert_level, driver, vision):
        alert_name = alert_level.name if hasattr(alert_level, "name") else str(alert_level)

        if alert_name in ("NONE", "LOW"):
            return

        driver = driver or {}
        vision = vision or {}

        if alert_name == "HIGH":
            if driver.get("drowsy") and self._can_speak("drowsy"):
                print(f"[VOICE] Playing 21.mp3 for drowsy")
                self._play_file("21.mp3")
                return

            if driver.get("yawning") and self._can_speak("yawning"):
                print(f"[VOICE] Playing 21.mp3 for yawning")
                self._play_file("21.mp3")
                return

            if driver.get("distracted") and self._can_speak("distracted"):
                print(f"[VOICE] Playing 21.mp3 for distracted")
                self._play_file("21.mp3")
                return

            # Fallback: HIGH triggered but specific cooldowns all active
            if self._can_speak("high_fallback"):
                print(f"[VOICE] Playing 21.mp3 for high_fallback")
                self._play_file("21.mp3")
                return

        if alert_name == "PHONE":
            # policy already confirmed phone_using=True, no need to re-check vision["phone"]
            if self._can_speak("phone"):
                print(f"[VOICE] Playing 22.mp3 for phone")
                self._play_file("22.mp3")
                return

    def stop(self):
        try:
            if self.ai_channel is not None:
                self.ai_channel.stop()
        except Exception:
            pass

    def reset(self):
        self.last_spoken_at.clear()

    def close(self):
        self.stop()