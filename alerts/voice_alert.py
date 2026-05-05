import os
import time
import logging
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
        self._ensure_audio()

    def _ensure_audio(self):
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init()
            self.ai_channel = pygame.mixer.Channel(0)   # Channel 0 dành riêng cho AI
        except Exception as e:
            self.logger.error(f"Failed to initialize pygame mixer: {e}")
            self.ai_channel = None

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

        if self.ai_channel is None:
            return

        file_path = self._voice_path(file_name)
        if not os.path.exists(file_path):
            self.logger.warning(f"AI audio file not found: {file_path}")
            return

        try:
            snd = pygame.mixer.Sound(file_path)
            self.ai_channel.play(snd)

            while self.ai_channel.get_busy():
                time.sleep(0.05)

        except Exception as e:
            self.logger.error(f"AI audio error: {e}")

    def speak(self, alert_level, driver, vision):
        alert_name = alert_level.name if hasattr(alert_level, "name") else str(alert_level)

        if alert_name in ("NONE", "LOW"):
            return

        driver = driver or {}
        vision = vision or {}

        if alert_name == "HIGH":
            if driver.get("drowsy") and self._can_speak("drowsy"):
                self._play_file("21.mp3")
                return

            if driver.get("yawning") and self._can_speak("yawning"):
                self._play_file("21.mp3")
                return

            if driver.get("distracted") and self._can_speak("distracted"):
                self._play_file("21.mp3")
                return

            # Fallback: HIGH triggered but specific cooldowns all active
            if self._can_speak("high_fallback"):
                self._play_file("21.mp3")
                return

        if alert_name == "PHONE":
            # policy already confirmed phone_using=True, no need to re-check vision["phone"]
            if self._can_speak("phone"):
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