import time
import pyttsx3

class VoiceAlert:
    def __init__(self, cooldown_sec=5):
        self.engine = pyttsx3.init()

        # 🔴 CHỌN GIỌNG VIỆT
        for voice in self.engine.getProperty("voices"):
            if "vi" in voice.id.lower() or "vietnam" in voice.name.lower():
                self.engine.setProperty("voice", voice.id)
                break

        self.engine.setProperty("rate", 160)
        self.engine.setProperty("volume", 1.0)

        self.cooldown_sec = cooldown_sec
        self.last_spoken_at = {}

    def _can_speak(self, key):
        now = time.time()
        last = self.last_spoken_at.get(key, 0)
        if now - last >= self.cooldown_sec:
            self.last_spoken_at[key] = now
            return True
        return False

    def speak(self, alert_level, driver, vision):
        if alert_level.name == "LOW":
            return

        if alert_level.name == "MEDIUM":
            if vision.get("phone") and self._can_speak("phone"):
                self.engine.say("Cảnh báo. Vui lòng không sử dụng điện thoại khi đang lái xe.")
                self.engine.runAndWait()

        if alert_level.name == "HIGH":
            if driver.get("drowsy") and self._can_speak("drowsy"):
                self.engine.say("Nguy hiểm. Người lái có dấu hiệu buồn ngủ. Vui lòng dừng xe và nghỉ ngơi ngay.")
                self.engine.runAndWait()

            if driver.get("distracted") and self._can_speak("distracted"):
                self.engine.say("Nguy hiểm. Người lái mất tập trung. Vui lòng chú ý quan sát.")
                self.engine.runAndWait()
