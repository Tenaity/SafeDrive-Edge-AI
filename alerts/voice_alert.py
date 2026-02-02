import time
import pyttsx3

class VoiceAlert:
    def __init__(self, cooldown_sec=5, window_sec=60, max_per_window=2):
        self.engine = pyttsx3.init()
        self.engine.setProperty("voice", "aav/vi+f1")
        self.engine.setProperty("rate", 160)
        self.engine.setProperty("volume", 1.0)

        # anti-spam config
        self.cooldown_sec = cooldown_sec
        self.window_sec = window_sec
        self.max_per_window = max_per_window

        self.last_spoken_at = {}    # key -> last_ts
        self.window_start = {}      # key -> window_start_ts
        self.window_count = {}      # key -> count in current window

    def _can_speak(self, key):
        now = time.time()

        # cooldown ngắn (chống nói liên tục từng frame)
        last = self.last_spoken_at.get(key, 0)
        if now - last < self.cooldown_sec:
            return False

        # window quota (2 lần / 60s)
        ws = self.window_start.get(key)
        if ws is None or (now - ws) >= self.window_sec:
            self.window_start[key] = now
            self.window_count[key] = 0

        cnt = self.window_count.get(key, 0)
        if cnt >= self.max_per_window:
            return False

        # pass
        self.window_count[key] = cnt + 1
        self.last_spoken_at[key] = now
        return True

    def speak(self, alert_level, driver, vision):
        if alert_level.name in ("NONE", "LOW"):
            return

        # ưu tiên HIGH, và chỉ nói 1 câu mỗi lần gọi
        if alert_level.name == "HIGH":
            if driver.get("drowsy") and self._can_speak("drowsy"):
                self.engine.say("Nguy hiểm. Người lái có dấu hiệu buồn ngủ. Vui lòng dừng xe và nghỉ ngơi ngay.")
                self.engine.runAndWait()
                return

            if driver.get("distracted") and self._can_speak("distracted"):
                self.engine.say("Nguy hiểm. Người lái mất tập trung. Vui lòng chú ý quan sát.")
                self.engine.runAndWait()
                return

        if alert_level.name == "MEDIUM":
            if vision.get("phone") and self._can_speak("phone"):
                self.engine.say("Cảnh báo. Vui lòng không sử dụng điện thoại khi đang lái xe.")
                self.engine.runAndWait()
                return