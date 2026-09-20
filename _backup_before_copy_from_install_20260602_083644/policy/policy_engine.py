import time
from enum import Enum
from typing import Any


class AlertLevel(Enum):
    NONE = 0
    PHONE = 1
    MEDIUM = 2
    HIGH = 3


class PolicyEngine:
    """
    Policy giảm nhạy buồn ngủ cho lái cẩu.

    Bản này sửa theo yêu cầu:
    - Nhắm mắt phải liên tục 10 giây mới báo HIGH.
    - Ngáp giữ nguyên, vẫn báo nhanh hơn.
    - Cúi đầu về trước không báo buồn ngủ vì đó là thao tác làm hàng.
    - Người ngủ gật thường là:
        + nhắm mắt rất lâu,
        + hoặc ngửa cổ lên,
        + hoặc cúi đầu kèm nghiêng/gục sang bên.
    - Khi mở mắt / hết tư thế ngủ thì reset nhanh về bình thường.
    """

    # =========================
    # DROWSY TUNING
    # =========================

    # Nhắm mắt phải rất lâu mới báo.
    EYES_CLOSED_HIGH_SEC = 10.0

    # Nếu đã báo do nhắm mắt, chỉ báo lại mỗi 10 giây nếu vẫn tiếp tục nhắm.
    EYES_CLOSED_REPEAT_SEC = 10.0

    # Không còn dùng raw drowsy để báo sớm.
    # Chỉ giữ làm tín hiệu phụ cho tư thế ngủ rõ.
    RAW_DROWSY_HIGH_SEC = 999.0

    # Ngửa cổ / gục nghiêng kéo dài mới báo.
    SLEEP_HEAD_POSE_SEC = 5.0

    # Mở mắt lại hoặc hết tư thế ngủ thì reset rất nhanh.
    RECOVERY_RESET_SEC = 0.4

    # =========================
    # OTHER ALERT TUNING
    # =========================

    # Ngáp giữ nguyên.
    YAWNING_HIGH_SEC = 1.2

    DISTRACTED_HIGH_SEC = 2.5
    NO_HAND_MEDIUM_SEC = 2.0
    FACE_LOST_MEDIUM_SEC = 5.0

    # radian xấp xỉ
    HEAD_UP_PITCH_THRESH = -0.25       # ngửa cổ lên
    HEAD_DOWN_PITCH_THRESH = 0.28      # cúi đầu về trước
    HEAD_ROLL_SLEEP_THRESH = 0.35      # nghiêng đầu/gục sang bên

    def __init__(self) -> None:
        self._since: dict[str, float] = {}
        self._last_alarm: dict[str, float] = {}

    def _hold_seconds(self, key: str, condition: bool, now: float) -> float:
        if condition:
            if key not in self._since:
                self._since[key] = now
            return now - self._since[key]

        self._since.pop(key, None)
        return 0.0

    def _reset_key(self, key: str) -> None:
        self._since.pop(key, None)
        self._last_alarm.pop(key, None)

    def _pulse_allowed(self, key: str, now: float, repeat_sec: float) -> bool:
        last = self._last_alarm.get(key)
        if last is None or now - last >= repeat_sec:
            self._last_alarm[key] = now
            return True
        return False

    @staticmethod
    def _bool(data: dict[str, Any], *keys: str) -> bool:
        for key in keys:
            value = data.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
        return False

    @staticmethod
    def _float(data: dict[str, Any], *keys: str, default: float = 0.0) -> float:
        for key in keys:
            value = data.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return default

    def _is_crane_working_context(
        self,
        hands: dict[str, Any],
        crane: dict[str, Any],
    ) -> bool:
        crane_active = self._bool(
            crane,
            "crane_active",
            "active",
            "running",
            "moving",
            "lifting",
            "lowering",
            "plc_start",
        )

        hand_in_zone = self._bool(
            hands,
            "hand_in_zone",
            "left_hand_in_zone",
            "right_hand_in_zone",
            "hands_ok",
        )

        no_hand = self._bool(hands, "no_hand", "no_hands")

        return crane_active or hand_in_zone or not no_hand

    def _drowsy_confirmed(
        self,
        driver: dict[str, Any],
        hands: dict[str, Any],
        crane: dict[str, Any],
        now: float,
    ) -> bool:
        raw_drowsy = self._bool(driver, "drowsy", "sleepy", "sleeping")

        eyes_closed = self._bool(
            driver,
            "eyes_closed",
            "eye_closed",
            "both_eyes_closed",
            "closed_eye",
        )

        eyes_closed_secs_from_driver = self._float(
            driver,
            "eyes_closed_secs",
            "eye_closed_secs",
            "eyes_closed_time",
            default=-1.0,
        )

        head_down = self._bool(
            driver,
            "head_down",
            "looking_down",
            "head_pitch_down",
        )

        head_up = self._bool(
            driver,
            "head_up",
            "looking_up",
            "head_pitch_up",
            "neck_back",
        )

        pitch = self._float(
            driver,
            "pitch",
            "head_pitch",
            "head_pitch_norm",
            "pitch_norm",
            default=0.0,
        )

        roll = self._float(
            driver,
            "roll",
            "head_roll",
            "head_roll_norm",
            "roll_norm",
            default=0.0,
        )

        # Suy luận thêm từ pitch/roll nếu pipeline trả số.
        if pitch <= self.HEAD_UP_PITCH_THRESH:
            head_up = True

        if pitch >= self.HEAD_DOWN_PITCH_THRESH:
            head_down = True

        head_tilted_sleep = abs(roll) >= self.HEAD_ROLL_SLEEP_THRESH

        crane_working_context = self._is_crane_working_context(hands, crane)

        # Cúi đầu về trước là đang làm hàng nếu:
        # - đầu cúi,
        # - không nghiêng/gục sang bên,
        # - đang ở ngữ cảnh vận hành.
        # Không xét eyes_closed ở đây để tránh nhắm mắt 1-2s khi cúi đầu cũng bị báo ngay.
        look_down_working = (
            head_down
            and not head_tilted_sleep
            and crane_working_context
            and not head_up
        )

        if look_down_working:
            # Cúi thẳng về trước: reset nhanh các timer ngủ.
            self._reset_key("raw_drowsy")
            self._reset_key("sleep_head_pose")
            # Không reset eyes_closed ngay nếu thật sự đang nhắm rất lâu,
            # nhưng chỉ báo khi đủ 10 giây.
            pass

        # Nếu mắt đã mở lại thì reset nhanh, tránh trạng thái HIGH giữ lâu.
        if not eyes_closed:
            self._reset_key("eyes_closed")

        # Trường hợp 1: pipeline đã tính sẵn thời gian mắt nhắm.
        # Chỉ báo theo chu kỳ 10s/lần.
        if eyes_closed_secs_from_driver >= self.EYES_CLOSED_HIGH_SEC:
            return self._pulse_allowed(
                "eyes_closed_driver",
                now,
                self.EYES_CLOSED_REPEAT_SEC,
            )

        if eyes_closed_secs_from_driver >= 0.0 and eyes_closed_secs_from_driver < self.RECOVERY_RESET_SEC:
            self._reset_key("eyes_closed_driver")

        # Trường hợp 2: policy tự đếm thời gian mắt nhắm.
        eyes_closed_hold = self._hold_seconds("eyes_closed", eyes_closed, now)

        if eyes_closed_hold >= self.EYES_CLOSED_HIGH_SEC:
            return self._pulse_allowed(
                "eyes_closed",
                now,
                self.EYES_CLOSED_REPEAT_SEC,
            )

        # Trường hợp 3: tư thế ngủ rõ.
        # - ngửa cổ lên ngủ
        # - hoặc cúi đầu + nghiêng/gục sang bên
        # Cúi thẳng về trước không được tính là ngủ.
        sleep_head_pose = head_up or (head_down and head_tilted_sleep)

        if not sleep_head_pose:
            self._reset_key("sleep_head_pose")

        sleep_pose_hold = self._hold_seconds("sleep_head_pose", sleep_head_pose, now)

        if sleep_pose_hold >= self.SLEEP_HEAD_POSE_SEC and (raw_drowsy or eyes_closed or head_tilted_sleep or head_up):
            return True

        # Không dùng raw_drowsy đơn lẻ để báo buồn ngủ nữa.
        # Vì raw_drowsy từ pipeline hiện đang quá nhạy.
        self._reset_key("raw_drowsy")

        return False

    def decide(
        self,
        vision: dict[str, Any] | None,
        driver: dict[str, Any] | None,
        hands: dict[str, Any] | None,
        crane: dict[str, Any] | None,
        phone_usage: dict[str, Any] | None = None,
    ) -> AlertLevel:
        vision = vision or {}
        driver = driver or {}
        hands = hands or {}
        crane = crane or {}
        phone_usage = phone_usage or {}

        now = time.monotonic()

        # PHONE giữ riêng.
        if self._bool(phone_usage, "phone_using"):
            return AlertLevel.PHONE

        # Ngáp giữ nguyên.
        yawning = self._bool(driver, "yawning", "yawn")
        yawning_hold = self._hold_seconds("yawning", yawning, now)
        if yawning_hold >= self.YAWNING_HIGH_SEC:
            return AlertLevel.HIGH

        # Buồn ngủ đã giảm nhạy mạnh.
        if self._drowsy_confirmed(driver, hands, crane, now):
            return AlertLevel.HIGH

        # Distracted debounce.
        distracted = self._bool(driver, "distracted", "looking_away")
        distracted_hold = self._hold_seconds("distracted", distracted, now)
        if distracted_hold >= self.DISTRACTED_HIGH_SEC:
            return AlertLevel.HIGH

        # No hand debounce.
        no_hand = self._bool(hands, "no_hand", "no_hands")
        no_hand_hold = self._hold_seconds("no_hand", no_hand, now)
        if no_hand_hold >= self.NO_HAND_MEDIUM_SEC:
            return AlertLevel.MEDIUM

        # Face lost nếu pipeline có trả.
        face_lost = self._bool(driver, "face_lost", "face_missing", "no_face")
        face_lost_hold = self._hold_seconds("face_lost", face_lost, now)
        if face_lost_hold >= self.FACE_LOST_MEDIUM_SEC:
            return AlertLevel.MEDIUM

        return AlertLevel.NONE